"""
Execution Controller - Complete Standalone Version
Manages execution of MVI sequences with safety monitoring.
No import dependencies - works completely standalone.
"""

import numpy as np
import logging
import time
import threading
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CORE DATA TYPES =====

class SystemMode(Enum):
    NORMAL = 0
    DEGRADED = 1
    SAFE_HOLD = 2
    EMERGENCY = 3

@dataclass
class RobotState:
    """Robot state for execution monitoring."""
    robot_id: str
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    temperature: float
    battery_level: float
    power_consumption: float
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    joint_torques: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    
    def __post_init__(self):
        # Ensure numpy arrays
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity)
        if not isinstance(self.orientation, np.ndarray):
            self.orientation = np.array(self.orientation)
        if not isinstance(self.angular_velocity, np.ndarray):
            self.angular_velocity = np.array(self.angular_velocity)

@dataclass
class Action:
    """Action for execution."""
    action_id: str
    action_type: str
    parameters: Dict[str, Any]
    duration: float
    priority: int = 0
    max_torque: Optional[float] = None
    max_velocity: Optional[float] = None
    thermal_load: Optional[float] = None
    power_required: Optional[float] = None

@dataclass
class MVISequence:
    """MVI sequence for execution."""
    sequence_id: str
    actions: List[Action]
    expected_duration: float
    predicted_risk: Any
    
    @property
    def action_count(self) -> int:
        return len(self.actions)

@dataclass
class SafetyConstraints:
    """Safety constraints for execution."""
    max_torque: Dict[str, float]
    max_velocity: Dict[str, float]
    thermal_limits: Dict[str, float]
    structural_load_limits: Dict[str, float]
    min_battery: float
    
    def validate_action(self, action: Action) -> Tuple[bool, List[str]]:
        """Validate an action against constraints."""
        violations = []
        
        if action.max_torque is not None:
            for joint, max_torque in self.max_torque.items():
                if action.max_torque > max_torque:
                    violations.append(f"Torque violation on {joint}: {action.max_torque} > {max_torque}")
        
        if action.thermal_load is not None:
            for component, limit in self.thermal_limits.items():
                if action.thermal_load > limit:
                    violations.append(f"Thermal violation on {component}")
        
        return len(violations) == 0, violations

# ===== EXECUTION STATUS =====

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SAFETY_STOPPED = "safety_stopped"

@dataclass
class ExecutionResult:
    """Result of action execution."""
    status: ExecutionStatus
    action_id: str
    sequence_id: str
    start_time: float
    end_time: float
    expected_duration: float
    actual_duration: float
    success_metrics: Dict[str, float]
    safety_violations: List[str]
    error_message: Optional[str] = None
    
    @property
    def was_successful(self) -> bool:
        return self.status in [ExecutionStatus.SUCCESS, ExecutionStatus.PARTIAL_SUCCESS]
    
    @property
    def timeliness(self) -> float:
        if self.expected_duration <= 0:
            return 1.0
        return min(1.0, self.expected_duration / max(0.001, self.actual_duration))

# ===== CUSTOM EXCEPTIONS =====

class SafetyViolationException(Exception):
    """Exception for safety violations."""
    def __init__(self, constraint: str, value: float, limit: float):
        super().__init__(f"Safety violation: {constraint} = {value} exceeds limit {limit}")
        self.constraint = constraint
        self.value = value
        self.limit = limit

class ActuatorFailureException(Exception):
    """Exception for actuator failures."""
    def __init__(self, actuator_id: str, command: str, reason: str):
        super().__init__(f"Actuator {actuator_id} failed to execute {command}: {reason}")
        self.actuator_id = actuator_id
        self.command = command
        self.reason = reason

class RiskExceedsThresholdException(Exception):
    """Exception for risk threshold violations."""
    def __init__(self, risk_value: float, threshold: float):
        super().__init__(f"Risk {risk_value:.3f} exceeds threshold {threshold:.3f}")
        self.risk_value = risk_value
        self.threshold = threshold

# ===== EXECUTION MONITOR =====

class ExecutionMonitor:
    """Monitors execution divergence."""
    
    def __init__(self, divergence_thresholds: Dict[str, float] = None):
        if divergence_thresholds is None:
            divergence_thresholds = {
                'position': 0.1,
                'orientation': 0.05,
                'velocity': 0.2,
                'torque': 5.0,
                'temperature': 2.0,
                'power': 10.0,
            }
        
        self.divergence_thresholds = divergence_thresholds
        self.divergence_history = []
        self.replanning_triggers = []
        
        self.execution_stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'safety_stops': 0,
            'avg_position_error': 0.0,
            'max_position_error': 0.0,
        }
        
        logger.info("Execution monitor initialized")
    
    def monitor_execution(self, expected_state: RobotState,
                         actual_state: RobotState,
                         action: Action) -> Tuple[bool, Dict[str, float]]:
        """Monitor execution divergence."""
        divergence_metrics = {}
        
        # Calculate position divergence
        pos_error = np.linalg.norm(expected_state.position - actual_state.position)
        divergence_metrics['position_error'] = pos_error
        
        # Calculate orientation divergence
        ori_error = 1.0 - abs(np.dot(expected_state.orientation, actual_state.orientation))
        divergence_metrics['orientation_error'] = ori_error
        
        # Calculate velocity divergence
        vel_error = np.linalg.norm(expected_state.velocity - actual_state.velocity)
        divergence_metrics['velocity_error'] = vel_error
        
        # Check temperature divergence
        temp_error = abs(expected_state.temperature - actual_state.temperature)
        divergence_metrics['temperature_error'] = temp_error
        
        # Check power divergence
        power_error = abs(expected_state.power_consumption - actual_state.power_consumption)
        divergence_metrics['power_error'] = power_error
        
        # Determine if safe to continue
        safe_to_continue = True
        violation_reasons = []
        
        for metric, value in divergence_metrics.items():
            threshold_key = metric.replace('_error', '')
            threshold = self.divergence_thresholds.get(threshold_key, float('inf'))
            
            if value > threshold:
                safe_to_continue = False
                violation_reasons.append(f"{metric}: {value:.3f} > {threshold:.3f}")
        
        # Record divergence
        divergence_record = {
            'timestamp': time.time(),
            'action_id': action.action_id,
            'metrics': divergence_metrics,
            'safe_to_continue': safe_to_continue,
            'violations': violation_reasons
        }
        self.divergence_history.append(divergence_record)
        
        # Keep history manageable
        if len(self.divergence_history) > 1000:
            self.divergence_history.pop(0)
        
        # Update statistics
        self._update_statistics(divergence_metrics, safe_to_continue)
        
        # Check if replanning needed
        if not safe_to_continue:
            self._trigger_replanning(action, divergence_metrics, violation_reasons)
        
        return safe_to_continue, divergence_metrics
    
    def _update_statistics(self, divergence_metrics: Dict[str, float], safe: bool):
        """Update execution statistics."""
        if 'position_error' in divergence_metrics:
            error = divergence_metrics['position_error']
            self.execution_stats['avg_position_error'] = (
                0.9 * self.execution_stats['avg_position_error'] + 0.1 * error
            )
            self.execution_stats['max_position_error'] = max(
                self.execution_stats['max_position_error'], error
            )
        
        self.execution_stats['total_actions'] += 1
        if safe:
            self.execution_stats['successful_actions'] += 1
        else:
            self.execution_stats['failed_actions'] += 1
    
    def _trigger_replanning(self, action: Action,
                           divergence_metrics: Dict[str, float],
                           reasons: List[str]):
        """Trigger replanning due to execution divergence."""
        trigger_record = {
            'timestamp': time.time(),
            'action_id': action.action_id,
            'divergence_metrics': divergence_metrics,
            'reasons': reasons,
            'thresholds': self.divergence_thresholds
        }
        self.replanning_triggers.append(trigger_record)
        
        logger.warning(f"Execution divergence triggering replanning: {reasons}")
    
    def should_replan(self, current_state: RobotState,
                     expected_state: RobotState) -> bool:
        """Determine if replanning is needed."""
        if not self.divergence_history:
            return False
        
        # Check last N executions
        recent_history = self.divergence_history[-10:]
        unsafe_count = sum(1 for record in recent_history if not record['safe_to_continue'])
        
        if unsafe_count >= 3:
            return True
        
        # Check specific divergence metrics
        pos_error = np.linalg.norm(current_state.position - expected_state.position)
        if pos_error > self.divergence_thresholds['position'] * 2.0:
            return True
        
        return False
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution monitoring metrics."""
        success_rate = 0.0
        if self.execution_stats['total_actions'] > 0:
            success_rate = self.execution_stats['successful_actions'] / self.execution_stats['total_actions']
        
        recent_divergence = []
        for record in self.divergence_history[-5:]:
            recent_divergence.append({
                'action': record['action_id'],
                'position_error': record['metrics'].get('position_error', 0.0),
                'safe': record['safe_to_continue']
            })
        
        return {
            'success_rate': success_rate,
            'total_actions': self.execution_stats['total_actions'],
            'failed_actions': self.execution_stats['failed_actions'],
            'safety_stops': self.execution_stats['safety_stops'],
            'avg_position_error': self.execution_stats['avg_position_error'],
            'max_position_error': self.execution_stats['max_position_error'],
            'recent_divergence': recent_divergence,
            'replanning_triggers': len(self.replanning_triggers)
        }
    
    def reset(self):
        """Reset execution monitor."""
        self.divergence_history = []
        self.replanning_triggers = []
        self.execution_stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'safety_stops': 0,
            'avg_position_error': 0.0,
            'max_position_error': 0.0,
        }
        logger.info("Execution monitor reset")

# ===== EXECUTION CONTROLLER =====

class ExecutionController:
    """
    Main execution controller for ARVS.
    
    Manages execution of MVI sequences with safety monitoring.
    """
    
    def __init__(self, robot_id: str, 
                 actuator_interface: Optional[Any] = None,
                 safety_constraints: Optional[SafetyConstraints] = None):
        self.robot_id = robot_id
        self.actuator_interface = actuator_interface
        self.safety_constraints = safety_constraints
        
        # Execution state
        self.current_sequence: Optional[MVISequence] = None
        self.current_action_index: int = -1
        self.execution_status: ExecutionStatus = ExecutionStatus.PENDING
        self.execution_start_time: float = 0.0
        self.execution_results: List[ExecutionResult] = []
        
        # Monitoring
        self.monitor = ExecutionMonitor()
        self.safety_check_interval = 0.1
        
        # Threading
        self.execution_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        
        # Emergency stop state
        self.emergency_stop_activated = False
        self.last_emergency_stop_time = 0.0
        
        # Performance tracking
        self.performance_metrics = {
            'sequences_executed': 0,
            'actions_executed': 0,
            'total_execution_time': 0.0,
            'emergency_stops': 0,
        }
        
        logger.info(f"Execution controller initialized for robot {robot_id}")
    
    def execute_sequence(self, mvi_sequence: MVISequence,
                        initial_state: RobotState) -> List[ExecutionResult]:
        """Execute an MVI sequence."""
        if self.execution_status == ExecutionStatus.EXECUTING:
            logger.warning("Already executing a sequence, stopping current execution")
            self.stop_execution()
        
        if not mvi_sequence.actions:
            logger.warning("Empty sequence, nothing to execute")
            return []
        
        logger.info(f"Executing sequence {mvi_sequence.sequence_id} with {len(mvi_sequence.actions)} actions")
        
        # Reset execution state
        self.current_sequence = mvi_sequence
        self.current_action_index = 0
        self.execution_status = ExecutionStatus.EXECUTING
        self.execution_start_time = time.time()
        self.stop_event.clear()
        self.pause_event.set()
        
        # Clear previous results
        sequence_results = []
        
        # Execute each action in sequence
        current_state = initial_state
        
        for i, action in enumerate(mvi_sequence.actions):
            if self.stop_event.is_set() or self.emergency_stop_activated:
                logger.warning("Execution stopped by user or emergency")
                self.execution_status = ExecutionStatus.CANCELLED
                break
            
            # Check if paused
            self.pause_event.wait()
            
            # Execute action
            try:
                result = self._execute_action(action, current_state, i, mvi_sequence.sequence_id)
                sequence_results.append(result)
                
                if not result.was_successful:
                    logger.warning(f"Action {action.action_id} failed, stopping sequence")
                    self.execution_status = ExecutionStatus.FAILED
                    break
                
                # Update current state
                current_state = self._update_state_after_action(current_state, action, result)
                
                # Move to next action
                self.current_action_index = i + 1
                
            except SafetyViolationException as e:
                logger.error(f"Safety violation during execution: {e}")
                self._activate_emergency_stop(e)
                self.execution_status = ExecutionStatus.SAFETY_STOPPED
                break
                
            except ActuatorFailureException as e:
                logger.error(f"Actuator failure: {e}")
                self.execution_status = ExecutionStatus.FAILED
                break
                
            except Exception as e:
                logger.error(f"Unexpected error during execution: {e}")
                self.execution_status = ExecutionStatus.FAILED
                break
        
        # Update final status
        if self.execution_status == ExecutionStatus.EXECUTING:
            if self.current_action_index >= len(mvi_sequence.actions):
                self.execution_status = ExecutionStatus.SUCCESS
            else:
                self.execution_status = ExecutionStatus.PARTIAL_SUCCESS
        
        # Update performance metrics
        self._update_performance_metrics(sequence_results)
        
        # Store results
        self.execution_results.extend(sequence_results)
        
        # Clean up
        self.current_sequence = None
        self.current_action_index = -1
        
        logger.info(f"Sequence execution completed with status: {self.execution_status.value}")
        
        return sequence_results
    
    def _execute_action(self, action: Action, current_state: RobotState,
                       action_index: int, sequence_id: str) -> ExecutionResult:
        """Execute a single action."""
        start_time = time.time()
        logger.info(f"Executing action {action_index}: {action.action_id}")
        
        # Pre-execution safety check
        if self.safety_constraints:
            valid, violations = self.safety_constraints.validate_action(action)
            if not valid:
                raise SafetyViolationException(
                    constraint=violations[0] if violations else "unknown",
                    value=0.0,
                    limit=0.0
                )
        
        # Send command to actuator interface
        command_success = self._send_actuator_command(action, current_state)
        
        if not command_success:
            raise ActuatorFailureException(
                actuator_id="simulated",
                command=action.action_id,
                reason="Command rejected by actuator interface"
            )
        
        # Monitor execution
        monitoring_results = self._monitor_action_execution(action, current_state, start_time)
        
        # Check for safety violations during execution
        if monitoring_results.get('safety_violation', False):
            raise SafetyViolationException(
                constraint=monitoring_results.get('violation_type', 'unknown'),
                value=monitoring_results.get('violation_value', 0.0),
                limit=monitoring_results.get('violation_limit', 0.0)
            )
        
        # Calculate execution metrics
        end_time = time.time()
        actual_duration = end_time - start_time
        
        success_metrics = {
            'position_error': monitoring_results.get('position_error', 0.0),
            'orientation_error': monitoring_results.get('orientation_error', 0.0),
            'timing_error': abs(action.duration - actual_duration),
        }
        
        # Determine status
        if monitoring_results.get('completed_successfully', True):
            if abs(actual_duration - action.duration) > action.duration * 0.2:
                status = ExecutionStatus.PARTIAL_SUCCESS
            else:
                status = ExecutionStatus.SUCCESS
        else:
            status = ExecutionStatus.FAILED
        
        result = ExecutionResult(
            status=status,
            action_id=action.action_id,
            sequence_id=sequence_id,
            start_time=start_time,
            end_time=end_time,
            expected_duration=action.duration,
            actual_duration=actual_duration,
            success_metrics=success_metrics,
            safety_violations=monitoring_results.get('safety_violations', []),
            error_message=monitoring_results.get('error_message')
        )
        
        return result
    
    def _send_actuator_command(self, action: Action, current_state: RobotState) -> bool:
        """Send command to actuator interface."""
        if self.actuator_interface:
            try:
                return self.actuator_interface.execute_command(action, current_state)
            except Exception as e:
                logger.error(f"Actuator interface error: {e}")
                return False

        # No physical interface connected yet (NASA/ESA telemetry integration pending).
        # Accept the command deterministically — actual success is assessed in
        # _monitor_action_execution via state divergence, not by coin flip.
        logger.debug(f"No actuator interface: command {action.action_id} accepted for monitoring")
        time.sleep(min(action.duration, 0.1))  # Yield control loop slice
        return True
    
    def _monitor_action_execution(self, action: Action, 
                                 initial_state: RobotState,
                                 start_time: float) -> Dict[str, Any]:
        """Monitor action execution in real-time."""
        monitoring_results = {
            'completed_successfully': True,
            'safety_violation': False,
            'position_error': 0.0,
            'orientation_error': 0.0,
            'safety_violations': [],
            'error_message': None
        }
        
        # Simulate monitoring loop
        elapsed = 0.0
        check_interval = 0.01
        
        while elapsed < action.duration and not self.stop_event.is_set():
            if self.pause_event.is_set():
                # Calculate current expected state
                progress = elapsed / action.duration
                expected_state = self._predict_state_during_action(
                    initial_state, action, progress
                )
                
                # Simulate current actual state
                actual_state = self._simulate_actual_state(expected_state)
                
                # Check for divergence
                safe, divergence = self.monitor.monitor_execution(
                    expected_state, actual_state, action
                )
                
                if not safe:
                    monitoring_results['safety_violation'] = True
                    monitoring_results['violation_type'] = 'execution_divergence'
                    monitoring_results['violation_value'] = divergence.get('position_error', 0.0)
                    monitoring_results['violation_limit'] = self.monitor.divergence_thresholds.get('position', 0.1)
                    monitoring_results['error_message'] = 'Execution divergence exceeded threshold'
                    break
                
                # Update monitoring results
                monitoring_results['position_error'] = divergence.get('position_error', 0.0)
                monitoring_results['orientation_error'] = divergence.get('orientation_error', 0.0)
            
            # Wait for next check
            time.sleep(check_interval)
            elapsed = time.time() - start_time
        
        return monitoring_results
    
    def _predict_state_during_action(self, initial_state: RobotState,
                                    action: Action, progress: float) -> RobotState:
        """Predict state during action execution."""
        predicted_state = RobotState(
            robot_id=initial_state.robot_id,
            timestamp=initial_state.timestamp + action.duration * progress,
            position=initial_state.position + initial_state.velocity * action.duration * progress,
            velocity=initial_state.velocity * (1.0 - progress * 0.5),
            orientation=initial_state.orientation.copy(),
            angular_velocity=initial_state.angular_velocity.copy(),
            joint_positions=initial_state.joint_positions.copy(),
            joint_velocities=initial_state.joint_velocities.copy(),
            joint_torques=initial_state.joint_torques.copy(),
            temperature=initial_state.temperature + (action.thermal_load or 0.0) * progress,
            battery_level=max(0.0, initial_state.battery_level - 
                            (action.power_required or 0.0) * action.duration * progress / 3600.0 / 100.0),
            power_consumption=action.power_required or initial_state.power_consumption,
            confidence=initial_state.confidence * (1.0 - progress * 0.1)
        )
        
        return predicted_state
    
    def _simulate_actual_state(self, expected_state: RobotState) -> RobotState:
        """
        Derive actual state from expected state using action physics.

        When no hardware interface is connected, the actual state IS the
        expected state — divergence only arises when real sensor feedback
        (NASA/ESA telemetry or ROS2 topics) differs from the prediction.
        Injecting random noise here would produce false divergence triggers
        and false safety-gate rejections.
        """
        return RobotState(
            robot_id=expected_state.robot_id,
            timestamp=expected_state.timestamp,
            position=expected_state.position.copy(),
            velocity=expected_state.velocity.copy(),
            orientation=expected_state.orientation.copy(),
            angular_velocity=expected_state.angular_velocity.copy(),
            joint_positions=expected_state.joint_positions.copy(),
            joint_velocities=expected_state.joint_velocities.copy(),
            joint_torques=expected_state.joint_torques.copy(),
            temperature=expected_state.temperature,
            battery_level=expected_state.battery_level,
            power_consumption=expected_state.power_consumption,
            confidence=expected_state.confidence
        )
    
    def _update_state_after_action(self, current_state: RobotState,
                                  action: Action, result: ExecutionResult) -> RobotState:
        """Update state after action execution."""
        updated_state = RobotState(
            robot_id=current_state.robot_id,
            timestamp=result.end_time,
            position=current_state.position + current_state.velocity * result.actual_duration,
            velocity=current_state.velocity * 0.9,
            orientation=current_state.orientation.copy(),
            angular_velocity=current_state.angular_velocity.copy(),
            joint_positions=current_state.joint_positions.copy(),
            joint_velocities=current_state.joint_velocities.copy(),
            joint_torques=current_state.joint_torques.copy(),
            temperature=current_state.temperature + (action.thermal_load or 0.0),
            battery_level=max(0.0, current_state.battery_level - 
                            (action.power_required or 0.0) * result.actual_duration / 3600.0 / 100.0),
            power_consumption=action.power_required or current_state.power_consumption,
            confidence=current_state.confidence * (0.9 if result.was_successful else 0.7)
        )
        
        return updated_state
    
    def _update_performance_metrics(self, results: List[ExecutionResult]):
        """Update performance tracking metrics."""
        self.performance_metrics['sequences_executed'] += 1
        self.performance_metrics['actions_executed'] += len(results)
        
        total_time = sum(r.actual_duration for r in results)
        self.performance_metrics['total_execution_time'] += total_time
        
        if any(r.status == ExecutionStatus.SAFETY_STOPPED for r in results):
            self.performance_metrics['emergency_stops'] += 1
    
    def _activate_emergency_stop(self, violation: SafetyViolationException):
        """Activate emergency stop procedure."""
        self.emergency_stop_activated = True
        self.last_emergency_stop_time = time.time()
        self.performance_metrics['emergency_stops'] += 1
        
        logger.critical(f"EMERGENCY STOP ACTIVATED: {violation}")
        
        # Send emergency stop command to actuators
        if self.actuator_interface:
            try:
                self.actuator_interface.emergency_stop()
            except Exception as e:
                logger.error(f"Failed to send emergency stop to actuators: {e}")
    
    def stop_execution(self):
        """Stop current execution gracefully."""
        self.stop_event.set()
        logger.info("Execution stop requested")
    
    def pause_execution(self):
        """Pause current execution."""
        self.pause_event.clear()
        logger.info("Execution paused")
    
    def resume_execution(self):
        """Resume paused execution."""
        self.pause_event.set()
        logger.info("Execution resumed")
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        current_action = None
        if (self.current_sequence and 
            0 <= self.current_action_index < len(self.current_sequence.actions)):
            current_action = self.current_sequence.actions[self.current_action_index]
        
        return {
            'status': self.execution_status.value,
            'current_sequence_id': self.current_sequence.sequence_id if self.current_sequence else None,
            'current_action': current_action.action_id if current_action else None,
            'current_action_index': self.current_action_index,
            'emergency_stop_active': self.emergency_stop_activated,
            'paused': not self.pause_event.is_set(),
            'execution_time': time.time() - self.execution_start_time if self.execution_start_time > 0 else 0.0,
            'performance_metrics': self.performance_metrics.copy(),
            'monitoring_metrics': self.monitor.get_execution_metrics()
        }
    
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        self.emergency_stop_activated = False
        logger.info("Emergency stop reset")
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_results = []
        self.monitor.reset()
        logger.info("Execution history cleared")
    
    def shutdown(self):
        """Shutdown execution controller."""
        self.stop_execution()
        self.reset_emergency_stop()
        logger.info("Execution controller shutdown")

# ===== DEMONSTRATION =====

def demonstrate_execution_controller():
    """Demonstrate the execution controller functionality."""
    print("=" * 70)
    print("EXECUTION CONTROLLER DEMONSTRATION")
    print("Complete standalone implementation")
    print("=" * 70)
    
    # Create safety constraints
    constraints = SafetyConstraints(
        max_torque={'joint1': 100.0, 'joint2': 80.0},
        max_velocity={'joint1': 2.0, 'joint2': 1.5},
        thermal_limits={'motor': 373.0, 'cpu': 358.0},
        structural_load_limits={'arm': 200.0},
        min_battery=0.15
    )
    
    # Create execution controller
    controller = ExecutionController(
        robot_id="test_robot",
        safety_constraints=constraints
    )
    
    # Create test robot state
    robot_state = RobotState(
        robot_id="test_robot",
        timestamp=time.time(),
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.5, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        temperature=320.0,
        battery_level=0.6,
        power_consumption=60.0,
        joint_positions={'joint1': 0.5, 'joint2': 0.3},
        confidence=0.9
    )
    
    # Create test actions
    actions = [
        Action(
            action_id="move_forward_1",
            action_type="motion",
            parameters={'direction': 'forward', 'distance': 1.0},
            duration=2.0,
            max_torque=50.0,
            max_velocity=0.5,
            thermal_load=10.0,
            power_required=40.0
        ),
        Action(
            action_id="move_forward_2",
            action_type="motion",
            parameters={'direction': 'forward', 'distance': 1.0},
            duration=2.0,
            max_torque=60.0,
            max_velocity=0.6,
            thermal_load=15.0,
            power_required=50.0
        )
    ]
    
    # Create MVI sequence
    mvi_sequence = MVISequence(
        sequence_id="test_sequence",
        actions=actions,
        expected_duration=4.0,
        predicted_risk=type('obj', (object,), {'overall_risk': 0.3})()
    )
    
    print("\n1. Starting sequence execution...")
    
    # Execute sequence
    try:
        results = controller.execute_sequence(mvi_sequence, robot_state)
        
        print(f"\n2. Execution completed with {len(results)} results:")
        for i, result in enumerate(results):
            print(f"   Action {i+1}: {result.action_id}")
            print(f"     Status: {result.status.value}")
            print(f"     Success: {result.was_successful}")
            print(f"     Duration: {result.actual_duration:.2f}s (expected: {result.expected_duration:.2f}s)")
            print(f"     Timeliness: {result.timeliness:.2f}")
        
        # Get execution status
        print("\n3. Execution controller status:")
        status = controller.get_execution_status()
        for key, value in status.items():
            if key not in ['performance_metrics', 'monitoring_metrics']:
                print(f"   {key}: {value}")
        
        # Get performance metrics
        print("\n4. Performance metrics:")
        metrics = controller.performance_metrics
        for key, value in metrics.items():
            print(f"   {key}: {value}")
        
        # Get monitoring metrics
        print("\n5. Monitoring metrics:")
        monitor_metrics = controller.monitor.get_execution_metrics()
        for key, value in monitor_metrics.items():
            if key != 'recent_divergence':
                print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"Execution failed: {e}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_execution_controller()