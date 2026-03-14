"""
ARVS Core System Integration
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
import threading
from dataclasses import dataclass
from enum import Enum

from arvs.core.data_types import (
    RobotState, SystemMode, SystemTelemetry, SafetyConstraints,
    Observation, FeatureVector, RiskAssessment, MVISequence,
    Action, OptimizationProblem, OptimizationResult
)
from arvs.core.exceptions import (
    SafetyViolationException, FaultDetectedException,
    OptimizationTimeoutException, DegradedModeException
)
from arvs.core.constants import (
    CONTROL_LOOP_PERIOD, MAX_REPLANNING_TIME,
    SYSTEM_VERSION, IMPLEMENTATION_VERSION
)

# Import all ARVS components
from arvs.perception.abstraction import PerceptionAbstraction
from arvs.state.estimation import StateEstimator
from arvs.risk.quantification import RiskQuantifier
from arvs.optimization.engine import OptimizationEngine, SolverType
from arvs.decision.mvi_logic import MVILogic, MVIProfile
from arvs.safety.safety_gate import SafetyGate
from arvs.fault.detection import FaultDetector
from arvs.execution.controller import ExecutionController
from arvs.learning.adaptive_models import AdaptiveLearner
from arvs.audit.logger import AuditLogger
from arvs.coordination.multi_robot import MultiRobotCoordinator

logger = logging.getLogger(__name__)

@dataclass
class ARVSConfig:
    """Configuration for ARVS system."""
    robot_id: str
    initial_mode: SystemMode = SystemMode.NORMAL
    control_loop_period: float = CONTROL_LOOP_PERIOD
    max_replanning_time: float = MAX_REPLANNING_TIME
    enable_learning: bool = True
    enable_audit_logging: bool = True
    coordination_mode: str = "centralized"
    safety_margins: Dict[str, float] = None
    
    def __post_init__(self):
        """Set default safety margins if not provided."""
        if self.safety_margins is None:
            self.safety_margins = {
                'torque': 0.8,
                'thermal': 0.9,
                'structural': 0.7,
                'velocity': 0.8,
                'battery': 0.3,
            }

class ARVSCore:
    """
    Main ARVS system implementation.
    
    Integrates all components into the triple-layer architecture:
    1. QUBO encoding of real-time robotic state
    2. Minimal-Viable-Intervention (MVI) optimization  
    3. Safety-Gated Verification Module
    
    From ARVS document: Operates independently of Earth-based control loops.
    """
    
    def __init__(self, config: ARVSConfig):
        """
        Initialize ARVS core system.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.robot_id = config.robot_id
        
        # System state
        self.system_mode = config.initial_mode
        self.previous_mode = config.initial_mode
        self.is_running = False
        self.control_loop_thread: Optional[threading.Thread] = None
        
        # Initialize components
        self._initialize_components()
        
        # State tracking
        self.current_state: Optional[RobotState] = None
        self.current_risk: Optional[RiskAssessment] = None
        self.current_mvi: Optional[MVISequence] = None
        self.execution_results: List[Dict] = []
        
        # Performance metrics
        self.performance_metrics = {
            'control_cycles': 0,
            'avg_cycle_time': 0.0,
            'replanning_count': 0,
            'safety_violations_prevented': 0,
            'emergency_stops': 0,
            'uptime': 0.0
        }
        
        # Mission objectives
        self.mission_objectives: List[Dict] = []
        self.current_objective_index = 0
        
        # Start time
        self.start_time = time.time()
        
        logger.info(f"ARVS Core initialized for robot {self.robot_id}")
        logger.info(f"System version: {SYSTEM_VERSION}, Implementation: {IMPLEMENTATION_VERSION}")
    
    def _initialize_components(self):
        """Initialize all ARVS components."""
        # Initialize safety constraints (these would come from robot configuration)
        self.safety_constraints = SafetyConstraints(
            max_torque={'joint1': 100.0, 'joint2': 100.0, 'joint3': 150.0},
            max_velocity={'joint1': 2.0, 'joint2': 2.0, 'joint3': 1.5},
            thermal_limits={'motor': 373.0, 'cpu': 358.0, 'battery': 323.0},
            structural_load_limits={'arm': 200.0, 'base': 500.0},
            min_battery=0.15,
            collision_zones=[
                {
                    'type': 'cylinder',
                    'center': [0.0, 0.0, 0.0],
                    'radius': 5.0,
                    'height': 10.0,
                    'name': 'keepout_zone'
                }
            ],
            communication_blackouts=[]  # Would be populated from mission plan
        )
        
        # Perception and state estimation
        sensor_config = {
            'imu': {'type': 'imu', 'update_rate': 100},
            'joint_state': {'type': 'joint_state', 'num_joints': 3},
            'thermal': {'type': 'thermal', 'components': ['motor', 'cpu', 'battery']}
        }
        
        self.perception = PerceptionAbstraction(self.robot_id, sensor_config)
        self.state_estimator = StateEstimator(self.robot_id)
        
        # Risk and safety
        self.risk_quantifier = RiskQuantifier(self.safety_constraints)
        self.safety_gate = SafetyGate(self.safety_constraints)
        
        # Optimization and decision making
        self.optimization_engine = OptimizationEngine(
            solver_preference=[
                SolverType.CLASSICAL,  # Start with classical for reliability
                SolverType.HYBRID,
                SolverType.QUANTUM_ANNEALER
            ]
        )
        
        self.mvi_logic = MVILogic(self.safety_constraints)
        
        # Fault detection and execution
        self.fault_detector = FaultDetector(self.robot_id, self.system_mode)
        self.execution_controller = ExecutionController(
            self.robot_id,
            safety_constraints=self.safety_constraints
        )
        
        # Learning and adaptation
        self.adaptive_learner = AdaptiveLearner(self.robot_id)
        self.adaptive_learner.enable_learning(self.config.enable_learning)
        
        # Audit and coordination
        self.audit_logger = AuditLogger(self.robot_id)
        self.coordinator = MultiRobotCoordinator()
        
        # Register this robot as a single-robot swarm
        self.coordinator.register_swarm(f"swarm_{self.robot_id}", [self.robot_id])
    
    def set_mission_objectives(self, objectives: List[Dict]):
        """
        Set mission objectives for the robot.
        
        Args:
            objectives: List of mission objectives with priorities, constraints, etc.
        """
        self.mission_objectives = objectives
        self.current_objective_index = 0
        
        logger.info(f"Set {len(objectives)} mission objectives")
        
        # Log mission start
        self.audit_logger.log_telemetry(SystemTelemetry(
            timestamp=time.time(),
            system_mode=self.system_mode,
            robot_state=self.current_state or self._get_default_state(),
            risk_assessment=self.current_risk or RiskAssessment(
                timestamp=time.time(),
                overall_risk=0.0,
                component_risks={},
                risk_factors={},
                confidence=1.0
            ),
            active_constraints=self.safety_constraints,
            selected_mvi=None,
            executed_actions=[],
            fault_status={},
            optimization_metrics={},
            safety_violations=[]
        ))
    
    def start(self):
        """Start the ARVS control loop."""
        if self.is_running:
            logger.warning("ARVS already running")
            return
        
        self.is_running = True
        self.control_loop_thread = threading.Thread(
            target=self._control_loop,
            daemon=True,
            name="ARVS_Control_Loop"
        )
        self.control_loop_thread.start()
        
        logger.info("ARVS control loop started")
        self.audit_logger.log_mode_change(
            SystemMode.NORMAL, self.system_mode, "system_start"
        )
    
    def stop(self):
        """Stop the ARVS control loop."""
        self.is_running = False
        
        if self.control_loop_thread:
            self.control_loop_thread.join(timeout=2.0)
        
        # Stop execution controller
        self.execution_controller.shutdown()
        
        # Shutdown audit logger
        self.audit_logger.log_system_shutdown()
        self.audit_logger.shutdown()
        
        logger.info("ARVS stopped")
    
    def _control_loop(self):
        """Main control loop of ARVS."""
        logger.info("ARVS control loop started")
        
        last_cycle_time = time.time()
        
        while self.is_running:
            cycle_start = time.time()
            
            try:
                # 1. Get observations (simulated for now)
                observations = self._get_simulated_observations()
                
                # 2. Process observations through perception
                feature_vector = self.perception.process_observations(observations)
                
                # 3. Update state estimation
                belief_state = self.state_estimator.update_belief(feature_vector)
                self.current_state = belief_state.mean_state
                
                # 4. Monitor for faults
                new_mode, detected_faults = self.fault_detector.monitor_system(
                    self.current_state, observations, feature_vector
                )
                
                # Update system mode if changed
                if new_mode != self.system_mode:
                    self.audit_logger.log_mode_change(
                        self.system_mode, new_mode,
                        f"faults_detected: {len(detected_faults)}"
                    )
                    self.previous_mode = self.system_mode
                    self.system_mode = new_mode
                
                # 5. Quantify risk
                self.current_risk = self.risk_quantifier.assess_risk(
                    self.current_state, belief_state
                )
                
                # Log risk assessment
                self.audit_logger.log_risk_assessment(
                    self.current_risk, self.system_mode
                )
                
                # 6. Check if replanning is needed
                if self._should_replan():
                    # 7. Generate optimization problem
                    optimization_problem = self.optimization_engine.formulate_problem(
                        self.current_state,
                        self.current_risk,
                        self.safety_constraints,
                        self._get_available_actions()
                    )
                    
                    # 8. Solve optimization problem
                    try:
                        optimization_result = self.optimization_engine.solve(
                            optimization_problem,
                            timeout=self.config.max_replanning_time
                        )
                        
                        # Log optimization
                        self.audit_logger.log_optimization(
                            optimization_problem.problem_id,
                            optimization_problem.num_variables,
                            optimization_result.solver_type.value,
                            optimization_result.solver_time,
                            optimization_result.success,
                            self.system_mode
                        )
                        
                        if optimization_result.success:
                            # 9. Generate MVI sequence
                            mvi_profile = self._get_mvi_profile_for_mode()
                            self.current_mvi = self.mvi_logic.generate_mvi(
                                self.current_state,
                                self.current_risk,
                                optimization_result,
                                optimization_problem,
                                mvi_profile
                            )
                            
                            # 10. Safety gate verification
                            safety_result = self.safety_gate.check_mvi_sequence(
                                self.current_mvi, self.current_state
                            )
                            
                            if safety_result.safe:
                                # 11. Execute MVI sequence
                                self._execute_mvi_sequence(self.current_mvi)
                                
                                # Log decision
                                self.audit_logger.log_decision(
                                    self.current_mvi,
                                    self.system_mode,
                                    {
                                        'solver_type': optimization_result.solver_type.value,
                                        'solve_time': optimization_result.solver_time,
                                        'objective_value': optimization_result.objective_value
                                    },
                                    True
                                )
                            else:
                                # Safety gate rejected the MVI
                                logger.warning(f"MVI rejected by safety gate: {safety_result.get_violation_summary()}")
                                self.performance_metrics['safety_violations_prevented'] += 1
                                
                                # Log safety violation
                                for violation in safety_result.violations:
                                    self.audit_logger.log_safety_violation(
                                        violation[0].value, violation[1],
                                        violation[2], violation[3],
                                        self.system_mode,
                                        self.current_mvi.sequence_id if self.current_mvi else None
                                    )
                                
                                # Enter safe hold if MVI was rejected
                                if self.system_mode != SystemMode.EMERGENCY:
                                    self._enter_safe_hold("mvi_rejected_by_safety_gate")
                        
                        else:
                            logger.error("Optimization failed, entering degraded mode")
                            self._enter_degraded_mode("optimization_failed")
                            
                    except OptimizationTimeoutException as e:
                        logger.error(f"Optimization timeout: {e}")
                        self._enter_degraded_mode("optimization_timeout")
                        
                    except Exception as e:
                        logger.error(f"Optimization error: {e}")
                        self._enter_degraded_mode("optimization_error")
                
                # 12. Update performance metrics
                self._update_performance_metrics(cycle_start)
                
                # 13. Sleep to maintain control loop period
                cycle_time = time.time() - cycle_start
                sleep_time = max(0.0, self.config.control_loop_period - cycle_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning(f"Control loop overrun: {cycle_time:.3f}s > {self.config.control_loop_period:.3f}s")
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}", exc_info=True)
                self._handle_control_loop_error(e)
                
                # Prevent tight error loop
                time.sleep(0.1)
    
    def _get_simulated_observations(self) -> List[Observation]:
        """Generate simulated observations for testing."""
        # In practice, this would interface with actual sensors
        current_time = time.time()
        
        observations = [
            Observation(
                sensor_id='imu_1',
                timestamp=current_time,
                data=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # quaternion + angular velocity
                data_type='imu',
                covariance=np.diag([0.001, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01]),
                valid=True
            ),
            Observation(
                sensor_id='joint_state_1',
                timestamp=current_time,
                data=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 3 joints: pos, vel
                data_type='joint_state',
                valid=True
            ),
            Observation(
                sensor_id='thermal_1',
                timestamp=current_time,
                data=np.array([303.0, 323.0, 300.0]),  # motor, cpu, battery temps
                data_type='thermal',
                valid=True
            )
        ]
        
        return observations
    
    def _get_default_state(self) -> RobotState:
        """Get default robot state."""
        return RobotState(
            robot_id=self.robot_id,
            timestamp=time.time(),
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            joint_positions={'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0},
            joint_velocities={'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0},
            joint_torques={'joint1': 0.0, 'joint2': 0.0, 'joint3': 0.0},
            temperature=293.0,
            battery_level=0.8,
            power_consumption=50.0,
            confidence=1.0
        )
    
    def _should_replan(self) -> bool:
        """Determine if replanning is needed."""
        # Always replan in first cycle
        if self.performance_metrics['control_cycles'] == 0:
            return True
        
        # Replan based on risk level
        if self.current_risk and self.current_risk.overall_risk > 0.5:
            return True
        
        # Replan if no current MVI
        if self.current_mvi is None:
            return True
        
        # Replan if MVI execution completed
        execution_status = self.execution_controller.get_execution_status()
        if execution_status['status'] in ['success', 'failed', 'cancelled']:
            return True
        
        # Replan based on execution monitor recommendation
        if self.execution_controller.monitor.should_replan(
            self.current_state,
            self._predict_state_from_mvi()
        ):
            return True
        
        # Periodic replanning
        if self.performance_metrics['control_cycles'] % 10 == 0:
            return True
        
        return False
    
    def _predict_state_from_mvi(self) -> RobotState:
        """Predict state based on current MVI execution."""
        if not self.current_mvi or not self.current_state:
            return self.current_state or self._get_default_state()
        
        # Simplified prediction - in practice would use dynamics model
        predicted_state = RobotState(
            robot_id=self.robot_id,
            timestamp=self.current_state.timestamp + self.current_mvi.expected_duration,
            position=self.current_state.position + self.current_state.velocity * self.current_mvi.expected_duration,
            velocity=self.current_state.velocity * 0.9,  # Some deceleration
            orientation=self.current_state.orientation.copy(),
            angular_velocity=self.current_state.angular_velocity.copy(),
            joint_positions=self.current_state.joint_positions.copy(),
            joint_velocities=self.current_state.joint_velocities.copy(),
            joint_torques=self.current_state.joint_torques.copy(),
            temperature=self.current_state.temperature + 5.0,  # Some heating
            battery_level=max(0.0, self.current_state.battery_level - 0.01),  # Some battery usage
            power_consumption=self.current_state.power_consumption,
            confidence=self.current_state.confidence * 0.8
        )
        
        return predicted_state
    
    def _get_available_actions(self) -> List[Action]:
        """Get available actions based on current state and mode."""
        # Start with basic safety actions
        available_actions = []
        
        # Always available safety actions
        available_actions.extend([
            Action(
                action_id='halt',
                action_type='safety',
                parameters={'mode': 'immediate'},
                duration=0.1,
                priority=100,
                max_torque=0.0,
                max_velocity=0.0,
                thermal_load=0.0,
                power_required=10.0
            ),
            Action(
                action_id='safe_hold',
                action_type='safety',
                parameters={'mode': 'stabilized'},
                duration=1.0,
                priority=90,
                max_torque=5.0,
                max_velocity=0.1,
                thermal_load=5.0,
                power_required=20.0
            )
        ])
        
        # Add movement actions if not in emergency mode
        if self.system_mode != SystemMode.EMERGENCY:
            available_actions.extend([
                Action(
                    action_id='move_forward',
                    action_type='motion',
                    parameters={'direction': 'forward', 'distance': 1.0},
                    duration=2.0,
                    priority=50,
                    max_torque=30.0,
                    max_velocity=0.5,
                    thermal_load=20.0,
                    power_required=100.0
                ),
                Action(
                    action_id='move_backward',
                    action_type='motion',
                    parameters={'direction': 'backward', 'distance': 1.0},
                    duration=2.0,
                    priority=50,
                    max_torque=30.0,
                    max_velocity=0.5,
                    thermal_load=20.0,
                    power_required=100.0
                ),
                Action(
                    action_id='turn_left',
                    action_type='motion',
                    parameters={'angle': 45.0},
                    duration=1.0,
                    priority=50,
                    max_torque=20.0,
                    max_velocity=0.3,
                    thermal_load=15.0,
                    power_required=80.0
                )
            ])
        
        # Add manipulation actions if battery sufficient
        if self.current_state and self.current_state.battery_level > 0.3:
            available_actions.append(
                Action(
                    action_id='manipulate_object',
                    action_type='manipulation',
                    parameters={'object': 'unknown', 'grasp_force': 10.0},
                    duration=3.0,
                    priority=60,
                    max_torque=50.0,
                    max_velocity=0.2,
                    thermal_load=30.0,
                    power_required=150.0
                )
            )
        
        return available_actions
    
    def _get_mvi_profile_for_mode(self) -> MVIProfile:
        """Get MVI profile based on current system mode."""
        if self.system_mode == SystemMode.EMERGENCY:
            return MVIProfile.emergency_profile()
        elif self.system_mode == SystemMode.DEGRADED:
            return MVIProfile.degraded_profile()
        else:
            return MVIProfile.normal_profile()
    
    def _execute_mvi_sequence(self, mvi_sequence: MVISequence):
        """Execute MVI sequence through execution controller."""
        if not mvi_sequence.actions:
            logger.warning("Empty MVI sequence, nothing to execute")
            return
        
        logger.info(f"Executing MVI sequence {mvi_sequence.sequence_id}")
        
        # Execute sequence
        results = self.execution_controller.execute_sequence(
            mvi_sequence, self.current_state
        )
        
        # Log execution results
        for result in results:
            self.audit_logger.log_action_execution(
                next(a for a in mvi_sequence.actions if a.action_id == result.action_id),
                {
                    'status': result.status.value,
                    'duration': result.actual_duration,
                    'success': result.was_successful,
                    'timeliness': result.timeliness
                },
                self.system_mode
            )
        
        # Store results
        self.execution_results.extend(results)
        
        # Update safety gate with last safe state
        if results and results[-1].was_successful:
            self.safety_gate.update_last_safe_state(self.current_state)
    
    def _enter_safe_hold(self, reason: str):
        """Enter safe hold mode."""
        logger.warning(f"Entering safe hold: {reason}")
        
        old_mode = self.system_mode
        self.system_mode = SystemMode.SAFE_HOLD
        
        self.audit_logger.log_mode_change(
            old_mode, self.system_mode, reason
        )
        
        # Execute safe hold action
        safe_hold_action = Action(
            action_id='emergency_safe_hold',
            action_type='safety',
            parameters={'reason': reason},
            duration=10.0,
            priority=100,
            max_torque=0.0,
            max_velocity=0.0,
            thermal_load=0.0,
            power_required=20.0
        )
        
        safe_mvi = MVISequence(
            sequence_id=f"safe_hold_{time.time()}",
            actions=[safe_hold_action],
            expected_duration=10.0,
            predicted_risk=RiskAssessment(
                timestamp=time.time(),
                overall_risk=0.1,
                component_risks={},
                risk_factors={},
                confidence=0.9
            )
        )
        
        self._execute_mvi_sequence(safe_mvi)
    
    def _enter_degraded_mode(self, reason: str):
        """Enter degraded mode."""
        logger.warning(f"Entering degraded mode: {reason}")
        
        old_mode = self.system_mode
        self.system_mode = SystemMode.DEGRADED
        
        self.audit_logger.log_mode_change(
            old_mode, self.system_mode, reason
        )
        
        # Reduce optimization complexity
        self.optimization_engine.switch_solver(SolverType.CLASSICAL)
    
    def _update_performance_metrics(self, cycle_start: float):
        """Update performance metrics."""
        cycle_time = time.time() - cycle_start
        
        self.performance_metrics['control_cycles'] += 1
        self.performance_metrics['avg_cycle_time'] = (
            0.9 * self.performance_metrics['avg_cycle_time'] + 0.1 * cycle_time
        )
        self.performance_metrics['uptime'] = time.time() - self.start_time
        
        # Update from execution controller
        exec_status = self.execution_controller.get_execution_status()
        self.performance_metrics['emergency_stops'] = exec_status['performance_metrics']['emergency_stops']
    
    def _handle_control_loop_error(self, error: Exception):
        """Handle control loop errors gracefully."""
        logger.error(f"Control loop error: {error}")
        
        # Enter degraded mode on error
        if self.system_mode != SystemMode.EMERGENCY:
            self._enter_degraded_mode(f"control_loop_error: {str(error)}")
        
        # Log error
        self.audit_logger.log_safety_violation(
            'software_error',
            'control_loop',
            1.0, 0.0,  # Always a violation
            self.system_mode
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        exec_status = self.execution_controller.get_execution_status()
        fault_report = self.fault_detector.get_system_health_report()
        
        return {
            'robot_id': self.robot_id,
            'system_mode': self.system_mode.name,
            'previous_mode': self.previous_mode.name,
            'uptime': self.performance_metrics['uptime'],
            'control_cycle': self.performance_metrics['control_cycles'],
            'avg_cycle_time': self.performance_metrics['avg_cycle_time'],
            'current_state': {
                'position': self.current_state.position.tolist() if self.current_state else None,
                'battery': self.current_state.battery_level if self.current_state else None,
                'temperature': self.current_state.temperature if self.current_state else None
            },
            'current_risk': self.current_risk.overall_risk if self.current_risk else None,
            'current_mvi': self.current_mvi.sequence_id if self.current_mvi else None,
            'execution_status': exec_status['status'],
            'fault_report': fault_report,
            'performance_metrics': self.performance_metrics,
            'mission_progress': {
                'current_objective': self.current_objective_index,
                'total_objectives': len(self.mission_objectives)
            }
        }
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all components."""
        return {
            'perception': {
                'sensor_health': self.perception._compute_sensor_health_score()
            },
            'state_estimation': {
                'confidence': self.state_estimator.current_belief.confidence if self.state_estimator.current_belief else 0.0
            },
            'risk_quantification': self.risk_quantifier.get_component_breakdown() if self.current_risk else {},
            'optimization': self.optimization_engine.get_performance_stats(),
            'mvi_logic': self.mvi_logic.get_statistics(),
            'safety_gate': self.safety_gate.get_safety_metrics(),
            'fault_detection': self.fault_detector.get_system_health_report(),
            'execution': self.execution_controller.get_execution_status(),
            'learning': self.adaptive_learner.get_learning_stats(),
            'audit': self.audit_logger.get_statistics(),
            'coordination': self.coordinator.get_coordination_stats()
        }
    
    def get_telemetry(self) -> SystemTelemetry:
        """Get complete system telemetry."""
        return SystemTelemetry(
            timestamp=time.time(),
            system_mode=self.system_mode,
            robot_state=self.current_state or self._get_default_state(),
            risk_assessment=self.current_risk or RiskAssessment(
                timestamp=time.time(),
                overall_risk=0.0,
                component_risks={},
                risk_factors={},
                confidence=1.0
            ),
            active_constraints=self.safety_constraints,
            selected_mvi=self.current_mvi,
            executed_actions=self.execution_controller.execution_results[-10:] if self.execution_controller.execution_results else [],
            fault_status={},  # Would populate from fault detector
            optimization_metrics=self.optimization_engine.get_performance_stats(),
            safety_violations=[]  # Would populate from safety gate
        )
    
    def emergency_stop(self):
        """Activate emergency stop."""
        logger.critical("EMERGENCY STOP ACTIVATED")
        
        self.execution_controller._activate_emergency_stop(
            SafetyViolationException('emergency_stop', 1.0, 0.0)
        )
        
        old_mode = self.system_mode
        self.system_mode = SystemMode.EMERGENCY
        
        self.audit_logger.log_mode_change(
            old_mode, self.system_mode, "manual_emergency_stop"
        )
        
        self.audit_logger.log_safety_violation(
            'emergency_stop',
            'system',
            1.0, 0.0,
            self.system_mode
        )
    
    def reset(self):
        """Reset ARVS system to initial state."""
        logger.info("Resetting ARVS system")
        
        # Stop execution
        self.execution_controller.stop_execution()
        
        # Reset components
        self.perception.reset()
        self.state_estimator.reset()
        self.risk_quantifier.reset()
        self.optimization_engine.reset()
        self.mvi_logic.reset()
        self.safety_gate.reset()
        self.fault_detector.reset_all_health()
        self.adaptive_learner.clear_buffer()
        
        # Reset state
        self.current_state = None
        self.current_risk = None
        self.current_mvi = None
        self.execution_results = []
        
        # Reset performance metrics
        self.performance_metrics = {
            'control_cycles': 0,
            'avg_cycle_time': 0.0,
            'replanning_count': 0,
            'safety_violations_prevented': 0,
            'emergency_stops': 0,
            'uptime': 0.0
        }
        
        # Return to normal mode
        old_mode = self.system_mode
        self.system_mode = SystemMode.NORMAL
        
        self.audit_logger.log_mode_change(
            old_mode, self.system_mode, "system_reset"
        )
        
        logger.info("ARVS system reset complete")