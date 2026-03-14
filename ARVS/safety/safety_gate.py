

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import time

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
    """Robot state for safety checking."""
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
    """Action for safety checking."""
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
    """MVI sequence for safety checking."""
    sequence_id: str
    actions: List[Action]
    expected_duration: float
    predicted_risk: Any  # Simplified for standalone version
    
    @property
    def action_count(self) -> int:
        return len(self.actions)

@dataclass
class SafetyConstraints:
    """Safety constraints."""
    max_torque: Dict[str, float]
    max_velocity: Dict[str, float]
    thermal_limits: Dict[str, float]
    structural_load_limits: Dict[str, float]
    min_battery: float
    collision_zones: List[Dict[str, Any]] = field(default_factory=list)
    communication_blackouts: List[Tuple[float, float]] = field(default_factory=list)
    
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

# ===== SAFETY GATE CLASSES =====

class SafetyViolationType(Enum):
    """Types of safety violations."""
    TORQUE_EXCEEDED = "torque_exceeded"
    LOAD_FACTOR_EXCEEDED = "load_factor_exceeded"
    TEMPERATURE_EXCEEDED = "temperature_exceeded"
    VIBRATION_EXCEEDED = "vibration_exceeded"
    BATTERY_LOW = "battery_low"
    HAZARD_PROXIMITY = "hazard_proximity"
    JOINT_COLLISION = "joint_collision"
    COMMUNICATION_BLACKOUT = "communication_blackout"
    STRUCTURAL_OVERLOAD = "structural_overload"

@dataclass
class SafetyCheckResult:
    """Result of safety check."""
    safe: bool
    violations: List[Tuple[SafetyViolationType, str, float, float]]
    warnings: List[str]
    confidence: float
    
    def has_critical_violations(self) -> bool:
        """Check for critical violations."""
        critical_types = {
            SafetyViolationType.TORQUE_EXCEEDED,
            SafetyViolationType.LOAD_FACTOR_EXCEEDED,
            SafetyViolationType.BATTERY_LOW,
            SafetyViolationType.JOINT_COLLISION
        }
        return any(violation[0] in critical_types for violation in self.violations)
    
    def get_violation_summary(self) -> str:
        """Get human-readable summary."""
        if not self.violations:
            return "No violations"
        summary = []
        for violation_type, component, value, limit in self.violations:
            summary.append(f"{violation_type.value} on {component}: {value:.2f} > {limit:.2f}")
        return "; ".join(summary)

# ===== SAFETY MARGINS =====
SAFETY_MARGIN_STRUCTURAL = 1.5
SAFETY_MARGIN_THERMAL = 0.8
SAFETY_MARGIN_POWER = 0.2

class SafetyGate:
    """
    Safety gate that validates all decisions before execution.
    Complete standalone implementation with no external dependencies.
    """
    
    def __init__(self, safety_constraints: SafetyConstraints):
        """Initialize safety gate."""
        self.safety_constraints = safety_constraints
        
        # Additional safety margins
        self.safety_margins = {
            'torque': 0.8,
            'thermal': 0.9,
            'structural': 0.7,
            'velocity': 0.8,
            'battery': 0.3,
        }
        
        # State tracking
        self.last_safe_state: Optional[RobotState] = None
        self.violation_history: List[Dict] = []
        self.safety_scores: List[float] = []
        
        # Recovery rules
        self.recovery_rules = {
            SafetyViolationType.TORQUE_EXCEEDED: 'reduce_torque',
            SafetyViolationType.TEMPERATURE_EXCEEDED: 'activate_cooling',
            SafetyViolationType.BATTERY_LOW: 'conserve_power',
            SafetyViolationType.HAZARD_PROXIMITY: 'move_away',
            SafetyViolationType.JOINT_COLLISION: 'adjust_pose'
        }
        
        logger.info("Safety gate initialized")
    
    def check_action(self, action: Action, 
                    current_state: RobotState,
                    predicted_state: Optional[RobotState] = None) -> SafetyCheckResult:
        """Check if an action is safe to execute."""
        violations = []
        warnings = []
        
        # Check all constraint types
        violations.extend(self._check_torque_constraints(action, current_state))
        violations.extend(self._check_thermal_constraints(action, current_state))
        violations.extend(self._check_structural_constraints(action, current_state))
        violations.extend(self._check_power_constraints(action, current_state))
        violations.extend(self._check_collision_constraints(action, current_state, predicted_state))
        violations.extend(self._check_hazard_proximity(current_state, predicted_state))
        violations.extend(self._check_joint_collision(action, current_state))
        violations.extend(self._check_communication_rules(action, current_state))
        
        # Determine overall safety
        safe = len(violations) == 0
        
        # Calculate confidence
        confidence = getattr(current_state, 'confidence', 0.8)
        if predicted_state is None:
            confidence *= 0.7
        
        # Update history
        self._update_violation_history(violations, action.action_id)
        
        # Calculate safety score
        safety_score = self._calculate_safety_score(violations, current_state)
        self.safety_scores.append(safety_score)
        if len(self.safety_scores) > 100:
            self.safety_scores.pop(0)
        
        return SafetyCheckResult(
            safe=safe,
            violations=violations,
            warnings=warnings,
            confidence=confidence
        )
    
    def check_mvi_sequence(self, mvi_sequence: MVISequence,
                          current_state: RobotState) -> SafetyCheckResult:
        """Check entire MVI sequence for safety."""
        all_violations = []
        all_warnings = []
        min_confidence = 1.0
        
        # Check each action in sequence
        state = current_state
        for action in mvi_sequence.actions:
            predicted_state = self._predict_state_after_action(state, action)
            result = self.check_action(action, state, predicted_state)
            
            all_violations.extend(result.violations)
            all_warnings.extend(result.warnings)
            min_confidence = min(min_confidence, result.confidence)
            
            # Update state for next iteration
            state = predicted_state
        
        # Check sequence properties
        all_violations.extend(self._check_sequence_properties(mvi_sequence, current_state))
        
        # Overall safety
        safe = len(all_violations) == 0
        
        return SafetyCheckResult(
            safe=safe,
            violations=all_violations,
            warnings=all_warnings,
            confidence=min_confidence
        )
    
    # ===== CONSTRAINT CHECKING METHODS =====
    
    def _check_torque_constraints(self, action: Action,
                                 state: RobotState) -> List[Tuple[SafetyViolationType, str, float, float]]:
        """Check torque constraints."""
        violations = []
        if action.max_torque is None:
            return violations
        
        safety_margin = self.safety_margins['torque']
        
        for joint, max_torque in self.safety_constraints.max_torque.items():
            safe_torque = max_torque * safety_margin
            if action.max_torque > safe_torque:
                violations.append((
                    SafetyViolationType.TORQUE_EXCEEDED,
                    joint,
                    action.max_torque,
                    safe_torque
                ))
        return violations
    
    def _check_thermal_constraints(self, action: Action,
                                  state: RobotState) -> List[Tuple[SafetyViolationType, str, float, float]]:
        """Check thermal constraints."""
        violations = []
        if action.thermal_load is None:
            return violations
        
        predicted_temp = state.temperature + action.thermal_load
        safety_margin = self.safety_margins['thermal']
        
        for component, max_temp in self.safety_constraints.thermal_limits.items():
            safe_temp = max_temp * safety_margin
            if predicted_temp > safe_temp:
                violations.append((
                    SafetyViolationType.TEMPERATURE_EXCEEDED,
                    component,
                    predicted_temp,
                    safe_temp
                ))
        return violations
    
    def _check_structural_constraints(self, action: Action,
                                     state: RobotState) -> List[Tuple[SafetyViolationType, str, float, float]]:
        """Check structural constraints."""
        violations = []
        
        load_factors = []
        if action.max_torque:
            load_factors.append(action.max_torque / 100.0)
        if action.max_velocity:
            load_factors.append(action.max_velocity / 5.0)
        if action.duration:
            load_factors.append(min(1.0, action.duration / 30.0) * 0.5)
        
        if load_factors:
            load_factor = sum(load_factors) / len(load_factors)
            max_load_factor = 1.0 / SAFETY_MARGIN_STRUCTURAL
            
            if load_factor > max_load_factor:
                violations.append((
                    SafetyViolationType.LOAD_FACTOR_EXCEEDED,
                    'structure',
                    load_factor,
                    max_load_factor
                ))
        return violations
    
    def _check_power_constraints(self, action: Action,
                                state: RobotState) -> List[Tuple[SafetyViolationType, str, float, float]]:
        """Check power constraints."""
        violations = []
        if action.power_required is None:
            return violations
        
        energy_required = action.power_required * action.duration / 3600.0
        battery_capacity = 100.0
        predicted_battery = state.battery_level - (energy_required / battery_capacity)
        min_battery = self.safety_constraints.min_battery + SAFETY_MARGIN_POWER
        
        if predicted_battery < min_battery:
            violations.append((
                SafetyViolationType.BATTERY_LOW,
                'battery',
                predicted_battery,
                min_battery
            ))
        return violations
    
    def _check_collision_constraints(self, action: Action,
                                    current_state: RobotState,
                                    predicted_state: Optional[RobotState]) -> List[Tuple]:
        """Check collision constraints."""
        violations = []
        if predicted_state:
            for zone in self.safety_constraints.collision_zones:
                if self._point_in_zone(predicted_state.position, zone):
                    violations.append((
                        SafetyViolationType.HAZARD_PROXIMITY,
                        'collision_zone',
                        1.0,
                        0.0
                    ))
                    break
        return violations
    
    def _check_hazard_proximity(self, current_state: RobotState,
                               predicted_state: Optional[RobotState]) -> List[Tuple]:
        """Check hazard proximity."""
        violations = []
        state_to_check = predicted_state if predicted_state else current_state
        
        for zone in self.safety_constraints.collision_zones:
            distance = self._distance_to_zone(state_to_check.position, zone)
            if 'safe_distance' in zone and distance < zone['safe_distance']:
                violations.append((
                    SafetyViolationType.HAZARD_PROXIMITY,
                    zone.get('name', 'hazard'),
                    distance,
                    zone['safe_distance']
                ))
        return violations
    
    def _check_joint_collision(self, action: Action,
                              state: RobotState) -> List[Tuple]:
        """Check joint collision risk."""
        violations = []
        for joint_name, joint_pos in state.joint_positions.items():
            joint_limits = {'min': -3.14, 'max': 3.14}
            if joint_pos < joint_limits['min'] + 0.1 or joint_pos > joint_limits['max'] - 0.1:
                limit = joint_limits['max'] if joint_pos > 0 else joint_limits['min']
                violations.append((
                    SafetyViolationType.JOINT_COLLISION,
                    joint_name,
                    joint_pos,
                    limit
                ))
        return violations
    
    def _check_communication_rules(self, action: Action,
                                  state: RobotState) -> List[Tuple]:
        """Check communication blackout rules."""
        violations = []
        action_end_time = state.timestamp + action.duration
        
        for blackout_start, blackout_end in self.safety_constraints.communication_blackouts:
            if state.timestamp < blackout_end and action_end_time > blackout_start:
                if not self._is_action_allowed_during_blackout(action):
                    violations.append((
                        SafetyViolationType.COMMUNICATION_BLACKOUT,
                        'communication',
                        action.duration,
                        0.0
                    ))
        return violations
    
    def _check_sequence_properties(self, mvi_sequence: MVISequence,
                                  current_state: RobotState) -> List[Tuple]:
        """Check overall sequence properties."""
        violations = []
        
        if mvi_sequence.expected_duration > 120.0:
            violations.append((
                SafetyViolationType.STRUCTURAL_OVERLOAD,
                'sequence_duration',
                mvi_sequence.expected_duration,
                120.0
            ))
        
        # Check predicted risk if available
        if hasattr(mvi_sequence.predicted_risk, 'overall_risk'):
            if mvi_sequence.predicted_risk.overall_risk > 0.8:
                violations.append((
                    SafetyViolationType.STRUCTURAL_OVERLOAD,
                    'predicted_risk',
                    mvi_sequence.predicted_risk.overall_risk,
                    0.8
                ))
        
        return violations
    
    # ===== UTILITY METHODS =====
    
    def _point_in_zone(self, point: np.ndarray, zone: Dict) -> bool:
        """Check if point is inside a hazard zone."""
        if zone.get('type') == 'cylinder':
            center = np.array(zone.get('center', [0, 0, 0]))
            radius = zone.get('radius', 0)
            height = zone.get('height', 10.0)
            
            distance_2d = np.linalg.norm(point[:2] - center[:2])
            height_diff = abs(point[2] - center[2])
            
            return distance_2d <= radius and height_diff <= height / 2
        return False
    
    def _distance_to_zone(self, point: np.ndarray, zone: Dict) -> float:
        """Calculate distance to hazard zone."""
        if zone.get('type') == 'cylinder':
            center = np.array(zone.get('center', [0, 0, 0]))
            radius = zone.get('radius', 0)
            distance_2d = np.linalg.norm(point[:2] - center[:2])
            return max(0.0, distance_2d - radius)
        return float('inf')
    
    def _is_action_allowed_during_blackout(self, action: Action) -> bool:
        """Check if action is allowed during communication blackout."""
        if action.action_type == 'safety':
            return True
        if (action.power_required and action.power_required < 20.0 and 
            action.duration < 5.0):
            return True
        return False
    
    def _predict_state_after_action(self, state: RobotState,
                                   action: Action) -> RobotState:
        """Simple state prediction after action execution."""
        predicted_state = RobotState(
            robot_id=state.robot_id,
            timestamp=state.timestamp + action.duration,
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            orientation=state.orientation.copy(),
            angular_velocity=state.angular_velocity.copy(),
            joint_positions=state.joint_positions.copy(),
            joint_velocities=state.joint_velocities.copy(),
            joint_torques=state.joint_torques.copy(),
            temperature=state.temperature + (action.thermal_load or 0.0),
            battery_level=max(0.0, state.battery_level - 
                             (action.power_required or 0.0) * action.duration / 3600.0 / 100.0),
            power_consumption=action.power_required or state.power_consumption,
            confidence=getattr(state, 'confidence', 1.0) * 0.9
        )
        return predicted_state
    
    def _calculate_safety_score(self, violations: List[Tuple],
                               state: RobotState) -> float:
        """Calculate safety score (0-1, higher is safer)."""
        if not violations:
            return 1.0
        
        total_penalty = 0.0
        weights = {
            SafetyViolationType.TORQUE_EXCEEDED: 0.3,
            SafetyViolationType.LOAD_FACTOR_EXCEEDED: 0.25,
            SafetyViolationType.TEMPERATURE_EXCEEDED: 0.2,
            SafetyViolationType.BATTERY_LOW: 0.15,
            SafetyViolationType.HAZARD_PROXIMITY: 0.1,
        }
        
        for violation_type, component, value, limit in violations:
            if limit > 0:
                excess = max(0.0, (value - limit) / limit)
            else:
                excess = 1.0 if value > 0 else 0.0
            
            weight = weights.get(violation_type, 0.05)
            total_penalty += excess * weight
        
        safety_score = max(0.0, 1.0 - total_penalty)
        safety_score *= getattr(state, 'confidence', 0.8)
        
        return safety_score
    
    def _update_violation_history(self, violations: List[Tuple],
                                 action_id: str):
        """Update violation history."""
        if violations:
            violation_record = {
                'timestamp': time.time(),
                'action_id': action_id,
                'violations': [(v[0].value, v[1], float(v[2]), float(v[3])) 
                              for v in violations]
            }
            self.violation_history.append(violation_record)
            if len(self.violation_history) > 1000:
                self.violation_history.pop(0)
    
    # ===== PUBLIC METHODS =====
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety metrics and statistics."""
        if not self.safety_scores:
            avg_score = 1.0
        else:
            avg_score = float(np.mean(self.safety_scores))
        
        recent_violations = []
        for record in self.violation_history[-10:]:
            recent_violations.append({
                'action': record['action_id'],
                'count': len(record['violations'])
            })
        
        return {
            'average_safety_score': avg_score,
            'current_safety_score': self.safety_scores[-1] if self.safety_scores else 1.0,
            'total_violations': len(self.violation_history),
            'recent_violations': recent_violations,
            'last_safe_state': self.last_safe_state is not None
        }
    
    def get_recovery_suggestion(self, violation_type: SafetyViolationType) -> str:
        """Get recovery suggestion for a violation type."""
        return self.recovery_rules.get(violation_type, 'unknown')
    
    def update_last_safe_state(self, state: RobotState):
        """Update the last known safe state."""
        self.last_safe_state = RobotState(
            robot_id=state.robot_id,
            timestamp=state.timestamp,
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            orientation=state.orientation.copy(),
            angular_velocity=state.angular_velocity.copy(),
            joint_positions=state.joint_positions.copy(),
            joint_velocities=state.joint_velocities.copy(),
            joint_torques=state.joint_torques.copy(),
            temperature=state.temperature,
            battery_level=state.battery_level,
            power_consumption=state.power_consumption,
            confidence=getattr(state, 'confidence', 1.0)
        )
        logger.debug("Updated last safe state")
    
    def get_last_safe_state(self) -> Optional[RobotState]:
        """Get the last known safe state."""
        return self.last_safe_state
    
    def reset(self):
        """Reset safety gate."""
        self.last_safe_state = None
        self.violation_history = []
        self.safety_scores = []
        logger.info("Safety gate reset")

# ===== DEMONSTRATION =====

def demonstrate_safety_gate():
    """Demonstrate the safety gate functionality."""
    print("=" * 70)
    print("SAFETY GATE DEMONSTRATION")
    print("Complete standalone implementation")
    print("=" * 70)
    
    # Create safety constraints
    constraints = SafetyConstraints(
        max_torque={'joint1': 100.0, 'joint2': 80.0},
        max_velocity={'joint1': 2.0, 'joint2': 1.5},
        thermal_limits={'motor': 373.0, 'cpu': 358.0},
        structural_load_limits={'arm': 200.0},
        min_battery=0.15,
        collision_zones=[
            {
                'type': 'cylinder',
                'center': [10.0, 0.0, 0.0],
                'radius': 2.0,
                'height': 5.0,
                'name': 'hazard_zone',
                'safe_distance': 3.0
            }
        ]
    )
    
    # Create safety gate
    safety_gate = SafetyGate(constraints)
    
    # Create test robot state
    robot_state = RobotState(
        robot_id="test_robot",
        timestamp=time.time(),
        position=np.array([5.0, 0.0, 0.0]),  # 5m from hazard
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
    safe_action = Action(
        action_id="move_slow",
        action_type="motion",
        parameters={'direction': 'forward', 'distance': 1.0},
        duration=2.0,
        max_torque=50.0,
        max_velocity=0.5,
        thermal_load=10.0,
        power_required=40.0
    )
    
    unsafe_action = Action(
        action_id="move_fast",
        action_type="motion",
        parameters={'direction': 'forward', 'distance': 5.0},
        duration=1.0,
        max_torque=120.0,  # Exceeds 100N-m limit
        max_velocity=3.0,   # Exceeds 2.0 m/s limit
        thermal_load=50.0,  # Will cause overheating
        power_required=150.0  # High power consumption
    )
    
    # Test safe action
    print("\n1. Testing SAFE action:")
    result1 = safety_gate.check_action(safe_action, robot_state)
    print(f"   Safe: {result1.safe}")
    print(f"   Violations: {len(result1.violations)}")
    if result1.violations:
        print(f"   Details: {result1.get_violation_summary()}")
    
    # Test unsafe action
    print("\n2. Testing UNSAFE action:")
    result2 = safety_gate.check_action(unsafe_action, robot_state)
    print(f"   Safe: {result2.safe}")
    print(f"   Violations: {len(result2.violations)}")
    if result2.violations:
        print(f"   Details: {result2.get_violation_summary()}")
        print(f"   Critical violations: {result2.has_critical_violations()}")
    
    # Test MVI sequence
    print("\n3. Testing MVI sequence:")
    mvi_sequence = MVISequence(
        sequence_id="test_mvi",
        actions=[safe_action, unsafe_action],
        expected_duration=3.0,
        predicted_risk=type('obj', (object,), {'overall_risk': 0.9})()  # Simulated risk
    )
    
    result3 = safety_gate.check_mvi_sequence(mvi_sequence, robot_state)
    print(f"   Safe: {result3.safe}")
    print(f"   Total violations: {len(result3.violations)}")
    
    # Get safety metrics
    print("\n4. Safety metrics:")
    metrics = safety_gate.get_safety_metrics()
    for key, value in metrics.items():
        if key != 'recent_violations':
            print(f"   {key}: {value}")
    
    # Test recovery suggestions
    print("\n5. Recovery suggestions for violations:")
    for violation_type in SafetyViolationType:
        suggestion = safety_gate.get_recovery_suggestion(violation_type)
        print(f"   {violation_type.value}: {suggestion}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_safety_gate()
    