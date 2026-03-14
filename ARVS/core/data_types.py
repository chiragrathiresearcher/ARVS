"""
ARVS Core Data Types
Defines all data structures used throughout the ARVS system.
Ensures type safety and clear interfaces between components.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union
from enum import Enum, IntEnum
import numpy as np
from datetime import datetime

class SystemMode(IntEnum):
    """Operational modes of the ARVS system."""
    NORMAL = 0
    DEGRADED = 1
    SAFE_HOLD = 2
    EMERGENCY = 3
    FAULT_RECOVERY = 4

class FaultSeverity(IntEnum):
    """Severity levels for detected faults."""
    NONE = 0
    MINOR = 1
    MODERATE = 2
    MAJOR = 3
    CRITICAL = 4

@dataclass
class RobotState:
    """Complete state representation of a single robot."""
    # Identification
    robot_id: str
    timestamp: float
    
    # Kinematic state
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    orientation: np.ndarray  # quaternion [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    
    # Actuator state
    joint_positions: Dict[str, float]  # joint name -> position
    joint_velocities: Dict[str, float]
    joint_torques: Dict[str, float]
    
    # Environmental state
    temperature: float  # Kelvin
    battery_level: float  # 0.0 to 1.0
    power_consumption: float  # Watts
    
    # Uncertainty/covariance (if available)
    position_cov: Optional[np.ndarray] = None  # 3x3 covariance
    orientation_cov: Optional[np.ndarray] = None  # 3x3 covariance
    
    # Metadata
    confidence: float = 1.0  # Overall state confidence [0, 1]
    
    def __post_init__(self):
        """Validate state after initialization."""
        assert 0.0 <= self.battery_level <= 1.0, "Battery level must be in [0, 1]"
        assert 0.0 <= self.confidence <= 1.0, "Confidence must be in [0, 1]"
        assert self.temperature > 0, "Temperature must be positive"
        assert self.position.shape == (3,), "Position must be 3D vector"
        assert self.velocity.shape == (3,), "Velocity must be 3D vector"

@dataclass
class Observation:
    """Raw observation from sensors."""
    sensor_id: str
    timestamp: float
    data: np.ndarray
    data_type: str  # 'lidar', 'camera', 'imu', 'joint_state', 'thermal', etc.
    covariance: Optional[np.ndarray] = None
    valid: bool = True

@dataclass
class FeatureVector:
    """Minimal state features extracted from observations."""
    timestamp: float
    features: Dict[str, float]  # feature name -> value
    uncertainties: Dict[str, float]  # feature name -> uncertainty
    source_sensors: List[str]  # Which sensors contributed

@dataclass
class RiskAssessment:
    """Quantified risk assessment."""
    timestamp: float
    overall_risk: float  # [0, 1], 0 = no risk, 1 = maximum risk
    component_risks: Dict[str, float]  # component/axis -> risk
    risk_factors: Dict[str, float]  # factor name -> contribution
    confidence: float = 1.0
    
    def is_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if risk is below acceptable threshold."""
        return self.overall_risk <= threshold

@dataclass
class Action:
    """Atomic action representation."""
    action_id: str
    action_type: str  # 'move', 'manipulate', 'wait', 'reconfigure', etc.
    parameters: Dict[str, Any]
    duration: float  # Expected duration in seconds
    priority: int = 0
    
    # Safety properties
    max_torque: Optional[float] = None
    max_velocity: Optional[float] = None
    thermal_load: Optional[float] = None
    power_required: Optional[float] = None

@dataclass
class MVISequence:
    """Minimum Viable Intervention sequence."""
    sequence_id: str
    actions: List[Action]
    expected_duration: float
    predicted_risk: RiskAssessment
    qubo_solution: Optional[np.ndarray] = None  # Raw QUBO solution if available
    
    @property
    def action_count(self) -> int:
        """Number of actions in sequence."""
        return len(self.actions)

@dataclass
class SafetyConstraints:
    """Hard safety constraints for the system."""
    max_torque: Dict[str, float]  # joint name -> max torque
    max_velocity: Dict[str, float]  # joint name -> max velocity
    thermal_limits: Dict[str, float]  # component -> max temperature
    structural_load_limits: Dict[str, float]  # component -> max load
    min_battery: float  # Minimum safe battery level
    collision_zones: List[Dict[str, Any]]  # Forbidden zones
    communication_blackouts: List[Tuple[float, float]]  # [start, end] times
    
    def validate_action(self, action: Action) -> Tuple[bool, List[str]]:
        """Validate an action against constraints."""
        violations = []
        
        # Check torque if specified
        if action.max_torque is not None:
            for joint, torque in action.parameters.get('torques', {}).items():
                if joint in self.max_torque and torque > self.max_torque[joint]:
                    violations.append(f"Torque violation on {joint}: {torque} > {self.max_torque[joint]}")
        
        # Check thermal limits
        if action.thermal_load is not None:
            for component, limit in self.thermal_limits.items():
                # Simplified check - in practice would use thermal model
                if action.thermal_load > limit:
                    violations.append(f"Thermal violation on {component}")
        
        return len(violations) == 0, violations

@dataclass
class SystemTelemetry:
    """Complete system telemetry for auditing."""
    timestamp: float
    system_mode: SystemMode
    robot_state: RobotState
    risk_assessment: RiskAssessment
    active_constraints: SafetyConstraints
    selected_mvi: Optional[MVISequence]
    executed_actions: List[Action]
    fault_status: Dict[str, FaultSeverity]
    optimization_metrics: Dict[str, float]  # solver time, iterations, etc.
    safety_violations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'system_mode': self.system_mode.name,
            'robot_state': {
                'robot_id': self.robot_state.robot_id,
                'position': self.robot_state.position.tolist(),
                'battery': self.robot_state.battery_level
            },
            'risk': self.risk_assessment.overall_risk,
            'selected_mvi_id': self.selected_mvi.sequence_id if self.selected_mvi else None,
            'faults': {k: v.name for k, v in self.fault_status.items()}
        }

@dataclass
class OptimizationProblem:
    """Formal optimization problem definition."""
    problem_id: str
    objective_matrix: np.ndarray  # Q matrix for QUBO: min x^T Q x
    constraint_matrices: List[np.ndarray]  # Additional constraint matrices
    variable_names: List[str]  # Names for each binary variable
    variable_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    time_limit: float = 0.5  # 500ms limit from ARVS document
    
    @property
    def num_variables(self) -> int:
        """Number of variables in the problem."""
        return len(self.variable_names)

@dataclass
class MultiRobotState:
    """State representation for multiple robots."""
    timestamp: float
    robot_states: Dict[str, RobotState]  # robot_id -> state
    relative_positions: Dict[Tuple[str, str], np.ndarray]  # (robot_a, robot_b) -> relative pos
    shared_resources: Dict[str, Any]  # Shared resources status
    coordination_constraints: Dict[str, Any]  # Inter-robot constraints