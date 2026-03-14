import numpy as np
from enum import Enum

class SolverType(Enum):
    """Types of optimization solvers available."""
    CLASSICAL = "classical"
    QUANTUM_ANNEALER = "quantum_annealer"
    QAOA = "qaoa"
    HYBRID = "hybrid"
    TENSOR_NETWORK = "tensor_network"

# Timing constants (from ARVS document)
MAX_REPLANNING_TIME = 0.5  # 500 milliseconds
CONTROL_LOOP_PERIOD = 0.1  # 100 milliseconds
TELEMETRY_RATE = 1.0  # 1 Hz

# Risk thresholds (normalized 0-1)
RISK_THRESHOLD_SAFE = 0.3
RISK_THRESHOLD_WARNING = 0.6
RISK_THRESHOLD_CRITICAL = 0.8
RISK_THRESHOLD_UNACCEPTABLE = 0.95

# Safety margins (from ARVS document Section 5.6)
SAFETY_MARGIN_STRUCTURAL = 1.5  # 50% safety margin
SAFETY_MARGIN_THERMAL = 0.8  # 20% thermal margin
SAFETY_MARGIN_POWER = 0.2  # 20% minimum battery buffer

# QUBO parameters (from ARVS document)
MAX_QUBO_VARIABLES = 20000
MIN_QUBO_VARIABLES = 500
DEFAULT_QUBO_PENALTY_WEIGHT = 100.0

# Fault detection thresholds
FAULT_THRESHOLD_TORQUE = 0.9  # 90% of max torque
FAULT_THRESHOLD_TEMPERATURE = 0.85  # 85% of max temperature
FAULT_THRESHOLD_BATTERY = 0.15  # 15% remaining
FAULT_THRESHOLD_VIBRATION = 0.75  # 75% of max vibration

# Degradation parameters
DEGRADATION_GRACE_PERIODS = 5  # Number of cycles before mode change
MAX_DEGRADED_PERFORMANCE_LOSS = 0.5  # Max 50% performance loss in degraded mode

# Coordinate system conventions
COORDINATE_EPSILON = 1e-6  # Numerical tolerance for coordinate comparisons
ORIENTATION_EPSILON = 1e-3  # Numerical tolerance for orientation comparisons

# Default physical limits (should be overridden by mission-specific config)
DEFAULT_MAX_TORQUE = 100.0  # N-m
DEFAULT_MAX_VELOCITY = 2.0  # m/s
DEFAULT_MAX_ACCELERATION = 1.0  # m/s²
DEFAULT_MAX_TEMPERATURE = 373.0  # Kelvin (100°C)
DEFAULT_MIN_TEMPERATURE = 123.0  # Kelvin (-150°C, from ARVS doc)

# Multi-robot coordination
MAX_ROBOTS_IN_SWARM = 10
MIN_INTER_ROBOT_DISTANCE = 1.0  # meters
MAX_INTER_ROBOT_DISTANCE = 50.0  # meters

# Learning parameters (bounded adaptive learning)
MAX_LEARNING_RATE = 0.01
MIN_TRAINING_SAMPLES = 100
MODEL_UPDATE_INTERVAL = 10.0  # seconds

# Numerical stability
MIN_PROBABILITY = 1e-6
MAX_PROBABILITY = 1.0 - 1e-6
LOG_EPSILON = 1e-10

# Default covariance matrices (if sensor covariance not provided)
DEFAULT_POSITION_COV = np.diag([0.01, 0.01, 0.01])  # 10cm variance
DEFAULT_ORIENTATION_COV = np.diag([0.001, 0.001, 0.001])  # Small variance
DEFAULT_VELOCITY_COV = np.diag([0.1, 0.1, 0.1])  # 0.1 m/s variance

# File paths and logging
TELEMETRY_LOG_PATH = "/tmp/arvs_telemetry.log"
DECISION_LOG_PATH = "/tmp/arvs_decisions.log"
SAFETY_LOG_PATH = "/tmp/arvs_safety.log"