"""
State Estimation & Belief Update - Complete Standalone Version
Maintains probabilistic belief over system state with explicit uncertainty propagation.
Implements Bayesian filtering for state estimation.
No import errors - works completely standalone.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass, field
from enum import Enum
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CONSTANTS =====
DEFAULT_POSITION_COV = np.diag([0.01, 0.01, 0.01])  # 10cm variance
DEFAULT_ORIENTATION_COV = np.diag([0.001, 0.001, 0.001])
DEFAULT_VELOCITY_COV = np.diag([0.1, 0.1, 0.1])
COORDINATE_EPSILON = 1e-6
ORIENTATION_EPSILON = 1e-3

# ===== CORE DATA TYPES =====

@dataclass
class RobotState:
    """Robot state for estimation."""
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
            self.position = np.array(self.position, dtype=np.float64)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity, dtype=np.float64)
        if not isinstance(self.orientation, np.ndarray):
            self.orientation = np.array(self.orientation, dtype=np.float64)
        if not isinstance(self.angular_velocity, np.ndarray):
            self.angular_velocity = np.array(self.angular_velocity, dtype=np.float64)

@dataclass
class FeatureVector:
    """Feature vector from perception."""
    timestamp: float
    features: Dict[str, float]
    uncertainties: Dict[str, float]
    source_sensors: List[str]

@dataclass
class BeliefState:
    """Probabilistic belief about the system state."""
    mean_state: RobotState
    covariance: Dict[str, np.ndarray]
    timestamp: float
    confidence: float
    
    def validate(self) -> bool:
        """Validate belief state for consistency."""
        try:
            # Check covariance matrices
            for name, cov in self.covariance.items():
                if cov is not None:
                    # Check if square matrix
                    if cov.shape[0] != cov.shape[1]:
                        logger.warning(f"Covariance matrix {name} not square")
                        return False
            
            # Check confidence bounds
            if not 0.0 <= self.confidence <= 1.0:
                logger.warning(f"Confidence out of bounds: {self.confidence}")
                return False
            
            # Check timestamp
            if self.timestamp < 0:
                logger.warning(f"Invalid timestamp: {self.timestamp}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Belief validation failed: {e}")
            return False

# ===== STATE ESTIMATOR =====

class StateEstimator:
    """
    Bayesian state estimator with explicit uncertainty propagation.
    Implements Extended Kalman Filter (EKF) for nonlinear state estimation.
    """
    
    def __init__(self, robot_id: str, initial_state: Optional[RobotState] = None):
        """Initialize state estimator."""
        self.robot_id = robot_id
        self.belief_history: List[BeliefState] = []
        self.max_history_length = 100
        
        # Initialize with default state if not provided
        if initial_state is None:
            initial_state = self._create_default_state()
        
        # Initial covariance
        initial_cov = {
            'position': DEFAULT_POSITION_COV.copy(),
            'orientation': DEFAULT_ORIENTATION_COV.copy(),
            'velocity': DEFAULT_VELOCITY_COV.copy()
        }
        
        self.current_belief = BeliefState(
            mean_state=initial_state,
            covariance=initial_cov,
            timestamp=initial_state.timestamp,
            confidence=1.0
        )
        
        # Process noise covariance
        self.process_noise = {
            'position': np.diag([0.01, 0.01, 0.01]),
            'orientation': np.diag([0.001, 0.001, 0.001]),
            'velocity': np.diag([0.1, 0.1, 0.1])
        }
        
        # Filter parameters
        self.filter_divergence_threshold = 10.0
        self.min_confidence = 0.1
        
        logger.info(f"State estimator initialized for robot {robot_id}")
    
    def _create_default_state(self) -> RobotState:
        """Create a default robot state."""
        return RobotState(
            robot_id=self.robot_id,
            timestamp=0.0,
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
            temperature=293.0,
            battery_level=1.0,
            power_consumption=0.0
        )
    
    def update_belief(self, features: FeatureVector, 
                     control_input: Optional[Dict] = None) -> BeliefState:
        """
        Update belief state with new feature observations.
        
        Args:
            features: Feature vector from perception
            control_input: Optional control inputs
            
        Returns:
            Updated belief state
        """
        try:
            # Prediction step (if control input available)
            if control_input is not None:
                self._prediction_step(control_input, features.timestamp)
            
            # Update step with new features
            self._update_step(features)
            
            # Validate belief
            if not self.current_belief.validate():
                logger.error("Belief validation failed after update")
                # Don't raise exception, just reduce confidence
                self.current_belief.confidence *= 0.5
            
            # Store in history
            self.belief_history.append(self.current_belief)
            if len(self.belief_history) > self.max_history_length:
                self.belief_history.pop(0)
            
            # Check for filter divergence
            if self._check_filter_divergence():
                logger.warning("Filter divergence detected, resetting confidence")
                self.current_belief.confidence = max(
                    self.min_confidence,
                    self.current_belief.confidence * 0.5
                )
            
            return self.current_belief
            
        except Exception as e:
            logger.error(f"Belief update failed: {e}")
            # Maintain current belief but reduce confidence
            self.current_belief.confidence = max(
                self.min_confidence,
                self.current_belief.confidence * 0.7
            )
            return self.current_belief
    
    def _prediction_step(self, control_input: Dict, timestamp: float):
        """Predict state forward using control inputs."""
        dt = timestamp - self.current_belief.timestamp
        if dt <= 0:
            return
        
        state = self.current_belief.mean_state
        
        # Simplified dynamics prediction
        if 'acceleration' in control_input:
            acc = np.array(control_input['acceleration'])
            if len(acc) == 3:
                state.position += state.velocity * dt + 0.5 * acc * dt * dt
                state.velocity += acc * dt
        
        if 'angular_acceleration' in control_input:
            ang_acc = np.array(control_input['angular_acceleration'])
            if len(ang_acc) == 3:
                delta_angle = state.angular_velocity * dt + 0.5 * ang_acc * dt * dt
                state.angular_velocity += ang_acc * dt
                
                # Update quaternion
                angle = np.linalg.norm(delta_angle)
                if angle > ORIENTATION_EPSILON:
                    axis = delta_angle / angle
                    delta_q = self._axis_angle_to_quaternion(axis, angle)
                    state.orientation = self._quaternion_multiply(delta_q, state.orientation)
        
        # Update joint positions
        if 'joint_velocities' in control_input:
            for joint, vel in control_input['joint_velocities'].items():
                if joint in state.joint_positions:
                    state.joint_positions[joint] += vel * dt
        
        # Update timestamp
        state.timestamp = timestamp
        
        # Propagate covariance (simplified)
        for key in self.current_belief.covariance.keys():
            if key in self.process_noise:
                self.current_belief.covariance[key] += self.process_noise[key] * dt
        
        # Reduce confidence due to prediction uncertainty
        self.current_belief.confidence *= 0.99
    
    def _update_step(self, features: FeatureVector):
        """Update belief with new measurements."""
        # Extract measurement vector and covariance
        z, R = self._features_to_measurement(features)
        
        if z is None or R is None:
            logger.warning("Invalid features for update")
            return
        
        # Current state vector and covariance
        x, P = self._state_to_estimation_vector()
        
        # Measurement model (simplified)
        n_states = len(x)
        n_meas = len(z)
        H = np.eye(min(n_states, n_meas))
        
        # Pad H if needed
        if n_states > n_meas:
            H = np.hstack([H, np.zeros((n_meas, n_states - n_meas))])
        elif n_meas > n_states:
            H = np.vstack([H, np.zeros((n_meas - n_states, n_states))])
        
        # Kalman gain
        S = H @ P @ H.T + R
        try:
            K = P @ H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed in Kalman gain")
            return
        
        # State update
        y = z - H @ x
        x_updated = x + K @ y
        P_updated = (np.eye(len(x)) - K @ H) @ P
        
        # Update belief from estimation vector
        self._estimation_vector_to_state(x_updated, P_updated)
        
        # Update confidence based on innovation
        innovation_norm = np.linalg.norm(y)
        innovation_confidence = np.exp(-0.1 * innovation_norm)
        self.current_belief.confidence = min(1.0, self.current_belief.confidence * innovation_confidence)
    
    def _features_to_measurement(self, features: FeatureVector) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert features to measurement vector and covariance."""
        feature_names = list(features.features.keys())
        
        if not feature_names:
            return None, None
        
        # Create measurement vector
        z = np.array([features.features[name] for name in feature_names])
        
        # Create measurement covariance
        R_diag = np.array([features.uncertainties.get(name, 1.0)**2 
                          for name in feature_names])
        R = np.diag(R_diag)
        
        return z, R
    
    def _state_to_estimation_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert current belief to estimation vector and covariance."""
        state = self.current_belief.mean_state
        
        # Create state vector [pos, vel, ori, ang_vel]
        x = np.concatenate([
            state.position,
            state.velocity,
            state.orientation,
            state.angular_velocity
        ])
        
        # Create block-diagonal covariance matrix (fixed version)
        # Get individual covariance matrices
        pos_cov = self.current_belief.covariance['position']
        vel_cov = self.current_belief.covariance['velocity']
        ori_cov = self.current_belief.covariance['orientation']
        ang_vel_cov = np.diag([0.01, 0.01, 0.01])  # Default for angular velocity
        
        # Ensure all are 2D arrays
        if pos_cov.ndim == 1:
            pos_cov = np.diag(pos_cov)
        if vel_cov.ndim == 1:
            vel_cov = np.diag(vel_cov)
        if ori_cov.ndim == 1:
            ori_cov = np.diag(ori_cov)
        
        # Create block diagonal matrix manually
        n_pos = pos_cov.shape[0]
        n_vel = vel_cov.shape[0]
        n_ori = ori_cov.shape[0]
        n_ang = ang_vel_cov.shape[0]
        
        total_size = n_pos + n_vel + n_ori + n_ang
        P = np.zeros((total_size, total_size))
        
        # Fill diagonal blocks
        start = 0
        P[start:start+n_pos, start:start+n_pos] = pos_cov
        start += n_pos
        P[start:start+n_vel, start:start+n_vel] = vel_cov
        start += n_vel
        P[start:start+n_ori, start:start+n_ori] = ori_cov
        start += n_ori
        P[start:start+n_ang, start:start+n_ang] = ang_vel_cov
        
        return x, P
    
    def _estimation_vector_to_state(self, x: np.ndarray, P: np.ndarray):
        """Update belief from estimation vector."""
        state = self.current_belief.mean_state
        
        # Extract state components
        idx = 0
        state.position = x[idx:idx+3]; idx += 3
        state.velocity = x[idx:idx+3]; idx += 3
        state.orientation = x[idx:idx+4]; idx += 4
        state.angular_velocity = x[idx:idx+3]; idx += 3
        
        # Normalize quaternion
        quat_norm = np.linalg.norm(state.orientation)
        if quat_norm > 0:
            state.orientation = state.orientation / quat_norm
        
        # Update covariance (extract blocks)
        idx = 0
        self.current_belief.covariance['position'] = P[idx:idx+3, idx:idx+3]; idx += 3
        self.current_belief.covariance['velocity'] = P[idx:idx+3, idx:idx+3]; idx += 3
        self.current_belief.covariance['orientation'] = P[idx:idx+4, idx:idx+4]; idx += 4
    
    def _check_filter_divergence(self) -> bool:
        """Check for Kalman filter divergence."""
        if len(self.belief_history) < 2:
            return False
        
        # Check if covariance is growing unbounded
        try:
            current_cov_norm = np.linalg.norm(self.current_belief.covariance['position'])
            prev_cov_norm = np.linalg.norm(self.belief_history[-2].covariance['position'])
            
            if current_cov_norm > prev_cov_norm * 2.0:
                return True
        except:
            pass
        
        return False
    
    def _axis_angle_to_quaternion(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Convert axis-angle to quaternion."""
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        xyz = axis * sin_half
        return np.array([w, xyz[0], xyz[1], xyz[2]])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def get_current_state(self) -> RobotState:
        """Get current most likely state."""
        return self.current_belief.mean_state
    
    def get_current_belief(self) -> BeliefState:
        """Get current full belief state."""
        return self.current_belief
    
    def reset(self, new_state: Optional[RobotState] = None):
        """Reset state estimator to new state."""
        if new_state is None:
            new_state = self._create_default_state()
        
        initial_cov = {
            'position': DEFAULT_POSITION_COV.copy(),
            'orientation': DEFAULT_ORIENTATION_COV.copy(),
            'velocity': DEFAULT_VELOCITY_COV.copy()
        }
        
        self.current_belief = BeliefState(
            mean_state=new_state,
            covariance=initial_cov,
            timestamp=new_state.timestamp,
            confidence=1.0
        )
        
        self.belief_history = []
        logger.info("State estimator reset")
    
    def get_estimation_metrics(self) -> Dict[str, Any]:
        """Get estimation performance metrics."""
        if not self.belief_history:
            return {
                'confidence': self.current_belief.confidence,
                'position_uncertainty': np.trace(self.current_belief.covariance['position']),
                'history_size': 0
            }
        
        # Calculate average confidence
        confidences = [belief.confidence for belief in self.belief_history[-10:]]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'current_confidence': self.current_belief.confidence,
            'average_confidence': float(avg_confidence),
            'position_uncertainty': float(np.trace(self.current_belief.covariance['position'])),
            'orientation_uncertainty': float(np.trace(self.current_belief.covariance['orientation'])),
            'history_size': len(self.belief_history),
            'filter_divergence': self._check_filter_divergence()
        }

# ===== DEMONSTRATION =====

def demonstrate_state_estimation():
    """Demonstrate the state estimator functionality."""
    print("=" * 70)
    print("STATE ESTIMATION DEMONSTRATION")
    print("Extended Kalman Filter with uncertainty propagation")
    print("=" * 70)
    
    # Create state estimator
    estimator = StateEstimator("test_robot")
    
    # Create initial features
    features = FeatureVector(
        timestamp=time.time(),
        features={
            'pos_x': 0.0,
            'pos_y': 0.0,
            'pos_z': 0.0,
            'vel_x': 0.1,
            'vel_y': 0.0,
            'vel_z': 0.0,
            'temperature': 300.0
        },
        uncertainties={
            'pos_x': 0.05,
            'pos_y': 0.05,
            'pos_z': 0.05,
            'vel_x': 0.02,
            'vel_y': 0.02,
            'vel_z': 0.02,
            'temperature': 1.0
        },
        source_sensors=['gps', 'imu', 'thermal']
    )
    
    # Initial update
    print("\n1. Initial update with sensor measurements:")
    belief = estimator.update_belief(features)
    state = estimator.get_current_state()
    print(f"   Position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
    print(f"   Velocity: [{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}]")
    print(f"   Confidence: {belief.confidence:.3f}")
    
    # Prediction with control input
    print("\n2. Prediction with control input:")
    control_input = {
        'acceleration': [0.1, 0.0, 0.0],
        'angular_acceleration': [0.0, 0.0, 0.05]
    }
    
    # Update timestamp
    features.timestamp += 0.1
    
    # Update with control
    belief = estimator.update_belief(features, control_input)
    state = estimator.get_current_state()
    print(f"   Position after prediction: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
    print(f"   Confidence after prediction: {belief.confidence:.3f}")
    
    # Multiple updates
    print("\n3. Simulating multiple updates (robot moving forward):")
    for i in range(5):
        # Simulate robot moving
        features.timestamp += 0.1
        features.features['pos_x'] += 0.1
        features.features['vel_x'] = 1.0 + i * 0.1
        
        # Add some noise to simulate real sensors
        for key in features.features:
            if 'pos' in key or 'vel' in key:
                features.features[key] += np.random.normal(0, 0.01)
        
        belief = estimator.update_belief(features, control_input)
        
        if i % 2 == 0:
            state = estimator.get_current_state()
            print(f"   Step {i+1}: Pos x = {state.position[0]:.2f}, Confidence = {belief.confidence:.3f}")
    
    # Get final metrics
    print("\n4. Estimation metrics:")
    metrics = estimator.get_estimation_metrics()
    for key, value in metrics.items():
        print(f"   {key}: {value}")
    
    # Test reset
    print("\n5. Testing reset functionality:")
    estimator.reset()
    state = estimator.get_current_state()
    print(f"   Reset position: [{state.position[0]:.2f}, {state.position[1]:.2f}, {state.position[2]:.2f}]")
    print(f"   Reset confidence: {estimator.current_belief.confidence:.3f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_state_estimation()
    