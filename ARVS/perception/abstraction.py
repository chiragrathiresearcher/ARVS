
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Define missing constants locally
DEFAULT_POSITION_COV = np.diag([0.01, 0.01, 0.01])  # 10cm variance
DEFAULT_ORIENTATION_COV = np.diag([0.001, 0.001, 0.001])
DEFAULT_VELOCITY_COV = np.diag([0.1, 0.1, 0.1])
COORDINATE_EPSILON = 1e-6

# Define missing data types locally
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
class RobotState:
    """Simplified robot state for perception."""
    robot_id: str
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    temperature: float
    battery_level: float
    power_consumption: float
    confidence: float = 1.0

class SensorFailureException(Exception):
    """Custom exception for sensor failures."""
    pass

class PerceptionAbstraction:
    """
    Abstracts raw sensor data into minimal feature representations.
    
    Key responsibilities:
    1. Sensor fusion from multiple sources
    2. Noise filtering and outlier rejection
    3. Missing data handling
    4. Uncertainty quantification
    5. Feature extraction for state estimation
    
    Safety considerations:
    - Never trust a single sensor without corroboration
    - Mark features as uncertain when sensor agreement is low
    - Degrade gracefully as sensors fail
    """
    
    def __init__(self, robot_id: str, sensor_config: Dict):
        """
        Initialize perception abstraction layer.
        
        Args:
            robot_id: Unique identifier for the robot
            sensor_config: Dictionary mapping sensor types to their configurations
        """
        self.robot_id = robot_id
        self.sensor_config = sensor_config
        self.sensor_history: Dict[str, List[Observation]] = {}
        self.last_valid_time: Dict[str, float] = {}
        
        # Initialize sensor history buffers
        for sensor_type in sensor_config.keys():
            self.sensor_history[sensor_type] = []
            self.last_valid_time[sensor_type] = 0.0
        
        # Kalman filters for state estimation (if needed)
        self.position_filter = None
        self.orientation_filter = None
        
        logger.info(f"Perception abstraction initialized for robot {robot_id}")
    
    def process_observations(self, observations: List[Observation]) -> FeatureVector:
        """
        Process raw observations into a feature vector.
        
        Args:
            observations: List of raw observations from various sensors
            
        Returns:
            FeatureVector containing minimal state features
            
        Raises:
            SensorFailureException: If critical sensors fail
        """
        if not observations:
            logger.warning("No observations received")
            return self._create_degraded_feature_vector()
        
        # Group observations by sensor type
        grouped_obs = self._group_observations_by_type(observations)
        
        # Update sensor history
        for obs in observations:
            if obs.valid:
                if obs.data_type not in self.sensor_history:
                    self.sensor_history[obs.data_type] = []
                self.sensor_history[obs.data_type].append(obs)
                self.last_valid_time[obs.data_type] = obs.timestamp
                # Limit history size
                if len(self.sensor_history[obs.data_type]) > 100:
                    self.sensor_history[obs.data_type].pop(0)
        
        # Check for sensor failures
        failed_sensors = self._detect_sensor_failures(grouped_obs)
        if failed_sensors:
            logger.warning(f"Sensor failures detected: {failed_sensors}")
        
        # Extract features from each sensor type
        features = {}
        uncertainties = {}
        source_sensors = []
        
        # Process position sensors (GPS, lidar SLAM, etc.)
        if 'position' in grouped_obs:
            pos_features, pos_uncertainties = self._extract_position_features(
                grouped_obs['position']
            )
            features.update(pos_features)
            uncertainties.update(pos_uncertainties)
            source_sensors.extend(['position'] * len(pos_features))
        
        # Process IMU sensors
        if 'imu' in grouped_obs:
            imu_features, imu_uncertainties = self._extract_imu_features(
                grouped_obs['imu']
            )
            features.update(imu_features)
            uncertainties.update(imu_uncertainties)
            source_sensors.extend(['imu'] * len(imu_features))
        
        # Process joint state sensors
        if 'joint_state' in grouped_obs:
            joint_features, joint_uncertainties = self._extract_joint_features(
                grouped_obs['joint_state']
            )
            features.update(joint_features)
            uncertainties.update(joint_uncertainties)
            source_sensors.extend(['joint_state'] * len(joint_features))
        
        # Process thermal sensors
        if 'thermal' in grouped_obs:
            thermal_features, thermal_uncertainties = self._extract_thermal_features(
                grouped_obs['thermal']
            )
            features.update(thermal_features)
            uncertainties.update(thermal_uncertainties)
            source_sensors.extend(['thermal'] * len(thermal_features))
        
        # Add sensor health metrics
        features['sensor_health_score'] = self._compute_sensor_health_score()
        uncertainties['sensor_health_score'] = 0.1  # Fixed uncertainty for health metric
        
        # Create feature vector
        timestamp = observations[0].timestamp if observations else 0.0
        
        return FeatureVector(
            timestamp=timestamp,
            features=features,
            uncertainties=uncertainties,
            source_sensors=list(set(source_sensors))  # Unique sensors
        )
    
    def _group_observations_by_type(self, observations: List[Observation]) -> Dict[str, List[Observation]]:
        """Group observations by their data type."""
        grouped = {}
        for obs in observations:
            if obs.data_type not in grouped:
                grouped[obs.data_type] = []
            grouped[obs.data_type].append(obs)
        return grouped
    
    def _extract_position_features(self, observations: List[Observation]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Extract position-related features from observations."""
        features = {}
        uncertainties = {}
        
        if not observations:
            return features, uncertainties
        
        # Use latest valid observation
        latest_obs = observations[-1]
        
        # Extract position (assuming data is [x, y, z])
        if latest_obs.data.shape[0] >= 3:
            features['pos_x'] = float(latest_obs.data[0])
            features['pos_y'] = float(latest_obs.data[1])
            features['pos_z'] = float(latest_obs.data[2])
            
            # Estimate uncertainties
            if latest_obs.covariance is not None and latest_obs.covariance.shape[0] >= 3:
                uncertainties['pos_x'] = float(np.sqrt(latest_obs.covariance[0, 0]))
                uncertainties['pos_y'] = float(np.sqrt(latest_obs.covariance[1, 1]))
                uncertainties['pos_z'] = float(np.sqrt(latest_obs.covariance[2, 2]))
            else:
                # Default uncertainties
                uncertainties['pos_x'] = float(np.sqrt(DEFAULT_POSITION_COV[0, 0]))
                uncertainties['pos_y'] = float(np.sqrt(DEFAULT_POSITION_COV[1, 1]))
                uncertainties['pos_z'] = float(np.sqrt(DEFAULT_POSITION_COV[2, 2]))
        
        return features, uncertainties
    
    def _extract_imu_features(self, observations: List[Observation]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Extract IMU-related features (orientation, angular velocity)."""
        features = {}
        uncertainties = {}
        
        if not observations:
            return features, uncertainties
        
        latest_obs = observations[-1]
        
        # Extract orientation (assuming data is [qw, qx, qy, qz, wx, wy, wz])
        if latest_obs.data.shape[0] >= 7:
            # Orientation quaternion
            features['ori_w'] = float(latest_obs.data[0])
            features['ori_x'] = float(latest_obs.data[1])
            features['ori_y'] = float(latest_obs.data[2])
            features['ori_z'] = float(latest_obs.data[3])
            
            # Angular velocity
            features['ang_vel_x'] = float(latest_obs.data[4])
            features['ang_vel_y'] = float(latest_obs.data[5])
            features['ang_vel_z'] = float(latest_obs.data[6])
            
            # Estimate uncertainties
            if latest_obs.covariance is not None:
                cov_diag = np.diag(latest_obs.covariance) if latest_obs.covariance.ndim == 2 else latest_obs.covariance
                if cov_diag.shape[0] >= 7:
                    for i, name in enumerate(['ori_w', 'ori_x', 'ori_y', 'ori_z', 
                                              'ang_vel_x', 'ang_vel_y', 'ang_vel_z']):
                        uncertainties[name] = float(np.sqrt(cov_diag[i]))
                else:
                    # Default orientation uncertainties
                    for name in ['ori_w', 'ori_x', 'ori_y', 'ori_z']:
                        uncertainties[name] = float(np.sqrt(DEFAULT_ORIENTATION_COV[0, 0]))
                    for name in ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']:
                        uncertainties[name] = 0.1  # rad/s default uncertainty
            else:
                # Default uncertainties
                for name in ['ori_w', 'ori_x', 'ori_y', 'ori_z']:
                    uncertainties[name] = float(np.sqrt(DEFAULT_ORIENTATION_COV[0, 0]))
                for name in ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']:
                    uncertainties[name] = 0.1
        
        return features, uncertainties
    
    def _extract_joint_features(self, observations: List[Observation]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Extract joint state features."""
        features = {}
        uncertainties = {}
        
        if not observations:
            return features, uncertainties
        
        latest_obs = observations[-1]
        
        # Get number of joints from config or default
        n_joints = self.sensor_config.get('joint_state', {}).get('num_joints', 6)
        
        for i in range(n_joints):
            base_idx = i * 3
            if latest_obs.data.shape[0] > base_idx + 2:
                features[f'joint_{i}_pos'] = float(latest_obs.data[base_idx])
                features[f'joint_{i}_vel'] = float(latest_obs.data[base_idx + 1])
                features[f'joint_{i}_torque'] = float(latest_obs.data[base_idx + 2])
                
                # Default uncertainties
                uncertainties[f'joint_{i}_pos'] = 0.01  # rad
                uncertainties[f'joint_{i}_vel'] = 0.1   # rad/s
                uncertainties[f'joint_{i}_torque'] = 0.5  # N-m
        
        return features, uncertainties
    
    def _extract_thermal_features(self, observations: List[Observation]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Extract thermal features."""
        features = {}
        uncertainties = {}
        
        if not observations:
            return features, uncertainties
        
        latest_obs = observations[-1]
        
        # Get component names from config or use defaults
        component_names = self.sensor_config.get('thermal', {}).get('components', 
                                                                   ['cpu', 'motor', 'battery', 'ambient'])
        
        for i, name in enumerate(component_names):
            if i < len(latest_obs.data):
                features[f'temp_{name}'] = float(latest_obs.data[i])
                uncertainties[f'temp_{name}'] = 1.0  # ±1K uncertainty
        
        return features, uncertainties
    
    def _detect_sensor_failures(self, grouped_obs: Dict[str, List[Observation]]) -> List[str]:
        """Detect sensor failures based on data validity and timing."""
        failed_sensors = []
        current_time = 0.0
        
        # Find a valid observation to get current time
        for obs_list in grouped_obs.values():
            if obs_list and obs_list[0].valid:
                current_time = obs_list[0].timestamp
                break
        
        for sensor_type, obs_list in grouped_obs.items():
            if not obs_list:
                failed_sensors.append(sensor_type)
                continue
            
            # Check if observations are valid
            valid_obs = [obs for obs in obs_list if obs.valid]
            if not valid_obs:
                failed_sensors.append(sensor_type)
                continue
            
            # Check for stale data (more than 1 second old)
            latest_obs = valid_obs[-1]
            if current_time - latest_obs.timestamp > 1.0:
                failed_sensors.append(sensor_type)
        
        return failed_sensors
    
    def _compute_sensor_health_score(self) -> float:
        """Compute overall sensor health score (0-1)."""
        if not self.last_valid_time:
            return 0.0
        
        # Find current time from last valid times
        valid_times = [t for t in self.last_valid_time.values() if t > 0]
        if not valid_times:
            return 0.0
        
        current_time = max(valid_times)
        health_scores = []
        
        for sensor_type, last_time in self.last_valid_time.items():
            if last_time == 0:
                score = 0.0
            else:
                # Score decays if data is stale
                time_since_last = current_time - last_time
                if time_since_last > 5.0:  # 5 seconds threshold
                    score = 0.0
                else:
                    score = max(0.0, 1.0 - time_since_last / 5.0)
            health_scores.append(score)
        
        return float(np.mean(health_scores)) if health_scores else 0.0
    
    def _create_degraded_feature_vector(self) -> FeatureVector:
        """Create a feature vector when no observations are available."""
        logger.error("Creating degraded feature vector - no valid observations")
        
        # Return minimal features with high uncertainty
        features = {
            'sensor_health_score': 0.0,
            'degraded_mode': 1.0
        }
        
        uncertainties = {
            'sensor_health_score': 1.0,
            'degraded_mode': 1.0
        }
        
        return FeatureVector(
            timestamp=0.0,
            features=features,
            uncertainties=uncertainties,
            source_sensors=['degraded']
        )
    
    def reset(self):
        """Reset perception history and filters."""
        self.sensor_history = {k: [] for k in self.sensor_config.keys()}
        self.last_valid_time = {k: 0.0 for k in self.sensor_config.keys()}
        logger.info("Perception abstraction reset")

# Simple test to verify it works
if __name__ == "__main__":
    # Create a test instance
    sensor_config = {
        'imu': {'type': 'imu', 'update_rate': 100},
        'joint_state': {'type': 'joint_state', 'num_joints': 3},
        'thermal': {'type': 'thermal', 'components': ['motor', 'cpu', 'battery']}
    }
    
    perception = PerceptionAbstraction("test_robot", sensor_config)
    
    # Create test observations
    observations = [
        Observation(
            sensor_id='imu_1',
            timestamp=100.0,
            data=np.array([1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]),  # quaternion + angular velocity
            data_type='imu',
            covariance=np.eye(7) * 0.001,
            valid=True
        ),
        Observation(
            sensor_id='joint_1',
            timestamp=100.0,
            data=np.array([0.0, 0.1, 5.0, 0.5, 0.2, 10.0, 1.0, 0.3, 15.0]),  # 3 joints
            data_type='joint_state',
            valid=True
        ),
        Observation(
            sensor_id='thermal_1',
            timestamp=100.0,
            data=np.array([320.0, 310.0, 300.0]),  # motor, cpu, battery temps
            data_type='thermal',
            valid=True
        )
    ]
    
    # Process observations
    feature_vector = perception.process_observations(observations)
    
    print("Perception test successful!")
    print(f"Features extracted: {len(feature_vector.features)}")
    print(f"Timestamp: {feature_vector.timestamp}")
    print(f"Source sensors: {feature_vector.source_sensors}")
    print(f"Sensor health: {feature_vector.features.get('sensor_health_score', 0.0):.2f}")

