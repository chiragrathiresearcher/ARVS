"""
Feature Extraction
Extracts minimal state features from processed perception data.
"""

import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extracts minimal feature representations for state estimation.
    
    Key principles:
    1. Extract only necessary features for decision making
    2. Maintain uncertainty information
    3. Handle missing data gracefully
    4. Normalize features for consistency
    """
    
    def __init__(self, robot_id: str):
        self.robot_id = robot_id
        self.feature_history = []
        self.max_history = 100
        
        # Feature normalization parameters
        self.normalization_params = {}
        
        logger.info(f"Feature extractor initialized for {robot_id}")
    
    def extract_features(self, processed_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract minimal features from processed perception data.
        
        Args:
            processed_data: Dictionary of processed sensor data
            
        Returns:
            Dictionary of extracted features with values normalized [0, 1]
        """
        features = {}
        
        try:
            # Position features
            if 'position' in processed_data:
                pos = np.array(processed_data['position'])
                features['pos_x'] = float(pos[0])
                features['pos_y'] = float(pos[1])
                features['pos_z'] = float(pos[2])
                features['pos_magnitude'] = float(np.linalg.norm(pos))
            
            # Velocity features
            if 'velocity' in processed_data:
                vel = np.array(processed_data['velocity'])
                features['vel_x'] = float(vel[0])
                features['vel_y'] = float(vel[1])
                features['vel_z'] = float(vel[2])
                features['speed'] = float(np.linalg.norm(vel))
            
            # Orientation features
            if 'orientation' in processed_data:
                ori = np.array(processed_data['orientation'])
                features['ori_w'] = float(ori[0])
                features['ori_x'] = float(ori[1])
                features['ori_y'] = float(ori[2])
                features['ori_z'] = float(ori[3])
            
            # Environmental features
            if 'temperature' in processed_data:
                features['temperature'] = float(processed_data['temperature'])
            
            if 'battery' in processed_data:
                features['battery_level'] = float(processed_data['battery'])
            
            # Sensor health features
            if 'sensor_health' in processed_data:
                features['sensor_health'] = float(processed_data['sensor_health'])
            
            # Obstacle features
            if 'obstacles' in processed_data:
                obstacles = processed_data['obstacles']
                if obstacles:
                    # Closest obstacle distance
                    distances = [obs.get('distance', float('inf')) for obs in obstacles]
                    if distances:
                        features['closest_obstacle'] = float(min(distances))
            
            # Normalize features
            features = self._normalize_features(features)
            
            # Store in history
            self.feature_history.append(features)
            if len(self.feature_history) > self.max_history:
                self.feature_history.pop(0)
            
            logger.debug(f"Extracted {len(features)} features")
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            features = self._get_default_features()
        
        return features
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features to [0, 1] range.
        
        Args:
            features: Raw feature dictionary
            
        Returns:
            Normalized feature dictionary
        """
        normalized = {}
        
        # Define normalization ranges for common features
        normalization_ranges = {
            'pos_x': (-100.0, 100.0),      # meters
            'pos_y': (-100.0, 100.0),
            'pos_z': (-10.0, 10.0),
            'speed': (0.0, 5.0),           # m/s
            'temperature': (200.0, 400.0),  # Kelvin
            'battery_level': (0.0, 1.0),
            'closest_obstacle': (0.0, 50.0)  # meters
        }
        
        for name, value in features.items():
            if name in normalization_ranges:
                min_val, max_val = normalization_ranges[name]
                if max_val > min_val:
                    normalized_val = (value - min_val) / (max_val - min_val)
                    normalized_val = max(0.0, min(1.0, normalized_val))
                    normalized[name] = normalized_val
                else:
                    normalized[name] = 0.0
            else:
                normalized[name] = value
        
        return normalized
    
    def _get_default_features(self) -> Dict[str, float]:
        """
        Get default features when extraction fails.
        
        Returns:
            Default feature dictionary with safe values
        """
        return {
            'pos_x': 0.5,           # Center of normalized range
            'pos_y': 0.5,
            'pos_z': 0.5,
            'speed': 0.0,           # Stopped
            'temperature': 0.5,     # Middle of range
            'battery_level': 0.7,   # 70% battery
            'sensor_health': 0.0,   # Unknown health
            'degraded_mode': 1.0    # In degraded mode
        }
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about extracted features.
        
        Returns:
            Feature statistics dictionary
        """
        if not self.feature_history:
            return {'error': 'No feature history available'}
        
        stats = {}
        feature_names = list(self.feature_history[0].keys())
        
        for name in feature_names:
            values = [entry.get(name, 0.0) for entry in self.feature_history if name in entry]
            if values:
                stats[name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values)
                }
        
        return {
            'robot_id': self.robot_id,
            'total_features_extracted': sum(len(entry) for entry in self.feature_history),
            'feature_statistics': stats,
            'history_size': len(self.feature_history)
        }
    
    def reset(self):
        """Reset feature extractor."""
        self.feature_history = []
        logger.info("Feature extractor reset")
        