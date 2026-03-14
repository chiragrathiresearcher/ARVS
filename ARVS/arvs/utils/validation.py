

import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union, Callable
import logging
import time
from dataclasses import is_dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Exception for validation failures."""
    pass

class SpacecraftValidator:
    """
    Spacecraft-grade validation system.
    
    Features:
    1. Multi-level validation (range, type, physics, mission)
    2. Graceful degradation on validation failure
    3. Historical tracking of validation issues
    4. Automatic correction of minor issues
    """
    
    def __init__(self):
        self.validation_history = []
        self.correction_count = 0
        self.rejection_count = 0
        self.anomaly_detected = False
        
    def validate_robot_state(self, state: Any, context: Dict = None) -> Tuple[bool, List[str]]:
        """
        Comprehensive robot state validation.
        
        Args:
            state: RobotState object to validate
            context: Additional context (mission phase, etc.)
            
        Returns:
            Tuple of (is_valid, list_of_warnings)
        """
        warnings = []
        
        try:
            # 1. Type validation
            if not hasattr(state, 'robot_id'):
                raise ValidationError("State missing robot_id")
            
            # 2. Range validation
            warnings.extend(self._validate_ranges(state))
            
            # 3. Physics validation
            warnings.extend(self._validate_physics(state))
            
            # 4. Consistency validation
            warnings.extend(self._validate_consistency(state))
            
            # 5. Mission-specific validation
            if context:
                warnings.extend(self._validate_mission_constraints(state, context))
            
            is_valid = len(warnings) == 0 or all('WARNING' in w for w in warnings)
            
            # Log validation result
            self._log_validation(state, is_valid, warnings)
            
            return is_valid, warnings
            
        except ValidationError as e:
            logger.error(f"State validation failed: {e}")
            self.rejection_count += 1
            return False, [f"ERROR: {str(e)}"]
    
    def _validate_ranges(self, state: Any) -> List[str]:
        """Validate parameter ranges."""
        warnings = []
        
        # Position validation
        if hasattr(state, 'position'):
            pos = np.array(state.position)
            if np.any(np.isnan(pos)):
                warnings.append("ERROR: Position contains NaN")
            if np.any(np.abs(pos) > 1e6):  # 1000km limit
                warnings.append("WARNING: Position magnitude suspiciously large")
        
        # Velocity validation
        if hasattr(state, 'velocity'):
            vel = np.array(state.velocity)
            if np.any(np.isnan(vel)):
                warnings.append("ERROR: Velocity contains NaN")
            if np.linalg.norm(vel) > 100.0:  # 100 m/s limit
                warnings.append("WARNING: Velocity too high")
        
        # Temperature validation
        if hasattr(state, 'temperature'):
            temp = state.temperature
            if temp < 100.0 or temp > 400.0:  # 100K to 400K reasonable range
                warnings.append(f"WARNING: Temperature out of range: {temp}K")
        
        # Battery validation
        if hasattr(state, 'battery_level'):
            batt = state.battery_level
            if batt < 0.0 or batt > 1.0:
                warnings.append(f"ERROR: Battery level invalid: {batt}")
            elif batt < 0.1:
                warnings.append("WARNING: Battery critically low")
        
        # Confidence validation
        if hasattr(state, 'confidence'):
            conf = state.confidence
            if conf < 0.0 or conf > 1.0:
                warnings.append(f"ERROR: Confidence invalid: {conf}")
        
        return warnings
    
    def _validate_physics(self, state: Any) -> List[str]:
        """Validate physical consistency."""
        warnings = []
        
        # Check velocity vs acceleration capability
        if hasattr(state, 'velocity') and hasattr(state, 'angular_velocity'):
            vel_norm = np.linalg.norm(state.velocity)
            ang_vel_norm = np.linalg.norm(state.angular_velocity)
            
            # Empirical limits for spacecraft
            if vel_norm > 10.0 and ang_vel_norm > 1.0:
                warnings.append("WARNING: High linear and angular velocity combination")
        
        # Check temperature vs power consumption correlation
        if hasattr(state, 'temperature') and hasattr(state, 'power_consumption'):
            temp = state.temperature
            power = state.power_consumption
            
            # Simple thermal model check
            expected_temp = 300.0 + (power / 100.0) * 10.0
            if abs(temp - expected_temp) > 50.0:
                warnings.append(f"WARNING: Temperature-power mismatch: {temp}K at {power}W")
        
        return warnings
    
    def _validate_consistency(self, state: Any) -> List[str]:
        """Validate internal consistency."""
        warnings = []
        
        # Check timestamp
        if hasattr(state, 'timestamp'):
            current_time = time.time()
            time_diff = current_time - state.timestamp
            
            if time_diff < -1.0:
                warnings.append("ERROR: Timestamp in future")
            elif time_diff > 10.0:
                warnings.append("WARNING: Stale state data")
        
        # Check orientation normalization
        if hasattr(state, 'orientation'):
            orientation = np.array(state.orientation)
            norm = np.linalg.norm(orientation)
            
            if abs(norm - 1.0) > 0.01:
                warnings.append(f"ERROR: Orientation not normalized: norm={norm}")
                # Auto-correct if not too far off
                if 0.9 < norm < 1.1:
                    state.orientation = orientation / norm
                    self.correction_count += 1
                    warnings.append("INFO: Orientation auto-corrected")
        
        # Check joint consistency
        if hasattr(state, 'joint_positions') and hasattr(state, 'joint_velocities'):
            if set(state.joint_positions.keys()) != set(state.joint_velocities.keys()):
                warnings.append("WARNING: Joint position/velocity keys mismatch")
        
        return warnings
    
    def _validate_mission_constraints(self, state: Any, context: Dict) -> List[str]:
        """Validate mission-specific constraints."""
        warnings = []
        
        mission_phase = context.get('mission_phase', 'unknown')
        
        if mission_phase == 'launch':
            # Stricter constraints during launch
            if hasattr(state, 'velocity') and np.linalg.norm(state.velocity) > 5.0:
                warnings.append("ERROR: Velocity too high during launch")
        
        elif mission_phase == 'cruise':
            # Cruise phase validations
            if hasattr(state, 'battery_level') and state.battery_level < 0.3:
                warnings.append("WARNING: Battery low during cruise")
        
        elif mission_phase == 'landing':
            # Landing phase validations
            if hasattr(state, 'position'):
                altitude = state.position[2]
                if altitude < 0.0:
                    warnings.append("ERROR: Negative altitude during landing")
        
        # Sun vector validation (for solar panels)
        if 'sun_vector' in context:
            sun_vector = np.array(context['sun_vector'])
            if hasattr(state, 'orientation'):
                # Check if solar panels are facing sun
                orientation = state.orientation
                # Simplified check - in practice would compute panel normal
                pass
        
        return warnings
    
    def _log_validation(self, state: Any, is_valid: bool, warnings: List[str]):
        """Log validation results."""
        entry = {
            'timestamp': time.time(),
            'robot_id': getattr(state, 'robot_id', 'unknown'),
            'is_valid': is_valid,
            'warning_count': len([w for w in warnings if 'WARNING' in w]),
            'error_count': len([w for w in warnings if 'ERROR' in w]),
            'correction_count': self.correction_count,
            'anomaly': self.anomaly_detected
        }
        
        self.validation_history.append(entry)
        
        # Keep history manageable
        if len(self.validation_history) > 1000:
            self.validation_history.pop(0)
        
        # Update anomaly detection
        if entry['error_count'] > 0:
            self.anomaly_detected = True
    
    def validate_action(self, action: Any, state: Any = None) -> Tuple[bool, List[str]]:
        """Validate action before execution."""
        warnings = []
        
        if not hasattr(action, 'action_id'):
            return False, ["ERROR: Action missing action_id"]
        
        # Check action parameters
        if hasattr(action, 'parameters'):
            params = action.parameters
            if not isinstance(params, dict):
                warnings.append("ERROR: Action parameters must be dict")
        
        # Check duration
        if hasattr(action, 'duration'):
            if action.duration <= 0.0:
                warnings.append("ERROR: Action duration must be positive")
            if action.duration > 3600.0:  # 1 hour max
                warnings.append("WARNING: Action duration very long")
        
        # Check resource requirements
        if hasattr(action, 'power_required'):
            if action.power_required < 0.0:
                warnings.append("ERROR: Power requirement negative")
        
        # Check against current state if provided
        if state:
            if hasattr(action, 'power_required') and hasattr(state, 'battery_level'):
                # Estimate battery usage
                energy_needed = action.power_required * action.duration / 3600.0
                battery_capacity = 100.0  # Wh
                predicted_battery = state.battery_level - (energy_needed / battery_capacity)
                
                if predicted_battery < 0.05:
                    warnings.append("ERROR: Action would deplete battery")
        
        is_valid = len([w for w in warnings if 'ERROR' in w]) == 0
        
        return is_valid, warnings
    
    def validate_mvi_sequence(self, mvi: Any, state: Any) -> Tuple[bool, List[str]]:
        """Validate MVI sequence."""
        warnings = []
        
        if not hasattr(mvi, 'actions'):
            return False, ["ERROR: MVI missing actions"]
        
        if not hasattr(mvi, 'expected_duration'):
            return False, ["ERROR: MVI missing expected_duration"]
        
        # Check individual actions
        for i, action in enumerate(mvi.actions):
            action_valid, action_warnings = self.validate_action(action, state)
            if not action_valid:
                warnings.append(f"ERROR: Action {i} invalid: {action_warnings}")
            else:
                warnings.extend([f"Action {i}: {w}" for w in action_warnings])
        
        # Check total duration
        if hasattr(mvi, 'expected_duration'):
            if mvi.expected_duration > 7200.0:  # 2 hours max
                warnings.append("WARNING: MVI duration very long")
        
        # Check predicted risk if available
        if hasattr(mvi, 'predicted_risk'):
            risk = mvi.predicted_risk
            if hasattr(risk, 'overall_risk'):
                if risk.overall_risk > 0.9:
                    warnings.append("WARNING: MVI predicted risk very high")
        
        is_valid = len([w for w in warnings if 'ERROR' in w]) == 0
        
        return is_valid, warnings
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        recent_history = self.validation_history[-100:] if self.validation_history else []
        
        stats = {
            'total_validations': len(self.validation_history),
            'corrections_applied': self.correction_count,
            'rejections': self.rejection_count,
            'anomaly_detected': self.anomaly_detected,
            'recent_success_rate': 0.0,
            'avg_warnings_per_validation': 0.0
        }
        
        if recent_history:
            stats['recent_success_rate'] = sum(1 for h in recent_history if h['is_valid']) / len(recent_history)
            stats['avg_warnings_per_validation'] = np.mean([h['warning_count'] for h in recent_history])
        
        return stats
    
    def reset(self):
        """Reset validator (for new mission phase)."""
        self.validation_history = []
        self.correction_count = 0
        self.rejection_count = 0
        self.anomaly_detected = False
        logger.info("Validator reset")

# Global instance for convenience
validator = SpacecraftValidator()

# Decorators for validation
def validate_input(func: Callable) -> Callable:
    """Decorator to validate function inputs."""
    def wrapper(*args, **kwargs):
        # Extract state from args/kwargs
        state = None
        for arg in args:
            if hasattr(arg, 'robot_id'):
                state = arg
                break
        
        if 'state' in kwargs:
            state = kwargs['state']
        
        # Validate if state found
        if state:
            is_valid, warnings = validator.validate_robot_state(state)
            if not is_valid:
                logger.error(f"Input validation failed: {warnings}")
                raise ValidationError(f"Input validation failed: {warnings}")
        
        return func(*args, **kwargs)
    return wrapper

def validate_output(func: Callable) -> Callable:
    """Decorator to validate function outputs."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Check if result is a state
        if hasattr(result, 'robot_id'):
            is_valid, warnings = validator.validate_robot_state(result)
            if not is_valid:
                logger.error(f"Output validation failed: {warnings}")
                # Try to correct if possible
                if all('WARNING' in w for w in warnings):
                    logger.warning("Output has warnings but is acceptable")
                else:
                    raise ValidationError(f"Output validation failed: {warnings}")
        
        return result
    return wrapper

# Example usage
if __name__ == "__main__":
    # Create a test state
    class TestState:
        def __init__(self):
            self.robot_id = "test_rover"
            self.timestamp = time.time()
            self.position = [1.0, 2.0, 3.0]
            self.velocity = [0.1, 0.0, 0.0]
            self.temperature = 320.0
            self.battery_level = 0.7
            self.confidence = 0.9
            self.orientation = [0.707, 0.0, 0.707, 0.0]  # Not quite normalized
    
    state = TestState()
    
    # Validate state
    is_valid, warnings = validator.validate_robot_state(state)
    print(f"State valid: {is_valid}")
    print(f"Warnings: {warnings}")
    
    # Get stats
    stats = validator.get_validation_stats()
    print(f"Validation stats: {stats}")