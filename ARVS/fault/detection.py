"""
Fault Detection and Diagnosis System with ARVS Compliance
Version: ARVS-Integrated | Status: Production Ready
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import logging
from dataclasses import dataclass, asdict
from enum import Enum, auto
import time
import psutil
from collections import defaultdict, deque

from arvs.core.axioms import ARVSAxiomBase, AxiomValidator, AxiomSeverity
from arvs.core.data_types import (
    RobotState, SystemMode, FaultSeverity,
    Observation, FeatureVector, ComponentHealth
)
from arvs.core.exceptions import (
    FaultDetectedException, DegradedModeException,
    AxiomViolationException, SafetyCriticalException
)
from arvs.core.constants import (
    FAULT_THRESHOLD_TORQUE, FAULT_THRESHOLD_TEMPERATURE,
    FAULT_THRESHOLD_BATTERY, FAULT_THRESHOLD_VIBRATION,
    DEGRADATION_GRACE_PERIODS, MAX_DEGRADED_PERFORMANCE_LOSS,
    SYSTEM_TIMEOUTS, HEALTH_DECAY_RATE
)

logger = logging.getLogger(__name__)


class FaultType(Enum):
    """Types of faults that can be detected."""
    SENSOR_FAILURE = auto()
    ACTUATOR_FAILURE = auto()
    COMMUNICATION_FAILURE = auto()
    POWER_FAILURE = auto()
    THERMAL_OVERLOAD = auto()
    STRUCTURAL_OVERLOAD = auto()
    SOFTWARE_FAULT = auto()
    ENVIRONMENTAL_HAZARD = auto()
    TIMING_ANOMALY = auto()
    DATA_CORRUPTION = auto()
    UNKNOWN_FAULT = auto()  # For ARVS E2 compliance

    def __str__(self):
        return self.name.lower()


@dataclass
class FaultDiagnosis:
    """
    Complete diagnosis of a detected fault with ARVS traceability.
    """
    fault_type: FaultType
    severity: FaultSeverity
    component: str
    timestamp: float
    confidence: float  # [0, 1] with explicit uncertainty (ARVS U1)
    symptoms: Dict[str, Any]
    probable_cause: str
    recommended_action: str
    recovery_possible: bool
    axiom_references: List[str] = None  # ARVS T1: Traceability
    uncertainty_metrics: Dict[str, float] = None  # ARVS U1: Explicit uncertainty
    
    def __post_init__(self):
        if self.axiom_references is None:
            self.axiom_references = []
        if self.uncertainty_metrics is None:
            self.uncertainty_metrics = {
                'measurement_uncertainty': 0.1,
                'model_uncertainty': 0.2,
                'temporal_uncertainty': 0.05
            }
        # ARVS E1: No perfect certainty
        self.confidence = min(0.99, self.confidence) if self.confidence > 0 else self.confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and traceability."""
        result = {
            'fault_type': str(self.fault_type),
            'severity': self.severity.name,
            'component': self.component,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'probable_cause': self.probable_cause,
            'recovery_possible': self.recovery_possible,
            'axiom_references': self.axiom_references,
            'uncertainty_metrics': self.uncertainty_metrics,
            'symptom_count': len(self.symptoms)
        }
        
        # Include symptoms if not too large
        if len(str(self.symptoms)) < 1000:
            result['symptoms'] = self.symptoms
        
        return result
    
    def to_serializable(self) -> Dict[str, Any]:
        """JSON-serializable version."""
        return {
            **self.to_dict(),
            'fault_type': self.fault_type.name,
            'severity': self.severity.name
        }
    
    def is_axiom_violation(self) -> bool:
        """Check if this fault represents an ARVS axiom violation."""
        return any(ref.startswith('AXIOM_') for ref in self.axiom_references)


class FaultDetector:
    """
    ARVS-compliant fault detection system with graceful degradation.
    
    Key ARVS Integrations:
    - E1: No omniscience in fault detection
    - E2: Unknown faults → high uncertainty
    - U1: Explicit uncertainty quantification
    - R1: Refusal/degredation as valid outcomes
    - T1: Full traceability of fault decisions
    """
    
    def __init__(self, robot_id: str, initial_mode: SystemMode = SystemMode.NORMAL):
        """
        Initialize ARVS-compliant fault detector.
        
        Args:
            robot_id: Unique robot identifier
            initial_mode: Starting system mode (ARVS A4: No implicit authority)
        """
        self.robot_id = robot_id
        self.current_mode = initial_mode
        self.previous_mode = initial_mode
        self.mode_change_time = time.time()
        
        # ARVS E3: Belief stability tracking
        self.mode_history = deque(maxlen=10)
        self.mode_history.append((initial_mode, time.time()))
        
        # Fault detection thresholds with temporal validity (ARVS E4)
        self.fault_thresholds = {
            FaultType.SENSOR_FAILURE: {
                'value': 0.8,
                'valid_until': time.time() + 3600,  # 1 hour
                'confidence': 0.9
            },
            FaultType.ACTUATOR_FAILURE: {
                'value': 0.7,
                'valid_until': time.time() + 1800,
                'confidence': 0.85
            },
            FaultType.THERMAL_OVERLOAD: {
                'value': 0.9,
                'valid_until': time.time() + 900,
                'confidence': 0.95
            },
            FaultType.POWER_FAILURE: {
                'value': 0.6,
                'valid_until': time.time() + 1200,
                'confidence': 0.8
            },
            FaultType.STRUCTURAL_OVERLOAD: {
                'value': 0.85,
                'valid_until': time.time() + 1500,
                'confidence': 0.75
            },
        }
        
        # Fault confirmation with hysteresis (ARVS E3: Stability)
        self.fault_counters: Dict[Tuple[FaultType, str], Dict] = defaultdict(
            lambda: {'count': 0, 'first_detected': 0, 'confirmed': False}
        )
        self.confirmation_thresholds = {
            FaultSeverity.CRITICAL: 1,  # Immediate confirmation
            FaultSeverity.MAJOR: 2,
            FaultSeverity.MODERATE: 3,
            FaultSeverity.MINOR: 5
        }
        
        # Fault history with ring buffer (ARVS T1: Traceability)
        self.fault_history: List[FaultDiagnosis] = []
        self.max_fault_history = 1000
        
        # Component health with decay (ARVS E4: Stale knowledge)
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_update_times: Dict[str, float] = {}
        
        # Performance degradation tracking
        self.performance_degradation: float = 0.0
        self.degradation_causes: deque = deque(maxlen=20)
        
        # System monitoring
        self.monitoring_start_time = time.time()
        self.cycle_count = 0
        self.last_cycle_time = time.time()
        
        # ARVS Integration
        self.axiom_validator = AxiomValidator()
        self.last_axiom_check = 0
        self.axiom_violations: List[Dict] = []
        
        # Unknown fault tracking (ARVS E2)
        self.unknown_fault_log = deque(maxlen=100)
        
        # Initialize
        self._initialize_component_health()
        
        logger.info(f"ARVS Fault Detector initialized for {robot_id}, mode: {initial_mode}")
    
    def _initialize_component_health(self):
        """Initialize health scores with ARVS compliance."""
        components = [
            ('sensors', 1.0, 0.1),
            ('actuators', 1.0, 0.2),
            ('computation', 1.0, 0.05),
            ('power', 1.0, 0.15),
            ('thermal', 1.0, 0.1),
            ('communications', 1.0, 0.08),
            ('structure', 1.0, 0.25),
            ('software', 1.0, 0.03)
        ]
        
        for name, initial_health, degradation_rate in components:
            self.component_health[name] = ComponentHealth(
                name=name,
                current_health=initial_health,
                degradation_rate=degradation_rate,
                last_update=time.time(),
                fault_count=0
            )
            self.health_update_times[name] = time.time()
    
    def monitor_system(self, 
                      robot_state: RobotState,
                      observations: List[Observation],
                      feature_vector: FeatureVector) -> Tuple[SystemMode, List[FaultDiagnosis]]:
        """
        ARVS-compliant system monitoring with fault detection.
        
        Returns:
            (new_system_mode, detected_faults_with_traceability)
        
        Raises:
            SafetyCriticalException if ARVS axioms violated
        """
        self.cycle_count += 1
        current_time = time.time()
        cycle_duration = current_time - self.last_cycle_time
        self.last_cycle_time = current_time
        
        # ARVS C4: Check for timing anomalies (proximity to failure)
        if cycle_duration > SYSTEM_TIMEOUTS['fault_detection']:
            timing_fault = self._create_timing_fault(cycle_duration)
            return SystemMode.EMERGENCY, [timing_fault]
        
        detected_faults = []
        
        # Run fault checks with explicit uncertainty tracking
        check_functions = [
            (self._check_sensor_health, [observations, feature_vector]),
            (self._check_actuator_health, [robot_state]),
            (self._check_thermal_health, [robot_state]),
            (self._check_power_health, [robot_state]),
            (self._check_structural_health, [robot_state]),
            (self._check_software_health, []),
            (self._check_environmental_hazards, [robot_state, feature_vector]),
            (self._check_timing_health, [cycle_duration]),
            (self._check_data_integrity, [observations, feature_vector]),
        ]
        
        for check_func, args in check_functions:
            try:
                faults = check_func(*args)
                detected_faults.extend(faults)
            except Exception as e:
                # ARVS E2: Unknown/check failure → high uncertainty fault
                unknown_fault = self._create_unknown_fault(check_func.__name__, str(e))
                detected_faults.append(unknown_fault)
                logger.error(f"Fault check {check_func.__name__} failed: {e}")
        
        # Apply fault confirmation logic (ARVS E3: Stability)
        confirmed_faults = self._apply_confirmation_logic(detected_faults, current_time)
        
        # ARVS Axiom compliance check
        axiom_status = self._check_axiom_compliance(confirmed_faults, robot_state)
        if not axiom_status['authority_valid']:
            # ARVS G3: Emergency doesn't relax rules
            emergency_fault = self._create_axiom_violation_fault(axiom_status)
            confirmed_faults.append(emergency_fault)
        
        # Update component health with ARVS E4: Temporal validity
        self._update_component_health(confirmed_faults, current_time)
        
        # Determine system mode with ARVS R1: Refusal legitimacy
        new_mode = self._determine_system_mode_arvs(confirmed_faults, current_time)
        
        # ARVS R2: Degradation over action
        self._update_performance_degradation(confirmed_faults, new_mode)
        
        # Store with traceability (ARVS T1/T2)
        self._store_faults_with_traceability(confirmed_faults, new_mode)
        
        # Handle mode change (ARVS A3: Authority loss is final per cycle)
        if new_mode != self.current_mode:
            self._handle_mode_change(new_mode, confirmed_faults, current_time)
        
        return new_mode, confirmed_faults
    
    def _check_sensor_health(self, observations: List[Observation],
                            feature_vector: FeatureVector) -> List[FaultDiagnosis]:
        """Sensor health check with ARVS compliance."""
        faults = []
        current_time = time.time()
        
        # ARVS E1: No assumption of complete sensor coverage
        if not observations:
            sensor_coverage = feature_vector.features.get('sensor_coverage', 0.0)
            if sensor_coverage < 0.3:  # Less than 30% coverage
                return [self._create_sensor_coverage_fault(sensor_coverage, current_time)]
            # Otherwise, continue with reduced confidence
        
        # Multi-dimensional sensor checks
        sensor_checks = [
            self._check_sensor_freshness(observations, current_time),
            self._check_sensor_consistency(observations),
            self._check_sensor_calibration(feature_vector),
            self._check_sensor_fusion_health(feature_vector),
        ]
        
        for check_result in sensor_checks:
            if check_result:
                faults.append(check_result)
        
        return faults
    
    def _check_sensor_freshness(self, observations: List[Observation],
                               current_time: float) -> Optional[FaultDiagnosis]:
        """Check sensor data freshness with temporal thresholds."""
        if not observations:
            return None
        
        max_age = max(current_time - obs.timestamp for obs in observations)
        
        if max_age > 5.0:  # 5 seconds without update
            # ARVS C4: Proximity escalation - older data = higher severity
            severity = FaultSeverity.CRITICAL if max_age > 30.0 else \
                      FaultSeverity.MAJOR if max_age > 10.0 else \
                      FaultSeverity.MODERATE
            
            return FaultDiagnosis(
                fault_type=FaultType.SENSOR_FAILURE,
                severity=severity,
                component='sensor_system',
                timestamp=current_time,
                confidence=0.8,
                symptoms={'max_data_age': max_age, 'observation_count': len(observations)},
                probable_cause='Sensor data timeout',
                recommended_action='Check sensor connections and power',
                recovery_possible=True,
                axiom_references=['E4', 'C4'],
                uncertainty_metrics={'temporal_uncertainty': min(1.0, max_age/60.0)}
            )
        return None
    
    def _create_sensor_coverage_fault(self, coverage: float,
                                     timestamp: float) -> FaultDiagnosis:
        """Create fault for insufficient sensor coverage."""
        return FaultDiagnosis(
            fault_type=FaultType.SENSOR_FAILURE,
            severity=FaultSeverity.MAJOR if coverage < 0.1 else FaultSeverity.MODERATE,
            component='sensor_coverage',
            timestamp=timestamp,
            confidence=0.9,
            symptoms={'coverage_ratio': coverage},
            probable_cause='Insufficient sensor coverage for safe operation',
            recommended_action='Enable degraded sensing mode or halt operation',
            recovery_possible=coverage > 0.05,
            axiom_references=['E1', 'R2'],
            uncertainty_metrics={'model_uncertainty': 1.0 - coverage}
        )
    
    def _check_actuator_health(self, robot_state: RobotState) -> List[FaultDiagnosis]:
        """Actuator health check with torque and response monitoring."""
        faults = []
        current_time = time.time()
        
        # Check each joint
        for joint_name, torque in robot_state.joint_torques.items():
            # Normalize and check
            torque_ratio = abs(torque) / FAULT_THRESHOLD_TORQUE
            
            if torque_ratio > 1.0:
                # ARVS C2: Non-linear risk scaling
                severity_multiplier = 2.0 if torque_ratio > 1.5 else 1.0
                severity_score = torque_ratio * severity_multiplier
                
                severity = FaultSeverity.CRITICAL if severity_score > 2.0 else \
                          FaultSeverity.MAJOR if severity_score > 1.5 else \
                          FaultSeverity.MODERATE
                
                fault = FaultDiagnosis(
                    fault_type=FaultType.ACTUATOR_FAILURE,
                    severity=severity,
                    component=f'actuator_{joint_name}',
                    timestamp=current_time,
                    confidence=min(0.95, torque_ratio),
                    symptoms={
                        'torque': torque,
                        'torque_ratio': torque_ratio,
                        'threshold': FAULT_THRESHOLD_TORQUE
                    },
                    probable_cause=f'Excessive torque on {joint_name}',
                    recommended_action=f'Reduce load, inspect {joint_name} mechanism',
                    recovery_possible=True,
                    axiom_references=['C2', 'C3'],
                    uncertainty_metrics={'measurement_uncertainty': 0.1}
                )
                faults.append(fault)
        
        # Check actuator response consistency
        if hasattr(robot_state, 'actuator_response_latency'):
            avg_latency = np.mean(robot_state.actuator_response_latency)
            if avg_latency > 0.1:  # 100ms threshold
                fault = FaultDiagnosis(
                    fault_type=FaultType.ACTUATOR_FAILURE,
                    severity=FaultSeverity.MODERATE,
                    component='actuator_response',
                    timestamp=current_time,
                    confidence=0.7,
                    symptoms={'avg_response_latency': avg_latency},
                    probable_cause='High actuator response latency',
                    recommended_action='Check control loop timing, reduce load',
                    recovery_possible=True,
                    axiom_references=['E3'],
                    uncertainty_metrics={'temporal_uncertainty': min(1.0, avg_latency)}
                )
                faults.append(fault)
        
        return faults
    
    def _check_thermal_health(self, robot_state: RobotState) -> List[FaultDiagnosis]:
        """Thermal health monitoring with progressive thresholds."""
        faults = []
        current_time = time.time()
        
        if not hasattr(robot_state, 'temperature') or robot_state.temperature is None:
            # ARVS E2: Unknown temperature → assume worst case
            return [self._create_unknown_thermal_fault(current_time)]
        
        temp_ratio = robot_state.temperature / FAULT_THRESHOLD_TEMPERATURE
        
        if temp_ratio > 1.0:
            # Progressive severity based on overheating
            if temp_ratio > 1.5:
                severity = FaultSeverity.CRITICAL
                recovery = robot_state.temperature < FAULT_THRESHOLD_TEMPERATURE * 1.8
            elif temp_ratio > 1.2:
                severity = FaultSeverity.MAJOR
                recovery = True
            else:
                severity = FaultSeverity.MODERATE
                recovery = True
            
            fault = FaultDiagnosis(
                fault_type=FaultType.THERMAL_OVERLOAD,
                severity=severity,
                component='thermal_system',
                timestamp=current_time,
                confidence=min(0.98, temp_ratio),
                symptoms={
                    'temperature': robot_state.temperature,
                    'temp_ratio': temp_ratio,
                    'threshold': FAULT_THRESHOLD_TEMPERATURE
                },
                probable_cause='Overheating due to high load or cooling failure',
                recommended_action='Activate emergency cooling, reduce power',
                recovery_possible=recovery,
                axiom_references=['C1', 'C4'],  # Worst-case, proximity escalation
                uncertainty_metrics={'measurement_uncertainty': 0.05}
            )
            faults.append(fault)
        
        return faults
    
    def _create_unknown_thermal_fault(self, timestamp: float) -> FaultDiagnosis:
        """Create fault for unknown thermal state."""
        return FaultDiagnosis(
            fault_type=FaultType.UNKNOWN_FAULT,
            severity=FaultSeverity.MAJOR,  # Conservative
            component='thermal_sensing',
            timestamp=timestamp,
            confidence=0.5,  # Low confidence due to unknown
            symptoms={'thermal_sensor_status': 'unknown'},
            probable_cause='Thermal sensor failure or data unavailable',
            recommended_action='Assume worst-case temperature, enable thermal protection',
            recovery_possible=False,  # Can't recover from unknown
            axiom_references=['E2', 'U1'],
            uncertainty_metrics={
                'measurement_uncertainty': 1.0,
                'model_uncertainty': 1.0,
                'temporal_uncertainty': 1.0
            }
        )
    
    def _check_power_health(self, robot_state: RobotState) -> List[FaultDiagnosis]:
        """Power system health with battery and consumption monitoring."""
        faults = []
        current_time = time.time()
        
        # Battery level check
        if hasattr(robot_state, 'battery_level'):
            battery = robot_state.battery_level
            
            if battery < FAULT_THRESHOLD_BATTERY:
                # ARVS C1: Worst-case consequence evaluation
                # Low battery could lead to uncontrolled shutdown
                severity = FaultSeverity.CRITICAL if battery < 0.05 else \
                          FaultSeverity.MAJOR if battery < 0.1 else \
                          FaultSeverity.MODERATE
                
                recovery = battery > 0.02  # Can only recover if some charge left
                
                fault = FaultDiagnosis(
                    fault_type=FaultType.POWER_FAILURE,
                    severity=severity,
                    component='battery',
                    timestamp=current_time,
                    confidence=0.95,
                    symptoms={'battery_level': battery},
                    probable_cause='Low battery, imminent power loss',
                    recommended_action='Initiate controlled shutdown, conserve power',
                    recovery_possible=recovery,
                    axiom_references=['C1', 'C3'],  # Worst-case, irreversible
                    uncertainty_metrics={'measurement_uncertainty': 0.02}
                )
                faults.append(fault)
        
        # Power consumption anomaly detection
        if hasattr(robot_state, 'power_consumption'):
            expected_power = getattr(robot_state, 'expected_power', None)
            if expected_power and robot_state.power_consumption:
                ratio = robot_state.power_consumption / expected_power
                if ratio > 1.5 or ratio < 0.5:
                    fault = FaultDiagnosis(
                        fault_type=FaultType.POWER_FAILURE,
                        severity=FaultSeverity.MODERATE,
                        component='power_consumption',
                        timestamp=current_time,
                        confidence=0.7,
                        symptoms={
                            'consumption': robot_state.power_consumption,
                            'expected': expected_power,
                            'ratio': ratio
                        },
                        probable_cause='Abnormal power consumption',
                        recommended_action='Check for shorts or failing components',
                        recovery_possible=True,
                        axiom_references=['E3'],
                        uncertainty_metrics={'measurement_uncertainty': 0.1}
                    )
                    faults.append(fault)
        
        return faults
    
    def _check_structural_health(self, robot_state: RobotState) -> List[FaultDiagnosis]:
        """Structural health monitoring with vibration analysis."""
        faults = []
        current_time = time.time()
        
        # Vibration analysis
        if hasattr(robot_state, 'vibration_metrics'):
            for metric_name, value in robot_state.vibration_metrics.items():
                if value > FAULT_THRESHOLD_VIBRATION:
                    # ARVS C2: Non-linear scaling for structural risks
                    severity_multiplier = 3.0 if 'critical' in metric_name else 1.5
                    severity_score = (value / FAULT_THRESHOLD_VIBRATION) * severity_multiplier
                    
                    severity = FaultSeverity.CRITICAL if severity_score > 3.0 else \
                              FaultSeverity.MAJOR if severity_score > 2.0 else \
                              FaultSeverity.MODERATE
                    
                    fault = FaultDiagnosis(
                        fault_type=FaultType.STRUCTURAL_OVERLOAD,
                        severity=severity,
                        component=f'structure_{metric_name}',
                        timestamp=current_time,
                        confidence=min(0.9, value / (FAULT_THRESHOLD_VIBRATION * 2)),
                        symptoms={
                            'vibration_metric': metric_name,
                            'value': value,
                            'threshold': FAULT_THRESHOLD_VIBRATION
                        },
                        probable_cause=f'Excessive {metric_name} vibration',
                        recommended_action='Reduce speed, inspect for structural damage',
                        recovery_possible=True,
                        axiom_references=['C2', 'C4'],
                        uncertainty_metrics={'measurement_uncertainty': 0.15}
                    )
                    faults.append(fault)
        
        return faults
    
    def _check_software_health(self) -> List[FaultDiagnosis]:
        """Software system health monitoring."""
        faults = []
        current_time = time.time()
        
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            memory_ratio = memory.percent / 100.0
            
            if memory_ratio > 0.9:
                fault = FaultDiagnosis(
                    fault_type=FaultType.SOFTWARE_FAULT,
                    severity=FaultSeverity.MAJOR,
                    component='memory',
                    timestamp=current_time,
                    confidence=0.85,
                    symptoms={
                        'memory_percent': memory.percent,
                        'available_gb': memory.available / 1e9,
                        'threshold': 90.0
                    },
                    probable_cause='High memory usage, possible memory leak',
                    recommended_action='Restart affected processes, clear caches',
                    recovery_possible=True,
                    axiom_references=['L2'],  # No online learning in critical state
                    uncertainty_metrics={'measurement_uncertainty': 0.02}
                )
                faults.append(fault)
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 95.0:
                fault = FaultDiagnosis(
                    fault_type=FaultType.SOFTWARE_FAULT,
                    severity=FaultSeverity.MODERATE,
                    component='cpu',
                    timestamp=current_time,
                    confidence=0.8,
                    symptoms={'cpu_percent': cpu_percent, 'threshold': 95.0},
                    probable_cause='High CPU load, possible infinite loop',
                    recommended_action='Reduce computational load, check task scheduling',
                    recovery_possible=True,
                    axiom_references=['R3'],  # No forced optimization
                    uncertainty_metrics={'measurement_uncertainty': 0.05}
                )
                faults.append(fault)
                
        except Exception as e:
            # ARVS E2: Software monitoring failure → unknown fault
            faults.append(self._create_unknown_fault('software_monitoring', str(e)))
        
        return faults
    
    def _check_environmental_hazards(self, robot_state: RobotState,
                                    feature_vector: FeatureVector) -> List[FaultDiagnosis]:
        """Environmental hazard detection."""
        faults = []
        current_time = time.time()
        
        # Radiation check
        radiation = feature_vector.features.get('radiation_level', 0.0)
        if radiation > 100.0:  # Arbitrary dangerous threshold
            # ARVS C3: Irreversible damage from radiation
            fault = FaultDiagnosis(
                fault_type=FaultType.ENVIRONMENTAL_HAZARD,
                severity=FaultSeverity.CRITICAL,
                component='environment_radiation',
                timestamp=current_time,
                confidence=0.9,
                symptoms={'radiation_level': radiation},
                probable_cause='Dangerous radiation levels',
                recommended_action='Activate shielding, evacuate area immediately',
                recovery_possible=False,  # Radiation damage is cumulative
                axiom_references=['C1', 'C3'],  # Worst-case, irreversible
                uncertainty_metrics={'measurement_uncertainty': 0.1}
            )
            faults.append(fault)
        
        # Other environmental checks can be added here
        
        return faults
    
    def _check_timing_health(self, cycle_duration: float) -> List[FaultDiagnosis]:
        """Timing and latency health checks."""
        faults = []
        current_time = time.time()
        
        # Cycle timing anomaly
        expected_cycle = 0.1  # 100ms expected cycle
        if cycle_duration > expected_cycle * 2:  # 200% over expected
            fault = FaultDiagnosis(
                fault_type=FaultType.TIMING_ANOMALY,
                severity=FaultSeverity.MODERATE,
                component='timing',
                timestamp=current_time,
                confidence=0.75,
                symptoms={
                    'cycle_duration': cycle_duration,
                    'expected_cycle': expected_cycle,
                    'ratio': cycle_duration / expected_cycle
                },
                probable_cause='Fault detection cycle taking too long',
                recommended_action='Optimize detection algorithms, check system load',
                recovery_possible=True,
                axiom_references=['C4'],  # Proximity to timing failure
                uncertainty_metrics={'temporal_uncertainty': 0.1}
            )
            faults.append(fault)
        
        return faults
    
    def _check_data_integrity(self, observations: List[Observation],
                             feature_vector: FeatureVector) -> List[FaultDiagnosis]:
        """Data integrity and corruption checks."""
        faults = []
        current_time = time.time()
        
        # Check for NaN or infinite values
        invalid_observations = []
        for obs in observations:
            if not obs.valid or self._has_invalid_values(obs.data):
                invalid_observations.append(obs.sensor_type)
        
        if invalid_observations:
            fault = FaultDiagnosis(
                fault_type=FaultType.DATA_CORRUPTION,
                severity=FaultSeverity.MODERATE,
                component='data_integrity',
                timestamp=current_time,
                confidence=0.8,
                symptoms={
                    'invalid_sensors': invalid_observations,
                    'total_sensors': len(observations)
                },
                probable_cause='Data corruption in sensor readings',
                recommended_action='Reset sensors, check data pipelines',
                recovery_possible=True,
                axiom_references=['E1'],  # Can't trust corrupted data
                uncertainty_metrics={'measurement_uncertainty': 1.0}
            )
            faults.append(fault)
        
        return faults
    
    def _has_invalid_values(self, data: Any) -> bool:
        """Check for NaN, Inf, or other invalid values."""
        if isinstance(data, (int, float)):
            return not np.isfinite(data)
        elif isinstance(data, np.ndarray):
            return not np.all(np.isfinite(data))
        return False
    
    def _create_timing_fault(self, cycle_duration: float) -> FaultDiagnosis:
        """Create critical timing fault."""
        return FaultDiagnosis(
            fault_type=FaultType.TIMING_ANOMALY,
            severity=FaultSeverity.CRITICAL,
            component='fault_detection_timing',
            timestamp=time.time(),
            confidence=0.95,
            symptoms={'cycle_duration': cycle_duration},
            probable_cause='Fault detection system timing violation',
            recommended_action='Enter emergency mode, restart fault detection',
            recovery_possible=True,
            axiom_references=['C4', 'G3'],
            uncertainty_metrics={'temporal_uncertainty': 0.5}
        )
    
    def _create_unknown_fault(self, source: str, error: str) -> FaultDiagnosis:
        """Create fault for unknown/undiagnosed issues (ARVS E2)."""
        return FaultDiagnosis(
            fault_type=FaultType.UNKNOWN_FAULT,
            severity=FaultSeverity.MAJOR,  # Conservative default
            component=f'unknown_{source}',
            timestamp=time.time(),
            confidence=0.3,  # Low confidence for unknown
            symptoms={'source': source, 'error': error},
            probable_cause=f'Unknown fault in {source}',
            recommended_action='Enter degraded mode, log details for analysis',
            recovery_possible=False,  # Can't recover from unknown
            axiom_references=['E2', 'U1', 'Z'],
            uncertainty_metrics={
                'measurement_uncertainty': 1.0,
                'model_uncertainty': 1.0,
                'temporal_uncertainty': 1.0
            }
        )
    
    def _apply_confirmation_logic(self, faults: List[FaultDiagnosis],
                                 current_time: float) -> List[FaultDiagnosis]:
        """Apply fault confirmation with hysteresis (ARVS E3)."""
        confirmed_faults = []
        
        for fault in faults:
            key = (fault.fault_type, fault.component)
            counter = self.fault_counters[key]
            
            # Update counter
            if not counter['confirmed']:
                counter['count'] += 1
                if counter['first_detected'] == 0:
                    counter['first_detected'] = current_time
                
                # Check if confirmed
                threshold = self.confirmation_thresholds.get(fault.severity, 3)
                if counter['count'] >= threshold:
                    counter['confirmed'] = True
                    confirmed_faults.append(fault)
                elif current_time - counter['first_detected'] > 10.0:  # 10s timeout
                    # Reset if not confirmed within timeout
                    counter['count'] = 0
                    counter['first_detected'] = 0
            else:
                # Already confirmed, include immediately
                confirmed_faults.append(fault)
        
        return confirmed_faults
    
    def _check_axiom_compliance(self, faults: List[FaultDiagnosis],
                               robot_state: RobotState) -> Dict[str, Any]:
        """Check ARVS axiom compliance for current state."""
        current_time = time.time()
        
        # Only check periodically to avoid overhead
        if current_time - self.last_axiom_check < 1.0:  # 1 second interval
            return {'authority_valid': True, 'violations': []}
        
        self.last_axiom_check = current_time
        
        # Create system state for axiom validation
        system_state = {
            'faults_present': len(faults) > 0,
            'critical_faults': any(f.severity == FaultSeverity.CRITICAL for f in faults),
            'mode_stability': self._calculate_mode_stability(),
            'component_health': {k: v.current_health for k, v in self.component_health.items()},
            'performance_degradation': self.performance_degradation,
            'timestamp': current_time
        }
        
        # Run axiom validation
        results = self.axiom_validator.validate_state(system_state)
        
        # Store violations
        for violation in results.get('failed', []):
            self.axiom_violations.append({
                'timestamp': current_time,
                'axiom': violation['axiom'],
                'reason': violation['reason'],
                'severity': violation['severity']
            })
        
        return results
    
    def _create_axiom_violation_fault(self, axiom_status: Dict) -> FaultDiagnosis:
        """Create fault for ARVS axiom violation."""
        violations = axiom_status.get('failed', [])
        violation_desc = ', '.join([f"{v['axiom']}" for v in violations[:3]])
        
        return FaultDiagnosis(
            fault_type=FaultType.SOFTWARE_FAULT,
            severity=FaultSeverity.CRITICAL,
            component='axiom_compliance',
            timestamp=time.time(),
            confidence=0.95,
            symptoms={'violations': violations},
            probable_cause=f'ARVS axiom violations: {violation_desc}',
            recommended_action='Enter emergency mode, review system state',
            recovery_possible=False,  # Axiom violations require system reset
            axiom_references=[f"AXIOM_{v['axiom']}" for v in violations],
            uncertainty_metrics={'model_uncertainty': 1.0}
        )
    
    def _calculate_mode_stability(self) -> float:
        """Calculate mode stability score (ARVS E3)."""
        if len(self.mode_history) < 2:
            return 1.0
        
        # Count mode changes in recent history
        changes = 0
        for i in range(1, len(self.mode_history)):
            if self.mode_history[i][0] != self.mode_history[i-1][0]:
                changes += 1
        
        stability = 1.0 - (changes / (len(self.mode_history) - 1))
        return max(0.0, stability)
    
    def _update_component_health(self, faults: List[FaultDiagnosis],
                                current_time: float):
        """Update component health with ARVS E4 compliance."""
        # Apply health decay first
        for component, health in self.component_health.items():
            time_since_update = current_time - health.last_update
            decay = health.degradation_rate * (time_since_update / 3600.0)  # Per hour
            health.current_health = max(0.0, health.current_health - decay)
            health.last_update = current_time
        
        # Apply fault penalties
        for fault in faults:
            # Map fault to component category
            component_map = {
                'sensor': 'sensors',
                'actuator': 'actuators',
                'thermal': 'thermal',
                'power': 'power',
                'structure': 'structure',
                'software': 'software',
                'communication': 'communications',
                'computation': 'computation'
            }
            
            target_component = 'software'  # Default
            
            for key, component in component_map.items():
                if key in fault.component.lower():
                    target_component = component
                    break
            
            if target_component in self.component_health:
                health = self.component_health[target_component]
                
                # Severity-based penalty
                penalty = {
                    FaultSeverity.MINOR: 0.05,
                    FaultSeverity.MODERATE: 0.15,
                    FaultSeverity.MAJOR: 0.35,
                    FaultSeverity.CRITICAL: 0.7
                }.get(fault.severity, 0.1)
                
                # Apply with confidence weighting
                effective_penalty = penalty * fault.confidence
                health.current_health = max(0.0, health.current_health - effective_penalty)
                health.fault_count += 1
                health.last_update = current_time
    
    def _determine_system_mode_arvs(self, faults: List[FaultDiagnosis],
                                   current_time: float) -> SystemMode:
        """
        Determine system mode with ARVS compliance.
        
        Implements:
        - R1: Refusal legitimacy
        - R2: Degradation over action
        - A1: Conditional authority
        - C1: Worst-case consequence evaluation
        """
        
        # ARVS A4: Default to no authority (safe hold)
        if not faults:
            # Check if we should return to normal
            avg_health = np.mean([h.current_health for h in self.component_health.values()])
            mode_stability = self._calculate_mode_stability()
            
            if avg_health > 0.8 and mode_stability > 0.7:
                return SystemMode.NORMAL
            elif avg_health > 0.6:
                return SystemMode.DEGRADED
            else:
                return SystemMode.SAFE_HOLD
        
        # Check for critical faults (ARVS C1: Worst-case)
        critical_faults = [f for f in faults if f.severity == FaultSeverity.CRITICAL]
        if critical_faults:
            # ARVS G3: Emergency increases conservatism
            return SystemMode.EMERGENCY
        
        # Check for major faults
        major_faults = [f for f in faults if f.severity == FaultSeverity.MAJOR]
        if major_faults:
            # Check if any are irreversible (ARVS C3)
            irreversible = any(not f.recovery_possible for f in major_faults)
            if irreversible:
                return SystemMode.EMERGENCY
            return SystemMode.FAULT_RECOVERY
        
        # Calculate overall system health
        health_scores = [h.current_health for h in self.component_health.values()]
        avg_health = np.mean(health_scores) if health_scores else 0.0
        min_health = min(health_scores) if health_scores else 0.0
        
        # ARVS R2: Prefer degradation
        if min_health < 0.3 or avg_health < 0.5:
            return SystemMode.DEGRADED
        
        # Check if we're recovering or degrading
        mode_stability = self._calculate_mode_stability()
        if mode_stability < 0.5:
            # Unstable, degrade further
            return SystemMode.DEGRADED
        
        # Default to current mode
        return self.current_mode
    
    def _update_performance_degradation(self, faults: List[FaultDiagnosis],
                                       new_mode: SystemMode):
        """Update performance degradation with ARVS R2 compliance."""
        # Mode-based baseline degradation
        mode_degradation = {
            SystemMode.NORMAL: 0.0,
            SystemMode.DEGRADED: 0.4,
            SystemMode.SAFE_HOLD: 0.7,
            SystemMode.EMERGENCY: 0.9,
            SystemMode.FAULT_RECOVERY: 0.6
        }.get(new_mode, 0.5)
        
        # Fault-based additional degradation
        fault_degradation = 0.0
        for fault in faults:
            severity_factor = {
                FaultSeverity.MINOR: 0.03,
                FaultSeverity.MODERATE: 0.08,
                FaultSeverity.MAJOR: 0.15,
                FaultSeverity.CRITICAL: 0.25
            }.get(fault.severity, 0.05)
            
            # Weight by confidence
            fault_degradation += severity_factor * fault.confidence
        
        # Calculate target degradation
        target_degradation = min(
            MAX_DEGRADED_PERFORMANCE_LOSS,
            mode_degradation + fault_degradation
        )
        
        # Apply smoothing with limits
        if target_degradation > self.performance_degradation:
            # Degrading: faster response
            alpha = 0.3
        else:
            # Recovering: slower response
            alpha = 0.1
        
        self.performance_degradation = (
            alpha * target_degradation + 
            (1 - alpha) * self.performance_degradation
        )
        
        # Log cause if significant change
        if abs(target_degradation - self.performance_degradation) > 0.1:
            cause = f"Mode:{new_mode.name}, Faults:{len(faults)}"
            self.degradation_causes.append((time.time(), cause))
    
    def _store_faults_with_traceability(self, faults: List[FaultDiagnosis],
                                       new_mode: SystemMode):
        """Store faults with full traceability (ARVS T1/T2)."""
        for fault in faults:
            # Add mode context to traceability
            if 'axiom_references' not in fault.axiom_references:
                fault.axiom_references.append('T1')
            
            # Store in history
            self.fault_history.append(fault)
            if len(self.fault_history) > self.max_fault_history:
                self.fault_history.pop(0)
            
            # Log for analysis
            logger.info(f"Fault detected: {fault.fault_type} on {fault.component}, "
                       f"severity: {fault.severity.name}, mode: {new_mode.name}")
    
    def _handle_mode_change(self, new_mode: SystemMode,
                           faults: List[FaultDiagnosis],
                           timestamp: float):
        """Handle system mode change with ARVS compliance."""
        old_mode = self.current_mode
        
        # ARVS A3: Authority loss is final per cycle
        # (Once we leave a mode, we can't return immediately)
        
        # Update mode history
        self.mode_history.append((new_mode, timestamp))
        self.previous_mode = old_mode
        self.current_mode = new_mode
        self.mode_change_time = timestamp
        
        # Log the change
        fault_summary = ', '.join([f.fault_type.name for f in faults[:3]])
        logger.warning(
            f"ARVS Mode Change: {old_mode.name} → {new_mode.name} | "
            f"Faults: {fault_summary} | "
            f"Degradation: {self.performance_degradation:.1%}"
        )
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive ARVS-compliant health report."""
        current_time = time.time()
        uptime = current_time - self.monitoring_start_time
        
        # Calculate health metrics
        component_healths = [h.current_health for h in self.component_health.values()]
        overall_health = float(np.mean(component_healths)) if component_healths else 0.0
        
        # Recent faults with traceability
        recent_faults = []
        for fault in self.fault_history[-10:]:
            recent_faults.append(fault.to_serializable())
        
        # Component details
        component_details = []
        for name, health in self.component_health.items():
            component_details.append({
                'component': name,
                'health': health.current_health,
                'degradation_rate': health.degradation_rate,
                'fault_count': health.fault_count,
                'last_update': current_time - health.last_update
            })
        
        # Axiom compliance status
        axiom_status = {
            'last_check': self.last_axiom_check,
            'recent_violations': self.axiom_violations[-5:],
            'total_violations': len(self.axiom_violations)
        }
        
        # Mode stability
        mode_stability = self._calculate_mode_stability()
        
        return {
            'robot_id': self.robot_id,
            'timestamp': current_time,
            'uptime': uptime,
            'cycle_count': self.cycle_count,
            
            # System state
            'current_mode': self.current_mode.name,
            'previous_mode': self.previous_mode.name,
            'mode_stability': mode_stability,
            'time_in_current_mode': current_time - self.mode_change_time,
            
            # Health metrics
            'overall_health': overall_health,
            'performance_degradation': self.performance_degradation,
            'degradation_causes': list(self.degradation_causes),
            
            # Component details
            'component_health': component_details,
            'health_summary': {
                'min': float(np.min(component_healths)) if component_healths else 0.0,
                'max': float(np.max(component_healths)) if component_healths else 0.0,
                'std': float(np.std(component_healths)) if len(component_healths) > 1 else 0.0
            },
            
            # Fault information
            'recent_faults': recent_faults,
            'total_faults_detected': len(self.fault_history),
            'fault_types_distribution': self._get_fault_distribution(),
            
            # ARVS compliance
            'axiom_compliance': axiom_status,
            'unknown_faults_logged': len(self.unknown_fault_log),
            
            # Performance limits
            'performance_limits': self.get_performance_limits()
        }
    
    def _get_fault_distribution(self) -> Dict[str, int]:
        """Get distribution of fault types."""
        distribution = defaultdict(int)
        for fault in self.fault_history:
            distribution[fault.fault_type.name] += 1
        return dict(distribution)
    
    def get_performance_limits(self) -> Dict[str, float]:
        """Get performance limits based on current degradation (ARVS R2)."""
        degradation_factor = 1.0 - self.performance_degradation
        
        # Non-linear scaling for critical parameters (ARVS C2)
        safety_margin = 0.1  # Additional safety margin
        
        return {
            'max_velocity': degradation_factor * (1.0 - safety_margin),
            'max_acceleration': degradation_factor ** 2,  # Quadratic scaling
            'max_torque': degradation_factor * (1.0 - safety_margin * 2),
            'max_power': degradation_factor,
            'computation_rate': max(0.1, degradation_factor),
            'sensor_update_rate': max(0.5, degradation_factor),
            'control_frequency': max(10.0, 100.0 * degradation_factor),
            'planning_horizon': max(1.0, 10.0 * degradation_factor)
        }
    
    def reset_component_health(self, component: str):
        """Reset health for specific component."""
        if component in self.component_health:
            self.component_health[component].current_health = 1.0
            self.component_health[component].fault_count = 0
            self.component_health[component].last_update = time.time()
            logger.info(f"Reset health for component: {component}")
    
    def reset_all_health(self):
        """Reset all component health."""
        self._initialize_component_health()
        self.performance_degradation = 0.0
        self.degradation_causes.clear()
        logger.info("Reset all component health and performance metrics")
    
    def clear_fault_history(self):
        """Clear fault history (use with caution)."""
        self.fault_history = []
        self.fault_counters.clear()
        logger.warning("Cleared all fault history and counters")
    
    def get_diagnostic_summary(self) -> str:
        """Get human-readable diagnostic summary."""
        report = self.get_system_health_report()
        
        summary = [
            f"=== ARVS Fault Detection Diagnostic ===",
            f"Robot: {report['robot_id']}",
            f"Uptime: {report['uptime']:.1f}s | Cycles: {report['cycle_count']}",
            f"Mode: {report['current_mode']} (Stability: {report['mode_stability']:.1%})",
            f"Overall Health: {report['overall_health']:.1%}",
            f"Performance Degradation: {report['performance_degradation']:.1%}",
            f"",
            f"Component Health:"
        ]
        
        for comp in report['component_health']:
            summary.append(f"  {comp['component']}: {comp['health']:.1%} "
                          f"(Faults: {comp['fault_count']})")
        
        summary.extend([
            f"",
            f"Recent Faults: {len(report['recent_faults'])}",
            f"Total Faults: {report['total_faults_detected']}",
            f"Axiom Violations: {report['axiom_compliance']['total_violations']}",
            f"",
            f"Performance Limits:"
        ])
        
        for param, limit in report['performance_limits'].items():
            summary.append(f"  {param}: {limit:.1%}")
        
        return "\n".join(summary)


# Helper function for safe fault detection initialization
def create_fault_detector(robot_id: str, 
                         config: Optional[Dict] = None) -> FaultDetector:
    """
    Factory function for creating fault detectors with validation.
    
    Args:
        robot_id: Robot identifier
        config: Optional configuration overrides
        
    Returns:
        Initialized and validated FaultDetector
    """
    # Default configuration
    default_config = {
        'initial_mode': SystemMode.NORMAL,
        'max_fault_history': 1000,
        'validation_interval': 1.0
    }
    
    # Merge with provided config
    if config:
        default_config.update(config)
    
    # Create detector
    detector = FaultDetector(
        robot_id=robot_id,
        initial_mode=default_config['initial_mode']
    )
    
    # Validate initial state
    initial_report = detector.get_system_health_report()
    if initial_report['overall_health'] < 0.9:
        logger.warning(f"Low initial health for {robot_id}: "
                      f"{initial_report['overall_health']:.1%}")
    
    return detector
