import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

from arvs.core.data_types import RiskAssessment, RobotState, BeliefState, SafetyConstraints
from arvs.core.constants import (
    RISK_THRESHOLD_SAFE, RISK_THRESHOLD_WARNING,
    RISK_THRESHOLD_CRITICAL, RISK_THRESHOLD_UNACCEPTABLE,
    SAFETY_MARGIN_STRUCTURAL, SAFETY_MARGIN_THERMAL,
    SAFETY_MARGIN_POWER
)

logger = logging.getLogger(__name__)

@dataclass
class RiskComponent:
    """Individual risk component with weight and value."""
    name: str
    value: float  # [0, 1]
    weight: float  # Contribution to overall risk
    description: str
    thresholds: Tuple[float, float, float]  # (warning, critical, unacceptable)

class RiskQuantifier:
    """
    Quantifies risk across multiple dimensions.
    
    Risk factors considered (from ARVS document Section 5.6):
    1. Torque limits
    2. Structural load limits
    3. Thermal limits
    4. Battery/power limits
    5. Collision risk
    6. Communication blackouts
    7. Joint collision risk
    8. Hazard proximity
    
    All risks are normalized to [0, 1] for comparability.
    """
    
    def __init__(self, safety_constraints: SafetyConstraints):
        """
        Initialize risk quantifier.
        
        Args:
            safety_constraints: Hard safety constraints for risk calculation
        """
        self.safety_constraints = safety_constraints
        
        # Define risk components with weights (sum to 1.0)
        self.risk_components = {
            'torque': RiskComponent(
                name='torque',
                value=0.0,
                weight=0.15,
                description='Joint torque exceeding limits',
                thresholds=(0.7, 0.85, 0.95)
            ),
            'structural': RiskComponent(
                name='structural',
                value=0.0,
                weight=0.15,
                description='Structural load exceeding limits',
                thresholds=(0.7, 0.85, 0.95)
            ),
            'thermal': RiskComponent(
                name='thermal',
                value=0.0,
                weight=0.15,
                description='Temperature exceeding limits',
                thresholds=(0.7, 0.85, 0.95)
            ),
            'power': RiskComponent(
                name='power',
                value=0.0,
                weight=0.15,
                description='Battery/power constraints',
                thresholds=(0.3, 0.15, 0.05)  # Lower is worse for battery
            ),
            'collision': RiskComponent(
                name='collision',
                value=0.0,
                weight=0.15,
                description='Collision risk with environment',
                thresholds=(0.5, 0.7, 0.9)
            ),
            'hazard_proximity': RiskComponent(
                name='hazard_proximity',
                value=0.0,
                weight=0.10,
                description='Proximity to known hazards',
                thresholds=(0.5, 0.7, 0.9)
            ),
            'communication': RiskComponent(
                name='communication',
                value=0.0,
                weight=0.05,
                description='Communication blackout risk',
                thresholds=(0.5, 0.7, 0.9)
            ),
            'uncertainty': RiskComponent(
                name='uncertainty',
                value=0.0,
                weight=0.10,
                description='State estimation uncertainty',
                thresholds=(0.5, 0.7, 0.9)
            )
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(comp.weight for comp in self.risk_components.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Risk weights sum to {total_weight}, normalizing")
            for comp in self.risk_components.values():
                comp.weight /= total_weight
        
        logger.info("Risk quantifier initialized")
    
    def assess_risk(self, robot_state: RobotState, 
                   belief_state: Optional[BeliefState] = None,
                   additional_context: Optional[Dict] = None) -> RiskAssessment:
        """
        Compute comprehensive risk assessment.
        
        Args:
            robot_state: Current robot state
            belief_state: Optional belief state for uncertainty assessment
            additional_context: Additional context (obstacles, etc.)
            
        Returns:
            RiskAssessment with normalized risk metrics
        """
        # Update all risk components
        self._update_torque_risk(robot_state)
        self._update_structural_risk(robot_state)
        self._update_thermal_risk(robot_state)
        self._update_power_risk(robot_state)
        self._update_collision_risk(robot_state, additional_context)
        self._update_hazard_proximity_risk(robot_state, additional_context)
        self._update_communication_risk(robot_state.timestamp)
        self._update_uncertainty_risk(belief_state)
        
        # Calculate weighted overall risk
        overall_risk = sum(
            comp.value * comp.weight 
            for comp in self.risk_components.values()
        )
        
        # Apply risk aggregation heuristics
        overall_risk = self._apply_risk_aggregation(overall_risk)
        
        # Create component risks dictionary
        component_risks = {
            comp.name: comp.value 
            for comp in self.risk_components.values()
        }
        
        # Create risk factors dictionary (raw values before weighting)
        risk_factors = {
            comp.name: comp.value 
            for comp in self.risk_components.values()
        }
        
        # Determine confidence based on state confidence
        confidence = robot_state.confidence if hasattr(robot_state, 'confidence') else 1.0
        
        return RiskAssessment(
            timestamp=robot_state.timestamp,
            overall_risk=float(overall_risk),
            component_risks=component_risks,
            risk_factors=risk_factors,
            confidence=confidence
        )
    
    def _update_torque_risk(self, robot_state: RobotState):
        """Compute risk from joint torque limits."""
        if not robot_state.joint_torques or not self.safety_constraints.max_torque:
            self.risk_components['torque'].value = 0.0
            return
        
        max_torque_risk = 0.0
        for joint, torque in robot_state.joint_torques.items():
            if joint in self.safety_constraints.max_torque:
                max_torque = self.safety_constraints.max_torque[joint]
                # Normalized risk: 0 at 0 torque, 1 at max torque (with safety margin)
                safe_max = max_torque / SAFETY_MARGIN_STRUCTURAL
                torque_risk = min(1.0, abs(torque) / safe_max)
                max_torque_risk = max(max_torque_risk, torque_risk)
        
        self.risk_components['torque'].value = max_torque_risk
    
    def _update_structural_risk(self, robot_state: RobotState):
        """Compute risk from structural load limits."""
        # Simplified structural risk calculation
        # In practice, would use finite element analysis or precomputed load envelopes
        
        if not self.safety_constraints.structural_load_limits:
            self.risk_components['structural'].value = 0.0
            return
        
        # Combine multiple load factors (simplified)
        load_factors = []
        
        # Velocity contributes to dynamic loads
        velocity_norm = np.linalg.norm(robot_state.velocity)
        load_factors.append(min(1.0, velocity_norm / 5.0))  # Normalize by 5 m/s
        
        # Joint torques contribute
        if robot_state.joint_torques:
            max_torque = max(abs(t) for t in robot_state.joint_torques.values()) if robot_state.joint_torques else 0
            load_factors.append(min(1.0, max_torque / 100.0))  # Normalize by 100 N-m
        
        # Take maximum load factor
        structural_risk = max(load_factors) if load_factors else 0.0
        self.risk_components['structural'].value = structural_risk
    
    def _update_thermal_risk(self, robot_state: RobotState):
        """Compute risk from thermal limits."""
        if not self.safety_constraints.thermal_limits:
            self.risk_components['thermal'].value = 0.0
            return
        
        max_thermal_risk = 0.0
        
        # Check robot temperature against limits
        for component, limit in self.safety_constraints.thermal_limits.items():
            # Map component to actual temperature measurement
            # Simplified: use robot state temperature for all components
            temp = robot_state.temperature
            
            # Apply thermal safety margin
            safe_limit = limit * SAFETY_MARGIN_THERMAL
            
            # Normalized risk: 0 at nominal temp, 1 at safe limit
            # Assuming nominal temp is 293K (20°C)
            nominal_temp = 293.0
            if temp <= nominal_temp:
                thermal_risk = 0.0
            else:
                thermal_risk = min(1.0, (temp - nominal_temp) / (safe_limit - nominal_temp))
            
            max_thermal_risk = max(max_thermal_risk, thermal_risk)
        
        self.risk_components['thermal'].value = max_thermal_risk
    
    def _update_power_risk(self, robot_state: RobotState):
        """Compute risk from power/battery constraints."""
        battery_level = robot_state.battery_level
        
        # Risk increases as battery depletes
        # 0 risk at 100%, 1 risk at min_battery level
        min_battery = self.safety_constraints.min_battery
        
        if battery_level <= min_battery:
            power_risk = 1.0
        elif battery_level >= 1.0:
            power_risk = 0.0
        else:
            # Linear risk increase as battery depletes
            power_risk = (min_battery + SAFETY_MARGIN_POWER - battery_level) / SAFETY_MARGIN_POWER
            power_risk = max(0.0, min(1.0, power_risk))
        
        self.risk_components['power'].value = power_risk
    
    def _update_collision_risk(self, robot_state: RobotState, 
                              additional_context: Optional[Dict] = None):
        """Compute risk from collision with environment."""
        collision_risk = 0.0
        
        if additional_context and 'obstacles' in additional_context:
            obstacles = additional_context['obstacles']
            
            for obstacle in obstacles:
                if 'position' in obstacle and 'radius' in obstacle:
                    obs_pos = np.array(obstacle['position'])
                    obs_radius = obstacle['radius']
                    
                    # Distance to obstacle
                    distance = np.linalg.norm(robot_state.position - obs_pos)
                    
                    # Critical distance (touching)
                    critical_distance = obs_radius + 0.5  # Robot radius (0.5m)
                    
                    if distance <= critical_distance:
                        collision_risk = 1.0
                        break
                    else:
                        # Risk decreases with distance
                        safe_distance = critical_distance * 2.0
                        if distance < safe_distance:
                            risk = 1.0 - (distance - critical_distance) / (safe_distance - critical_distance)
                            collision_risk = max(collision_risk, risk)
        
        self.risk_components['collision'].value = collision_risk
    
    def _update_hazard_proximity_risk(self, robot_state: RobotState,
                                     additional_context: Optional[Dict] = None):
        """Compute risk from proximity to known hazards."""
        hazard_risk = 0.0
        
        # Check collision zones from safety constraints
        for zone in self.safety_constraints.collision_zones:
            if 'type' in zone and zone['type'] == 'cylinder':
                # Cylindrical hazard zone
                center = np.array(zone['center'])
                radius = zone['radius']
                height = zone.get('height', 10.0)  # Default height
                
                distance_2d = np.linalg.norm(robot_state.position[:2] - center[:2])
                height_diff = abs(robot_state.position[2] - center[2])
                
                if distance_2d <= radius and height_diff <= height/2:
                    # Inside hazard zone
                    hazard_risk = 1.0
                    break
                elif distance_2d <= radius * 1.5:
                    # Near hazard zone
                    proximity_risk = 1.0 - (distance_2d - radius) / (radius * 0.5)
                    hazard_risk = max(hazard_risk, proximity_risk)
        
        self.risk_components['hazard_proximity'].value = hazard_risk
    
    def _update_communication_risk(self, timestamp: float):
        """Compute risk from communication blackouts."""
        comm_risk = 0.0
        
        for blackout_start, blackout_end in self.safety_constraints.communication_blackouts:
            if blackout_start <= timestamp <= blackout_end:
                # Currently in blackout
                comm_risk = 1.0
                break
            elif blackout_start - 60 <= timestamp <= blackout_start:
                # Approaching blackout (within 60 seconds)
                time_to_blackout = blackout_start - timestamp
                comm_risk = max(comm_risk, 1.0 - time_to_blackout / 60.0)
        
        self.risk_components['communication'].value = comm_risk
    
    def _update_uncertainty_risk(self, belief_state: Optional[BeliefState] = None):
        """Compute risk from state estimation uncertainty."""
        if belief_state is None:
            self.risk_components['uncertainty'].value = 0.0
            return
        
        # Combine uncertainties from different state components
        uncertainties = []
        
        # Position uncertainty
        if 'position' in belief_state.covariance:
            pos_cov = belief_state.covariance['position']
            pos_uncertainty = np.sqrt(np.trace(pos_cov)) / 10.0  # Normalize by 10m
            uncertainties.append(min(1.0, pos_uncertainty))
        
        # Orientation uncertainty
        if 'orientation' in belief_state.covariance:
            ori_cov = belief_state.covariance['orientation']
            ori_uncertainty = np.sqrt(np.trace(ori_cov)) / 1.0  # Normalize by 1 rad
            uncertainties.append(min(1.0, ori_uncertainty))
        
        # Overall belief confidence
        confidence_risk = 1.0 - belief_state.confidence
        uncertainties.append(confidence_risk)
        
        # Take maximum uncertainty
        uncertainty_risk = max(uncertainties) if uncertainties else 0.0
        self.risk_components['uncertainty'].value = uncertainty_risk
    
    def _apply_risk_aggregation(self, weighted_risk: float) -> float:
        """
        Apply risk aggregation heuristics.
        
        Implements non-linear aggregation to account for:
        1. Risk amplification when multiple risks are high
        2. Risk saturation effects
        3. Critical risk domination
        """
        # Get maximum individual risk
        max_component_risk = max(comp.value for comp in self.risk_components.values())
        
        # Non-linear aggregation: amplify if multiple risks are high
        high_risks = sum(1 for comp in self.risk_components.values() 
                        if comp.value > 0.7)
        
        if high_risks >= 2:
            # Multiple high risks - amplify
            aggregated_risk = min(1.0, weighted_risk * (1.0 + 0.2 * high_risks))
        elif max_component_risk > 0.9:
            # Single critical risk dominates
            aggregated_risk = max_component_risk
        else:
            aggregated_risk = weighted_risk
        
        # Apply sigmoid-like saturation
        aggregated_risk = 1.0 / (1.0 + np.exp(-10.0 * (aggregated_risk - 0.5)))
        
        return float(aggregated_risk)
    
    def get_risk_level(self, risk_value: float) -> str:
        """Convert risk value to qualitative level."""
        if risk_value <= RISK_THRESHOLD_SAFE:
            return "SAFE"
        elif risk_value <= RISK_THRESHOLD_WARNING:
            return "WARNING"
        elif risk_value <= RISK_THRESHOLD_CRITICAL:
            return "CRITICAL"
        else:
            return "UNACCEPTABLE"
    
    def get_component_breakdown(self) -> Dict[str, Dict[str, float]]:
        """Get detailed breakdown of all risk components."""
        breakdown = {}
        for name, comp in self.risk_components.items():
            breakdown[name] = {
                'value': comp.value,
                'weight': comp.weight,
                'weighted_contribution': comp.value * comp.weight,
                'level': self.get_risk_level(comp.value),
                'description': comp.description
            }
        return breakdown
    
    def reset(self):
        """Reset all risk components to zero."""
        for comp in self.risk_components.values():
            comp.value = 0.0
        logger.info("Risk quantifier reset")
        