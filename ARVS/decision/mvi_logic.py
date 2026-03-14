

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CORE DATA TYPES =====

@dataclass
class RobotState:
    """Robot state for MVI generation."""
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

@dataclass
class RiskAssessment:
    """Risk assessment for MVI generation."""
    timestamp: float
    overall_risk: float
    component_risks: Dict[str, float]
    risk_factors: Dict[str, float]
    confidence: float = 1.0

@dataclass
class Action:
    """Action for MVI generation."""
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
    """MVI sequence."""
    sequence_id: str
    actions: List[Action]
    expected_duration: float
    predicted_risk: RiskAssessment
    qubo_solution: Optional[np.ndarray] = None
    
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
    
    def validate_action(self, action: Action) -> Tuple[bool, List[str]]:
        """Validate action against constraints."""
        violations = []
        
        if action.max_torque is not None:
            for joint, max_torque in self.max_torque.items():
                if action.max_torque > max_torque:
                    violations.append(f"Torque violation on {joint}")
        
        if action.thermal_load is not None:
            for component, limit in self.thermal_limits.items():
                if action.thermal_load > limit:
                    violations.append(f"Thermal violation on {component}")
        
        return len(violations) == 0, violations

@dataclass
class OptimizationResult:
    """Optimization result."""
    success: bool
    solution: Optional[np.ndarray]
    objective_value: float
    solver_time: float
    solver_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationProblem:
    """Optimization problem."""
    problem_id: str
    objective_matrix: np.ndarray
    variable_names: List[str]
    time_limit: float = 0.5
    
    @property
    def num_variables(self) -> int:
        return len(self.variable_names)

# ===== MVI LOGIC =====

class MVIStrategy(Enum):
    """MVI strategies."""
    FAULT_ISOLATION = "fault_isolation"
    TORQUE_BALANCING = "torque_balancing"
    THERMAL_BLEED_OFF = "thermal_bleed_off"
    PATH_CORRECTION = "path_correction"
    RESOURCE_REALLOCATION = "resource_reallocation"
    SWARM_RE_COORDINATION = "swarm_re_coordination"

@dataclass
class MVIProfile:
    """MVI profile."""
    max_actions: int
    allowed_strategies: set
    risk_tolerance: float
    time_limit: float
    
    @classmethod
    def emergency_profile(cls):
        return cls(
            max_actions=3,
            allowed_strategies={
                MVIStrategy.FAULT_ISOLATION,
                MVIStrategy.THERMAL_BLEED_OFF
            },
            risk_tolerance=0.1,
            time_limit=0.1
        )
    
    @classmethod
    def normal_profile(cls):
        return cls(
            max_actions=10,
            allowed_strategies=set(MVIStrategy),
            risk_tolerance=0.3,
            time_limit=0.5
        )
    
    @classmethod
    def degraded_profile(cls):
        return cls(
            max_actions=5,
            allowed_strategies={
                MVIStrategy.FAULT_ISOLATION,
                MVIStrategy.PATH_CORRECTION,
                MVIStrategy.RESOURCE_REALLOCATION
            },
            risk_tolerance=0.2,
            time_limit=0.3
        )

class MVILogic:
    """
    Generates Minimum Viable Intervention sequences.
    Complete standalone implementation.
    """
    
    def __init__(self, safety_constraints: SafetyConstraints):
        self.safety_constraints = safety_constraints
        self.action_library: Dict[str, Action] = {}
        self.strategy_weights = {
            MVIStrategy.FAULT_ISOLATION: 0.3,
            MVIStrategy.TORQUE_BALANCING: 0.2,
            MVIStrategy.THERMAL_BLEED_OFF: 0.15,
            MVIStrategy.PATH_CORRECTION: 0.15,
            MVIStrategy.RESOURCE_REALLOCATION: 0.1,
            MVIStrategy.SWARM_RE_COORDINATION: 0.1
        }
        
        self._initialize_action_library()
        self.generation_stats = {
            'times': [],
            'action_counts': [],
            'risk_reductions': []
        }
        
        logger.info("MVI logic initialized")
    
    def _initialize_action_library(self):
        """Initialize action library."""
        self.action_library['halt'] = Action(
            action_id='halt',
            action_type='safety',
            parameters={'mode': 'immediate'},
            duration=0.1,
            priority=100,
            max_torque=0.0,
            max_velocity=0.0,
            thermal_load=0.0,
            power_required=10.0
        )
        
        self.action_library['safe_hold'] = Action(
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
        
        self.action_library['reduce_speed'] = Action(
            action_id='reduce_speed',
            action_type='motion',
            parameters={'factor': 0.5},
            duration=2.0,
            priority=50,
            max_torque=30.0,
            max_velocity=1.0,
            thermal_load=15.0,
            power_required=50.0
        )
        
        self.action_library['thermal_bleed'] = Action(
            action_id='thermal_bleed',
            action_type='thermal',
            parameters={'mode': 'active_cooling'},
            duration=5.0,
            priority=60,
            max_torque=0.0,
            max_velocity=0.0,
            thermal_load=-20.0,
            power_required=100.0
        )
        
        self.action_library['torque_balance'] = Action(
            action_id='torque_balance',
            action_type='manipulation',
            parameters={'method': 'redistribute'},
            duration=3.0,
            priority=70,
            max_torque=40.0,
            max_velocity=0.5,
            thermal_load=25.0,
            power_required=60.0
        )
    
    def generate_mvi(self, robot_state: RobotState,
                    risk_assessment: RiskAssessment,
                    optimization_result: OptimizationResult,
                    optimization_problem: OptimizationProblem,
                    profile: MVIProfile = None) -> MVISequence:
        """Generate MVI sequence."""
        start_time = time.time()
        
        if profile is None:
            profile = self._select_profile(risk_assessment)
        
        logger.info(f"Generating MVI with profile: {profile}")
        
        try:
            # Extract selected actions
            selected_actions = self._extract_actions_from_solution(
                optimization_result, optimization_problem
            )
            
            # Apply strategy filtering
            filtered_actions = self._apply_strategy_filtering(
                selected_actions, profile.allowed_strategies, risk_assessment
            )
            
            # Limit number of actions
            if len(filtered_actions) > profile.max_actions:
                filtered_actions = self._select_minimal_action_set(
                    filtered_actions, profile.max_actions, risk_assessment
                )
            
            # Order actions
            ordered_actions = self._order_actions(filtered_actions, robot_state)
            
            # Create MVI sequence
            mvi_sequence = self._create_mvi_sequence(
                ordered_actions, robot_state, risk_assessment, 
                optimization_result, profile
            )
            
            # Validate MVI
            if not self._validate_mvi(mvi_sequence, robot_state):
                logger.warning("Generated MVI failed validation, generating fallback")
                mvi_sequence = self._generate_fallback_mvi(robot_state, risk_assessment, profile)
            
            # Update statistics
            self._update_statistics(start_time, mvi_sequence, risk_assessment)
            
            return mvi_sequence
            
        except Exception as e:
            logger.error(f"MVI generation failed: {e}")
            return self._generate_emergency_mvi(robot_state, risk_assessment)
    
    def _select_profile(self, risk_assessment: RiskAssessment) -> MVIProfile:
        """Select MVI profile based on risk."""
        if risk_assessment.overall_risk > 0.8:
            return MVIProfile.emergency_profile()
        elif risk_assessment.overall_risk > 0.5:
            return MVIProfile.degraded_profile()
        else:
            return MVIProfile.normal_profile()
    
    def _extract_actions_from_solution(self,
                                     optimization_result: OptimizationResult,
                                     optimization_problem: OptimizationProblem) -> List[Action]:
        """Extract selected actions from solution."""
        selected_actions = []
        
        if optimization_result.solution is None:
            return selected_actions
        
        for i, var_name in enumerate(optimization_problem.variable_names):
            if var_name.startswith('action_') and optimization_result.solution[i] > 0.5:
                parts = var_name.split('_')
                if len(parts) >= 3:
                    action_id = '_'.join(parts[2:])
                    if action_id in self.action_library:
                        selected_actions.append(self.action_library[action_id])
        
        return selected_actions
    
    def _apply_strategy_filtering(self, actions: List[Action],
                                 allowed_strategies: set,
                                 risk_assessment: RiskAssessment) -> List[Action]:
        """Filter actions based on allowed strategies."""
        if not allowed_strategies:
            return actions
        
        filtered = []
        for action in actions:
            action_strategy = self._classify_action_strategy(action, risk_assessment)
            if action_strategy in allowed_strategies:
                filtered.append(action)
        
        return filtered
    
    def _classify_action_strategy(self, action: Action,
                                 risk_assessment: RiskAssessment) -> MVIStrategy:
        """Classify action strategy."""
        action_id = action.action_id.lower()
        
        if 'halt' in action_id or 'hold' in action_id:
            return MVIStrategy.FAULT_ISOLATION
        elif 'torque' in action_id or 'balance' in action_id:
            return MVIStrategy.TORQUE_BALANCING
        elif 'thermal' in action_id or 'cool' in action_id or 'bleed' in action_id:
            return MVIStrategy.THERMAL_BLEED_OFF
        elif 'path' in action_id or 'correct' in action_id or 'adjust' in action_id:
            return MVIStrategy.PATH_CORRECTION
        elif 'resource' in action_id or 'reallocate' in action_id:
            return MVIStrategy.RESOURCE_REALLOCATION
        elif 'swarm' in action_id or 'coord' in action_id:
            return MVIStrategy.SWARM_RE_COORDINATION
        else:
            # Default based on highest risk
            max_risk_component = max(
                risk_assessment.component_risks.items(),
                key=lambda x: x[1]
            )[0]
            
            if 'thermal' in max_risk_component:
                return MVIStrategy.THERMAL_BLEED_OFF
            elif 'torque' in max_risk_component or 'structural' in max_risk_component:
                return MVIStrategy.TORQUE_BALANCING
            else:
                return MVIStrategy.FAULT_ISOLATION
    
    def _select_minimal_action_set(self, actions: List[Action],
                                  max_actions: int,
                                  risk_assessment: RiskAssessment) -> List[Action]:
        """Select minimal action set."""
        if len(actions) <= max_actions:
            return actions
        
        # Score actions
        action_scores = []
        for action in actions:
            score = self._score_action_effectiveness(action, risk_assessment)
            action_scores.append((score, action))
        
        # Sort and select top N
        action_scores.sort(key=lambda x: x[0], reverse=True)
        selected = [action for _, action in action_scores[:max_actions]]
        
        # Ensure safety action if high risk
        if risk_assessment.overall_risk > 0.7:
            safety_actions = [a for a in selected if a.action_type == 'safety']
            if not safety_actions and actions:
                safety_candidates = [a for a in actions if a.action_type == 'safety']
                if safety_candidates:
                    selected[-1] = safety_candidates[0]
        
        return selected
    
    def _score_action_effectiveness(self, action: Action,
                                   risk_assessment: RiskAssessment) -> float:
        """Score action effectiveness."""
        score = 0.0
        
        # Base score from priority
        score += action.priority / 100.0
        
        # Address thermal risks
        if action.thermal_load is not None and action.thermal_load < 0:
            thermal_risk = risk_assessment.component_risks.get('thermal', 0.0)
            score += thermal_risk * 2.0
        
        # Address torque/structural risks
        if action.max_torque is not None and action.max_torque < 30.0:
            torque_risk = risk_assessment.component_risks.get('torque', 0.0)
            structural_risk = risk_assessment.component_risks.get('structural', 0.0)
            score += (torque_risk + structural_risk) * 1.5
        
        # Penalize high resource usage
        if action.power_required and action.power_required > 50.0:
            power_risk = risk_assessment.component_risks.get('power', 0.0)
            score -= power_risk * 0.5
        
        # Prefer shorter duration in high-risk
        if risk_assessment.overall_risk > 0.7:
            duration_score = max(0.0, 1.0 - action.duration / 10.0)
            score += duration_score
        
        return max(0.0, score)
    
    def _order_actions(self, actions: List[Action], robot_state: RobotState) -> List[Action]:
        """Order actions."""
        if not actions:
            return []
        
        # Group by type
        safety_actions = [a for a in actions if a.action_type == 'safety']
        thermal_actions = [a for a in actions if 'thermal' in a.action_id]
        motion_actions = [a for a in actions if a.action_type == 'motion']
        other_actions = [a for a in actions if a not in safety_actions + thermal_actions + motion_actions]
        
        # Sort by priority
        safety_actions.sort(key=lambda x: x.priority, reverse=True)
        thermal_actions.sort(key=lambda x: x.priority, reverse=True)
        motion_actions.sort(key=lambda x: x.priority, reverse=True)
        other_actions.sort(key=lambda x: x.priority, reverse=True)
        
        # Combine: safety first, then thermal, then motion, then others
        ordered = safety_actions + thermal_actions + motion_actions + other_actions
        
        # Battery critical: power-saving actions first
        if robot_state.battery_level < 0.2:
            power_saving = [a for a in ordered if a.power_required and a.power_required < 20.0]
            others = [a for a in ordered if a not in power_saving]
            ordered = power_saving + others
        
        return ordered
    
    def _create_mvi_sequence(self, actions: List[Action], robot_state: RobotState,
                            risk_assessment: RiskAssessment,
                            optimization_result: OptimizationResult,
                            profile: MVIProfile) -> MVISequence:
        """Create MVI sequence."""
        expected_duration = sum(action.duration for action in actions) * 1.2
        current_risk = risk_assessment.overall_risk
        risk_reduction = self._estimate_risk_reduction(actions, risk_assessment)
        predicted_risk = max(0.0, current_risk - risk_reduction)
        
        predicted_risk_assessment = RiskAssessment(
            timestamp=robot_state.timestamp + expected_duration,
            overall_risk=predicted_risk,
            component_risks={k: max(0.0, v - risk_reduction * 0.5) 
                           for k, v in risk_assessment.component_risks.items()},
            risk_factors=risk_assessment.risk_factors.copy(),
            confidence=risk_assessment.confidence * 0.9
        )
        
        sequence_id = f"mvi_{robot_state.timestamp}_{len(actions)}a_{int(predicted_risk*100)}r"
        
        return MVISequence(
            sequence_id=sequence_id,
            actions=actions,
            expected_duration=expected_duration,
            predicted_risk=predicted_risk_assessment,
            qubo_solution=optimization_result.solution
        )
    
    def _estimate_risk_reduction(self, actions: List[Action],
                                risk_assessment: RiskAssessment) -> float:
        """Estimate risk reduction."""
        total_reduction = 0.0
        
        for action in actions:
            if action.action_type == 'safety':
                total_reduction += 0.2
            elif 'thermal' in action.action_id and action.thermal_load < 0:
                thermal_risk = risk_assessment.component_risks.get('thermal', 0.0)
                total_reduction += thermal_risk * 0.3
            elif 'torque' in action.action_id and action.max_torque < 30.0:
                torque_risk = risk_assessment.component_risks.get('torque', 0.0)
                structural_risk = risk_assessment.component_risks.get('structural', 0.0)
                total_reduction += (torque_risk + structural_risk) * 0.2
            elif action.power_required and action.power_required < 20.0:
                power_risk = risk_assessment.component_risks.get('power', 0.0)
                total_reduction += power_risk * 0.1
        
        max_reduction = min(0.8, 0.2 * len(actions))
        return min(total_reduction, max_reduction)
    
    def _validate_mvi(self, mvi_sequence: MVISequence,
                     robot_state: RobotState) -> bool:
        """Validate MVI sequence."""
        if not mvi_sequence.actions:
            return False
        
        for action in mvi_sequence.actions:
            valid, violations = self.safety_constraints.validate_action(action)
            if not valid:
                logger.warning(f"Action {action.action_id} violates constraints: {violations}")
                return False
        
        if mvi_sequence.expected_duration > 60.0:
            logger.warning(f"MVI duration too long: {mvi_sequence.expected_duration}s")
            return False
        
        if mvi_sequence.predicted_risk.overall_risk > 0.9:
            logger.warning(f"Predicted risk too high: {mvi_sequence.predicted_risk.overall_risk}")
            return False
        
        return True
    
    def _generate_fallback_mvi(self, robot_state: RobotState,
                              risk_assessment: RiskAssessment,
                              profile: MVIProfile) -> MVISequence:
        """Generate fallback MVI."""
        logger.info("Generating fallback MVI")
        
        fallback_actions = [self.action_library['halt']]
        
        max_risk_component = max(
            risk_assessment.component_risks.items(),
            key=lambda x: x[1]
        )[0]
        
        if 'thermal' in max_risk_component:
            fallback_actions.append(self.action_library['thermal_bleed'])
        elif 'torque' in max_risk_component or 'structural' in max_risk_component:
            fallback_actions.append(self.action_library['torque_balance'])
        
        if len(fallback_actions) > profile.max_actions:
            fallback_actions = fallback_actions[:profile.max_actions]
        
        return self._create_mvi_sequence(
            fallback_actions, robot_state, risk_assessment,
            OptimizationResult(
                success=False,
                solution=None,
                objective_value=float('inf'),
                solver_time=0.0,
                metadata={'fallback': True}
            ),
            profile
        )
    
    def _generate_emergency_mvi(self, robot_state: RobotState,
                               risk_assessment: RiskAssessment) -> MVISequence:
        """Generate emergency MVI."""
        logger.warning("Generating emergency MVI")
        
        emergency_actions = [
            self.action_library['halt'],
            self.action_library['safe_hold']
        ]
        
        expected_duration = 5.0
        
        emergency_risk = RiskAssessment(
            timestamp=robot_state.timestamp + expected_duration,
            overall_risk=min(1.0, risk_assessment.overall_risk * 0.7),
            component_risks={k: v * 0.7 for k, v in risk_assessment.component_risks.items()},
            risk_factors=risk_assessment.risk_factors.copy(),
            confidence=0.5
        )
        
        return MVISequence(
            sequence_id=f"emergency_{robot_state.timestamp}",
            actions=emergency_actions,
            expected_duration=expected_duration,
            predicted_risk=emergency_risk,
            qubo_solution=None
        )
    
    def _update_statistics(self, start_time: float,
                          mvi_sequence: MVISequence,
                          risk_assessment: RiskAssessment):
        """Update statistics."""
        gen_time = time.time() - start_time
        
        self.generation_stats['times'].append(gen_time)
        self.generation_stats['action_counts'].append(len(mvi_sequence.actions))
        
        if len(self.generation_stats['times']) > 100:
            for key in self.generation_stats:
                if self.generation_stats[key]:
                    self.generation_stats[key].pop(0)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistics."""
        stats = {}
        
        for key, values in self.generation_stats.items():
            if values:
                stats[f'avg_{key}'] = float(np.mean(values))
                stats[f'max_{key}'] = float(np.max(values))
                stats[f'min_{key}'] = float(np.min(values))
        
        return stats
    
    def add_custom_action(self, action: Action):
        """Add custom action."""
        self.action_library[action.action_id] = action
        logger.info(f"Added custom action: {action.action_id}")
    
    def remove_action(self, action_id: str):
        """Remove action."""
        if action_id in self.action_library:
            del self.action_library[action_id]
            logger.info(f"Removed action: {action_id}")
    
    def reset(self):
        """Reset MVI logic."""
        self.generation_stats = {'times': [], 'action_counts': [], 'risk_reductions': []}
        logger.info("MVI logic reset")

# ===== DEMONSTRATION =====

def demonstrate_mvi_logic():
    """Demonstrate MVI logic functionality."""
    print("=" * 70)
    print("MVI LOGIC DEMONSTRATION")
    print("Complete standalone implementation")
    print("=" * 70)
    
    # Create safety constraints
    constraints = SafetyConstraints(
        max_torque={'joint1': 100.0, 'joint2': 80.0},
        max_velocity={'joint1': 2.0, 'joint2': 1.5},
        thermal_limits={'motor': 373.0, 'cpu': 358.0},
        structural_load_limits={'arm': 200.0},
        min_battery=0.15
    )
    
    # Create MVI logic
    mvi_logic = MVILogic(constraints)
    
    # Create test robot state
    robot_state = RobotState(
        robot_id="test_robot",
        timestamp=time.time(),
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.5, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        temperature=330.0,  # High temperature
        battery_level=0.4,  # Medium battery
        power_consumption=60.0
    )
    
    # Create test risk assessment
    risk_assessment = RiskAssessment(
        timestamp=time.time(),
        overall_risk=0.65,  # High risk
        component_risks={
            'thermal': 0.8,
            'power': 0.5,
            'structural': 0.3
        },
        risk_factors={
            'temperature': 330.0,
            'battery': 0.4,
            'velocity': 0.5
        }
    )
    
    # Create mock optimization result
    optimization_result = OptimizationResult(
        success=True,
        solution=np.array([1, 0, 1, 0, 1]),  # Select halt, reduce_speed, torque_balance
        objective_value=150.0,
        solver_time=0.1,
        solver_type="classical"
    )
    
    # Create mock optimization problem
    optimization_problem = OptimizationProblem(
        problem_id="test_problem",
        objective_matrix=np.eye(5),
        variable_names=['action_halt', 'action_safe_hold', 'action_reduce_speed', 
                       'action_thermal_bleed', 'action_torque_balance']
    )
    
    print("\n1. Testing MVI generation with high risk (emergency profile):")
    mvi_sequence = mvi_logic.generate_mvi(
        robot_state,
        risk_assessment,
        optimization_result,
        optimization_problem,
        MVIProfile.emergency_profile()
    )
    
    print(f"   Sequence ID: {mvi_sequence.sequence_id}")
    print(f"   Number of actions: {mvi_sequence.action_count}")
    print(f"   Expected duration: {mvi_sequence.expected_duration:.2f}s")
    print(f"   Predicted risk: {mvi_sequence.predicted_risk.overall_risk:.3f}")
    print(f"   Actions:")
    for i, action in enumerate(mvi_sequence.actions):
        print(f"     {i+1}. {action.action_id} ({action.action_type}) - {action.duration}s")
    
    print("\n2. Testing with normal risk (normal profile):")
    low_risk_assessment = RiskAssessment(
        timestamp=time.time(),
        overall_risk=0.3,
        component_risks={'thermal': 0.2, 'power': 0.1},
        risk_factors={'temperature': 310.0, 'battery': 0.7}
    )
    
    mvi_sequence2 = mvi_logic.generate_mvi(
        robot_state,
        low_risk_assessment,
        optimization_result,
        optimization_problem
    )
    
    print(f"   Sequence ID: {mvi_sequence2.sequence_id}")
    print(f"   Number of actions: {mvi_sequence2.action_count}")
    print(f"   Expected duration: {mvi_sequence2.expected_duration:.2f}s")
    
    print("\n3. MVI Logic statistics:")
    stats = mvi_logic.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n4. Testing action library management:")
    custom_action = Action(
        action_id='custom_scan',
        action_type='perception',
        parameters={'mode': 'full_scan'},
        duration=3.0,
        priority=40,
        power_required=30.0
    )
    
    mvi_logic.add_custom_action(custom_action)
    print(f"   Added custom action: {custom_action.action_id}")
    print(f"   Total actions in library: {len(mvi_logic.action_library)}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_mvi_logic()