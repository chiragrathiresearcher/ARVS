"""
Multi-Robot Coordination Layer
Implements joint QUBO model for robot swarms from ARVS document Section 5.8.
Enables coordinated assembly, collision-free motion, shared resource handling,
and distributed fault recovery.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import time
from dataclasses import dataclass
from enum import Enum

from arvs.core.data_types import (
    RobotState, MultiRobotState, Action, MVISequence,
    OptimizationProblem, SafetyConstraints
)
from arvs.core.exceptions import NoFeasibleSolutionException
from arvs.core.constants import (
    MAX_ROBOTS_IN_SWARM, MIN_INTER_ROBOT_DISTANCE,
    MAX_INTER_ROBOT_DISTANCE
)

logger = logging.getLogger(__name__)

class CoordinationMode(Enum):
    """Modes of robot coordination."""
    CENTRALIZED = "centralized"  # Single joint optimization
    DISTRIBUTED = "distributed"  # Distributed optimization
    HIERARCHICAL = "hierarchical"  # Leader-follower
    DECENTRALIZED = "decentralized"  # Fully decentralized

class CoordinationConstraint(Enum):
    """Types of coordination constraints."""
    COLLISION_AVOIDANCE = "collision_avoidance"
    FORMATION_MAINTENANCE = "formation_maintenance"
    RESOURCE_SHARING = "resource_sharing"
    TASK_ALLOCATION = "task_allocation"
    COMMUNICATION_CONSTRAINT = "communication_constraint"

@dataclass
class RobotRole:
    """Role assignment for a robot in a swarm."""
    robot_id: str
    role: str  # 'leader', 'follower', 'specialist', etc.
    capabilities: Set[str]  # What this robot can do
    constraints: Dict[str, Any]  # Role-specific constraints

@dataclass
class SwarmState:
    """Complete state of a robot swarm."""
    swarm_id: str
    timestamp: float
    robot_states: Dict[str, RobotState]  # robot_id -> state
    robot_roles: Dict[str, RobotRole]  # robot_id -> role
    formation: Optional[Dict[str, Any]] = None  # Formation geometry
    shared_resources: Dict[str, Any] = None  # Shared resources state
    communication_graph: Dict[str, List[str]] = None  # Who can talk to whom
    
    @property
    def num_robots(self) -> int:
        """Number of robots in swarm."""
        return len(self.robot_states)
    
    @property
    def center_of_mass(self) -> np.ndarray:
        """Compute center of mass of swarm."""
        positions = [state.position for state in self.robot_states.values()]
        return np.mean(positions, axis=0) if positions else np.zeros(3)

class MultiRobotCoordinator:
    """
    Coordinates multiple robots using joint QUBO optimization.
    
    From ARVS document Section 5.8:
    - Builds joint QUBO model for N robots
    - Enables coordinated assembly
    - Ensures collision-free motion
    - Manages shared resources
    - Handles distributed fault recovery
    """
    
    def __init__(self, coordination_mode: CoordinationMode = CoordinationMode.CENTRALIZED):
        """
        Initialize multi-robot coordinator.
        
        Args:
            coordination_mode: Mode of coordination
        """
        self.coordination_mode = coordination_mode
        self.swarms: Dict[str, SwarmState] = {}
        self.coordination_constraints: Dict[str, List[CoordinationConstraint]] = {}
        
        # Communication parameters
        self.communication_range = 100.0  # meters
        self.communication_update_interval = 1.0  # seconds
        self.last_communication_update = 0.0
        
        # Optimization parameters for joint QUBO
        self.joint_qubo_penalty_weight = 1000.0  # Higher for inter-robot constraints
        
        # Formation templates
        self.formation_templates = {
            'line': self._create_line_formation,
            'grid': self._create_grid_formation,
            'circle': self._create_circle_formation,
            'v_shape': self._create_v_formation
        }
        
        # Performance tracking
        self.coordination_stats = {
            'joint_optimizations': 0,
            'collisions_prevented': 0,
            'resource_conflicts_resolved': 0,
            'formation_adjustments': 0,
            'avg_coordination_time': 0.0
        }
        
        logger.info(f"Multi-robot coordinator initialized with mode: {coordination_mode.value}")
    
    def register_swarm(self, swarm_id: str, robot_ids: List[str],
                      formation_type: str = None) -> SwarmState:
        """
        Register a new swarm of robots.
        
        Args:
            swarm_id: Unique swarm identifier
            robot_ids: List of robot IDs in swarm
            formation_type: Optional formation type
            
        Returns:
            SwarmState for the new swarm
        """
        if swarm_id in self.swarms:
            logger.warning(f"Swarm {swarm_id} already registered")
            return self.swarms[swarm_id]
        
        # Create initial robot states (will be updated later)
        robot_states = {}
        robot_roles = {}
        
        for i, robot_id in enumerate(robot_ids):
            # Create placeholder state
            robot_states[robot_id] = RobotState(
                robot_id=robot_id,
                timestamp=time.time(),
                position=np.zeros(3),
                velocity=np.zeros(3),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity=np.zeros(3),
                joint_positions={},
                joint_velocities={},
                joint_torques={},
                temperature=293.0,
                battery_level=1.0,
                power_consumption=0.0
            )
            
            # Assign roles (first robot is leader)
            if i == 0:
                role = RobotRole(
                    robot_id=robot_id,
                    role='leader',
                    capabilities={'navigation', 'coordination', 'decision'},
                    constraints={}
                )
            else:
                role = RobotRole(
                    robot_id=robot_id,
                    role='follower',
                    capabilities={'navigation', 'execution'},
                    constraints={'follow_leader': True}
                )
            robot_roles[robot_id] = role
        
        # Create formation if specified
        formation = None
        if formation_type and formation_type in self.formation_templates:
            formation = self.formation_templates[formation_type](robot_ids)
        
        # Create swarm state
        swarm_state = SwarmState(
            swarm_id=swarm_id,
            timestamp=time.time(),
            robot_states=robot_states,
            robot_roles=robot_roles,
            formation=formation,
            shared_resources={},
            communication_graph=self._create_communication_graph(robot_ids)
        )
        
        self.swarms[swarm_id] = swarm_state
        self.coordination_constraints[swarm_id] = [
            CoordinationConstraint.COLLISION_AVOIDANCE,
            CoordinationConstraint.FORMATION_MAINTENANCE
        ]
        
        logger.info(f"Registered swarm {swarm_id} with {len(robot_ids)} robots")
        return swarm_state
    
    def _create_line_formation(self, robot_ids: List[str]) -> Dict[str, Any]:
        """Create line formation geometry."""
        formation = {
            'type': 'line',
            'spacing': 2.0,  # meters between robots
            'orientation': [1.0, 0.0, 0.0],  # Direction of line
            'robot_positions': {}
        }
        
        for i, robot_id in enumerate(robot_ids):
            position = np.array([i * formation['spacing'], 0.0, 0.0])
            formation['robot_positions'][robot_id] = position.tolist()
        
        return formation
    
    def _create_grid_formation(self, robot_ids: List[str]) -> Dict[str, Any]:
        """Create grid formation geometry."""
        formation = {
            'type': 'grid',
            'row_spacing': 2.0,
            'col_spacing': 2.0,
            'rows': int(np.ceil(np.sqrt(len(robot_ids)))),
            'robot_positions': {}
        }
        
        for i, robot_id in enumerate(robot_ids):
            row = i // formation['rows']
            col = i % formation['rows']
            position = np.array([col * formation['col_spacing'], 
                                row * formation['row_spacing'], 0.0])
            formation['robot_positions'][robot_id] = position.tolist()
        
        return formation
    
    def _create_circle_formation(self, robot_ids: List[str]) -> Dict[str, Any]:
        """Create circle formation geometry."""
        formation = {
            'type': 'circle',
            'radius': 5.0,
            'robot_positions': {}
        }
        
        n = len(robot_ids)
        for i, robot_id in enumerate(robot_ids):
            angle = 2 * np.pi * i / n
            position = np.array([
                formation['radius'] * np.cos(angle),
                formation['radius'] * np.sin(angle),
                0.0
            ])
            formation['robot_positions'][robot_id] = position.tolist()
        
        return formation
    
    def _create_v_formation(self, robot_ids: List[str]) -> Dict[str, Any]:
        """Create V formation geometry (like migrating birds)."""
        formation = {
            'type': 'v_shape',
            'spacing': 3.0,
            'angle': np.pi / 6,  # 30 degrees
            'robot_positions': {}
        }
        
        leader_id = robot_ids[0]
        formation['robot_positions'][leader_id] = [0.0, 0.0, 0.0]
        
        # Assign positions in V shape
        for i, robot_id in enumerate(robot_ids[1:], 1):
            side = 1 if i % 2 == 0 else -1  # Alternate sides
            row = (i + 1) // 2
            
            x = row * formation['spacing'] * np.cos(formation['angle'])
            y = side * row * formation['spacing'] * np.sin(formation['angle'])
            
            formation['robot_positions'][robot_id] = [x, y, 0.0]
        
        return formation
    
    def _create_communication_graph(self, robot_ids: List[str]) -> Dict[str, List[str]]:
        """Create initial communication graph (fully connected)."""
        graph = {}
        for robot_id in robot_ids:
            graph[robot_id] = [other_id for other_id in robot_ids if other_id != robot_id]
        return graph
    
    def update_swarm_state(self, swarm_id: str, 
                          robot_states: Dict[str, RobotState]):
        """
        Update state of a swarm.
        
        Args:
            swarm_id: Swarm identifier
            robot_states: Updated robot states
        """
        if swarm_id not in self.swarms:
            logger.warning(f"Swarm {swarm_id} not registered")
            return
        
        swarm_state = self.swarms[swarm_id]
        swarm_state.timestamp = time.time()
        
        # Update individual robot states
        for robot_id, state in robot_states.items():
            if robot_id in swarm_state.robot_states:
                swarm_state.robot_states[robot_id] = state
        
        # Update communication graph based on distances
        current_time = time.time()
        if current_time - self.last_communication_update > self.communication_update_interval:
            self._update_communication_graph(swarm_state)
            self.last_communication_update = current_time
        
        # Check for constraint violations
        self._check_swarm_constraints(swarm_state)
    
    def _update_communication_graph(self, swarm_state: SwarmState):
        """Update communication graph based on current positions."""
        robot_ids = list(swarm_state.robot_states.keys())
        new_graph = {robot_id: [] for robot_id in robot_ids}
        
        for i, robot_id1 in enumerate(robot_ids):
            for robot_id2 in robot_ids[i+1:]:
                pos1 = swarm_state.robot_states[robot_id1].position
                pos2 = swarm_state.robot_states[robot_id2].position
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance <= self.communication_range:
                    new_graph[robot_id1].append(robot_id2)
                    new_graph[robot_id2].append(robot_id1)
        
        swarm_state.communication_graph = new_graph
    
    def _check_swarm_constraints(self, swarm_state: SwarmState):
        """Check for constraint violations in swarm."""
        violations = []
        
        # Check inter-robot distances
        robot_ids = list(swarm_state.robot_states.keys())
        for i, robot_id1 in enumerate(robot_ids):
            for robot_id2 in robot_ids[i+1:]:
                pos1 = swarm_state.robot_states[robot_id1].position
                pos2 = swarm_state.robot_states[robot_id2].position
                distance = np.linalg.norm(pos1 - pos2)
                
                if distance < MIN_INTER_ROBOT_DISTANCE:
                    violations.append(f"Robots {robot_id1} and {robot_id2} too close: {distance:.2f}m")
                elif distance > MAX_INTER_ROBOT_DISTANCE:
                    violations.append(f"Robots {robot_id1} and {robot_id2} too far: {distance:.2f}m")
        
        if violations:
            logger.warning(f"Swarm {swarm_state.swarm_id} constraint violations: {violations}")
    
    def create_joint_qubo(self, swarm_id: str,
                         individual_problems: Dict[str, OptimizationProblem],
                         shared_constraints: Dict[str, Any] = None) -> OptimizationProblem:
        """
        Create joint QUBO problem for entire swarm.
        
        Args:
            swarm_id: Swarm identifier
            individual_problems: Individual robot optimization problems
            shared_constraints: Additional swarm-wide constraints
            
        Returns:
            Joint OptimizationProblem for the swarm
            
        Raises:
            ValueError: If swarm not found or problems incompatible
        """
        if swarm_id not in self.swarms:
            raise ValueError(f"Swarm {swarm_id} not found")
        
        swarm_state = self.swarms[swarm_id]
        
        # Combine variable names and matrices
        all_variable_names = []
        all_variable_bounds = {}
        problem_offsets = {}
        
        current_offset = 0
        
        for robot_id, problem in individual_problems.items():
            if robot_id not in swarm_state.robot_states:
                logger.warning(f"Robot {robot_id} not in swarm {swarm_id}, skipping")
                continue
            
            # Store offset for this robot's variables
            problem_offsets[robot_id] = current_offset
            
            # Add variables
            for var_name in problem.variable_names:
                joint_var_name = f"{robot_id}_{var_name}"
                all_variable_names.append(joint_var_name)
                
                if problem.variable_bounds and var_name in problem.variable_bounds:
                    all_variable_bounds[joint_var_name] = problem.variable_bounds[var_name]
            
            current_offset += problem.num_variables
        
        # Create combined Q matrix
        total_vars = len(all_variable_names)
        Q_combined = np.zeros((total_vars, total_vars))
        
        # Add individual robot objectives
        for robot_id, problem in individual_problems.items():
            if robot_id not in problem_offsets:
                continue
            
            offset = problem_offsets[robot_id]
            n_vars = problem.num_variables
            
            # Copy robot's Q matrix to combined matrix
            Q_combined[offset:offset+n_vars, offset:offset+n_vars] += problem.objective_matrix
        
        # Add inter-robot constraints
        Q_combined = self._add_inter_robot_constraints(
            Q_combined, swarm_state, problem_offsets, individual_problems
        )
        
        # Add shared constraints if provided
        if shared_constraints:
            Q_combined = self._add_shared_constraints(
                Q_combined, swarm_state, problem_offsets, shared_constraints
            )
        
        # Create joint problem
        joint_problem_id = f"joint_{swarm_id}_{time.time()}"
        
        joint_problem = OptimizationProblem(
            problem_id=joint_problem_id,
            objective_matrix=Q_combined,
            constraint_matrices=[],  # All constraints in QUBO
            variable_names=all_variable_names,
            variable_bounds=all_variable_bounds if all_variable_bounds else None,
            time_limit=0.5  # 500ms for joint optimization
        )
        
        self.coordination_stats['joint_optimizations'] += 1
        
        logger.info(f"Created joint QUBO for swarm {swarm_id} with {total_vars} variables")
        return joint_problem
    
    def _add_inter_robot_constraints(self, Q: np.ndarray, swarm_state: SwarmState,
                                    problem_offsets: Dict[str, int],
                                    individual_problems: Dict[str, OptimizationProblem]) -> np.ndarray:
        """Add constraints between robots."""
        penalty = self.joint_qubo_penalty_weight
        
        robot_ids = list(swarm_state.robot_states.keys())
        
        # Add collision avoidance constraints
        for i, robot_id1 in enumerate(robot_ids):
            for robot_id2 in robot_ids[i+1:]:
                if robot_id1 not in problem_offsets or robot_id2 not in problem_offsets:
                    continue
                
                offset1 = problem_offsets[robot_id1]
                offset2 = problem_offsets[robot_id2]
                
                # Find action selection variables
                # (This is simplified - in practice would need to map actions to positions)
                
                # For collision avoidance: penalize both robots moving to nearby positions
                # Simplified: add penalty if both robots select "move" actions
                problem1 = individual_problems[robot_id1]
                problem2 = individual_problems[robot_id2]
                
                # Find move action variables (simplified assumption)
                move_vars1 = [j for j, name in enumerate(problem1.variable_names) 
                            if 'move' in name.lower()]
                move_vars2 = [j for j, name in enumerate(problem2.variable_names) 
                            if 'move' in name.lower()]
                
                for var1 in move_vars1:
                    for var2 in move_vars2:
                        idx1 = offset1 + var1
                        idx2 = offset2 + var2
                        
                        # Add penalty for both moving
                        Q[idx1, idx2] += penalty * 0.1
                        Q[idx2, idx1] += penalty * 0.1
        
        # Add formation maintenance constraints
        if swarm_state.formation:
            Q = self._add_formation_constraints(Q, swarm_state, problem_offsets, penalty)
        
        return Q
    
    def _add_formation_constraints(self, Q: np.ndarray, swarm_state: SwarmState,
                                  problem_offsets: Dict[str, int],
                                  penalty: float) -> np.ndarray:
        """Add formation maintenance constraints."""
        formation = swarm_state.formation
        
        if formation['type'] == 'line':
            # For line formation: maintain relative positions
            robot_ids = list(swarm_state.robot_states.keys())
            
            for i in range(len(robot_ids) - 1):
                robot_id1 = robot_ids[i]
                robot_id2 = robot_ids[i + 1]
                
                if robot_id1 not in problem_offsets or robot_id2 not in problem_offsets:
                    continue
                
                # Simplified: penalize deviation from expected spacing
                # In practice, would use position variables
                pass
        
        return Q
    
    def _add_shared_constraints(self, Q: np.ndarray, swarm_state: SwarmState,
                               problem_offsets: Dict[str, int],
                               shared_constraints: Dict[str, Any]) -> np.ndarray:
        """Add shared resource constraints."""
        penalty = self.joint_qubo_penalty_weight
        
        # Example: Shared power source constraint
        if 'max_total_power' in shared_constraints:
            max_power = shared_constraints['max_total_power']
            
            # Find power variables for each robot
            # (Simplified - would need to map actions to power consumption)
            
            # For each robot, find variables representing high-power actions
            for robot_id, offset in problem_offsets.items():
                # Simplified: assume first variable is correlated with power
                # In practice, would need proper mapping
                power_var_idx = offset  # First variable
                
                # Penalize high power usage
                Q[power_var_idx, power_var_idx] += penalty * 0.01
        
        return Q
    
    def decompose_joint_solution(self, joint_solution: np.ndarray,
                                swarm_id: str,
                                individual_problems: Dict[str, OptimizationProblem],
                                problem_offsets: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Decompose joint solution into individual robot solutions.
        
        Args:
            joint_solution: Solution to joint QUBO problem
            swarm_id: Swarm identifier
            individual_problems: Original individual problems
            problem_offsets: Variable offsets for each robot
            
        Returns:
            Dictionary mapping robot_id to individual solution
        """
        if swarm_id not in self.swarms:
            raise ValueError(f"Swarm {swarm_id} not found")
        
        individual_solutions = {}
        
        for robot_id, problem in individual_problems.items():
            if robot_id not in problem_offsets:
                continue
            
            offset = problem_offsets[robot_id]
            n_vars = problem.num_variables
            
            # Extract this robot's portion of the solution
            robot_solution = joint_solution[offset:offset+n_vars]
            individual_solutions[robot_id] = robot_solution
        
        return individual_solutions
    
    def coordinate_actions(self, swarm_id: str,
                          robot_actions: Dict[str, List[Action]]) -> Dict[str, List[Action]]:
        """
        Coordinate actions between robots to avoid conflicts.
        
        Args:
            swarm_id: Swarm identifier
            robot_actions: Proposed actions for each robot
            
        Returns:
            Coordinated actions with conflicts resolved
        """
        if swarm_id not in self.swarms:
            logger.warning(f"Swarm {swarm_id} not found")
            return robot_actions
        
        swarm_state = self.swarms[swarm_id]
        coordinated_actions = robot_actions.copy()
        
        # Check for spatial conflicts
        spatial_conflicts = self._detect_spatial_conflicts(swarm_state, robot_actions)
        
        for conflict in spatial_conflicts:
            robot_id1, robot_id2, conflict_type = conflict
            
            # Resolve conflict based on roles
            role1 = swarm_state.robot_roles[robot_id1].role
            role2 = swarm_state.robot_roles[robot_id2].role
            
            # Leader has priority over followers
            if role1 == 'leader' and role2 != 'leader':
                # Robot 2 yields
                coordinated_actions[robot_id2] = self._modify_actions_to_avoid_conflict(
                    coordinated_actions[robot_id2], conflict_type
                )
                self.coordination_stats['collisions_prevented'] += 1
            elif role2 == 'leader' and role1 != 'leader':
                # Robot 1 yields
                coordinated_actions[robot_id1] = self._modify_actions_to_avoid_conflict(
                    coordinated_actions[robot_id1], conflict_type
                )
                self.coordination_stats['collisions_prevented'] += 1
            else:
                # Both equal priority - both modify
                coordinated_actions[robot_id1] = self._modify_actions_to_avoid_conflict(
                    coordinated_actions[robot_id1], conflict_type
                )
                coordinated_actions[robot_id2] = self._modify_actions_to_avoid_conflict(
                    coordinated_actions[robot_id2], conflict_type
                )
                self.coordination_stats['collisions_prevented'] += 2
        
        # Check for resource conflicts
        resource_conflicts = self._detect_resource_conflicts(swarm_state, robot_actions)
        
        for conflict in resource_conflicts:
            # Simple resolution: sequentialize access
            self._resolve_resource_conflict(swarm_state, coordinated_actions, conflict)
            self.coordination_stats['resource_conflicts_resolved'] += 1
        
        return coordinated_actions
    
    def _detect_spatial_conflicts(self, swarm_state: SwarmState,
                                 robot_actions: Dict[str, List[Action]]) -> List[Tuple]:
        """Detect potential spatial conflicts between robots."""
        conflicts = []
        
        robot_ids = list(swarm_state.robot_states.keys())
        
        for i, robot_id1 in enumerate(robot_ids):
            for robot_id2 in robot_ids[i+1:]:
                if robot_id1 not in robot_actions or robot_id2 not in robot_actions:
                    continue
                
                # Simplified conflict detection
                # In practice, would check predicted trajectories
                
                actions1 = robot_actions[robot_id1]
                actions2 = robot_actions[robot_id2]
                
                # Check if any actions involve movement
                has_movement1 = any('move' in a.action_id.lower() for a in actions1)
                has_movement2 = any('move' in a.action_id.lower() for a in actions2)
                
                if has_movement1 and has_movement2:
                    # Check current positions
                    pos1 = swarm_state.robot_states[robot_id1].position
                    pos2 = swarm_state.robot_states[robot_id2].position
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    if distance < MIN_INTER_ROBOT_DISTANCE * 2:
                        conflicts.append((robot_id1, robot_id2, 'proximity_conflict'))
        
        return conflicts
    
    def _modify_actions_to_avoid_conflict(self, actions: List[Action],
                                         conflict_type: str) -> List[Action]:
        """Modify actions to avoid conflicts."""
        if not actions:
            return actions
        
        modified_actions = []
        
        for action in actions:
            if conflict_type == 'proximity_conflict' and 'move' in action.action_id.lower():
                # Replace move with wait or reduced speed
                wait_action = Action(
                    action_id='wait_avoid_conflict',
                    action_type='safety',
                    parameters={'duration': 1.0},
                    duration=1.0,
                    priority=action.priority + 10,  # Higher priority to avoid conflict
                    max_torque=0.0,
                    max_velocity=0.0,
                    thermal_load=0.0,
                    power_required=10.0
                )
                modified_actions.append(wait_action)
            else:
                # Keep original action
                modified_actions.append(action)
        
        return modified_actions
    
    def _detect_resource_conflicts(self, swarm_state: SwarmState,
                                  robot_actions: Dict[str, List[Action]]) -> List[Dict]:
        """Detect resource conflicts between robots."""
        conflicts = []
        
        # Simplified: check for shared tool usage
        # In practice, would check actual resource requirements
        
        tool_usage = {}
        
        for robot_id, actions in robot_actions.items():
            for action in actions:
                if 'tool' in action.action_id.lower():
                    tool_name = action.action_id.split('_')[-1]
                    
                    if tool_name not in tool_usage:
                        tool_usage[tool_name] = []
                    tool_usage[tool_name].append(robot_id)
        
        # Detect conflicts (multiple robots using same tool)
        for tool_name, users in tool_usage.items():
            if len(users) > 1:
                conflicts.append({
                    'resource': tool_name,
                    'type': 'tool_conflict',
                    'users': users
                })
        
        return conflicts
    
    def _resolve_resource_conflict(self, swarm_state: SwarmState,
                                  robot_actions: Dict[str, List[Action]],
                                  conflict: Dict):
        """Resolve resource conflict by sequencing access."""
        resource = conflict['resource']
        users = conflict['users']
        
        # Sort users by priority (leaders first)
        users_sorted = sorted(users, key=lambda uid: 
                            0 if swarm_state.robot_roles[uid].role == 'leader' else 1)
        
        # Assign sequential access
        for i, robot_id in enumerate(users_sorted):
            # Find tool usage actions for this robot
            for action in robot_actions[robot_id]:
                if resource in action.action_id:
                    # Add delay based on position in sequence
                    delay_action = Action(
                        action_id=f"wait_for_{resource}_{i}",
                        action_type='safety',
                        parameters={'delay': i * 2.0},  # 2 second intervals
                        duration=i * 2.0,
                        priority=50,
                        max_torque=0.0,
                        max_velocity=0.0,
                        thermal_load=0.0,
                        power_required=10.0
                    )
                    
                    # Insert delay before tool usage
                    actions = robot_actions[robot_id]
                    idx = actions.index(action)
                    actions.insert(idx, delay_action)
    
    def get_swarm_info(self, swarm_id: str) -> Dict[str, Any]:
        """Get information about a swarm."""
        if swarm_id not in self.swarms:
            return {'error': f"Swarm {swarm_id} not found"}
        
        swarm_state = self.swarms[swarm_id]
        
        # Calculate swarm metrics
        positions = [state.position for state in swarm_state.robot_states.values()]
        velocities = [state.velocity for state in swarm_state.robot_states.values()]
        
        if positions:
            centroid = np.mean(positions, axis=0)
            spread = np.std(positions, axis=0)
            avg_speed = np.mean([np.linalg.norm(v) for v in velocities])
        else:
            centroid = np.zeros(3)
            spread = np.zeros(3)
            avg_speed = 0.0
        
        # Communication connectivity
        comm_graph = swarm_state.communication_graph
        if comm_graph:
            connectivity = sum(len(neighbors) for neighbors in comm_graph.values()) / (2 * len(comm_graph))
        else:
            connectivity = 0.0
        
        return {
            'swarm_id': swarm_id,
            'num_robots': swarm_state.num_robots,
            'formation_type': swarm_state.formation['type'] if swarm_state.formation else None,
            'centroid': centroid.tolist(),
            'spread': spread.tolist(),
            'average_speed': float(avg_speed),
            'communication_connectivity': float(connectivity),
            'roles': {rid: role.role for rid, role in swarm_state.robot_roles.items()},
            'constraints': [c.value for c in self.coordination_constraints.get(swarm_id, [])],
            'timestamp': swarm_state.timestamp
        }
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            **self.coordination_stats,
            'num_swarms': len(self.swarms),
            'coordination_mode': self.coordination_mode.value,
            'total_robots': sum(swarm.num_robots for swarm in self.swarms.values())
        }
    
    def remove_swarm(self, swarm_id: str):
        """Remove a swarm from coordination."""
        if swarm_id in self.swarms:
            del self.swarms[swarm_id]
            if swarm_id in self.coordination_constraints:
                del self.coordination_constraints[swarm_id]
            logger.info(f"Removed swarm {swarm_id}")
    
    def reset(self):
        """Reset coordinator to initial state."""
        self.swarms = {}
        self.coordination_constraints = {}
        self.coordination_stats = {
            'joint_optimizations': 0,
            'collisions_prevented': 0,
            'resource_conflicts_resolved': 0,
            'formation_adjustments': 0,
            'avg_coordination_time': 0.0
        }
        logger.info("Multi-robot coordinator reset")
        