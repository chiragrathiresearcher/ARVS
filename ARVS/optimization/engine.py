

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== CORE DATA TYPES =====

@dataclass
class RobotState:
    """Robot state for optimization."""
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
    """Risk assessment for optimization."""
    timestamp: float
    overall_risk: float
    component_risks: Dict[str, float]
    risk_factors: Dict[str, float]
    confidence: float = 1.0

@dataclass
class Action:
    """Action for optimization."""
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
class SafetyConstraints:
    """Safety constraints for optimization."""
    max_torque: Dict[str, float]
    max_velocity: Dict[str, float]
    thermal_limits: Dict[str, float]
    structural_load_limits: Dict[str, float]
    min_battery: float

# ===== OPTIMIZATION TYPES =====

class SolverType(Enum):
    """Types of optimization solvers."""
    CLASSICAL = "classical"
    QUANTUM_ANNEALER = "quantum_annealer"
    QAOA = "qaoa"
    HYBRID = "hybrid"
    TENSOR_NETWORK = "tensor_network"

@dataclass
class OptimizationProblem:
    """Optimization problem definition."""
    problem_id: str
    objective_matrix: np.ndarray
    constraint_matrices: List[np.ndarray] = field(default_factory=list)
    variable_names: List[str] = field(default_factory=list)
    variable_bounds: Optional[Dict[str, Tuple[float, float]]] = None
    time_limit: float = 0.5
    
    @property
    def num_variables(self) -> int:
        return len(self.variable_names)

@dataclass
class OptimizationResult:
    """Result of optimization process."""
    success: bool
    solution: Optional[np.ndarray]
    objective_value: float
    solver_time: float
    solver_type: SolverType
    metadata: Dict[str, Any]
    
    def validate(self, problem: OptimizationProblem) -> bool:
        """Validate solution against problem."""
        if not self.success or self.solution is None:
            return False
        
        if len(self.solution) != problem.num_variables:
            logger.error(f"Solution dimension mismatch")
            return False
        
        if problem.variable_bounds:
            for var_name, (lower, upper) in problem.variable_bounds.items():
                if var_name in problem.variable_names:
                    idx = problem.variable_names.index(var_name)
                    value = self.solution[idx]
                    if value < lower - 1e-6 or value > upper + 1e-6:
                        logger.warning(f"Variable {var_name} out of bounds")
                        return False
        
        return True

# ===== EXCEPTIONS =====

class OptimizationTimeoutException(Exception):
    """Raised when optimization exceeds time limit."""
    def __init__(self, time_limit: float, actual_time: float):
        super().__init__(f"Optimization timeout: {actual_time:.3f}s > {time_limit:.3f}s")

class NoFeasibleSolutionException(Exception):
    """Raised when no feasible solution can be found."""
    def __init__(self, problem_description: str):
        super().__init__(f"No feasible solution for: {problem_description}")

class QuantumSolverUnavailableException(Exception):
    """Raised when quantum solver is unavailable."""
    def __init__(self, solver_type: SolverType, reason: str):
        super().__init__(f"Quantum solver {solver_type.value} unavailable: {reason}")

# ===== CONSTANTS =====

MAX_REPLANNING_TIME = 0.5  # 500ms
MAX_QUBO_VARIABLES = 20000
MIN_QUBO_VARIABLES = 500
DEFAULT_QUBO_PENALTY_WEIGHT = 100.0

# ===== SOLVER IMPLEMENTATIONS =====

class ClassicalSolver:
    """Classical optimization solver."""
    
    def __init__(self):
        self.name = "ClassicalSolver"
        self.is_available_flag = True
    
    def is_available(self) -> bool:
        return self.is_available_flag
    
    def solve(self, Q: np.ndarray, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """Solve QUBO using simulated annealing."""
        start_time = time.time()
        n = Q.shape[0]
        
        # Simulated annealing
        best_solution = None
        best_energy = float('inf')
        
        # Multiple random restarts
        for restart in range(10):
            if time.time() - start_time > timeout:
                break
            
            # Start with random solution
            current_solution = np.random.choice([0, 1], size=n)
            current_energy = self._calculate_energy(current_solution, Q)
            
            # Annealing parameters
            temperature = 10.0
            cooling_rate = 0.95
            
            for iteration in range(100):
                if time.time() - start_time > timeout:
                    break
                
                # Generate neighbor
                neighbor = current_solution.copy()
                flip_index = random.randint(0, n-1)
                neighbor[flip_index] = 1 - neighbor[flip_index]
                
                neighbor_energy = self._calculate_energy(neighbor, Q)
                
                # Accept if better or with probability
                if neighbor_energy < current_energy or random.random() < np.exp((current_energy - neighbor_energy) / temperature):
                    current_solution = neighbor
                    current_energy = neighbor_energy
                
                # Cool down
                temperature *= cooling_rate
                
                # Update best
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
        
        if best_solution is not None:
            return best_solution, best_energy
        else:
            # Fallback: random solution
            solution = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
            energy = self._calculate_energy(solution, Q)
            return solution, energy
    
    def _calculate_energy(self, x: np.ndarray, Q: np.ndarray) -> float:
        """Calculate energy for QUBO: x^T Q x."""
        return float(x.T @ Q @ x)

class QuantumAnnealerInterface:
    """
    Placeholder for D-Wave Leap quantum annealer.

    Currently unavailable — real hardware integration requires
    dwave-ocean-sdk and a valid D-Wave API token.  When the SDK is
    installed and DWAVE_API_TOKEN is set in the environment, this class
    will delegate to the real sampler.  Until then it raises
    QuantumSolverUnavailableException so the engine falls back to
    ClassicalSolver deterministically.
    """

    def __init__(self):
        self.name = "QuantumAnnealer_DWave"
        self._sdk_available = self._check_sdk()

    def _check_sdk(self) -> bool:
        """Return True only if dwave-ocean-sdk is installed and token exists."""
        try:
            import importlib
            importlib.import_module("dwave.system")
            import os
            return bool(os.environ.get("DWAVE_API_TOKEN", "").strip())
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._sdk_available

    def solve(self, Q: np.ndarray, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        """Delegate to D-Wave sampler when SDK is present; raise otherwise."""
        if not self._sdk_available:
            raise QuantumSolverUnavailableException(
                SolverType.QUANTUM_ANNEALER,
                "dwave-ocean-sdk not installed or DWAVE_API_TOKEN not set"
            )
        # Real D-Wave path (reached only when SDK + token are present)
        from dwave.system import DWaveSampler, EmbeddingComposite
        import dimod
        n = Q.shape[0]
        bqm = dimod.BinaryQuadraticModel.from_qubo(
            {(i, j): float(Q[i, j]) for i in range(n) for j in range(n) if Q[i, j] != 0}
        )
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=100, time_limit=int(timeout))
        best = response.first.sample
        solution = np.array([best[i] for i in range(n)], dtype=float)
        energy = float(solution.T @ Q @ solution)
        return solution, energy


class QAOASolverInterface:
    """
    Placeholder for gate-based QAOA solver.

    Unavailable until a quantum circuit framework (Qiskit / Cirq) and
    hardware backend are configured.  Raises deterministically so the
    engine falls back to ClassicalSolver.
    """

    def __init__(self):
        self.name = "QAOA_placeholder"
        self._sdk_available = self._check_sdk()

    def _check_sdk(self) -> bool:
        try:
            import importlib
            importlib.import_module("qiskit")
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._sdk_available

    def solve(self, Q: np.ndarray, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        if not self._sdk_available:
            raise QuantumSolverUnavailableException(
                SolverType.QAOA,
                "Qiskit not installed — QAOA solver unavailable"
            )
        # Real QAOA path reserved for future implementation
        raise QuantumSolverUnavailableException(
            SolverType.QAOA, "QAOA backend not yet configured"
        )


class HybridSolverInterface:
    """
    Hybrid solver: tries D-Wave if available, falls back to ClassicalSolver.
    Always available because the classical fallback is always present.
    """

    def __init__(self):
        self.name = "HybridSolver"
        self._quantum = QuantumAnnealerInterface()
        self._classical = ClassicalSolver()

    def is_available(self) -> bool:
        return True  # Classical fallback guarantees availability

    def solve(self, Q: np.ndarray, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        start_time = time.time()
        if self._quantum.is_available():
            try:
                solution, energy = self._quantum.solve(Q, timeout / 2)
                if solution is not None:
                    return solution, energy
            except Exception:
                pass  # Fall through to classical
        remaining = timeout - (time.time() - start_time)
        return self._classical.solve(Q, max(0.05, remaining))


class TensorNetworkSolver:
    """
    Placeholder for tensor-network contraction solver.
    Unavailable until cotengra / quimb is installed.
    """

    def __init__(self):
        self.name = "TensorNetwork_placeholder"
        self._sdk_available = self._check_sdk()

    def _check_sdk(self) -> bool:
        try:
            import importlib
            importlib.import_module("cotengra")
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        return self._sdk_available

    def solve(self, Q: np.ndarray, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float]:
        if not self._sdk_available:
            raise QuantumSolverUnavailableException(
                SolverType.TENSOR_NETWORK,
                "cotengra not installed — tensor network solver unavailable"
            )
        raise QuantumSolverUnavailableException(
            SolverType.TENSOR_NETWORK, "Tensor network backend not yet configured"
        )

# ===== MAIN OPTIMIZATION ENGINE =====

class OptimizationEngine:
    """
    Main optimization engine for ARVS decision-making.
    
    From ARVS document:
    - Must solve ≤ 20,000 variables in < 500ms
    - Supports quantum, classical, and hybrid solvers
    - Includes fallback to classical when quantum unavailable
    """
    
    def __init__(self, solver_preference: List[SolverType] = None):
        """Initialize optimization engine."""
        if solver_preference is None:
            solver_preference = [
                SolverType.CLASSICAL,  # Start with classical for reliability
                SolverType.HYBRID,
                SolverType.QUANTUM_ANNEALER,
                SolverType.QAOA,
                SolverType.TENSOR_NETWORK
            ]
        
        self.solver_preference = solver_preference
        self.active_solver: Optional[SolverType] = None
        self.solver_status: Dict[SolverType, bool] = {}
        
        # Initialize solvers
        self._initialize_solvers()
        
        # Problem formulation
        self.penalty_weight = DEFAULT_QUBO_PENALTY_WEIGHT
        
        # Performance tracking
        self.solve_times: List[float] = []
        self.success_rate: float = 1.0
        
        logger.info(f"Optimization engine initialized with preference: {[s.value for s in solver_preference]}")
    
    def _initialize_solvers(self):
        """Initialize all available solvers."""
        self.solvers = {
            SolverType.CLASSICAL: ClassicalSolver(),
            SolverType.QUANTUM_ANNEALER: QuantumAnnealerInterface(),
            SolverType.QAOA: QAOASolverInterface(),
            SolverType.HYBRID: HybridSolverInterface(),
            SolverType.TENSOR_NETWORK: TensorNetworkSolver()
        }
        
        # Check availability
        for solver_type, solver in self.solvers.items():
            self.solver_status[solver_type] = solver.is_available()
        
        # Set active solver
        for solver_type in self.solver_preference:
            if self.solver_status.get(solver_type, False):
                self.active_solver = solver_type
                logger.info(f"Active solver set to: {solver_type.value}")
                break
        
        if self.active_solver is None:
            logger.warning("No preferred solvers available, using classical")
            self.active_solver = SolverType.CLASSICAL
    
    def formulate_problem(self, robot_state: RobotState,
                         risk_assessment: RiskAssessment,
                         safety_constraints: SafetyConstraints,
                         available_actions: List[Action]) -> OptimizationProblem:
        """
        Formulate robotic decision problem as optimization problem.
        
        Converts mission constraints and robot state into QUBO formulation
        as specified in ARVS document Section 5.3.
        """
        # Create binary variables
        variable_names = []
        variable_bounds = {}
        
        # Action selection variables
        for i, action in enumerate(available_actions):
            var_name = f"action_{i}_{action.action_id}"
            variable_names.append(var_name)
            variable_bounds[var_name] = (0.0, 1.0)  # Binary
        
        # Constraint satisfaction variables
        constraint_vars = [
            "torque_constraint",
            "thermal_constraint", 
            "structural_constraint",
            "power_constraint",
            "collision_constraint"
        ]
        
        for constraint in constraint_vars:
            variable_names.append(constraint)
            variable_bounds[constraint] = (0.0, 1.0)
        
        # Create QUBO matrix
        n_vars = len(variable_names)
        Q = np.zeros((n_vars, n_vars))
        
        # Objective: minimize risk while achieving mission objectives
        Q = self._add_objective_terms(Q, variable_names, risk_assessment, available_actions)
        
        # Constraints: hard constraints as penalty terms
        Q = self._add_constraint_terms(Q, variable_names, safety_constraints, available_actions)
        
        # Action sequencing constraints
        Q = self._add_sequencing_terms(Q, variable_names, available_actions)
        
        # Risk-aware weighting
        Q = self._apply_risk_weighting(Q, risk_assessment)
        
        # Create optimization problem
        problem_id = f"prob_{robot_state.timestamp}_{robot_state.robot_id}"
        
        return OptimizationProblem(
            problem_id=problem_id,
            objective_matrix=Q,
            variable_names=variable_names,
            variable_bounds=variable_bounds,
            time_limit=MAX_REPLANNING_TIME
        )
    
    def _add_objective_terms(self, Q: np.ndarray, variable_names: List[str],
                            risk_assessment: RiskAssessment,
                            available_actions: List[Action]) -> np.ndarray:
        """Add objective function terms to QUBO matrix."""
        for i, action in enumerate(available_actions):
            var_idx = variable_names.index(f"action_{i}_{action.action_id}")
            
            # Base cost from action properties
            base_cost = 0.0
            
            if action.power_required:
                base_cost += action.power_required / 100.0
            
            if action.thermal_load:
                base_cost += action.thermal_load / 50.0
            
            # Risk-aware cost adjustment
            risk_factor = risk_assessment.overall_risk
            if risk_factor > 0.7:
                base_cost *= (1.0 + risk_factor)
            
            Q[var_idx, var_idx] += base_cost
        
        return Q
    
    def _add_constraint_terms(self, Q: np.ndarray, variable_names: List[str],
                             safety_constraints: SafetyConstraints,
                             available_actions: List[Action]) -> np.ndarray:
        """Add constraint terms to QUBO matrix as penalties."""
        penalty = self.penalty_weight
        
        # Get constraint indices
        torque_idx = variable_names.index("torque_constraint")
        thermal_idx = variable_names.index("thermal_constraint")
        structural_idx = variable_names.index("structural_constraint")
        
        for i, action in enumerate(available_actions):
            action_idx = variable_names.index(f"action_{i}_{action.action_id}")
            
            # Torque constraints
            if action.max_torque:
                violates_torque = False
                for joint, max_torque in safety_constraints.max_torque.items():
                    if action.max_torque > max_torque:
                        violates_torque = True
                        break
                
                if violates_torque:
                    Q[action_idx, torque_idx] += penalty
                    Q[torque_idx, action_idx] += penalty
            
            # Thermal constraints
            if action.thermal_load:
                violates_thermal = False
                for component, limit in safety_constraints.thermal_limits.items():
                    if action.thermal_load > limit:
                        violates_thermal = True
                        break
                
                if violates_thermal:
                    Q[action_idx, thermal_idx] += penalty
                    Q[thermal_idx, action_idx] += penalty
            
            # Structural constraints
            structural_stress = 0.0
            if action.max_velocity and action.max_velocity > 1.0:
                structural_stress += (action.max_velocity - 1.0) / 2.0
            
            if action.duration > 10.0:
                structural_stress += action.duration / 60.0
            
            if structural_stress > 0.5:
                Q[action_idx, structural_idx] += penalty * structural_stress
                Q[structural_idx, action_idx] += penalty * structural_stress
        
        # Encourage constraint satisfaction
        for constraint in ["torque_constraint", "thermal_constraint", 
                          "structural_constraint", "power_constraint", 
                          "collision_constraint"]:
            if constraint in variable_names:
                idx = variable_names.index(constraint)
                Q[idx, idx] -= penalty  # Penalize violation (value=0)
        
        return Q
    
    def _add_sequencing_terms(self, Q: np.ndarray, variable_names: List[str],
                             available_actions: List[Action]) -> np.ndarray:
        """Add action sequencing constraints."""
        action_indices = []
        
        for i, action in enumerate(available_actions):
            var_name = f"action_{i}_{action.action_id}"
            if var_name in variable_names:
                action_indices.append(variable_names.index(var_name))
        
        # Penalize selecting many actions (encourage minimal intervention)
        for idx1 in action_indices:
            for idx2 in action_indices:
                if idx1 != idx2:
                    Q[idx1, idx2] += 0.1 * self.penalty_weight
        
        return Q
    
    def _apply_risk_weighting(self, Q: np.ndarray, 
                             risk_assessment: RiskAssessment) -> np.ndarray:
        """Adjust QUBO based on current risk level."""
        risk_factor = risk_assessment.overall_risk
        
        if risk_factor > 0.7:
            scaling = 1.0 + 2.0 * (risk_factor - 0.7) / 0.3
            Q = Q * scaling
        
        return Q
    
    def solve(self, problem: OptimizationProblem, 
              timeout: Optional[float] = None) -> OptimizationResult:
        """
        Solve optimization problem using available solvers.
        """
        if timeout is None:
            timeout = problem.time_limit
        
        start_time = time.time()
        
        # Try preferred solvers in order
        for solver_type in self.solver_preference:
            if not self.solver_status.get(solver_type, False):
                continue
            
            if time.time() - start_time > timeout:
                raise OptimizationTimeoutException(timeout, time.time() - start_time)
            
            try:
                solver = self.solvers[solver_type]
                solution, objective_value = solver.solve(
                    problem.objective_matrix, 
                    timeout - (time.time() - start_time)
                )
                
                success = solution is not None
                solver_time = time.time() - start_time
                
                result = OptimizationResult(
                    success=success,
                    solution=solution,
                    objective_value=objective_value if success else float('inf'),
                    solver_time=solver_time,
                    solver_type=solver_type,
                    metadata={
                        'problem_id': problem.problem_id,
                        'num_variables': problem.num_variables,
                        'solver_name': solver.name
                    }
                )
                
                if success and result.validate(problem):
                    self.active_solver = solver_type
                    self._update_performance_metrics(solver_time, True)
                    return result
                else:
                    logger.warning(f"Solver {solver_type.value} failed")
                    
            except Exception as e:
                logger.error(f"Solver {solver_type.value} error: {e}")
                self.solver_status[solver_type] = False
        
        # All solvers failed
        logger.warning("All solvers failed, generating fallback solution")
        
        # Generate simple fallback solution
        n = problem.num_variables
        solution = np.zeros(n)
        
        # Select first action
        if n > 0:
            solution[0] = 1
        
        # Set constraints to satisfied
        for i in range(max(0, n-5), n):
            solution[i] = 1
        
        solver_time = time.time() - start_time
        
        result = OptimizationResult(
            success=True,
            solution=solution,
            objective_value=float(solution.T @ problem.objective_matrix @ solution),
            solver_time=solver_time,
            solver_type=SolverType.CLASSICAL,
            metadata={
                'problem_id': problem.problem_id,
                'num_variables': n,
                'fallback': True,
                'warning': 'All solvers failed, using fallback'
            }
        )
        
        self._update_performance_metrics(solver_time, True)
        return result
    
    def _update_performance_metrics(self, solve_time: float, success: bool):
        """Update performance tracking."""
        self.solve_times.append(solve_time)
        if len(self.solve_times) > 100:
            self.solve_times.pop(0)
        
        if success:
            self.success_rate = 0.95 * self.success_rate + 0.05
        else:
            self.success_rate = 0.95 * self.success_rate
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.solve_times:
            avg_time = 0.0
            max_time = 0.0
            min_time = 0.0
        else:
            avg_time = float(np.mean(self.solve_times))
            max_time = float(np.max(self.solve_times))
            min_time = float(np.min(self.solve_times))
        
        return {
            'avg_solve_time': avg_time,
            'max_solve_time': max_time,
            'min_solve_time': min_time,
            'success_rate': self.success_rate,
            'active_solver': self.active_solver.value if self.active_solver else None,
            'solver_status': {k.value: v for k, v in self.solver_status.items()},
            'total_solves': len(self.solve_times)
        }
    
    def switch_solver(self, solver_type: SolverType) -> bool:
        """Switch to a different solver type."""
        if self.solver_status.get(solver_type, False):
            self.active_solver = solver_type
            logger.info(f"Switched to solver: {solver_type.value}")
            return True
        else:
            logger.warning(f"Cannot switch to unavailable solver: {solver_type.value}")
            return False
    
    def reset(self):
        """Reset optimization engine."""
        self.solve_times = []
        self.success_rate = 1.0
        self._initialize_solvers()
        logger.info("Optimization engine reset")

# ===== DEMONSTRATION =====

def demonstrate_optimization_engine():
    """Demonstrate the optimization engine functionality."""
    print("=" * 70)
    print("OPTIMIZATION ENGINE DEMONSTRATION")
    print("Complete standalone implementation")
    print("=" * 70)
    
    # Create optimization engine
    engine = OptimizationEngine()
    
    # Create test data
    robot_state = RobotState(
        robot_id="test_robot",
        timestamp=time.time(),
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([0.5, 0.0, 0.0]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        temperature=320.0,
        battery_level=0.7,
        power_consumption=50.0
    )
    
    risk_assessment = RiskAssessment(
        timestamp=time.time(),
        overall_risk=0.4,
        component_risks={'thermal': 0.3, 'power': 0.2, 'structural': 0.1},
        risk_factors={'temperature': 320.0, 'battery': 0.7}
    )
    
    safety_constraints = SafetyConstraints(
        max_torque={'joint1': 100.0, 'joint2': 80.0},
        max_velocity={'joint1': 2.0, 'joint2': 1.5},
        thermal_limits={'motor': 373.0, 'cpu': 358.0},
        structural_load_limits={'arm': 200.0},
        min_battery=0.15
    )
    
    available_actions = [
        Action(
            action_id="move_slow",
            action_type="motion",
            parameters={'direction': 'forward', 'distance': 1.0},
            duration=2.0,
            max_torque=50.0,
            thermal_load=10.0,
            power_required=40.0
        ),
        Action(
            action_id="move_fast",
            action_type="motion",
            parameters={'direction': 'forward', 'distance': 3.0},
            duration=1.0,
            max_torque=90.0,
            thermal_load=30.0,
            power_required=100.0
        ),
        Action(
            action_id="halt",
            action_type="safety",
            parameters={'mode': 'immediate'},
            duration=0.1,
            max_torque=0.0,
            thermal_load=0.0,
            power_required=10.0
        )
    ]
    
    print("\n1. Formulating optimization problem...")
    problem = engine.formulate_problem(
        robot_state, risk_assessment, safety_constraints, available_actions
    )
    print(f"   Problem ID: {problem.problem_id}")
    print(f"   Variables: {problem.num_variables}")
    print(f"   QUBO matrix shape: {problem.objective_matrix.shape}")
    
    print("\n2. Solving optimization problem...")
    try:
        result = engine.solve(problem)
        
        print(f"   Success: {result.success}")
        print(f"   Solver: {result.solver_type.value}")
        print(f"   Solve time: {result.solver_time:.3f}s")
        print(f"   Objective value: {result.objective_value:.3f}")
        
        if result.solution is not None:
            print(f"   Solution shape: {result.solution.shape}")
            
            # Interpret solution
            print("\n3. Interpreting solution:")
            for i, var_name in enumerate(problem.variable_names):
                if i < len(available_actions):
                    action_name = available_actions[i].action_id
                    selected = "SELECTED" if result.solution[i] > 0.5 else "not selected"
                    print(f"   {var_name} ({action_name}): {selected}")
                else:
                    constraint_status = "SATISFIED" if result.solution[i] > 0.5 else "VIOLATED"
                    print(f"   {var_name}: {constraint_status}")
        
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n4. Performance statistics:")
    stats = engine.get_performance_stats()
    for key, value in stats.items():
        if key != 'solver_status':
            print(f"   {key}: {value}")
    
    print("\n5. Solver status:")
    for solver_type, available in engine.solver_status.items():
        status = "AVAILABLE" if available else "UNAVAILABLE"
        print(f"   {solver_type.value}: {status}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_optimization_engine()
    