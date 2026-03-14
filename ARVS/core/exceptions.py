"""
ARVS System Exceptions - Simplified Version
No type hints to avoid import issues.
"""

class ARVSException(Exception):
    """Base exception for all ARVS-related errors."""
    pass

class SafetyViolationException(ARVSException):
    """Raised when a safety constraint would be violated."""
    def __init__(self, constraint, value, limit):
        super().__init__(f"Safety violation: {constraint} = {value} exceeds limit {limit}")
        self.constraint = constraint
        self.value = value
        self.limit = limit

class OptimizationTimeoutException(ARVSException):
    """Raised when optimization exceeds time limit."""
    def __init__(self, time_limit, actual_time):
        super().__init__(f"Optimization timeout: {actual_time:.3f}s > {time_limit:.3f}s limit")
        self.time_limit = time_limit
        self.actual_time = actual_time

# ... [keep all other exception classes the same but remove type hints]

class QuantumSolverUnavailableException(ARVSException):
    """Raised when quantum solver is unavailable."""
    def __init__(self, solver_type, reason):
        super().__init__(f"Quantum solver {solver_type} unavailable: {reason}")
        self.solver_type = solver_type
        self.reason = reason 
        