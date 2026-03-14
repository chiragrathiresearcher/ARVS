# API Reference & Core Data Types

This document serves as the technical reference for the core data structures and interfaces within the ARVS framework. Consistency in data representation is mandatory to maintain the integrity of the **Axiom Constitution** and the **Safety Gate**.

---

## 1. Primary System Enums
Located in `core/data_types.py` and `core/constants.py`, these enums govern the state machine of the entire system.

### `SystemMode`
Defines the high-level operational state of the ARVS.
* `NORMAL (0)`: Full autonomous operation.
* `DEGRADED (1)`: Restricted performance due to minor faults or elevated risk.
* `SAFE_HOLD (2)`: Minimal power state; awaiting operator intervention.
* `EMERGENCY (3)`: Critical failure state; all actuators disabled.
* `FAULT_RECOVERY (4)`: Attempting automated subsystem reset.

### `FaultSeverity`
* `NONE`, `MINOR`, `MODERATE`, `MAJOR`, `CRITICAL`.

---

## 2. Core Data Structures

### `RobotState`
The comprehensive snapshot of a single agent at a specific timestamp.
| Field | Type | Description |
| :--- | :--- | :--- |
| `robot_id` | `str` | Unique identifier for the agent. |
| `timestamp` | `float` | MET (Mission Elapsed Time) from `timing.py`. |
| `position` | `np.ndarray` | 3D coordinates [x, y, z]. |
| `orientation` | `np.ndarray` | Quaternion [w, x, y, z]. |
| `confidence` | `float` | 0.0 to 1.0 probability of state accuracy. |

### `OptimizationProblem` (QUBO)
The mathematical structure sent to the `OptimizationEngine`.
* **`objective_matrix`**: The $Q$ matrix ($N \times N$) defining the costs and penalties.
* **`variable_names`**: Mapping of binary decision bits to physical actions.
* **`time_limit`**: Hard-cap for solver (default 500ms).



---

## 3. Utility Interfaces

### `RadiationHardenedMath` (`utils/math_utils.py`)
* `hardened_dot(a, b)`: Returns `(result, confidence)`. Performs triple-redundant dot product.
* `verify_checksum(data, expected)`: Validates memory integrity against SHA-256 hashes.

### `SpacecraftClock` (`utils/timing.py`)
* `get_mission_elapsed_time()`: Returns the stable, drift-compensated MET.
* `watchdog_pet()`: Resets the hardware safety timer.

---

## 4. Exception Handling
ARVS uses specific exceptions in `core/exceptions.py` to trigger automated safety protocols:

* **`SafetyViolationException`**: Raised by the `SafetyGate` when a command exceeds physical limits.
* **`AxiomViolationException`**: Raised when telemetry contradicts the Safety Constitution.
* **`QuantumSolverUnavailableException`**: Triggered if the D-Wave/Leap connection is lost, forcing a fallback to classical Tabu search.

---

## 5. Usage Example: Initializing the System

```python
from arvs.core.data_types import SystemMode
from arvs.core.arvs_system import ARVSSystem

# Initialize a new agent
arvs = ARVSSystem(robot_id="rover_v1")

# Start the high-frequency control loop
arvs.start()

# Check system health
info = arvs.get_system_info()
if info['system_mode'] == SystemMode.NORMAL:
    print("System is verified and operational.")
