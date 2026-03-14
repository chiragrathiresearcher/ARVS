# Risk Quantification & QUBO Optimization

This document details the mathematical framework used by ARVS to make optimal safety decisions under uncertainty. The system utilizes **Quadratic Unconstrained Binary Optimization (QUBO)** to select the "Minimal Viable Intervention" (MVI) that maximizes mission success while minimizing risk.

---

## 1. 8-Dimensional Risk Quantification
Before an optimization problem is formulated, the `RiskQuantifier` (see `risk/quantification.py`) evaluates the current `BeliefState` across eight distinct dimensions:

1.  **Thermal Risk**: Proximity to critical hardware temperature limits.
2.  **Structural Risk**: Mechanical stress and joint torque saturation.
3.  **Power Risk**: Battery depletion and consumption rates.
4.  **Proximity Risk**: Distance to obstacles or other swarm members.
5.  **Navigation Risk**: Divergence from the planned trajectory.
6.  **Communication Risk**: Signal latency and packet loss.
7.  **Epistemic Risk**: High levels of sensor uncertainty/noise.
8.  **Actuator Risk**: Performance degradation in motors or joints.



---

## 2. The QUBO Formulation
ARVS maps the current state and risks into a quadratic cost function. The goal is to find a binary vector $x$ (representing a set of possible actions) that minimizes the total energy $H$:

$$H(x) = \sum_{i} Q_{ii}x_i + \sum_{i<j} Q_{ij}x_i x_j$$

* **Linear Terms ($Q_{ii}$)**: Represent the individual cost or benefit of an action (e.g., "How much does stopping reduce thermal risk?").
* **Quadratic Terms ($Q_{ij}$)**: Represent interactions between actions (e.g., "We cannot 'Sprint' and 'Limp Home' at the same time").



---

## 3. Solver Architecture
The `OptimizationEngine` (see `optimization/engine.py`) is hardware-agnostic and supports multiple solver backends:

| Solver Type | Description | Use Case |
| :--- | :--- | :--- |
| **CLASSICAL** | Simulated Annealing / Tabu Search | Standard edge-computing scenarios. |
| **QUANTUM** | D-Wave Leap / Quantum Annealing | Complex, high-variable swarm coordination. |
| **HYBRID** | Combined Classical-Quantum | Real-time large-scale trajectory planning. |

---

## 4. Minimal Viable Intervention (MVI)
Once the solver finds the optimal binary string, the `MVILogic` (see `decision/mvi_logic.py`) translates those bits into an executable `MVISequence`. 

* **Conservative Profile**: Prioritizes stability; triggered when overall risk > 0.6.
* **Aggressive Profile**: Prioritizes mission speed; only active when risk < 0.3.
* **Intervention Selection**: The logic ensures that the chosen intervention is the *smallest possible deviation* from the original mission goal required to restore safety.



---

## 5. Multi-Robot Joint Optimization
In swarm scenarios, `multi_robot.py` expands the QUBO matrix to include cross-robot constraints. If Robot A and Robot B are on a collision course, a massive "Penalty Weight" is added to the $Q_{ij}$ term where $i$ is Robot A's path and $j$ is Robot B's path, forcing the solver to find a non-conflicting solution.
