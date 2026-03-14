# Swarm Intelligence & Multi-Robot Coordination

This document describes how ARVS extends individual safety verification to collective swarm behavior. By utilizing **Joint QUBO Formulations**, the system ensures that multiple agents can operate in close proximity without violating safety axioms or physical constraints.

---

## 1. Coordination Modes
The `MultiRobotCoordinator` (see `coordination/multi_robot.py`) supports four distinct modes of operation, depending on communication bandwidth and mission requirements:

* **CENTRALIZED**: A single master node solves a high-dimensional joint QUBO for the entire swarm. Best for high-precision formation flying.
* **DISTRIBUTED**: Robots solve local problems but exchange "Constraint Buffers" to ensure global compatibility.
* **HIERARCHICAL**: A designated 'Leader' robot sets the safety boundaries, which 'Follower' robots must treat as hard axioms.
* **DECENTRALIZED**: Fully autonomous agents use reactive collision avoidance based on the estimated states of neighbors.



---

## 2. The Joint QUBO Model
When robots operate as a swarm, the individual optimization problem is expanded into a **Joint Objective Function**. This is calculated in Section 5.8 of the ARVS Specification.

The total energy $H_{swarm}$ includes:
1.  **Individual Goals**: Each robot's mission objectives.
2.  **Collision Penalties**: Massive quadratic penalties ($Q_{ij}$) are added if the predicted trajectories of two robots $i$ and $j$ overlap.
3.  **Formation Constraints**: Penalties for drifting beyond the `MAX_INTER_ROBOT_DISTANCE` (50.0m).



---

## 3. Swarm Roles & Responsibilities
To optimize resource usage, robots are assigned specific roles defined in `MultiRobotState`:

| Role | Responsibility | Verification Priority |
| :--- | :--- | :--- |
| **LEADER** | Path Planning & Comms Relay | High Epistemic Certainty |
| **FOLLOWER** | Task Execution | Proximity & Collision Avoidance |
| **SCOUT** | Environmental Mapping | Adaptive Model Updates |
| **WORKER** | Resource Handling | Power & Thermal Management |

---

## 4. Distributed Fault Recovery
A key strength of the ARVS coordination layer is its resilience to individual unit failure. 

* **Heartbeat Monitoring**: If a robot stops sending telemetry, the coordinator marks it as `COMMUNICATION_FAILURE`.
* **Dynamic Re-tasking**: The Joint QUBO is immediately re-solved to route other robots around the disabled unit.
* **Safety Buffering**: Nearby robots automatically increase their `MIN_INTER_ROBOT_DISTANCE` (1.0m) around a failing unit to provide a "Safety Buffer" for recovery maneuvers.



---

## 5. Resource Sharing & Deadlock Avoidance
The `coordination/multi_robot.py` module manages shared assets (e.g., charging docks, narrow transit corridors).
* **Mutual Exclusion**: The QUBO logic ensures that only one robot can occupy a "Shared Resource" state in the binary solution vector.
* **Priority Arbitration**: If two robots require the same resource, the one with the higher **Risk Score** or lower **Battery Level** is granted priority by the optimizer.
