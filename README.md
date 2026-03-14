# ARVS — Adaptive Robust Verification System

**Autonomous Risk-aware Vehicle System**  
Research-grade autonomy architecture for spacecraft, rovers, and safety-critical autonomous systems.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--1682--4369-green)](https://orcid.org/0009-0008-1682-4369)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          ARVS Pipeline                          │
│  Hardware → Sense → Think → Verify → Optimise → Enforce/Act    │
│   (HAL)   (Perc.) (Bayes/ (Frozen  (QUBO/     (C++ sub-ms     │
│           Fusion) 8D Risk) Axioms)   Tabu)      Safety Gate)   │
└─────────────────────────────────────────────────────────────────┘
```

Five-stage pipeline: **Sense → Think → Verify → Optimise → Enforce**

1. **Sense** — Hardware Abstraction Layer, pub/sub TelemetryBus, mock drivers (IMU, GPS, thermal, power)
2. **Think** — EKF/Bayesian state estimation, 8-D risk, MDP/POMDP formal decision model
3. **Verify** — Frozen Axiom Constitution (25 axioms, immutable, cryptographic audit chain)
4. **Optimise** — QUBO on D-Wave Leap / Tabu fallback, A* mission planner
5. **Enforce** — C++ safety gate (sub-ms), SCHED_FIFO watchdog, ROS2 nodes

---

## Repository Structure

```
ARVS-main/
├── ARVS/                          # Python core
│   ├── hardware/hal.py            # HAL + TelemetryBus (pub/sub)
│   ├── core/                      # Axioms, constants, data types
│   ├── state/                     # EKF estimation + belief model
│   ├── risk/quantification.py     # 8-dimensional risk quantifier
│   ├── safety/safety_gate.py      # Python safety gate
│   ├── planning/
│   │   ├── mdp.py                 # MDP/POMDP + reward function
│   │   └── mission_planner.py     # Long-horizon planner + A*
│   ├── learning/
│   │   ├── adaptive_models.py     # Adaptive learner (±15% bounded)
│   │   └── experience_db.py       # SQLite experience DB + policy update
│   ├── optimization/engine.py     # QUBO / D-Wave / Tabu
│   ├── decision/mvi_logic.py      # MVI strategy logic
│   ├── execution/controller.py    # Python execution controller
│   ├── audit/logger.py            # SHA-256 hash-chain audit log
│   ├── fault/detection.py         # Fault detection
│   └── coordination/multi_robot.py
│
├── cpp/                           # C++ real-time components
│   ├── include/                   # arvs_types, safety_gate, axiom_validator,
│   │                              # watchdog_timer, execution_controller, ros_convert
│   ├── src/                       # Implementations
│   ├── ros2/                      # ROS2 node wrappers (safety_gate, axiom_validator,
│   │                              # execution_controller)
│   ├── CMakeLists.txt
│   └── package.xml
│
├── simulation/
│   ├── run_simulation.py          # Entry point
│   ├── data_loaders/telemetry_loader.py  # NASA REMS, SPICE, ESA MEX, ISS OSDR
│   ├── engine/simulation_engine.py
│   ├── scenarios/scenarios.py     # 4 scenarios
│   ├── outputs/                   # Plots, forensics, CI report (pre-generated)
│   └── tests/
│       ├── test_arvs_full.py      # 69 tests
│       └── run_tests.py           # Standalone runner
│
└── Docs/                          # Architecture, API, axioms, reliability
```

---

## Quick Start

### Run simulation
```bash
cd ARVS-main/simulation
pip install numpy pandas matplotlib
python run_simulation.py
```
Runs 4 scenarios × 4 data sources = 16 runs. Outputs: 16 PNG plots, forensic JSON trails, CI report.

### Run tests
```bash
cd ARVS-main
python simulation/tests/run_tests.py
```

### Build C++ / ROS2
```bash
cd ARVS-main/cpp
colcon build   # requires ROS2 Humble or Iron
```

---

## Key Design Decisions

**No random values in the safety path** — all stubs replaced with real SDK detection, physics-based logic, and hardware telemetry.

**Frozen Axiom Constitution** — 25 axioms in 9 categories as immutable dataclasses. Closure axiom Z passes only if all others pass or system is in sanctioned degraded mode. Identical logic in Python and C++.

**Risk integrated into reward** — `R(s,a) = gain - λ·risk - μ·energy`. The MDP decision optimiser trades mission progress against safety automatically.

**±15% Bounded Learning** — parameters cannot change more than ±15% per cycle. `risk_penalty` can never decrease below 80% of baseline (Axiom L1 guard).

**Zero-heap C++ hot path** — fixed-size arrays, no `std::vector` in structs, `SCHED_FIFO` watchdog thread, latching emergency stop.

---

## Simulation Scenarios

| Scenario | PASS Criterion |
|---|---|
| Normal Operations | Zero gate blocks |
| Fault Injection | Gate must block 1.8× torque spike at frame 50 |
| Communication Blackout | 300s blackout window logged |
| Axiom Cascade → Safe Hold | SAFE_HOLD triggered when confidence < 0.30 |

---

## Telemetry Sources

| Source | Live URL | Fallback |
|---|---|---|
| NASA REMS | pds-atmospheres.nmsu.edu | Martian diurnal cycle (REMS published stats) |
| NASA SPICE | ssd.jpl.nasa.gov/horizons | Circular Mars orbit 400 km |
| ESA Mars Express | archives.esac.esa.int/psa | Elliptical orbit 300×10,000 km |
| ISS OSDR | osdr.nasa.gov | Circular LEO 408 km, 51.6° inc. |

---

## Citation

```bibtex
@software{rathi2025arvs,
  author  = {Rathi, Chirag},
  title   = {ARVS: Adaptive Robust Verification System},
  year    = {2025},
  url     = {https://github.com/chiragrathiresearcher/ARVS},
  license = {Apache-2.0},
  orcid   = {0009-0008-1682-4369}
}
```

**Chirag Rathi** — Independent Researcher  
chiragrathiresearcher@gmail.com | ORCID: 0009-0008-1682-4369

*Apache-2.0 License*
