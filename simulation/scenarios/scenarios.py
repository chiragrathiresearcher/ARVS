"""
scenarios.py
ARVS Simulation — Scenario Catalogue

Four scenarios, each parameterised and documented.
"""

from simulation.engine.simulation_engine import ScenarioConfig, ScenarioType


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normal Operations
# ─────────────────────────────────────────────────────────────────────────────

NORMAL_OPS = ScenarioConfig(
    scenario_type = ScenarioType.NORMAL_OPS,
    name          = "Normal Operations",
    description   = (
        "Full Martian sol of nominal rover driving. "
        "Temperature follows diurnal cycle, battery discharges then recharges "
        "with solar, all torques within limits. "
        "PASS criterion: ZERO gate blocks across all frames."
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Fault Injection
# ─────────────────────────────────────────────────────────────────────────────

FAULT_INJECTION = ScenarioConfig(
    scenario_type     = ScenarioType.FAULT_INJECTION,
    name              = "Fault Injection — Torque Spike + Sensor Dropout",
    description       = (
        "At frame 50: motor torque spikes to 180 N·m (1.8× the 100 N·m limit). "
        "At frames 80–90: IMU + temperature sensor dropout, confidence falls to 0.40. "
        "PASS criterion: safety gate MUST block the torque-spike action."
    ),
    fault_start_frame = 50,
    fault_torque_nm   = 180.0,
    dropout_frames    = list(range(80, 91)),
)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Communication Blackout
# ─────────────────────────────────────────────────────────────────────────────

COMM_BLACKOUT = ScenarioConfig(
    scenario_type         = ScenarioType.COMM_BLACKOUT,
    name                  = "Communication Blackout",
    description           = (
        "20,000-second blackout window starting at t=10,000s — representative "
        "of a Mars occultation pass (~40 min) scaled to the Martian sol "
        "telemetry frame spacing (one frame every ~444s). "
        "High-power non-safety actions (>20 W, duration >5s) must be deferred. "
        "PASS criterion: blackout window is correctly identified and logged."
    ),
    blackout_start_s      = 10_000.0,
    blackout_duration_s   = 20_000.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Axiom Violation Cascade
# ─────────────────────────────────────────────────────────────────────────────

AXIOM_CASCADE = ScenarioConfig(
    scenario_type             = ScenarioType.AXIOM_CASCADE,
    name                      = "Axiom Violation Cascade → Safe Hold",
    description               = (
        "Starting at frame 60, confidence decays exponentially "
        "(e^{-0.05·n}) until it falls below 0.30. "
        "At that point: E3 oscillation axiom fails, C2 confidence-harm "
        "axiom fails, Z closure fails, system transitions to SAFE_HOLD. "
        "PASS criterion: SAFE_HOLD must be triggered before confidence hits floor."
    ),
    confidence_collapse_frame = 60,
    confidence_floor          = 0.05,
)

ALL_SCENARIOS = [NORMAL_OPS, FAULT_INJECTION, COMM_BLACKOUT, AXIOM_CASCADE]
