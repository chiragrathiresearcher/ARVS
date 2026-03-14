"""
run_simulation.py
ARVS Full Simulation Runner
============================
Runs all 4 scenarios × all 4 telemetry sources (REMS, SPICE/HORIZONS,
ESA Mars Express, ISS OSDR).  Each run with real data where network is
available, physics-consistent synthetic fallback otherwise.

Outputs (all written to simulation/outputs/):
  plots/        — PNG time-series diagnostic plots (6 panels each)
  forensics/    — per-scenario JSON event trails
  reports/      — ci_report.txt + ci_report.json  (CI-compatible)

Usage:
  python run_simulation.py                  # all scenarios × all sources
  python run_simulation.py --source REMS    # single source
  python run_simulation.py --scenario fault # single scenario
"""

import sys
import os
import logging
import argparse
import time
import json

# ── Make ARVS-main root importable ───────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SIM_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ARVS.sim")

from data_loaders.telemetry_loader import (
    load_rems, load_spice_horizons, load_esa_housekeeping,
    load_iss_telemetry, frames_to_dataframe
)
from engine.simulation_engine import SimulationEngine, SimulationResult
from scenarios.scenarios import ALL_SCENARIOS
from outputs.report import (
    plot_scenario, write_forensic_json, write_ci_report, validate_audit_log
)

# ─────────────────────────────────────────────────────────────────────────────
# Output directory setup
# ─────────────────────────────────────────────────────────────────────────────

PLOTS_DIR     = os.path.join(SIM_ROOT, "outputs", "plots")
FORENSICS_DIR = os.path.join(SIM_ROOT, "outputs", "forensics")
REPORTS_DIR   = os.path.join(SIM_ROOT, "outputs", "reports")

for d in [PLOTS_DIR, FORENSICS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Telemetry source registry
# ─────────────────────────────────────────────────────────────────────────────

TELEMETRY_SOURCES = {
    "REMS":       lambda: load_rems(200),
    "SPICE":      lambda: load_spice_horizons(100),
    "ESA_MEX":    lambda: load_esa_housekeeping(200),
    "ISS":        lambda: load_iss_telemetry(200),
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(source_filter=None, scenario_filter=None):
    engine   = SimulationEngine()
    all_results: list[SimulationResult] = []

    sources   = {k: v for k, v in TELEMETRY_SOURCES.items()
                 if source_filter is None or k == source_filter.upper()}
    scenarios = [s for s in ALL_SCENARIOS
                 if scenario_filter is None or
                 scenario_filter.lower() in s.name.lower()]

    logger.info("=" * 72)
    logger.info("ARVS SIMULATION START")
    logger.info(f"  Sources:   {list(sources.keys())}")
    logger.info(f"  Scenarios: {[s.name for s in scenarios]}")
    logger.info("=" * 72)

    total_start = time.time()

    for source_name, loader in sources.items():
        logger.info(f"\n── Loading telemetry: {source_name} ──────────────")
        t0 = time.time()
        frames = loader()
        logger.info(f"   {len(frames)} frames loaded in {time.time()-t0:.2f}s  "
                    f"(source label: {frames[0].source if frames else 'N/A'})")

        # Export raw telemetry as CSV for inspection
        df = frames_to_dataframe(frames)
        csv_path = os.path.join(REPORTS_DIR, f"telemetry_{source_name}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"   Raw telemetry → {csv_path}")

        for scenario in scenarios:
            logger.info(f"\n   ▶ Scenario: {scenario.name}")
            t1 = time.time()

            result = engine.run(frames, scenario, robot_id=f"SIM_{source_name}")

            elapsed = time.time() - t1
            status  = "PASS ✓" if result.pass_fail else "FAIL ✗"
            logger.info(f"     {status}  |  {result.total_frames} frames  |  "
                        f"{result.summary['gate_blocks']} blocks  |  "
                        f"risk avg={result.summary['avg_risk_score']:.3f}  |  "
                        f"{elapsed:.2f}s")

            # ── Outputs ────────────────────────────────────────────────────

            # 1. Time-series plot
            plot_path = plot_scenario(result, PLOTS_DIR)

            # 2. Forensic JSON
            forensic_path = write_forensic_json(result, FORENSICS_DIR)

            # 3. Audit log validation
            audit = validate_audit_log(result)
            audit_path = os.path.join(
                FORENSICS_DIR,
                f"audit_{_slug(result.scenario)}_{source_name}.json")
            with open(audit_path, "w") as f:
                json.dump(audit, f, indent=2)

            if audit["missed_blocks"] > 0:
                logger.warning(
                    f"     AUDIT WARNING: {audit['missed_blocks']} missed blocks "
                    f"at frames {audit['missed_at_frames']}")

            all_results.append(result)

    # ── Master CI report ──────────────────────────────────────────────────────
    ci_path = write_ci_report(all_results, REPORTS_DIR)

    total_elapsed = time.time() - total_start
    overall_pass  = all(r.pass_fail for r in all_results)

    logger.info("\n" + "=" * 72)
    logger.info("ARVS SIMULATION COMPLETE")
    logger.info(f"  Total time : {total_elapsed:.1f}s")
    logger.info(f"  Scenarios  : {len(all_results)}")
    logger.info(f"  Overall    : {'ALL PASS ✓' if overall_pass else 'FAILURES DETECTED ✗'}")
    logger.info(f"  CI report  : {ci_path}")
    logger.info("=" * 72)

    return 0 if overall_pass else 1


def _slug(s):
    return s.lower().replace(" ", "_").replace("—", "").replace("/", "_") \
            .replace("→", "").replace("+", "")[:50]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARVS Simulation Runner")
    parser.add_argument("--source",   default=None,
                        help="Single source: REMS | SPICE | ESA_MEX | ISS")
    parser.add_argument("--scenario", default=None,
                        help="Filter scenario by keyword (e.g. 'fault', 'normal')")
    args = parser.parse_args()
    sys.exit(run(args.source, args.scenario))
