"""
report.py  —  ARVS Simulation Output Layer
Plots + Forensic JSON + CI Report + Audit Validation
"""
import os, json, logging
from typing import List, Dict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from engine.simulation_engine import SimulationResult

logger = logging.getLogger(__name__)
_DEFAULT = os.path.join(os.path.dirname(__file__))

STYLE = {
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#e6edf3",
    "text.color":"#e6edf3","xtick.color":"#8b949e","ytick.color":"#8b949e",
    "grid.color":"#21262d","grid.linestyle":"--","grid.alpha":0.5,
    "lines.linewidth":1.6,"font.family":"monospace",
}

def _ensure(d): os.makedirs(d, exist_ok=True); return d
def _slug(s):
    for c in " —/→+": s=s.replace(c,"_")
    return "_".join(s.lower().split())[:50]
def _mark(ax, times, color, label):
    if not times: return
    ax.axvline(times[0],color=color,lw=1.5,ls="--",label=label,alpha=0.85)
    for t in times[1:]: ax.axvline(t,color=color,lw=0.7,ls="--",alpha=0.45)


def plot_scenario(result: SimulationResult, output_dir: str = _DEFAULT) -> str:
    _ensure(output_dir)
    steps = result.steps
    if not steps: return ""

    t           = [s.mission_time_s        for s in steps]
    temp        = [s.temperature_k         for s in steps]
    batt        = [s.battery_level*100     for s in steps]
    risk        = [s.risk_score            for s in steps]
    conf        = [s.confidence            for s in steps]
    gate_ok     = [1 if s.gate_passed else 0 for s in steps]
    axiom_fails = [len(s.axiom_failures)   for s in steps]
    power_w     = [s.power_w               for s in steps]

    fault_t    = [s.mission_time_s for s in steps if s.fault_active]
    blackout_t = [s.mission_time_s for s in steps if s.blackout_active]
    block_t    = [s.mission_time_s for s in steps if not s.gate_passed]
    safehold_t = [s.mission_time_s for s in steps if "SAFE_HOLD" in s.system_mode]

    n    = len(steps)
    span = (t[-1]-t[0]) if n>1 else 1.0
    bw   = span/n

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(20,16))
        ok_col = "#58a6ff" if result.pass_fail else "#f85149"
        fig.suptitle(
            f"ARVS SIMULATION  ·  {result.scenario.upper()}\n"
            f"Source: {result.source}  |  Frames: {result.total_frames}  |  "
            f"Blocks: {result.summary['gate_blocks']}  |  "
            f"Avg risk: {result.summary['avg_risk_score']:.3f}  |  "
            f"{'✓ PASS' if result.pass_fail else '✗ FAIL'}",
            fontsize=11,fontweight="bold",color=ok_col,y=0.99)

        gs = GridSpec(4,2,figure=fig,hspace=0.50,wspace=0.35,
                      left=0.07,right=0.97,top=0.93,bottom=0.05)

        # Panel 1 — Temperature
        ax1=fig.add_subplot(gs[0,0])
        ax1.plot(t,temp,color="#ff7b72",lw=1.8,label="Temp (K)")
        ax1.axhline(373*0.9,color="#f85149",ls="--",lw=1.2,label="Safe limit (90%)")
        ax1.axhline(373,color="#f85149",ls=":",lw=0.8,label="Hard max")
        _mark(ax1,fault_t,"#ffa657","Fault"); _mark(ax1,safehold_t,"#bf8700","Safe Hold")
        ax1.set_ylabel("Temperature (K)"); ax1.set_title("Thermal State")
        ax1.legend(fontsize=7,loc="upper right"); ax1.grid(True)

        # Panel 2 — Battery + Power
        ax2=fig.add_subplot(gs[0,1])
        ax2.plot(t,batt,color="#3fb950",lw=1.8,label="Battery (%)")
        ax2.fill_between(t,batt,alpha=0.12,color="#3fb950")
        ax2.axhline(35,color="#f85149",ls="--",lw=1.2,label="Min 35%")
        ax2r=ax2.twinx()
        ax2r.plot(t,power_w,color="#79c0ff",alpha=0.55,lw=1.0)
        ax2r.set_ylabel("Power (W)",color="#79c0ff",fontsize=8)
        ax2r.tick_params(axis="y",colors="#79c0ff",labelsize=7)
        _mark(ax2,block_t,"#f85149","Blocked")
        ax2.set_ylabel("SoC (%)"); ax2.set_title("Power / Battery")
        ax2.set_ylim(0,115); ax2.legend(fontsize=7,loc="upper right"); ax2.grid(True)

        # Panel 3 — Risk
        ax3=fig.add_subplot(gs[1,0])
        ax3.plot(t,risk,color="#79c0ff",lw=1.8,label="Risk")
        ax3.fill_between(t,risk,alpha=0.12,color="#79c0ff")
        ax3.axhspan(0,0.3,alpha=0.07,color="#3fb950")
        ax3.axhspan(0.3,0.6,alpha=0.07,color="#d29922")
        ax3.axhspan(0.6,1,alpha=0.07,color="#f85149")
        ax3.axhline(0.6,color="#f85149",ls="--",lw=1.2,label="High-risk thr")
        ax3.set_ylabel("Risk [0–1]"); ax3.set_title("8-Dimensional Risk Score")
        ax3.set_ylim(-0.02,1.05); ax3.legend(fontsize=7); ax3.grid(True)

        # Panel 4 — Confidence
        ax4=fig.add_subplot(gs[1,1])
        ax4.plot(t,conf,color="#d2a8ff",lw=1.8,label="Confidence")
        ax4.fill_between(t,conf,alpha=0.12,color="#d2a8ff")
        ax4.axhline(0.30,color="#f85149",ls="--",lw=1.2,label="Safe-hold thr")
        ax4.axhline(0.95,color="#ffa657",ls=":",lw=0.9,label="Irreversible thr")
        _mark(ax4,safehold_t,"#bf8700","Safe Hold")
        ax4.set_ylabel("Confidence [0–1]"); ax4.set_title("Epistemic Confidence")
        ax4.set_ylim(-0.02,1.05); ax4.legend(fontsize=7); ax4.grid(True)

        # Panel 5 — Gate timeline
        ax5=fig.add_subplot(gs[2,:])
        colours=["#3fb950" if g else "#f85149" for g in gate_ok]
        ax5.bar(t,[1]*n,width=bw,color=colours,align="edge",linewidth=0)
        for bt in blackout_t: ax5.axvline(bt,color="#d29922",alpha=0.35,lw=0.6)
        for ft in fault_t:    ax5.axvline(ft,color="#ffa657",alpha=0.70,lw=1.5,ls="--")
        ax5.set_yticks([]); ax5.set_xlim(t[0],t[-1]+bw)
        ax5.set_ylabel("Gate"); ax5.set_title("Safety Gate Decision Timeline  (green=pass · red=blocked)")
        ax5.legend(handles=[
            mpatches.Patch(color="#3fb950",label="Pass"),
            mpatches.Patch(color="#f85149",label="Blocked"),
            mpatches.Patch(color="#d29922",label="Comm blackout"),
            mpatches.Patch(color="#ffa657",label="Fault event"),
        ],fontsize=8,loc="upper right")

        # Panel 6 — Axiom failures
        ax6=fig.add_subplot(gs[3,:])
        ax6.fill_between(t,axiom_fails,step="mid",alpha=0.65,color="#ffa657",label="Axiom failures")
        ax6.plot(t,axiom_fails,color="#ffa657",lw=0.9,drawstyle="steps-mid")
        ax6.axhline(0,color="#3fb950",lw=0.8,ls=":")
        ax6.set_ylabel("# Axiom Failures"); ax6.set_xlabel("Mission Time (s)")
        ax6.set_title("Axiom Validation — Failure Count per Step  (25 axioms checked)")
        ax6.set_ylim(-0.2,max(max(axiom_fails)+2,5)); ax6.legend(fontsize=8); ax6.grid(True)

        for ax in [ax1,ax2,ax3,ax4]:
            ax.set_xlabel("Mission Time (s)",fontsize=8)

        if not result.pass_fail:
            fig.text(0.5,0.005,f"FAILURE: {result.failure_reason}",
                     ha="center",fontsize=10,color="#f85149",
                     bbox=dict(facecolor="#161b22",edgecolor="#f85149",pad=4))

        fname = os.path.join(output_dir,f"{_slug(result.scenario)}_{result.source[:20]}.png")
        fig.savefig(fname,dpi=150,bbox_inches="tight",facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Plot → {fname}")
        return fname


def write_forensic_json(result: SimulationResult, output_dir: str = _DEFAULT) -> str:
    _ensure(output_dir)
    fname = os.path.join(output_dir, f"forensic_{_slug(result.scenario)}_{result.source[:20]}.json")
    payload = {
        "scenario": result.scenario, "source": result.source,
        "pass": result.pass_fail, "failure_reason": result.failure_reason,
        "summary": result.summary, "forensic_log": result.forensic_log,
        "anomalous_steps": [
            {"frame":s.frame_idx,"t":round(s.mission_time_s,3),
             "events":s.events,"gate_violations":s.gate_violations,
             "axiom_failures":s.axiom_failures,"risk":round(s.risk_score,4),
             "confidence":round(s.confidence,4),"temp_k":round(s.temperature_k,2),
             "battery_pct":round(s.battery_level*100,1)}
            for s in result.steps
            if s.events or not s.gate_passed or s.axiom_failures
        ]
    }
    with open(fname,"w") as f: json.dump(payload,f,indent=2)
    logger.info(f"Forensic → {fname}")
    return fname


def write_ci_report(all_results: List[SimulationResult], output_dir: str = _DEFAULT) -> str:
    _ensure(output_dir)
    lines = ["="*80,"ARVS SIMULATION  —  CI PASS / FAIL REPORT","="*80,""]
    overall_pass = True
    records = []
    for r in all_results:
        if not r.pass_fail: overall_pass = False
        stat = "PASS ✓" if r.pass_fail else "FAIL ✗"
        lines += [
            f"  Scenario  : {r.scenario}",
            f"  Source    : {r.source}",
            f"  Status    : {stat}",
            f"  Frames    : {r.total_frames}",
            f"  Blocks    : {r.summary['gate_blocks']} ({r.summary['gate_block_rate']*100:.1f}%)",
            f"  Avg Risk  : {r.summary['avg_risk_score']:.4f}",
            f"  Max Risk  : {r.summary['max_risk_score']:.4f}",
            f"  Max Temp  : {r.summary['max_temperature_k']:.1f} K",
            f"  Min Batt  : {r.summary['min_battery_level']*100:.1f}%",
            f"  Safe Hold : {r.summary['safe_hold_triggered']}",
        ]
        if not r.pass_fail: lines.append(f"  !! REASON : {r.failure_reason}")
        lines.append("")
        records.append({"scenario":r.scenario,"source":r.source,
                        "pass":r.pass_fail,"summary":r.summary})
    lines += ["─"*80,
              f"  OVERALL : {'ALL SCENARIOS PASSED ✓' if overall_pass else 'ONE OR MORE FAILURES ✗'}",
              "─"*80]
    txt  = os.path.join(output_dir,"ci_report.txt")
    jpath = os.path.join(output_dir,"ci_report.json")
    with open(txt,"w") as f: f.write("\n".join(lines))
    import json as _json
    with open(jpath,"w") as f: _json.dump({"overall_pass":overall_pass,"scenarios":records},f,indent=2)
    logger.info(f"CI report → {txt}")
    return txt


def validate_audit_log(result: SimulationResult) -> Dict:
    expected, missed, spurious = [], [], []
    for s in result.steps:
        should = s.fault_active or (not s.sensor_healthy and s.confidence < 0.4)
        did    = not s.gate_passed
        if should and did: expected.append(s.frame_idx)
        elif should and not did: missed.append(s.frame_idx)
        elif not should and did:
            if not any("AXIOM" in v.get("type","") for v in s.gate_violations):
                spurious.append(s.frame_idx)
    return {"scenario":result.scenario,"expected_blocks":len(expected),
            "missed_blocks":len(missed),"spurious_blocks":len(spurious),
            "audit_pass":len(missed)==0,
            "missed_at_frames":missed[:10],"spurious_at_frames":spurious[:10]}
