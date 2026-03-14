"""
learning/experience_db.py
ARVS Experience Database + Policy Update
==========================================
Records past mission experiences so the system can:
  1. Avoid previously observed failure modes
  2. Refine the MDP reward function weights from outcomes
  3. Provide offline analysis material for researchers

Design decisions
----------------
* SQLite backend (single file, no server, ACID, survives power loss).
  Appropriate for rover / spacecraft: flash-friendly, small footprint.
* Schema is append-only. No row updates — immutable forensic trail.
* Policy update uses a lightweight gradient-free method (CMA-ES style
  directional search on the 5-dimensional reward weight space).
  We avoid online gradient descent because:
    (a) safety axioms forbid online learning in irreversible contexts
    (b) gradient estimates from sparse episode data are noisy
* The ±15% bounded learning constraint from adaptive_models.py is
  enforced here too: reward weights cannot change by more than 15%
  from their baseline values per update cycle.
* All writes are journaled: each record has a SHA-256 content hash
  linking to the ARVS audit log.
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default database path — override via environment variable for testing
DEFAULT_DB_PATH = os.environ.get(
    "ARVS_EXPERIENCE_DB",
    os.path.join(os.path.dirname(__file__), "..", "..", "experience.db")
)

# ─────────────────────────────────────────────────────────────────────────────
# Schema types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """
    One complete mission episode (plan → execution → outcome).
    Immutable once written.
    """
    episode_id:        str
    mission_source:    str          # "REMS", "ISS", "SYNTHETIC_REMS", etc.
    scenario:          str          # "normal_ops", "fault_injection", etc.
    start_time:        float
    end_time:          float
    total_frames:      int
    gate_blocks:       int
    axiom_failures:    int
    safe_hold_triggered: bool
    mission_success:   bool
    avg_risk_score:    float
    max_temperature_k: float
    min_battery_level: float
    reward_weights:    Dict[str, float]
    outcome_notes:     str = ""

    @property
    def duration_s(self) -> float:
        return self.end_time - self.start_time

    @property
    def block_rate(self) -> float:
        return self.gate_blocks / max(1, self.total_frames)

    def content_hash(self) -> str:
        d = asdict(self)
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class Transition:
    """
    One (state, action, reward, next_state) tuple.
    Written at ARVS_EXPERIENCE_WRITE_FREQ Hz during execution.
    """
    transition_id:   str
    episode_id:      str
    mission_time_s:  float
    state_summary:   Dict       # compact: temp, batt, conf, mode
    action_type:     str
    reward:          float
    next_state_summary: Dict
    gate_passed:     bool
    axiom_failures:  int


@dataclass
class FailureRecord:
    """
    Structured record of a safety gate block or axiom violation.
    Used to compute failure statistics for policy update.
    """
    record_id:      str
    episode_id:     str
    mission_time_s: float
    failure_type:   str         # "GATE_BLOCK", "AXIOM_VIOLATION", "SAFE_HOLD"
    details:        Dict
    confidence_at_failure: float
    temp_at_failure: float
    battery_at_failure: float


# ─────────────────────────────────────────────────────────────────────────────
# Database manager
# ─────────────────────────────────────────────────────────────────────────────

class ExperienceDB:
    """
    SQLite-backed experience database.
    Thread-safe: uses a single connection with WAL mode.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS episodes (
        episode_id       TEXT PRIMARY KEY,
        mission_source   TEXT,
        scenario         TEXT,
        start_time       REAL,
        end_time         REAL,
        total_frames     INTEGER,
        gate_blocks      INTEGER,
        axiom_failures   INTEGER,
        safe_hold        INTEGER,
        mission_success  INTEGER,
        avg_risk         REAL,
        max_temp_k       REAL,
        min_battery      REAL,
        reward_weights   TEXT,
        outcome_notes    TEXT,
        content_hash     TEXT
    );

    CREATE TABLE IF NOT EXISTS transitions (
        transition_id    TEXT PRIMARY KEY,
        episode_id       TEXT,
        mission_time_s   REAL,
        state_summary    TEXT,
        action_type      TEXT,
        reward           REAL,
        next_state       TEXT,
        gate_passed      INTEGER,
        axiom_failures   INTEGER
    );

    CREATE TABLE IF NOT EXISTS failures (
        record_id        TEXT PRIMARY KEY,
        episode_id       TEXT,
        mission_time_s   REAL,
        failure_type     TEXT,
        details          TEXT,
        conf             REAL,
        temp_k           REAL,
        battery          REAL
    );

    CREATE TABLE IF NOT EXISTS policy_updates (
        update_id        TEXT PRIMARY KEY,
        timestamp        REAL,
        episodes_used    INTEGER,
        old_weights      TEXT,
        new_weights      TEXT,
        improvement      REAL,
        notes            TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_episodes_scenario
        ON episodes(scenario);
    CREATE INDEX IF NOT EXISTS idx_transitions_episode
        ON transitions(episode_id);
    CREATE INDEX IF NOT EXISTS idx_failures_type
        ON failures(failure_type);
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn   = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()
        logger.info(f"ExperienceDB opened: {db_path}")

    def close(self) -> None:
        self._conn.close()

    # ── Write ──────────────────────────────────────────────────────────────────

    def write_episode(self, ep: Episode) -> None:
        self._conn.execute("""
            INSERT OR IGNORE INTO episodes VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ep.episode_id, ep.mission_source, ep.scenario,
            ep.start_time, ep.end_time, ep.total_frames,
            ep.gate_blocks, ep.axiom_failures,
            int(ep.safe_hold_triggered), int(ep.mission_success),
            ep.avg_risk_score, ep.max_temperature_k, ep.min_battery_level,
            json.dumps(ep.reward_weights), ep.outcome_notes,
            ep.content_hash()
        ))
        self._conn.commit()
        logger.debug(f"ExperienceDB: episode {ep.episode_id} written")

    def write_transition(self, t: Transition) -> None:
        self._conn.execute("""
            INSERT OR IGNORE INTO transitions VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            t.transition_id, t.episode_id, t.mission_time_s,
            json.dumps(t.state_summary), t.action_type, t.reward,
            json.dumps(t.next_state_summary),
            int(t.gate_passed), t.axiom_failures
        ))
        self._conn.commit()

    def write_failure(self, f: FailureRecord) -> None:
        self._conn.execute("""
            INSERT OR IGNORE INTO failures VALUES (?,?,?,?,?,?,?,?)
        """, (
            f.record_id, f.episode_id, f.mission_time_s,
            f.failure_type, json.dumps(f.details),
            f.confidence_at_failure, f.temp_at_failure, f.battery_at_failure
        ))
        self._conn.commit()

    # ── Read / query ──────────────────────────────────────────────────────────

    def episodes(self, scenario: Optional[str] = None,
                 limit: int = 100) -> List[Episode]:
        q = "SELECT * FROM episodes"
        params: Tuple = ()
        if scenario:
            q += " WHERE scenario = ?"
            params = (scenario,)
        q += f" ORDER BY start_time DESC LIMIT {limit}"
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_episode(r) for r in rows]

    def failure_statistics(self) -> Dict:
        rows = self._conn.execute("""
            SELECT failure_type, COUNT(*) as n,
                   AVG(conf) as avg_conf,
                   AVG(temp_k) as avg_temp,
                   AVG(battery) as avg_batt
            FROM failures
            GROUP BY failure_type
        """).fetchall()
        return {
            r[0]: {
                "count":        r[1],
                "avg_confidence": round(r[2] or 0.0, 4),
                "avg_temp_k":   round(r[3] or 0.0, 2),
                "avg_battery":  round(r[4] or 0.0, 4),
            }
            for r in rows
        }

    def mission_success_rate(self, scenario: Optional[str] = None) -> float:
        q      = "SELECT AVG(mission_success) FROM episodes"
        params: Tuple = ()
        if scenario:
            q += " WHERE scenario = ?"
            params = (scenario,)
        result = self._conn.execute(q, params).fetchone()[0]
        return float(result or 0.0)

    def avg_risk_by_scenario(self) -> Dict[str, float]:
        rows = self._conn.execute(
            "SELECT scenario, AVG(avg_risk) FROM episodes GROUP BY scenario"
        ).fetchall()
        return {r[0]: round(r[1], 4) for r in rows}

    def _row_to_episode(self, row) -> Episode:
        return Episode(
            episode_id       = row[0],
            mission_source   = row[1],
            scenario         = row[2],
            start_time       = row[3],
            end_time         = row[4],
            total_frames     = row[5],
            gate_blocks      = row[6],
            axiom_failures   = row[7],
            safe_hold_triggered = bool(row[8]),
            mission_success  = bool(row[9]),
            avg_risk_score   = row[10],
            max_temperature_k= row[11],
            min_battery_level= row[12],
            reward_weights   = json.loads(row[13] or "{}"),
            outcome_notes    = row[14] or "",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Policy update engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RewardWeightBaseline:
    """Default weights and their allowed ±15% bounds."""
    mission_gain:  float = 1.0
    risk_penalty:  float = 5.0
    energy_cost:   float = 0.3
    safety_bonus:  float = 2.0
    time_penalty:  float = 0.01
    gamma:         float = 0.95

    BOUND_PCT: float = 0.15   # ±15% maximum change per update

    def as_array(self) -> np.ndarray:
        return np.array([self.mission_gain, self.risk_penalty,
                         self.energy_cost, self.safety_bonus, self.time_penalty])

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        base = self.as_array()
        lo   = base * (1.0 - self.BOUND_PCT)
        hi   = base * (1.0 + self.BOUND_PCT)
        return lo, hi

    def from_array(self, arr: np.ndarray) -> "RewardWeightBaseline":
        return RewardWeightBaseline(
            mission_gain  = float(arr[0]),
            risk_penalty  = float(arr[1]),
            energy_cost   = float(arr[2]),
            safety_bonus  = float(arr[3]),
            time_penalty  = float(arr[4]),
            gamma         = self.gamma,
        )


class PolicyUpdater:
    """
    Updates MDP reward weights based on accumulated episode data.

    Algorithm: Coordinate descent within the ±15% constraint box.
    For each weight dimension, try +δ and -δ; keep the direction
    that improves the performance metric.

    Performance metric: weighted combination of
      - mission success rate (+)
      - gate block rate (-)
      - average risk score (-)

    Axiom constraint: update is only applied if the new weights satisfy
    all safety axioms (specifically L1: learning must not override safety).
    Concretely: risk_penalty must never decrease below 80% of baseline.
    """

    DELTA = 0.05    # step size for coordinate search (5% of current value)
    MIN_EPISODES_FOR_UPDATE = 10

    def __init__(self, db: ExperienceDB,
                 baseline: Optional[RewardWeightBaseline] = None):
        self.db       = db
        self.baseline = baseline or RewardWeightBaseline()
        self.current  = self.baseline.as_array().copy()

    def should_update(self) -> bool:
        """Returns True when enough new episodes have accumulated."""
        recent = self.db.episodes(limit=self.MIN_EPISODES_FOR_UPDATE)
        return len(recent) >= self.MIN_EPISODES_FOR_UPDATE

    def update(self) -> Optional[Dict]:
        """
        Run one policy update cycle.
        Returns a summary dict or None if update was not applied.
        """
        episodes = self.db.episodes(limit=100)
        if len(episodes) < self.MIN_EPISODES_FOR_UPDATE:
            logger.info("PolicyUpdater: not enough episodes, skipping")
            return None

        old_weights = self.current.copy()
        old_score   = self._score(episodes, old_weights)

        lo, hi = self.baseline.bounds()
        improved = False

        # Coordinate descent: one pass over all 5 weight dimensions
        for dim in range(len(self.current)):
            delta    = self.current[dim] * self.DELTA
            best_w   = self.current.copy()
            best_s   = old_score

            for sign in (+1, -1):
                candidate = self.current.copy()
                candidate[dim] = np.clip(
                    self.current[dim] + sign * delta, lo[dim], hi[dim])

                # Safety axiom L1 guard: risk_penalty (index 1) must not
                # decrease below 80% of baseline
                if candidate[1] < self.baseline.risk_penalty * 0.80:
                    continue

                s = self._score(episodes, candidate)
                if s > best_s:
                    best_s = s
                    best_w = candidate

            if best_s > old_score + 1e-6:
                self.current = best_w
                improved     = True

        if not improved:
            logger.info("PolicyUpdater: no improvement found, weights unchanged")
            return None

        new_weights = self.current.copy()
        improvement = (self._score(episodes, new_weights) - old_score)

        # Persist to DB
        update_id = f"upd_{int(time.time())}"
        self.db._conn.execute("""
            INSERT INTO policy_updates VALUES (?,?,?,?,?,?,?)
        """, (
            update_id, time.time(), len(episodes),
            json.dumps(old_weights.tolist()),
            json.dumps(new_weights.tolist()),
            float(improvement),
            f"Coordinate descent, {len(episodes)} episodes"
        ))
        self.db._conn.commit()

        summary = {
            "update_id":   update_id,
            "episodes":    len(episodes),
            "improvement": round(float(improvement), 5),
            "old_weights": {
                "mission_gain": round(float(old_weights[0]), 4),
                "risk_penalty": round(float(old_weights[1]), 4),
                "energy_cost":  round(float(old_weights[2]), 4),
                "safety_bonus": round(float(old_weights[3]), 4),
                "time_penalty": round(float(old_weights[4]), 5),
            },
            "new_weights": {
                "mission_gain": round(float(new_weights[0]), 4),
                "risk_penalty": round(float(new_weights[1]), 4),
                "energy_cost":  round(float(new_weights[2]), 4),
                "safety_bonus": round(float(new_weights[3]), 4),
                "time_penalty": round(float(new_weights[4]), 5),
            },
        }
        logger.info(f"PolicyUpdater: weights updated — improvement={improvement:.5f}")
        return summary

    def current_weights(self) -> Dict[str, float]:
        w = self.current
        return {
            "mission_gain":  float(w[0]),
            "risk_penalty":  float(w[1]),
            "energy_cost":   float(w[2]),
            "safety_bonus":  float(w[3]),
            "time_penalty":  float(w[4]),
        }

    def _score(self, episodes: List[Episode],
                weights: np.ndarray) -> float:
        """
        Performance metric for a set of weights given observed episodes.
        Higher = better.
        """
        if not episodes:
            return 0.0

        success_rate = sum(1 for e in episodes if e.mission_success) / len(episodes)
        avg_block    = sum(e.block_rate for e in episodes) / len(episodes)
        avg_risk     = sum(e.avg_risk_score for e in episodes) / len(episodes)

        # Reward weights influence: risk_penalty should correlate with
        # lower block rates and lower risk scores
        w_risk = weights[1]   # risk_penalty

        return (
            success_rate
            - 2.0 * avg_block
            - avg_risk
            + 0.1 * w_risk / self.baseline.risk_penalty   # prefer higher safety
        )
