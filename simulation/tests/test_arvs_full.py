"""
tests/test_arvs_full.py
ARVS Full Test Suite
=====================
pytest-based, CI-compatible.  Zero random values — all test inputs are
deterministic and physically motivated.

Covers:
  1. HAL / TelemetryBus / MockSensorDrivers
  2. MDP + POMDP formal decision model
  3. MissionPlanner + A* path planning
  4. ExperienceDB + PolicyUpdater
  5. SafetyGate (Python)
  6. AxiomValidator (Python)
  7. SimulationEngine × all 4 scenarios
  8. Integration: HAL → SimulationEngine → ExperienceDB

Run with:
  cd ARVS-main
  pip install pytest numpy pandas matplotlib --break-system-packages
  pytest simulation/tests/test_arvs_full.py -v
"""

import os
import sys
import math
import time
import tempfile
import logging

import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SIM  = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, SIM)

logging.basicConfig(level=logging.WARNING)   # suppress INFO noise in test output


# ═════════════════════════════════════════════════════════════════════════════
# 1. HAL & TelemetryBus
# ═════════════════════════════════════════════════════════════════════════════

class TestTelemetryBus:

    def setup_method(self):
        from ARVS.hardware.hal import TelemetryBus, SensorType, SensorHealth, TelemetryFrame
        self.TelemetryBus  = TelemetryBus
        self.SensorType    = SensorType
        self.SensorHealth  = SensorHealth
        self.TelemetryFrame = TelemetryFrame
        self.bus = TelemetryBus(history_depth=100)

    def _make_frame(self, sid="IMU_0", seq=1, health=None):
        from ARVS.hardware.hal import SensorHealth
        return self.TelemetryFrame(
            sensor_id="IMU_0",
            sensor_type=self.SensorType.IMU,
            acquisition_time=time.time(),
            sequence_number=seq,
            acceleration=(0.0, 0.0, -3.72),
            health=health or SensorHealth.HEALTHY,
        )

    def test_publish_and_latest(self):
        frame = self._make_frame(seq=1)
        self.bus.publish(frame)
        latest = self.bus.latest(self.SensorType.IMU)
        assert latest is not None
        assert latest.sensor_id == "IMU_0"
        assert latest.sequence_number == 1

    def test_subscriber_called(self):
        received = []
        self.bus.subscribe(self.SensorType.IMU, lambda f: received.append(f))
        frame = self._make_frame(seq=1)
        self.bus.publish(frame)
        assert len(received) == 1
        assert received[0].sequence_number == 1

    def test_subscriber_unsubscribe(self):
        received = []
        cb = lambda f: received.append(f)
        self.bus.subscribe(self.SensorType.IMU, cb)
        self.bus.unsubscribe(self.SensorType.IMU, cb)
        self.bus.publish(self._make_frame(seq=1))
        assert len(received) == 0

    def test_drop_detection(self):
        self.bus.publish(self._make_frame(seq=1))
        self.bus.publish(self._make_frame(seq=5))   # gap of 3
        assert self.bus.total_drops == 3

    def test_system_confidence_all_healthy(self):
        from ARVS.hardware.hal import SensorHealth
        self.bus.publish(self._make_frame(seq=1, health=SensorHealth.HEALTHY))
        assert self.bus.system_confidence() == pytest.approx(1.0)

    def test_system_confidence_degraded(self):
        from ARVS.hardware.hal import SensorHealth
        self.bus.publish(self._make_frame(seq=1, health=SensorHealth.DEGRADED))
        assert self.bus.system_confidence() == pytest.approx(0.6)

    def test_history_returns_n_frames(self):
        for i in range(1, 6):
            self.bus.publish(self._make_frame(seq=i))
        h = self.bus.history(self.SensorType.IMU, n=3)
        assert len(h) == 3
        assert h[-1].sequence_number == 5

    def test_frame_is_immutable(self):
        """TelemetryFrame is frozen — subscribers cannot mutate data."""
        frame = self._make_frame(seq=1)
        with pytest.raises(Exception):
            frame.sequence_number = 999   # type: ignore

    def test_frame_confidence_healthy(self):
        from ARVS.hardware.hal import SensorHealth
        f = self._make_frame(seq=1, health=SensorHealth.HEALTHY)
        assert f.confidence == 1.0

    def test_frame_confidence_failed(self):
        from ARVS.hardware.hal import SensorHealth
        f = self._make_frame(seq=1, health=SensorHealth.FAILED)
        assert f.confidence == 0.0


class TestMockSensorDrivers:

    def test_imu_driver_produces_valid_frame(self):
        from ARVS.hardware.hal import (
            TelemetryBus, MockIMUDriver, SensorType
        )
        bus = TelemetryBus()
        drv = MockIMUDriver("IMU_0", bus, rate_hz=10.0)
        frame = drv.read()
        assert frame is not None
        assert frame.sensor_type == SensorType.IMU
        assert frame.acceleration is not None
        # Mars gravity: z ≈ -3.72 m/s²
        assert abs(frame.acceleration[2] + 3.72) < 0.5

    def test_gps_driver_produces_valid_frame(self):
        from ARVS.hardware.hal import TelemetryBus, MockGPSDriver, SensorType
        bus = TelemetryBus()
        drv = MockGPSDriver("GPS_0", bus)
        frame = drv.read()
        assert frame is not None
        assert frame.position is not None
        assert len(frame.position) == 3

    def test_thermistor_temperature_in_martian_range(self):
        from ARVS.hardware.hal import TelemetryBus, MockThermistorDriver
        bus = TelemetryBus()
        drv = MockThermistorDriver("THERM_0", bus)
        frame = drv.read()
        assert 150.0 < frame.temperature_k < 400.0   # published REMS range

    def test_power_monitor_soc_bounded(self):
        from ARVS.hardware.hal import TelemetryBus, MockPowerMonitorDriver
        bus = TelemetryBus()
        drv = MockPowerMonitorDriver("PWR_0", bus)
        for _ in range(20):
            frame = drv.read()
        assert 0.0 <= frame.battery_soc <= 1.0

    def test_hal_health_summary(self):
        from ARVS.hardware.hal import HardwareAbstractionLayer
        hal = HardwareAbstractionLayer(use_mock_sensors=True)
        # don't start threads — just test registration
        health = hal.health()
        assert "driver_count" in health
        assert health["driver_count"] == 4


# ═════════════════════════════════════════════════════════════════════════════
# 2. MDP Decision Model
# ═════════════════════════════════════════════════════════════════════════════

class TestMDP:

    def setup_method(self):
        from ARVS.planning.mdp import (
            build_mission_mdp, MDPState, MDPAction, ActionType,
            RewardWeights, RewardFunction, TransitionModel, BeliefState
        )
        self.MDPState   = MDPState
        self.MDPAction  = MDPAction
        self.ActionType = ActionType
        self.RewardWeights  = RewardWeights
        self.RewardFunction = RewardFunction
        self.TransitionModel = TransitionModel
        self.BeliefState = BeliefState
        self.build = build_mission_mdp

    def test_mdp_builds_correct_state_count(self):
        solver, states, actions = self.build(grid_size=2)
        # 2×2 grid × 4 modes × 10 batt × 10 temp × 10 conf = 16,000
        assert len(states) == 2*2*4*10*10*10

    def test_reward_function_emergency_stop_safe(self):
        rf = self.RewardFunction()
        s = self.MDPState((0,0), "DEGRADED", 2, 8, 3)
        a_estop = self.MDPAction(self.ActionType.EMERGENCY_STOP, 1.0, 10.0, 0.0, False)
        a_move  = self.MDPAction(self.ActionType.MOVE_FORWARD, 10.0, 80.0, 2.0, True)
        ns = s
        r_stop  = rf(s, a_estop, ns)
        r_move  = rf(s, a_move,  ns)
        # Emergency stop should be rewarded more than risky motion in degraded
        assert r_stop > r_move

    def test_transition_emergency_stop_deterministic(self):
        tm = self.TransitionModel()
        s  = self.MDPState((2,2), "NORMAL", 5, 3, 7)
        a  = self.MDPAction(self.ActionType.EMERGENCY_STOP, 1.0, 10.0, 0.0, False)
        outcomes = tm.transitions(s, a)
        assert len(outcomes) == 1
        assert outcomes[0][1] == 1.0   # probability = 1
        assert outcomes[0][0].system_mode == "EMERGENCY"

    def test_transition_probabilities_sum_to_one(self):
        tm = self.TransitionModel()
        s  = self.MDPState((1,1), "NORMAL", 5, 3, 7)
        for action_type in [
            self.ActionType.MOVE_FORWARD,
            self.ActionType.CHARGE,
            self.ActionType.SAFE_HOLD,
            self.ActionType.SCIENCE_SAMPLE,
        ]:
            a = self.MDPAction(action_type, 10.0, 60.0, 2.0, True)
            outcomes = tm.transitions(s, a)
            total = sum(p for _, p in outcomes)
            assert abs(total - 1.0) < 1e-9, \
                f"{action_type}: probabilities sum to {total}"

    def test_mdp_solves_small_grid(self):
        # Use grid_size=2 to keep state space tractable for unit test
        solver, states, actions = self.build(grid_size=2)
        sol = solver.solve(epsilon=0.01, max_iterations=50)
        assert sol.n_states == len(states)
        assert sol.n_actions == len(actions)
        # Value function must be non-empty
        assert len(sol.value_function) > 0
        # Policy must cover all states
        assert len(sol.policy) == len(states)

    def test_belief_state_normalised(self):
        s1 = self.MDPState((0,0), "NORMAL", 5, 3, 7)
        s2 = self.MDPState((0,1), "NORMAL", 5, 3, 7)
        b  = self.BeliefState({s1: 3.0, s2: 1.0})
        assert abs(sum(b.distribution.values()) - 1.0) < 1e-9

    def test_belief_most_likely(self):
        s1 = self.MDPState((0,0), "NORMAL", 5, 3, 7)
        s2 = self.MDPState((0,1), "NORMAL", 5, 3, 7)
        b  = self.BeliefState({s1: 0.9, s2: 0.1})
        assert b.most_likely() == s1

    def test_reward_weights_gamma_default(self):
        w = self.RewardWeights()
        assert w.gamma == pytest.approx(0.95)

    def test_irreversible_action_in_degraded_mode_penalised(self):
        rf = self.RewardFunction()
        s  = self.MDPState((1,1), "DEGRADED", 3, 4, 5)
        a_irrev = self.MDPAction(self.ActionType.SCIENCE_SAMPLE, 60.0, 100.0, 8.0, False)
        a_rev   = self.MDPAction(self.ActionType.COMMUNICATE,    30.0, 40.0,  3.0, True)
        ns = s
        # Irreversible in degraded should be more penalised
        assert rf(s, a_irrev, ns) < rf(s, a_rev, ns)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Mission Planner & A*
# ═════════════════════════════════════════════════════════════════════════════

class TestMissionPlanner:

    def setup_method(self):
        from ARVS.planning.mission_planner import (
            MissionPlanner, MissionGoal, GoalType, TerrainMap, AStarPlanner
        )
        self.MissionPlanner = MissionPlanner
        self.MissionGoal    = MissionGoal
        self.GoalType       = GoalType
        self.TerrainMap     = TerrainMap
        self.AStarPlanner   = AStarPlanner

    def _state(self, pos=(0.0, 0.0, 0.0), batt=0.8, mode="NORMAL"):
        return {"position": list(pos), "battery_level": batt,
                "system_mode": mode}

    def test_plan_reach_waypoint(self):
        planner = self.MissionPlanner()
        planner.add_goal(self.MissionGoal(
            "g1", self.GoalType.REACH_WAYPOINT, priority=5,
            target_pos=(3.0, 3.0)))
        plan = planner.plan_next(self._state())
        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.is_valid

    def test_plan_science_sample(self):
        planner = self.MissionPlanner()
        planner.add_goal(self.MissionGoal(
            "g1", self.GoalType.COLLECT_SAMPLE, priority=5))
        plan = planner.plan_next(self._state())
        assert plan is not None
        assert any(s.action_type == "science_sample" for s in plan.steps)

    def test_plan_recharge(self):
        planner = self.MissionPlanner()
        planner.add_goal(self.MissionGoal(
            "g1", self.GoalType.RECHARGE, priority=8,
            target_value=0.95))
        plan = planner.plan_next(self._state(batt=0.50))
        assert plan is not None
        assert plan.steps[0].action_type == "charge"
        assert plan.total_duration_s > 0

    def test_plan_energy_budget_positive(self):
        planner = self.MissionPlanner()
        planner.add_goal(self.MissionGoal(
            "g1", self.GoalType.COLLECT_SAMPLE, priority=5))
        plan = planner.plan_next(self._state())
        assert plan.total_energy_wh >= 0.0
        assert plan.worst_case_energy_wh >= plan.total_energy_wh

    def test_no_plan_in_emergency(self):
        planner = self.MissionPlanner()
        planner.add_goal(self.MissionGoal(
            "g1", self.GoalType.REACH_WAYPOINT, priority=5,
            target_pos=(5.0, 5.0)))
        plan = planner.plan_next(self._state(mode="EMERGENCY"))
        assert plan is None

    def test_action_sequence_export(self):
        planner = self.MissionPlanner()
        planner.add_goal(self.MissionGoal(
            "g1", self.GoalType.COLLECT_SAMPLE, priority=5))
        plan = planner.plan_next(self._state())
        seq = plan.action_sequence()
        assert isinstance(seq, list)
        assert all("action_type" in a for a in seq)
        assert all("duration" in a for a in seq)

    def test_astar_finds_path_on_open_grid(self):
        terrain = self.TerrainMap(10, 10, resolution_m=1.0)
        planner_astar = self.AStarPlanner(terrain)
        path = planner_astar.plan((0.0, 0.0), (5.0, 5.0))
        assert path is not None
        assert len(path) > 0
        # Last waypoint should be close to goal
        gx, gy = path[-1]
        assert abs(gx - 5.0) <= 1.0
        assert abs(gy - 5.0) <= 1.0

    def test_astar_avoids_obstacles(self):
        terrain = self.TerrainMap(20, 20, resolution_m=1.0)
        terrain.set_obstacle(10.0, 5.0, radius_m=1.0)
        planner_astar = self.AStarPlanner(terrain)
        path = planner_astar.plan((0.0, 0.0), (15.0, 5.0))
        if path:
            # No waypoint should be inside the obstacle
            for wx, wy in path:
                dist = math.hypot(wx - 10.0, wy - 5.0)
                assert dist > 0.5, f"Path passed through obstacle at ({wx},{wy})"

    def test_astar_returns_none_for_blocked_goal(self):
        terrain = self.TerrainMap(10, 10, resolution_m=1.0)
        terrain.set_obstacle(5.0, 5.0, radius_m=3.0)
        planner_astar = self.AStarPlanner(terrain)
        path = planner_astar.plan((0.0, 0.0), (5.0, 5.0))
        assert path is None

    def test_goal_priority_ordering(self):
        planner = self.MissionPlanner()
        planner.add_goal(self.MissionGoal("g_low",  self.GoalType.COLLECT_SAMPLE, priority=2))
        planner.add_goal(self.MissionGoal("g_high", self.GoalType.RECHARGE,       priority=9, target_value=0.9))
        plan = planner.plan_next(self._state(batt=0.3))
        # High priority goal should be planned first
        assert plan is not None
        assert plan.goal_id == "g_high"


# ═════════════════════════════════════════════════════════════════════════════
# 4. ExperienceDB & PolicyUpdater
# ═════════════════════════════════════════════════════════════════════════════

class TestExperienceDB:

    def setup_method(self):
        from ARVS.learning.experience_db import (
            ExperienceDB, Episode, Transition, FailureRecord,
            PolicyUpdater, RewardWeightBaseline
        )
        self.ExperienceDB        = ExperienceDB
        self.Episode             = Episode
        self.Transition          = Transition
        self.FailureRecord       = FailureRecord
        self.PolicyUpdater       = PolicyUpdater
        self.RewardWeightBaseline = RewardWeightBaseline

        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db   = ExperienceDB(self._tmp.name)

    def teardown_method(self):
        self.db.close()
        os.unlink(self._tmp.name)

    def _episode(self, eid="ep1", success=True, gate_blocks=0):
        return self.Episode(
            episode_id       = eid,
            mission_source   = "SYNTHETIC_REMS",
            scenario         = "normal_ops",
            start_time       = 1000.0,
            end_time         = 2000.0,
            total_frames     = 200,
            gate_blocks      = gate_blocks,
            axiom_failures   = 0,
            safe_hold_triggered = False,
            mission_success  = success,
            avg_risk_score   = 0.05,
            max_temperature_k = 305.0,
            min_battery_level = 0.55,
            reward_weights   = {"mission_gain": 1.0, "risk_penalty": 5.0},
        )

    def test_write_and_read_episode(self):
        ep = self._episode()
        self.db.write_episode(ep)
        episodes = self.db.episodes()
        assert len(episodes) == 1
        assert episodes[0].episode_id == "ep1"

    def test_idempotent_write(self):
        ep = self._episode()
        self.db.write_episode(ep)
        self.db.write_episode(ep)   # duplicate — should be ignored
        assert len(self.db.episodes()) == 1

    def test_mission_success_rate(self):
        self.db.write_episode(self._episode("e1", success=True))
        self.db.write_episode(self._episode("e2", success=True))
        self.db.write_episode(self._episode("e3", success=False))
        rate = self.db.mission_success_rate()
        assert rate == pytest.approx(2/3)

    def test_failure_statistics(self):
        from ARVS.learning.experience_db import FailureRecord
        fr = FailureRecord(
            record_id="f1", episode_id="ep1", mission_time_s=100.0,
            failure_type="GATE_BLOCK", details={"type": "TORQUE_EXCEEDED"},
            confidence_at_failure=0.8, temp_at_failure=310.0,
            battery_at_failure=0.6,
        )
        self.db.write_failure(fr)
        stats = self.db.failure_statistics()
        assert "GATE_BLOCK" in stats
        assert stats["GATE_BLOCK"]["count"] == 1

    def test_episode_content_hash_stable(self):
        ep = self._episode()
        h1 = ep.content_hash()
        h2 = ep.content_hash()
        assert h1 == h2

    def test_policy_updater_requires_minimum_episodes(self):
        updater = self.PolicyUpdater(self.db)
        assert not updater.should_update()   # 0 episodes < 10

    def test_policy_updater_runs_when_enough_episodes(self):
        for i in range(15):
            ep = self._episode(
                f"ep_{i}",
                success=(i % 3 != 0),
                gate_blocks=i % 5,
            )
            self.db.write_episode(ep)

        updater = self.PolicyUpdater(self.db)
        assert updater.should_update()

    def test_policy_update_bounded(self):
        """Updated weights must stay within ±15% of baseline."""
        import numpy as np
        baseline = self.RewardWeightBaseline()

        for i in range(15):
            self.db.write_episode(self._episode(f"ep_{i}", success=i % 2 == 0))

        updater = self.PolicyUpdater(self.db, baseline)
        result  = updater.update()

        lo, hi = baseline.bounds()
        current = updater.current
        for dim in range(len(current)):
            assert current[dim] >= lo[dim] - 1e-9, \
                f"dim {dim} below lower bound: {current[dim]} < {lo[dim]}"
            assert current[dim] <= hi[dim] + 1e-9, \
                f"dim {dim} above upper bound: {current[dim]} > {hi[dim]}"

    def test_risk_penalty_never_decreases_below_80pct(self):
        """Axiom L1 guard: risk_penalty ≥ 80% of baseline."""
        baseline = self.RewardWeightBaseline()
        for i in range(20):
            self.db.write_episode(self._episode(f"ep_{i}", success=False, gate_blocks=100))

        updater = self.PolicyUpdater(self.db, baseline)
        updater.update()
        assert updater.current[1] >= baseline.risk_penalty * 0.80


# ═════════════════════════════════════════════════════════════════════════════
# 5. Python SafetyGate
# ═════════════════════════════════════════════════════════════════════════════

class TestSafetyGate:

    def setup_method(self):
        from ARVS.safety.safety_gate import (
            SafetyGate, SafetyConstraints, RobotState, Action,
            SafetyViolationType
        )
        self.SafetyGate      = SafetyGate
        self.SafetyConstraints = SafetyConstraints
        self.RobotState      = RobotState
        self.Action          = Action
        self.ViolationType   = SafetyViolationType

        self.constraints = SafetyConstraints(
            max_torque={"joint1": 100.0, "joint2": 80.0},
            max_velocity={"joint1": 2.0},
            thermal_limits={"motor": 373.0},
            structural_load_limits={"arm": 200.0},
            min_battery=0.15,
        )
        self.gate  = SafetyGate(self.constraints)
        self.state = RobotState(
            robot_id="TEST_01", timestamp=1000.0,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.05, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 0.0]),
            temperature=310.0, battery_level=0.75,
            power_consumption=60.0, confidence=0.95
        )

    def _action(self, torque=50.0, thermal=5.0, power=40.0, dur=10.0):
        return self.Action(
            action_id="a1", action_type="motion",
            parameters={}, duration=dur,
            max_torque=torque, max_velocity=0.05,
            thermal_load=thermal, power_required=power,
        )

    def test_safe_action_passes(self):
        result = self.gate.check_action(self._action(), self.state)
        assert result.safe

    def test_torque_exceeded_blocked(self):
        result = self.gate.check_action(self._action(torque=180.0), self.state)
        assert not result.safe
        types = [v[0] for v in result.violations]
        assert self.ViolationType.TORQUE_EXCEEDED in types

    def test_thermal_exceeded_blocked(self):
        # State temp 310 K + 60 K load = 370 K > 373*0.9 = 335.7 K limit
        result = self.gate.check_action(self._action(thermal=60.0), self.state)
        assert not result.safe

    def test_battery_low_blocked(self):
        low_batt_state = self.RobotState(
            robot_id="TEST_02", timestamp=1000.0,
            position=np.array([0.0,0.0,0.0]),
            velocity=np.array([0.0,0.0,0.0]),
            orientation=np.array([1.0,0.0,0.0,0.0]),
            angular_velocity=np.array([0.0,0.0,0.0]),
            temperature=300.0, battery_level=0.20,
            power_consumption=50.0, confidence=0.9,
        )
        # High power action drains past minimum
        result = self.gate.check_action(
            self._action(power=5000.0, dur=3600.0), low_batt_state)
        assert not result.safe

    def test_safe_action_updates_last_safe_state(self):
        self.gate.check_action(self._action(), self.state)
        self.gate.update_last_safe_state(self.state)
        saved = self.gate.get_last_safe_state()
        assert saved is not None
        assert saved.robot_id == "TEST_01"

    def test_metrics_updated(self):
        self.gate.check_action(self._action(), self.state)
        m = self.gate.get_safety_metrics()
        assert "average_safety_score" in m

    def test_reset_clears_history(self):
        self.gate.check_action(self._action(torque=180.0), self.state)
        self.gate.reset()
        m = self.gate.get_safety_metrics()
        assert m["total_violations"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# 6. Axiom Validator
# ═════════════════════════════════════════════════════════════════════════════

class TestAxiomValidator:
    """Tests the StandaloneAxiomValidator from simulation_engine."""

    def setup_method(self):
        from engine.simulation_engine import StandaloneAxiomValidator
        self.validator = StandaloneAxiomValidator()
        self.t = 1000.0
        self._base = {
            "confidence": 0.85,
            "uncertainty_explicit": True,
            "uncertainty_current": 0.15,
            "uncertainty_max_credible": 0.15,
            "uncertainty_previous": 0.15,
            "has_new_evidence": True,
            "belief_oscillation_rate": 0.5,
            "belief_timestamp": 999.0,
            "belief_validity_window": 60.0,
            "n_active_authorities": 1,
            "authority_revoked_this_cycle": False,
            "authority_explicitly_defined": True,
            "is_acting": True,
            "evaluation_basis": "worst_case_credible",
            "potential_harm": 0.3,
            "action_is_reversible": True,
            "distance_to_harm": 0.8,
            "constraint_tightness": 1.0,
            "refusal_is_illegal": False,
            "action_is_full_capability": False,
            "is_safe_for_full_capability": True,
            "is_optimizing": False,
            "is_safe": True,
            "all_actions_gated": True,
            "gate_decision_final": True,
            "learning_overrides": False,
            "is_irreversible_context": False,
            "online_learning_active": False,
            "action_has_explanation": True,
            "justification_timestamp": 998.0,
            "action_timestamp": 999.0,
            "system_mode": "NORMAL",
        }

    def test_all_axioms_pass_on_good_state(self):
        passed, failures = self.validator.validate(self._base, self.t)
        assert passed, f"Failures: {failures}"

    def test_e1_fails_on_omniscience(self):
        s = {**self._base, "confidence": 1.0}
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("E1" in f for f in failures)

    def test_e3_fails_on_rapid_oscillation(self):
        s = {**self._base, "belief_oscillation_rate": 3.0}
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("E3" in f for f in failures)

    def test_a2_fails_with_two_authorities(self):
        s = {**self._base, "n_active_authorities": 2}
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("A2" in f for f in failures)

    def test_g3_fails_in_emergency_without_tightening(self):
        s = {**self._base,
             "system_mode": "EMERGENCY",
             "constraint_tightness": 1.0}   # should be ≥ 1.2
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("G3" in f for f in failures)

    def test_l1_fails_when_learning_overrides(self):
        s = {**self._base, "learning_overrides": True}
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("L1" in f for f in failures)

    def test_l2_fails_online_learning_in_irreversible(self):
        s = {**self._base,
             "is_irreversible_context": True,
             "online_learning_active": True}
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("L2" in f for f in failures)

    def test_t2_fails_when_justification_after_action(self):
        s = {**self._base,
             "justification_timestamp": 1001.0,
             "action_timestamp": 1000.0}
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("T2" in f for f in failures)

    def test_z_passes_in_degraded_mode_despite_failures(self):
        s = {**self._base,
             "confidence": 1.0,           # E1 fails
             "system_mode": "DEGRADED"}   # Z should still pass
        passed, failures = self.validator.validate(s, self.t)
        # Z passes because system is in degraded mode
        assert not any("Z:" in f for f in failures), \
            f"Z should pass in DEGRADED mode: {failures}"

    def test_c3_fails_irreversible_low_confidence(self):
        s = {**self._base,
             "action_is_reversible": False,
             "confidence": 0.80}   # below 0.95 threshold
        passed, failures = self.validator.validate(s, self.t)
        assert not passed
        assert any("C3" in f for f in failures)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Simulation Engine — all 4 scenarios × REMS source
# ═════════════════════════════════════════════════════════════════════════════

class TestSimulationEngine:

    def setup_method(self):
        from data_loaders.telemetry_loader import _synthetic_rems
        from engine.simulation_engine import SimulationEngine
        from scenarios.scenarios import (
            NORMAL_OPS, FAULT_INJECTION, COMM_BLACKOUT, AXIOM_CASCADE
        )
        # 200 frames covers a full Martian sol — needed for cascade (frame 60+)
        # and comm blackout (t > 500 s requires frames spanning ~500s)
        self.frames = _synthetic_rems(200)
        self.engine = SimulationEngine()
        self.NORMAL_OPS    = NORMAL_OPS
        self.FAULT_INJECTION = FAULT_INJECTION
        self.COMM_BLACKOUT   = COMM_BLACKOUT
        self.AXIOM_CASCADE   = AXIOM_CASCADE

    def test_normal_ops_zero_gate_blocks(self):
        result = self.engine.run(self.frames, self.NORMAL_OPS)
        assert result.pass_fail
        assert result.summary["gate_blocks"] == 0

    def test_fault_injection_blocks_at_fault_frame(self):
        result = self.engine.run(self.frames, self.FAULT_INJECTION)
        assert result.pass_fail
        fault_step = next(
            (s for s in result.steps if s.fault_active), None)
        # Fault frame should either be blocked or not present (if frames < fault_start)
        if fault_step:
            assert not fault_step.gate_passed, \
                "Fault frame should have been blocked by safety gate"

    def test_axiom_cascade_triggers_safe_hold(self):
        result = self.engine.run(self.frames, self.AXIOM_CASCADE)
        assert result.pass_fail, f"Scenario failed: {result.failure_reason}"
        assert result.summary["safe_hold_triggered"], \
            f"safe_hold was not triggered. modes={set(s.system_mode for s in result.steps)}"

    def test_comm_blackout_logged(self):
        result = self.engine.run(self.frames, self.COMM_BLACKOUT)
        assert result.pass_fail, f"Scenario failed: {result.failure_reason}"
        blackout_steps = [s for s in result.steps if s.blackout_active]
        if any(s.mission_time_s > 500.0 for s in result.steps):
            assert len(blackout_steps) > 0, \
                "Frames span past t=500s but no blackout_active steps found"
            blackout_events = [
                e for s in result.steps for e in s.events
                if "COMM" in e or "blackout" in e.lower()
            ]
            assert len(blackout_events) > 0, "No COMM blackout events logged"

    def test_simulation_result_has_all_fields(self):
        result = self.engine.run(self.frames, self.NORMAL_OPS)
        assert result.total_frames == len(self.frames)
        assert len(result.steps) == len(self.frames)
        assert "avg_risk_score" in result.summary
        assert "max_temperature_k" in result.summary

    def test_risk_scores_bounded(self):
        result = self.engine.run(self.frames, self.NORMAL_OPS)
        for step in result.steps:
            assert 0.0 <= step.risk_score <= 1.0, \
                f"Risk out of bounds: {step.risk_score}"

    def test_confidence_bounded(self):
        result = self.engine.run(self.frames, self.AXIOM_CASCADE)
        for step in result.steps:
            assert 0.0 <= step.confidence <= 1.0, \
                f"Confidence out of bounds: {step.confidence}"


# ═════════════════════════════════════════════════════════════════════════════
# 8. Integration: HAL → SimulationEngine → ExperienceDB
# ═════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """
    Full pipeline: HAL produces telemetry → simulation runs scenarios
    → results written to ExperienceDB → policy updater analyses data.
    """

    def test_hal_feeds_simulation_engine(self):
        """HAL mock drivers produce frames compatible with the sim engine."""
        from ARVS.hardware.hal import MockThermistorDriver, TelemetryBus, SensorType
        from engine.simulation_engine import SimulationEngine, ScenarioConfig, ScenarioType
        from data_loaders.telemetry_loader import TelemetryFrame as TF
        import numpy as np

        bus = TelemetryBus()
        drv = MockThermistorDriver("THERM_0", bus)

        # Read 20 frames from mock driver, convert to simulation TelemetryFrames
        frames = []
        for i in range(20):
            hw_frame = drv.read()
            tf = TF(
                source           = "HAL_MOCK",
                mission_time_s   = i * 1.0,
                utc_timestamp    = "",
                position_m       = np.array([i * 0.01, 0.0, 0.0]),
                velocity_ms      = np.array([0.01, 0.0, 0.0]),
                orientation_quat = np.array([1.0, 0.0, 0.0, 0.0]),
                angular_velocity_rads = np.array([0.0, 0.0, 0.0]),
                temperature_k    = hw_frame.temperature_k,
                thermal_gradient_k_per_s = 0.0,
                battery_level    = 0.85,
                power_consumption_w = 60.0,
                solar_irradiance_w_m2 = 590.0,
                ambient_pressure_pa   = 800.0,
                wind_speed_ms    = 3.0,
                uv_index         = 2.0,
                confidence       = 0.92,
                sensor_flags     = {"temp": True},
                metadata         = {"real_data": False},
            )
            frames.append(tf)

        engine = SimulationEngine()
        config = ScenarioConfig(
            scenario_type=ScenarioType.NORMAL_OPS,
            name="HAL Integration Test",
            description="HAL mock → sim engine"
        )
        result = engine.run(frames, config)
        assert result.pass_fail
        assert result.total_frames == 20

    def test_simulation_to_experience_db_pipeline(self):
        from data_loaders.telemetry_loader import _synthetic_rems
        from engine.simulation_engine import SimulationEngine
        from scenarios.scenarios import NORMAL_OPS
        from ARVS.learning.experience_db import ExperienceDB, Episode

        frames = _synthetic_rems(50)
        engine = SimulationEngine()
        result = engine.run(frames, NORMAL_OPS)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        db = ExperienceDB(db_path)

        ep = Episode(
            episode_id        = "integration_ep_1",
            mission_source    = result.source,
            scenario          = result.scenario,
            start_time        = 0.0,
            end_time          = result.steps[-1].mission_time_s if result.steps else 0.0,
            total_frames      = result.total_frames,
            gate_blocks       = result.summary["gate_blocks"],
            axiom_failures    = result.summary["axiom_failure_frames"],
            safe_hold_triggered = result.summary["safe_hold_triggered"],
            mission_success   = result.pass_fail,
            avg_risk_score    = result.summary["avg_risk_score"],
            max_temperature_k = result.summary["max_temperature_k"],
            min_battery_level = result.summary["min_battery_level"],
            reward_weights    = {"mission_gain": 1.0, "risk_penalty": 5.0},
        )
        db.write_episode(ep)

        episodes = db.episodes()
        assert len(episodes) == 1
        assert episodes[0].mission_success == result.pass_fail

        db.close()
        os.unlink(db_path)
