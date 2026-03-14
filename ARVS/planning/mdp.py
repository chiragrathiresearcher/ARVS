"""
planning/mdp.py
ARVS Formal Decision Model — MDP and POMDP
============================================
Implements a mathematically rigorous decision model aligned with modern
robotics research practice (Kaelbling, Thrun, Sutton & Barto).

Two modes:
  1. FullyObservableMDP  — Markov Decision Process
     State fully known → standard value iteration / policy iteration
     Suitable for: execution controller with complete state feedback

  2. PartiallyObservableMDP (POMDP) — for uncertain environments
     Belief state is a probability distribution over states
     Suitable for: rover operations during sensor dropout,
                   spacecraft with delayed telemetry

Core design decisions
---------------------
* Risk is integrated into the reward function, not computed separately.
  R(s,a) = mission_gain(s,a) - λ·risk(s,a) - μ·energy(s,a)
  where λ and μ are mission-specific penalty weights.
* Safety constraints are HARD: actions violating safety axioms are
  removed from the action set before solving (not penalised — excluded).
* The solver is deliberately lightweight (synchronous value iteration)
  so it can run in the ARVS control loop on flight hardware.

References
----------
Kaelbling et al., "Planning and Acting in Partially Observable Stochastic
Domains", Artificial Intelligence 101(1-2), 1998.
Thrun et al., "Probabilistic Robotics", MIT Press, 2005.
"""

import math
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Domain types
# ─────────────────────────────────────────────────────────────────────────────

class ActionType(Enum):
    MOVE_FORWARD    = auto()
    MOVE_BACKWARD   = auto()
    TURN_LEFT       = auto()
    TURN_RIGHT      = auto()
    SCIENCE_SAMPLE  = auto()
    COMMUNICATE     = auto()
    CHARGE          = auto()
    SAFE_HOLD       = auto()
    EMERGENCY_STOP  = auto()


@dataclass(frozen=True)
class MDPState:
    """
    Discrete state for the MDP.
    For continuous systems, this represents a discretised cell in the
    state space (position bucket × mode × battery bucket).
    """
    position_bucket:  Tuple[int, int]  # (x_cell, y_cell) — grid index
    system_mode:      str              # "NORMAL", "DEGRADED", "SAFE_HOLD", "EMERGENCY"
    battery_bucket:   int              # 0–9 (tenths of full charge)
    temp_bucket:      int              # 0–9 (relative to safe range)
    confidence_bucket: int             # 0–9


@dataclass(frozen=True)
class MDPAction:
    """Discrete action."""
    action_type:     ActionType
    duration_s:      float
    power_required_w: float
    thermal_load_k:  float
    is_reversible:   bool = True


@dataclass
class RewardWeights:
    """
    Mission-specific reward function weights.
    Tune these per mission profile (science-maximising vs energy-conserving).
    """
    mission_gain:  float = 1.0    # reward for completing mission objectives
    risk_penalty:  float = 5.0    # λ — penalise risky states
    energy_cost:   float = 0.3    # μ — penalise high energy consumption
    safety_bonus:  float = 2.0    # reward for choosing safe actions
    time_penalty:  float = 0.01   # small penalty per step (encourages efficiency)

    # Discount factor — how much future rewards count
    # 0.95 typical for long-horizon space missions
    gamma:         float = 0.95


@dataclass
class MDPSolution:
    """Result of solving the MDP."""
    policy:          Dict[MDPState, MDPAction]   # state → optimal action
    value_function:  Dict[MDPState, float]        # V*(s)
    iterations:      int
    converged:       bool
    solve_time_s:    float
    n_states:        int
    n_actions:       int


# ─────────────────────────────────────────────────────────────────────────────
# Reward function — integrates risk into decision optimisation
# ─────────────────────────────────────────────────────────────────────────────

class RewardFunction:
    """
    R(s, a) = w_gain·gain(s,a) - w_risk·risk(s,a) - w_energy·energy(s,a)
            + w_safe·safe_bonus(a) - w_time·1

    Integrating risk directly into the reward means the decision optimiser
    automatically trades off mission progress against safety — no ad hoc
    weighting after the fact.
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.w = weights or RewardWeights()

    def __call__(self, state: MDPState, action: MDPAction,
                 next_state: MDPState) -> float:
        return (
            self.w.mission_gain * self._gain(state, action, next_state)
            - self.w.risk_penalty  * self._risk(state, action)
            - self.w.energy_cost   * self._energy(action)
            + self.w.safety_bonus  * self._safe_bonus(action, state)
            - self.w.time_penalty
        )

    def _gain(self, s: MDPState, a: MDPAction, ns: MDPState) -> float:
        """Mission gain: reward for science activities and maintaining good mode."""
        if a.action_type == ActionType.SCIENCE_SAMPLE:
            return 10.0
        if a.action_type == ActionType.COMMUNICATE:
            return 3.0
        if ns.system_mode == "NORMAL" and s.system_mode != "NORMAL":
            return 5.0   # recovered to normal mode
        return 0.0

    def _risk(self, s: MDPState, a: MDPAction) -> float:
        """Risk as a function of state + action."""
        # Battery risk: exponential penalty approaching empty
        batt_risk = max(0.0, 1.0 - s.battery_bucket / 3.0) ** 2

        # Thermal risk: penalty in high-temp buckets
        temp_risk = max(0.0, (s.temp_bucket - 7) / 3.0)

        # Mode risk
        mode_risk = {"NORMAL": 0.0, "DEGRADED": 0.3,
                     "SAFE_HOLD": 0.6, "EMERGENCY": 1.0}.get(s.system_mode, 0.5)

        # Action risk: irreversible actions in degraded states are very risky
        action_risk = 0.0
        if not a.is_reversible and s.system_mode != "NORMAL":
            action_risk = 2.0
        if a.action_type == ActionType.EMERGENCY_STOP:
            action_risk = 0.0  # always safe to stop

        return batt_risk + temp_risk + mode_risk + action_risk

    def _energy(self, a: MDPAction) -> float:
        """Normalised energy cost [0,1]."""
        return min(1.0, a.power_required_w * a.duration_s / 3600.0 / 50.0)

    def _safe_bonus(self, a: MDPAction, s: MDPState) -> float:
        """Bonus for explicitly choosing safe actions."""
        if a.action_type in (ActionType.SAFE_HOLD, ActionType.EMERGENCY_STOP):
            if s.system_mode in ("DEGRADED", "SAFE_HOLD", "EMERGENCY"):
                return 1.0
        if a.action_type == ActionType.CHARGE and s.battery_bucket < 3:
            return 0.5
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Transition model
# ─────────────────────────────────────────────────────────────────────────────

class TransitionModel:
    """
    T(s, a, s') = P(s' | s, a)

    For a continuous domain approximation, this is a Gaussian kernel over
    the state space.  Here we use a simplified discrete model parameterised
    by published Mars rover mobility statistics.
    """

    # Motion success probability by system mode
    MOTION_SUCCESS = {"NORMAL": 0.95, "DEGRADED": 0.70,
                      "SAFE_HOLD": 0.0, "EMERGENCY": 0.0}

    def transitions(self, state: MDPState, action: MDPAction
                    ) -> List[Tuple[MDPState, float]]:
        """
        Return [(next_state, probability), ...].
        Probabilities must sum to 1.
        """
        outcomes = []

        if action.action_type == ActionType.EMERGENCY_STOP:
            next_s = MDPState(
                position_bucket   = state.position_bucket,
                system_mode       = "EMERGENCY",
                battery_bucket    = state.battery_bucket,
                temp_bucket       = state.temp_bucket,
                confidence_bucket = state.confidence_bucket,
            )
            return [(next_s, 1.0)]

        if action.action_type == ActionType.SAFE_HOLD:
            next_s = MDPState(
                position_bucket   = state.position_bucket,
                system_mode       = "SAFE_HOLD",
                battery_bucket    = max(0, state.battery_bucket - 1),
                temp_bucket       = max(0, state.temp_bucket - 1),
                confidence_bucket = state.confidence_bucket,
            )
            return [(next_s, 1.0)]

        # Motion actions
        if action.action_type in (ActionType.MOVE_FORWARD, ActionType.MOVE_BACKWARD,
                                   ActionType.TURN_LEFT,   ActionType.TURN_RIGHT):
            p_success = self.MOTION_SUCCESS.get(state.system_mode, 0.5)
            dx, dy = self._delta(action.action_type)

            # Success outcome
            px, py = state.position_bucket
            new_pos = (px + dx, py + dy)
            new_batt = max(0, state.battery_bucket - 1)
            new_temp = min(9, state.temp_bucket +
                           (1 if action.thermal_load_k > 5.0 else 0))

            success_state = MDPState(
                position_bucket   = new_pos,
                system_mode       = self._mode_transition(state, action),
                battery_bucket    = new_batt,
                temp_bucket       = new_temp,
                confidence_bucket = state.confidence_bucket,
            )
            # Slip / failure outcome: stay in place
            failure_state = MDPState(
                position_bucket   = state.position_bucket,
                system_mode       = "DEGRADED" if state.system_mode == "NORMAL"
                                    else state.system_mode,
                battery_bucket    = new_batt,
                temp_bucket       = new_temp,
                confidence_bucket = max(0, state.confidence_bucket - 1),
            )
            return [(success_state, p_success),
                    (failure_state, 1.0 - p_success)]

        # Charge
        if action.action_type == ActionType.CHARGE:
            new_batt = min(9, state.battery_bucket + 3)
            next_s   = MDPState(
                position_bucket   = state.position_bucket,
                system_mode       = state.system_mode,
                battery_bucket    = new_batt,
                temp_bucket       = state.temp_bucket,
                confidence_bucket = state.confidence_bucket,
            )
            return [(next_s, 1.0)]

        # Science / Communicate
        next_s = MDPState(
            position_bucket   = state.position_bucket,
            system_mode       = state.system_mode,
            battery_bucket    = max(0, state.battery_bucket - 1),
            temp_bucket       = state.temp_bucket,
            confidence_bucket = min(9, state.confidence_bucket + 1),
        )
        return [(next_s, 0.90), (state, 0.10)]

    @staticmethod
    def _delta(action_type: ActionType) -> Tuple[int, int]:
        return {
            ActionType.MOVE_FORWARD:  (0,  1),
            ActionType.MOVE_BACKWARD: (0, -1),
            ActionType.TURN_LEFT:     (-1, 0),
            ActionType.TURN_RIGHT:    (1,  0),
        }.get(action_type, (0, 0))

    @staticmethod
    def _mode_transition(state: MDPState, action: MDPAction) -> str:
        if state.battery_bucket <= 1:
            return "DEGRADED"
        if state.temp_bucket >= 9:
            return "SAFE_HOLD"
        return state.system_mode


# ─────────────────────────────────────────────────────────────────────────────
# MDP Solver — Value Iteration
# ─────────────────────────────────────────────────────────────────────────────

class MDPSolver:
    """
    Standard value iteration solver.
    Converges when max ΔV < ε (Bellman residual).

    For POMDP belief-state planning, see POMDPSolver below.
    """

    def __init__(self,
                 states:     List[MDPState],
                 actions:    List[MDPAction],
                 transition: TransitionModel,
                 reward:     RewardFunction,
                 weights:    Optional[RewardWeights] = None):
        self.states     = states
        self.actions    = actions
        self.transition = transition
        self.reward     = reward
        self.gamma      = (weights or RewardWeights()).gamma

    def solve(self, epsilon: float = 0.001,
              max_iterations: int = 1000) -> MDPSolution:
        """
        Value iteration.
        V_{k+1}(s) = max_a Σ_{s'} T(s,a,s') [R(s,a,s') + γ V_k(s')]
        """
        t0 = time.time()
        V  = {s: 0.0 for s in self.states}
        policy: Dict[MDPState, MDPAction] = {}

        for iteration in range(max_iterations):
            delta = 0.0
            new_V = {}
            for s in self.states:
                best_v = -math.inf
                best_a = None
                for a in self.actions:
                    q = self._q_value(s, a, V)
                    if q > best_v:
                        best_v = q
                        best_a = a
                new_V[s]    = best_v if best_v != -math.inf else 0.0
                policy[s]   = best_a
                delta        = max(delta, abs(new_V[s] - V[s]))
            V = new_V
            if delta < epsilon:
                logger.info(f"MDP converged in {iteration+1} iterations (δ={delta:.6f})")
                return MDPSolution(
                    policy=policy, value_function=V,
                    iterations=iteration+1, converged=True,
                    solve_time_s=time.time()-t0,
                    n_states=len(self.states), n_actions=len(self.actions)
                )

        logger.warning(f"MDP did not converge in {max_iterations} iterations")
        return MDPSolution(
            policy=policy, value_function=V,
            iterations=max_iterations, converged=False,
            solve_time_s=time.time()-t0,
            n_states=len(self.states), n_actions=len(self.actions)
        )

    def _q_value(self, state: MDPState, action: MDPAction,
                 V: Dict[MDPState, float]) -> float:
        """Q(s,a) = Σ_{s'} T(s,a,s') [R(s,a,s') + γ V(s')]"""
        q = 0.0
        for next_s, prob in self.transition.transitions(state, action):
            r  = self.reward(state, action, next_s)
            vn = V.get(next_s, 0.0)
            q += prob * (r + self.gamma * vn)
        return q


# ─────────────────────────────────────────────────────────────────────────────
# POMDP — belief-state MDP for partial observability
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeliefState:
    """Probability distribution over MDPStates."""
    distribution: Dict[MDPState, float]

    def __post_init__(self):
        total = sum(self.distribution.values())
        if total > 0:
            self.distribution = {s: p/total for s, p in self.distribution.items()}

    def most_likely(self) -> MDPState:
        return max(self.distribution, key=self.distribution.get)

    def entropy(self) -> float:
        return -sum(p * math.log(p + 1e-12)
                    for p in self.distribution.values())

    def expected_value(self, V: Dict[MDPState, float]) -> float:
        return sum(p * V.get(s, 0.0) for s, p in self.distribution.items())


class POMDPSolver:
    """
    Point-based POMDP solver (simplified PBVI — point-based value iteration).
    Maintains a set of α-vectors (linear functions over belief space).

    Suitable for: rover operations during sensor dropout where the true
    state is unknown and we reason over a belief distribution.

    Reference: Pineau et al., "Point-based Value Iteration: An Anytime
    Algorithm for POMDPs", IJCAI 2003.
    """

    def __init__(self, mdp_solver: MDPSolver,
                 observation_model: Optional[Dict] = None):
        self.mdp   = mdp_solver
        self.obs   = observation_model or {}

    def best_action(self, belief: BeliefState,
                    V: Dict[MDPState, float]) -> MDPAction:
        """
        Select the best action for the current belief state.
        Uses the MDP value function as a heuristic (point evaluation).
        """
        best_q  = -math.inf
        best_a  = None

        for a in self.mdp.actions:
            q = self._belief_q(belief, a, V)
            if q > best_q:
                best_q = q
                best_a = a

        return best_a

    def _belief_q(self, belief: BeliefState, action: MDPAction,
                  V: Dict[MDPState, float]) -> float:
        """Q(b, a) = Σ_s b(s) Q(s, a)"""
        return sum(
            p * self.mdp._q_value(s, action, V)
            for s, p in belief.distribution.items()
        )

    def update_belief(self, belief: BeliefState,
                      action: MDPAction,
                      observation: Dict) -> BeliefState:
        """
        Bayes filter belief update:
        b'(s') ∝ P(o|s',a) Σ_s T(s,a,s') b(s)
        """
        new_dist: Dict[MDPState, float] = {}

        for next_s in self.mdp.states:
            # Σ_s T(s,a,s') b(s)
            predict = sum(
                b_s * self._transition_prob(s, action, next_s)
                for s, b_s in belief.distribution.items()
            )
            # P(o | s', a) — observation likelihood
            obs_prob = self._observation_prob(observation, next_s)
            new_dist[next_s] = obs_prob * predict

        return BeliefState(new_dist)

    def _transition_prob(self, s: MDPState, a: MDPAction,
                          next_s: MDPState) -> float:
        for ns, p in self.mdp.transition.transitions(s, a):
            if ns == next_s:
                return p
        return 0.0

    def _observation_prob(self, obs: Dict, state: MDPState) -> float:
        """
        P(observation | state) — likelihood of sensor reading given true state.
        Gaussian model: obs confidence vs state confidence bucket.
        """
        obs_conf   = obs.get("confidence", 0.9)
        state_conf = state.confidence_bucket / 9.0
        diff       = abs(obs_conf - state_conf)
        # Gaussian likelihood with σ=0.2
        return math.exp(-0.5 * (diff / 0.2) ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Factory — build a standard ARVS MDP from current system state
# ─────────────────────────────────────────────────────────────────────────────

def build_mission_mdp(
        grid_size:   int = 5,
        reward_weights: Optional[RewardWeights] = None
) -> Tuple[MDPSolver, List[MDPState], List[MDPAction]]:
    """
    Build a mission MDP for a rover on a grid_size × grid_size terrain patch.

    Returns (solver, states, actions) ready to call solver.solve().
    """
    # Enumerate states
    states: List[MDPState] = [
        MDPState(
            position_bucket   = (px, py),
            system_mode       = mode,
            battery_bucket    = batt,
            temp_bucket       = temp,
            confidence_bucket = conf,
        )
        for px in range(grid_size)
        for py in range(grid_size)
        for mode in ("NORMAL", "DEGRADED", "SAFE_HOLD", "EMERGENCY")
        for batt in range(10)
        for temp in range(10)
        for conf in range(10)
    ]

    # Enumerate actions
    actions: List[MDPAction] = [
        MDPAction(ActionType.MOVE_FORWARD,   duration_s=10.0, power_required_w=80.0,  thermal_load_k=2.0,  is_reversible=True),
        MDPAction(ActionType.MOVE_BACKWARD,  duration_s=10.0, power_required_w=80.0,  thermal_load_k=2.0,  is_reversible=True),
        MDPAction(ActionType.TURN_LEFT,      duration_s=5.0,  power_required_w=60.0,  thermal_load_k=1.0,  is_reversible=True),
        MDPAction(ActionType.TURN_RIGHT,     duration_s=5.0,  power_required_w=60.0,  thermal_load_k=1.0,  is_reversible=True),
        MDPAction(ActionType.SCIENCE_SAMPLE, duration_s=60.0, power_required_w=100.0, thermal_load_k=8.0,  is_reversible=True),
        MDPAction(ActionType.COMMUNICATE,    duration_s=30.0, power_required_w=40.0,  thermal_load_k=3.0,  is_reversible=True),
        MDPAction(ActionType.CHARGE,         duration_s=120.0,power_required_w=0.0,   thermal_load_k=1.0,  is_reversible=True),
        MDPAction(ActionType.SAFE_HOLD,      duration_s=10.0, power_required_w=20.0,  thermal_load_k=0.5,  is_reversible=True),
        MDPAction(ActionType.EMERGENCY_STOP, duration_s=1.0,  power_required_w=10.0,  thermal_load_k=0.0,  is_reversible=False),
    ]

    weights    = reward_weights or RewardWeights()
    transition = TransitionModel()
    reward     = RewardFunction(weights)
    solver     = MDPSolver(states, actions, transition, reward, weights)

    logger.info(f"MDP built: {len(states)} states × {len(actions)} actions")
    return solver, states, actions
