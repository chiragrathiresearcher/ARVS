"""
planning/mission_planner.py
ARVS Mission Planner
=====================
Handles long-horizon mission objectives.  Bridges the gap between:
  - The MDP/POMDP decision model (tactical, per-step decisions)
  - The MVI execution controller (immediate action sequences)

Responsibilities
----------------
1. Maintain a mission goal stack (ordered by priority)
2. Decompose high-level goals into MVI-compatible action sequences
3. Path planning for mobile systems (A* on a 2-D terrain grid)
4. Resource allocation: energy budget per goal, time windows
5. Replanning when execution diverges from plan

The planner operates at a LOWER frequency (1 Hz or on-demand) than the
safety gate (100 Hz) and execution controller (10 Hz).  It produces plans;
it does NOT execute them.

Design decisions
----------------
* Goals are represented as a priority queue (highest priority first).
* Path planning uses A* with a real terrain cost map.
  When no terrain map is available, Manhattan distance heuristic.
* Energy allocation uses a simple greedy budget: each goal is allocated
  a fraction of the current battery budget proportional to its priority.
* All plans are annotated with worst-case energy cost so the safety gate
  can pre-screen them before execution begins.
"""

import heapq
import math
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Goal representation
# ─────────────────────────────────────────────────────────────────────────────

class GoalType(Enum):
    REACH_WAYPOINT     = auto()   # navigate to a position
    COLLECT_SAMPLE     = auto()   # science: collect at current position
    TRANSMIT_DATA      = auto()   # communicate with ground station
    RECHARGE           = auto()   # charge battery to target level
    SURVEY_AREA        = auto()   # coverage path in a bounding box
    SAFE_HOLD          = auto()   # wait until conditions improve
    EMERGENCY_RETURN   = auto()   # return to last safe position


@dataclass
class MissionGoal:
    goal_id:      str
    goal_type:    GoalType
    priority:     int           # higher = more urgent (1–10)
    target_pos:   Optional[Tuple[float, float]] = None   # m, x-y
    target_value: Optional[float] = None                 # e.g. battery target SoC
    deadline_s:   Optional[float] = None                 # mission time deadline
    energy_budget_wh: float = 10.0
    completed:    bool = False
    notes:        str  = ""

    def __lt__(self, other: "MissionGoal") -> bool:
        # Higher priority = pops first from min-heap (negate)
        return self.priority > other.priority


# ─────────────────────────────────────────────────────────────────────────────
# Planned action step
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PlanStep:
    step_id:      int
    action_type:  str
    parameters:   Dict
    duration_s:   float
    power_w:      float
    waypoint:     Optional[Tuple[float,float]] = None
    notes:        str = ""

    @property
    def energy_wh(self) -> float:
        return self.power_w * self.duration_s / 3600.0


@dataclass
class MissionPlan:
    plan_id:         str
    goal_id:         str
    steps:           List[PlanStep]
    total_duration_s: float
    total_energy_wh:  float
    worst_case_energy_wh: float   # for safety gate pre-screening
    created_at:      float
    is_valid:        bool = True
    invalidation_reason: str = ""

    def action_sequence(self) -> List[Dict]:
        """Export as list of ARVS Action dicts for the execution controller."""
        return [
            {
                "action_id":      f"{self.plan_id}_step{s.step_id}",
                "action_type":    s.action_type,
                "parameters":     s.parameters,
                "duration":       s.duration_s,
                "power_required": s.power_w,
                "max_torque":     s.parameters.get("max_torque", 60.0),
                "max_velocity":   s.parameters.get("max_velocity", 0.05),
                "thermal_load":   s.parameters.get("thermal_load", 2.0),
            }
            for s in self.steps
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Terrain map for path planning
# ─────────────────────────────────────────────────────────────────────────────

class TerrainMap:
    """
    2-D occupancy + cost grid.
    Cell cost represents traversal difficulty:
      0.0 = open flat terrain
      1.0 = passable but challenging (slope, loose regolith)
      inf = impassable (rock, crater)
    """

    def __init__(self, width: int, height: int, resolution_m: float = 1.0):
        self.width       = width
        self.height      = height
        self.resolution  = resolution_m
        self.cost_grid   = np.zeros((height, width), dtype=float)
        self.obstacle_grid = np.zeros((height, width), dtype=bool)

    def set_obstacle(self, x_m: float, y_m: float, radius_m: float = 1.0) -> None:
        cx, cy = self._world_to_grid(x_m, y_m)
        r      = int(math.ceil(radius_m / self.resolution))
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    gx, gy = cx+dx, cy+dy
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        self.obstacle_grid[gy, gx] = True
                        self.cost_grid[gy, gx]     = math.inf

    def set_cost(self, x_m: float, y_m: float, cost: float) -> None:
        cx, cy = self._world_to_grid(x_m, y_m)
        if 0 <= cx < self.width and 0 <= cy < self.height:
            self.cost_grid[cy, cx] = cost

    def traversal_cost(self, x: int, y: int) -> float:
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.cost_grid[y, x])
        return math.inf

    def is_free(self, x: int, y: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return not self.obstacle_grid[y, x]
        return False

    def _world_to_grid(self, x_m: float, y_m: float) -> Tuple[int, int]:
        return int(x_m / self.resolution), int(y_m / self.resolution)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.resolution, gy * self.resolution


# ─────────────────────────────────────────────────────────────────────────────
# A* path planner
# ─────────────────────────────────────────────────────────────────────────────

class AStarPlanner:
    """
    A* path planning on a TerrainMap.
    Heuristic: octile distance (8-connected grid).
    Returns a list of (x_m, y_m) waypoints.
    """

    MOVES_8 = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
    ]

    def __init__(self, terrain: TerrainMap):
        self.terrain = terrain

    def plan(self, start_m: Tuple[float,float],
             goal_m:  Tuple[float,float]) -> Optional[List[Tuple[float,float]]]:
        """Returns list of world-frame waypoints from start to goal, or None."""
        t = self.terrain
        sx, sy = t._world_to_grid(*start_m)
        gx, gy = t._world_to_grid(*goal_m)

        if not t.is_free(gx, gy):
            logger.warning(f"A*: goal ({gx},{gy}) is in obstacle")
            return None

        # f-score heap: (f, g, (x,y))
        open_heap: List[Tuple[float, float, Tuple[int,int]]] = []
        heapq.heappush(open_heap, (0.0, 0.0, (sx, sy)))

        came_from: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {(sx,sy): None}
        g_score:   Dict[Tuple[int,int], float] = {(sx,sy): 0.0}

        while open_heap:
            _, g_curr, curr = heapq.heappop(open_heap)

            if curr == (gx, gy):
                return self._reconstruct(came_from, curr)

            cx, cy = curr
            for dx, dy, step_cost in self.MOVES_8:
                nx, ny = cx+dx, cy+dy
                if not t.is_free(nx, ny):
                    continue
                terrain_cost = t.traversal_cost(nx, ny)
                if terrain_cost == math.inf:
                    continue
                new_g = g_curr + step_cost * (1.0 + terrain_cost)
                nb    = (nx, ny)
                if new_g < g_score.get(nb, math.inf):
                    g_score[nb]    = new_g
                    h              = self._heuristic(nx, ny, gx, gy)
                    came_from[nb]  = curr
                    heapq.heappush(open_heap, (new_g + h, new_g, nb))

        logger.warning("A*: no path found")
        return None

    def _heuristic(self, x: int, y: int, gx: int, gy: int) -> float:
        """Octile distance."""
        dx, dy = abs(x - gx), abs(y - gy)
        return max(dx, dy) + (1.414 - 1.0) * min(dx, dy)

    def _reconstruct(self, came_from: Dict, current: Tuple[int,int]
                     ) -> List[Tuple[float,float]]:
        path = []
        node: Optional[Tuple[int,int]] = current
        while node is not None:
            path.append(self.terrain.grid_to_world(*node))
            node = came_from[node]
        path.reverse()
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Mission Planner
# ─────────────────────────────────────────────────────────────────────────────

class MissionPlanner:
    """
    Long-horizon mission planner.

    Usage:
        planner = MissionPlanner(terrain_map)
        planner.add_goal(MissionGoal(...))
        plan = planner.plan_next(current_state)
        # feed plan.action_sequence() to ExecutionController
    """

    ROVER_SPEED_MS  = 0.05     # m/s — MSL/Perseverance typical traverse
    DRIVE_POWER_W   = 80.0
    SAMPLE_POWER_W  = 100.0
    COMM_POWER_W    = 40.0
    CHARGE_POWER_W  = 0.0      # RTG always on; charging = reducing load
    RTG_OUTPUT_W    = 110.0    # Perseverance MMRTG beginning-of-mission

    def __init__(self, terrain: Optional[TerrainMap] = None):
        self.terrain    = terrain or TerrainMap(50, 50, resolution_m=1.0)
        self.path_planner = AStarPlanner(self.terrain)
        self._goal_heap: List[MissionGoal] = []
        self._plan_counter = 0
        self._history:  List[MissionPlan] = []

    def add_goal(self, goal: MissionGoal) -> None:
        heapq.heappush(self._goal_heap, goal)
        logger.info(f"MissionPlanner: added goal {goal.goal_id} "
                    f"(type={goal.goal_type.name}, priority={goal.priority})")

    def plan_next(self, current_state: Dict) -> Optional[MissionPlan]:
        """
        Generate a MissionPlan for the highest-priority incomplete goal.
        Returns None if no goals remain or current state is EMERGENCY.
        """
        if current_state.get("system_mode", "NORMAL") == "EMERGENCY":
            logger.warning("MissionPlanner: EMERGENCY mode — no new plans")
            return None

        # Find highest-priority incomplete goal
        while self._goal_heap:
            goal = heapq.heappop(self._goal_heap)
            if not goal.completed:
                break
        else:
            logger.info("MissionPlanner: no goals remaining")
            return None

        plan = self._decompose(goal, current_state)
        self._plan_counter += 1
        self._history.append(plan)
        logger.info(f"MissionPlanner: plan {plan.plan_id} "
                    f"({len(plan.steps)} steps, "
                    f"{plan.total_energy_wh:.2f} Wh, "
                    f"{plan.total_duration_s:.0f}s)")
        return plan

    # ── Goal decomposition ────────────────────────────────────────────────────

    def _decompose(self, goal: MissionGoal,
                   state: Dict) -> MissionPlan:
        self._plan_counter += 1
        plan_id = f"plan_{self._plan_counter:04d}"
        steps:  List[PlanStep] = []
        sid = 0

        cur_pos = tuple(state.get("position", [0.0, 0.0])[:2])

        # ── REACH_WAYPOINT / SURVEY ───────────────────────────────────────────
        if goal.goal_type in (GoalType.REACH_WAYPOINT, GoalType.SURVEY_AREA,
                              GoalType.EMERGENCY_RETURN):
            target = goal.target_pos or (0.0, 0.0)
            path   = self.path_planner.plan(cur_pos, target)

            if path is None:
                # Fallback: straight line waypoints (no obstacle data)
                dx = target[0] - cur_pos[0]
                dy = target[1] - cur_pos[1]
                dist = math.hypot(dx, dy)
                n_steps = max(1, int(dist / 2.0))   # 2 m segments
                path = [
                    (cur_pos[0] + dx * i / n_steps,
                     cur_pos[1] + dy * i / n_steps)
                    for i in range(1, n_steps + 1)
                ]

            for waypoint in path:
                dx   = waypoint[0] - (cur_pos[0] if sid == 0 else
                                      steps[-1].waypoint[0])
                dy   = waypoint[1] - (cur_pos[1] if sid == 0 else
                                      steps[-1].waypoint[1])
                dist = math.hypot(dx, dy)
                dur  = dist / self.ROVER_SPEED_MS

                steps.append(PlanStep(
                    step_id    = sid,
                    action_type= "motion",
                    parameters = {
                        "direction": "forward",
                        "distance":  dist,
                        "max_torque":   60.0,
                        "max_velocity": self.ROVER_SPEED_MS,
                        "thermal_load": 2.0,
                    },
                    duration_s = dur,
                    power_w    = self.DRIVE_POWER_W,
                    waypoint   = waypoint,
                ))
                sid += 1

        # ── COLLECT_SAMPLE ────────────────────────────────────────────────────
        elif goal.goal_type == GoalType.COLLECT_SAMPLE:
            steps.append(PlanStep(
                step_id    = sid,
                action_type= "science_sample",
                parameters = {"sample_type": "core", "depth_mm": 50,
                              "max_torque": 80.0, "thermal_load": 8.0},
                duration_s = 120.0,
                power_w    = self.SAMPLE_POWER_W,
            )); sid += 1

        # ── TRANSMIT_DATA ─────────────────────────────────────────────────────
        elif goal.goal_type == GoalType.TRANSMIT_DATA:
            steps.append(PlanStep(
                step_id    = sid,
                action_type= "communicate",
                parameters = {"mode": "X-band", "data_volume_mb": 100,
                              "max_torque": 10.0, "thermal_load": 3.0},
                duration_s = 300.0,
                power_w    = self.COMM_POWER_W,
            )); sid += 1

        # ── RECHARGE ──────────────────────────────────────────────────────────
        elif goal.goal_type == GoalType.RECHARGE:
            target_soc = goal.target_value or 0.90
            cur_soc    = state.get("battery_level", 0.5)
            needed_wh  = (target_soc - cur_soc) * 43.2  # Perseverance 43.2 Wh
            net_w      = self.RTG_OUTPUT_W - 20.0        # idle load
            dur        = max(0.0, needed_wh / net_w * 3600.0)
            steps.append(PlanStep(
                step_id    = sid,
                action_type= "charge",
                parameters = {"target_soc": target_soc, "thermal_load": 0.5,
                              "max_torque": 0.0},
                duration_s = dur,
                power_w    = self.CHARGE_POWER_W,
            )); sid += 1

        # ── SAFE_HOLD ─────────────────────────────────────────────────────────
        elif goal.goal_type == GoalType.SAFE_HOLD:
            dur = goal.target_value or 600.0   # default 10-minute hold
            steps.append(PlanStep(
                step_id    = sid,
                action_type= "safe_hold",
                parameters = {"thermal_load": 0.5, "max_torque": 0.0},
                duration_s = dur,
                power_w    = 20.0,
            )); sid += 1

        # ── Compute totals ────────────────────────────────────────────────────
        total_dur    = sum(s.duration_s for s in steps)
        total_energy = sum(s.energy_wh  for s in steps)
        # Worst case: +20% energy contingency (ECSS standard)
        worst_energy = total_energy * 1.20

        return MissionPlan(
            plan_id              = plan_id,
            goal_id              = goal.goal_id,
            steps                = steps,
            total_duration_s     = total_dur,
            total_energy_wh      = total_energy,
            worst_case_energy_wh = worst_energy,
            created_at           = time.time(),
            is_valid             = True,
        )

    # ── Replanning trigger ────────────────────────────────────────────────────

    def invalidate_current_plan(self, plan: MissionPlan, reason: str) -> None:
        plan.is_valid = False
        plan.invalidation_reason = reason
        logger.warning(f"Plan {plan.plan_id} invalidated: {reason}")

    def replan(self, current_state: Dict,
               failed_plan: MissionPlan) -> Optional[MissionPlan]:
        """
        Re-add the goal with higher priority and generate a new plan.
        Called by the execution controller on divergence.
        """
        # Find the original goal
        for plan in self._history:
            if plan.plan_id == failed_plan.plan_id:
                logger.info(f"Replanning for goal {failed_plan.goal_id}")
                break

        # Check energy before replanning
        batt     = current_state.get("battery_level", 0.5)
        avail_wh = batt * 43.2 * 0.8   # 80% usable capacity

        new_goal = MissionGoal(
            goal_id      = failed_plan.goal_id + "_replan",
            goal_type    = GoalType.REACH_WAYPOINT,  # simplified recovery
            priority     = 8,                         # high priority
            energy_budget_wh = avail_wh,
            notes        = f"Replan after: {failed_plan.invalidation_reason}",
        )
        self.add_goal(new_goal)
        return self.plan_next(current_state)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "goals_pending":  len(self._goal_heap),
            "plans_generated": self._plan_counter,
            "history_depth":   len(self._history),
        }
