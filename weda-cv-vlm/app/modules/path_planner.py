"""Potential Field path planner -- real-time obstacle-aware navigation.

Computes a smooth path from camera (origin) to a targeted object by:
  - Attractive field:  linear gradient pulling toward target
  - Repulsive field:   inverse-square pushing away from each obstacle
  - Path:              gradient descent from origin to target
  - Smoothing:         moving-average on waypoints

Performance:  < 2 ms for 20 obstacles, zero GPU, pure stdlib math.
Integration:  inserted as Stage 5.5 in pipeline (after spatial, before
              physical reasoning), only when a target track is active.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NavigationPlan:
    """Result of a potential-field path plan."""
    target_track_id: int = -1
    target_pos: Optional[tuple[float, float]] = None  # (x, z) world meters
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    path_length_m: float = 0.0
    obstacles_avoided: int = 0
    path_clear: bool = True


class PotentialFieldPlanner:
    """Gradient-descent path planner using attractive + repulsive fields.

    Parameters match YAML ``navigation:`` section in ``pipeline.yaml``.
    """

    def __init__(
        self,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 0.8,
        obstacle_radius: float = 0.5,
        step_size: float = 0.15,
        max_steps: int = 150,
        convergence_m: float = 0.2,
    ) -> None:
        self._att_gain = attractive_gain
        self._rep_gain = repulsive_gain
        self._obs_radius = obstacle_radius
        self._step_size = step_size
        self._max_steps = max_steps
        self._conv = convergence_m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        target_xz: tuple[float, float],
        obstacle_xzs: list[tuple[float, float]],
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> NavigationPlan:
        """Plan a path from *origin* (camera) to *target_xz* avoiding obstacles.

        All coordinates are in world meters (X = lateral, Z = depth).

        Returns a :class:`NavigationPlan` with waypoints and metadata.
        """
        nav = NavigationPlan(target_pos=target_xz)
        tx, tz = target_xz
        cx, cz = origin
        waypoints: list[tuple[float, float]] = [(cx, cz)]
        obstacles_avoided = 0

        for _ in range(self._max_steps):
            dx_goal = tx - cx
            dz_goal = tz - cz
            dist_goal = math.sqrt(dx_goal * dx_goal + dz_goal * dz_goal)
            if dist_goal < self._conv:
                break

            # --- Attractive force (unit direction toward target) ---
            inv_d = 1.0 / max(dist_goal, 0.01)
            fx_att = self._att_gain * dx_goal * inv_d
            fz_att = self._att_gain * dz_goal * inv_d

            # --- Repulsive force (inverse-square from obstacles within radius) ---
            fx_rep = 0.0
            fz_rep = 0.0
            _any_repulsion = False
            for ox, oz in obstacle_xzs:
                dx_obs = cx - ox
                dz_obs = cz - oz
                dist_obs = math.sqrt(dx_obs * dx_obs + dz_obs * dz_obs)
                if 0.01 < dist_obs < self._obs_radius:
                    strength = (
                        self._rep_gain
                        * (1.0 / dist_obs - 1.0 / self._obs_radius)
                        / (dist_obs * dist_obs)
                    )
                    inv_do = 1.0 / dist_obs
                    fx_rep += strength * dx_obs * inv_do
                    fz_rep += strength * dz_obs * inv_do
                    _any_repulsion = True
                    obstacles_avoided += 1

            # Break head-on symmetry: if obstacle is directly ahead (no lateral
            # repulsion component), add a small perpendicular nudge so the path
            # deflects left or right instead of stalling.
            if _any_repulsion and abs(fx_rep) < 0.01:
                fx_rep += 0.3 * self._rep_gain

            # --- Combined force ---
            fx = fx_att + fx_rep
            fz = fz_att + fz_rep
            mag = math.sqrt(fx * fx + fz * fz)
            if mag < 0.001:
                # Stuck in local minimum — apply lateral perturbation to escape
                fx += 0.3 * self._att_gain
                fz += 0.05 * self._att_gain
                mag = math.sqrt(fx * fx + fz * fz)
                if mag < 0.001:
                    nav.path_clear = False
                    break

            # Step along the field gradient
            inv_m = self._step_size / mag
            cx += fx * inv_m
            cz += fz * inv_m
            waypoints.append((cx, cz))

        # --- Smooth path (moving average, window = 3) ---
        if len(waypoints) > 4:
            smoothed = [waypoints[0]]
            for i in range(1, len(waypoints) - 1):
                sx = (waypoints[i - 1][0] + waypoints[i][0] + waypoints[i + 1][0]) / 3.0
                sz = (waypoints[i - 1][1] + waypoints[i][1] + waypoints[i + 1][1]) / 3.0
                smoothed.append((sx, sz))
            smoothed.append(waypoints[-1])
            waypoints = smoothed

        # --- Path length ---
        length = 0.0
        for i in range(1, len(waypoints)):
            dx = waypoints[i][0] - waypoints[i - 1][0]
            dz = waypoints[i][1] - waypoints[i - 1][1]
            length += math.sqrt(dx * dx + dz * dz)

        nav.waypoints = waypoints
        nav.path_length_m = length
        nav.obstacles_avoided = obstacles_avoided
        return nav
