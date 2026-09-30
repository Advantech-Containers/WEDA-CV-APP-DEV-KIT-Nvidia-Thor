"""Tests for the Potential Field path planner."""

import math
import pytest
from app.modules.path_planner import PotentialFieldPlanner, NavigationPlan


class TestPotentialFieldPlanner:
    def _planner(self, **kw):
        defaults = dict(
            attractive_gain=1.0,
            repulsive_gain=0.8,
            obstacle_radius=0.5,
            step_size=0.15,
            max_steps=200,
            convergence_m=0.2,
        )
        defaults.update(kw)
        return PotentialFieldPlanner(**defaults)

    def test_straight_line_no_obstacles(self):
        """Path to target with no obstacles should be roughly a straight line."""
        p = self._planner()
        nav = p.plan(target_xz=(0.0, 5.0), obstacle_xzs=[])
        assert isinstance(nav, NavigationPlan)
        assert nav.path_clear is True
        assert nav.obstacles_avoided == 0
        assert len(nav.waypoints) > 2
        # Path length should be close to Euclidean distance (5.0m)
        assert abs(nav.path_length_m - 5.0) < 1.0
        # All waypoints should stay near x=0 (straight ahead)
        for wx, wz in nav.waypoints:
            assert abs(wx) < 0.5

    def test_avoids_obstacle(self):
        """Path should route around a single obstacle placed between origin and target."""
        p = self._planner()
        # Obstacle at (0.0, 2.5) — directly in path from (0,0) to (0,5)
        nav = p.plan(target_xz=(0.0, 5.0), obstacle_xzs=[(0.0, 2.5)])
        assert nav.path_clear is True
        assert nav.obstacles_avoided > 0
        # Path should deviate laterally to avoid the obstacle
        max_lateral = max(abs(wx) for wx, _ in nav.waypoints)
        assert max_lateral > 0.05  # must move sideways

    def test_multiple_obstacles(self):
        """Path should navigate around multiple obstacles."""
        p = self._planner()
        obstacles = [(0.0, 2.0), (0.0, 4.0), (0.3, 3.0)]
        nav = p.plan(target_xz=(0.0, 6.0), obstacle_xzs=obstacles)
        assert len(nav.waypoints) > 5
        assert nav.path_length_m > 6.0  # longer than straight line due to avoidance

    def test_lateral_target(self):
        """Path to a target that's to the right, not straight ahead."""
        p = self._planner()
        nav = p.plan(target_xz=(3.0, 3.0), obstacle_xzs=[])
        assert nav.path_clear is True
        # Final waypoint should be near the target
        last = nav.waypoints[-1]
        dist_to_target = math.sqrt((last[0] - 3.0) ** 2 + (last[1] - 3.0) ** 2)
        assert dist_to_target < 0.5

    def test_close_target(self):
        """Very close target should produce a short path."""
        p = self._planner()
        nav = p.plan(target_xz=(0.0, 0.3), obstacle_xzs=[])
        assert nav.path_length_m < 1.0
        assert len(nav.waypoints) >= 2

    def test_empty_obstacles(self):
        """Empty obstacle list should not crash."""
        p = self._planner()
        nav = p.plan(target_xz=(1.0, 4.0), obstacle_xzs=[])
        assert nav.path_clear is True

    def test_navigation_plan_defaults(self):
        """NavigationPlan should have sensible defaults."""
        nav = NavigationPlan()
        assert nav.target_track_id == -1
        assert nav.target_pos is None
        assert nav.waypoints == []
        assert nav.path_length_m == 0.0
        assert nav.obstacles_avoided == 0
        assert nav.path_clear is True

    def test_custom_gains(self):
        """Custom attractive/repulsive gains should not crash."""
        p = self._planner(attractive_gain=2.0, repulsive_gain=1.5, obstacle_radius=1.0)
        nav = p.plan(target_xz=(0.0, 5.0), obstacle_xzs=[(0.0, 2.5)])
        assert nav.path_clear is True
        assert len(nav.waypoints) > 2

    def test_path_smoothed(self):
        """Path with > 4 waypoints should be smoothed (moving average)."""
        p = self._planner(step_size=0.1)  # smaller steps = more waypoints
        nav = p.plan(target_xz=(0.0, 5.0), obstacle_xzs=[(0.0, 2.5)])
        # With step_size=0.1, distance 5m → at least 30+ waypoints before smoothing
        assert len(nav.waypoints) > 10
