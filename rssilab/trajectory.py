"""Experiment geometry: static test points (positioning) and routes (tracking).

Waypoint, random-walk and lawnmower routes avoid motion-blocking obstacles with
an A* planner on an occupancy grid (geometry.route_between); the circle route
does not (a warning is issued when a route crosses an obstacle)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import OccGrid, crosses_obstacles, inside_obstacles, route_between
from .model import Scenario

MODES = ("Positioning", "Tracking")
POINT_SOURCES = ("Points on map", "Random points", "Grid")
TRAJ_TYPES = ("Waypoints (click on map)", "Random walk", "Circle", "Lawnmower")


@dataclass
class TrajParams:
    mode: str = "Positioning"        # "Positioning" (static target) or "Tracking" (moving target)
    # positioning
    point_source: str = "Points on map"
    n_random_points: int = 10
    fixes_per_point: int = 10        # position estimates per point and per trial
    grid_spacing: float = 2.0
    grid_z: Optional[float] = None   # 3D: height of the evaluation grid (None = tag height)
    # tracking
    kind: str = TRAJ_TYPES[0]
    speed: float = 1.0               # m/s
    use_astar: bool = True
    astar_res: float = 0.5           # m
    clearance: float = 0.3           # m, distance kept from obstacles
    n_random_wp: int = 6
    circle_frac: float = 0.8
    circle_laps: float = 1.0
    lawn_spacing: float = 5.0

    @property
    def positioning(self) -> bool:
        return self.mode == "Positioning"


# ---------------------------------------------------------------------------
# Positioning: static test points
# ---------------------------------------------------------------------------
def grid_points(scn: Scenario, spacing: float, offset: float = 0.0, z: Optional[float] = None,
                z_layers: Optional[np.ndarray] = None):
    """Grid (N, 3) over the target area at height z (default tag height), or on
    several height layers; points inside obstacles are removed."""
    x0, y0, x1, y1 = scn.area_bounds()
    xs = np.arange(x0 + offset, x1 + 1e-9, spacing)
    ys = np.arange(y0 + offset, y1 + 1e-9, spacing)
    zs = np.atleast_1d(z_layers if z_layers is not None else (scn.tag_h if z is None else z))
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    mask = ~inside_obstacles(pts, scn.obstacles, margin=0.0, room_z=scn.room_z)
    return pts[mask], {"xs": xs, "ys": ys, "mask": mask, "z": float(zs[0])}


def random_points(scn: Scenario, n: int, rng: np.random.Generator, clearance: float = 0.3) -> np.ndarray:
    """n points (n, 3) uniformly distributed in the free part of the target area
    (also in height in 3D)."""
    x0, y0, z0, x1, y1, z1 = scn.area_bounds3()
    out = np.empty((0, 3))
    for _ in range(50):
        c = np.column_stack([rng.uniform(x0, x1, 4 * n), rng.uniform(y0, y1, 4 * n),
                             rng.uniform(z0, z1, 4 * n) if z1 > z0 else np.full(4 * n, z0)])
        c = c[~inside_obstacles(c, scn.obstacles, margin=clearance, room_z=scn.room_z)]
        out = np.vstack([out, c])
        if len(out) >= n:
            return out[:n]
    raise ValueError("Could not place the random points: the target area is almost fully occupied.")


def static_points(scn: Scenario, p: TrajParams, rng: np.random.Generator, warn: Optional[list] = None):
    """Test points (P, 3) and grid info (only for the Grid source)."""
    src = p.point_source
    if src == "Grid":
        gz = p.grid_z if (scn.dim == 3 and p.grid_z is not None) else scn.tag_h
        pts, info = grid_points(scn, p.grid_spacing, z=gz)
        if len(pts) == 0:
            raise ValueError("The grid has no free points: reduce the spacing.")
        return pts, info
    if src == "Random points":
        if p.n_random_points < 1:
            raise ValueError("Use at least 1 random point.")
        return random_points(scn, p.n_random_points, rng, p.clearance), None
    if not scn.test_points:
        raise ValueError("Add at least one test point (Test point tool, click on the map), "
                         "or choose Random points / Grid.")
    pts = scn.points3(scn.test_points)
    bad = inside_obstacles(pts, scn.obstacles, margin=0.0, room_z=scn.room_z)
    if warn is not None and bad.any():
        warn.append("Test points inside an obstacle: " +
                    ", ".join(f"P{i + 1}" for i in np.flatnonzero(bad)) + ".")
    return pts, None


# ---------------------------------------------------------------------------
# Tracking: routes
# ---------------------------------------------------------------------------
def _dedup(route: np.ndarray) -> np.ndarray:
    if len(route) < 2:
        return route
    keep = np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-9]
    return route[keep]


def _chain(occ, pts, use_astar, warn=None):
    """Polyline (K, 3) through the 3D points; with A* each leg is routed in plan
    view and the height varies linearly along the leg."""
    pts = np.asarray(pts, dtype=float)
    if not use_astar or occ is None:
        return pts
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        leg = np.array(route_between(occ, tuple(a[:2]), tuple(b[:2]), warn), dtype=float)
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(leg, axis=0).T))]
        frac = s / s[-1] if s[-1] > 0 else np.zeros_like(s)
        z = a[2] + frac * (b[2] - a[2])
        out += list(np.column_stack([leg, z])[1:])
    return np.array(out, dtype=float)


def build_route(scn: Scenario, p: TrajParams, rng: np.random.Generator, warn: Optional[list] = None) -> np.ndarray:
    """Polyline (K, 3) followed by the target. Routing problems go to `warn`."""
    x0, y0, x1, y1 = scn.area_bounds()
    occ = None
    if p.use_astar or p.kind == "Random walk":
        occ = OccGrid(scn.area_bounds(), p.astar_res, scn.obstacles, p.clearance)

    if p.kind.startswith("Waypoints"):
        if len(scn.waypoints) < 2:
            raise ValueError("Add at least 2 waypoints (Waypoint tool, click on the map).")
        route = _chain(occ, scn.points3(scn.waypoints), p.use_astar, warn)
    elif p.kind == "Random walk":
        free = occ.centers[occ.grid.ravel() != 1]
        if len(free) == 0:
            raise ValueError("The target area has no free cells.")
        pts = free[rng.integers(0, len(free), max(2, p.n_random_wp))]
        _, _, z0, _, _, z1 = scn.area_bounds3()
        zz = rng.uniform(z0, z1, len(pts)) if z1 > z0 else np.full(len(pts), z0)
        route = _chain(occ, np.column_stack([pts, zz]), True, warn)
    elif p.kind == "Circle":
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = p.circle_frac * (x1 - x0) / 2, p.circle_frac * (y1 - y0) / 2
        th = np.linspace(0, 2 * np.pi * p.circle_laps, int(180 * max(p.circle_laps, 0.1)) + 1)
        route = np.column_stack([cx + rx * np.cos(th), cy + ry * np.sin(th), np.full(len(th), scn.tag_h)])
    elif p.kind == "Lawnmower":
        m = max(p.clearance, 0.5)
        ys = np.arange(y0 + m, y1 - m + 1e-9, max(p.lawn_spacing, 0.5))
        pts = []
        for i, y in enumerate(ys):
            row = [(x0 + m, y, scn.tag_h), (x1 - m, y, scn.tag_h)]
            pts += row if i % 2 == 0 else row[::-1]
        route = _chain(occ, pts, p.use_astar, warn)
    else:
        raise ValueError(f"'{p.kind}' is not a route type.")
    route = _dedup(route)
    if warn is not None and scn.obstacles and len(route) > 1:
        cross = crosses_obstacles(route[:-1, :2], route[1:, :2], scn.obstacles)
        if cross.any():
            why = (" (the circle route ignores obstacles)" if p.kind == "Circle" else
                   " (A* routing is off)" if not p.use_astar and p.kind != "Random walk" else "")
            warn.append(f"The route crosses motion-blocking obstacles in {int(cross.sum())} "
                        f"segment(s){why}: the target walks through them.")
    return route


def sample_route(route: np.ndarray, speed: float, rate_hz: float, max_samples: int = 400_000):
    """Positions (S, 3) and times (S,) along the route at constant (3D) speed."""
    route = _dedup(np.asarray(route, dtype=float))
    seg = np.linalg.norm(np.diff(route, axis=0), axis=1) if len(route) > 1 else np.zeros(0)
    cum = np.r_[0.0, np.cumsum(seg)]
    total = cum[-1]
    step = max(speed, 1e-6) / max(rate_hz, 1e-6)
    n = int(total / step) + 1
    if n > max_samples:
        raise ValueError(f"The route needs {n} RSS samples (> {max_samples}). "
                         f"Lower the sample rate, raise the speed or shorten the route.")
    s = np.arange(n) * step
    pos = np.column_stack([np.interp(s, cum, route[:, k]) for k in range(route.shape[1])])
    return pos, s / max(speed, 1e-6)
