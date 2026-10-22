"""Geometry: vectorised intersections, multi-wall losses, occupancy grid, A* routing."""
from __future__ import annotations

import math
from heapq import heappop, heappush
from typing import List, Optional, Sequence

import numpy as np

from .grid_layouts import LAYOUT_NAMES, grid_layout
from .model import Obstacle, Scenario


# ---------------------------------------------------------------------------
# Segment / obstacle intersection (all inputs broadcastable numpy arrays)
# ---------------------------------------------------------------------------
def _z_range(ob: Obstacle, room_z: float):
    """Vertical extent of an obstacle (z1 = None means floor-to-ceiling)."""
    return ob.z0, (room_z if ob.z1 is None else ob.z1)


def seg_rect_hits(ax, ay, bx, by, rect, az=None, bz=None, zr=None) -> np.ndarray:
    """Liang-Barsky test: does segment A->B touch the axis-aligned rect
    (or, with az/bz/zr, the box rect x [z0, z1])?"""
    xmin, ymin, xmax, ymax = rect
    slabs = [(bx - ax, ax - xmin, xmax - ax), (by - ay, ay - ymin, ymax - ay)]
    shape_args = [ax, ay, bx, by]
    if az is not None:
        slabs.append((bz - az, az - zr[0], zr[1] - az))
        shape_args += [az, bz]
    t0 = np.zeros(np.broadcast(*shape_args).shape)
    t1 = np.ones_like(t0)
    ok = np.ones(t0.shape, dtype=bool)
    for d, qlo, qhi in slabs:
        for p, q in ((-d, qlo), (d, qhi)):
            p = np.broadcast_to(p, t0.shape)
            q = np.broadcast_to(q, t0.shape)
            par = p == 0
            ok &= ~(par & (q < 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(par, 0.0, q / np.where(par, 1.0, p))
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
    return ok & (t0 <= t1)


def seg_seg_hits(ax, ay, bx, by, cx, cy, dx_, dy_, return_t=False):
    """Crossing test of segments A->B (arrays) and C->D (scalars), in plan view.
    With return_t also gives the parameter t along A->B of the crossing."""
    rx, ry = bx - ax, by - ay
    sx, sy = dx_ - cx, dy_ - cy
    den = rx * sy - ry * sx
    qpx, qpy = cx - ax, cy - ay
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qpx * sy - qpy * sx) / den
        u = (qpx * ry - qpy * rx) / den
    hit = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    return (hit, t) if return_t else hit


def obstacle_hits(ax, ay, bx, by, ob: Obstacle, az=None, bz=None, room_z: float = 3.0) -> np.ndarray:
    """Does segment A->B cross the obstacle? With az/bz the test is in 3D:
    the link must pass below the top of the obstacle (z1) and above its bottom (z0)."""
    if ob.kind == "rect":
        zr = None if az is None else _z_range(ob, room_z)
        return seg_rect_hits(ax, ay, bx, by, (ob.x0, ob.y0, ob.x1, ob.y1), az, bz, zr)
    hit, t = seg_seg_hits(ax, ay, bx, by, ob.x0, ob.y0, ob.x1, ob.y1, return_t=True)
    if az is None:
        return hit
    z0, z1 = _z_range(ob, room_z)
    with np.errstate(invalid="ignore"):
        zc = az + t * (bz - az)
        return hit & (zc >= z0 - 1e-9) & (zc <= z1 + 1e-9)


def wall_loss_matrix(points: np.ndarray, anchors: np.ndarray, obstacles: Sequence[Obstacle],
                     f_ghz: float, room_z: float = 3.0) -> np.ndarray:
    """(P, A) total obstacle loss in dB along each anchor-point link (Multi-Wall).
    points / anchors are (N, 3): obstacle heights are taken into account."""
    P, A = len(points), len(anchors)
    loss = np.zeros((P, A))
    if not obstacles or P == 0 or A == 0:
        return loss
    px, py, pz = points[:, 0:1], points[:, 1:2], points[:, 2:3]
    qx, qy, qz = anchors[None, :, 0], anchors[None, :, 1], anchors[None, :, 2]
    for ob in obstacles:
        L = ob.loss(f_ghz)
        if L == 0:
            continue
        loss += L * obstacle_hits(px, py, qx, qy, ob, pz, qz, room_z)
    return loss


def point_seg_dist(px, py, ax, ay, bx, by) -> np.ndarray:
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / (L2 if L2 > 0 else 1.0), 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def inside_obstacles(points: np.ndarray, obstacles: Sequence[Obstacle], margin: float = 0.0,
                     motion_only: bool = True, room_z: float = 3.0) -> np.ndarray:
    """(P,) True where a point lies inside any obstacle grown by `margin`.
    (P, 2) points: plan view. (P, 3) points: the obstacle height is checked too,
    so a point above a desk is free."""
    out = np.zeros(len(points), dtype=bool)
    x, y = points[:, 0], points[:, 1]
    z = points[:, 2] if points.shape[1] > 2 else None
    for ob in obstacles:
        if motion_only and not ob.blocks_motion:
            continue
        if ob.kind == "rect":
            m = (x >= ob.x0 - margin) & (x <= ob.x1 + margin) & (y >= ob.y0 - margin) & (y <= ob.y1 + margin)
        else:
            m = point_seg_dist(x, y, ob.x0, ob.y0, ob.x1, ob.y1) <= ob.thickness / 2 + margin
        if z is not None:
            z0, z1 = _z_range(ob, room_z)
            m &= (z >= z0 - margin) & (z <= z1 + margin)
        out |= m
    return out


def crosses_obstacles(a: np.ndarray, b: np.ndarray, obstacles: Sequence[Obstacle],
                      room_z: float = 3.0) -> np.ndarray:
    """(N,) True where the move a->b crosses a motion-blocking obstacle
    ((N, 2) plan view, (N, 3) with obstacle heights)."""
    out = np.zeros(len(a), dtype=bool)
    three = a.shape[1] > 2
    for ob in obstacles:
        if ob.blocks_motion:
            if three:
                out |= obstacle_hits(a[:, 0], a[:, 1], b[:, 0], b[:, 1], ob, a[:, 2], b[:, 2], room_z)
            else:
                out |= obstacle_hits(a[:, 0], a[:, 1], b[:, 0], b[:, 1], ob)
    return out


# ---------------------------------------------------------------------------
# Occupancy grid over the target area (rows = y, cols = x)
# ---------------------------------------------------------------------------
class OccGrid:
    def __init__(self, bounds, res: float, obstacles: Sequence[Obstacle], clearance: float = 0.3):
        x0, y0, x1, y1 = bounds
        self.x0, self.y0 = x0, y0
        self.nx = max(2, int(round((x1 - x0) / res)))
        self.ny = max(2, int(round((y1 - y0) / res)))
        self.rx, self.ry = (x1 - x0) / self.nx, (y1 - y0) / self.ny
        xs = x0 + (np.arange(self.nx) + 0.5) * self.rx
        ys = y0 + (np.arange(self.ny) + 0.5) * self.ry
        X, Y = np.meshgrid(xs, ys)
        self.centers = np.column_stack([X.ravel(), Y.ravel()])
        # grow by half a cell diagonal so that thin walls never leak through
        grow = clearance + 0.71 * max(self.rx, self.ry)
        blocked = inside_obstacles(self.centers, obstacles, margin=grow)
        self.grid = blocked.reshape(self.ny, self.nx).astype(float)

    def to_cell(self, x, y):
        c = int(np.clip((x - self.x0) / self.rx, 0, self.nx - 1))
        r = int(np.clip((y - self.y0) / self.ry, 0, self.ny - 1))
        return r, c

    def to_xy(self, r, c):
        return (self.x0 + (c + 0.5) * self.rx, self.y0 + (r + 0.5) * self.ry)

    def is_free(self, r, c) -> bool:
        return self.grid[r, c] != 1

    def nearest_free(self, r, c):
        if self.is_free(r, c):
            return r, c
        rr, cc = np.nonzero(self.grid != 1)
        if len(rr) == 0:
            return r, c
        k = np.argmin((rr - r) ** 2 + (cc - c) ** 2)
        return int(rr[k]), int(cc[k])

    def los(self, p, q) -> bool:
        """Line of sight on the occupancy grid (used to smooth A* staircases)."""
        L = np.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(L / (0.4 * min(self.rx, self.ry))) + 1)
        t = np.linspace(0, 1, n)
        xs, ys = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
        cs = np.clip(((xs - self.x0) / self.rx).astype(int), 0, self.nx - 1)
        rs = np.clip(((ys - self.y0) / self.ry).astype(int), 0, self.ny - 1)
        return bool(np.all(self.grid[rs, cs] != 1))


def astar(grid: np.ndarray, start, goal):
    """A* on a 4-connected occupancy grid (cells != 1 are free), Euclidean heuristic.

    Returns (path, visited): path is the list of (row, col) cells from start to
    goal, or None when the goal cannot be reached."""
    start, goal = tuple(int(v) for v in start), tuple(int(v) for v in goal)
    rows, cols = grid.shape
    h = lambda a: math.hypot(a[0] - goal[0], a[1] - goal[1])  # noqa: E731
    open_set = [(h(start), start)]
    came, g = {}, {start: 0}
    visited = set()
    while open_set:
        _, cur = heappop(open_set)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1], list(visited)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nb = (cur[0] + dr, cur[1] + dc)
            if 0 <= nb[0] < rows and 0 <= nb[1] < cols and grid[nb] != 1:
                tg = g[cur] + 1
                if tg < g.get(nb, math.inf):
                    came[nb], g[nb] = cur, tg
                    heappush(open_set, (tg + h(nb), nb))
    return None, list(visited)


def smooth_path(pts: List[tuple], occ: OccGrid) -> List[tuple]:
    if len(pts) <= 2:
        return pts
    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = i + 1
        while j + 1 < len(pts) and occ.los(pts[i], pts[j + 1]):
            j += 1
        out.append(pts[j])
        i = j
    return out


def route_between(occ: OccGrid, a, b, warnings_out: Optional[list] = None) -> List[tuple]:
    """A* route between two points on the occupancy grid, smoothed.
    If no route exists (e.g. grid layout 14, walled goal) a straight segment is
    used and a message is appended to warnings_out."""
    s = occ.nearest_free(*occ.to_cell(*a))
    g = occ.nearest_free(*occ.to_cell(*b))
    path, _ = astar(occ.grid, s, g)
    if not path:
        if warnings_out is not None:
            warnings_out.append(f"No obstacle-free route from ({a[0]:.1f}, {a[1]:.1f}) to "
                                f"({b[0]:.1f}, {b[1]:.1f}): straight segment used.")
        return [tuple(a), tuple(b)]
    pts = [occ.to_xy(r, c) for r, c in path]
    a_ok = occ.is_free(*occ.to_cell(*a))
    b_ok = occ.is_free(*occ.to_cell(*b))
    pts = ([tuple(a)] if a_ok else []) + pts + ([tuple(b)] if b_ok else [])
    return smooth_path(pts, occ)


# ---------------------------------------------------------------------------
# Import of the 20 x 20 grid-world layouts (grid_layouts.py)
# ---------------------------------------------------------------------------
def grid_to_rects(grid: np.ndarray, skip_border: bool = True):
    """Merge occupied cells into rectangles (row runs, then vertical merge)."""
    rows, cols = grid.shape
    r0, r1 = (1, rows - 1) if skip_border else (0, rows)
    c0, c1 = (1, cols - 1) if skip_border else (0, cols)
    open_r, done = {}, []
    for r in range(r0, r1):
        runs, c = [], c0
        while c < c1:
            if grid[r, c] == 1:
                s = c
                while c < c1 and grid[r, c] == 1:
                    c += 1
                runs.append((s, c - 1))
            else:
                c += 1
        nxt = {}
        for run in runs:
            if run in open_r:
                rs, _ = open_r.pop(run)
                nxt[run] = (rs, r)
            else:
                nxt[run] = (r, r)
        done += [(rs, re, a, b) for (a, b), (rs, re) in open_r.items()]
        open_r = nxt
    done += [(rs, re, a, b) for (a, b), (rs, re) in open_r.items()]
    return done  # (row_start, row_end, col_start, col_end) inclusive


def import_grid_layout(scn: Scenario, level: int, material: str, onto_area: bool = True,
                       append: bool = False) -> str:
    """Turns grid layout `level` (1-14, see grid_layouts.py) into blocks of `material`
    scaled onto the target area (or the whole room); the border wall is skipped
    and start/goal become the tracking waypoints. Returns a short description."""
    grid, start, goal = grid_layout(level)
    rows, cols = grid.shape
    x0, y0, x1, y1 = scn.area_bounds() if onto_area else scn.room_bounds()
    cw, ch = (x1 - x0) / cols, (y1 - y0) / rows

    def cell_xy(r, c):  # row 0 is the top row
        return (x0 + (c + 0.5) * cw, y1 - (r + 0.5) * ch)

    if not append:
        scn.obstacles = []
    for rs, re, a, b in grid_to_rects(np.asarray(grid)):
        scn.obstacles.append(Obstacle("rect", x0 + a * cw, y1 - (re + 1) * ch,
                                      x0 + (b + 1) * cw, y1 - rs * ch, material=material))
    scn.waypoints = [cell_xy(*start), cell_xy(*goal)]
    return (f"Imported grid layout {int(level)} ({LAYOUT_NAMES[int(level)]}): "
            f"{len(scn.obstacles)} blocks, start and goal as waypoints")


import_grid_scenario = import_grid_layout      # name used by RSSILab <= 0.2
