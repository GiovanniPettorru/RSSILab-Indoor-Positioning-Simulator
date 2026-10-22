"""Grid-world layouts: 14 obstacle maps on a 20 x 20 cell grid.

Each layout is a 0/1 occupancy grid (row 0 is the top row, 1 = obstacle) with a
border wall, a start cell and a goal cell given as (row, col). The layouts are
classic path-planning test cases (open field, U-trap, narrow passage, maze,
spiral, walled goal, ...). In RSSILab they are imported as obstacle maps
(geometry.import_grid_layout): the occupied cells become blocks scaled onto the
target area (or the whole room) and start/goal become tracking waypoints.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

GRID_SIZE = 20
DEFAULT_START = (1, 1)
DEFAULT_GOAL = (18, 18)

LAYOUT_NAMES: Dict[int, str] = {
    1: "Open Field", 2: "The Wall", 3: "Smart City", 4: "U-Trap", 5: "Narrow Passage",
    6: "Dual Path", 7: "Maze", 8: "Random Debris", 9: "Diagonal", 10: "The Room",
    11: "The Bunker", 12: "The Spiral", 13: "Checkerboard", 14: "Mission Impossible",
}


def grid_layout(level: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Returns (grid (20, 20) of 0/1, start (row, col), goal (row, col)).

    Level 14 has no route from start to goal (the goal is walled in): it is
    meant to exercise the 'no route' handling of the route planner.
    """
    level = int(level)
    if level not in LAYOUT_NAMES:
        raise ValueError(f"Unknown grid layout {level}: use 1..{len(LAYOUT_NAMES)}.")
    g = np.zeros((GRID_SIZE, GRID_SIZE))
    g[0, :] = g[-1, :] = 1                     # border walls
    g[:, 0] = g[:, -1] = 1
    start, goal = DEFAULT_START, DEFAULT_GOAL

    if level == 2:                             # The Wall
        g[8:14, 10] = 1
    elif level == 3:                           # Smart City
        g[5:10, 3] = 1
        g[12, 5:15] = 1
        g[2:6, 12] = 1
        g[15:19, 8] = 1
        g[8:10, 15:18] = 1
    elif level == 4:                           # U-Trap, opening towards the top
        g[5:15, 12] = 1
        g[5:15, 8] = 1
        g[15, 8:13] = 1
        start, goal = (1, 1), (18, 10)
    elif level == 5:                           # Narrow Passage
        g[10, 0:15] = 1
        g[10, 16:20] = 1
        start, goal = (1, 10), (18, 10)
    elif level == 6:                           # Dual Path
        g[4:16, 6] = 1
        g[4:16, 14] = 1
        g[8, 6:10] = 1
        start, goal = (10, 2), (10, 18)
    elif level == 7:                           # Maze
        for r in (2, 6, 10, 14):
            g[r, 2:18] = 1
        g[2, 2] = g[6, 17] = g[10, 2] = g[14, 17] = 0
    elif level == 8:                           # Random Debris (fixed seed, same cells every time)
        rs = np.random.RandomState(42)
        for _ in range(35):
            rx, ry = rs.randint(1, 19), rs.randint(1, 19)
            if (rx, ry) != start and (rx, ry) != goal:
                g[rx, ry] = 1
    elif level == 9:                           # Diagonal
        for i in range(3, 17):
            g[i, i] = 1
        g[9, 9] = 0
        start, goal = (1, 18), (18, 1)
    elif level == 10:                          # The Room, one door
        g[13, 13:] = 1
        g[13:, 13] = 1
        g[16, 13] = 0
        start, goal = (2, 2), (17, 17)
    elif level in (11, 14):                    # The Bunker (11) / walled goal (14)
        g[6:13, 6] = 1
        g[6:13, 13] = 1
        g[6, 6:14] = 1
        g[13, 6:14] = 1
        if level == 11:
            g[13, 10] = 0                      # the only door
        start, goal = (1, 1), (10, 10)
    elif level == 12:                          # The Spiral
        g[2, 2:18] = 1
        g[17, 2:18] = 1
        g[2:18, 2] = 1
        g[2:18, 17] = 1
        g[4, 4:16] = 1
        g[15, 4:16] = 1
        g[4:16, 4] = 1
        g[4:16, 15] = 1
        g[2, 10] = g[15, 12] = g[4, 8] = 0     # staggered openings
        start, goal = (0, 10), (10, 10)
    elif level == 13:                          # Checkerboard
        for r in range(3, 17, 2):
            for c in range(3, 17, 2):
                g[r, c] = 1
        start, goal = (1, 1), (18, 18)
    return g, start, goal
