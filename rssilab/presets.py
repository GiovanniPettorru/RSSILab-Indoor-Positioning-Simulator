"""Ready-made scenarios and technology presets."""
from __future__ import annotations

import copy

from .model import _OLD_PRESET_NAMES, PRESETS, Anchor, Obstacle, Scenario, Technology

# Room, target area and anchors of the reference scenario (the layout of the
# original RSSILab dataset): 90 x 60 m room centred at (0, 0), 60 x 30 m target
# area, 12 anchors on the walls (= Scenario.place_anchors_on_perimeter(12)).
REFERENCE_ANCHORS = [(-33.75, 30.0), (-11.25, 30.0), (11.25, 30.0), (33.75, 30.0), (45.0, 15.0),
                     (45.0, -15.0), (33.75, -30.0), (11.25, -30.0), (-11.25, -30.0),
                     (-33.75, -30.0), (-45.0, -15.0), (-45.0, 15.0)]


def technology(name: str) -> Technology:
    """A fresh copy of a technology preset (see model.PRESETS)."""
    name = _OLD_PRESET_NAMES.get(name, name)
    if name not in PRESETS:
        raise KeyError(f"Unknown technology preset '{name}'. Available: {', '.join(PRESETS)}")
    return copy.deepcopy(PRESETS[name])


def default_scenario() -> Scenario:
    """Reference scenario: 90 x 60 m room, 60 x 30 m target area, 12 anchors on the walls."""
    s = Scenario(room_w=90.0, room_h=60.0, area_w=60.0, area_h=30.0)
    s.anchors = [Anchor(f"a{i + 1}", x, y, z=s.anchor_h) for i, (x, y) in enumerate(REFERENCE_ANCHORS)]
    s.waypoints = [(-25.0, -10.0), (25.0, -10.0), (25.0, 10.0), (-25.0, 10.0), (-25.0, -8.0)]
    s.test_points = [(0.0, 0.0), (-20.0, -8.0), (-10.0, 8.0), (15.0, -6.0), (25.0, 12.0)]
    return s


def demo_office() -> Scenario:
    """Reference room with an office floor plan: corridor, rooms with doors, racks."""
    s = default_scenario()
    W = lambda x0, y0, x1, y1, m, t=0.2: Obstacle("wall", x0, y0, x1, y1, material=m, thickness=t)  # noqa: E731
    s.obstacles = [
        # corridor (y = +-4), one 3 m door per room
        W(-30, 4, -24, 4, "Brick", 0.3), W(-21, 4, -9, 4, "Brick", 0.3), W(-6, 4, 6, 4, "Brick", 0.3),
        W(9, 4, 21, 4, "Brick", 0.3), W(24, 4, 30, 4, "Brick", 0.3),
        W(-30, -4, -22, -4, "Brick", 0.3), W(-19, -4, -1.5, -4, "Brick", 0.3),
        W(1.5, -4, 18, -4, "Brick", 0.3), W(21, -4, 30, -4, "Brick", 0.3),
        # partitions
        W(-15, 4, -15, 15, "Drywall"), W(0, 4, 0, 15, "Glass", 0.05), W(15, 4, 15, 15, "Drywall"),
        W(-10, -15, -10, -4, "Concrete", 0.3), W(10, -15, 10, -4, "Concrete", 0.3),
        # furniture / racks
        Obstacle("rect", 20, -13, 27, -11, material="Metal"),
        Obstacle("rect", -27, 8, -22, 11, material="Furniture"),
        Obstacle("rect", 4, 9, 10, 11, material="Wood"),
    ]
    s.waypoints = [(-28.0, 0.0), (-8.0, 10.0), (8.0, 12.5), (25.0, -8.0), (-20.0, -10.0), (-28.0, 0.0)]
    s.test_points = [(-25.0, 0.0), (-8.0, 10.0), (8.0, 12.5), (22.0, 10.0), (-20.0, -10.0),
                     (0.0, -10.0), (24.0, -7.0)]
    return s


def small_room() -> Scenario:
    """20 x 15 m room, 4 anchors at the centres of the walls."""
    s = Scenario(room_w=20.0, room_h=15.0, area_w=18.0, area_h=13.0)
    s.place_anchors_on_perimeter(4)
    s.waypoints = [(-7.0, -5.0), (7.0, -5.0), (7.0, 5.0), (-7.0, 5.0), (-7.0, -4.0)]
    s.test_points = [(0.0, 0.0), (-6.0, -4.0), (6.0, 4.0), (5.0, -4.5)]
    return s


SCENARIO_PRESETS = {
    "Reference 90 x 60 m, 12 anchors": default_scenario,
    "Demo office (walls, doors, racks)": demo_office,
    "Small room 20 x 15 m": small_room,
}
# names used by RSSILab <= 0.2
OLD_SCENARIO_PRESET_NAMES = {"Default (dataset_generation.py)": "Reference 90 x 60 m, 12 anchors"}
