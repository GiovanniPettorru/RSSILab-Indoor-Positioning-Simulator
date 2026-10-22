"""Data model: materials, anchors, obstacles, wireless technologies, scenario.

Coordinates are in metres, with the room centred at (0, 0); heights (z) are
measured from the floor.
"""
from __future__ import annotations

import copy
import json
import math
import warnings
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Materials (Multi-Wall / Motley-Keenan model): loss per crossing, in dB.
# Indicative values at 0.868 / 2.4 / 5.0 / 6.5 GHz from the usual indoor
# literature (ITU-R P.1238/P.2040, COST 231). Calibrate with your measurements
# and edit freely: the table is the single source of truth for the simulator.
# ---------------------------------------------------------------------------
MATERIAL_FREQS_GHZ = (0.868, 2.4, 5.0, 6.5)
MATERIAL_TABLE = {
    "Drywall":   (2.0, 3.0, 4.0, 5.0),
    "Glass":     (1.5, 2.5, 4.0, 5.0),
    "Wood":      (2.5, 4.0, 6.0, 7.0),
    "Brick":     (5.0, 8.0, 12.0, 14.0),
    "Concrete":  (9.0, 13.0, 20.0, 24.0),
    "Metal":     (20.0, 26.0, 30.0, 32.0),
    "Furniture": (1.0, 2.0, 3.0, 3.5),
}
MATERIAL_COLORS = {
    "Drywall": "#b9b2a6", "Glass": "#7fc4d8", "Wood": "#a0703c",
    "Brick": "#b5523b", "Concrete": "#6e6e6e", "Metal": "#2f3b4c",
    "Furniture": "#c9a96b",
}


def material_loss(material: str, f_ghz: float) -> float:
    """Per-crossing loss (dB), log-frequency interpolation of MATERIAL_TABLE."""
    vals = MATERIAL_TABLE.get(material, MATERIAL_TABLE["Brick"])
    lf = np.log10(np.asarray(MATERIAL_FREQS_GHZ))
    return float(np.interp(math.log10(max(f_ghz, 1e-3)), lf, vals))


# ---------------------------------------------------------------------------
@dataclass
class Anchor:
    name: str
    x: float
    y: float
    enabled: bool = True
    z: float = 0.0


@dataclass
class Obstacle:
    """kind = 'wall' (segment with thickness, any angle) or 'rect' (axis-aligned block)."""
    kind: str
    x0: float
    y0: float
    x1: float
    y1: float
    material: str = "Brick"
    thickness: float = 0.2
    loss_db: Optional[float] = None      # None -> use MATERIAL_TABLE
    blocks_motion: bool = True
    z0: float = 0.0                      # bottom height (m)
    z1: Optional[float] = None           # top height (m), None = up to the ceiling

    def __post_init__(self):
        self.normalized()                # blocks: corners may be given in any order

    def loss(self, f_ghz: float) -> float:
        return self.loss_db if self.loss_db is not None else material_loss(self.material, f_ghz)

    def normalized(self) -> "Obstacle":
        if self.kind == "rect":
            self.x0, self.x1 = min(self.x0, self.x1), max(self.x0, self.x1)
            self.y0, self.y1 = min(self.y0, self.y1), max(self.y0, self.y1)
        return self

    def polygon(self) -> np.ndarray:
        """Corner points for drawing."""
        if self.kind == "rect":
            return np.array([[self.x0, self.y0], [self.x1, self.y0],
                             [self.x1, self.y1], [self.x0, self.y1]])
        d = np.array([self.x1 - self.x0, self.y1 - self.y0])
        n = np.linalg.norm(d)
        if n < 1e-9:
            d = np.array([1.0, 0.0]); n = 1.0
        nrm = np.array([-d[1], d[0]]) / n * self.thickness / 2
        a, b = np.array([self.x0, self.y0]), np.array([self.x1, self.y1])
        return np.array([a + nrm, b + nrm, b - nrm, a - nrm])


# ---------------------------------------------------------------------------
LEGACY_FADING = "Legacy (Rayleigh amplitude in dB)"
FADING_MODELS = ("None", "Rayleigh", "Rician", LEGACY_FADING)
SENSITIVITY_OFF = -150.0   # sensitivities below this are treated as "no cut-off"
LEGACY_PRESET = "Legacy log-normal model"
# names used by RSSILab <= 0.2 (still accepted when loading projects)
_OLD_FADING_NAMES = {"Legacy (receive_RSS.py)": LEGACY_FADING}
_OLD_PRESET_NAMES = {"Legacy (receive_RSS.py)": LEGACY_PRESET}


def _known_fields(cls, d: dict, what: str) -> dict:
    """Keeps the keys of d that are fields of the dataclass cls (warns about the others),
    so that projects written by newer versions still load."""
    names = {f.name for f in fields(cls)}
    extra = sorted(set(d) - names)
    if extra:
        warnings.warn(f"Ignoring unknown {what} fields: {', '.join(extra)}", stacklevel=3)
    return {k: v for k, v in d.items() if k in names}


@dataclass
class Technology:
    name: str = "Custom"
    freq_ghz: float = 2.4
    tx_power_dbm: float = 0.0
    gains_db: float = 0.0            # Gt + Gr - system losses
    p0_auto: bool = True             # P0 from free-space path loss at d0
    p0_dbm: float = -40.0            # RSS at d0 (used when p0_auto is False)
    d0_m: float = 1.0
    path_loss_exp: float = 3.0
    shadow_sigma_db: float = 4.0
    shadow_corr_m: float = 0.0       # 0 -> i.i.d. per sample (legacy model)
    fading: str = "Rayleigh"
    rician_k_db: float = 6.0
    legacy_fading_sigma: float = 2.0
    sensitivity_dbm: float = -95.0
    quantization_db: float = 1.0     # 0 -> no quantization
    sample_rate_hz: float = 10.0     # RSS readings per anchor per second
    samples_per_fix: int = 5         # readings averaged per position fix

    def __post_init__(self):
        self.fading = _OLD_FADING_NAMES.get(self.fading, self.fading)
        self.name = _OLD_PRESET_NAMES.get(self.name, self.name)

    @staticmethod
    def from_dict(d: dict) -> "Technology":
        return Technology(**_known_fields(Technology, d, "technology"))

    def p0(self) -> float:
        if not self.p0_auto:
            return self.p0_dbm
        fspl = 20 * math.log10(max(self.d0_m, 1e-3)) + 20 * math.log10(self.freq_ghz * 1e9) - 147.55
        return self.tx_power_dbm + self.gains_db - fspl

    def has_sensitivity(self) -> bool:
        return self.sensitivity_dbm > SENSITIVITY_OFF

    def nominal_range_m(self) -> float:
        """Free-space-of-walls range where the mean RSS hits the sensitivity."""
        if not self.has_sensitivity():
            return float("inf")
        return self.d0_m * 10 ** ((self.p0() - self.sensitivity_dbm) / (10 * self.path_loss_exp))


def _tech(**kw) -> Technology:
    return Technology(**kw)


# Indicative presets: starting points, not certified radio parameters.
# The legacy model is the channel of the original RSSILab dataset generator:
# P0 = -40 dBm at 1 m, n = 4, i.i.d. log-normal shadowing (3 dB) plus a
# Rayleigh-distributed term (sigma 2) added in dB, no sensitivity, no quantisation.
PRESETS = {
    LEGACY_PRESET: _tech(
        name=LEGACY_PRESET, freq_ghz=2.4, p0_auto=False, p0_dbm=-40.0,
        path_loss_exp=4.0, shadow_sigma_db=3.0, shadow_corr_m=0.0,
        fading=LEGACY_FADING, legacy_fading_sigma=2.0,
        sensitivity_dbm=-200.0, quantization_db=0.0, sample_rate_hz=10.0, samples_per_fix=1),
    "Wi-Fi 2.4 GHz": _tech(
        name="Wi-Fi 2.4 GHz", freq_ghz=2.437, tx_power_dbm=20.0, gains_db=0.0,
        path_loss_exp=3.0, shadow_sigma_db=4.0, shadow_corr_m=3.0, fading="Rician",
        rician_k_db=3.0, sensitivity_dbm=-90.0, quantization_db=1.0,
        sample_rate_hz=10.0, samples_per_fix=5),
    "Wi-Fi 5 GHz": _tech(
        name="Wi-Fi 5 GHz", freq_ghz=5.5, tx_power_dbm=20.0, path_loss_exp=3.3,
        shadow_sigma_db=5.0, shadow_corr_m=2.0, fading="Rician", rician_k_db=3.0,
        sensitivity_dbm=-88.0, quantization_db=1.0, sample_rate_hz=10.0, samples_per_fix=5),
    "Wi-Fi HaLow 868 MHz (802.11ah)": _tech(
        name="Wi-Fi HaLow 868 MHz (802.11ah)", freq_ghz=0.866, tx_power_dbm=14.0,
        path_loss_exp=2.8, shadow_sigma_db=5.0, shadow_corr_m=5.0, fading="Rician",
        rician_k_db=3.0, sensitivity_dbm=-100.0, quantization_db=1.0,
        sample_rate_hz=5.0, samples_per_fix=5),
    "BLE 2.4 GHz": _tech(
        name="BLE 2.4 GHz", freq_ghz=2.426, tx_power_dbm=0.0, gains_db=-5.0,
        path_loss_exp=2.5, shadow_sigma_db=4.0, shadow_corr_m=2.0, fading="Rayleigh",
        sensitivity_dbm=-95.0, quantization_db=1.0, sample_rate_hz=10.0, samples_per_fix=10),
    "Zigbee / 802.15.4": _tech(
        name="Zigbee / 802.15.4", freq_ghz=2.405, tx_power_dbm=3.0, path_loss_exp=2.8,
        shadow_sigma_db=4.0, shadow_corr_m=2.0, fading="Rayleigh",
        sensitivity_dbm=-97.0, quantization_db=1.0, sample_rate_hz=4.0, samples_per_fix=4),
    "LoRa 868 MHz": _tech(
        name="LoRa 868 MHz", freq_ghz=0.868, tx_power_dbm=14.0, path_loss_exp=2.7,
        shadow_sigma_db=6.0, shadow_corr_m=6.0, fading="Rician", rician_k_db=3.0,
        sensitivity_dbm=-137.0, quantization_db=1.0, sample_rate_hz=0.5, samples_per_fix=1),
    "UWB CH5 (RSS only)": _tech(
        name="UWB CH5 (RSS only)", freq_ghz=6.5, tx_power_dbm=-14.3, path_loss_exp=2.2,
        shadow_sigma_db=3.0, shadow_corr_m=1.5, fading="Rician", rician_k_db=6.0,
        sensitivity_dbm=-95.0, quantization_db=0.25, sample_rate_hz=20.0, samples_per_fix=5),
}


# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    room_w: float = 90.0
    room_h: float = 60.0
    area_cx: float = 0.0
    area_cy: float = 0.0
    area_w: float = 60.0
    area_h: float = 30.0
    anchor_h: float = 0.0        # default height of new anchors
    tag_h: float = 0.0           # 2D: known target height; 3D: height of new points
    dim: int = 2                 # 2 = estimate (x, y) with known height, 3 = estimate (x, y, z)
    room_z: float = 3.0          # ceiling height (full-height walls reach it)
    area_z0: float = 0.0         # 3D: vertical extent of the target area
    area_z1: float = 2.5
    anchors: List[Anchor] = field(default_factory=list)
    obstacles: List[Obstacle] = field(default_factory=list)
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    test_points: List[Tuple[float, float]] = field(default_factory=list)

    def room_bounds(self):
        return (-self.room_w / 2, -self.room_h / 2, self.room_w / 2, self.room_h / 2)

    def area_bounds(self):
        return (self.area_cx - self.area_w / 2, self.area_cy - self.area_h / 2,
                self.area_cx + self.area_w / 2, self.area_cy + self.area_h / 2)

    def area_bounds3(self):
        """(x0, y0, z0, x1, y1, z1); in 2D the vertical extent collapses on tag_h."""
        x0, y0, x1, y1 = self.area_bounds()
        z0, z1 = (self.area_z0, self.area_z1) if self.dim == 3 else (self.tag_h, self.tag_h)
        return (x0, y0, z0, x1, y1, z1)

    def anchor_xy(self, only_enabled: bool = True) -> np.ndarray:
        a = [(an.x, an.y) for an in self.anchors if an.enabled or not only_enabled]
        return np.array(a, dtype=float).reshape(-1, 2)

    def anchor_xyz(self, only_enabled: bool = True) -> np.ndarray:
        a = [(an.x, an.y, an.z) for an in self.anchors if an.enabled or not only_enabled]
        return np.array(a, dtype=float).reshape(-1, 3)

    def points3(self, pts) -> np.ndarray:
        """(N, 3) from a list of (x, y) or (x, y, z); z = tag_h in 2D or when missing."""
        out = np.empty((len(pts), 3))
        for i, p in enumerate(pts):
            z = p[2] if (len(p) > 2 and self.dim == 3) else self.tag_h
            out[i] = (p[0], p[1], z)
        return out

    def anchor_names(self, only_enabled: bool = True) -> List[str]:
        return [an.name for an in self.anchors if an.enabled or not only_enabled]

    def place_anchors_on_perimeter(self, n: int) -> None:
        """n anchors on the room walls, at the centres of equal segments, split
        between the sides in proportion to their length (opposite walls get the same
        number; with an odd n the last anchor goes to the top-left corner). With
        n = 12 in the 90 x 60 m room this gives the 12 anchors of the reference
        scenario."""
        n = int(n)
        w, h = self.room_w, self.room_h
        if n <= 1:
            self.anchors = [Anchor("a1", 0.0, h / 2, z=self.anchor_h)] if n == 1 else []
            return
        n_long = max(1, int(round(n * w / (2 * (w + h)))))
        n_short = max(0, (n - 2 * n_long) // 2)
        rest = n - 2 * n_long - 2 * n_short
        pts = []
        pts += [(-w / 2 + (i + 0.5) * w / n_long, h / 2) for i in range(n_long)]            # top
        pts += [(w / 2, h / 2 - (i + 0.5) * h / n_short) for i in range(n_short)]           # right
        pts += [(w / 2 - (i + 0.5) * w / n_long, -h / 2) for i in range(n_long)]            # bottom
        pts += [(-w / 2, -h / 2 + (i + 0.5) * h / n_short) for i in range(n_short)]         # left
        corners = [(-w / 2, h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, -h / 2)]
        pts += corners[:rest]
        self.anchors = [Anchor(f"a{i + 1}", float(x), float(y), z=self.anchor_h) for i, (x, y) in enumerate(pts)]

    def rename_anchors(self) -> None:
        for i, a in enumerate(self.anchors):
            a.name = f"a{i + 1}"

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["waypoints"] = [list(p) for p in self.waypoints]
        d["test_points"] = [list(p) for p in self.test_points]
        return d

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        d = copy.deepcopy(d)
        h = d.get("anchor_h", 0.0)                 # projects saved before per-anchor heights
        anchors = [Anchor(**_known_fields(Anchor, dict({"z": h}, **a), "anchor")) for a in d.pop("anchors", [])]
        obstacles = [Obstacle(**_known_fields(Obstacle, o, "obstacle")) for o in d.pop("obstacles", [])]
        wps = [tuple(p) for p in d.pop("waypoints", [])]
        tps = [tuple(p) for p in d.pop("test_points", [])]
        s = Scenario(**_known_fields(Scenario, d, "scenario"))
        s.anchors, s.obstacles, s.waypoints, s.test_points = anchors, obstacles, wps, tps
        return s


PROJECT_FORMAT = "rssilab-2"


def save_project(path: str, scn: Scenario, tech: Technology, settings: Optional[dict] = None) -> None:
    """Writes scenario, technology and (GUI) settings to a JSON project file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"format": PROJECT_FORMAT, "scenario": scn.to_dict(),
                   "technology": asdict(tech), "settings": settings or {}}, f, indent=2)


def load_project(path: str):
    """Returns (scenario, technology or None, settings dict) from a JSON project file."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or "scenario" not in d:
        raise ValueError("not an RSSILab project (no 'scenario' entry)")
    scn = Scenario.from_dict(d["scenario"])
    tech = Technology.from_dict(d["technology"]) if d.get("technology") else None
    return scn, tech, d.get("settings", {})
