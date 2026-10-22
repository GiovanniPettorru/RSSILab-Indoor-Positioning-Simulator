"""Channel model.

RSS = P0 - 10 n log10(d/d0) - sum(wall losses) + shadowing + fading,
then receiver quantisation and sensitivity cut-off (missing reading = NaN).

Shadowing:
  * shadow_corr_m == 0: i.i.d. Gaussian per sample (legacy model);
  * shadow_corr_m  > 0: static, spatially correlated Gaussian field per anchor
    (correlation 1/e at shadow_corr_m), so the same place always sees the same
    shadowing - this is what makes fingerprinting beat model-based methods.

Fading (per sample, in dB):
  * Rayleigh / Rician: power of a unit-mean channel, 10 log10(|h|^2);
  * Legacy: sigma * sqrt(-2 ln U), as in the original RSSILab dataset generator.
    That term is a Rayleigh-distributed *amplitude* added in dB: always positive,
    mean sigma*sqrt(pi/2) (= +2.51 dB for sigma = 2). Kept for reproducibility.
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache

import numpy as np

from .geometry import wall_loss_matrix
from .model import Technology

LN10 = math.log(10.0)
_EULER = 0.5772156649015329


def distances(points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """(P, A) 3D distances between points (P, 3) and anchors (A, 3)."""
    d = points[:, None, :3] - anchors[None, :, :3]
    return np.sqrt((d * d).sum(-1))


def mean_rss(points, anchors, obstacles, tech: Technology, use_walls=True,
             p0=None, n=None, room_z: float = 3.0) -> np.ndarray:
    """(P, A) deterministic part of the RSS (path loss + walls), points and anchors in 3D."""
    p0 = tech.p0() if p0 is None else p0
    n = tech.path_loss_exp if n is None else n
    d = np.maximum(distances(points, anchors), 0.1 * tech.d0_m)
    rss = p0 - 10.0 * n * np.log10(d / tech.d0_m)
    if use_walls and obstacles:
        rss = rss - wall_loss_matrix(points, anchors, obstacles, tech.freq_ghz, room_z)
    return rss


# ---------------------------------------------------------------------------
def fading_db(tech: Technology, rng: np.random.Generator, shape) -> np.ndarray:
    m = tech.fading
    if m == "Rayleigh":
        return 10 * np.log10(np.maximum(rng.exponential(1.0, shape), 1e-12))
    if m == "Rician":
        K = 10 ** (tech.rician_k_db / 10)
        los = math.sqrt(K / (K + 1))
        s = math.sqrt(1 / (2 * (K + 1)))
        h_re = los + s * rng.standard_normal(shape)
        h_im = s * rng.standard_normal(shape)
        return 10 * np.log10(np.maximum(h_re ** 2 + h_im ** 2, 1e-12))
    if m.startswith("Legacy"):
        u = 1.0 - rng.random(shape)            # (0, 1]: avoids log(0)
        return tech.legacy_fading_sigma * np.sqrt(-2.0 * np.log(u))
    if m != "None":
        raise ValueError(f"Unknown fading model '{m}': use one of None, Rayleigh, Rician, Legacy.")
    return np.zeros(shape)


def _log_i0(z: np.ndarray) -> np.ndarray:
    """log of the modified Bessel function I0, without overflow for large z."""
    small = z < 600
    out = np.empty_like(z)
    out[small] = np.log(np.i0(z[small]))
    zl = z[~small]
    out[~small] = zl - 0.5 * np.log(2 * math.pi * zl) + np.log1p(1 / (8 * zl) + 9 / (128 * zl * zl))
    return out


@lru_cache(maxsize=64)
def _rician_stats(k_db: float):
    """Exact mean and std (dB) of 10 log10 |h|^2 for a unit-mean Rician channel,
    by numerical integration of the density of ln |h|^2."""
    K = 10 ** (k_db / 10)
    v, mu2 = 1 / (K + 1), K / (K + 1)          # scattered power, line-of-sight power
    u = np.linspace(-50.0, 5.0, 400_001)       # u = ln |h|^2
    x = np.exp(u)
    logf = u - math.log(v) - (x + mu2) / v + _log_i0(2 * np.sqrt(mu2 * x) / v)
    w = np.exp(logf - logf.max())
    w /= w.sum()
    db = (10 / LN10) * u
    m = float(w @ db)
    return m, float(np.sqrt(w @ (db - m) ** 2))


def fading_stats(tech: Technology):
    """(mean, std) in dB of one fading sample."""
    m = tech.fading
    if m == "Rayleigh":
        return -10 * _EULER / LN10, (math.pi / math.sqrt(6)) * 10 / LN10
    if m == "Rician":
        return _rician_stats(round(tech.rician_k_db, 3))
    if m.startswith("Legacy"):
        s = tech.legacy_fading_sigma
        return s * math.sqrt(math.pi / 2), s * math.sqrt(2 - math.pi / 2)
    return 0.0, 0.0


def effective_sigma(tech: Technology) -> float:
    """Std (dB) of the per-fix averaged RSS around its calibrated mean."""
    _, sf = fading_stats(tech)
    ns = max(1, int(tech.samples_per_fix))
    if tech.shadow_corr_m > 0:      # static shadowing does not average out
        return math.sqrt(tech.shadow_sigma_db ** 2 + sf ** 2 / ns)
    return math.sqrt((tech.shadow_sigma_db ** 2 + sf ** 2) / ns)


# ---------------------------------------------------------------------------
class ShadowingField:
    """Static Gaussian random field per anchor (FFT-filtered white noise) with
    Gaussian correlation exp(-r^2 / corr^2), i.e. 1/e at the correlation distance.

    The field covers `bounds` (the region where the target can be) on a grid with
    8 nodes per correlation distance, coarser only when that would exceed
    `max_cells`, and is interpolated bilinearly. The interpolated values are
    rescaled by the exact variance of the bilinear interpolation, so the marginal
    standard deviation is `sigma` everywhere; when the grid is coarser than the
    correlation distance allows, `note` explains how the correlation is smoothed."""

    def __init__(self, bounds, n_anchors: int, sigma: float, corr: float,
                 rng: np.random.Generator, max_cells: int = 250_000):
        x0, y0, x1, y1 = bounds
        pad = 3 * corr                           # no wrap-around correlation across the FFT period
        self.x0, self.y0 = x0 - pad, y0 - pad
        W, H = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad
        res = max(corr / 8, 0.1)
        while (W / res) * (H / res) > max_cells:
            res *= 1.25
        self.res = res
        self.nx, self.ny = int(math.ceil(W / res)) + 1, int(math.ceil(H / res)) + 1
        sk = (corr / 2) / res                    # kernel std in cells -> corr(corr) = 1/e
        self.note = ""
        if res > corr / 3:
            self.note = (f"Correlated shadowing: the field is sampled every {res:.2f} m, coarser than the "
                         f"correlation distance ({corr:g} m) can resolve over this region; correlations below "
                         f"~{res:.1f} m are smoothed (the standard deviation is preserved).")
        kx = np.fft.rfftfreq(self.nx)[None, :]
        ky = np.fft.fftfreq(self.ny)[:, None]
        Hf = np.exp(-2 * (math.pi ** 2) * sk ** 2 * (kx ** 2 + ky ** 2))
        self.fields = np.empty((n_anchors, self.ny, self.nx), dtype=np.float32)
        rx, ry = [], []
        for a in range(n_anchors):
            w = rng.standard_normal((self.ny, self.nx))
            f = np.fft.irfft2(np.fft.rfft2(w) * Hf, s=w.shape)
            f -= f.mean()
            f /= max(f.std(), 1e-12)
            rx.append(np.mean(f[:, :-1] * f[:, 1:]))       # correlation between neighbouring nodes
            ry.append(np.mean(f[:-1, :] * f[1:, :]))
            self.fields[a] = f * sigma
        self.rho_x = float(np.clip(np.mean(rx), 0.0, 1.0)) if rx else 1.0
        self.rho_y = float(np.clip(np.mean(ry), 0.0, 1.0)) if ry else 1.0

    def sample(self, points: np.ndarray) -> np.ndarray:
        """(P, A) bilinear interpolation of each anchor's field."""
        gx = np.clip((points[:, 0] - self.x0) / self.res, 0, self.nx - 1.001)
        gy = np.clip((points[:, 1] - self.y0) / self.res, 0, self.ny - 1.001)
        i, j = gy.astype(int), gx.astype(int)
        fy, fx = (gy - i)[None, :], (gx - j)[None, :]
        F = self.fields
        v = (F[:, i, j] * (1 - fx) * (1 - fy) + F[:, i, j + 1] * fx * (1 - fy)
             + F[:, i + 1, j] * (1 - fx) * fy + F[:, i + 1, j + 1] * fx * fy)
        # variance of a bilinear blend of correlated nodes (separable Gaussian correlation)
        vx = (1 - fx) ** 2 + fx ** 2 + 2 * self.rho_x * fx * (1 - fx)
        vy = (1 - fy) ** 2 + fy ** 2 + 2 * self.rho_y * fy * (1 - fy)
        return (v / np.sqrt(vx * vy)).T


class Environment:
    """One realisation of the propagation environment (static shadowing).

    The shadowing field varies in the horizontal plane (x, y) only and covers
    `bounds` (x0, y0, x1, y1): by default the target area; pass the bounding box
    of all the positions that will be measured when some lie outside it."""

    def __init__(self, scn, tech: Technology, rng: np.random.Generator, bounds=None):
        self.scn, self.tech = scn, tech
        self.anchors = scn.anchor_xyz()
        self.field = None
        if tech.shadow_corr_m > 0 and tech.shadow_sigma_db > 0 and len(self.anchors):
            self.field = ShadowingField(bounds or scn.area_bounds(), len(self.anchors),
                                        tech.shadow_sigma_db, tech.shadow_corr_m, rng)

    @property
    def note(self) -> str:
        return self.field.note if self.field is not None else ""

    def mean(self, points: np.ndarray) -> np.ndarray:
        return mean_rss(points, self.anchors, self.scn.obstacles, self.tech, room_z=self.scn.room_z)

    def measure(self, points: np.ndarray, rng: np.random.Generator, mean=None) -> np.ndarray:
        """(P, A) raw RSS readings (one per point and anchor), NaN = not received."""
        t = self.tech
        mu = self.mean(points) if mean is None else mean
        if self.field is not None:
            sh = self.field.sample(points)
        elif t.shadow_sigma_db > 0:
            sh = rng.normal(0.0, t.shadow_sigma_db, mu.shape)
        else:
            sh = 0.0
        rss = mu + sh + fading_db(t, rng, mu.shape)
        if t.quantization_db > 0:
            rss = np.round(rss / t.quantization_db) * t.quantization_db
        if t.has_sensitivity():
            rss = np.where(rss < t.sensitivity_dbm, np.nan, rss)
        return rss


def average_fixes(samples: np.ndarray) -> np.ndarray:
    """(F, Ns, A) -> (F, A) mean in dB over the received readings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(samples, axis=1)
