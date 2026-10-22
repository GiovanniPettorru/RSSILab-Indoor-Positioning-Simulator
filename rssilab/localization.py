"""Positioning methods and trackers, in 2D (x, y with known target height) or 3D (x, y, z).

Every method belongs to one of two families:

* Distance-based: the RSS is inverted through the path-loss model, into distances
  (Min-Max, weighted centroid, trilateration, LLS / WLS / NLS multilateration) or
  into a likelihood (maximum-likelihood estimators). Defined in this module.
* Fingerprinting: the mapping from RSS to position is learnt from a simulated
  survey of the scenario (k-NN, WKNN on the radio map, decision tree, random
  forest, neural network). Defined in rssilab/fingerprinting.py.

Every method has the signature

    f(rss: ndarray (F, A), ctx: Context) -> ndarray (F, ctx.dim)

where rss holds one averaged RSS vector per position fix (NaN = anchor not
received). ctx.anchors are the anchor coordinates in the estimation space
((A, 2) in 2D, (A, 3) in 3D) and ctx.ranges(rss) the matching distances.
Register your own with @register("Name", family=...) (see plugins/user_algorithms.py):
it appears in the GUI automatically. A method that returns (F, 2) in 3D mode gets
the tag height as z.
"""
from __future__ import annotations

import importlib
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .channel import LN10, mean_rss
from .geometry import inside_obstacles, obstacle_hits

DISTANCE = "Distance-based"
FINGERPRINTING = "Fingerprinting"
FAMILIES = (DISTANCE, FINGERPRINTING)

ALGORITHMS: Dict[str, Callable] = {}
# names used before version 1.0, still accepted by the engine and in old projects
ALIASES = {"WCL": "Weighted centroid", "LLS trilateration": "LLS multilateration",
           "NLS (ML-LM)": "MLE (Levenberg-Marquardt)", "ML grid search": "MLE (grid search)",
           "WKNN fingerprint": "WKNN (radio map)"}
_CHUNK_ELEMS = 2_000_000            # bounds the (fixes x candidates) blocks of the vectorised methods


def register(name: str, needs: tuple = (), family: Optional[str] = None):
    """Decorator that adds a positioning method to the registry (and to the GUI).

    family is DISTANCE or FINGERPRINTING; by default FINGERPRINTING when needs
    contains 'fingerprints', DISTANCE otherwise. For fingerprinting methods the
    engine runs the survey and fills ctx.survey and ctx.fingerprints. needs may
    also contain 'ml_grid'. Registering an existing name replaces that method."""
    needs = tuple(needs)
    if family is None:
        family = FINGERPRINTING if "fingerprints" in needs else DISTANCE
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, not {family!r}")
    if family == FINGERPRINTING and "fingerprints" not in needs:
        needs += ("fingerprints",)

    def deco(fn):
        fn.needs, fn.family = needs, family
        ALGORITHMS[name] = fn
        return fn
    return deco


def family_of(name: str) -> str:
    """DISTANCE or FINGERPRINTING."""
    return getattr(ALGORITHMS[name], "family", DISTANCE)


def methods(family: Optional[str] = None) -> List[str]:
    """Registered method names, in registration order, optionally of one family."""
    return [n for n in ALGORITHMS if family is None or family_of(n) == family]


def canonical(name: str) -> str:
    """Current name of a method (maps the names used before version 1.0)."""
    return name if name in ALGORITHMS else ALIASES.get(name, name)


def _chunk(n_per_row: int) -> int:
    return max(1, _CHUNK_ELEMS // max(int(n_per_row), 1))


@dataclass
class EstimatorModel:
    """What the positioning engine *believes* about the channel."""
    p0: float      # calibrated mean RSS at d0 (fading mean included)
    n: float
    d0: float
    sigma: float   # std of the per-fix RSS (dB)


class Context:
    """What a positioning method may use besides the RSS.

    anchors (A, dim) / anchors3 (A, 3) enabled anchor positions; dim 2 or 3;
    z_fixed known target height (2D); area (x0, y0, x1, y1) and area3 (with z);
    lo / hi bounds of the estimation space; model (EstimatorModel: p0, n, d0, sigma);
    obstacles; tech (true Technology, use with care: an estimator should rely on
    model); params (algorithm parameters, see engine.LocParams); rng (a random
    stream of the method, independent of the other methods).
    Fingerprinting methods also get survey (fingerprinting.Survey: every
    fingerprint with its position) and fingerprints = (points (R, 3), radio map
    (R, A)), the mean fingerprint of each reference point.
    Helpers: rss_to_dist(), ranges(), ml_grid(), cell(), lift(), features(),
    training(), trained()."""

    def __init__(self, scn, tech, model: EstimatorModel, params: dict, rng):
        self.scn, self.tech, self.model = scn, tech, model
        self.params, self.rng = params, rng
        self.dim = 3 if scn.dim == 3 else 2
        self.anchors3 = scn.anchor_xyz()
        self.anchors = self.anchors3[:, :self.dim]
        self.z_fixed = float(scn.tag_h)
        self.area = scn.area_bounds()                 # (x0, y0, x1, y1)
        self.area3 = scn.area_bounds3()               # (x0, y0, z0, x1, y1, z1)
        self.lo = np.array(self.area3[:3])[:self.dim]
        self.hi = np.array(self.area3[3:])[:self.dim]
        self.room_z = scn.room_z
        self.obstacles = scn.obstacles
        self.map_aware = bool(params.get("map_aware", True))
        self.cancel = None                  # set by the engine: raises InterruptedError when stopped
        sens = tech.sensitivity_dbm if tech.has_sensitivity() else -150.0
        self.floor = sens - 5.0             # value used for missing readings
        self.train_time = 0.0               # seconds spent training models (read and reset by the engine)
        self._ml = None
        self.set_survey(None)

    def check_cancel(self) -> None:
        """Long-running methods call this now and then, so that Stop is responsive."""
        if self.cancel is not None:
            self.cancel()

    def lift(self, p: np.ndarray) -> np.ndarray:
        """(F, 2 or 3) -> (F, 3), using the known height in 2D."""
        p = np.asarray(p, dtype=float)
        if p.shape[1] >= 3:
            return p[:, :3]
        return np.column_stack([p, np.full(len(p), self.z_fixed)])

    def rss_to_dist(self, rss: np.ndarray) -> np.ndarray:
        """3D distance from RSS by inverting the log-distance model."""
        m = self.model
        return m.d0 * 10 ** ((m.p0 - rss) / (10 * m.n))

    def ranges(self, rss: np.ndarray) -> np.ndarray:
        """Distances in the estimation space: 3D in 3D, horizontal in 2D
        (the known height difference to each anchor is removed)."""
        d = self.rss_to_dist(rss)
        if self.dim == 3:
            return d
        dz = self.anchors3[None, :, 2] - self.z_fixed
        return np.sqrt(np.maximum(d * d - dz * dz, 1e-6))

    def ml_grid(self) -> dict:
        """Candidate grid (x, y layers; several z layers in 3D) and model RSS,
        walls included if map-aware."""
        if self._ml is None:
            res = float(self.params.get("ml_res", 0.5))
            x0, y0, z0, x1, y1, z1 = self.area3
            nx, ny = max(2, int(round((x1 - x0) / res))), max(2, int(round((y1 - y0) / res)))
            rx, ry = (x1 - x0) / nx, (y1 - y0) / ny
            xs = x0 + (np.arange(nx) + 0.5) * rx
            ys = y0 + (np.arange(ny) + 0.5) * ry
            if self.dim == 3 and z1 > z0:
                zres = float(self.params.get("z_res", 0.5))
                nz = max(1, int(round((z1 - z0) / zres)))
                rz = (z1 - z0) / nz
                zs = z0 + (np.arange(nz) + 0.5) * rz
            else:
                nz, rz, zs = 1, 1.0, np.array([self.z_fixed if self.dim == 2 else z0])
            Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
            cand3 = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            obs = self.obstacles if self.map_aware else []
            m = self.model
            H = mean_rss(cand3, self.anchors3, obs, self.tech, use_walls=bool(obs),
                         p0=m.p0, n=m.n, room_z=self.room_z)
            free = ~inside_obstacles(cand3, obs, 0.0, room_z=self.room_z) if obs else np.ones(len(cand3), bool)
            self._ml = dict(cand=cand3[:, :self.dim], cand3=cand3, H=H, free=free, x0=x0, y0=y0,
                            z0=zs[0] - rz / 2, rx=rx, ry=ry, rz=rz, nx=nx, ny=ny, nz=nz)
        return self._ml

    def cell(self, P: np.ndarray) -> np.ndarray:
        """Flat ML-grid index of each point (F, dim)."""
        g = self.ml_grid()
        c = np.clip(((P[:, 0] - g["x0"]) / g["rx"]).astype(int), 0, g["nx"] - 1)
        r = np.clip(((P[:, 1] - g["y0"]) / g["ry"]).astype(int), 0, g["ny"] - 1)
        k = np.zeros_like(c)
        if self.dim == 3 and g["nz"] > 1:
            k = np.clip(((P[:, 2] - g["z0"]) / g["rz"]).astype(int), 0, g["nz"] - 1)
        return (k * g["ny"] + r) * g["nx"] + c

    # ------------------------------------------------------------ fingerprinting
    def set_survey(self, survey) -> None:
        """Sets the fingerprint survey (fingerprinting.Survey) and forgets the models
        trained on the previous one."""
        self.survey = survey
        self.fingerprints = None if survey is None else (survey.points, survey.radio_map)
        self._training = None
        self._models = {}

    def features(self, rss: np.ndarray) -> np.ndarray:
        """Input of the fingerprinting models: the RSS with missing readings
        (NaN) replaced by sensitivity - 5 dB."""
        return np.where(np.isnan(rss), self.floor, rss)

    def training(self):
        """(X, y): features (N, A) and positions (N, dim) of every survey fingerprint."""
        if self._training is None:
            if self.survey is None:
                raise ValueError("No fingerprint survey: register the method with family=FINGERPRINTING.")
            self._training = (self.features(self.survey.fingerprints), self.survey.positions[:, :self.dim])
        return self._training

    def trained(self, key: str, fit: Callable):
        """The model returned by fit(X, y) on the current survey (see training()):
        fitted once per survey and kept until the survey changes. The fitting time
        is reported as training time, not as time per fix."""
        if key not in self._models:
            X, y = self.training()
            t0 = time.perf_counter()
            self._models[key] = fit(X, y)
            self.train_time += time.perf_counter() - t0
        return self._models[key]


# ---------------------------------------------------------------------------
# Distance-based: geometric
# ---------------------------------------------------------------------------
@register("Min-Max")
def minmax(rss, ctx):
    """Centre of the intersection of the bounding boxes [a - d, a + d]."""
    d, A = ctx.ranges(rss), ctx.anchors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanmax(A[None] - d[..., None], axis=1)
        hi = np.nanmin(A[None] + d[..., None], axis=1)
    return (lo + hi) / 2


@register("Weighted centroid")
def wcl(rss, ctx):
    """Weighted centroid of the anchors, weights (1/d)^g over the k strongest anchors."""
    A = ctx.anchors
    k = int(ctx.params.get("wcl_k", 4))
    g = float(ctx.params.get("wcl_g", 1.0))
    d = ctx.rss_to_dist(rss)
    w = np.where(np.isnan(d), 0.0, np.maximum(d, 1e-3) ** (-g))
    if 0 < k < A.shape[0]:
        order = np.argsort(-np.nan_to_num(rss, nan=-1e9), axis=1)[:, :k]
        keep = np.zeros_like(w, dtype=bool)
        np.put_along_axis(keep, order, True, axis=1)
        w = w * keep
    sw = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        est = (w @ A) / sw[:, None]
    est[sw == 0] = np.nan
    return est


# ---------------------------------------------------------------------------
# Distance-based: trilateration / multilateration
# ---------------------------------------------------------------------------
def _linearised(rss, ctx, keep=None, weighted=False):
    """Multilateration from the linearised range equations |p - a_i|^2 = d_i^2,
    one least-squares problem per fix (keep: optional (F, A) mask of the anchors
    to use).

    Unweighted: the equation of the strongest used anchor is subtracted from the
    others, which removes |p|^2 (classic LLS). Weighted: |p|^2 is kept as an extra
    unknown and each equation is weighted by the inverse variance of its squared
    range under log-normal shadowing, proportional to 1/d^4 (WLS).
    In 3D, when the used anchors are all at one height, (x, y) come from the linear
    system and z from the residual range, on the side of the target volume.
    A fix whose used anchors cannot determine the position (all on one line in 2D,
    or on one non-horizontal plane in 3D) gets no estimate (NaN)."""
    A, D = ctx.anchors, ctx.dim
    d = ctx.ranges(rss)
    M = ~np.isnan(d)
    if keep is not None:
        M &= keep
    F, nA = d.shape
    out = np.full((F, D), np.nan)
    c = A.mean(axis=0)
    Ac = A - c                                     # centred coordinates: better conditioned systems
    sq = (Ac ** 2).sum(1)
    d2 = np.where(M, d, 0.0) ** 2
    cnt = M.sum(1)
    if weighted:
        dd = np.where(M, np.maximum(ctx.rss_to_dist(rss), 0.1 * ctx.model.d0), np.inf)
        with np.errstate(invalid="ignore"):
            w = (np.min(dd, axis=1, keepdims=True) / dd) ** 4      # inverse variance of d^2, up to a factor
        w[~np.isfinite(w)] = 0.0
    else:
        w = M.astype(float)
    copl = np.zeros(F, bool)
    if D == 3:
        zu = np.where(M, A[None, :, 2], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            copl = (np.nanmax(zu, axis=1) - np.nanmin(zu, axis=1)) < 0.05
    for cols, rows in ((D, ~copl), (2, copl)):
        sel = np.flatnonzero(rows & (cnt >= cols + 1))
        B = _chunk(nA * (cols + 1))
        for s in range(0, sel.size, B):
            ctx.check_cancel()
            f = sel[s:s + B]
            n = len(f)
            Ak = Ac[:, :cols]
            if weighted:                       # unknowns: p - c and |p - c|^2 (or |p-c|^2 - 2 h z, coplanar)
                G = np.concatenate([np.broadcast_to(-2.0 * Ak, (n, nA, cols)), np.ones((n, nA, 1))], axis=2)
                b = d2[f] - sq[None]
                ww = w[f]
            else:                              # reference equation subtracted
                ref = np.argmax(np.where(M[f], rss[f], -np.inf), axis=1)
                G = 2.0 * (Ak[None] - Ak[ref][:, None])
                b = sq[None] - sq[ref][:, None] - d2[f] + d2[f, ref][:, None]
                ww = w[f].copy()
                ww[np.arange(n), ref] = 0.0
            Nm = np.einsum("fa,fai,faj->fij", ww, G, G)
            g = np.einsum("fa,fai,fa->fi", ww, G, b)
            ev = np.linalg.eigvalsh(Nm)                   # singular system: degenerate anchor geometry
            ok = ev[:, 0] > 1e-12 * np.maximum(ev[:, -1], 1e-300)
            f, Nm, g = f[ok], Nm[ok], g[ok]
            sol = np.einsum("fij,fj->fi", np.linalg.pinv(Nm), g)[:, :cols] + c[:cols]
            if cols == D:
                out[f] = sol
                continue
            # 3D with the used anchors at one height h: (x, y) solved, |z - h| from the residual range
            used = M[f]
            h2 = d2[f] - ((A[None, :, :2] - sol[:, None, :]) ** 2).sum(-1)
            dz = np.sqrt(np.maximum((h2 * used).sum(1) / used.sum(1), 0.0))
            za = (A[None, :, 2] * used).sum(1) / used.sum(1)
            zmid = (ctx.lo[2] + ctx.hi[2]) / 2
            z = np.where(np.abs(za - dz - zmid) <= np.abs(za + dz - zmid), za - dz, za + dz)
            out[f] = np.column_stack([sol, z])
    return out


def _strongest_independent(rss, A, k, tol: float = 1e-3):
    """Mask (F, A) of the k strongest received anchors of each fix that are affinely
    independent (in RSS order, an anchor is skipped when it lies on the line, or the
    plane, of the anchors already chosen, within `tol` metres)."""
    F, nA = rss.shape
    D = A.shape[1]
    order = np.argsort(-np.where(np.isnan(rss), -np.inf, rss), axis=1)
    heard = ~np.isnan(np.take_along_axis(rss, order, axis=1))
    keep = np.zeros((F, nA), bool)
    rows = np.arange(F)
    first = order[:, 0]
    keep[rows, first] = heard[:, 0]
    basis = np.zeros((F, D, D))                    # orthonormal directions spanned so far
    rank = np.zeros(F, int)
    for j in range(1, nA):
        c = order[:, j]
        v = A[c] - A[first]
        r = v - np.einsum("fkd,fk->fd", basis, np.einsum("fkd,fd->fk", basis, v))
        nr = np.linalg.norm(r, axis=1)
        ok = np.flatnonzero(heard[:, j] & (rank < k - 1) & (nr > tol))
        keep[ok, c[ok]] = True
        basis[ok, rank[ok]] = r[ok] / nr[ok, None]
        rank[ok] += 1
    return keep


@register("Trilateration")
def trilateration(rss, ctx):
    """Classic trilateration: the D + 1 strongest anchors of each fix (3 in 2D,
    4 in 3D), skipping anchors aligned with the stronger ones (on a line in 2D,
    on a plane in 3D); linearised equations solved exactly."""
    A = ctx.anchors
    if ctx.dim == 3 and np.ptp(ctx.anchors3[:, 2]) < 0.05:
        A = A[:, :2]                   # anchors at one height: 3 of them, z from the residual range
    return _linearised(rss, ctx, keep=_strongest_independent(rss, A, A.shape[1] + 1))


@register("LLS multilateration")
def lls(rss, ctx):
    """Linear least squares on all received anchors, strongest anchor as reference equation."""
    return _linearised(rss, ctx)


@register("WLS multilateration")
def wls(rss, ctx):
    """Weighted least squares on the linearised equations, weights 1/d^4 (inverse
    variance of the squared ranges under log-normal shadowing)."""
    return _linearised(rss, ctx, weighted=True)


def _levenberg_marquardt(p, model, M, ctx, iters: int = 200, tol: float = 1e-4):
    """Batched Levenberg-Marquardt. model(q, rows) returns the residuals (n, A) and
    their Jacobian (n, A, D) for the fixes `rows` at positions q (zero for missing
    readings). Each fix iterates until its step is below `tol` metres (or `iters`
    steps); fixes with fewer than D + 1 readings get NaN."""
    D = p.shape[1]
    res, J = model(p, np.arange(len(p)))
    cost = (res ** 2).sum(1)
    lam = np.full(len(p), 1e-2)
    eye = np.eye(D)
    active = M.sum(1) >= D + 1
    for it in range(iters):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        if it % 10 == 0:
            ctx.check_cancel()
        Jf = J[idx]
        JTJ = np.einsum("fai,faj->fij", Jf, Jf)
        g = np.einsum("fai,fa->fi", Jf, res[idx])
        Hm = JTJ + lam[idx, None, None] * (JTJ * eye) + 1e-9 * eye
        step = np.linalg.solve(Hm, g[..., None])[..., 0]
        pn = p[idx] - step
        rn, Jn = model(pn, idx)
        cn = (rn ** 2).sum(1)
        acc = cn < cost[idx]
        ia = idx[acc]
        p[ia], cost[ia], res[ia], J[ia] = pn[acc], cn[acc], rn[acc], Jn[acc]
        lam[idx] = np.clip(np.where(acc, lam[idx] * 0.3, lam[idx] * 5.0), 1e-7, 1e7)
        done = (np.linalg.norm(step, axis=1) < tol) | (lam[idx] >= 1e7)
        active[idx[done]] = False
    p[M.sum(1) < D + 1] = np.nan
    return p


def _start(rss, ctx, init):
    """Starting point of the iterative methods: the weighted centroid (z at mid
    height in 3D) or init."""
    D = ctx.dim
    p = wcl(rss, ctx) if init is None else np.array(init, dtype=float)[:, :D]
    centre = (ctx.lo + ctx.hi) / 2
    p[np.isnan(p).any(axis=1)] = centre
    if D == 3 and init is None:
        p[:, 2] = centre[2]             # anchors often share a height: start off their plane
    return p


@register("NLS multilateration")
def nls(rss, ctx, init=None, iters: int = 200, tol: float = 1e-4):
    """Nonlinear least squares on the ranges, min sum (|p - a_i| - d_i)^2,
    Levenberg-Marquardt from the weighted centroid."""
    A = ctx.anchors
    M = ~np.isnan(rss)
    d = np.nan_to_num(ctx.ranges(rss))

    def model(q, rows):
        v = q[:, None, :] - A[None]
        r = np.sqrt(np.maximum((v * v).sum(-1), 1e-12))
        m = M[rows]
        return np.where(m, r - d[rows], 0.0), np.where(m[..., None], v / r[..., None], 0.0)

    return _levenberg_marquardt(_start(rss, ctx, init), model, M, ctx, iters, tol)


@register("MLE (Levenberg-Marquardt)")
def mle_lm(rss, ctx, init=None, iters: int = 200, tol: float = 1e-4):
    """Maximum likelihood in the RSS domain (log-normal shadowing), i.e. least
    squares on the RSS residuals, Levenberg-Marquardt from the weighted centroid."""
    m, A3, D = ctx.model, ctx.anchors3, ctx.dim
    M = ~np.isnan(rss)
    r = np.nan_to_num(rss)
    k = 10 * m.n / LN10
    dmin2 = (0.1 * m.d0) ** 2

    def model(q, rows):
        dv = ctx.lift(q)[:, None, :] - A3[None, :, :]
        d2 = np.maximum((dv * dv).sum(-1), dmin2)
        h = m.p0 - 5 * m.n * np.log10(d2 / m.d0 ** 2)
        mk = M[rows]
        return (np.where(mk, r[rows] - h, 0.0),
                np.where(mk[..., None], k * dv[..., :D] / d2[..., None], 0.0))

    return _levenberg_marquardt(_start(rss, ctx, init), model, M, ctx, iters, tol)


@register("MLE (grid search)", needs=("ml_grid",))
def mle_grid(rss, ctx):
    """Exhaustive maximum likelihood over a grid (several height layers in 3D);
    with 'map-aware' the model includes the walls and estimates are restricted to
    free space."""
    g = ctx.ml_grid()
    free = g["free"] if g["free"].any() else np.ones_like(g["free"])
    C, H = g["cand"][free], g["H"][free]
    H2T = (H ** 2).T
    M = (~np.isnan(rss)).astype(float)
    r = np.nan_to_num(rss)
    out = np.full((len(rss), ctx.dim), np.nan)
    B = _chunk(len(C))
    for s in range(0, len(rss), B):
        ctx.check_cancel()
        Ms, rs = M[s:s + B], r[s:s + B]
        cost = (Ms * rs * rs).sum(1)[:, None] - 2 * (Ms * rs) @ H.T + Ms @ H2T
        out[s:s + B] = C[np.argmin(cost, axis=1)]
    out[M.sum(1) == 0] = np.nan
    return out


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------
def kalman_cv(z: np.ndarray, dt: float, q_acc: float, r_std: float) -> np.ndarray:
    """Constant-velocity Kalman filter on position fixes (F, D); NaN fixes = predict only."""
    D = z.shape[1]
    I, O = np.eye(D), np.zeros((D, D))
    Fm = np.block([[I, dt * I], [O, I]])
    G = np.vstack([dt * dt / 2 * I, dt * I])
    Q = (q_acc ** 2) * G @ G.T
    Hm = np.hstack([I, O])
    R = (r_std ** 2) * I
    out = np.full_like(z, np.nan)
    x, P = None, None
    for f, zf in enumerate(z):
        ok = not np.isnan(zf).any()
        if x is None:
            if ok:
                x = np.r_[zf, np.zeros(D)]
                P = np.diag(np.r_[np.full(D, r_std ** 2), np.full(D, 4.0)])
                out[f] = zf
            continue
        x = Fm @ x
        P = Fm @ P @ Fm.T + Q
        if ok:
            S = Hm @ P @ Hm.T + R
            K = P @ Hm.T @ np.linalg.inv(S)
            x = x + K @ (zf - Hm @ x)
            P = (np.eye(2 * D) - K @ Hm) @ P
        out[f] = x[:D]
    return out


def particle_filter(rss: np.ndarray, ctx: Context, n: int, dt: float, acc_std: float,
                    jitter: float, v0: float, sigma: float, rng,
                    inject: float = 0.02) -> np.ndarray:
    """SIR particle filter, constant-velocity motion, RSS likelihood of the
    path-loss model looked up on the ML grid (2D or 3D). Map-aware: particles stay
    in free space and moves through walls are rejected (the particle stops and
    picks a new velocity). A fraction `inject` of particles is redrawn uniformly
    in free space at every step, so the filter recovers when the cloud gets
    trapped behind a wall."""
    g = ctx.ml_grid()
    C, H, free = g["cand"], g["H"], g["free"]
    D = ctx.dim
    lo, hi = ctx.lo, ctx.hi
    cellsize = np.array([g["rx"], g["ry"], g["rz"]])[:D]
    if D == 3 and g["nz"] == 1:
        cellsize[2] = 0.0
    walls = [o for o in ctx.obstacles if o.kind == "wall" and o.blocks_motion] if ctx.map_aware else []
    fidx = np.flatnonzero(free) if free.any() else np.arange(len(free))
    vstd = v0 / np.sqrt(D)

    def draw(m):
        return C[rng.choice(fidx, m)] + rng.uniform(-0.5, 0.5, (m, D)) * cellsize

    P = draw(n)
    V = rng.normal(0.0, vstd, (n, D))
    if D == 3:
        V[:, 2] *= 0.3                    # targets move mostly horizontally
    w = np.full(n, 1.0 / n)
    out = np.full((len(rss), D), np.nan)
    for f in range(len(rss)):
        if f % 64 == 0:
            ctx.check_cancel()
        V = V + rng.normal(0.0, acc_std * dt, V.shape)
        prop = P + V * dt + rng.normal(0.0, jitter, P.shape)
        if D == 3:                      # the height stays in the target volume (stops at its floor/top)
            zc = np.clip(prop[:, 2], lo[2], hi[2])
            V[zc != prop[:, 2], 2] = 0.0
            prop[:, 2] = zc
        ok = np.all((prop >= lo - 1e-9) & (prop <= hi + 1e-9), axis=1)
        ok &= free[ctx.cell(prop)]
        if walls:
            a3, b3 = ctx.lift(P), ctx.lift(prop)
            for wl in walls:
                ok &= ~obstacle_hits(a3[:, 0], a3[:, 1], b3[:, 0], b3[:, 1], wl, a3[:, 2], b3[:, 2],
                                     ctx.room_z)
        P = np.where(ok[:, None], prop, P)
        nb = int((~ok).sum())
        if nb:
            V[~ok] = rng.normal(0.0, vstd, (nb, D))
        ni = int(round(inject * n))
        if ni:
            kk = rng.choice(n, ni, replace=False)
            P[kk] = draw(ni)
            V[kk] = rng.normal(0.0, vstd, (ni, D))
            w[kk] = 1.0 / n
            w = w / w.sum()
        r = rss[f]
        m = ~np.isnan(r)
        if m.any():
            h = H[ctx.cell(P)][:, m]
            ll = -0.5 * (((r[m] - h) / sigma) ** 2).sum(1)
            w = w * np.exp(ll - ll.max())
        sw = w.sum()
        w = np.full(n, 1.0 / n) if (not np.isfinite(sw) or sw <= 0) else w / sw
        out[f] = w @ P
        if 1.0 / (w @ w) < n / 2:                       # systematic resampling
            cs = np.cumsum(w); cs[-1] = 1.0
            kk = np.searchsorted(cs, (rng.random() + np.arange(n)) / n)
            P, V = P[kk], V[kk]
            w = np.full(n, 1.0 / n)
    return out


importlib.import_module(".fingerprinting", __package__)        # registers the fingerprinting methods
