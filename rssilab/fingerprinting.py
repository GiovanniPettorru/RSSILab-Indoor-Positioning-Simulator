"""Fingerprinting: the survey of the scenario and the models trained on it.

The survey simulates the offline phase of fingerprinting. Reference points lie
on a grid over the free target area, with the chosen spacing (the resolution of
the radio map) and offset by half a spacing from the corner of the area, so that
they never coincide with the points of a test grid with the same spacing; in 3D
the grid is repeated on height layers. At every reference point the survey
records S fingerprints, each obtained like a position fix (the average of
`samples per fix` readings) through the same channel as the experiment: walls,
shadowing realisation, fading and receiver. The models are trained on the survey
once and retrained whenever the shadowing realisation changes. Missing readings
enter the models as sensitivity - 5 dB.

k-NN and WKNN are implemented with NumPy; the decision tree, the random forest
and the neural network use scikit-learn.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

from .channel import average_fixes
from .localization import FINGERPRINTING, _chunk, register
from .trajectory import grid_points


@dataclass
class Survey:
    points: np.ndarray        # (R, 3) reference points
    radio_map: np.ndarray     # (R, A) mean fingerprint of each reference point (dBm, NaN = never received)
    positions: np.ndarray     # (N, 3) position of every fingerprint, N = R x fingerprints per point
    fingerprints: np.ndarray  # (N, A) every fingerprint (dBm, NaN = not received)


def survey_points(scn, spacing: float, z_res: float = 0.5) -> np.ndarray:
    """Reference points (R, 3) of a survey with the given spacing: at the tag height
    in 2D; in 3D on height layers z_res apart across the target volume (one layer
    at its height when the volume is flat)."""
    layers = None
    if scn.dim == 3:
        nz = max(1, int(round((scn.area_z1 - scn.area_z0) / max(z_res, 0.05))))
        rz = (scn.area_z1 - scn.area_z0) / nz
        layers = scn.area_z0 + (np.arange(nz) + 0.5) * rz
    rp, _ = grid_points(scn, spacing, offset=spacing / 2, z_layers=layers)
    return rp


def build_survey(scn, env, spacing: float, per_point: int, rng, samples_per_fix: int = 1,
                 z_res: float = 0.5, check=None, block_readings: int = 100_000) -> Survey:
    """Simulates the survey in the environment `env` (channel.Environment), in
    blocks of reference points of at most `block_readings` readings (bounded
    memory); check() is called between blocks (to stop a long survey)."""
    rp = survey_points(scn, spacing, z_res)
    if len(rp) == 0:
        raise ValueError("The fingerprint survey has no reference point in the free target area: "
                         "reduce the survey grid spacing.")
    S, Ns = max(1, int(per_point)), max(1, int(samples_per_fix))
    mu = env.mean(rp)
    fps = np.empty((len(rp) * S, mu.shape[1]))
    B = max(1, block_readings // (S * Ns))
    for s in range(0, len(rp), B):
        if check is not None:
            check()
        e = min(s + B, len(rp))
        raw = env.measure(np.repeat(rp[s:e], S * Ns, axis=0), rng, mean=np.repeat(mu[s:e], S * Ns, axis=0))
        fps[s * S:e * S] = average_fixes(raw.reshape((e - s) * S, Ns, -1))
    return Survey(rp, average_fixes(fps.reshape(len(rp), S, -1)), np.repeat(rp, S, axis=0), fps)


# ---------------------------------------------------------------------------
_SKLEARN = {"Decision tree": ("tree", "DecisionTreeRegressor"),
            "Random forest": ("ensemble", "RandomForestRegressor"),
            "Neural network (MLP)": ("neural_network", "MLPRegressor")}


def _sk(method: str):
    """The scikit-learn class of a model, imported when first needed."""
    module, name = _SKLEARN[method]
    try:
        return getattr(importlib.import_module(f"sklearn.{module}"), name)
    except ImportError as e:
        raise ImportError(f"'{method}' needs scikit-learn, install it with: pip install scikit-learn") from e


def preload(names) -> None:
    """Imports scikit-learn for the selected models before a run, so that a missing
    installation is reported at once and the import time is not counted as training."""
    for n in names:
        if n in _SKLEARN:
            _sk(n)


def _seed(ctx) -> int:
    return int(ctx.rng.integers(2 ** 31 - 1))


def _apply(rss, ctx, predict):
    """predict(features (n, A)) -> (n, dim) on the fixes with at least one reading,
    in blocks; NaN (no estimate) for the fixes with no reading."""
    out = np.full((len(rss), ctx.dim), np.nan)
    ok = np.flatnonzero(~np.all(np.isnan(rss), axis=1))
    X = ctx.features(rss[ok])
    B = 50_000
    for s in range(0, len(ok), B):
        ctx.check_cancel()
        out[ok[s:s + B]] = np.asarray(predict(X[s:s + B]), dtype=float).reshape(-1, ctx.dim)
    return out


def _knn(Q, X, y, k, weighted, ctx, x2=None):
    """Mean position of the k fingerprints of X nearest to each row of Q (Euclidean
    distance in dB), weighted by 1/distance or uniform."""
    k = max(1, min(int(k), len(X)))
    x2 = (X ** 2).sum(1) if x2 is None else x2
    out = np.empty((len(Q), y.shape[1]))
    B = _chunk(len(X))
    for s in range(0, len(Q), B):
        ctx.check_cancel()
        q = Q[s:s + B]
        d2 = np.maximum((q ** 2).sum(1)[:, None] - 2 * q @ X.T + x2[None], 0.0)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        if weighted:
            w = 1.0 / (np.sqrt(np.take_along_axis(d2, idx, axis=1)) + 1e-6)
        else:
            w = np.ones(idx.shape)
        out[s:s + B] = (w[..., None] * y[idx]).sum(1) / w.sum(1)[:, None]
    return out


@register("k-NN", family=FINGERPRINTING)
def knn(rss, ctx):
    """k nearest neighbours in signal space among all the survey fingerprints,
    weighted by 1/distance (or uniform)."""
    X, y = ctx.training()
    x2 = ctx.trained("k-NN", lambda X, y: (X ** 2).sum(1))
    k = int(ctx.params.get("knn_k", 10))
    weighted = bool(ctx.params.get("knn_weighted", True))
    return _apply(rss, ctx, lambda q: _knn(q, X, y, k, weighted, ctx, x2))


@register("WKNN (radio map)", family=FINGERPRINTING)
def wknn(rss, ctx):
    """Weighted k nearest neighbours over the radio map: the mean fingerprint of
    each reference point, weights 1/distance."""
    pts, radio = ctx.fingerprints
    X = ctx.features(radio)
    k = int(ctx.params.get("wknn_k", 4))
    return _apply(rss, ctx, lambda q: _knn(q, X, pts[:, :ctx.dim], k, True, ctx))


@register("Decision tree", family=FINGERPRINTING)
def decision_tree(rss, ctx):
    """Regression tree (CART) from the fingerprint to the position."""
    leaf = max(1, int(ctx.params.get("dt_min_leaf", 5)))

    def fit(X, y):
        return _sk("Decision tree")(min_samples_leaf=leaf, random_state=_seed(ctx)).fit(X, y)
    return _apply(rss, ctx, ctx.trained("Decision tree", fit).predict)


@register("Random forest", family=FINGERPRINTING)
def random_forest(rss, ctx):
    """Random forest of regression trees (bootstrap samples, sqrt(A) anchors tried
    at each split), grown 10 trees at a time so that Stop stays responsive."""
    trees = max(1, int(ctx.params.get("rf_trees", 100)))

    def fit(X, y):
        rf = _sk("Random forest")(
            n_estimators=min(10, trees), max_features="sqrt", warm_start=True, n_jobs=-1,
            random_state=_seed(ctx))
        rf.fit(X, y)
        while rf.n_estimators < trees:         # warm start: same forest as one fit of all the trees
            ctx.check_cancel()
            rf.n_estimators = min(rf.n_estimators + 10, trees)
            rf.fit(X, y)
        return rf
    return _apply(rss, ctx, ctx.trained("Random forest", fit).predict)


class _ScaledMLP:
    """Network trained on standardised fingerprints and positions."""

    def __init__(self, net, mx, sx, my, sy):
        self.net, self.mx, self.sx, self.my, self.sy = net, mx, sx, my, sy

    def __call__(self, X):
        return self.net.predict((X - self.mx) / self.sx).reshape(len(X), -1) * self.sy + self.my


def _train_mlp(X, y, hidden, seed, ctx, max_epochs: int = 300, patience: int = 15) -> _ScaledMLP:
    """Adam, mini-batches of 512, early stopping on 10% of the fingerprints
    (the weights of the best epoch are kept); Stop is checked at every epoch."""
    MLP = _sk("Neural network (MLP)")
    rs = np.random.RandomState(seed)
    mx, sx = X.mean(axis=0), X.std(axis=0)
    sx[sx < 1e-9] = 1.0
    my, sy = y.mean(axis=0), float(max(y.std(axis=0).max(), 1e-9))
    Xs, ys = (X - mx) / sx, (y - my) / sy
    perm = rs.permutation(len(Xs))
    nv = len(Xs) // 10
    va, tr = perm[:nv], perm[nv:]
    net = MLP(hidden_layer_sizes=tuple(hidden), batch_size=min(512, len(tr)), learning_rate_init=3e-3,
              random_state=rs)
    best, keep, bad = np.inf, None, 0
    for _ in range(max_epochs):
        ctx.check_cancel()
        net.partial_fit(Xs[tr], ys[tr])
        if nv == 0:                            # tiny survey: no validation set, fixed number of epochs
            continue
        loss = float(np.mean((net.predict(Xs[va]).reshape(nv, -1) - ys[va]) ** 2))
        if loss < best * (1 - 1e-4):
            best, bad = loss, 0
            keep = ([c.copy() for c in net.coefs_], [b.copy() for b in net.intercepts_])
        else:
            bad += 1
            if bad >= patience:
                break
    if keep is not None:
        net.coefs_, net.intercepts_ = keep
    return _ScaledMLP(net, mx, sx, my, sy)


@register("Neural network (MLP)", family=FINGERPRINTING)
def neural_network(rss, ctx):
    """Multilayer perceptron (ReLU) regressing the position from the standardised fingerprint."""
    hidden = tuple(int(h) for h in ctx.params.get("mlp_hidden", (64, 64)))
    model = ctx.trained("Neural network (MLP)", lambda X, y: _train_mlp(X, y, hidden, _seed(ctx), ctx))
    return _apply(rss, ctx, model)
