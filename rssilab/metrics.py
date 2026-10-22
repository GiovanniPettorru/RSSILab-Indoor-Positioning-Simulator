"""Error statistics (overall, per test point, per trial) and empirical CDF."""
from __future__ import annotations

import warnings

import numpy as np

STAT_KEYS = ("n", "valid", "mean", "mse", "rmse", "std", "min", "median", "p90", "p95", "max")


def summarize(err: np.ndarray) -> dict:
    """Error statistics in metres (MSE in m^2). valid = % of fixes with an estimate."""
    e = np.asarray(err, float).ravel()
    n_all = e.size
    e = e[np.isfinite(e)]
    if e.size == 0:
        nan = float("nan")
        return dict(n=n_all, valid=0.0, mean=nan, mse=nan, rmse=nan, std=nan, min=nan,
                    median=nan, p90=nan, p95=nan, max=nan)
    mse = float(np.mean(e ** 2))
    return dict(n=n_all, valid=100.0 * e.size / max(n_all, 1), mean=float(e.mean()), mse=mse,
                rmse=float(np.sqrt(mse)), std=float(e.std()), min=float(e.min()),
                median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                p95=float(np.percentile(e, 95)), max=float(e.max()))


def _rowstats(e: np.ndarray) -> dict:
    """summarize() applied to each row of e (rows = groups), NaN-aware."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ok = np.isfinite(e)
        mse = np.nanmean(e ** 2, axis=1)
        return dict(n=np.full(len(e), e.shape[1]), valid=100.0 * ok.mean(axis=1),
                    mean=np.nanmean(e, axis=1), mse=mse, rmse=np.sqrt(mse),
                    std=np.nanstd(e, axis=1), min=np.nanmin(e, axis=1),
                    median=np.nanmedian(e, axis=1), p90=np.nanpercentile(e, 90, axis=1),
                    p95=np.nanpercentile(e, 95, axis=1), max=np.nanmax(e, axis=1))


def point_stats(est: np.ndarray, points: np.ndarray, K: int, dim: int = 2) -> dict:
    """Per test point, over all trials and repeated fixes.

    est: (R, P*K, 3) estimates ordered point by point (K fixes each); points (P, 3).
    Errors use the first `dim` coordinates. Besides the error statistics returns the
    mean estimate, the bias |mean estimate - true| and the spread
    sqrt(mean |estimate - mean estimate|^2): for every point MSE = bias^2 + spread^2.
    In 3D also the horizontal (xy) and vertical (z) RMSE."""
    R = est.shape[0]
    P = len(points)
    E = est.reshape(R, P, K, 3).transpose(1, 0, 2, 3).reshape(P, R * K, 3)
    D = E - points[:, None, :3]
    err = np.sqrt((D[..., :dim] ** 2).sum(-1))
    out = _rowstats(err)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(E, axis=1)
        dev = E[..., :dim] - m[:, None, :dim]
        out["spread"] = np.sqrt(np.nanmean((dev ** 2).sum(-1), axis=1))
        out["rmse_xy"] = np.sqrt(np.nanmean((D[..., :2] ** 2).sum(-1), axis=1))
        out["rmse_z"] = np.sqrt(np.nanmean(D[..., 2] ** 2, axis=1)) if dim == 3 else np.full(P, np.nan)
    out["est_x"], out["est_y"], out["est_z"] = m[:, 0], m[:, 1], m[:, 2]
    out["bias"] = np.sqrt(((m[:, :dim] - points[:, :dim]) ** 2).sum(-1))
    out["x"], out["y"], out["z"] = points[:, 0], points[:, 1], points[:, 2]
    return out


def trial_stats(err: np.ndarray) -> dict:
    """Per trial (row of err, shape (R, F))."""
    return _rowstats(np.asarray(err, float))


def ecdf(err: np.ndarray):
    e = np.sort(np.asarray(err, float).ravel())
    e = e[np.isfinite(e)]
    return e, np.arange(1, e.size + 1) / max(e.size, 1)
