"""Masked intrinsic dimension estimators: TwoNN, Levina-Bickel MLE-k, GRIDE, and MST-PHD."""

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.distance import cdist


def estimate_twonn_mle(dmat: np.ndarray) -> tuple[float, float, float]:
    """Returns (10%-trimmed Facco TwoNN, inverse-Pareto TwoNN, harmonic Levina-Bickel MLE-5)."""
    sd = np.sort(dmat, axis=1)
    r1, r2 = sd[:, 0], sd[:, 1]
    ok = (r1 > 1e-6) & np.isfinite(r2) & (r2 > r1 * 1.000001)
    if ok.sum() < 15:
        return float("nan"), float("nan"), float("nan")

    mu = np.sort(r2[ok] / r1[ok])
    nv = len(mu)
    pareto_id = float(nv / np.sum(np.log(mu)))

    keep = max(10, int(0.90 * nv))
    cdf = np.arange(1, nv + 1, dtype=np.float64) / (nv + 1.0)
    x = np.log(mu[:keep])
    y = -np.log(1.0 - cdf[:keep])
    facco_id = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))

    ok5 = (r1 > 1e-6) & np.isfinite(sd[:, 4]) & (sd[:, 4] > r1 * 1.000001)
    if ok5.sum() >= 15:
        inv_m = np.mean(np.log(np.maximum(sd[ok5, 4:5] / sd[ok5, :4], 1.000001)), axis=1)
        mle5 = float(1.0 / max(float(np.mean(inv_m)), 1e-9))
    else:
        mle5 = float("nan")
    return facco_id, pareto_id, mle5


def estimate_mle_k(dmat: np.ndarray, k: int = 5) -> float:
    """Harmonic Levina-Bickel MLE-k estimator on a (possibly masked) distance matrix."""
    ds = np.sort(dmat, axis=1)[:, :k]
    ok = np.all(np.isfinite(ds), axis=1) & (ds[:, 0] > 1e-12) & (ds[:, k - 1] > ds[:, 0])
    if np.sum(ok) < 10:
        return float("nan")
    ds = ds[ok]
    inv_m = np.mean(np.log(ds[:, [k - 1]] / ds[:, : k - 1]), axis=1)
    inv_m = inv_m[inv_m > 1e-12]
    return float(1.0 / np.mean(inv_m)) if len(inv_m) >= 10 else float("nan")


def estimate_gride(dmat: np.ndarray, n1: int = 2, n2: int = 4) -> float:
    """Generalized Ratios Intrinsic Dimension Estimator (Denti et al., 2022)."""
    ds = np.sort(dmat, axis=1)
    if ds.shape[1] < n2:
        return float("nan")
    r1, r2 = ds[:, n1 - 1], ds[:, n2 - 1]
    ok = np.isfinite(r1) & np.isfinite(r2) & (r1 > 1e-12) & (r2 > r1 * (1.0 + 1e-12))
    if np.sum(ok) < 12:
        return float("nan")
    log_mu = np.log(r2[ok] / r1[ok])
    mean_l = float(np.mean(log_mu))
    if mean_l <= 1e-12:
        return float("nan")
    if n2 - n1 == 1:
        return float(1.0 / (n1 * mean_l))

    def grad(d: float) -> float:
        u = d * log_mu
        return (1.0 / d) + (n2 - n1 - 1) * float(np.mean(log_mu / (-np.expm1(-u)))) - (n2 - 1) * mean_l

    if grad(1e-3) <= 0:
        return 1e-3
    if grad(500.0) >= 0:
        return 500.0
    return float(brentq(grad, 1e-3, 500.0, xtol=1e-6))


def _prim_mst_edges_dense(sub_d: np.ndarray):
    m = sub_d.shape[0]
    in_tree = np.zeros(m, dtype=bool)
    in_tree[0] = True
    min_d = sub_d[0].copy()
    w_list = np.empty(m - 1, dtype=np.float64)
    for step in range(m - 1):
        cand = np.where(in_tree, np.inf, min_d)
        nxt = int(np.argmin(cand))
        w = float(cand[nxt])
        if not np.isfinite(w):
            return None
        w_list[step] = w
        in_tree[nxt] = True
        better = (~in_tree) & (sub_d[nxt] < min_d)
        min_d[better] = sub_d[nxt, better]
    return w_list


def estimate_phd_mst(
    dmat: np.ndarray,
    alpha: float = 1.0,
    min_points: int = 45,
    n_steps: int = 7,
    n_reruns: int = 5,
    seed: int = 42,
) -> float:
    """Subsampled Prim MST Persistent Homology Dimension (Schweinhart, 2021)."""
    n_total = dmat.shape[0]
    if n_total < min_points + 15:
        return float("nan")
    rng = np.random.RandomState(seed)
    sizes = np.unique(np.linspace(min_points, n_total, n_steps, dtype=int))
    log_n, log_e = [], []

    for sz in sizes:
        vals = []
        draws = 1 if sz == n_total else n_reruns
        for _ in range(draws):
            idx = np.arange(n_total) if sz == n_total else np.sort(rng.choice(n_total, size=int(sz), replace=False))
            w = _prim_mst_edges_dense(dmat[np.ix_(idx, idx)])
            if w is not None:
                vals.append(float(np.sum(w ** alpha)))
        if vals:
            log_n.append(math.log(float(sz)))
            log_e.append(math.log(float(np.mean(vals))))

    if len(log_n) < 4:
        return float("nan")
    x, y = np.asarray(log_n), np.asarray(log_e)
    xm, ym = x - x.mean(), y - y.mean()
    kappa = float(np.dot(xm, ym) / max(np.dot(xm, xm), 1e-12))
    if kappa >= 0.97 or kappa <= -0.5:
        return float("nan")
    return float(alpha / (1.0 - kappa))


def masked_id_from_hidden_states(
    hidden_states: np.ndarray,
    mask: np.ndarray | None = None,
    drop_pos0: bool = False,
    unit_sphere: bool = False,
    compute_phd: bool = False,
) -> dict[str, float]:
    """Computes TwoNN, Pareto-TwoNN, MLE-5, GRIDE(2,4), and optionally MST-PHD on [L, D] hidden states."""
    h = np.asarray(hidden_states, dtype=np.float64)
    if drop_pos0:
        h = h[1:]
        if mask is not None:
            mask = mask[1:, 1:]
    if unit_sphere:
        h = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)

    dmat = cdist(h, h, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    dmat[dmat <= 1e-6] = np.inf
    if mask is not None:
        dmat[mask] = np.inf

    twonn, pareto, mle5 = estimate_twonn_mle(dmat)
    out = {
        "twonn": twonn,
        "pareto_twonn": pareto,
        "mle5": mle5,
        "gride_2_4": estimate_gride(dmat, 2, 4),
    }
    if compute_phd:
        out["mst_phd"] = estimate_phd_mst(dmat)
    return out
