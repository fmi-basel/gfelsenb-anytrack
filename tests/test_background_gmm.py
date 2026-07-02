"""Vectorized GMM background: equivalence to the sklearn reference + speedup."""
from __future__ import annotations

import time

import numpy as np
import pytest

from anytrack.background import fit_gmm_background, _fit_gmm_background_sklearn


def _make_samples(seed=0):
    """(N,H,W) with three pixel regimes: constant bg, bimodal (bg+object), noisy unimodal."""
    rng = np.random.default_rng(seed)
    N, H, W = 80, 64, 64
    s = np.empty((N, H, W), dtype=np.uint8)
    # constant background (skipped: std < min_std)
    s[:, 0:20, :] = np.clip(rng.normal(205, 1.0, (N, 20, W)), 0, 255)
    # bimodal: ~75% bright ~205, ~25% dark object ~30  -> 2-component, bg = bright mode
    for f in range(N):
        dark = rng.random((24, W)) < 0.25
        band = np.where(dark, rng.normal(30, 3, (24, W)), rng.normal(205, 2, (24, W)))
        s[f, 20:44, :] = np.clip(band, 0, 255)
    # noisy unimodal ~150 (std > min_std, 1-component)
    s[:, 44:64, :] = np.clip(rng.normal(150, 25, (N, 20, W)), 0, 255)
    return s


def test_vectorized_matches_sklearn():
    s = _make_samples()
    kw = dict(bic_improvement=10.0, lowp=120.0, min_std=10.0, reg_covar=1e-3)
    bg_sk, _ = _fit_gmm_background_sklearn(s, **kw)
    bg_v, _ = fit_gmm_background(s, **kw)

    assert bg_v.shape == bg_sk.shape == (64, 64)
    diff = np.abs(bg_sk.astype(int) - bg_v.astype(int))

    # Both recover the bright background mode (~205) in the bimodal band.
    assert abs(int(np.median(bg_sk[20:44])) - 205) <= 6
    assert abs(int(np.median(bg_v[20:44])) - 205) <= 6

    # Overall equivalence: most pixels identical, negligible mean difference.
    # (bg feeds detection as bg - gray > thr; a few-level diff vs ~175 object
    #  contrast cannot change detection.)
    assert int(np.median(diff)) == 0
    assert float(diff.mean()) < 2.0
    assert float(np.mean(diff <= 4)) > 0.95


def test_vectorized_is_faster(capsys):
    s = _make_samples(seed=1)
    t0 = time.perf_counter()
    _fit_gmm_background_sklearn(s, min_std=10.0)
    t_sk = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit_gmm_background(s, min_std=10.0)
    t_v = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n  GMM background: sklearn={t_sk:.3f}s  vectorized={t_v:.3f}s  "
              f"speedup={t_sk / max(t_v, 1e-6):.1f}x")
    assert t_v < t_sk
