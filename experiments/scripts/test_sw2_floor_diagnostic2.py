#!/usr/bin/env python3
"""
Diagnostic part 2: ABC posterior bias at moderate epsilon.

Shows that raw-data ABC with K*n_obs dimensional observations
produces biased posteriors (sigma2 → 0) even at small epsilon.

Compares: raw data (20D) vs summary stats (4D) vs raw data n_obs=1 (2D).

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_floor_diagnostic2.py
"""
import sys
import time as _time
from pathlib import Path
import numpy as np
from jax import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta
from diagnostics import sample_true_posterior, sliced_w2_joint


def main():
    K = 2
    seed = 42
    N_ref = 5000

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    # === Setup 1: raw data, n_obs=10 (20D observation space) ===
    model_20d = GaussianWithNoSummaryStats(K=K, n_obs=10, sigma_0=1.0, alpha=5.0, beta=5.0)
    true_theta = Theta(loc=np.array([[[0.0], [2.0]]]), glob=np.array([[1.0]]))
    y_obs_20d = model_20d.data_generator(k2, true_theta)

    # === Setup 2: raw data, n_obs=1 (2D observation space) ===
    model_2d = GaussianWithNoSummaryStats(K=K, n_obs=1, sigma_0=1.0, alpha=5.0, beta=5.0)
    key, k_y2d = random.split(key)
    y_obs_2d = model_2d.data_generator(k_y2d, true_theta)

    # ── Floors ──
    print("=" * 60)
    print("FLOORS")
    print("=" * 60)
    floor_20d = sliced_w2_joint(model_20d, y_obs_20d,
        Theta(loc=sample_true_posterior(model_20d, y_obs_20d, N_ref, rng=np.random.default_rng(2000))[0][:, :, None],
              glob=sample_true_posterior(model_20d, y_obs_20d, N_ref, rng=np.random.default_rng(2000))[1][:, None]),
        n_projections=200, n_ref_samples=N_ref, seed=0)

    floor_2d = sliced_w2_joint(model_2d, y_obs_2d,
        Theta(loc=sample_true_posterior(model_2d, y_obs_2d, N_ref, rng=np.random.default_rng(2000))[0][:, :, None],
              glob=sample_true_posterior(model_2d, y_obs_2d, N_ref, rng=np.random.default_rng(2000))[1][:, None]),
        n_projections=200, n_ref_samples=N_ref, seed=0)

    print(f"  n_obs=10 (20D): floor = {floor_20d:.6f}")
    print(f"  n_obs=1  (2D):  floor = {floor_2d:.6f}")

    # ── Expected distance at true params ──
    print("\n" + "=" * 60)
    print("EXPECTED DISTANCE AT TRUE PARAMETERS")
    print("=" * 60)
    key, k_dist = random.split(key)
    for model, y_obs, label in [(model_20d, y_obs_20d, "n_obs=10 (20D)"),
                                 (model_2d, y_obs_2d, "n_obs=1 (2D)")]:
        model.reset_weights_distance()
        zs_true = model.data_generator(k_dist, Theta(
            loc=np.tile(true_theta.loc, (1000, 1, 1)),
            glob=np.tile(true_theta.glob, (1000, 1)),
        ))
        dists_true = np.array(model.distance(zs_true, y_obs))
        print(f"  {label}: mean={np.mean(dists_true):.3f}, "
              f"median={np.median(dists_true):.3f}, "
              f"min={np.min(dists_true):.3f}, "
              f"P(d<0.5)={np.mean(dists_true < 0.5):.4f}")

    # ── Vanilla rejection: compare 20D vs 2D ──
    print("\n" + "=" * 60)
    print("VANILLA REJECTION: 20D vs 2D")
    print("=" * 60)

    for model, y_obs, floor, label, n_obs_label in [
        (model_20d, y_obs_20d, floor_20d, "n_obs=10 (20D)", 10),
        (model_2d, y_obs_2d, floor_2d, "n_obs=1 (2D)", 1),
    ]:
        print(f"\n  --- {label} ---")
        N_sims = [100_000, 1_000_000, 10_000_000]
        for N_sim in N_sims:
            key, k_th, k_dat = random.split(key, 3)
            model.reset_weights_distance()
            thetas = model.prior_generator(k_th, N_sim)
            zs = model.data_generator(k_dat, thetas)
            dists = np.array(model.distance(zs, y_obs))

            top_idx = np.argsort(dists)[:N_ref]
            thetas_acc = Theta(
                loc=np.array(thetas.loc)[top_idx],
                glob=np.array(thetas.glob)[top_idx],
            )
            eps = float(dists[top_idx[-1]])

            # Check posterior sigma2 and mu
            sigma2_acc = np.array(thetas_acc.glob)[:, 0]
            mu_acc = np.array(thetas_acc.loc)[:, :, 0]

            sw2 = sliced_w2_joint(model, y_obs, thetas_acc,
                                   n_projections=200, n_ref_samples=N_ref, seed=0)

            print(f"    N_sim={N_sim:>10,}  eps={eps:.4f}  "
                  f"SW2={sw2:.4f} ({sw2/floor:.1f}x)  "
                  f"sigma2: mean={np.mean(sigma2_acc):.3f} med={np.median(sigma2_acc):.3f}  "
                  f"mu1: mean={np.mean(mu_acc[:,0]):.3f} std={np.std(mu_acc[:,0]):.3f}")

    # ── True posterior stats for comparison ──
    print("\n" + "=" * 60)
    print("TRUE POSTERIOR STATISTICS (for comparison)")
    print("=" * 60)
    for model, y_obs, label in [(model_20d, y_obs_20d, "n_obs=10"),
                                 (model_2d, y_obs_2d, "n_obs=1")]:
        mu_true, s2_true = sample_true_posterior(model, y_obs, N_ref, rng=np.random.default_rng(42))
        print(f"  {label}: sigma2 mean={np.mean(s2_true):.3f} med={np.median(s2_true):.3f}  "
              f"mu1 mean={np.mean(mu_true[:,0]):.3f} std={np.std(mu_true[:,0]):.3f}")


if __name__ == "__main__":
    main()
