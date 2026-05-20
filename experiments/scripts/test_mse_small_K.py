#!/usr/bin/env python3
"""
Q1 – MSE comparison of permABC vs ABC for small K where both accept.

For K=2,3,5 we compare rejection ABC (identity) vs permABC (optimal perm)
at varying epsilon, including regimes below and above epsilon*.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_mse_small_K.py
"""

import sys
from pathlib import Path
import numpy as np
from jax import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.assignment.dispatch import optimal_index_distance
from permabc.utils.functions import Theta
from diagnostics import sample_true_posterior


def run_mse_comparison(K, N=500_000, seed=42):
    print("=" * 80)
    print(f"K={K}, N_sim={N}")
    print("=" * 80)

    key = random.PRNGKey(seed)
    key, k1, k2, k3, k4 = random.split(key, 5)

    model = GaussianWithNoSummaryStats(
        K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0,
    )

    # True parameters
    true_theta = model.prior_generator(k1, 1)
    glob = np.array(true_theta.glob); loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0
    loc[0, :, 0] = 0.0
    true_theta = Theta(glob=glob, loc=loc)

    y_obs = model.data_generator(k2, true_theta)

    # Posterior mean as MSE target
    mu_ref, s2_ref = sample_true_posterior(model, y_obs, 20_000, rng=np.random.default_rng(99))
    post_mean_s2 = np.mean(s2_ref)
    post_mean_mu = np.mean(mu_ref, axis=0)

    # Simulate proposals
    thetas = model.prior_generator(k3, N)
    zs = model.data_generator(k4, thetas)

    # permABC distances + permuted thetas
    dists_perm, ys_idx, zs_idx, _ = optimal_index_distance(model, zs, y_obs, M=K)
    thetas_perm = thetas.apply_permutation(zs_idx)

    # ABC distances (identity)
    y_np = np.array(y_obs[0:1])
    zs_np = np.array(zs[:, :K])
    dists_abc = np.sqrt(np.sum((zs_np - y_np) ** 2, axis=(1, 2)))

    # epsilon*
    y_flat = np.array(y_obs[0])
    eps_star = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = np.sqrt(np.sum((y_flat[i] - y_flat[j]) ** 2))
            eps_star = min(eps_star, d_ij / 2)

    print(f"  eps* = {eps_star:.4f}")
    print(f"  min dist permABC = {np.min(dists_perm):.4f}")
    print(f"  min dist ABC     = {np.min(dists_abc):.4f}")
    print(f"  post mean sigma2 = {post_mean_s2:.4f}")

    # Epsilon grid: focus around eps* and where ABC also has samples
    abc_quantiles = np.quantile(dists_abc, [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    perm_quantiles = np.quantile(dists_perm, [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    manual = [eps_star * r for r in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]]
    epsilons = np.sort(np.unique(np.concatenate([abc_quantiles, perm_quantiles, manual])))

    print(f"\n  {'epsilon':>10s} {'eps/eps*':>8s} {'n_perm':>7s} {'n_abc':>7s}"
          f" {'MSE_s2_perm':>12s} {'MSE_s2_abc':>12s} {'ratio':>7s}"
          f" {'MSE_mu_perm':>12s} {'MSE_mu_abc':>12s} {'ratio':>7s}")
    print("  " + "-" * 110)

    for eps in epsilons:
        mask_p = dists_perm <= eps
        mask_a = dists_abc <= eps
        n_p, n_a = int(np.sum(mask_p)), int(np.sum(mask_a))

        mse_s2_p = mse_s2_a = mse_mu_p = mse_mu_a = np.nan
        ratio_s2 = ratio_mu = ""

        if n_p >= 20:
            s2_p = np.array(thetas_perm.glob[mask_p, 0])
            mu_p = np.array(thetas_perm.loc[mask_p, :, 0])
            mse_s2_p = np.mean((s2_p - post_mean_s2) ** 2)
            mse_mu_p = np.mean(np.mean((mu_p - post_mean_mu[None, :]) ** 2, axis=1))

        if n_a >= 20:
            s2_a = np.array(thetas.glob[mask_a, 0])
            mu_a = np.array(thetas.loc[mask_a, :, 0])
            mse_s2_a = np.mean((s2_a - post_mean_s2) ** 2)
            mse_mu_a = np.mean(np.mean((mu_a - post_mean_mu[None, :]) ** 2, axis=1))

        if np.isfinite(mse_s2_p) and np.isfinite(mse_s2_a) and mse_s2_a > 0:
            ratio_s2 = f"{mse_s2_p / mse_s2_a:7.2f}"
        if np.isfinite(mse_mu_p) and np.isfinite(mse_mu_a) and mse_mu_a > 0:
            ratio_mu = f"{mse_mu_p / mse_mu_a:7.2f}"

        star = " <--" if abs(eps / eps_star - 1.0) < 0.02 else ""
        print(f"  {eps:10.4f} {eps/eps_star:8.2f} {n_p:7d} {n_a:7d}"
              f" {mse_s2_p:12.4f} {mse_s2_a:12.4f} {ratio_s2:>7s}"
              f" {mse_mu_p:12.4f} {mse_mu_a:12.4f} {ratio_mu:>7s}{star}")


if __name__ == "__main__":
    for K in [2, 3, 5]:
        run_mse_comparison(K=K, N=500_000, seed=42)
        print()
