#!/usr/bin/env python3
"""
Two diagnostic tests:

Q1 – MSE of permABC vs ABC for varying epsilon.
     On the Gaussian toy model with closed-form posterior,
     compute MSE(sigma2) and MSE(mu_avg) for:
       (a) permABC-Vanilla (with optimal permutation)
       (b) ABC-Vanilla     (identity ordering, no permutation)
     as a function of epsilon, including regimes above and below epsilon*.

Q2 – SW2 floor: what SW2 do you get from true posterior samples?
     Draw N particles from the exact posterior and compute SW2 against
     another independent draw of the same posterior.
     This gives the MC noise floor.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_mse_and_sw2_floor.py
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
from diagnostics import sample_true_posterior, sliced_w2_joint


# ══════════════════════════════════════════════════════════════════════════════
# Common setup
# ══════════════════════════════════════════════════════════════════════════════

def setup(K=10, n_obs=10, seed=42):
    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model = GaussianWithNoSummaryStats(
        K=K, n_obs=n_obs, sigma_0=3.0, alpha=1.01, beta=1.0,
    )

    # True parameters: sigma2=1, mu_k drawn from prior except mu_1=0
    true_theta = model.prior_generator(k1, 1)
    glob = np.array(true_theta.glob); loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0
    loc[0, :, 0] = 0.0  # all mu_k = 0 for clean MSE reference
    true_theta = Theta(glob=glob, loc=loc)

    y_obs = model.data_generator(k2, true_theta)
    return model, y_obs, true_theta, key


# ══════════════════════════════════════════════════════════════════════════════
# Q1 – MSE of permABC vs ABC for varying epsilon
# ══════════════════════════════════════════════════════════════════════════════

def test_mse_vs_epsilon(K=10, N=200_000, seed=42):
    print("=" * 70)
    print(f"Q1 – MSE of permABC vs ABC  (K={K}, N_sim={N})")
    print("=" * 70)

    model, y_obs, true_theta, key = setup(K=K, seed=seed)
    true_sigma2 = float(true_theta.glob[0, 0])
    true_mu = np.array(true_theta.loc[0, :, 0])  # (K,)

    # Draw true posterior samples for reference MSE target
    mu_ref, s2_ref = sample_true_posterior(model, y_obs, 10_000, rng=np.random.default_rng(99))
    post_mean_sigma2 = np.mean(s2_ref)
    post_mean_mu = np.mean(mu_ref, axis=0)  # (K,)
    print(f"  True sigma2 = {true_sigma2:.3f}")
    print(f"  Posterior mean sigma2 = {post_mean_sigma2:.3f}")
    print(f"  Posterior mean mu_1   = {post_mean_mu[0]:.3f}")

    # Simulate N proposals from prior
    key, k1, k2 = random.split(key, 3)
    thetas = model.prior_generator(k1, N)
    zs = model.data_generator(k2, thetas)

    # permABC distances
    dists_perm, ys_idx, zs_idx, _ = optimal_index_distance(model, zs, y_obs, M=K)
    thetas_perm = thetas.apply_permutation(zs_idx)

    # ABC distances (identity, no permutation)
    dists_abc = np.sqrt(np.sum((np.array(zs[:, :K]) - np.array(y_obs[0:1])) ** 2, axis=(1, 2)))

    # Compute epsilon* (half min distance between distinct obs permutations)
    y_np = np.array(y_obs[0])  # (K, n_obs)
    eps_star = np.inf
    for i in range(K):
        for j in range(i + 1, K):
            d_ij = np.sqrt(np.sum((y_np[i] - y_np[j]) ** 2))
            eps_star = min(eps_star, d_ij / 2)
    print(f"  epsilon* = {eps_star:.4f}")

    # Epsilon schedule: spanning below and above epsilon*
    quantiles_perm = np.quantile(dists_perm, [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
    epsilons = np.sort(np.unique(np.concatenate([
        quantiles_perm,
        [eps_star * 0.5, eps_star * 0.8, eps_star, eps_star * 1.5, eps_star * 2, eps_star * 5],
    ])))

    print(f"\n  {'epsilon':>10s}  {'eps/eps*':>8s}  {'n_perm':>7s}  {'n_abc':>7s}"
          f"  {'MSE_s2_perm':>12s}  {'MSE_s2_abc':>12s}"
          f"  {'MSE_mu_perm':>12s}  {'MSE_mu_abc':>12s}")
    print("  " + "-" * 95)

    for eps in epsilons:
        # permABC accepted
        mask_perm = dists_perm <= eps
        n_perm = int(np.sum(mask_perm))
        # ABC accepted
        mask_abc = dists_abc <= eps
        n_abc = int(np.sum(mask_abc))

        def mse(particles_s2, particles_mu):
            """MSE relative to posterior mean."""
            mse_s2 = np.mean((particles_s2 - post_mean_sigma2) ** 2)
            mse_mu = np.mean(np.mean((particles_mu - post_mean_mu[None, :]) ** 2, axis=1))
            return mse_s2, mse_mu

        mse_s2_perm = mse_mu_perm = mse_s2_abc = mse_mu_abc = np.nan

        if n_perm >= 10:
            s2_p = np.array(thetas_perm.glob[mask_perm, 0])
            mu_p = np.array(thetas_perm.loc[mask_perm, :, 0])
            mse_s2_perm, mse_mu_perm = mse(s2_p, mu_p)

        if n_abc >= 10:
            s2_a = np.array(thetas.glob[mask_abc, 0])
            mu_a = np.array(thetas.loc[mask_abc, :, 0])
            mse_s2_abc, mse_mu_abc = mse(s2_a, mu_a)

        ratio = eps / eps_star
        marker = " <-- eps*" if abs(ratio - 1.0) < 0.01 else ""
        print(f"  {eps:10.4f}  {ratio:8.2f}  {n_perm:7d}  {n_abc:7d}"
              f"  {mse_s2_perm:12.4f}  {mse_s2_abc:12.4f}"
              f"  {mse_mu_perm:12.4f}  {mse_mu_abc:12.4f}{marker}")


# ══════════════════════════════════════════════════════════════════════════════
# Q2 – SW2 floor from true posterior samples
# ══════════════════════════════════════════════════════════════════════════════

def test_sw2_floor(K=10, seed=42):
    print("\n" + "=" * 70)
    print(f"Q2 – SW2 floor from true posterior samples (K={K})")
    print("=" * 70)

    model, y_obs, true_theta, _ = setup(K=K, seed=seed)

    # Different sample sizes to see how floor scales
    n_particles_list = [100, 200, 500, 1000, 2000, 5000]
    n_repeats = 10

    print(f"\n  {'N_particles':>12s}  {'SW2_mean':>10s}  {'SW2_std':>10s}  {'SW2_min':>10s}  {'SW2_max':>10s}")
    print("  " + "-" * 60)

    for N in n_particles_list:
        sw2_values = []
        for rep in range(n_repeats):
            # Draw "particles" from true posterior (no ABC, just exact samples)
            rng = np.random.default_rng(seed + rep * 1000)
            mu_samples, s2_samples = sample_true_posterior(model, y_obs, N, rng=rng)

            # Package as Theta
            glob = s2_samples[:, None]            # (N, 1)
            loc = mu_samples[:, :, None]           # (N, K, 1)
            fake_thetas = Theta(glob=glob, loc=loc)

            sw2 = sliced_w2_joint(
                model, y_obs, fake_thetas,
                weights=None, perm=None,
                n_projections=200, n_ref_samples=5000, seed=rep + 7,
            )
            sw2_values.append(sw2)

        sw2_values = np.array(sw2_values)
        print(f"  {N:12d}  {np.mean(sw2_values):10.4f}  {np.std(sw2_values):10.4f}"
              f"  {np.min(sw2_values):10.4f}  {np.max(sw2_values):10.4f}")

    # Also test with K=20 (the Fig 4/6 setting)
    if K != 20:
        print(f"\n  --- Now with K=20 ---")
        test_sw2_floor(K=20, seed=seed)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_mse_vs_epsilon(K=10, N=200_000, seed=42)
    test_sw2_floor(K=10, seed=42)
