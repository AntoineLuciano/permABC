#!/usr/bin/env python3
"""
Diagnostic: does SW2 → floor as N_sim → ∞ for a simple K=2 Gaussian case?

K=2, peaked prior (sigma_0=1), true theta = (mu1=0, mu2=2, sigma2=1).
Vanilla rejection + permABC-SMC at increasing budgets.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_convergence_K2.py
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
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.utils.functions import Theta
from permabc.core.distances import optimal_index_distance
from diagnostics import sample_true_posterior, sliced_w2_joint


def main():
    K = 2
    n_obs = 10
    sigma_0 = 1.0   # peaked prior on mu
    alpha, beta = 5.0, 5.0  # peaked prior on sigma2 (mode ~ 1)
    N_ref = 5000
    seed = 42

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model = GaussianWithNoSummaryStats(
        K=K, n_obs=n_obs, sigma_0=sigma_0, alpha=alpha, beta=beta
    )

    # True theta: mu1=0, mu2=2, sigma2=1
    true_theta = Theta(
        loc=np.array([[[0.0], [2.0]]]),
        glob=np.array([[1.0]]),
    )
    y_obs = model.data_generator(k2, true_theta)
    print(f"y_obs shape: {y_obs.shape}")
    print(f"y_obs[0,0]: {y_obs[0,0,:3]}...  y_obs[0,1]: {y_obs[0,1,:3]}...")

    # ── SW2 floor ──
    print("\n=== SW2 floor (true vs true, N=5000) ===")
    floors = []
    for s in range(5):
        rng_a = np.random.default_rng(1000 + s)
        mu_a, s2_a = sample_true_posterior(model, y_obs, N_ref, rng=rng_a)
        t_a = Theta(loc=mu_a[:, :, None], glob=s2_a[:, None])
        f = sliced_w2_joint(model, y_obs, t_a,
                            n_projections=200, n_ref_samples=N_ref, seed=0)
        floors.append(f)
        print(f"  floor[{s}] = {f:.6f}")
    floor_mean = np.mean(floors)
    floor_std = np.std(floors)
    print(f"  floor mean = {floor_mean:.6f} ± {floor_std:.6f}")

    # ── Sanity: SW2 of exact true posterior samples ──
    print("\n=== SW2 of exact posterior samples (should ≈ floor) ===")
    rng_exact = np.random.default_rng(9999)
    mu_ex, s2_ex = sample_true_posterior(model, y_obs, N_ref, rng=rng_exact)
    t_exact = Theta(loc=mu_ex[:, :, None], glob=s2_ex[:, None])
    sw2_exact = sliced_w2_joint(model, y_obs, t_exact,
                                 n_projections=200, n_ref_samples=N_ref, seed=0)
    print(f"  SW2(exact_posterior, ref) = {sw2_exact:.6f}  (ratio = {sw2_exact/floor_mean:.2f}x)")

    # ── Vanilla rejection ABC at increasing N_sim ──
    print("\n=== Vanilla rejection ABC (no perm) ===")
    N_sims = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    for N in N_sims:
        key, k_theta, k_data = random.split(key, 3)
        t0 = _time.perf_counter()
        thetas = model.prior_generator(k_theta, N)
        zs = model.data_generator(k_data, thetas)
        dists = model.distance(zs, y_obs)
        elapsed = _time.perf_counter() - t0

        # Take top N_ref accepted
        top_idx = np.argsort(dists)[:N_ref]
        thetas_acc = Theta(
            loc=np.array(thetas.loc)[top_idx],
            glob=np.array(thetas.glob)[top_idx],
        )
        eps = float(dists[top_idx[-1]])

        sw2 = sliced_w2_joint(model, y_obs, thetas_acc,
                               n_projections=200, n_ref_samples=N_ref, seed=0)
        print(f"  N={N:>12,}  eps={eps:.6f}  SW2={sw2:.6f}  ratio={sw2/floor_mean:.2f}x  ({elapsed:.1f}s)")

    # ── Vanilla permABC rejection at increasing N_sim ──
    print("\n=== permABC Vanilla (with perm) ===")
    N_sims_perm = [10_000, 100_000, 1_000_000, 10_000_000]

    for N in N_sims_perm:
        key, k_theta, k_data = random.split(key, 3)
        t0 = _time.perf_counter()
        thetas = model.prior_generator(k_theta, N)
        zs = model.data_generator(k_data, thetas)
        model.reset_weights_distance()
        dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs)
        elapsed = _time.perf_counter() - t0

        top_idx = np.argsort(dists_perm)[:N_ref]
        thetas_acc = Theta(
            loc=np.array(thetas.loc)[top_idx],
            glob=np.array(thetas.glob)[top_idx],
        )
        perm_acc = np.array(zs_index)[top_idx]
        eps = float(dists_perm[top_idx[-1]])

        sw2 = sliced_w2_joint(model, y_obs, thetas_acc, perm=perm_acc,
                               n_projections=200, n_ref_samples=N_ref, seed=0)
        print(f"  N={N:>12,}  eps={eps:.6f}  SW2={sw2:.6f}  ratio={sw2/floor_mean:.2f}x  ({elapsed:.1f}s)")

    # ── permABC-SMC ──
    print("\n=== permABC-SMC (N_particles=5000) ===")
    key, k_smc = random.split(key)
    model.reset_weights_distance()
    out = perm_abc_smc(
        key=k_smc, model=model, n_particles=N_ref,
        epsilon_target=0, y_obs=y_obs, kernel=KernelTruncatedRW,
        N_sim_max=np.inf, stopping_accept_rate=0.01,
        Final_iteration=0, update_weights_distance=False,
        verbose=1,
    )
    n_sim_total = int(np.sum(out["N_sim"]))
    thetas_smc = out["Thetas"][-1]
    weights_smc = out["Weights"][-1]
    perm_smc = out.get("Zs_index", [None])[-1]
    eps_final = float(out["Eps_values"][-1])

    sw2_smc = sliced_w2_joint(model, y_obs, thetas_smc, weights=weights_smc,
                               perm=perm_smc,
                               n_projections=200, n_ref_samples=N_ref, seed=0)
    print(f"\n  N_sim={n_sim_total:,}  eps_final={eps_final:.6f}")
    print(f"  SW2={sw2_smc:.6f}  ratio={sw2_smc/floor_mean:.2f}x")
    print(f"  unique={out['unique_part'][-1]:.4f}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"SUMMARY  K={K}, floor={floor_mean:.6f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
