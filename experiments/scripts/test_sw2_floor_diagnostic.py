#!/usr/bin/env python3
"""
Diagnostic: why doesn't SW2 converge to the floor?

Tests:
1. Exact posterior → SW2 should = floor (sanity check)
2. Vanilla rejection at increasing N_sim → should approach floor
3. permABC-SMC → check ESS and compare SW2 with/without resampling
4. Effect of n_ref_samples on floor stability

K=2, peaked prior to make it easy.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_floor_diagnostic.py
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
from diagnostics import sample_true_posterior, sliced_w2_joint


def ess_from_weights(w):
    """Effective sample size from normalized weights."""
    w = np.asarray(w, dtype=float)
    w = w / np.sum(w)
    return 1.0 / np.sum(w ** 2)


def resample_particles(thetas, weights, n_out, rng):
    """Multinomial resampling to get n_out equally-weighted particles."""
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    idx = rng.choice(len(w), size=n_out, replace=True, p=w)
    return Theta(
        loc=np.array(thetas.loc)[idx],
        glob=np.array(thetas.glob)[idx],
    )


def main():
    K = 2
    N = 5000  # number of particles / reference samples
    seed = 42

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model = GaussianWithNoSummaryStats(
        K=K, n_obs=10, sigma_0=1.0, alpha=5.0, beta=5.0,
    )

    # True theta: mu = (0, 2), sigma2 = 1
    true_theta = Theta(
        loc=np.array([[[0.0], [2.0]]]),
        glob=np.array([[1.0]]),
    )
    y_obs = model.data_generator(k2, true_theta)
    print(f"K={K}, sigma_0=1, alpha=5, beta=5")
    print(f"True theta: mu=(0, 2), sigma2=1")
    print(f"y_obs means: {np.mean(y_obs[0], axis=1)}")
    print(f"y_obs stds:  {np.std(y_obs[0], axis=1)}\n")

    # ================================================================
    # Test 0: Floor stability
    # ================================================================
    print("=" * 60)
    print("TEST 0: Floor stability (SW2 between two true posterior samples)")
    print("=" * 60)
    for trial in range(5):
        rng = np.random.default_rng(1000 + trial)
        mu_t, s2_t = sample_true_posterior(model, y_obs, N, rng=rng)
        t_true = Theta(loc=mu_t[:, :, None], glob=s2_t[:, None])
        sw2 = sliced_w2_joint(model, y_obs, t_true,
                               n_projections=200, n_ref_samples=N, seed=0)
        print(f"  trial {trial}: SW2(true_{trial}, ref) = {sw2:.6f}")

    # Fixed floor for comparisons
    rng_floor = np.random.default_rng(2000)
    mu_f, s2_f = sample_true_posterior(model, y_obs, N, rng=rng_floor)
    t_floor = Theta(loc=mu_f[:, :, None], glob=s2_f[:, None])
    floor = sliced_w2_joint(model, y_obs, t_floor,
                             n_projections=200, n_ref_samples=N, seed=0)
    print(f"\n  → Floor = {floor:.6f}\n")

    # ================================================================
    # Test 1: Vanilla rejection at increasing N_sim
    # ================================================================
    print("=" * 60)
    print("TEST 1: Vanilla rejection (no perm) - does SW2 → floor?")
    print("=" * 60)
    N_sims = [10_000, 100_000, 1_000_000, 10_000_000]

    for N_sim in N_sims:
        key, k_th, k_dat = random.split(key, 3)
        model.reset_weights_distance()
        t0 = _time.perf_counter()
        thetas = model.prior_generator(k_th, N_sim)
        zs = model.data_generator(k_dat, thetas)
        dists = np.array(model.distance(zs, y_obs))
        elapsed = _time.perf_counter() - t0

        # Take top N
        top_idx = np.argsort(dists)[:N]
        thetas_acc = Theta(
            loc=np.array(thetas.loc)[top_idx],
            glob=np.array(thetas.glob)[top_idx],
        )
        eps = float(dists[top_idx[-1]])

        sw2 = sliced_w2_joint(model, y_obs, thetas_acc,
                               n_projections=200, n_ref_samples=N, seed=0)
        print(f"  N_sim={N_sim:>12,}  eps={eps:.4f}  SW2={sw2:.6f}  ratio={sw2/floor:.2f}x  ({elapsed:.1f}s)")

    # ================================================================
    # Test 2: permABC-SMC — check weight degeneracy
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: permABC-SMC — ESS and weight degeneracy")
    print("=" * 60)
    key, k_smc = random.split(key)
    model.reset_weights_distance()
    out = perm_abc_smc(
        key=k_smc, model=model, n_particles=N,
        epsilon_target=0, y_obs=y_obs, kernel=KernelTruncatedRW,
        N_sim_max=np.inf, stopping_accept_rate=0.01,
        Final_iteration=0, update_weights_distance=False,
        verbose=1,
    )

    n_pops = len(out["Thetas"])
    print(f"\n  {n_pops - 1} populations, total N_sim = {int(np.sum(out['N_sim'])):,}")

    # Check last 5 populations
    print(f"\n  {'Pop':>4s} {'eps':>10s} {'ESS':>8s} {'unique%':>8s} {'SW2_weighted':>14s} {'SW2_resampled':>14s} {'ratio_w':>8s} {'ratio_r':>8s}")
    for i in range(max(1, n_pops - 5), n_pops):
        thetas_i = out["Thetas"][i]
        weights_i = out["Weights"][i]
        eps_i = out["Eps_values"][i]
        unique_i = out["unique_part"][i]
        perm_i = out.get("Zs_index", [None] * n_pops)[i]

        ess = ess_from_weights(weights_i)

        # SW2 with weights
        sw2_w = sliced_w2_joint(model, y_obs, thetas_i, weights=weights_i,
                                 perm=perm_i,
                                 n_projections=200, n_ref_samples=N, seed=0)

        # SW2 after resampling (eliminates weight degeneracy)
        rng_rs = np.random.default_rng(42)
        thetas_rs = resample_particles(thetas_i, weights_i, N, rng_rs)
        # Apply permutation before resampling
        if perm_i is not None:
            loc_arr = np.array(thetas_i.loc)
            perm_arr = np.array(perm_i, dtype=int)
            idx_ax = np.arange(loc_arr.shape[0])[:, None]
            loc_perm = loc_arr[idx_ax, perm_arr]
            thetas_perm = Theta(loc=loc_perm, glob=np.array(thetas_i.glob))
            thetas_rs = resample_particles(thetas_perm, weights_i, N, rng_rs)
        sw2_r = sliced_w2_joint(model, y_obs, thetas_rs,
                                 n_projections=200, n_ref_samples=N, seed=0)

        print(f"  {i:4d} {eps_i:10.4f} {ess:8.1f} {unique_i:8.2%} {sw2_w:14.6f} {sw2_r:14.6f} {sw2_w/floor:8.2f} {sw2_r/floor:8.2f}")

    # ================================================================
    # Test 3: Sampling noise decomposition
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Does resampled SMC approach floor?")
    print("=" * 60)
    # Resample final population multiple times
    thetas_final = out["Thetas"][-1]
    weights_final = out["Weights"][-1]
    perm_final = out.get("Zs_index", [None] * n_pops)[-1]
    ess_final = ess_from_weights(weights_final)

    # Apply permutation
    if perm_final is not None:
        loc_arr = np.array(thetas_final.loc)
        perm_arr = np.array(perm_final, dtype=int)
        idx_ax = np.arange(loc_arr.shape[0])[:, None]
        loc_perm = loc_arr[idx_ax, perm_arr]
        thetas_perm_final = Theta(loc=loc_perm, glob=np.array(thetas_final.glob))
    else:
        thetas_perm_final = thetas_final

    print(f"  Final ESS = {ess_final:.1f} / {N}")
    print(f"  Final eps = {out['Eps_values'][-1]:.6f}")
    print(f"  Floor = {floor:.6f}\n")

    # Resample at different sizes
    for n_resample in [100, 500, 1000, 2000, 5000, 10000]:
        sw2_vals = []
        for trial in range(5):
            rng_rs = np.random.default_rng(trial * 100)
            thetas_rs = resample_particles(thetas_perm_final, weights_final, n_resample, rng_rs)
            sw2 = sliced_w2_joint(model, y_obs, thetas_rs,
                                   n_projections=200, n_ref_samples=N, seed=0)
            sw2_vals.append(sw2)
        mean_sw2 = np.mean(sw2_vals)
        std_sw2 = np.std(sw2_vals)
        print(f"  n_resample={n_resample:>6d}  SW2={mean_sw2:.6f} ± {std_sw2:.6f}  ratio={mean_sw2/floor:.2f}x")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    print(f"Floor = {floor:.6f}")
    print(f"If SW2_weighted >> floor but SW2_resampled ≈ floor → weight degeneracy")
    print(f"If SW2_resampled >> floor even after resampling → ABC bias (epsilon too large)")
    print(f"ESS / N = {ess_final:.1f} / {N} = {ess_final/N:.1%}")


if __name__ == "__main__":
    main()
