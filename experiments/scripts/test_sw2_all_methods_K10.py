#!/usr/bin/env python3
"""
Compare SW2 across ALL ABC methods for K=10, same simulation budget.

Budget = total N_sim (component-simulations) from permABC-SMC.
Reports: N_sim, unique particles / ESS, SW2.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_all_methods_K10.py
"""

import sys
import time as _time
from pathlib import Path

import numpy as np
from jax import random
from scipy.stats import invgamma

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.algorithms.smc import abc_smc, perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.assignment.dispatch import optimal_index_distance
from permabc.utils.functions import Theta
from diagnostics import sample_true_posterior, sliced_w2_joint


# ── Helpers ────────────────────────────────────────────────────────────────

def compute_sw2_floor(model, y_obs, n_particles, n_projections=200, seed=0):
    rng = np.random.default_rng(seed + 2000)
    mu, s2 = sample_true_posterior(model, y_obs, n_particles, rng=rng)
    thetas_true = Theta(loc=mu[:, :, None], glob=s2[:, None])
    return sliced_w2_joint(model, y_obs, thetas_true,
                           n_projections=n_projections, n_ref_samples=n_particles,
                           seed=seed)


def ess_from_weights(weights):
    """ESS from importance weights: 1/sum(w_i^2)."""
    w = np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    return 1.0 / np.sum(w**2)


def ess_autocorr_1d(chain):
    """ESS via truncated autocorrelation (FFT) for a 1D chain."""
    n = len(chain)
    if n < 10:
        return float(n)
    x = chain - np.mean(chain)
    var = np.var(chain)
    if var < 1e-30:
        return 1.0
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
    acf /= acf[0]
    # Initial positive sequence estimator (Geyer 1992)
    tau = 1.0
    for lag in range(1, n):
        if acf[lag] < 0.05:
            break
        tau += 2 * acf[lag]
    return n / max(tau, 1.0)


def ess_mcmc(mus_chain, s2_chain):
    """Min ESS across all parameters of the Gibbs chain."""
    ess_vals = [ess_autocorr_1d(s2_chain)]
    for k in range(mus_chain.shape[1]):
        ess_vals.append(ess_autocorr_1d(mus_chain[:, k]))
    return min(ess_vals), ess_vals


def unique_count(thetas):
    """Count unique particles based on loc parameters."""
    loc = np.asarray(thetas.loc)
    if loc.ndim == 3:
        loc = loc.reshape(loc.shape[0], -1)
    _, idx = np.unique(loc, axis=0, return_index=True)
    return len(idx)


# ── Summary stats model ───────────────────────────────────────────────────

class GaussianSummaryStats(GaussianWithNoSummaryStats):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_scale = np.ones(2)

    def data_generator(self, key, thetas):
        raw = super().data_generator(key, thetas)
        means = np.mean(raw, axis=2, keepdims=True)
        stds = np.std(raw, axis=2, keepdims=True)
        return np.concatenate([means, stds], axis=2)

    def update_weights_distance(self, zs, verbose=0):
        for d in range(zs.shape[2]):
            vals = zs[:, :, d].ravel()
            mad_d = float(np.median(np.abs(vals - np.median(vals))))
            if mad_d > 1e-10:
                self._obs_scale[d] = 1.0 / mad_d

    def distance(self, zs, y_obs):
        s = self._obs_scale[None, None, :]
        diff = (y_obs[0] - zs) * s
        return np.array(np.sqrt(np.sum(diff ** 2, axis=(1, 2))))

    def distance_component(self, z_k, y_k):
        diff = (y_k - z_k) * self._obs_scale
        return float(np.sum(diff ** 2))

    def distance_matrices_loc(self, zs, y_obs, M=0, L=0):
        if M == 0: M = self.K
        if L == 0: L = self.K
        s = self._obs_scale
        zs_s = zs[:, :M] * s[None, None, :]
        y_s = y_obs[0, :L] * s[None, :]
        n = zs_s.shape[0]
        matrices = np.zeros((n, 2 * self.K - L, self.K + M - L))
        for i in range(n):
            for k in range(L):
                for m in range(M):
                    matrices[i, k, m] = np.sum((y_s[k] - zs_s[i, m]) ** 2)
        return matrices


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    K = 10
    N_particles = 5000
    seed = 42

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model_raw = GaussianWithNoSummaryStats(
        K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0,
    )
    model_ss = GaussianSummaryStats(
        K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0,
    )

    # Generate data
    true_theta = model_raw.prior_generator(k1, 1)
    glob = np.array(true_theta.glob); loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0; loc[0, :, 0] = 0.0
    true_theta = Theta(glob=glob, loc=loc)
    y_obs_raw = model_raw.data_generator(k2, true_theta)

    y_means = np.mean(y_obs_raw, axis=2, keepdims=True)
    y_stds = np.std(y_obs_raw, axis=2, keepdims=True)
    y_obs_ss = np.concatenate([y_means, y_stds], axis=2)

    print(f"K={K}, N_particles={N_particles}")
    print(f"True params: mu_k=0.0, sigma2=1.0")
    print(f"N_sim counts component-simulations (each draw of K components = K sims)\n")

    # ── 0. SW2 floor ──
    print("Computing SW2 floor...")
    sw2_floor = compute_sw2_floor(model_raw, y_obs_raw, N_particles, seed=0)
    print(f"SW2 floor = {sw2_floor:.4f}\n")

    # results: (method, n_sim, epsilon, sw2, n_effective, eff_label)
    results = []

    # ── 1. permABC-SMC (summary stats) → defines budget ──
    print("=" * 60)
    print("permABC-SMC (summary stats + Prangle MAD)")
    print("=" * 60)
    key, k_smc = random.split(key)
    out_perm_ss = perm_abc_smc(
        key=k_smc, model=model_ss, n_particles=N_particles,
        epsilon_target=0, y_obs=y_obs_ss, kernel=KernelTruncatedRW,
        N_sim_max=np.inf, stopping_accept_rate=0.015,
        Final_iteration=200, update_weights_distance=True,
        verbose=1, try_lsa=True,
    )
    budget = int(np.sum(out_perm_ss["N_sim"]))
    thetas_perm_ss = out_perm_ss["Thetas"][-1]
    weights_perm_ss = out_perm_ss["Weights"][-1]
    eps_perm_ss = out_perm_ss["Eps_values"][-1]
    sw2_perm_ss = sliced_w2_joint(model_raw, y_obs_raw, thetas_perm_ss,
                                   weights=weights_perm_ss,
                                   n_projections=200, n_ref_samples=N_particles, seed=0)
    n_uniq_perm_ss = unique_count(thetas_perm_ss)
    ess_w_perm_ss = ess_from_weights(weights_perm_ss)
    results.append(("permABC-SMC (ss)", budget, eps_perm_ss, sw2_perm_ss,
                     n_uniq_perm_ss, int(ess_w_perm_ss), "unique/ESS_w"))
    n_iters_ss = len(out_perm_ss["Eps_values"]) - 1
    print(f"\npermABC-SMC (ss): N_sim={budget:,} ({n_iters_ss} iters × {N_particles}×{K})")
    print(f"  eps={eps_perm_ss:.4f}, SW2={sw2_perm_ss:.4f}")
    print(f"  unique={n_uniq_perm_ss}/{N_particles}, ESS_weights={ess_w_perm_ss:.0f}")
    print(f"\n*** BUDGET = {budget:,} component-sims ***\n")

    # ── 2. ABC-Vanilla (raw data) ──
    print("=" * 60)
    print(f"ABC-Vanilla (raw data)")
    print("=" * 60)
    N_vanilla = budget // K
    actual_van = N_vanilla * K
    print(f"  N_draws={N_vanilla:,}, N_sim={actual_van:,}")
    key, k_th, k_dat = random.split(key, 3)
    model_raw.reset_weights_distance()
    thetas_van = model_raw.prior_generator(k_th, N_vanilla)
    zs_van = model_raw.data_generator(k_dat, thetas_van)
    dists_van = np.array(model_raw.distance(zs_van, y_obs_raw))

    sorted_dists = np.sort(dists_van)
    eps_van = sorted_dists[min(N_particles - 1, N_vanilla - 1)]
    mask = dists_van <= eps_van
    n_acc_van = int(np.sum(mask))
    thetas_acc = thetas_van[mask]
    if n_acc_van > N_particles:
        rng_sub = np.random.default_rng(42)
        idx = rng_sub.choice(n_acc_van, N_particles, replace=False)
        thetas_acc = thetas_acc[idx]
        n_acc_van = N_particles
    sw2_van = sliced_w2_joint(model_raw, y_obs_raw, thetas_acc,
                               n_projections=200, n_ref_samples=N_particles, seed=0)
    results.append(("ABC-Vanilla", actual_van, eps_van, sw2_van,
                     n_acc_van, n_acc_van, "n_accepted"))
    print(f"  eps={eps_van:.4f}, n_accepted={n_acc_van}, SW2={sw2_van:.4f}")

    # ── 3. permABC-Vanilla (raw data) ──
    print("\n" + "=" * 60)
    print(f"permABC-Vanilla (raw data)")
    print("=" * 60)
    key, k_th, k_dat = random.split(key, 3)
    model_raw.reset_weights_distance()
    thetas_pv = model_raw.prior_generator(k_th, N_vanilla)
    zs_pv = model_raw.data_generator(k_dat, thetas_pv)
    t0 = _time.perf_counter()
    dists_pv, _, zs_idx_pv, _ = optimal_index_distance(
        model=model_raw, zs=zs_pv, y_obs=y_obs_raw, epsilon=0, verbose=1,
    )
    print(f"  Assignment time: {_time.perf_counter()-t0:.1f}s")
    thetas_pv_perm = thetas_pv.apply_permutation(zs_idx_pv)

    sorted_dists_pv = np.sort(dists_pv)
    eps_pv = sorted_dists_pv[min(N_particles - 1, N_vanilla - 1)]
    mask = dists_pv <= eps_pv
    n_acc_pv = int(np.sum(mask))
    thetas_acc = thetas_pv_perm[mask]
    if n_acc_pv > N_particles:
        rng_sub = np.random.default_rng(42)
        idx = rng_sub.choice(n_acc_pv, N_particles, replace=False)
        thetas_acc = thetas_acc[idx]
        n_acc_pv = N_particles
    sw2_pv = sliced_w2_joint(model_raw, y_obs_raw, thetas_acc,
                              n_projections=200, n_ref_samples=N_particles, seed=0)
    results.append(("permABC-Vanilla", actual_van, eps_pv, sw2_pv,
                     n_acc_pv, n_acc_pv, "n_accepted"))
    print(f"  eps={eps_pv:.4f}, n_accepted={n_acc_pv}, SW2={sw2_pv:.4f}")

    # ── 4. ABC-SMC (raw data) ──
    print("\n" + "=" * 60)
    print(f"ABC-SMC (raw data, N_sim_max={budget:,})")
    print("=" * 60)
    key, k_smc = random.split(key)
    model_raw.reset_weights_distance()
    out_abc_smc = abc_smc(
        key=k_smc, model=model_raw, n_particles=N_particles,
        epsilon_target=0, y_obs=y_obs_raw, kernel=KernelTruncatedRW,
        N_sim_max=budget, stopping_accept_rate=0.015,
        Final_iteration=200, update_weights_distance=False,
        verbose=1,
    )
    n_sim_abc_smc = int(np.sum(out_abc_smc["N_sim"]))
    thetas_abc_smc = out_abc_smc["Thetas"][-1]
    weights_abc_smc = out_abc_smc["Weights"][-1]
    eps_abc_smc = out_abc_smc["Eps_values"][-1]
    sw2_abc_smc = sliced_w2_joint(model_raw, y_obs_raw, thetas_abc_smc,
                                   weights=weights_abc_smc,
                                   n_projections=200, n_ref_samples=N_particles, seed=0)
    n_uniq_abc_smc = unique_count(thetas_abc_smc)
    ess_w_abc_smc = ess_from_weights(weights_abc_smc)
    results.append(("ABC-SMC (raw)", n_sim_abc_smc, eps_abc_smc, sw2_abc_smc,
                     n_uniq_abc_smc, int(ess_w_abc_smc), "unique/ESS_w"))
    print(f"\nABC-SMC: N_sim={n_sim_abc_smc:,}, eps={eps_abc_smc:.4f}, SW2={sw2_abc_smc:.4f}")
    print(f"  unique={n_uniq_abc_smc}/{N_particles}, ESS_w={ess_w_abc_smc:.0f}")

    # ── 5. permABC-SMC (raw data) ──
    print("\n" + "=" * 60)
    print(f"permABC-SMC (raw data, N_sim_max={budget:,})")
    print("=" * 60)
    key, k_smc = random.split(key)
    model_raw.reset_weights_distance()
    out_perm_raw = perm_abc_smc(
        key=k_smc, model=model_raw, n_particles=N_particles,
        epsilon_target=0, y_obs=y_obs_raw, kernel=KernelTruncatedRW,
        N_sim_max=budget, stopping_accept_rate=0.015,
        Final_iteration=200, update_weights_distance=False,
        verbose=1, try_lsa=True,
    )
    n_sim_perm_raw = int(np.sum(out_perm_raw["N_sim"]))
    thetas_perm_raw = out_perm_raw["Thetas"][-1]
    weights_perm_raw = out_perm_raw["Weights"][-1]
    eps_perm_raw = out_perm_raw["Eps_values"][-1]
    sw2_perm_raw = sliced_w2_joint(model_raw, y_obs_raw, thetas_perm_raw,
                                    weights=weights_perm_raw,
                                    n_projections=200, n_ref_samples=N_particles, seed=0)
    n_uniq_perm_raw = unique_count(thetas_perm_raw)
    ess_w_perm_raw = ess_from_weights(weights_perm_raw)
    results.append(("permABC-SMC (raw)", n_sim_perm_raw, eps_perm_raw, sw2_perm_raw,
                     n_uniq_perm_raw, int(ess_w_perm_raw), "unique/ESS_w"))
    print(f"\npermABC-SMC (raw): N_sim={n_sim_perm_raw:,}, eps={eps_perm_raw:.4f}, SW2={sw2_perm_raw:.4f}")
    print(f"  unique={n_uniq_perm_raw}/{N_particles}, ESS_w={ess_w_perm_raw:.0f}")

    # ── 6. ABC-Gibbs (summary stats, T=1000) ──
    print("\n" + "=" * 60)
    print("ABC-Gibbs (summary stats)")
    print("=" * 60)
    y_obs_2d = np.array(y_obs_raw[0])
    y_mean_k = np.mean(y_obs_2d, axis=1)
    y_std_k = np.std(y_obs_2d, axis=1)

    T_gibbs = 1000
    M = budget // (2 * K * T_gibbs)
    actual_nsim_gibbs = 2 * K * T_gibbs * M
    print(f"T={T_gibbs}, M={M}, N_sim=2×{K}×{T_gibbs}×{M}={actual_nsim_gibbs:,} (budget={budget:,})")

    rng_gibbs = np.random.default_rng(seed + 777)
    mus_chain = np.zeros((T_gibbs + 1, K))
    s2_chain = np.zeros(T_gibbs + 1)
    mus_chain[0] = rng_gibbs.normal(model_raw.mu_0, model_raw.sigma_0, size=K)
    s2_chain[0] = invgamma.rvs(model_raw.alpha, scale=model_raw.beta, random_state=rng_gibbs)

    t0 = _time.perf_counter()
    for t in range(T_gibbs):
        sigma_t = np.sqrt(max(s2_chain[t], 1e-15))
        for k in range(K):
            mu_cands = rng_gibbs.normal(model_raw.mu_0, model_raw.sigma_0, size=M)
            noise = rng_gibbs.normal(size=(M, model_raw.n_obs))
            sims = mu_cands[:, None] + sigma_t * noise
            sim_means = np.mean(sims, axis=1)
            dists_k = (sim_means - y_mean_k[k]) ** 2
            mus_chain[t + 1, k] = mu_cands[np.argmin(dists_k)]

        s2_cands = invgamma.rvs(model_raw.alpha, scale=model_raw.beta,
                                size=M, random_state=rng_gibbs)
        s2_cands = np.maximum(s2_cands, 1e-15)
        dists_s2 = np.empty(M)
        for m in range(M):
            sig_m = np.sqrt(s2_cands[m])
            noise = rng_gibbs.normal(size=(K, model_raw.n_obs))
            sims_m = mus_chain[t + 1, :, None] + sig_m * noise
            sim_means_m = np.mean(sims_m, axis=1)
            sim_stds_m = np.std(sims_m, axis=1)
            dists_s2[m] = np.sum((sim_means_m - y_mean_k) ** 2
                                 + (sim_stds_m - y_std_k) ** 2)
        s2_chain[t + 1] = s2_cands[np.argmin(dists_s2)]
    time_gibbs = _time.perf_counter() - t0
    print(f"Gibbs done in {time_gibbs:.1f}s")

    burnin = int(T_gibbs * 0.2)
    mus_post = mus_chain[burnin:]
    s2_post = s2_chain[burnin:]
    n_post = len(s2_post)

    thetas_gibbs = Theta(loc=mus_post[:, :, None], glob=s2_post[:, None])
    sw2_gibbs = sliced_w2_joint(model_raw, y_obs_raw, thetas_gibbs,
                                 n_projections=200, n_ref_samples=N_particles, seed=0)

    min_ess, ess_all = ess_mcmc(mus_post, s2_post)
    ess_sigma2 = ess_all[0]
    print(f"  n_post={n_post} (T={T_gibbs} - burnin={burnin})")
    print(f"  ESS_sigma2={ess_sigma2:.0f}, min_ESS={min_ess:.0f}")
    print(f"  ESS per param: sigma2={ess_all[0]:.0f}, " +
          ", ".join(f"mu{k}={ess_all[k+1]:.0f}" for k in range(K)))
    print(f"  SW2={sw2_gibbs:.4f}")

    results.append(("ABC-Gibbs", actual_nsim_gibbs, np.nan, sw2_gibbs,
                     n_post, int(min_ess), f"n_post/min_ESS"))

    # ── Budget verification ──
    print("\n" + "=" * 70)
    print("BUDGET VERIFICATION (component-simulations)")
    print("=" * 70)
    print(f"  Target budget:               {budget:>12,}")
    print(f"  ABC-Vanilla:    {N_vanilla:,} draws × {K} = {actual_van:>12,}")
    print(f"  permABC-Vanilla: same        = {actual_van:>12,}")
    n_iters_abc_smc = len(out_abc_smc["Eps_values"]) - 1
    print(f"  ABC-SMC:        {n_iters_abc_smc} iters × ~{N_particles}×{K}  = {n_sim_abc_smc:>12,}")
    n_iters_perm_raw = len(out_perm_raw["Eps_values"]) - 1
    print(f"  permABC-SMC(raw): {n_iters_perm_raw} iters × ~{N_particles}×{K} = {n_sim_perm_raw:>12,}")
    print(f"  permABC-SMC(ss): {n_iters_ss} iters × {N_particles}×{K}  = {budget:>12,}")
    print(f"  ABC-Gibbs:      2×{K}×{T_gibbs}×{M}    = {actual_nsim_gibbs:>12,}")

    # ── Final summary ──
    print("\n" + "=" * 90)
    print(f"SUMMARY  (K={K}, N_particles={N_particles}, budget={budget:,})")
    print("=" * 90)
    print(f"{'Method':<25s} {'N_sim':>12s} {'eps':>8s} {'SW2':>8s} {'ratio':>7s}"
          f" {'unique':>7s} {'ESS':>7s}")
    print("-" * 90)
    for label, n_sim, eps, sw2, n_eff1, n_eff2, eff_lbl in results:
        eps_str = f"{eps:.4f}" if np.isfinite(eps) else "n/a"
        print(f"{label:<25s} {n_sim:12,} {eps_str:>8s} {sw2:8.4f} {sw2/sw2_floor:7.2f}"
              f" {n_eff1:7d} {n_eff2:7d}")
    print(f"{'SW2 floor':<25s} {'':>12s} {'':>8s} {sw2_floor:8.4f} {1.00:7.2f}")


if __name__ == "__main__":
    main()
