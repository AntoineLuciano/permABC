#!/usr/bin/env python3
"""
ABC-Gibbs with T adjusted so n_post = unique(permABC-SMC), same budget.
K=10, same data as test_sw2_all_methods_K10.py.
"""
import sys, time as _time
from pathlib import Path
import numpy as np
from jax import random
from scipy.stats import invgamma

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.utils.functions import Theta
from diagnostics import sample_true_posterior, sliced_w2_joint

# ── Summary stats model (same as K10 script) ──
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


def ess_autocorr_1d(chain):
    n = len(chain)
    if n < 10: return float(n)
    x = chain - np.mean(chain)
    var = np.var(chain)
    if var < 1e-30: return 1.0
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n]
    acf /= acf[0]
    tau = 1.0
    for lag in range(1, n):
        if acf[lag] < 0.05: break
        tau += 2 * acf[lag]
    return n / max(tau, 1.0)


def ess_mcmc(mus_chain, s2_chain):
    ess_vals = [ess_autocorr_1d(s2_chain)]
    for k in range(mus_chain.shape[1]):
        ess_vals.append(ess_autocorr_1d(mus_chain[:, k]))
    return min(ess_vals), ess_vals


def unique_count(thetas):
    loc = np.asarray(thetas.loc)
    if loc.ndim == 3: loc = loc.reshape(loc.shape[0], -1)
    _, idx = np.unique(loc, axis=0, return_index=True)
    return len(idx)


def run_gibbs(model_raw, y_obs_raw, K, T, M, seed=42):
    """Run ABC-Gibbs with summary stats, return chain + diagnostics."""
    y_obs_2d = np.array(y_obs_raw[0])
    y_mean_k = np.mean(y_obs_2d, axis=1)
    y_std_k = np.std(y_obs_2d, axis=1)

    rng = np.random.default_rng(seed + 777)
    mus_chain = np.zeros((T + 1, K))
    s2_chain = np.zeros(T + 1)
    mus_chain[0] = rng.normal(model_raw.mu_0, model_raw.sigma_0, size=K)
    s2_chain[0] = invgamma.rvs(model_raw.alpha, scale=model_raw.beta, random_state=rng)

    t0 = _time.perf_counter()
    for t in range(T):
        sigma_t = np.sqrt(max(s2_chain[t], 1e-15))
        # Block 1: update each mu_k
        for k in range(K):
            mu_cands = rng.normal(model_raw.mu_0, model_raw.sigma_0, size=M)
            noise = rng.normal(size=(M, model_raw.n_obs))
            sims = mu_cands[:, None] + sigma_t * noise
            sim_means = np.mean(sims, axis=1)
            dists_k = (sim_means - y_mean_k[k]) ** 2
            mus_chain[t + 1, k] = mu_cands[np.argmin(dists_k)]
        # Block 2: update sigma2
        s2_cands = invgamma.rvs(model_raw.alpha, scale=model_raw.beta,
                                size=M, random_state=rng)
        s2_cands = np.maximum(s2_cands, 1e-15)
        dists_s2 = np.empty(M)
        for m in range(M):
            sig_m = np.sqrt(s2_cands[m])
            noise = rng.normal(size=(K, model_raw.n_obs))
            sims_m = mus_chain[t + 1, :, None] + sig_m * noise
            sim_means_m = np.mean(sims_m, axis=1)
            sim_stds_m = np.std(sims_m, axis=1)
            dists_s2[m] = np.sum((sim_means_m - y_mean_k) ** 2
                                 + (sim_stds_m - y_std_k) ** 2)
        s2_chain[t + 1] = s2_cands[np.argmin(dists_s2)]
    elapsed = _time.perf_counter() - t0
    return mus_chain, s2_chain, elapsed


def main():
    K = 10
    N_particles = 5000
    seed = 42
    burnin_frac = 0.2

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model_raw = GaussianWithNoSummaryStats(K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0)
    model_ss = GaussianSummaryStats(K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0)

    # Same data
    true_theta = model_raw.prior_generator(k1, 1)
    glob = np.array(true_theta.glob); loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0; loc[0, :, 0] = 0.0
    true_theta = Theta(glob=glob, loc=loc)
    y_obs_raw = model_raw.data_generator(k2, true_theta)
    y_obs_ss = np.concatenate([np.mean(y_obs_raw, axis=2, keepdims=True),
                                np.std(y_obs_raw, axis=2, keepdims=True)], axis=2)

    # ── SW2 floor ──
    print("Computing SW2 floor...")
    rng_floor = np.random.default_rng(2000)
    mu_f, s2_f = sample_true_posterior(model_raw, y_obs_raw, N_particles, rng=rng_floor)
    thetas_floor = Theta(loc=mu_f[:, :, None], glob=s2_f[:, None])
    sw2_floor = sliced_w2_joint(model_raw, y_obs_raw, thetas_floor,
                                 n_projections=200, n_ref_samples=N_particles, seed=0)
    print(f"SW2 floor = {sw2_floor:.4f}\n")

    # ── permABC-SMC (ss) → budget + unique count ──
    print("=" * 60)
    print("Running permABC-SMC (ss) to determine budget + unique count")
    print("=" * 60)
    key, k_smc = random.split(key)
    out = perm_abc_smc(
        key=k_smc, model=model_ss, n_particles=N_particles,
        epsilon_target=0, y_obs=y_obs_ss, kernel=KernelTruncatedRW,
        N_sim_max=np.inf, stopping_accept_rate=0.015,
        Final_iteration=200, update_weights_distance=True,
        verbose=1, try_lsa=True,
    )
    budget = int(np.sum(out["N_sim"]))
    thetas_smc = out["Thetas"][-1]
    weights_smc = out["Weights"][-1]
    n_unique_smc = unique_count(thetas_smc)
    sw2_smc = sliced_w2_joint(model_raw, y_obs_raw, thetas_smc,
                               weights=weights_smc,
                               n_projections=200, n_ref_samples=N_particles, seed=0)

    print(f"\npermABC-SMC (ss): budget={budget:,}, unique={n_unique_smc}/{N_particles}")
    print(f"  SW2={sw2_smc:.4f}, ratio={sw2_smc/sw2_floor:.2f}x floor\n")

    # ── ABC-Gibbs: T adjusted so n_post = n_unique_smc ──
    # n_post = T * (1 - burnin_frac) → T = ceil(n_unique_smc / (1 - burnin_frac))
    T_target = int(np.ceil(n_unique_smc / (1 - burnin_frac)))
    M_target = budget // (2 * K * T_target)
    actual_nsim = 2 * K * T_target * M_target
    burnin = int(T_target * burnin_frac)
    n_post_expected = T_target - burnin + 1  # includes t=0 in chain

    print("=" * 60)
    print(f"ABC-Gibbs (T={T_target}, M={M_target}) — matched unique")
    print("=" * 60)
    print(f"  Target n_post = unique_smc = {n_unique_smc}")
    print(f"  T={T_target}, burnin={burnin}, n_post={T_target - burnin + 1}")
    print(f"  N_sim = 2×{K}×{T_target}×{M_target} = {actual_nsim:,} (budget={budget:,})")

    mus_chain, s2_chain, elapsed = run_gibbs(model_raw, y_obs_raw, K, T_target, M_target, seed)
    print(f"  Done in {elapsed:.1f}s")

    mus_post = mus_chain[burnin:]
    s2_post = s2_chain[burnin:]
    n_post = len(s2_post)

    min_ess, ess_all = ess_mcmc(mus_post, s2_post)
    print(f"  n_post={n_post}, min_ESS={min_ess:.0f}")
    print(f"  ESS: sigma2={ess_all[0]:.0f}, " +
          ", ".join(f"mu{k}={ess_all[k+1]:.0f}" for k in range(K)))

    thetas_gibbs = Theta(loc=mus_post[:, :, None], glob=s2_post[:, None])
    sw2_gibbs = sliced_w2_joint(model_raw, y_obs_raw, thetas_gibbs,
                                 n_projections=200, n_ref_samples=N_particles, seed=0)
    print(f"  SW2={sw2_gibbs:.4f}, ratio={sw2_gibbs/sw2_floor:.2f}x floor")

    # ── Also run original T=1000 for comparison ──
    T_ref = 1000
    M_ref = budget // (2 * K * T_ref)
    actual_ref = 2 * K * T_ref * M_ref
    burnin_ref = int(T_ref * burnin_frac)

    print(f"\n{'='*60}")
    print(f"ABC-Gibbs (T={T_ref}, M={M_ref}) — reference")
    print(f"{'='*60}")
    print(f"  N_sim = 2×{K}×{T_ref}×{M_ref} = {actual_ref:,}")

    mus_ref, s2_ref, elapsed_ref = run_gibbs(model_raw, y_obs_raw, K, T_ref, M_ref, seed)
    print(f"  Done in {elapsed_ref:.1f}s")

    mus_post_ref = mus_ref[burnin_ref:]
    s2_post_ref = s2_ref[burnin_ref:]
    n_post_ref = len(s2_post_ref)
    min_ess_ref, ess_ref = ess_mcmc(mus_post_ref, s2_post_ref)
    print(f"  n_post={n_post_ref}, min_ESS={min_ess_ref:.0f}")

    thetas_ref = Theta(loc=mus_post_ref[:, :, None], glob=s2_post_ref[:, None])
    sw2_ref = sliced_w2_joint(model_raw, y_obs_raw, thetas_ref,
                               n_projections=200, n_ref_samples=N_particles, seed=0)
    print(f"  SW2={sw2_ref:.4f}, ratio={sw2_ref/sw2_floor:.2f}x floor")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"SUMMARY  (K={K}, budget={budget:,}, SW2_floor={sw2_floor:.4f})")
    print(f"{'='*80}")
    print(f"{'Method':<35s} {'N_sim':>12s} {'n_post':>7s} {'ESS':>7s} {'SW2':>8s} {'ratio':>7s}")
    print("-" * 80)
    print(f"{'permABC-SMC (ss)':<35s} {budget:12,} {n_unique_smc:7d} {n_unique_smc:7d} {sw2_smc:8.4f} {sw2_smc/sw2_floor:7.2f}")
    print(f"{'ABC-Gibbs (T='+str(T_target)+', M='+str(M_target)+')':<35s} {actual_nsim:12,} {n_post:7d} {int(min_ess):7d} {sw2_gibbs:8.4f} {sw2_gibbs/sw2_floor:7.2f}")
    print(f"{'ABC-Gibbs (T='+str(T_ref)+', M='+str(M_ref)+')':<35s} {actual_ref:12,} {n_post_ref:7d} {int(min_ess_ref):7d} {sw2_ref:8.4f} {sw2_ref/sw2_floor:7.2f}")
    print(f"{'SW2 floor':<35s} {'':>12s} {'':>7s} {'':>7s} {sw2_floor:8.4f} {'1.00':>7s}")


if __name__ == "__main__":
    main()
