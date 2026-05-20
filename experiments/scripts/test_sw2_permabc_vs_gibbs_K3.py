#!/usr/bin/env python3
"""
Compare SW2 convergence: permABC-SMC (summary stats + Prangle) vs ABC-Gibbs for K=3.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_permabc_vs_gibbs_K3.py
"""

import sys
from pathlib import Path

import numpy as np
from jax import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from scipy.stats import invgamma

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.utils.functions import Theta
from diagnostics import sample_true_posterior, sliced_w2_joint


def compute_sw2_floor(model, y_obs, n_particles, seed=0):
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1000)
    mu1, s2_1 = sample_true_posterior(model, y_obs, n_particles, rng=rng1)
    mu2, s2_2 = sample_true_posterior(model, y_obs, n_particles, rng=rng2)
    ref = np.column_stack([mu2, s2_2])
    abc = np.column_stack([mu1, s2_1])
    dim = abc.shape[1]
    rng_proj = np.random.default_rng(seed + 1)
    dirs = rng_proj.standard_normal((200, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    sq = 0.0
    for d in dirs:
        sq += np.mean((np.sort(abc @ d) - np.sort(ref @ d)) ** 2)
    return float(np.sqrt(sq / 200))


class GaussianSummaryStats(GaussianWithNoSummaryStats):
    """Summary stats (mean, std) + Prangle MAD normalization."""

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


def main():
    K = 3
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

    true_theta = model_raw.prior_generator(k1, 1)
    glob = np.array(true_theta.glob); loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0; loc[0, :, 0] = 0.0
    true_theta = Theta(glob=glob, loc=loc)
    y_obs_raw = model_raw.data_generator(k2, true_theta)

    y_means = np.mean(y_obs_raw, axis=2, keepdims=True)
    y_stds = np.std(y_obs_raw, axis=2, keepdims=True)
    y_obs_ss = np.concatenate([y_means, y_stds], axis=2)

    print(f"K={K}, N_particles={N_particles}")
    sw2_floor = compute_sw2_floor(model_raw, y_obs_raw, N_particles)
    print(f"SW2 floor = {sw2_floor:.4f}\n")

    # ── 1. permABC-SMC with summary stats ──
    print("=" * 60)
    print("permABC-SMC (summary stats + Prangle MAD)")
    print("=" * 60)
    key, k_smc = random.split(key)
    result_smc = perm_abc_smc(
        key=k_smc, model=model_ss, n_particles=N_particles,
        epsilon_target=0, y_obs=y_obs_ss, kernel=KernelTruncatedRW,
        N_sim_max=np.inf, stopping_accept_rate=0.015,
        Final_iteration=200, update_weights_distance=True,
        verbose=1, try_lsa=True,
    )

    n_sim_smc = int(np.sum(result_smc["N_sim"]))
    thetas_smc = result_smc["Thetas"][-1]
    weights_smc = result_smc["Weights"][-1]
    eps_smc = result_smc["Eps_values"][-1]
    sw2_smc = sliced_w2_joint(model_raw, y_obs_raw, thetas_smc, weights=weights_smc,
                               n_projections=200, n_ref_samples=5000, seed=0)
    print(f"\npermABC-SMC: N_sim={n_sim_smc:,}, eps={eps_smc:.4f}, SW2={sw2_smc:.4f}")

    # ── 2. ABC-Gibbs with same budget (summary stats) ──
    print("\n" + "=" * 60)
    print(f"ABC-Gibbs with summary stats (budget = {n_sim_smc:,} sims)")
    print("=" * 60)
    y_obs_2d = np.array(y_obs_raw[0])  # (K, n_obs)
    # Precompute observed summary stats
    y_mean_k = np.mean(y_obs_2d, axis=1)  # (K,)
    y_std_k = np.std(y_obs_2d, axis=1)    # (K,)

    M_mu = 100
    M_sigma2 = 100
    n_sim_per_iter = K * M_mu + M_sigma2
    T_gibbs = n_sim_smc // n_sim_per_iter
    print(f"M_mu={M_mu}, M_sigma2={M_sigma2}, T={T_gibbs}, sims/iter={n_sim_per_iter}")

    # Run ABC-Gibbs with summary stats
    rng_gibbs = np.random.default_rng(seed + 777)
    mus_chain = np.zeros((T_gibbs + 1, K))
    s2_chain = np.zeros(T_gibbs + 1)
    # Init from prior
    mus_chain[0] = rng_gibbs.normal(model_raw.mu_0, model_raw.sigma_0, size=K)
    s2_chain[0] = invgamma.rvs(model_raw.alpha, scale=model_raw.beta, random_state=rng_gibbs)

    for t in range(T_gibbs):
        sigma_t = np.sqrt(max(s2_chain[t], 1e-15))
        # Step 1: update mu_k using mean(z_k) as summary stat
        for k in range(K):
            mu_cands = rng_gibbs.normal(model_raw.mu_0, model_raw.sigma_0, size=M_mu)
            noise = rng_gibbs.normal(size=(M_mu, model_raw.n_obs))
            sims = mu_cands[:, None] + sigma_t * noise
            sim_means = np.mean(sims, axis=1)  # (M_mu,)
            dists = (sim_means - y_mean_k[k]) ** 2
            mus_chain[t + 1, k] = mu_cands[np.argmin(dists)]

        # Step 2: update sigma2 using all mean(z_k), std(z_k) as summary stats
        s2_cands = invgamma.rvs(model_raw.alpha, scale=model_raw.beta,
                                size=M_sigma2, random_state=rng_gibbs)
        s2_cands = np.maximum(s2_cands, 1e-15)
        dists_s2 = np.empty(M_sigma2)
        for m in range(M_sigma2):
            sig_m = np.sqrt(s2_cands[m])
            noise = rng_gibbs.normal(size=(K, model_raw.n_obs))
            sims = mus_chain[t + 1, :, None] + sig_m * noise
            sim_means = np.mean(sims, axis=1)
            sim_stds = np.std(sims, axis=1)
            dists_s2[m] = np.sum((sim_means - y_mean_k) ** 2 + (sim_stds - y_std_k) ** 2)
        s2_chain[t + 1] = s2_cands[np.argmin(dists_s2)]

    # Compute SW2 at various chain lengths
    checkpoints = [T_gibbs // 4, T_gibbs // 2, 3 * T_gibbs // 4, T_gibbs]
    burnin_frac = 0.2

    print(f"\n{'T':>8s} {'N_sim':>10s} {'SW2':>10s} {'ratio':>8s}")
    print("-" * 40)
    for T_check in checkpoints:
        burnin = int(T_check * burnin_frac)
        mus_sub = mus_chain[burnin:T_check]
        s2_sub = s2_chain[burnin:T_check]
        n_sub = len(mus_sub)
        thetas_gibbs = Theta(
            loc=mus_sub[:, :, None],
            glob=s2_sub[:, None],
        )
        sw2_gibbs = sliced_w2_joint(model_raw, y_obs_raw, thetas_gibbs,
                                     n_projections=200, n_ref_samples=5000, seed=0)
        nsim_check = T_check * n_sim_per_iter
        ratio = sw2_gibbs / sw2_floor
        print(f"  {T_check:6d} {nsim_check:10,} {sw2_gibbs:10.4f} {ratio:8.2f}")

    print(f"\n{'Method':<30s} {'N_sim':>10s} {'SW2':>10s} {'ratio':>8s}")
    print("-" * 60)
    print(f"{'permABC-SMC (summary+Prangle)':<30s} {n_sim_smc:10,} {sw2_smc:10.4f} {sw2_smc/sw2_floor:8.2f}")
    sw2_gibbs_final = sliced_w2_joint(model_raw, y_obs_raw,
        Theta(loc=mus_chain[int(T_gibbs*0.2):T_gibbs, :, None],
              glob=s2_chain[int(T_gibbs*0.2):T_gibbs, None]),
        n_projections=200, n_ref_samples=5000, seed=0)
    print(f"{'ABC-Gibbs (summary stats)':<30s} {n_sim_smc:10,} {sw2_gibbs_final:10.4f} {sw2_gibbs_final/sw2_floor:8.2f}")
    print(f"{'SW2 floor':<30s} {'':>10s} {sw2_floor:10.4f} {1.0:8.2f}")


if __name__ == "__main__":
    main()
