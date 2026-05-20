#!/usr/bin/env python3
"""
Test: does SW2(permABC-SMC, true) → SW2(true, true) as epsilon → 0 for K=5?

Runs permABC-SMC with many iterations to push epsilon down,
computing SW2 at each checkpoint.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_convergence_K5.py
"""

import sys
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


class GaussianWithSummaryStatsMeanStd(GaussianWithNoSummaryStats):
    """Gaussian model with summary stats (mean, std) + Prangle MAD normalization.

    data_generator returns (mean, std) per component: shape (n_particles, K, 2).
    update_weights_distance computes MAD per obs dimension (K*n_obs approach)
    and stores 1/MAD. distance() and distance_matrices_loc() apply scaling
    to both z and y before computing L2.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_scale = np.ones(2)  # 1/MAD per obs dim (mean, std)

    def data_generator(self, key, thetas):
        raw = super().data_generator(key, thetas)
        means = np.mean(raw, axis=2, keepdims=True)
        stds = np.std(raw, axis=2, keepdims=True)
        return np.concatenate([means, stds], axis=2)  # (n, K, 2)

    def update_weights_distance(self, zs, verbose=0):
        """Prangle: recompute 1/MAD per obs dimension from simulated data."""
        for d in range(zs.shape[2]):
            vals = zs[:, :, d].ravel()
            mad_d = float(np.median(np.abs(vals - np.median(vals))))
            if mad_d > 1e-10:
                self._obs_scale[d] = 1.0 / mad_d
        if verbose > 0:
            print(f"  Prangle scale (1/MAD): {self._obs_scale}")

    def distance(self, zs, y_obs):
        """L2 distance with Prangle scaling per obs dimension."""
        s = self._obs_scale[None, None, :]  # (1, 1, 2)
        diff = (y_obs[0] - zs) * s
        return np.array(np.sqrt(np.sum(diff ** 2, axis=(1, 2))))

    def distance_component(self, z_k, y_k):
        """Distance between single components with Prangle scaling."""
        diff = (y_k - z_k) * self._obs_scale
        return float(np.sum(diff ** 2))

    def distance_matrices_loc(self, zs, y_obs, M=0, L=0):
        """Pairwise distance matrices with Prangle scaling."""
        if M == 0:
            M = self.K
        if L == 0:
            L = self.K
        # Scale both z and y
        s = self._obs_scale  # (2,)
        zs_s = zs[:, :M] * s[None, None, :]
        y_s = y_obs[0, :L] * s[None, :]

        n_particles = zs_s.shape[0]
        matrices = np.zeros((n_particles, 2 * self.K - L, self.K + M - L))
        for i in range(n_particles):
            for k in range(L):
                for m in range(M):
                    matrices[i, k, m] = np.sum((y_s[k] - zs_s[i, m]) ** 2)
        return matrices


def compute_sw2_floor(model, y_obs, n_particles, n_projections=200, seed=0):
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed + 1000)
    mu1, s2_1 = sample_true_posterior(model, y_obs, n_particles, rng=rng1)
    mu2, s2_2 = sample_true_posterior(model, y_obs, n_particles, rng=rng2)

    ref_joint = np.column_stack([mu2, s2_2])
    abc_joint = np.column_stack([mu1, s2_1])
    dim = abc_joint.shape[1]

    rng_proj = np.random.default_rng(seed + 1)
    directions = rng_proj.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    sw2_sq = 0.0
    for omega in directions:
        a_sorted = np.sort(abc_joint @ omega)
        b_sorted = np.sort(ref_joint @ omega)
        sw2_sq += np.mean((a_sorted - b_sorted) ** 2)
    sw2_sq /= n_projections
    return float(np.sqrt(sw2_sq))


def main():
    K = 3
    N_particles = 5000
    seed = 42

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model_raw = GaussianWithNoSummaryStats(
        K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0,
    )
    model = GaussianWithSummaryStatsMeanStd(
        K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0,
    )

    # Generate true parameters and observed data
    true_theta = model_raw.prior_generator(k1, 1)
    glob = np.array(true_theta.glob); loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0
    loc[0, :, 0] = 0.0
    true_theta = Theta(glob=glob, loc=loc)
    y_obs_raw = model_raw.data_generator(k2, true_theta)
    # Summary stats for y_obs: (mean, std) per component
    y_means = np.mean(y_obs_raw, axis=2, keepdims=True)
    y_stds = np.std(y_obs_raw, axis=2, keepdims=True)
    y_obs = np.concatenate([y_means, y_stds], axis=2)  # (1, K, 2)

    # SW2 floor
    print(f"Computing SW2(true, true) floor for N={N_particles}...")
    sw2_floor = compute_sw2_floor(model_raw, y_obs_raw, N_particles)
    print(f"  SW2 floor = {sw2_floor:.4f}")

    # Run permABC-SMC with many iterations
    print(f"\nRunning permABC-SMC with K={K}, N_particles={N_particles}...")
    key, k_smc = random.split(key)

    result = perm_abc_smc(
        key=k_smc,
        model=model,
        n_particles=N_particles,
        epsilon_target=0,
        y_obs=y_obs,
        kernel=KernelTruncatedRW,
        N_sim_max=np.inf,
        stopping_accept_rate=0.015,
        Final_iteration=200,
        update_weights_distance=True,
        verbose=1,
        try_lsa=True,
    )

    thetas_history = result["Thetas"]
    epsilon_history = result["Eps_values"]
    weights_history = result["Weights"]

    print(f"\n{'iter':>5s} {'epsilon':>10s} {'SW2':>10s} {'SW2_floor':>10s} {'ratio':>8s}")
    print("-" * 50)

    for i, (thetas_i, eps_i, w_i) in enumerate(zip(thetas_history, epsilon_history, weights_history)):
        sw2_i = sliced_w2_joint(
            model_raw, y_obs_raw, thetas_i, weights=w_i,
            n_projections=200, n_ref_samples=5000, seed=0,
        )
        ratio = sw2_i / sw2_floor if sw2_floor > 0 else np.nan
        print(f"  {i:4d} {eps_i:10.4f} {sw2_i:10.4f} {sw2_floor:10.4f} {ratio:8.2f}")

    print(f"\nFinal epsilon: {epsilon_history[-1]:.4f}")
    print(f"Final SW2:     {sw2_i:.4f}")
    print(f"SW2 floor:     {sw2_floor:.4f}")
    print(f"Ratio:         {sw2_i / sw2_floor:.2f}")


if __name__ == "__main__":
    main()
