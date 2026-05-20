#!/usr/bin/env python3
"""
SW2 vs epsilon for n_obs=1 (2D) vs n_obs=10 (20D).
Shows that dimension is the key factor.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_vs_epsilon_nobs1.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from jax import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta
from diagnostics import sample_true_posterior, sliced_w2_joint


def compute_floor(model, y_obs, N_ref):
    rng = np.random.default_rng(2000)
    mu, s2 = sample_true_posterior(model, y_obs, N_ref, rng=rng)
    t = Theta(loc=mu[:, :, None], glob=s2[:, None])
    return sliced_w2_joint(model, y_obs, t,
                           n_projections=200, n_ref_samples=N_ref, seed=0)


def sweep_epsilon(model, y_obs, key, N_sim, N_ref, n_points=20):
    key, k_th, k_dat = random.split(key, 3)
    model.reset_weights_distance()
    thetas = model.prior_generator(k_th, N_sim)
    zs = model.data_generator(k_dat, thetas)
    dists = np.array(model.distance(zs, y_obs))

    min_q = max(N_ref / N_sim, 1e-5)
    quantiles = np.logspace(np.log10(0.1), np.log10(min_q), n_points)

    epsilons, sw2s, s2_means = [], [], []
    for q in quantiles:
        n_acc = max(int(q * N_sim), N_ref)
        if n_acc > N_sim:
            continue
        top_idx = np.argsort(dists)[:n_acc]
        eps = float(dists[top_idx[-1]])

        if n_acc > N_ref:
            rng = np.random.default_rng(42)
            sub_idx = rng.choice(n_acc, N_ref, replace=False)
            top_idx = top_idx[sub_idx]

        thetas_acc = Theta(
            loc=np.array(thetas.loc)[top_idx],
            glob=np.array(thetas.glob)[top_idx],
        )
        sw2 = sliced_w2_joint(model, y_obs, thetas_acc,
                               n_projections=200, n_ref_samples=N_ref, seed=0)
        s2_mean = float(np.mean(np.array(thetas_acc.glob)[:, 0]))

        epsilons.append(eps)
        sw2s.append(sw2)
        s2_means.append(s2_mean)
        print(f"    eps={eps:.4f}  SW2={sw2:.4f}  sigma2={s2_mean:.3f}")

    return epsilons, sw2s, s2_means


def main():
    K = 2
    seed = 42
    N_ref = 5000
    N_sim = 10_000_000

    key = random.PRNGKey(seed)

    true_theta = Theta(loc=np.array([[[0.0], [2.0]]]), glob=np.array([[1.0]]))

    configs = [
        ("n_obs=1 (2D)", 1, '#2ca02c'),
        ("n_obs=10 (20D)", 10, '#1f77b4'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, n_obs, color in configs:
        print(f"\n{'='*60}")
        print(f"{label}")
        print(f"{'='*60}")

        model = GaussianWithNoSummaryStats(
            K=K, n_obs=n_obs, sigma_0=1.0, alpha=5.0, beta=5.0)

        key, k_y = random.split(key)
        y_obs = model.data_generator(k_y, true_theta)

        floor = compute_floor(model, y_obs, N_ref)
        print(f"Floor = {floor:.6f}")

        # True posterior
        mu_t, s2_t = sample_true_posterior(model, y_obs, N_ref,
                                            rng=np.random.default_rng(42))
        s2_post_mean = np.mean(s2_t)
        print(f"True posterior: sigma2 mean = {s2_post_mean:.3f}")

        key, k_sw = random.split(key)
        epsilons, sw2s, s2_means = sweep_epsilon(
            model, y_obs, k_sw, N_sim, N_ref, n_points=20)

        ax1.plot(epsilons, sw2s, 'o-', color=color, markersize=4,
                 linewidth=1.5, label=label)
        ax1.axhline(floor, color=color, linestyle=':', linewidth=1, alpha=0.6)

        ax2.plot(epsilons, s2_means, 's-', color=color, markersize=4,
                 linewidth=1.5, label=label)
        ax2.axhline(s2_post_mean, color=color, linestyle=':', linewidth=1, alpha=0.6,
                     label=f'True post. mean ({label})')

    ax1.set_xlabel(r'$\varepsilon$')
    ax1.set_ylabel(r'$\mathrm{SW}_2$')
    ax1.set_title(r'$\mathrm{SW}_2$ vs $\varepsilon$ — curse of dimensionality')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(r'$\varepsilon$')
    ax2.set_ylabel(r'$\langle\sigma^2\rangle$ accepted')
    ax2.set_title(r'Posterior $\sigma^2$ bias vs $\varepsilon$')
    ax2.set_xscale('log')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = PROJECT_ROOT / "experiments" / "figures" / "diagnostic_sw2_vs_epsilon_dim.pdf"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
