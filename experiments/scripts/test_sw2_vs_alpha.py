#!/usr/bin/env python3
"""
SW2 vs alpha (acceptance rate) and vs epsilon for vanilla ABC.

- Generate N_sim draws from the prior
- For each alpha in logspace(0, -3, 20): accept top alpha fraction
- Subsample/pad to 1000 particles, compute SW2
- Plot: SW2 vs alpha, SW2 vs epsilon, sigma2 bias vs alpha

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_vs_alpha.py
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


def main():
    K = 2
    seed = 42
    N_sw2 = 1000       # particles for SW2 computation
    N_ref = 1000        # reference true posterior samples
    N_sim = 10_000_000  # total vanilla draws

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model = GaussianWithNoSummaryStats(
        K=K, n_obs=10, sigma_0=1.0, alpha=5.0, beta=5.0,
    )
    true_theta = Theta(loc=np.array([[[0.0], [2.0]]]), glob=np.array([[1.0]]))
    y_obs = model.data_generator(k2, true_theta)

    # Floor
    print("Computing floor (N_ref=1000)...")
    floors = []
    for s in range(5):
        rng = np.random.default_rng(1000 + s)
        mu_f, s2_f = sample_true_posterior(model, y_obs, N_ref, rng=rng)
        t_f = Theta(loc=mu_f[:, :, None], glob=s2_f[:, None])
        f = sliced_w2_joint(model, y_obs, t_f,
                             n_projections=200, n_ref_samples=N_ref, seed=0)
        floors.append(f)
    floor = np.mean(floors)
    print(f"Floor = {floor:.6f} (std={np.std(floors):.6f})")

    # True posterior stats
    rng_t = np.random.default_rng(42)
    mu_t, s2_t = sample_true_posterior(model, y_obs, 10000, rng=rng_t)
    print(f"True posterior: sigma2 mean={np.mean(s2_t):.3f}, mu1 mean={np.mean(mu_t[:,0]):.3f}")

    # Generate all vanilla draws at once
    print(f"\nGenerating {N_sim:,} vanilla draws...")
    key, k_th, k_dat = random.split(key, 3)
    model.reset_weights_distance()
    thetas = model.prior_generator(k_th, N_sim)
    zs = model.data_generator(k_dat, thetas)
    dists = np.array(model.distance(zs, y_obs))
    sort_idx = np.argsort(dists)
    print("Done.")

    # Alpha grid: 1 down to 0.001
    alphas = np.logspace(0, -3, 25)
    # Also ensure we can get at least N_sw2 particles
    alphas = alphas[alphas * N_sim >= N_sw2]

    epsilons, sw2s, s2_means, mu1_means = [], [], [], []

    print(f"\n{'alpha':>10s} {'n_acc':>10s} {'epsilon':>10s} {'SW2':>10s} {'sigma2':>10s} {'mu1':>10s}")
    print("-" * 65)
    for alpha in alphas:
        n_acc = int(alpha * N_sim)
        if n_acc < N_sw2:
            n_acc = N_sw2

        top = sort_idx[:n_acc]
        eps = float(dists[top[-1]])

        # Subsample to N_sw2
        rng_sub = np.random.default_rng(42)
        if n_acc > N_sw2:
            sub = rng_sub.choice(n_acc, N_sw2, replace=False)
            sel = top[sub]
        else:
            sel = top

        thetas_sel = Theta(
            loc=np.array(thetas.loc)[sel],
            glob=np.array(thetas.glob)[sel],
        )

        sw2 = sliced_w2_joint(model, y_obs, thetas_sel,
                               n_projections=200, n_ref_samples=N_ref, seed=0)

        s2_mean = float(np.mean(np.array(thetas_sel.glob)[:, 0]))
        mu1_mean = float(np.mean(np.array(thetas_sel.loc)[:, 0, 0]))

        epsilons.append(eps)
        sw2s.append(sw2)
        s2_means.append(s2_mean)
        mu1_means.append(mu1_mean)

        print(f"{alpha:10.4f} {n_acc:10,} {eps:10.4f} {sw2:10.4f} {s2_mean:10.3f} {mu1_mean:10.3f}")

    alphas_plot = alphas[:len(sw2s)]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: SW2 vs alpha
    ax = axes[0]
    ax.plot(alphas_plot, sw2s, 'o-', color='#1f77b4', markersize=5, linewidth=1.5)
    ax.axhline(floor, color='gray', linestyle=':', linewidth=1.5, label=f'Floor = {floor:.4f}')
    ax.set_xlabel(r'$\alpha$ (acceptance rate)')
    ax.set_ylabel(r'$\mathrm{SW}_2$')
    ax.set_title(r'$\mathrm{SW}_2$ vs acceptance rate $\alpha$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: SW2 vs epsilon
    ax = axes[1]
    ax.plot(epsilons, sw2s, 'o-', color='#1f77b4', markersize=5, linewidth=1.5)
    ax.axhline(floor, color='gray', linestyle=':', linewidth=1.5, label=f'Floor = {floor:.4f}')
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\mathrm{SW}_2$')
    ax.set_title(r'$\mathrm{SW}_2$ vs $\varepsilon$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: sigma2 mean vs alpha
    ax = axes[2]
    ax.plot(alphas_plot, s2_means, 's-', color='#d62728', markersize=5, linewidth=1.5)
    ax.axhline(np.mean(s2_t), color='gray', linestyle=':', linewidth=1.5,
               label=f'True post. mean = {np.mean(s2_t):.3f}')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label=r'True $\sigma^2 = 1$')
    ax.set_xlabel(r'$\alpha$ (acceptance rate)')
    ax.set_ylabel(r'$\langle\sigma^2\rangle$ accepted')
    ax.set_title(r'Posterior $\sigma^2$ bias')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'K={K}, n_obs=10, N_sim={N_sim:,}, N_sw2={N_sw2}', fontsize=13)
    fig.tight_layout()

    out = PROJECT_ROOT / "experiments" / "figures" / "diagnostic_sw2_vs_alpha.pdf"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
