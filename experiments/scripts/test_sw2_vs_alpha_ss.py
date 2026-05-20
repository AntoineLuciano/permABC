#!/usr/bin/env python3
"""
SW2 vs alpha for vanilla ABC with normalized summary statistics.

Summary stats: (mean, std) per component, MAD-normalized.
K=2, n_obs=10 → summary space = 4D (vs 20D raw).

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_vs_alpha_ss.py
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


def compute_summary_stats(data):
    """(N, K, n_obs) → (N, K, 2) with mean and std."""
    return np.concatenate([
        np.mean(data, axis=2, keepdims=True),
        np.std(data, axis=2, keepdims=True),
    ], axis=2)


def mad_scale(ss_sim):
    """Compute MAD normalization weights from (N, K, d_ss)."""
    scales = np.ones(ss_sim.shape[2])
    for d in range(ss_sim.shape[2]):
        vals = ss_sim[:, :, d].ravel()
        mad = float(np.median(np.abs(vals - np.median(vals))))
        if mad > 1e-10:
            scales[d] = 1.0 / mad
    return scales


def ss_distance(ss_sim, ss_obs, scales):
    """Euclidean distance on normalized summary stats.
    ss_sim: (N, K, d_ss), ss_obs: (1, K, d_ss), scales: (d_ss,)
    Returns: (N,)
    """
    diff = (ss_obs[0] - ss_sim) * scales[None, None, :]
    return np.sqrt(np.sum(diff ** 2, axis=(1, 2)))


def main():
    K = 2
    seed = 42
    N_sw2 = 1000
    N_ref = 1000
    N_sim = 10_000_000

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model = GaussianWithNoSummaryStats(
        K=K, n_obs=10, sigma_0=1.0, alpha=5.0, beta=5.0,
    )
    true_theta = Theta(loc=np.array([[[0.0], [2.0]]]), glob=np.array([[1.0]]))
    y_obs_raw = model.data_generator(k2, true_theta)
    y_obs_ss = compute_summary_stats(y_obs_raw)

    # Floor (SW2 always computed on (mu, sigma2) parameter space, not obs space)
    print("Computing floor (N_ref=1000)...")
    floors = []
    for s in range(5):
        rng = np.random.default_rng(1000 + s)
        mu_f, s2_f = sample_true_posterior(model, y_obs_raw, N_ref, rng=rng)
        t_f = Theta(loc=mu_f[:, :, None], glob=s2_f[:, None])
        f = sliced_w2_joint(model, y_obs_raw, t_f,
                             n_projections=200, n_ref_samples=N_ref, seed=0)
        floors.append(f)
    floor = np.mean(floors)
    print(f"Floor = {floor:.6f} (std={np.std(floors):.6f})")

    # True posterior stats
    rng_t = np.random.default_rng(42)
    mu_t, s2_t = sample_true_posterior(model, y_obs_raw, 10000, rng=rng_t)
    print(f"True posterior: sigma2 mean={np.mean(s2_t):.3f}, mu1 mean={np.mean(mu_t[:,0]):.3f}")

    # Generate all vanilla draws
    print(f"\nGenerating {N_sim:,} vanilla draws + summary stats...")
    key, k_th, k_dat = random.split(key, 3)
    thetas = model.prior_generator(k_th, N_sim)
    zs_raw = model.data_generator(k_dat, thetas)
    zs_ss = compute_summary_stats(np.array(zs_raw))

    # MAD normalization
    scales = mad_scale(zs_ss)
    print(f"MAD scales: {scales}")

    dists = ss_distance(zs_ss, y_obs_ss, scales)
    sort_idx = np.argsort(dists)
    print("Done.")

    # Alpha grid
    alphas = np.logspace(0, -4, 30)
    alphas = alphas[alphas * N_sim >= N_sw2]

    epsilons, sw2s, s2_means = [], [], []

    print(f"\n{'alpha':>10s} {'n_acc':>10s} {'eps_ss':>10s} {'SW2':>10s} {'sigma2':>10s}")
    print("-" * 55)
    for alpha_val in alphas:
        n_acc = max(int(alpha_val * N_sim), N_sw2)
        if n_acc > N_sim:
            continue

        top = sort_idx[:n_acc]
        eps = float(dists[top[-1]])

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

        # SW2 on parameter space (same as before — compares to true posterior)
        sw2 = sliced_w2_joint(model, y_obs_raw, thetas_sel,
                               n_projections=200, n_ref_samples=N_ref, seed=0)

        s2_mean = float(np.mean(np.array(thetas_sel.glob)[:, 0]))

        epsilons.append(eps)
        sw2s.append(sw2)
        s2_means.append(s2_mean)

        print(f"{alpha_val:10.4f} {n_acc:10,} {eps:10.4f} {sw2:10.4f} {s2_mean:10.3f}")

    alphas_plot = alphas[:len(sw2s)]

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(alphas_plot, sw2s, 'o-', color='#2ca02c', markersize=5, linewidth=1.5,
            label='Summary stats (4D)')
    ax.axhline(floor, color='gray', linestyle=':', linewidth=1.5, label=f'Floor = {floor:.4f}')
    ax.set_xlabel(r'$\alpha$ (acceptance rate)')
    ax.set_ylabel(r'$\mathrm{SW}_2$')
    ax.set_title(r'$\mathrm{SW}_2$ vs $\alpha$ — summary stats')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epsilons, sw2s, 'o-', color='#2ca02c', markersize=5, linewidth=1.5)
    ax.axhline(floor, color='gray', linestyle=':', linewidth=1.5, label=f'Floor = {floor:.4f}')
    ax.set_xlabel(r'$\varepsilon$ (summary stats)')
    ax.set_ylabel(r'$\mathrm{SW}_2$')
    ax.set_title(r'$\mathrm{SW}_2$ vs $\varepsilon$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(alphas_plot, s2_means, 's-', color='#d62728', markersize=5, linewidth=1.5)
    ax.axhline(np.mean(s2_t), color='gray', linestyle=':', linewidth=1.5,
               label=f'True post. mean = {np.mean(s2_t):.3f}')
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label=r'True $\sigma^2 = 1$')
    ax.set_xlabel(r'$\alpha$ (acceptance rate)')
    ax.set_ylabel(r'$\langle\sigma^2\rangle$ accepted')
    ax.set_title(r'Posterior $\sigma^2$ bias — summary stats')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'K={K}, summary stats (mean,std), MAD-normalized, N_sim={N_sim:,}', fontsize=13)
    fig.tight_layout()

    out = PROJECT_ROOT / "experiments" / "figures" / "diagnostic_sw2_vs_alpha_ss.pdf"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close(fig)

    # ── KDE plot: marginal posteriors vs true for selected alphas ──
    from scipy.stats import gaussian_kde

    # Select ~5 alphas spread across the range
    kde_alphas_idx = np.linspace(0, len(alphas_plot) - 1, 5, dtype=int)
    kde_alphas = alphas_plot[kde_alphas_idx]

    # True posterior samples for KDE
    rng_kde = np.random.default_rng(77)
    mu_true_kde, s2_true_kde = sample_true_posterior(model, y_obs_raw, 10000, rng=rng_kde)

    param_labels = [r'$\mu_1$', r'$\mu_2$', r'$\sigma^2$']

    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    cmap = plt.cm.viridis
    colors_kde = [cmap(x) for x in np.linspace(0.1, 0.9, len(kde_alphas))]

    for ax, pidx, plabel in zip(axes2, range(3), param_labels):
        # True posterior KDE
        if pidx < K:
            true_vals = mu_true_kde[:, pidx]
        else:
            true_vals = s2_true_kde
        kde_true = gaussian_kde(true_vals)
        xmin = np.percentile(true_vals, 0.5)
        xmax = np.percentile(true_vals, 99.5)

        for ci, alpha_val in enumerate(kde_alphas):
            n_acc = max(int(alpha_val * N_sim), N_sw2)
            top = sort_idx[:n_acc]
            eps = float(dists[top[-1]])

            rng_sub = np.random.default_rng(42)
            if n_acc > N_sw2:
                sub = rng_sub.choice(n_acc, N_sw2, replace=False)
                sel = top[sub]
            else:
                sel = top

            if pidx < K:
                vals = np.array(thetas.loc)[sel, pidx, 0]
            else:
                vals = np.array(thetas.glob)[sel, 0]

            xmin = min(xmin, np.percentile(vals, 1))
            xmax = max(xmax, np.percentile(vals, 99))

            kde_abc = gaussian_kde(vals)
            xs = np.linspace(xmin, xmax, 300)
            ax.plot(xs, kde_abc(xs), color=colors_kde[ci], linewidth=1.5,
                    label=rf'$\alpha$={alpha_val:.1e} ($\varepsilon$={eps:.2f})')

        xs = np.linspace(xmin, xmax, 300)
        ax.plot(xs, kde_true(xs), 'k--', linewidth=2, label='True posterior')
        ax.set_xlabel(plabel)
        ax.set_ylabel('Density')
        ax.set_title(f'Marginal {plabel}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig2.suptitle(f'Marginal posteriors: ABC (summary stats) vs true — K={K}', fontsize=13)
    fig2.tight_layout()
    out2 = PROJECT_ROOT / "experiments" / "figures" / "diagnostic_marginals_ss.pdf"
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
