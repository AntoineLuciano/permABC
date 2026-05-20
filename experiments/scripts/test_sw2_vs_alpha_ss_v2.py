#!/usr/bin/env python3
"""
SW2 vs alpha for vanilla ABC with normalized summary statistics.

1. Pilot 10K draws → calibrate epsilon thresholds + MAD scales
2. For each alpha: draw N_target/alpha samples, accept top alpha fraction
3. Compute SW2 on 1000 accepted particles

K=2, prior close to posterior.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_vs_alpha_ss_v2.py
"""
import sys
import time as _time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from jax import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from permabc.utils.functions import Theta
from diagnostics import sample_true_posterior, sliced_w2_joint


def compute_summary_stats(data):
    return np.concatenate([
        np.mean(data, axis=2, keepdims=True),
        np.std(data, axis=2, keepdims=True),
    ], axis=2)


def mad_scale(ss_sim):
    scales = np.ones(ss_sim.shape[2])
    for d in range(ss_sim.shape[2]):
        vals = ss_sim[:, :, d].ravel()
        mad = float(np.median(np.abs(vals - np.median(vals))))
        if mad > 1e-10:
            scales[d] = 1.0 / mad
    return scales


def ss_distance(ss_sim, ss_obs, scales):
    diff = (ss_obs[0] - ss_sim) * scales[None, None, :]
    return np.sqrt(np.sum(diff ** 2, axis=(1, 2)))


def main():
    K = 2
    seed = 42
    N_target = 10_000
    N_ref = 10_000
    N_pilot = 10_000
    BATCH = 1_000_000  # process in chunks to avoid OOM

    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)

    model = GaussianWithNoSummaryStats(K=K, n_obs=10, sigma_0=0.5, alpha=10.0, beta=10.0)
    true_theta = Theta(loc=np.array([[[0.0], [0.5]]]), glob=np.array([[1.0]]))
    y_obs_raw = model.data_generator(k2, true_theta)
    y_obs_ss = compute_summary_stats(y_obs_raw)

    # Floor
    print("Computing floor...")
    floors = []
    for s in range(5):
        rng = np.random.default_rng(1000 + s)
        mu_f, s2_f = sample_true_posterior(model, y_obs_raw, N_ref, rng=rng)
        t_f = Theta(loc=mu_f[:, :, None], glob=s2_f[:, None])
        floors.append(sliced_w2_joint(model, y_obs_raw, t_f,
                                       n_projections=200, n_ref_samples=N_ref, seed=0))
    floor = np.mean(floors)
    print(f"Floor = {floor:.4f} (std={np.std(floors):.4f})")

    rng_t = np.random.default_rng(42)
    mu_t, s2_t = sample_true_posterior(model, y_obs_raw, 10000, rng=rng_t)
    s2_post_mean = np.mean(s2_t)
    print(f"True posterior: sigma2={s2_post_mean:.3f}, mu1={np.mean(mu_t[:,0]):.3f}\n")

    # Pilot — calibrate epsilon + MAD
    print(f"Pilot: {N_pilot:,} draws...")
    key, k_th, k_dat = random.split(key, 3)
    thetas_pilot = model.prior_generator(k_th, N_pilot)
    zs_pilot = model.data_generator(k_dat, thetas_pilot)
    ss_pilot = compute_summary_stats(np.array(zs_pilot))
    scales = mad_scale(ss_pilot)
    dists_pilot = ss_distance(ss_pilot, y_obs_ss, scales)

    alphas = np.logspace(0, -4, 30)
    epsilons = np.quantile(dists_pilot, alphas)
    print(f"MAD scales: {scales}")
    print(f"Epsilon range: [{epsilons[-1]:.4f}, {epsilons[0]:.4f}]")

    # Estimate: total draws ~ sum(N_target/alpha), at ~3M draws/s
    total_draws = np.sum(np.ceil(N_target / alphas).astype(int))
    print(f"Total draws needed: ~{total_draws:,.0f} ({total_draws/3e6:.0f}s at 3M/s)")
    print(f"+ {len(alphas)} x ~1s for SW2 = ~{len(alphas)}s")
    print()

    # Sweep
    results = []
    stored_particles = {}

    print(f"{'#':>3s} {'alpha':>10s} {'N_draw':>12s} {'eps':>8s} "
          f"{'SW2':>8s} {'ratio':>6s} {'s2':>7s} {'time':>6s}")
    print("-" * 70)

    for i, (alpha_val, eps) in enumerate(zip(alphas, epsilons)):
        t0 = _time.perf_counter()

        # How many draws to get ~N_target accepted
        N_draw = int(np.ceil(N_target / alpha_val))
        # Add 20% margin for stochastic variation
        N_draw = int(N_draw * 1.2)

        # Generate in batches if too large for memory
        loc_acc, glob_acc = [], []
        n_remaining = N_draw
        while n_remaining > 0:
            n_batch = min(n_remaining, BATCH)
            key, k_th, k_dat = random.split(key, 3)
            thetas_b = model.prior_generator(k_th, n_batch)
            zs_b = model.data_generator(k_dat, thetas_b)
            ss_b = compute_summary_stats(np.array(zs_b))
            dists_b = ss_distance(ss_b, y_obs_ss, scales)
            mask = dists_b <= eps
            n_new = int(np.sum(mask))
            if n_new > 0:
                loc_acc.append(np.array(thetas_b.loc)[mask])
                glob_acc.append(np.array(thetas_b.glob)[mask])
            n_remaining -= n_batch

        if not loc_acc:
            print(f"{i+1:3d} {alpha_val:10.1e} {N_draw:12,} {eps:8.4f} {'SKIP':>8s}")
            continue

        loc_all = np.concatenate(loc_acc)
        glob_all = np.concatenate(glob_acc)
        n_total = len(loc_all)

        # Subsample to N_target
        rng_sub = np.random.default_rng(42)
        if n_total > N_target:
            idx = rng_sub.choice(n_total, N_target, replace=False)
            loc_all = loc_all[idx]
            glob_all = glob_all[idx]

        thetas_sel = Theta(loc=loc_all, glob=glob_all)
        sw2 = sliced_w2_joint(model, y_obs_raw, thetas_sel,
                               n_projections=200, n_ref_samples=N_ref, seed=0)
        s2_mean = float(np.mean(glob_all[:, 0]))
        elapsed = _time.perf_counter() - t0

        mu_means = [float(np.mean(loc_all[:, k, 0])) for k in range(K)]
        results.append((alpha_val, eps, sw2, s2_mean, mu_means))
        stored_particles[len(results) - 1] = (loc_all, glob_all)

        print(f"{i+1:3d} {alpha_val:10.1e} {N_draw:12,} {eps:8.4f} "
              f"{sw2:8.4f} {sw2/floor:6.1f}x {s2_mean:7.3f} {elapsed:5.1f}s")

    # ── Plots ──
    alphas_r = np.array([r[0] for r in results])
    sw2_r = np.array([r[2] for r in results])
    s2_r = np.array([r[3] for r in results])
    mu_means_r = np.array([r[4] for r in results])  # (n_res, K)

    # True posterior means
    mu_true_means = np.array([np.mean(mu_t[:, k]) for k in range(K)])
    s2_true_mean = np.mean(s2_t)

    # MSE = (posterior_mean_abc - posterior_mean_true)^2
    mse_mu = np.mean((mu_means_r - mu_true_means[None, :]) ** 2, axis=1)
    mse_s2 = (s2_r - s2_true_mean) ** 2

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) SW2 vs alpha
    ax = axes[0, 0]
    ax.plot(alphas_r, sw2_r, 'o-', color='#1f77b4', ms=5, lw=1.5)
    ax.axhline(floor, color='gray', ls=':', lw=1.5, label=f'Floor = {floor:.4f}')
    ax.set_xlabel(r'$\alpha$ (acceptance rate)')
    ax.set_ylabel(r'$\mathrm{SW}_2$')
    ax.set_title(r'$\mathrm{SW}_2$ vs $\alpha$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
    ax.legend(); ax.grid(True, alpha=0.3)

    # (0,1) MSE mu vs alpha
    ax = axes[0, 1]
    ax.plot(alphas_r, mse_mu, 's-', color='#2ca02c', ms=5, lw=1.5)
    ax.set_xlabel(r'$\alpha$ (acceptance rate)')
    ax.set_ylabel(r'$\mathrm{MSE}(\mu)$')
    ax.set_title(r'$\mathrm{MSE}(\mu)$ vs $\alpha$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # (0,2) MSE sigma2 vs alpha
    ax = axes[0, 2]
    ax.plot(alphas_r, mse_s2, 's-', color='#d62728', ms=5, lw=1.5)
    ax.set_xlabel(r'$\alpha$ (acceptance rate)')
    ax.set_ylabel(r'$\mathrm{MSE}(\sigma^2)$')
    ax.set_title(r'$\mathrm{MSE}(\sigma^2)$ vs $\alpha$')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # Row 2: KDE marginals at 5 selected levels
    n_res = len(results)
    kde_idx = np.linspace(0, n_res - 1, min(5, n_res), dtype=int)
    colors_kde = [plt.cm.viridis(x) for x in np.linspace(0.1, 0.9, len(kde_idx))]

    for col, pidx, plabel in [(0, 0, r'$\mu_1$'), (1, 1, r'$\mu_2$'), (2, -1, r'$\sigma^2$')]:
        ax = axes[1, col]
        true_vals = mu_t[:, pidx] if pidx >= 0 else s2_t
        kde_true = gaussian_kde(true_vals)
        xmin, xmax = np.percentile(true_vals, [0.5, 99.5])

        for ci, ri in enumerate(kde_idx):
            loc_k, glob_k = stored_particles[ri]
            alpha_val = results[ri][0]
            vals = loc_k[:, pidx, 0] if pidx >= 0 else glob_k[:, 0]
            if len(vals) < 10:
                continue
            xmin = min(xmin, np.percentile(vals, 1))
            xmax = max(xmax, np.percentile(vals, 99))
            kde_abc = gaussian_kde(vals)
            xs = np.linspace(xmin, xmax, 300)
            ax.plot(xs, kde_abc(xs), color=colors_kde[ci], lw=1.5,
                    label=rf'$\alpha$={alpha_val:.1e}')

        xs = np.linspace(xmin, xmax, 300)
        ax.plot(xs, kde_true(xs), 'k--', lw=2, label='True')
        ax.set_xlabel(plabel); ax.set_ylabel('Density')
        ax.set_title(f'Marginal {plabel}')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f'K={K}, summary stats (mean,std), MAD-norm, prior≈post', fontsize=14)
    fig.tight_layout()
    out = PROJECT_ROOT / "experiments" / "figures" / "diagnostic_sw2_ss_full.pdf"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
