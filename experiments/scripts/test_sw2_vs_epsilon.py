#!/usr/bin/env python3
"""
Plot SW2 vs epsilon for vanilla rejection ABC.

Shows the convergence (or lack thereof) of SW2 towards the floor
as epsilon decreases, for different observation dimensions.

K=2, peaked prior.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_vs_epsilon.py
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


def run_vanilla_at_quantiles(model, y_obs, key, N_sim, N_ref, n_quantiles=20):
    """Run vanilla ABC and compute SW2 at multiple epsilon quantiles."""
    key, k_th, k_dat = random.split(key, 3)
    model.reset_weights_distance()
    thetas = model.prior_generator(k_th, N_sim)
    zs = model.data_generator(k_dat, thetas)
    dists = np.array(model.distance(zs, y_obs))

    # Log-spaced quantiles from generous (0.1) to tight (N_ref/N_sim)
    min_q = max(N_ref / N_sim, 1e-5)
    quantiles = np.logspace(np.log10(min(0.1, 1.0)), np.log10(min_q), n_quantiles)
    quantiles = np.unique(np.clip(quantiles, min_q, 0.5))

    results = []
    for q in quantiles:
        n_acc = max(int(q * N_sim), N_ref)
        if n_acc > N_sim:
            continue
        top_idx = np.argsort(dists)[:n_acc]
        eps = float(dists[top_idx[-1]])

        # Subsample to N_ref if needed
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

        sigma2_mean = float(np.mean(np.array(thetas_acc.glob)[:, 0]))
        results.append((eps, sw2, sigma2_mean, n_acc))
        print(f"    q={q:.1e}  n_acc={n_acc:>7,}  eps={eps:.4f}  "
              f"SW2={sw2:.4f}  sigma2_mean={sigma2_mean:.3f}")

    return results


def main():
    K = 2
    seed = 42
    N_ref = 5000
    N_sim = 10_000_000

    key = random.PRNGKey(seed)
    key, k1, k2, k3 = random.split(key, 4)

    # Model: K=2, peaked prior
    model = GaussianWithNoSummaryStats(K=K, n_obs=10, sigma_0=1.0, alpha=5.0, beta=5.0)
    true_theta = Theta(loc=np.array([[[0.0], [2.0]]]), glob=np.array([[1.0]]))
    y_obs = model.data_generator(k2, true_theta)

    # Floor
    print("Computing floor...")
    rng_f = np.random.default_rng(2000)
    mu_f, s2_f = sample_true_posterior(model, y_obs, N_ref, rng=rng_f)
    t_f = Theta(loc=mu_f[:, :, None], glob=s2_f[:, None])
    floor = sliced_w2_joint(model, y_obs, t_f,
                             n_projections=200, n_ref_samples=N_ref, seed=0)
    print(f"Floor = {floor:.6f}")

    # True posterior stats
    print("\nTrue posterior: sigma2 mean={:.3f}, mu1 mean={:.3f}".format(
        np.mean(s2_f), np.mean(mu_f[:, 0])))

    # SW2 vs epsilon for vanilla rejection
    print(f"\nVanilla rejection ABC (N_sim={N_sim:,}, N_ref={N_ref})")
    print("=" * 70)
    results = run_vanilla_at_quantiles(model, y_obs, key, N_sim, N_ref, n_quantiles=25)

    # Plot
    epsilons = [r[0] for r in results]
    sw2s = [r[1] for r in results]
    sigma2s = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: SW2 vs epsilon
    ax1.plot(epsilons, sw2s, 'o-', color='#1f77b4', markersize=5, linewidth=1.5, label='ABC-Vanilla')
    ax1.axhline(floor, color='gray', linestyle=':', linewidth=1.5, label=f'Floor = {floor:.4f}')
    ax1.set_xlabel(r'$\varepsilon$')
    ax1.set_ylabel(r'$\mathrm{SW}_2$')
    ax1.set_title(f'SW2 vs epsilon (K={K}, n_obs=10, N_sim={N_sim:,})')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: mean sigma2 of accepted particles vs epsilon
    ax2.plot(epsilons, sigma2s, 's-', color='#d62728', markersize=5, linewidth=1.5, label=r'$\langle\sigma^2\rangle$ accepted')
    ax2.axhline(np.mean(s2_f), color='gray', linestyle=':', linewidth=1.5,
                label=f'True posterior mean = {np.mean(s2_f):.3f}')
    ax2.set_xlabel(r'$\varepsilon$')
    ax2.set_ylabel(r'$\langle\sigma^2\rangle$')
    ax2.set_title(r'Posterior $\sigma^2$ bias vs $\varepsilon$')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = PROJECT_ROOT / "experiments" / "figures" / "diagnostic_sw2_vs_epsilon.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
