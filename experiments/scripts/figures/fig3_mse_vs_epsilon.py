#!/usr/bin/env python3
"""
Figure 3 supplement: MSE vs epsilon for ABC and permABC on the uniform model.

Shows that permABC has higher MSE than ABC when epsilon > epsilon*
due to the projection bias from label-switching resolution.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/figures/fig3_mse_vs_epsilon.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from jax import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from plot_config import setup_matplotlib, save_figure
from permabc.models.uniform_known import Uniform_known
from permabc.assignment.dispatch import optimal_index_distance

setup_matplotlib()


def run_mse_vs_epsilon(K=2, N=1_000_000, seed=42):
    key = random.PRNGKey(seed)
    key, k1, k2, k3 = random.split(key, 4)

    model = Uniform_known(K=K)
    y_obs = np.array([0.0, 0.5])[None, :, None]  # shape (1, K, 1)

    # True posterior mean: mu_k | y_k ~ Uniform(y_k-1, y_k+1), mean = y_k
    post_mean = np.array([0.0, 0.5])

    # epsilon*
    eps_star = float(model.distance(y_obs, y_obs[:, ::-1])[0]) / 2
    print(f"epsilon* = {eps_star:.4f}")

    # Simulate
    thetas = model.prior_generator(k1, N)
    zs = model.data_generator(k2, thetas)

    # ABC distances (identity)
    dists_abc = np.array(model.distance(zs, y_obs))

    # permABC distances + permuted thetas
    dists_perm, _, zs_idx, _ = optimal_index_distance(
        model=model, zs=zs, y_obs=y_obs, epsilon=0, verbose=1
    )
    thetas_perm = thetas.apply_permutation(zs_idx)

    # Epsilon grid
    eps_grid = np.sort(np.unique(np.concatenate([
        np.linspace(0.01, eps_star * 3, 50),
        [eps_star * r for r in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]],
        np.quantile(dists_abc, [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]),
        np.quantile(dists_perm, [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]),
    ])))

    results = []
    for eps in eps_grid:
        mask_a = dists_abc <= eps
        mask_p = dists_perm <= eps
        n_a, n_p = int(np.sum(mask_a)), int(np.sum(mask_p))

        mse_a = mse_p = np.nan
        if n_a >= 30:
            mu_a = np.array(thetas.loc[mask_a, :, 0])
            mse_a = np.mean((mu_a - post_mean[None, :]) ** 2)
        if n_p >= 30:
            mu_p = np.array(thetas_perm.loc[mask_p, :, 0])
            mse_p = np.mean((mu_p - post_mean[None, :]) ** 2)

        results.append((eps, n_a, n_p, mse_a, mse_p))

    return np.array(results), eps_star


def plot_mse_vs_epsilon(results, eps_star, seed=42):
    eps = results[:, 0]
    n_abc = results[:, 1]
    n_perm = results[:, 2]
    mse_abc = results[:, 3]
    mse_perm = results[:, 4]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: MSE vs epsilon
    valid_a = np.isfinite(mse_abc)
    valid_p = np.isfinite(mse_perm)

    ax1.plot(eps[valid_a], mse_abc[valid_a], "o-", color="#d62728", label="ABC", markersize=4)
    ax1.plot(eps[valid_p], mse_perm[valid_p], "s-", color="#1f77b4", label="permABC", markersize=4)
    ax1.axvline(eps_star, color="gray", linestyle="--", linewidth=1, label=r"$\varepsilon^*$")
    ax1.set_xlabel(r"$\varepsilon$")
    ax1.set_ylabel(r"MSE$(\hat{\mu})$")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("MSE vs tolerance")

    # Panel 2: MSE ratio (permABC / ABC) vs epsilon
    both_valid = valid_a & valid_p & (mse_abc > 0)
    ratio = mse_perm[both_valid] / mse_abc[both_valid]
    ax2.plot(eps[both_valid], ratio, "D-", color="#2ca02c", markersize=4)
    ax2.axvline(eps_star, color="gray", linestyle="--", linewidth=1, label=r"$\varepsilon^*$")
    ax2.axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    ax2.set_xlabel(r"$\varepsilon$")
    ax2.set_ylabel(r"MSE(permABC) / MSE(ABC)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title(r"MSE ratio vs tolerance")

    fig.suptitle(f"Uniform model, K={2}, seed={seed}", fontsize=14)
    fig.tight_layout()
    return fig


def main():
    seed = 42
    results, eps_star = run_mse_vs_epsilon(K=2, N=1_000_000, seed=seed)

    # Print table
    print(f"\n{'epsilon':>10s} {'eps/eps*':>8s} {'n_abc':>8s} {'n_perm':>8s}"
          f" {'MSE_abc':>12s} {'MSE_perm':>12s} {'ratio':>8s}")
    print("-" * 70)
    for eps, n_a, n_p, mse_a, mse_p in results:
        ratio = ""
        if np.isfinite(mse_a) and np.isfinite(mse_p) and mse_a > 0:
            ratio = f"{mse_p / mse_a:8.3f}"
        star = " <--" if abs(eps / eps_star - 1.0) < 0.05 else ""
        print(f"  {eps:10.4f} {eps/eps_star:8.2f} {int(n_a):8d} {int(n_p):8d}"
              f" {mse_a:12.6f} {mse_p:12.6f} {ratio:>8s}{star}")

    fig = plot_mse_vs_epsilon(results, eps_star, seed=seed)
    out_dir = PROJECT_ROOT / "experiments" / "figures" / "fig3"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / f"fig3_mse_vs_epsilon_seed_{seed}.pdf")
    print(f"\nFigure saved to {out_dir / f'fig3_mse_vs_epsilon_seed_{seed}.pdf'}")


if __name__ == "__main__":
    main()
