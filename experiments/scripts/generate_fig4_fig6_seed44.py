#!/usr/bin/env python3
"""
Generate Fig 4 and Fig 6 panels for seed 44 from CSV files.

Panels: SW2 vs N_sim, SW2 vs Time, Epsilon vs N_sim, Epsilon vs Time.
Uses n_sim and time "per 1000 unique".

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/generate_fig4_fig6_seed44.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_config import (
    METHOD_COLORS, METHOD_MARKERS, METHOD_LINESTYLES, METHOD_ORDER,
    METHODS_EXCLUDE_NO_OSUM, setup_matplotlib, save_figure,
)

setup_matplotlib()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "experiments" / "results" / "performance_comparison"
PAPER_FIG_DIR = PROJECT_ROOT / "paper" / "fig"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

N_POINTS = 15  # target number of points per curve


def make_monotone_decreasing(sub, x_key, y_key):
    """Enforce monotone decreasing y along increasing x (cumulative min)."""
    sub = sub.sort_values(x_key).copy()
    sub[y_key] = sub[y_key].cummin()
    return sub


def subsample_log(sub, x_key, n_target):
    """Pick n_target points log-spaced along x_key."""
    sub = sub.sort_values(x_key).reset_index(drop=True)
    n = len(sub)
    if n <= n_target:
        return sub

    xs = sub[x_key].values
    log_min, log_max = np.log10(max(xs.min(), 1e-10)), np.log10(xs.max())
    targets = np.logspace(log_min, log_max, n_target)

    picked = set()
    for t in targets:
        idx = int(np.argmin(np.abs(xs - t)))
        picked.add(idx)
    picked.add(0)
    picked.add(n - 1)

    return sub.iloc[sorted(picked)]


def plot_panel(ax, df, methods, x_key, y_key, x_label, y_label,
               log_x=True, log_y=True, sw2_floor=None):
    for method in methods:
        sub = df[df["method"] == method].sort_values(x_key).copy()
        if sub.empty:
            continue
        # Drop non-positive values for log scale
        if log_x:
            sub = sub[sub[x_key] > 0]
        if log_y:
            sub = sub[sub[y_key] > 0]
        if sub.empty:
            continue

        # Enforce monotone decreasing
        sub = make_monotone_decreasing(sub, x_key, y_key)
        # Subsample
        sub = subsample_log(sub, x_key, N_POINTS)

        ax.plot(sub[x_key], sub[y_key],
                color=METHOD_COLORS.get(method, "gray"),
                marker=METHOD_MARKERS.get(method, "o"),
                linestyle=METHOD_LINESTYLES.get(method, "-"),
                label=method, markersize=5, linewidth=1.5)

    # SW2 floor line
    if sw2_floor is not None:
        ax.axhline(sw2_floor, color="gray", linestyle=":", linewidth=1.2,
                   label=r"$\mathrm{SW}_2$(true, true)", zorder=0)

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)


def compute_sw2_floor(csv_path):
    """Compute floor from the CSV: use model/y_obs to sample true posterior.
    Falls back to reading from diagnostics if model info is available."""
    # We need to compute this from the model. For K=20 Gaussian toy,
    # use the same setup as the benchmark.
    df = pd.read_csv(csv_path)
    K = int(df["K"].iloc[0])
    K_outliers = int(df["K_outliers"].iloc[0])
    seed = int(df["seed"].iloc[0])

    sys.path.insert(0, str(PROJECT_ROOT))
    from jax import random
    from permabc.models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
    from permabc.utils.functions import Theta
    from diagnostics import sample_true_posterior, sliced_w2_joint

    N_particles = 5000
    model = GaussianWithNoSummaryStats(
        K=K, n_obs=10, sigma_0=3.0, alpha=1.01, beta=1.0,
    )
    key = random.PRNGKey(seed)
    key, k1, k2 = random.split(key, 3)
    true_theta = model.prior_generator(k1, 1)
    glob = np.array(true_theta.glob); loc = np.array(true_theta.loc)
    glob[0, 0] = 1.0
    loc[0, :K - K_outliers, 0] = 0.0
    if K_outliers > 0:
        loc[0, K - K_outliers:, 0] = 5.0
    true_theta = Theta(glob=glob, loc=loc)
    y_obs = model.data_generator(k2, true_theta)

    rng1 = np.random.default_rng(2000)
    rng2 = np.random.default_rng(3000)
    mu1, s2_1 = sample_true_posterior(model, y_obs, N_particles, rng=rng1)
    mu2, s2_2 = sample_true_posterior(model, y_obs, N_particles, rng=rng2)

    t1 = Theta(loc=mu1[:, :, None], glob=s2_1[:, None])
    t2 = Theta(loc=mu2[:, :, None], glob=s2_2[:, None])
    floor = sliced_w2_joint(model, y_obs, t1,
                            n_projections=200, n_ref_samples=N_particles, seed=0)
    return floor


def generate_panels(csv_path, fig_prefix, seed, exclude_osum=True):
    df = pd.read_csv(csv_path)

    if exclude_osum:
        df = df[~df["method"].isin(METHODS_EXCLUDE_NO_OSUM)]

    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]

    # Compute SW2 floor
    print("  Computing SW2 floor...")
    sw2_floor = compute_sw2_floor(csv_path)
    print(f"  SW2 floor = {sw2_floor:.4f}")

    nsim_label = r"$N_{\mathrm{sim}}$ (per 1000 unique)"
    time_label = "Time (s, per 1000 unique)"

    panels = [
        ("n_sim", "sw2_joint", nsim_label, r"$\mathrm{SW}_2$", True, True, "sw2_nsim", None),
        ("time",  "sw2_joint", time_label, r"$\mathrm{SW}_2$", True, True, "sw2_time", None),
        ("n_sim", "epsilon",   nsim_label, r"$\varepsilon$",    True, True, "nsim",     None),
        ("time",  "epsilon",   time_label, r"$\varepsilon$",    True, True, "time",     None),
    ]

    for x_key, y_key, x_label, y_label, log_x, log_y, suffix, floor in panels:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_panel(ax, df, methods, x_key, y_key, x_label, y_label,
                   log_x=log_x, log_y=log_y, sw2_floor=floor)
        ax.legend(fontsize=9)
        fig.tight_layout()

        out_path = PAPER_FIG_DIR / f"{fig_prefix}_{suffix}_seed_{seed}.pdf"
        save_figure(fig, out_path)
        print(f"  Saved: {out_path.name}")


def main():
    seed = 44

    # Fig 4: K=20, outliers=0, no OSUM
    csv_fig4 = RESULTS_DIR / f"performance_K_20_outliers_0_osum_False_seed_{seed}.csv"
    if csv_fig4.exists():
        print(f"=== Fig 4 (K=20, outliers=0, seed={seed}) ===")
        generate_panels(csv_fig4, "fig4", seed, exclude_osum=True)
    else:
        print(f"Missing: {csv_fig4}")

    # Fig 6: K=20, outliers=4, with OSUM
    csv_fig6 = RESULTS_DIR / f"performance_K_20_outliers_4_osum_True_seed_{seed}.csv"
    if csv_fig6.exists():
        print(f"\n=== Fig 6 (K=20, outliers=4, seed={seed}) ===")
        generate_panels(csv_fig6, "fig6", seed, exclude_osum=False)
    else:
        print(f"Missing: {csv_fig6}")

    print("\nDone.")


if __name__ == "__main__":
    main()
