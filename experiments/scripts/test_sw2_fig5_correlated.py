#!/usr/bin/env python3
"""
SW2 comparison: permABC-SMC vs ABC-Gibbs on the CORRELATED Gaussian model (Fig 5).

Model: mu_k ~ N(0, sigma_mu²), alpha ~ N(0, sigma_alpha²), X_{k,i} ~ N(mu_k + alpha, 1)
Posterior is multivariate Gaussian (conjugate), so SW2 floor is well-defined.

Usage:
    PYENV_VERSION=permabc pyenv exec python experiments/scripts/test_sw2_fig5_correlated.py
"""
import sys, time as _time
from pathlib import Path
import numpy as np
from jax import random, vmap, jit
import jax.numpy as jnp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "scripts"))

from permabc.models.Gaussian_with_correlated_params import GaussianWithCorrelatedParams
from permabc.algorithms.smc import perm_abc_smc
from permabc.sampling.kernels import KernelTruncatedRW
from permabc.utils.functions import Theta


# ── True posterior (analytic) ─────────────────────────────────────────────────

def compute_true_posterior(model, y_obs):
    """Full (K+1)-dim Gaussian posterior: (mu_1,...,mu_K, alpha)."""
    K = model.K
    n = model.n_obs
    y = np.asarray(y_obs[0])  # (K, n_obs)
    S = np.sum(y, axis=1)     # (K,) = sum of obs per component

    # Prior precision
    diag_prior = np.concatenate([np.full(K, 1.0 / model.sigma_mu**2),
                                  [1.0 / model.sigma_alpha**2]])
    Lambda_prior = np.diag(diag_prior)

    # Likelihood precision: each obs X_{k,i} ~ N(mu_k + alpha, 1)
    Lambda_lik = np.zeros((K + 1, K + 1))
    for j in range(K):
        Lambda_lik[j, j] = n
        Lambda_lik[j, K] = n
        Lambda_lik[K, j] = n
    Lambda_lik[K, K] = n * K

    # Likelihood info vector
    eta_lik = np.zeros(K + 1)
    eta_lik[:K] = S
    eta_lik[K] = np.sum(S)

    Lambda_post = Lambda_prior + Lambda_lik
    Sigma_post = np.linalg.inv(Lambda_post)
    mu_post = Sigma_post @ eta_lik
    return mu_post, Sigma_post


def sample_true_posterior_correlated(model, y_obs, n_samples, rng):
    """Sample from the exact posterior. Returns (mus, alphas)."""
    mu_post, Sigma_post = compute_true_posterior(model, y_obs)
    K = model.K
    samples = rng.multivariate_normal(mu_post, Sigma_post, size=n_samples)
    mus = samples[:, :K]       # (n_samples, K)
    alphas = samples[:, K]     # (n_samples,)
    return mus, alphas


# ── SW2 computation ──────────────────────────────────────────────────────────

def sw2_joint(abc_mus, abc_alphas, ref_mus, ref_alphas, weights=None,
              n_projections=200, seed=0):
    """Sliced W2 in (mu_1,...,mu_K, alpha) space."""
    abc_joint = np.column_stack([abc_mus, abc_alphas])
    ref_joint = np.column_stack([ref_mus, ref_alphas])
    dim = abc_joint.shape[1]
    N_abc = len(abc_joint)
    N_ref = len(ref_joint)

    rng = np.random.default_rng(seed + 1)
    directions = rng.standard_normal((n_projections, dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    is_uniform = weights is None or np.allclose(weights, 1.0 / N_abc)

    sw2_sq = 0.0
    for omega in directions:
        proj_abc = abc_joint @ omega
        proj_ref = ref_joint @ omega

        if is_uniform and N_abc == N_ref:
            # Exact sort-diff
            a_sorted = np.sort(proj_abc)
            b_sorted = np.sort(proj_ref)
            sw2_sq += np.mean((a_sorted - b_sorted) ** 2)
        else:
            # Quantile interpolation for weighted
            proj_ref_sorted = np.sort(proj_ref)
            n_q = 2 * max(N_abc, N_ref)
            q_levels = np.linspace(0, 1, n_q + 2)[1:-1]
            ref_quantiles = np.interp(q_levels, np.linspace(0, 1, N_ref), proj_ref_sorted)

            if weights is not None:
                w = np.asarray(weights, dtype=float)
                w = w / np.sum(w)
            else:
                w = np.ones(N_abc) / N_abc
            order = np.argsort(proj_abc)
            abc_sorted = proj_abc[order]
            cdf = np.cumsum(w[order])
            abc_quantiles = np.interp(q_levels, cdf, abc_sorted)
            sw2_sq += np.mean((abc_quantiles - ref_quantiles) ** 2)

    sw2_sq /= n_projections
    return float(np.sqrt(sw2_sq))


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def ess_mcmc(mus_chain, alpha_chain):
    ess_vals = [ess_autocorr_1d(alpha_chain)]
    for k in range(mus_chain.shape[1]):
        ess_vals.append(ess_autocorr_1d(mus_chain[:, k]))
    return min(ess_vals), ess_vals


def unique_count(thetas):
    loc = np.asarray(thetas.loc)
    if loc.ndim == 3: loc = loc.reshape(loc.shape[0], -1)
    _, idx = np.unique(loc, axis=0, return_index=True)
    return len(idx)


# ── Gibbs sampler (raw data, same as fig5) ────────────────────────────────────

def run_gibbs_correlated(model, y_obs, K, T, M_mu, M_alpha, seed=42):
    """ABC-Gibbs for the correlated Gaussian model, using raw data distances."""

    @jit
    def distance_xs(xs, y):
        """(M, K, n_obs) vs (K, n_obs) -> (M, K)"""
        return vmap(lambda x: jnp.sum((x - y) ** 2, axis=1))(xs)

    @jit
    def distance_sum(xs, y):
        """(M, K, n_obs) vs (K, n_obs) -> (M,)"""
        return vmap(lambda x: jnp.mean(jnp.sum((x - y) ** 2, axis=1)))(xs)

    y = np.array(y_obs[0])  # (K, n_obs)

    key = random.PRNGKey(seed + 777)
    mus = np.zeros((T + 1, K))
    alphas = np.zeros(T + 1)

    key, k1, k2 = random.split(key, 3)
    mus[0] = np.array(random.normal(k1, shape=(K,)) * model.sigma_mu)
    alphas[0] = float(random.normal(k2) * model.sigma_alpha)

    t0 = _time.perf_counter()
    for t in range(T):
        alpha_t = alphas[t]

        # Block 1: update each mu_k independently
        key, k_mu, k_data = random.split(key, 3)
        mu_cands = np.array(random.normal(k_mu, shape=(M_mu, K)) * model.sigma_mu)
        # For each candidate mu_k, simulate n_obs observations N(mu_k + alpha, 1)
        # and pick the one closest to y_k
        thetas_mu = Theta(loc=jnp.array(mu_cands[:, :, None]),
                          glob=jnp.full((M_mu, 1), alpha_t))
        xs_mu = model.data_generator(k_data, thetas_mu)  # (M, K, n_obs)
        dists_mu = np.array(distance_xs(jnp.array(xs_mu), jnp.array(y)))  # (M, K)
        for k in range(K):
            mus[t + 1, k] = mu_cands[np.argmin(dists_mu[:, k]), k]

        # Block 2: update alpha
        key, k_alpha, k_data2 = random.split(key, 3)
        alpha_cands = np.array(random.normal(k_alpha, shape=(M_alpha,)) * model.sigma_alpha)
        thetas_alpha = Theta(loc=jnp.tile(mus[t + 1][None, :, None], (M_alpha, 1, 1)),
                             glob=alpha_cands[:, None])
        xs_alpha = model.data_generator(k_data2, thetas_alpha)  # (M, K, n_obs)
        dists_alpha = np.array(distance_sum(jnp.array(xs_alpha), jnp.array(y)))  # (M,)
        alphas[t + 1] = alpha_cands[np.argmin(dists_alpha)]

    elapsed = _time.perf_counter() - t0
    return mus, alphas, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    K = 20
    N_particles = 5000
    seed = 42
    burnin_frac = 0.01

    # Same setup as fig5
    key = random.PRNGKey(0)  # fig5 uses seed=0 when seed==42
    key, key_theta, key_yobs = random.split(key, 3)

    model = GaussianWithCorrelatedParams(K=K, n_obs=20, sigma_mu=10., sigma_alpha=10.)
    true_theta = model.prior_generator(key_theta, 1)
    true_theta = Theta(
        loc=true_theta.loc.at[0, 0, 0].set(0.),
        glob=true_theta.glob.at[0, 0].set(0.)
    )
    y_obs = model.data_generator(key_yobs, true_theta)
    y_obs = np.array(y_obs)
    y_obs[0, 0, :] -= np.mean(y_obs[0, 0, :])

    print(f"K={K}, N_particles={N_particles}, n_obs={model.n_obs}")
    print(f"sigma_mu={model.sigma_mu}, sigma_alpha={model.sigma_alpha}")
    print(f"True params: mu_1=0, alpha=0\n")

    # ── SW2 floor ──
    print("Computing SW2 floor...")
    rng1 = np.random.default_rng(2000)
    rng2 = np.random.default_rng(3000)
    mus_true1, alphas_true1 = sample_true_posterior_correlated(model, y_obs, N_particles, rng1)
    mus_true2, alphas_true2 = sample_true_posterior_correlated(model, y_obs, N_particles, rng2)
    sw2_floor = sw2_joint(mus_true1, alphas_true1, mus_true2, alphas_true2,
                          n_projections=200, seed=0)
    print(f"SW2 floor = {sw2_floor:.4f}\n")

    # ── permABC-SMC ──
    print("=" * 60)
    print("permABC-SMC (raw data)")
    print("=" * 60)
    key_run = random.PRNGKey(seed)
    key_run, k_smc = random.split(key_run)
    t0 = _time.perf_counter()
    out = perm_abc_smc(
        key=k_smc, model=model, n_particles=N_particles,
        epsilon_target=0, y_obs=y_obs, kernel=KernelTruncatedRW,
        N_sim_max=np.inf, stopping_accept_rate=0.015,
        Final_iteration=0, update_weights_distance=False,
        verbose=1, try_lsa=True,
    )
    time_smc = _time.perf_counter() - t0
    budget = int(np.sum(out["N_sim"]))
    thetas_smc = out["Thetas"][-1]
    weights_smc = out["Weights"][-1]
    mus_smc = np.asarray(thetas_smc.loc).reshape(-1, K)
    alphas_smc = np.asarray(thetas_smc.glob).squeeze()
    n_unique_smc = unique_count(thetas_smc)

    sw2_smc = sw2_joint(mus_smc, alphas_smc, mus_true1, alphas_true1,
                        weights=weights_smc, n_projections=200, seed=0)
    print(f"\npermABC-SMC: budget={budget:,}, time={time_smc:.1f}s")
    print(f"  unique={n_unique_smc}/{N_particles}")
    print(f"  SW2={sw2_smc:.4f}, ratio={sw2_smc/sw2_floor:.2f}x floor\n")

    # ── ABC-Gibbs (T=N_particles, M from budget) ──
    # Budget = (M_mu * K + M_alpha) * K * T component-sims
    # In fig5 Gibbs: M_mu candidates each simulate K components, K times -> M_mu * K * K
    #                M_alpha candidates each simulate K components -> M_alpha * K
    # Per iteration: M_mu * K + M_alpha * K component-sims (each sim = K obs)
    # Actually in fig5: ABCmus simulates M_mu particles * K components = M_mu * K comp-sims
    #                   ABCalpha simulates M_alpha particles * K components = M_alpha * K comp-sims
    # Per iteration: (M_mu + M_alpha) * K comp-sims
    # With M_mu = M_alpha = M: 2 * M * K comp-sims per iteration
    # Total: 2 * M * K * T

    T_gibbs = N_particles  # = 1000 like fig5
    M_gibbs = budget // (2 * K * T_gibbs)
    actual_nsim_gibbs = 2 * K * T_gibbs * M_gibbs
    burnin = int(T_gibbs * burnin_frac)

    print("=" * 60)
    print(f"ABC-Gibbs (T={T_gibbs}, M={M_gibbs})")
    print("=" * 60)
    print(f"  N_sim = 2×{K}×{T_gibbs}×{M_gibbs} = {actual_nsim_gibbs:,} (budget={budget:,})")

    mus_gibbs, alphas_gibbs, time_gibbs = run_gibbs_correlated(
        model, y_obs, K, T_gibbs, M_gibbs, M_gibbs, seed)
    print(f"  Done in {time_gibbs:.1f}s")

    mus_post = mus_gibbs[burnin:]
    alphas_post = alphas_gibbs[burnin:]
    n_post = len(alphas_post)

    min_ess, ess_all = ess_mcmc(mus_post, alphas_post)
    print(f"  n_post={n_post}, min_ESS={min_ess:.0f}")

    sw2_gibbs = sw2_joint(mus_post, alphas_post, mus_true1, alphas_true1,
                          n_projections=200, seed=0)
    print(f"  SW2={sw2_gibbs:.4f}, ratio={sw2_gibbs/sw2_floor:.2f}x floor")

    # ── Also try Gibbs matched unique ──
    T_matched = int(np.ceil(n_unique_smc / (1 - burnin_frac)))
    M_matched = budget // (2 * K * T_matched)
    actual_matched = 2 * K * T_matched * M_matched
    burnin_m = int(T_matched * burnin_frac)

    print(f"\n{'='*60}")
    print(f"ABC-Gibbs (T={T_matched}, M={M_matched}) — matched unique={n_unique_smc}")
    print(f"{'='*60}")
    print(f"  N_sim = 2×{K}×{T_matched}×{M_matched} = {actual_matched:,}")

    mus_m, alphas_m, time_m = run_gibbs_correlated(
        model, y_obs, K, T_matched, M_matched, M_matched, seed + 1)
    print(f"  Done in {time_m:.1f}s")

    mus_post_m = mus_m[burnin_m:]
    alphas_post_m = alphas_m[burnin_m:]
    n_post_m = len(alphas_post_m)
    min_ess_m, _ = ess_mcmc(mus_post_m, alphas_post_m)

    sw2_m = sw2_joint(mus_post_m, alphas_post_m, mus_true1, alphas_true1,
                      n_projections=200, seed=0)
    print(f"  n_post={n_post_m}, min_ESS={min_ess_m:.0f}")
    print(f"  SW2={sw2_m:.4f}, ratio={sw2_m/sw2_floor:.2f}x floor")

    # ── Summary ──
    print(f"\n{'='*80}")
    print(f"SUMMARY  (K={K}, budget={budget:,}, SW2_floor={sw2_floor:.4f})")
    print(f"{'='*80}")
    print(f"{'Method':<40s} {'N_sim':>12s} {'n_eff':>7s} {'ESS':>7s} {'SW2':>8s} {'ratio':>7s}")
    print("-" * 80)
    print(f"{'permABC-SMC (raw)':<40s} {budget:12,} {n_unique_smc:7d} {n_unique_smc:7d} {sw2_smc:8.4f} {sw2_smc/sw2_floor:7.2f}")
    print(f"{'ABC-Gibbs (T='+str(T_gibbs)+',M='+str(M_gibbs)+')':<40s} {actual_nsim_gibbs:12,} {n_post:7d} {int(min_ess):7d} {sw2_gibbs:8.4f} {sw2_gibbs/sw2_floor:7.2f}")
    print(f"{'ABC-Gibbs (T='+str(T_matched)+',M='+str(M_matched)+')':<40s} {actual_matched:12,} {n_post_m:7d} {int(min_ess_m):7d} {sw2_m:8.4f} {sw2_m/sw2_floor:7.2f}")
    print(f"{'SW2 floor':<40s} {'':>12s} {'':>7s} {'':>7s} {sw2_floor:8.4f} {'1.00':>7s}")

    # ── True posterior stats ──
    mu_post_mean, Sigma_post = compute_true_posterior(model, y_obs)
    corr = Sigma_post / np.sqrt(np.outer(np.diag(Sigma_post), np.diag(Sigma_post)))
    print(f"\nPosterior correlation mu_1-alpha: {corr[0, K]:.3f}")
    print(f"Posterior std(mu_k): {np.sqrt(Sigma_post[0,0]):.3f}")
    print(f"Posterior std(alpha): {np.sqrt(Sigma_post[K,K]):.3f}")


if __name__ == "__main__":
    main()
