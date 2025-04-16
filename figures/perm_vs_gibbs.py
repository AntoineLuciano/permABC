# Initialization
import os 

os.chdir("/Users/antoineluciano/Documents/Recherche/permABC_new/permABC")
import sys
sys.path.append(os.getcwd())

from vanilla import abc_vanilla, perm_abc_vanilla
from smc import abc_smc, perm_abc_smc
from over_sampling import perm_abc_smc_os
from under_matching import perm_abc_smc_um  
from kernels import KernelTruncatedRW
from distances import optimal_index_distance
from models.Gaussian_with_correlated_params import GaussianWithCorrelatedParams
from utils import Theta
from jax import random
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
N = 1000
N_smc = N
N_gibbs = N
N_pymc = N
key = random.PRNGKey(0)
key, key_theta, key_yobs = random.split(key, 3)
K = 10
n_obs = 20
sigma_mu, sigma_alpha = 10., 10.
model = GaussianWithCorrelatedParams(K = K, n_obs = n_obs, sigma_mu = sigma_mu, sigma_alpha = sigma_alpha)
true_theta = model.prior_generator(key_theta, 1)
true_theta.loc[0,0,0] = 0.
true_theta.glob[0,0] = 0.
y_obs = model.data_generator(key_yobs, true_theta)


print("permABC-SMC...")
key, subkey = random.split(key)
kernel = KernelTruncatedRW
out_perm_smc = perm_abc_smc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, kernel = kernel, verbose = 0, update_weights_distance=False, Final_iteration=0)

mus_perm_smc = out_perm_smc["Thetas"][-1].loc.squeeze()
betas_perm_smc = out_perm_smc["Thetas"][-1].glob.squeeze()
n_sim_perm_smc = np.sum(out_perm_smc["N_sim"])

import numpy as np
from tqdm.notebook import tqdm
from jax import vmap, jit, random
import jax.numpy as jnp
from scipy import stats
from scipy.optimize import linear_sum_assignment

@jit
def distance_one_silo(x_k,y_k):
    return jnp.sum((x_k-y_k)**2)

@jit
def distance_all_silo(x,y):
    return vmap(distance_one_silo, in_axes=(0, 0))(x,y)
@jit
def distance_xs(xs, y):
    return vmap(distance_all_silo, in_axes=(0, None))(xs, y)

@jit
def distance_sum_silo(x,y):
    return jnp.mean((distance_all_silo(x,y)))

@jit
def distance_sum(xs, y):
    return (vmap(distance_sum_silo, in_axes=(0, None))(xs, y))
# @jit 
# def dist_full_y(xs, y):
#     return jnp.sum(vmap(dist_full, in_axes=(0, None))(xs, y), axis = 1)

def ABCmus(key, M, y_obs, alpha):
    key, key_mus, key_data = random.split(key, 3)
    mus = random.normal(key_mus, shape=(M,K))*model.sigma_mu
    thetas = Theta(loc = mus[:,:,None], glob = np.repeat([alpha], M)[:,None])
    xs = model.data_generator(key_data, thetas)
    dists = distance_xs(xs, y_obs)
    index_min = jnp.argmin(dists, axis=0)
    Eps_betas = jnp.array([dists[index_min[i],i] for i in range(K)])
    mus_min = np.array([mus[index_min[i],i] for i in range(K)])
    return mus_min, Eps_betas
    

def ABCalpha(key, M, y_obs, mus):
    key, key_alpha, key_data = random.split(key, 3)
    alphas = random.normal(key_alpha, shape=(M,1))*model.sigma_alpha
    thetas = Theta(loc = np.repeat([mus], M, axis=0)[:,:,None], glob = alphas)
    xs = model.data_generator(key_data, thetas)
    dists = distance_sum(xs, y_obs)
    index_min = jnp.argmin(dists)
    Eps_alpha = dists[index_min]
    alpha_min = alphas[index_min]
    return alpha_min[0], Eps_alpha



def gibbs_robin(key, T, M_mu, M_alpha, y_obs):
    mus = np.zeros((T+1,K))
    alphas = np.zeros(T+1)
    
    Eps_mu = np.zeros((T, K))
    Eps_alpha = np.zeros(T)
    
    
    key, key_alpha, key_mu = random.split(key, 3)
    
    # alphas[0] = -10
    mus[0] = random.normal(key_mu, shape=(K,))*model.sigma_mu
    # mus[0] = -10
    alphas[0] = random.normal(key_alpha)*model.sigma_alpha
    for t in tqdm(range(T)):
        key, key_mus = random.split(key)
        mus[t+1], Eps_mu[t] = ABCmus(key_mus, M_mu, y_obs, alphas[t])
        key, key_alpha = random.split(key)
        alphas[t+1], Eps_alpha[t] = ABCalpha(key_alpha, M_alpha, y_obs, mus[t+1])
        
    return mus, alphas, Eps_mu, Eps_alpha 

M = n_sim_perm_smc//(2*K*N)
print("M = ", M)
key, subkey = random.split(key)
mus_gibbs, alphas_gibbs, Eps_mus_gibbs, Eps_alphas_gibbs = gibbs_robin(subkey, N, M, M, y_obs[0])

import pymc as pm

# Define the model
with pm.Model() as mod:
    # Priors for unknown model parameters

    sigma_x = 1.0

    mu = pm.Normal('mu', mu=0, sigma=sigma_mu, shape=(K, 1))
    alpha = pm.Normal('alpha', mu=0, sigma=sigma_alpha, shape=(1, 1))

    # Likelihood (sampling distribution) of observations
    x = pm.Normal('x', mu=mu + alpha, sigma=sigma_x, observed=y_obs[0])

    # Inference
    trace = pm.sample(N_pymc, return_inferencedata=True)

true_post_mu = np.array(trace.posterior.mu[:,:,:,0]).reshape(4*N_pymc, K)
true_post_alpha = np.array(trace.posterior.alpha).reshape(-1)


k = 0
title_size = 15
fontsize = 12
alpha_fig = 0.25

f, ax = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[1].scatter(x = mus_perm_smc[:,k], y = betas_perm_smc, label = "Perm ABC SMC", color = "blue", alpha = alpha_fig)
sns.kdeplot(x=true_post_mu[:, k], y=true_post_alpha, label="True posterior", color = "black", alpha =1, ax = ax[1])
ax[1].set_title("permABC-SMC", fontsize = title_size)
# ax[1].scatter(x = true_theta.loc[0,k,0], y = true_theta.glob[0,0], label = "True theta", color = "red", alpha = 1)

ax[1].set_xlabel("$\mu_1$", fontsize = fontsize)
# ax[1].set_ylabel("$\\beta$", fontsize = fontsize)
ax[0].set_title("ABC-Gibbs", fontsize = title_size)
ax[0].scatter(x = mus_gibbs[:,k], y = alphas_gibbs, label = "Gibbs", color = "red", alpha = alpha_fig)
sns.kdeplot(x=true_post_mu[:, k], y=true_post_alpha, label="True posterior",  color = "black", alpha =1, ax = ax[0])
# ax[0].scatter(x = true_theta.loc[0,k,0], y = true_theta.glob[0,0], label = "True theta", color = "red", alpha = 1)
ax[0].set_xlabel("$\mu_1$", fontsize = fontsize)
ax[0].set_ylabel("$\\beta$", fontsize = fontsize)
ax[0].set_xlim(-10, 10)
ax[0].set_ylim(-10, 10)
ax[1].set_xlim(-10, 10)
ax[1].set_ylim(-10, 10)
plt.savefig("figures/perm_vs_gibbs.svg", dpi=300)
plt.savefig("figures/perm_vs_gibbs.png", dpi=300)
plt.savefig("figures/perm_vs_gibbs.pdf", dpi=300)

dico = {"mus_perm_smc": mus_perm_smc, "betas_perm_smc": betas_perm_smc, "mus_gibbs": mus_gibbs, "alphas_gibbs": alphas_gibbs, "true_post_mu": true_post_mu, "true_post_alpha": true_post_alpha, "true_theta": true_theta, "y_obs": y_obs, "N_smc": N_smc, "N_gibbs": N_gibbs, "N_pymc": N_pymc, "model": model, "kernel": kernel, "N_sim_perm_smc": n_sim_perm_smc, "M": M}

import pickle
with open("figures/perm_vs_gibbs.pkl", "wb") as f:
    pickle.dump(dico, f)


# plt.scatter(x = true_theta.loc[0,k,0], y = true_theta.glob[0,0], label = "True theta", color = "red", alpha = 1)
# sns.kdeplot(x=true_post_mu[:, k], y=true_post_alpha, label="True posterior", level = 3, color = "black", alpha =1 )
