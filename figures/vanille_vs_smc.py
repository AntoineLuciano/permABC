import os 


os.chdir("/Users/antoineluciano/Documents/Recherche/permABC_new/permABC")
print("\n\nPATH:",os.getcwd())


import sys
sys.path.append("/Users/antoineluciano/Documents/Recherche/permABC_new/permABC")
from vanilla import abc_vanilla, perm_abc_vanilla
from smc import abc_smc, perm_abc_smc
from over_sampling import perm_abc_smc_os
from under_matching import perm_abc_smc_um  
from kernels import KernelTruncatedRW
from distances import optimal_index_distance
from models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from jax import random
import numpy as np
import jax.numpy as jnp
from scipy.stats import invgamma
import matplotlib.pyplot as plt
import seaborn as sns

key = random.PRNGKey(0)
key, subkey = random.split(key)
K = 15
n = 10
sigma0 = 10
alpha, beta = 5,5
model = GaussianWithNoSummaryStats(K = K, n_obs= n, sigma_0 = sigma0, alpha = alpha, beta = beta)
true_theta = model.prior_generator(subkey, 1)
# true_theta.loc = np.linspace(-2*sigma0, 2*sigma0, K)[None,:, None]
true_theta.glob = np.array([1.])[None,:]
key, subkey = random.split(key)
y_obs = model.data_generator(subkey, true_theta)
print(y_obs.shape)
# Vanilla 
## permABC
key, subkey = random.split(key)
N_points = 1000000
N_smc = 1000

model.reset_weights_distance()
key, key_theta, key_perm = random.split(key, 3)
thetas = model.prior_generator(key_theta, N_points)
zs = model.data_generator(key, thetas)
dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs)
thetas_perm = thetas.apply_permutation(zs_index)
mus_perm_van = thetas_perm.loc.squeeze()
betas_perm_van = thetas_perm.glob.squeeze()


## ABC
key, subkey = random.split(key)
thetas_van = model.prior_generator(subkey, N_points)
key, subkey = random.split(key)
zs_van = model.data_generator(subkey, thetas_van)
dists_van = model.distance(zs_van, y_obs)
n_sim_van = N_points*K


mus_van = thetas_van.loc.squeeze()
betas_van = thetas_van.glob.squeeze()

kernel = KernelTruncatedRW 
key, subkey = random.split(key)


out_smc = abc_smc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, kernel = kernel, verbose =1, Final_iteration = 0, update_weights_distance= False, N_sim_max= N_points*K, stopping_accept_rate= 0.)

key, subkey = random.split(key)
kernel = KernelTruncatedRW
model.reset_weights_distance()

out_perm_smc = perm_abc_smc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, kernel = kernel, verbose = 1, Final_iteration= 0, update_weights_distance= False, N_sim_max= N_points*K, stopping_accept_rate= 0.)

# mus_perm_smc = out_perm_smc["Thetas"][-1].loc.squeeze()
# betas_perm_smc = out_perm_smc["Thetas"][-1].glob.squeeze()
# n_sim_perm_smc = np.sum(out_perm_smc["N_sim"])

from pmc import abc_pmc
key, subkey = random.split(key)
model.reset_weights_distance()
out_pmc = abc_pmc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, verbose =1, update_weights_distance= False, N_sim_max= N_points*K, stopping_accept_rate= 0., alpha = .95)
# Comparison
import seaborn as sns
import matplotlib.pyplot as plt

# Plot
n_sim_smc = np.array(out_smc["N_sim"])[1:]
n_sim_perm_smc = np.array(out_perm_smc["N_sim"])[1:]
epsilon_smc = np.array(out_smc["Eps_values"])[1:]
epsilon_perm_smc = np.array(out_perm_smc["Eps_values"])[1:]
unique_smc = np.array(out_smc["unique_part"])[1:]
unique_perm_smc = np.array(out_perm_smc["unique_part"])[1:]
epsilon_pmc = np.array(out_pmc["Eps_values"])[1:]
n_sim_pmc = np.array(out_pmc["N_sim"])
ess_pmc = np.array(out_pmc["Ess"])
unique_pmc = np.array(out_pmc["unique_part"])


N_sample = 1000

alphas = np.logspace(0, -3, 10)
epsilons_van_perm = np.quantile(dists_perm, alphas)
epsilons_van = np.quantile(dists_van[:,0], alphas)

N_sims = N_sample/alphas

N_sim_pmc = np.cumsum(n_sim_pmc)/K/unique_pmc/N_smc*N_sample
N_sim_smc = np.cumsum(n_sim_smc)/K/unique_smc/N_smc*N_sample
N_sim_perm_smc = np.cumsum(n_sim_perm_smc)/K/unique_perm_smc/N_smc*N_sample


closest_index_smc = np.argmin(np.abs(N_sim_smc - N_sims[:, None]), axis=1)
closest_index_perm_smc = np.argmin(np.abs(N_sim_perm_smc - N_sims[:, None]), axis=1)
closest_index_pmc = np.argmin(np.abs(N_sim_pmc - N_sims[:, None]), axis=1)
f, ax = plt.subplots(1, 1, figsize=(10, 6))
plt.plot(N_sims, np.quantile(dists_perm, alphas), label="permABC Vanilla", linestyle="-", marker="x")
plt.plot(N_sims, np.quantile(dists_van[:,0], alphas), label="ABC Vanilla", linestyle="--", marker="x")
plt.plot(N_sim_perm_smc[closest_index_perm_smc], epsilon_perm_smc[closest_index_perm_smc], label="permABC-SMC", linestyle="-", marker="o")
plt.plot(N_sim_smc[closest_index_smc], epsilon_smc[closest_index_smc], label="ABC-SMC", linestyle="--", marker="o")
plt.plot(N_sim_pmc[closest_index_pmc], epsilon_pmc[closest_index_pmc], label="ABC-PMC", linestyle="--", marker="o")



plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of simulations per {} unique particles".format(N_sample)) 
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.savefig("figures/vanilla_vs_smc.png")


import pickle

data_to_save = {
    "N_sample": N_sample,
    "alphas": alphas,
    "dists_perm": dists_perm,
    "dists_van": dists_van,
    "n_sim_pmc": n_sim_pmc,
    "n_sim_smc": n_sim_smc,
    "n_sim_perm_smc": n_sim_perm_smc,
    "epsilon_pmc": epsilon_pmc,
    "epsilon_smc": epsilon_smc,
    "epsilon_perm_smc": epsilon_perm_smc,
    "unique_pmc": unique_pmc,
    "unique_smc": unique_smc,
    "unique_perm_smc": unique_perm_smc,
}

with open("figures/vanilla_vs_smc.pkl", "wb") as f:
    pickle.dump(data_to_save, f)