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
K = 10
n = 10
sigma0 = 10
alpha, beta = 5,5
model = GaussianWithNoSummaryStats(K = K, n_obs= n, sigma_0 = sigma0, alpha = alpha, beta = beta)
true_theta = model.prior_generator(subkey, 1)
# true_theta.loc = np.linspace(-2*sigma0, 2*sigma0, K)[None,:, None]
true_theta.glob = np.array([1.])[None,:]
true_theta.loc[0,0,0] = 3*sigma0
true_theta.loc[0,1,0] = -3*sigma0
key, subkey = random.split(key)
y_obs = model.data_generator(subkey, true_theta)
# Vanilla 
## permABC
key, subkey = random.split(key)
N_points = 1000000
N_smc = 1000

print("permABC Vanilla...")
model.reset_weights_distance()
key, key_theta, key_perm = random.split(key, 3)
thetas = model.prior_generator(key_theta, N_points)
zs = model.data_generator(key, thetas)
dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs)
thetas_perm = thetas.apply_permutation(zs_index)
mus_perm_van = thetas_perm.loc.squeeze()
betas_perm_van = thetas_perm.glob.squeeze()


## ABC
print("Vanilla ABC...")
model.reset_weights_distance()
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

print("ABC SMC...")
model.reset_weights_distance()
out_smc = abc_smc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, kernel = kernel, verbose =1, Final_iteration = 0, update_weights_distance= False, N_sim_max= N_points*K, stopping_accept_rate= 0.015)

key, subkey = random.split(key)
kernel = KernelTruncatedRW
model.reset_weights_distance()

print("permABC SMC...")
out_perm_smc = perm_abc_smc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, kernel = kernel, verbose = 1, Final_iteration= 0, update_weights_distance= False, N_sim_max= N_points*K, stopping_accept_rate= 0.015)

# mus_perm_smc = out_perm_smc["Thetas"][-1].loc.squeeze()
# betas_perm_smc = out_perm_smc["Thetas"][-1].glob.squeeze()
# n_sim_perm_smc = np.sum(out_perm_smc["N_sim"])

from pmc import abc_pmc
key, subkey = random.split(key)
model.reset_weights_distance()
print("ABC PMC...")
out_pmc = abc_pmc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, verbose =1, update_weights_distance= False, N_sim_max= N_points*K, stopping_accept_rate= 0.015, alpha = .95)

alphas = np.logspace(0, np.log10(N_smc/N_points), 10)
print(alphas)
epsilons_van_perm = np.quantile(dists_perm, alphas)
epsilons_van = np.quantile(dists_van, alphas)

# Over-sampling
print("Over-sampling...")
key, subkey = random.split(key)
alpha_epsilon = .95
# M0s_test = np.arange(K+1, 10*K)
# N_OS = 10000
# key, subkey = random.split(key)
# thetas = model.prior_generator(subkey, N_OS, n_silos= np.max(M0s_test))
# key, subkey = random.split(key)
# zs = model.data_generator(subkey, thetas)
# thetas_M = []
# epsilons_os_test = np.zeros(len(M0s_test))
# N_plot = 2
# for j, M0 in enumerate(M0s_test):
#     zs_os = zs.copy()[:, :M0]
#     dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs_os, y_obs, M = M0)
#     epsilons_os_test[j] = np.quantile(dists_perm, alpha_epsilon)
 
# epsilons_os = []   
# M0s = []
# for epsilon_van in epsilons_van_perm:
#     index = np.argmin(np.abs(epsilons_os_test - epsilon_van))
#     epsilons_os.append(epsilons_os_test[index])
#     M0s.append(M0s_test[index])

# print("M0s = {}".format(M0s))
from over_sampling import perm_abc_smc_os
M0s = np.array([1.2*K, 1.5*K, 1.8*K, 2*K,  5*K, 7*K, 10*K], dtype=int)
epsilons_os = []
alphas_M = [.95, .75]
dico_os = {}
N_os = 1000
for M0 in M0s: 
    print("M0 = {}".format(M0))
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, N_smc, M0)
    key, subkey = random.split(key)
    
    zs = model.data_generator(subkey, thetas)
    dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs, M = M0)
    thetas_perm = thetas.apply_permutation(zs_index)
    
    epsilon = np.quantile(dists_perm, alpha_epsilon)
    print("Epsilon = {}".format(epsilon))
    epsilons_os.append(epsilon)
    dico_os[M0] = {}
    for alpha_M in alphas_M:
        
        key, subkey = random.split(key)
        dico_os[M0][alpha_M] = perm_abc_smc_os(key = subkey, model = model, n_particles= N_os, y_obs = y_obs, kernel = kernel, M_0 = M0, epsilon = epsilon, alpha_M = alpha_M, update_weights_distance= False, verbose = 0, Final_iteration= 0, duplicate = True)
epsilons_os = np.array(epsilons_os)


from under_matching import perm_abc_smc_um
L0s = np.array(np.linspace(4, K, K), dtype=int)
alphas_L = np.array([.95,  .75])
dico_um = {}
kernel = KernelTruncatedRW
N_um = 1000

for L0 in L0s:
    print("L0 = {}".format(L0))
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, N_smc)
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)
    dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs, L = L0)
    epsilon = np.quantile(dists_perm, alpha_epsilon)
    dico_um[L0] = {}
    for alpha_L in alphas_L:
        key, subkey = random.split(key)
        dico_um[L0][alpha_L] = perm_abc_smc_um(key = subkey, model = model, n_particles= N_um, y_obs = y_obs, kernel = kernel, L_0 = L0, epsilon = epsilon, alpha_L = alpha_L, update_weights_distance= False, verbose = 0, Final_iteration= 0)
        

# Comparison
N_sample = 1



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

n_sim_os_0_95 = np.array([np.sum(dico_os[M0][.95]["N_sim"]) for M0 in M0s])
n_sim_os_0_75 = np.array([np.sum(dico_os[M0][.75]["N_sim"]) for M0 in M0s])
unique_os_0_95 = np.array([dico_os[M0][.95]["unique_part"][-1] for M0 in M0s])
unique_os_0_75 = np.array([dico_os[M0][.75]["unique_part"][-1] for M0 in M0s])
epsilons_os = np.array(epsilons_os)
n_sim_um_0_95 = np.array([np.sum(dico_um[L0][.95]["N_sim"]) if dico_um[L0][.95] is not None else 0 for L0 in L0s])
n_sim_um_0_75 = np.array([np.sum(dico_um[L0][.75]["N_sim"]) if dico_um[L0][.75] is not None else 0 for L0 in L0s])
unique_um_0_95 = np.array([dico_um[L0][.95]["unique_part"][-1] if dico_um[L0][.95] is not None else 0 for L0 in L0s])
unique_um_0_75 = np.array([dico_um[L0][.75]["unique_part"][-1] if dico_um[L0][.75] is not None else 0 for L0 in L0s])
epsilon_um_0_95 = np.array([dico_um[L0][.95]["Eps_values"][-1] if dico_um[L0][.95] is not None else 0 for L0 in L0s])
epsilon_um_0_75 = np.array([dico_um[L0][.75]["Eps_values"][-1] if dico_um[L0][.75] is not None else 0 for L0 in L0s])


alphas = np.logspace(0, -4, 10)
epsilons_van_perm = np.quantile(dists_perm, alphas)
epsilons_van = np.quantile(dists_van, alphas)


N_sims = N_sample/alphas

N_sim_pmc = np.cumsum(n_sim_pmc)/K/unique_pmc/N_smc*N_sample
N_sim_smc = np.cumsum(n_sim_smc)/K/unique_smc/N_smc*N_sample
N_sim_perm_smc = np.cumsum(n_sim_perm_smc)/K/unique_perm_smc/N_smc*N_sample


closest_index_smc = np.argmin(np.abs(N_sim_smc - N_sims[:, None]), axis=1)
closest_index_perm_smc = np.argmin(np.abs(N_sim_perm_smc - N_sims[:, None]), axis=1)
closest_index_pmc = np.argmin(np.abs(N_sim_pmc - N_sims[:, None]), axis=1)
f, ax = plt.subplots(1, 1, figsize=(10, 6))
plt.plot(N_sims, np.quantile(dists_perm, alphas), label="permABC Vanilla", linestyle="-", marker="x")
plt.plot(N_sims, np.quantile(dists_van, alphas), label="ABC Vanilla", linestyle="--", marker="x")
plt.plot(N_sim_perm_smc[closest_index_perm_smc], epsilon_perm_smc[closest_index_perm_smc], label="permABC-SMC", linestyle="-", marker="o")
plt.plot(N_sim_smc[closest_index_smc], epsilon_smc[closest_index_smc], label="ABC-SMC", linestyle="--", marker="o")
plt.plot(N_sim_pmc[closest_index_pmc], epsilon_pmc[closest_index_pmc], label="ABC-PMC", linestyle="--", marker="o")

N_sim_os_0_75 = n_sim_os_0_75/K/unique_os_0_75/N_os*N_sample
N_sim_os_0_95 = n_sim_os_0_95/K/unique_os_0_95/N_os*N_sample

plt.plot(N_sim_os_0_95, epsilons_os, label="permABC-SMC-OS alpha_M = 0.95", linestyle="-", marker="^")
plt.plot(N_sim_os_0_75, epsilons_os, label="permABC-SMC-OS alpha_M = 0.75", linestyle="-", marker="^")

N_sim_um_0_95 = n_sim_um_0_95/K/unique_um_0_95/N_um*N_sample
N_sim_um_0_75 = n_sim_um_0_75/K/unique_um_0_75/N_um*N_sample

plt.plot(N_sim_um_0_95, epsilon_um_0_95, label="permABC-SMC-UM alpha_L = 0.95", linestyle="-", marker="s")
plt.plot(N_sim_um_0_75, epsilon_um_0_75, label="permABC-SMC-UM alpha_L = 0.75", linestyle="-", marker="s")



plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of simulations per {} unique particles".format(N_sample)) 
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.savefig("figures/vanilla_vs_smc_vs_osum_2outliers.png")
plt.close()

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
    "epsilons_os": epsilons_os,
    "n_sim_os_0_95": n_sim_os_0_95,
    "n_sim_os_0_75": n_sim_os_0_75,
    "unique_os_0_95": unique_os_0_95,
    "unique_os_0_75": unique_os_0_75,
    "epsilon_um_0_95": epsilon_um_0_95,
    "epsilon_um_0_75": epsilon_um_0_75,
    "n_sim_um_0_95": n_sim_um_0_95,
    "n_sim_um_0_75": n_sim_um_0_75,
    "unique_um_0_95": unique_um_0_95,
    "unique_um_0_75": unique_um_0_75,
    "dico_os": dico_os,
    "true_theta": true_theta,
    "y_obs": y_obs,
   
}

with open("figures/vanilla_vs_smc_vs_osum_2outliers.pkl", "wb") as f:
    pickle.dump(data_to_save, f)