import os 

os.chdir("..")
os.chdir("..")
import sys
sys.path.append(os.getcwd())
sys.stdout.reconfigure(line_buffering=True)
from smc import abc_smc, perm_abc_smc
from pmc import abc_pmc
from over_sampling import perm_abc_smc_os
from under_matching import perm_abc_smc_um  
from kernels import KernelTruncatedRW
from distances import optimal_index_distance
from models.Gaussian_with_no_summary_stats import GaussianWithNoSummaryStats
from jax import random
import numpy as np
import matplotlib.pyplot as plt
import time
from jax import config
config.update("jax_disable_jit", True)
import os

N_points = 1000000
N_smc = 1000
N_os = 1000
N_um = 1000
stopping_rate = .015

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
\
if len(sys.argv) > 1:
    K = int(sys.argv[1])
else:
    K = 10
    
if len(sys.argv) > 2:
    K_outliers = int(sys.argv[2])
else:
    K_outliers = 0
    

key = random.PRNGKey(0)
key, subkey = random.split(key)
n = 10
sigma0 = 10
alpha, beta = 5,5
model = GaussianWithNoSummaryStats(K = K, n_obs= n, sigma_0 = sigma0, alpha = alpha, beta = beta)
true_theta = model.prior_generator(subkey, 1)
# true_theta.loc = np.linspace(-2*sigma0, 2*sigma0, K)[None,:, None]
true_theta.glob = np.array([1.])[None,:]

for i in range(K_outliers):
    key, subkey = random.split(key)
    sign = random.choice(subkey, a = np.array([-1, 1]), shape = (1,))
    key, subkey = random.split(key)
    if sign == 1:
        true_theta.loc[0,i,0] = random.uniform(subkey, shape=(1,), minval=-3*sigma0, maxval=-2*sigma0)[0]
    else:
        true_theta.loc[0,i,0] = random.uniform(subkey, shape=(1,), minval=2*sigma0, maxval=3*sigma0)[0]
    
key, subkey = random.split(key)
y_obs = model.data_generator(subkey, true_theta)
# Vanilla 
## permABC
key, subkey = random.split(key)


print("Vanilla...")
model.reset_weights_distance()
time_0 = time.time()
key, key_theta, key_data = random.split(key, 3)
thetas = model.prior_generator(key_theta, N_points)
zs = model.data_generator(key_data, thetas)
dists = model.distance(zs, y_obs)
time_van = time.time() - time_0
print("Time for vanilla: {}".format(time_van))


print("permABC Vanilla...")
model.reset_weights_distance()
time_0 = time.time()
key, key_theta, key_data = random.split(key, 3)
thetas = model.prior_generator(key_theta, N_points)
zs = model.data_generator(key_data, thetas)
dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs)
time_perm_van = time.time() - time_0
print("Time for permABC Vanilla: {}".format(time_perm_van))

print("ABC SMC...")
key, subkey = random.split(key)
kernel = KernelTruncatedRW
model.reset_weights_distance()

out_smc = abc_smc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, kernel = kernel, verbose = 1, Final_iteration= 0, update_weights_distance= False, stopping_accept_rate= stopping_rate, N_sim_max= N_points*K)
n_sim_smc = np.cumsum(out_smc["N_sim"][1:])
epsilons_smc = np.array(out_smc["Eps_values"])[1:]
unique_smc = np.array(out_smc["unique_part"])[1:]
time_smc = np.cumsum(out_smc["Time"][1:])

print("ABC PMC...")
key, subkey = random.split(key)
kernel = KernelTruncatedRW
model.reset_weights_distance()

out_pmc = abc_pmc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs,  alpha = .95, verbose = 1, update_weights_distance= False, stopping_accept_rate= stopping_rate, N_sim_max= N_points*K)
n_sim_pmc = np.cumsum(out_pmc["N_sim"][1:])
epsilons_pmc = np.array(out_pmc["Eps_values"])[1:]
unique_pmc = np.array(out_pmc["unique_part"])[1:]
time_pmc = np.cumsum(out_pmc["Time"][1:])


print("permABC SMC...")
out_perm_smc = perm_abc_smc(key = subkey, model = model, n_particles= N_smc, epsilon_target= 0, y_obs = y_obs, kernel = kernel, verbose = 1, Final_iteration= 0, update_weights_distance= False, stopping_accept_rate= stopping_rate, N_sim_max= N_points*K)

n_sim_perm_smc = np.cumsum(out_perm_smc["N_sim"][1:])
epsilons_perm_smc = np.array(out_perm_smc["Eps_values"])[1:]
unique_perm_smc = np.array(out_perm_smc["unique_part"])[1:]
time_perm_smc = np.cumsum(out_perm_smc["Time"][1:])


# Over-sampling
print("Over-sampling...")
key, subkey = random.split(key)
alpha_epsilon = .95

from over_sampling import perm_abc_smc_os
M0s = np.array([1.5*K, 2*K,  5*K, 7*K, 10*K, 15*K, 20*K, 25*K], dtype=int)

epsilons_os = []
time_os = []
n_sim_os = []
unique_os = []
alpha_M = .9

N_epsilon = 10000
for M0 in M0s: 
    print("M0 = {}".format(M0))
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, N_epsilon, M0)
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)
    dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs, M = M0)
    thetas_perm = thetas.apply_permutation(zs_index)
    epsilon = np.quantile(dists_perm, alpha_epsilon)
    print("Epsilon = {}".format(epsilon))
    epsilons_os.append(epsilon)
    model.reset_weights_distance()
    key, subkey = random.split(key)
    out_os = perm_abc_smc_os(key = subkey, model = model, n_particles= N_os, y_obs = y_obs, kernel = kernel, M_0 = M0, epsilon = epsilon, alpha_M = alpha_M, update_weights_distance= False, verbose = 0, Final_iteration= 0, duplicate = True)
    n_sim_os.append(np.sum(out_os["N_sim"]))
    unique_os.append(out_os["unique_part"][-1])
    time_os.append(out_os["time_final"])

epsilons_os = np.array(epsilons_os)
n_sim_os = np.array(n_sim_os)
unique_os = np.array(unique_os)
time_os = np.array(time_os)


from under_matching import perm_abc_smc_um
L0s = np.array(np.linspace(2, K, K), dtype=int)
alpha_L = .9
epsilons_um = []
n_sim_um = []
unique_um = []
time_um = []

kernel = KernelTruncatedRW

for L0 in L0s:
    print("L0 = {}".format(L0))
    key, subkey = random.split(key)
    thetas = model.prior_generator(subkey, N_epsilon)
    key, subkey = random.split(key)
    zs = model.data_generator(subkey, thetas)
    dists_perm, ys_index, zs_index, _ = optimal_index_distance(model, zs, y_obs, L = L0)
    epsilon = np.quantile(dists_perm, alpha_epsilon)
    
    key, subkey = random.split(key)
    model.reset_weights_distance()
    out_um = perm_abc_smc_um(key = subkey, model = model, n_particles= N_um, y_obs = y_obs, kernel = kernel, L_0 = L0, epsilon = epsilon, alpha_L = alpha_L, update_weights_distance= False, verbose = 0, Final_iteration= 0)
    if out_um!= None:
        n_sim_um.append(np.sum(out_um["N_sim"]))
        unique_um.append(out_um["unique_part"][-1])
        epsilons_um.append(out_um["Eps_values"][-1])
        time_um.append(out_um["time_final"])
    
epsilons_um = np.array(epsilons_um)
n_sim_um = np.array(n_sim_um)
unique_um = np.array(unique_um)
time_um = np.array(time_um)

# Comparison
import seaborn as sns
import matplotlib.pyplot as plt



N_sample = 1000

time_by_sim_van = time_van/N_points
alphas = np.logspace(0, -3, 10)
n_sim_van = 1/alphas*N_sample
time_van = n_sim_van * time_by_sim_van
epsilons_van = np.quantile(dists, alphas)

time_by_sim_perm_van = time_perm_van/N_points
n_sim_perm_van = 1/alphas*N_sample
time_perm_van = n_sim_perm_van * time_by_sim_perm_van
epsilons_perm_van = np.quantile(dists_perm, alphas)

n_sim_unique_perm_smc = n_sim_perm_smc/(K*unique_perm_smc*N_smc)*N_sample
n_sim_unique_os = n_sim_os/(K*unique_os*N_os)*N_sample
n_sim_unique_um = n_sim_um/(K*unique_um*N_um)*N_sample
n_sim_unique_pmc = n_sim_pmc/(K*unique_pmc*N_smc)*N_sample
n_sim_unique_smc = n_sim_smc/(K*unique_smc*N_smc)*N_sample

time_unique_perm_smc = time_perm_smc/(K*unique_perm_smc*N_smc)*N_sample
time_unique_os = time_os/(K*unique_os*N_os)*N_sample
time_unique_um = time_um/(K*unique_um*N_um)*N_sample
time_unique_pmc = time_pmc/(K*unique_pmc*N_smc)*N_sample
time_unique_smc = time_smc/(K*unique_smc*N_smc)*N_sample


f, ax = plt.subplots(1, 1, figsize=(10, 6))

plt.plot(n_sim_perm_van, epsilons_perm_van, label="permABC-Vanilla", linestyle="-", marker="o")
plt.plot(n_sim_van, epsilons_van, label="ABC-Vanilla", linestyle="-", marker="s")
plt.plot(n_sim_unique_perm_smc, epsilons_perm_smc, label="permABC-SMC", linestyle="--", marker="o")
plt.plot(n_sim_unique_os, epsilons_os, label="permABC-SMC-OS", linestyle="--", marker="^")
plt.plot(n_sim_unique_um, epsilons_um, label="permABC-SMC-UM", linestyle="--", marker="s")
plt.plot(n_sim_unique_smc, epsilons_smc, label="ABC-SMC", linestyle="--", marker="o")
plt.plot(n_sim_unique_pmc, epsilons_pmc, label="ABC-PMC", linestyle="--", marker="s")


plt.yscale("log")
plt.xscale("log")
plt.xlabel("Number of simulations per {} unique particles".format(N_sample)) 
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.savefig("figures/osum/n_sim/K_{}_K_outliers_{}.pdf".format(K, K_outliers))
plt.close()


f, ax = plt.subplots(1, 1, figsize=(10, 6))

plt.plot(time_perm_van, epsilons_perm_van, label="permABC-Vanilla", linestyle="-", marker="o")
plt.plot(time_van, epsilons_van, label="ABC-Vanilla", linestyle="-", marker="s")
plt.plot(time_perm_smc, epsilons_perm_smc, label="permABC-SMC", linestyle="-", marker="o")
plt.plot(time_unique_os, epsilons_os, label="permABC-SMC-OS", linestyle="-", marker="^")
plt.plot(time_unique_um, epsilons_um, label="permABC-SMC-UM", linestyle="-", marker="s")
plt.plot(time_smc, epsilons_smc, label="ABC-SMC", linestyle="-", marker="o")
plt.plot(time_pmc, epsilons_pmc, label="ABC-PMC", linestyle="-", marker="s")


plt.yscale("log")
plt.xscale("log")
plt.xlabel("Time per {} unique particles (in seconds)".format(N_sample))
plt.ylabel(r"$\epsilon$")
plt.legend()
plt.savefig("figures/osum/time/K_{}_K_outliers_{}.pdf".format(K, K_outliers))
plt.close()



dico = {"n_sim_perm_smc": n_sim_perm_smc,
        "n_sim_os": n_sim_os,
        "n_sim_um": n_sim_um,
        "n_sim_smc": n_sim_smc,
        "n_sim_pmc": n_sim_pmc,
        "n_sim_van": n_sim_van,
        "n_sim_perm_van": n_sim_perm_van,
        
        
        "time_perm_smc": time_perm_smc,
        "time_os": time_os,
        "time_um": time_um,
        "time_smc": time_smc,
        "time_pmc": time_pmc,
        "time_van": time_van,
        "time_perm_van": time_perm_van,
        
        
        "epsilons_perm_smc": epsilons_perm_smc,
        "epsilons_os": epsilons_os,
        "epsilons_um": epsilons_um,
        "epsilons_smc": epsilons_smc,
        "epsilons_pmc": epsilons_pmc,
        "epsilons_van": epsilons_van,
        "epsilons_perm_van": epsilons_perm_van,
        
        
        "unique_perm_smc": unique_perm_smc,
        "unique_os": unique_os,
        "unique_um": unique_um, 
        "unique_smc": unique_smc,
        "unique_pmc": unique_pmc,
        
        "K": K,
        "K_outliers": K_outliers,
        "N_points": N_points,
        "N_smc": N_smc,
        "N_os": N_os,
        "N_um": N_um,
        "M0s": M0s,
        "L0s": L0s,
        
        "alpha_epsilon": alpha_epsilon,
        "alpha_M": alpha_M,
        "alpha_L": alpha_L,
        "y_obs": y_obs,
        
        "n_sim_unique_perm_smc": n_sim_unique_perm_smc,
        "n_sim_unique_os": n_sim_unique_os,
        "n_sim_unique_um": n_sim_unique_um,
        "n_sim_unique_smc": n_sim_unique_smc,
        "n_sim_unique_pmc": n_sim_unique_pmc,
        
        "time_unique_perm_smc": time_unique_perm_smc,
        "time_unique_os": time_unique_os,
        "time_unique_um": time_unique_um,  
        "time_unique_smc": time_unique_smc,
        "time_unique_pmc": time_unique_pmc,
}

import pickle
import lzma

with open("figures/osum/K_{}_K_outliers_{}.pkl".format(K, K_outliers), "wb") as f:
    pickle.dump(dico, f)
