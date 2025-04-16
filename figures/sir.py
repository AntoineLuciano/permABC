# Package importation
import os 

os.chdir("/Users/antoineluciano/Documents/Recherche/permABC_new/permABC")
import sys
sys.path.append(os.getcwd())

# from vanilla import abc_vanilla, perm_abc_vanilla
from smc import abc_smc, perm_abc_smc
# from over_sampling import perm_abc_smc_os
# from under_matching import perm_abc_smc_um  
from kernels import KernelTruncatedRW
from distances import optimal_index_distance
from models.SIR import SIRWithKnownInit, SIRWithUnknownInit
from jax import random
import numpy as np
import jax.numpy as jnp
from scipy.stats import invgamma
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
def fit_data(mod, gen, y_obs, thetas, names, size = 100):
    K = y_obs.shape[0]

    index_random = np.random.choice(np.arange(1000),size=min(size, len(thetas)), replace = False)
    zs = mod.data_generator(gen, thetas[index_random])
    for k in range(K):
        for i in range(len(index_random)):
            plt.plot(zs[i][k], color = "blue", alpha = .1)
        plt.title(str(k)+": "+names[k])
        plt.plot(y_obs[k],label="Observed",color = "red")
        if k==0: plt.legend()
        plt.show()
# Data importation
## Population and name
df_pop = pd.read_csv("./datas/donnees_departements.csv", sep = ";")

dep_pop = {df_pop.loc[i]["CODDEP"]:df_pop.loc[i]["PTOT"] for i in range(df_pop.shape[0])}
dep_name = {df_pop.loc[i]["CODDEP"]:df_pop.loc[i]["DEP"] for i in range(df_pop.shape[0])}

reg = np.unique(df_pop["CODREG"])
reg_pop = {}
reg_name = {}
for r in reg: 
    reg_pop[r] = df_pop[df_pop["CODREG"]==r]["PTOT"].sum()
    reg_name[r] = df_pop[df_pop["CODREG"]==r]["REG"].iloc[0]

dep_reg = {df_pop.loc[i]["CODDEP"]:df_pop.loc[i]["CODREG"] for i in range(df_pop.shape[0])}
## Data by departement
df_dep = pd.read_csv("./datas/data-dep.csv",sep=";")
df_dep=df_dep.iloc[np.where(df_dep.loc[:]["sexe"]==0)]
df_dep = df_dep[~df_dep['dep'].isin(["971", "972", "973", "974", "976", "978","2A","2B"])]


date=np.unique(np.array(pd.to_datetime(df_dep.iloc[:]["jour"])))
df_dep["reg"] = ""
df_dep["pop"] = ""

dep_list = np.unique(df_dep["dep"])
for d in dep_list:  
    df_dep.loc[df_dep["dep"]==d,"reg"] = dep_reg[d]
    df_dep.loc[df_dep["dep"]==d,"pop"] = dep_pop[d]
    
I_obs_dep = np.array([df_dep.loc[df_dep["dep"]==d]["hosp"].rolling(window=7).mean()[6:]/dep_pop[d]*100000*15 for d in dep_list])

date = date[6:]
K_dep = I_obs_dep.shape[0]
I_obs_dep.shape



plt.show()
## Data by region
df_reg = df_dep.groupby(['jour', 'reg']).sum().reset_index()
reg_list = np.unique(df_reg["reg"])
I_obs_reg = np.array([df_reg.loc[df_reg["reg"]==r]["hosp"].rolling(window=7).mean()[6:]/reg_pop[r]*100000*15 for r in reg_list])
K_reg = I_obs_reg.shape[0]
I_obs_reg.shape


## Data at national scale
df_fr = df_dep.groupby(['jour']).sum().reset_index()

K_fr = 1
I_obs_fr = np.array(df_fr["hosp"].rolling(window=7).mean()[6:]/df_fr["pop"][0]*100000*15)

## Focus on the first wave
n_day = 120
date_V1 = date[:n_day]
print("From {} to {}".format(date_V1[0],date_V1[-1]))
I_obs_dep_V1 = I_obs_dep[:,:n_day]
I_obs_reg_V1 = I_obs_reg[:,:n_day]
I_obs_fr_V1 = I_obs_fr[:n_day]


high_I, high_R, high_beta, high_r0 = 1000, 1000, 5, 5
n_pop = 100000
mod_fr = SIRWithUnknownInit(K_fr, n_obs = n_day, n_pop = n_pop, high_I = high_I, high_R = high_R, high_beta = high_beta, high_r0 = high_r0)
y_obs_fr = I_obs_fr_V1[None, None, :]

weights_region = [reg_pop[r] for r in reg_list]
mod_reg = SIRWithUnknownInit(K = K_reg, n_obs = n_day, n_pop = n_pop, high_I = high_I, high_R = high_R, high_beta=high_beta, high_r0=high_r0, weights_distance = weights_region)
y_obs_reg = np.array(I_obs_reg_V1)[None,:]

weights_dep = [dep_pop[d] for d in dep_list]
mod_dep = SIRWithUnknownInit(K = K_dep, n_obs = n_day, n_pop = n_pop, high_I = high_I, high_R = high_R, high_beta = high_beta, high_r0 = high_r0, weights_distance = weights_dep)
y_obs_dep = np.array(I_obs_dep_V1)[None,:]

# Fit the data
key = random.PRNGKey(0)

key, key_smc = random.split(key)
N = 1000
alpha = .95
smc_fr = abc_smc(key = key_smc, model = mod_fr, y_obs = y_obs_fr, n_particles= N, kernel = KernelTruncatedRW, verbose = 1, epsilon_target = 0., update_weights_distance= False, Final_iteration= 50)

thetas_smc_fr = smc_fr["Thetas"][-1]
zs_smc_fr = smc_fr["Zs"][-1]
epsilon_smc_fr = smc_fr["Eps_values"][-1]
n_iterations_smc_fr = len(smc_fr["Eps_values"])


smc_reg= perm_abc_smc(key = key, model = mod_reg, y_obs = y_obs_reg, n_particles= N, kernel = KernelTruncatedRW, verbose = 1, epsilon_target = 0., update_weights_distance= False, Final_iteration= 50, num_blocks_gibbs= 2, both_loc_glob= True)
thetas_smc_reg = smc_reg["Thetas"][-1]
zs_smc_reg = smc_reg["Zs"][-1]
epsilon_smc_reg = smc_reg["Eps_values"][-1]
n_iterations_smc_reg = len(smc_reg["Eps_values"])


smc_dep= perm_abc_smc(key = key, model = mod_dep, y_obs = y_obs_dep, n_particles= N, kernel = KernelTruncatedRW, verbose = 1, epsilon_target = 0., update_weights_distance= False, Final_iteration= 50, num_blocks_gibbs= 5, both_loc_glob= True)
thetas_smc_dep = smc_dep["Thetas"][-1]
zs_smc_dep = smc_dep["Zs"][-1]
epsilon_smc_dep = smc_dep["Eps_values"][-1]
n_iterations_smc_dep = len(smc_dep["Eps_values"])

#Save the results
dico_save = {"Thetas_smc_fr": thetas_smc_fr, "Zs_smc_fr": zs_smc_fr, "Eps_smc_fr": epsilon_smc_fr, "n_iterations_smc_fr": n_iterations_smc_fr,
            "Thetas_smc_reg": thetas_smc_reg, "Zs_smc_reg": zs_smc_reg, "Eps_smc_reg": epsilon_smc_reg, "n_iterations_smc_reg": n_iterations_smc_reg,
            "Thetas_smc_dep": thetas_smc_dep, "Zs_smc_dep": zs_smc_dep, "Eps_smc_dep": epsilon_smc_dep, "n_iterations_smc_dep": n_iterations_smc_dep}

import pickle
with open("./datas/sir_results.pkl", "wb") as f:
    pickle.dump(dico_save, f)

# Plot the results

f, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.kdeplot(thetas_smc_dep.glob[:,0],  label='Departments scale', linewidth=2, linestyle='-', color='blue')
sns.kdeplot(thetas_smc_reg.glob[:,0], label='Regions scale', linewidth=2, color='orange', linestyle='--')
sns.kdeplot(thetas_smc_fr.glob[:,0], label='National scale', linewidth=2, color='green', linestyle='-.')

# Adding labels and title
plt.xlabel('$R_0$', fontsize=12)
plt.ylabel('Density', fontsize=12)


# Adding legend
# plt.legend()

# Display the plo
plt.savefig("SIR_R0_posterior.pdf")
plt.show()

