# Initialization
import os 

os.chdir("/Users/antoineluciano/Documents/Recherche/permABC_new/permABC")
import sys
sys.path.append(os.getcwd())

from distances import optimal_index_distance
from models.uniform_known import Uniform_known
from jax import random
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# With unknown Beta

model = Uniform_known(K = 2)
# y_obs = np.array([-1,1])[None, :, None]
y_obs = np.array([0., 0.5])[None, :, None]
key = random.PRNGKey(0)
y = y_obs[0,:,0]
epsilon_star = model.distance(y_obs, y_obs[:,::-1])[0]/2
print("Epsilon star", epsilon_star)

model_known = Uniform_known(K = 2)
## ABC
key, subkey = random.split(key)
Nsim = 1000000

key, subkey = random.split(key)
thetas = model.prior_generator(subkey, Nsim)
key, subkey = random.split(key)
zs = model.data_generator(subkey, thetas)
dists = model.distance(zs, y_obs)
dists_perm, _, zs_index, _ = optimal_index_distance(model, zs = zs, y_obs = y_obs, epsilon = 0, verbose= 2)
zs_perm = zs[np.arange(Nsim)[:,None], zs_index]
thetas_perm = thetas.apply_permutation(zs_index)


epsilons = [np.inf, epsilon_star+1, epsilon_star]
colors =["tab:blue", "tab:red", "tab:orange", "gold"]
colors = plt.cm.viridis(np.linspace(0,1,len(epsilons)))
N = 10000
s = 1
x = np.linspace(-2,2,100)
alpha_fig = 0.25
fontsize = 12
title_size = 15


f, ax = plt.subplots(1,2, figsize = (10,5), tight_layout=True)

for i, epsilon in enumerate(epsilons): 
    index = np.random.choice(np.where(dists<=epsilon)[0], N, replace = False)
    index_perm = np.random.choice(np.where(dists_perm<=epsilon)[0], N, replace = False)

    mus = thetas.loc[index].squeeze()
    
    mus_perm = thetas_perm.loc[index_perm].squeeze()
  
    ax[0].scatter(x=mus[:,0], y=mus[:,1], color = colors[i], alpha = alpha_fig,  s = 10)
    ax[1].scatter(x=mus_perm[:, 0], y=mus_perm[:, 1], color = colors[i], alpha = alpha_fig, s=10)
prior = np.array([[-2,-2],[2,-2],[2,2],[-2,2],[-2,-2]])
posterior = np.array([[y[0]-s, y[1]-s], [y[0]+s, y[1]-s], [y[0]+s, y[1]+s], [y[0]-s, y[1]+s], [y[0]-s, y[1]-s]])
ax[0].scatter(y_obs[0,0,0], y_obs[0,1,0], c='black', label='y',marker='x')
# ax[0].plot(x,x, '--', color='black')
ax[0].plot(posterior[:, 0], posterior[:, 1], color='black',linestyle='--')
ax[0].plot(prior[:, 0], prior[:, 1], color='grey',linestyle='--')
ax[0].set_title("ABC", fontsize = title_size)
ax[0].set_xlabel('$\\mu_1$', fontsize = fontsize)
ax[0].set_ylabel('$\\mu_2$', fontsize = fontsize)
ax[0].set_xlim(-2.2,2.2)
ax[0].set_ylim(-2.2,2.2)
ax[0].set_aspect('equal')
ax[1].set_aspect('equal')
ax[1].scatter(y_obs[0,0,0], y_obs[0,1,0], c='black', label='y',marker='x')
# ax[1].plot(x,x, '--', color='black')
ax[1].plot(posterior[:, 0], posterior[:, 1], color='black',linestyle='--')
ax[1].plot(prior[:, 0], prior[:, 1], color='grey',linestyle='--')
ax[1].set_title('permABC', fontsize = title_size)
ax[1].set_xlabel('$\\mu_1$', fontsize = fontsize)
# ax[1].set_ylabel('$\\mu_2$', fontsize = fontsize)
ax[1].set_xlim(-2.2,2.2)
ax[1].set_ylim(-2.2,2.2)
    
plt.savefig("figures/perm_vs_vanilla.svg", dpi=300)
plt.savefig("figures/perm_vs_vanilla.png", dpi=300)
plt.savefig("figures/perm_vs_vanilla.pdf", dpi=300)

    