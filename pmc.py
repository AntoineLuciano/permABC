import numpy as np
import numba as nb
import particles
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, multivariate_normal
from scipy.optimize import linear_sum_assignment
from jax import random,jit, vmap
import jax.numpy as jnp
from utils import Theta
from jax.scipy.stats import truncnorm
from jax.scipy.special import logsumexp
from tqdm import tqdm
import time
    
def init_pmc(key, model, n_particles, y_obs, update_weights_distance = True, verbose = 1):
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    thetas = model.prior_generator(key_thetas, n_particles, K)
    zs = model.data_generator(key_zs, thetas)
    if update_weights_distance: model.update_weights_distance(zs, verbose)
    distance_values = model.distance(zs, y_obs)
    weights = np.ones(n_particles) / n_particles
    ess_val = n_particles
    return thetas, zs, distance_values, weights, ess_val
    
def update_epsilon(alive_distances, epsilon_target, alpha):
    return max(np.quantile(alive_distances,alpha), epsilon_target)



        
   
def ess(weights):
    return 1 / np.sum(weights**2)

def move_pmc(key, model, thetas, weights, y_obs,size, std_loc, std_glob):
    key,key_index, key_loc, key_glob, key_data = random.split(key, 5)
    
    indexes = np.array(random.choice(key_index, a= thetas.loc.shape[0], shape=(size,), p=weights, replace=True))
    proposed_thetas = thetas[indexes]
    
    proposed_thetas_loc = random.truncated_normal(key_loc, lower = (model.support_par_loc[0,0]-proposed_thetas.loc)/std_loc, upper = (model.support_par_loc[0,1]-proposed_thetas.loc)/std_loc, shape=proposed_thetas.loc.shape)*std_loc + proposed_thetas.loc
    while np.isinf(proposed_thetas_loc).any():
        print("\n\nINFINITE VALUE IN PROPOSED THETAS LOC\n\n")
        key, key_loc = random.split(key)
        proposed_thetas_loc = random.truncated_normal(key_loc, lower = (model.support_par_loc[0,0]-proposed_thetas.loc)/std_loc, upper = (model.support_par_loc[0,1]-proposed_thetas.loc)/std_loc, shape=proposed_thetas.loc.shape)*std_loc + proposed_thetas.loc

    proposed_thetas_glob = random.truncated_normal(key_glob, lower = (model.support_par_glob[0,0]-proposed_thetas.glob)/std_glob, upper = (model.support_par_loc[0,1]-proposed_thetas.glob)/std_glob, shape=proposed_thetas.glob.shape)*std_glob + proposed_thetas.glob
    while np.isinf(proposed_thetas_glob).any():
        print("\n\nINFINITE VALUE IN PROPOSED THETAS GLOB\n\n")
        key, key_glob = random.split(key)
        proposed_thetas_glob = random.truncated_normal(key_glob, lower = (model.support_par_glob[0,0]-proposed_thetas.glob)/std_glob, upper = (model.support_par_loc[0,1]-proposed_thetas.glob)/std_glob, shape=proposed_thetas.glob.shape)*std_glob + proposed_thetas.glob
    
    proposed_thetas = Theta(loc = proposed_thetas_loc, glob = proposed_thetas_glob)

    proposed_zs = model.data_generator(key_data, proposed_thetas)
    
    proposed_distances = model.distance(proposed_zs, y_obs)
    
    return proposed_thetas, proposed_distances, proposed_zs



@jit 
def K_t_ij(thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, std_loc, std_glob, a_loc, b_loc, a_glob, b_glob):
    
    logpdf_loc = jnp.sum(truncnorm.logpdf(x = thetas_t_loc, a = (a_loc-thetas_t_1_loc)/std_loc, b = (b_loc-thetas_t_1_loc)/std_loc, loc = thetas_t_1_loc, scale = std_loc))
    logpdf_glob = (truncnorm.logpdf(x = thetas_t_glob, a = (a_glob-thetas_t_1_glob)/std_glob, b = (b_glob-thetas_t_1_glob)/std_glob, loc = thetas_t_1_glob, scale = std_glob))
    return (logpdf_loc + logpdf_glob + jnp.log(weights_t_1))


@jit
def K_t_i(thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, std_loc, std_glob, a_loc, b_loc, a_glob, b_glob):
    res = vmap(K_t_ij, (None,None,0,0,0,None,None,None,None,None,None))(thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, std_loc, std_glob, a_loc, b_loc, a_glob, b_glob)
    return logsumexp(res, axis = 0)

@jit
def K_t(thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, std_loc, std_glob, a_loc, b_loc, a_glob, b_glob):
    return (vmap(K_t_i, (0,0,None,None,None,None,None,None,None,None,None))(thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, std_loc, std_glob, a_loc, b_loc, a_glob, b_glob))


def update_weights(model, thetas_t, thetas_t_1, weights_t_1, std_loc, std_glob):
    a_loc = model.support_par_loc[0,0]
    b_loc = model.support_par_loc[0,1]
    a_glob = model.support_par_glob[0,0]
    b_glob = model.support_par_glob[0,1]
    std_loc = std_loc.squeeze()
    std_glob = std_glob.squeeze()
    thetas_t_loc, thetas_t_glob = np.array(thetas_t.loc).squeeze(), np.array(thetas_t.glob).squeeze()
    thetas_t_1_loc, thetas_t_1_glob = np.array(thetas_t_1.loc).squeeze(), np.array(thetas_t_1.glob).squeeze()
    weights_t_1 = np.array(weights_t_1).squeeze()
    logdenominateur = K_t(thetas_t_loc, thetas_t_glob, thetas_t_1_loc, thetas_t_1_glob, weights_t_1, std_loc, std_glob, a_loc, b_loc, a_glob, b_glob)    
    logprior = model.prior_logpdf(thetas_t).reshape(-1)
    weights = jnp.exp(logprior-logdenominateur)
    weights = weights / jnp.sum(weights)
  
    
    print("Log Denominateur: min = {:.3} (index = {}), max = {:.3} (index = {})".format(np.min(logdenominateur), np.argmin(logdenominateur), np.max(logdenominateur), np.argmax(logdenominateur)))
    print("Log prior: min {:.4} (index = {}), max = {:.4} (index = {})".format(np.min(logprior), np.argmin(logprior), np.max(logprior), np.argmax(logprior)))
    print("Weights: min = {:.3} (index = {}), max = {:.3} (index = {})".format(np.min(weights), np.argmin(weights), np.max(weights), np.argmax(weights)))
    return weights


def abc_pmc(key, model, n_particles, epsilon_target, alpha, y_obs, epsilon_1 = np.inf, N_sim_max = np.inf, update_weights_distance = True, verbose = 1, stopping_accept_rate = .015):
    K = model.K
    time_0 = time.time()
    key, key_thetas, key_zs = random.split(key, 3)
    new_thetas = model.prior_generator(key_thetas, n_particles)
    new_zs = model.data_generator(key_zs, new_thetas)
    if update_weights_distance: model.update_weights_distance(new_zs)
    distance_values = model.distance(new_zs, y_obs)
    if verbose >=1: print("DISTANCE VALUES: MIN = {:.3}, MAX = {:.3} MEAN = {:.3}".format(np.min(distance_values), np.max(distance_values), np.mean(distance_values)))  
    weights = np.ones(n_particles) / n_particles
    epsilon = np.inf
    ess_val = ess(weights)
    time_0 = time.time()- time_0
    Thetas, Weights, Ess, Epsilon, Dist, Nsim = [new_thetas], [weights], [ess_val], [epsilon], [distance_values], [n_particles*K]
    Zs, Dist, Acc_rate, Time, = [new_zs], [distance_values], [1], [time_0]
    Unique_p, Unique_c = [1.], [1.]
    t = 1
    accept_rate = 1
    
    if verbose >=1:print("Iteration 0: Epsilon = {:.3} ESS = {:0.0f} ({:.3%})".format(epsilon, ess_val, ess_val/n_particles))
    
    while epsilon > epsilon_target:
        if t == 1 and epsilon_1 != np.inf: epsilon = epsilon_1
        else: epsilon = max(np.quantile(distance_values, alpha), epsilon_target)
        
        print("Iteration {}: Epsilon = {:.3}".format(t, epsilon))
        
        # Epsilon = np.append(Epsilon, epsilon)
        zs = new_zs.copy()
        thetas = new_thetas.copy()
        
        if K > 1: 
            std_loc = np.sqrt(2*np.diag(np.cov(thetas.loc.squeeze().T, aweights=weights)))[None, :,None]
        else: 
            std_loc = np.sqrt(2*np.cov(thetas.loc.squeeze()[None, :], aweights=weights))
        std_glob = np.sqrt(2*np.cov(thetas.glob.squeeze(), aweights=weights))

        print("Std loc: min = {:.3}, max = {:.3}\nStd glob = {:.3}".format(np.min(std_loc), np.max(std_loc), std_glob))
        print("Simulating...")
        
        new_thetas = Theta()
        new_zs = np.empty((0, new_zs.shape[1], new_zs.shape[2]))
        distance_values = np.empty(0)
        n_accept=0
        n_sim = 0
        accept_ratee = 1. 
        while n_accept < n_particles:
            time_it = time.time()
            key, key_move = random.split(key)
            
            proposed_thetas, proposed_distances, proposed_zs = move_pmc(key = key_move, model = model, thetas = thetas,  weights = weights, y_obs = y_obs, size = int((n_particles-n_accept)/accept_ratee), std_loc = std_loc, std_glob = std_glob)
            
            accept = np.where(proposed_distances < epsilon)[0]
            new_thetas.append(proposed_thetas[accept])
            distance_values = np.append(distance_values, proposed_distances[accept])

            new_zs = np.append(new_zs, proposed_zs[accept], axis = 0)
            accept_ratee = 1. 
            n_sim += int((n_particles-n_accept)/accept_ratee)
            accept_rate = max(len(accept)/int((n_particles-n_accept)/accept_ratee), .001)
            n_accept += len(accept)
        Nsim.append(n_sim*K)
        print("NUMBER OF SIMULATIONS: {}/{}".format(np.sum(Nsim), N_sim_max))
        
        if accept_rate < stopping_accept_rate or np.sum(Nsim)> N_sim_max: 
            print("Accept rate too low, we stop the algorithm")
            epsilon_target = epsilon
        new_thetas = new_thetas[:n_particles]
        new_zs = new_zs[:n_particles]
        distance_values = distance_values[:n_particles]
        
        print("Number of simulations: {:.2E} (Accept rate {:.5%})\nWeight update...".format(n_sim, n_particles/n_sim))
        print("DISTANCE VALUES: MIN = {:.3}, MAX = {:.3} MEAN = {:.3}".format(np.min(distance_values), np.max(distance_values), np.mean(distance_values)))
        weights = update_weights(model,new_thetas, thetas ,weights, std_loc, std_glob)
        weights = np.array(weights)
        ess_val = ess(weights)
        
        print("ESS = {:0.0f} ({:.3%})".format(ess_val, ess_val/n_particles))
        # model.update_weights_distance(new_zs)
        t += 1
        if ess_val < n_particles / 2:
            print(f"Resampling...")
            # index = particles.resampling.systematic(weights)
            key, key_index = random.split(key)
            index = random.choice(key_index, a= np.arange(n_particles), shape = (n_particles,), p = weights, replace = True)
            
            index = np.array(index, dtype = np.int64)
            counts, indexx = np.unique(index, return_counts = True)
            print("Number of unique index: ", counts[0], indexx[0])

            print(f"Number of unique particles: {len(counts)}")
            new_thetas = new_thetas[index]
            print(f"After resampling, new thetas shape: {new_thetas.loc.shape}")
            new_zs = new_zs[index]
            distance_values = distance_values[index]
            weights = np.ones(n_particles) / n_particles
            ess_val = ess(weights)
        unique_p = len(np.unique(new_thetas.loc, axis = 0))/n_particles
        unique_c = len(np.unique(new_thetas.glob, axis = 0))/n_particles
        time_it = time.time() - time_it  
        Weights.append(weights)
        Ess.append(ess_val)
        Epsilon = np.append(Epsilon, epsilon)
        Thetas.append(new_thetas)
        Dist.append(distance_values)
        Zs.append(new_zs)
        Acc_rate.append(accept_rate)
        Time.append(time_it)
        Unique_p.append(unique_p)
        Unique_c.append(unique_c)
        
            
        print("")  
    out = {"Thetas": Thetas, "Zs": Zs, "Weights": Weights, "Ess": Ess, "Eps_values": Epsilon, "Dist": Dist, "N_sim": Nsim, "Time": Time, "unique_part": Unique_p, "unique_comp": Unique_c, "Acc_rate": Acc_rate, "time_final": time.time()-time_0}
    return out
