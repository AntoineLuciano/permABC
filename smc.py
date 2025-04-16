from utils import ess, resampling
from distances import optimal_index_distance
from moves import move_smc, move_smc_gibbs_blocks
import numpy as np
from jax import random
import time


def init_perm_smc(key, model, n_particles, y_obs, verbose = 1, update_weight_distance = True):
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    thetas = model.prior_generator(key_thetas, n_particles, K)
    zs = model.data_generator(key_zs, thetas)
    print(update_weight_distance)
    if update_weight_distance: 
        model.update_weights_distance(zs, verbose)
    distance_values, ys_index, zs_index, n_lsa = optimal_index_distance(zs = zs, y_obs = y_obs, model = model, verbose = verbose, epsilon = np.inf)
    if update_weight_distance: 
        zs_permuted = zs[np.arange(n_particles)[:, np.newaxis], zs_index]
        model.update_weights_distance(zs_permuted, verbose)
    weights = np.ones(n_particles) / n_particles
    ess_val = ess(weights)
    return thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa

def init_smc(key, model, n_particles, y_obs, update_weight_distance = True, verbose = 1):
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    thetas = model.prior_generator(key_thetas, n_particles, K)
    zs = model.data_generator(key_zs, thetas)
    if update_weight_distance: model.update_weights_distance(zs, verbose)
    distance_values = model.distance(zs, y_obs)
    weights = np.ones(n_particles) / n_particles
    ess_val = n_particles
    return thetas, zs, distance_values, weights, ess_val

def update_epsilon(alive_distances, epsilon_target, alpha):
    return max(np.quantile(alive_distances,alpha), epsilon_target)

def update_weights(weights, distance_values, epsilon):
    weights = weights * np.where(distance_values <= epsilon, 1., 0.)
    if np.sum(weights)==0: 
        print("POIDS NULS")
        return weights
    return weights / np.sum(weights)
        


def abc_smc(key, model, n_particles, epsilon_target, y_obs, kernel, alpha_epsilon = .95, Final_iteration = 0, alpha_resample = 0.5, num_blocks_gibbs = 0, update_weights_distance = False, verbose = 1, N_sim_max = np.inf, stopping_accept_rate = .015, both_loc_glob = False):
    
    time_0 = time.time()
    if update_weights_distance:
        model.reset_weights_distance()
    K = model.K
    if y_obs.ndim == 1: y_obs = y_obs.reshape(1,-1)
        
    thetas, zs, distance_values, weights, ess_val = init_smc(key, model, n_particles, y_obs, verbose = verbose, update_weight_distance= update_weights_distance)
    alive = np.where(weights > 0.0)[0]
    
    Acc_rate, Thetas, Zs, Weights, Ess, Epsilon, Dist, Nsim, Unique_p, Unique_c, Time = [1.], [thetas], [zs], [weights], [ess_val], [np.inf], [distance_values], [n_particles*model.K], [1], [1], [time.time()-time_0]
    
    epsilon = np.inf
    
    if verbose>0:print(f"Iteration 0: Epsilon = {epsilon}, ESS = {ess_val:0.0f} Acc. rate = 100% Numb. unique particles = {len(np.unique(thetas.reshape_2d(),axis=0)):0.0f}")
    t = 1
        
    while epsilon > epsilon_target or Final_iteration >=0:
        time_it = time.time()
        epsilon = update_epsilon(distance_values[alive], epsilon_target, alpha_epsilon)
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        ess_val = np.round(ess(weights))
        
        if ess_val < n_particles * alpha_resample or (epsilon == epsilon_target and ess_val < n_particles):
            key, key_resample = random.split(key)
            thetas, zs, distance_values = resampling(key_resample, weights, [thetas, zs, distance_values])
            weights, ess_val = np.ones(n_particles) / n_particles, n_particles
            alive = np.where(weights > 0.0)[0]
            if verbose>0:print(f"Resampling... {len(np.unique(distance_values[alive]))} unique particles left")
        
        key, key_move = random.split(key)
        if num_blocks_gibbs == 0:
            thetas[alive], zs[alive], distance_values[alive], _, _, _, acc_rate, n_sim = move_smc(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, verbose = verbose, perm = False)
        else:            
            thetas[alive], zs[alive], distance_values[alive], _, _, _, acc_rate, n_sim = move_smc_gibbs_blocks(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, H = num_blocks_gibbs, verbose = verbose, both_loc_glob= both_loc_glob, perm = False)
            
        if verbose>0:
            print(f"Iteration {t}: Espilon = {epsilon:0.4f}, ESS = {ess_val:0.0f} Acc. rate = {acc_rate:.2%} Uniqueness rate particules = {len(np.unique(thetas.reshape_2d(),axis=0))/n_particles:.1%} Uniqueness rate components = {len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape):.1%} Global parameters uniqueness rate = {len(np.unique(thetas.glob))/n_particles:.1%}")
            
        if update_weights_distance: 
            if verbose >1: print("f) Update weights distance:", end = " ")
            model.update_weights_distance(zs, verbose)
            distance_values[alive] = model.distance(zs[alive], y_obs)
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive]> epsilon)
            if verbose>1:print("After update weights update: {} particles alive ({} particles killed)".format(len(alive), n_killed))
            
        Acc_rate.append(acc_rate)
        Thetas.append(thetas)
        Zs.append(zs)
        Weights.append(weights)
        Ess.append(ess_val)
        Epsilon.append(epsilon)
        Dist.append(distance_values)
        Nsim.append(n_sim)
        Unique_p.append(len(np.unique(thetas.reshape_2d(),axis=0))/n_particles)
        Unique_c.append(len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape))
        Time.append(time.time()-time_it)
        t += 1
        if verbose > 1 and N_sim_max<np.inf:print("NUMBER OF SIMULATIONS {}/{}".format(np.sum(Nsim), N_sim_max))
        if epsilon == epsilon_target:
            Final_iteration -= 1
            
        if acc_rate < stopping_accept_rate or epsilon == epsilon_target or np.sum(Nsim) >= N_sim_max:
            epsilon_target = epsilon
            
        if acc_rate < stopping_accept_rate and verbose>0:
            print("Acceptance rate is too low, stopping the algorithm for epsilon = ", epsilon)
        if verbose>0: print()
      
    out = {"Thetas": Thetas, "Zs": Zs, "Weights": Weights, "Ess": Ess, "Eps_values": Epsilon, "Dist": Dist, "N_sim": Nsim, "Time": Time, "Acc_rate": Acc_rate,  "unique_part": Unique_p, "unique_comp": Unique_c, "time_final": time.time()-time_0}
    return out
       




def perm_abc_smc(key, model, n_particles, epsilon_target, y_obs, kernel, alpha_epsilon = .95, Final_iteration = 0, alpha_resample = 0.5, num_blocks_gibbs = 0, update_weights_distance = False, verbose = 1, N_sim_max = np.inf, stopping_accept_rate = .015, both_loc_glob = False):
    time_0 = time.time()
    if update_weights_distance:
        model.reset_weights_distance()
    K = model.K
    if y_obs.ndim == 1: y_obs = y_obs.reshape(1,-1)
        
    thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa = init_perm_smc(key, model, n_particles, y_obs, verbose = verbose, update_weight_distance= update_weights_distance)
    alive = np.where(weights > 0.0)[0]
    
    Acc_rate, Thetas, Zs, Weights, Ys_index, Zs_index, Ess, Epsilon, Dist, Nsim, Nlsa, Unique_p, Unique_c, Time = [1.], [thetas], [zs], [weights], [ys_index], [zs_index], [ess_val], [epsilon_target], [distance_values], [n_particles*model.K], [n_lsa], [1], [1], [time.time()-time_0]
    
    epsilon = np.inf
    
    if verbose>0:print(f"Iteration 0: Epsilon = {epsilon}, ESS = {ess_val:0.0f} Acc. rate = 100% Numb. unique particles = {len(np.unique(thetas.reshape_2d(),axis=0)):0.0f}")
    t = 1

    acc_rate = 1.
    
    apply_permutation = False
    
    while epsilon > epsilon_target or Final_iteration >=0:
        time_it = time.time()
        epsilon = update_epsilon(distance_values[alive], epsilon_target, alpha_epsilon)
        if verbose > 1: print("a) Update Epsilon: new epsilon = {:.3}".format(epsilon))
        old_ess = ess_val
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        ess_val = (ess(weights))
        if verbose > 1: 
            print("b) Update weights: Old ESS = {:0.0f} New ESS = {:0.0f} ({:.2%} of particles killed)".format(old_ess, ess_val, (old_ess-ess_val)/old_ess))
            print("c) Resampling:", end = " ")
        
        if ess_val < n_particles * alpha_resample or (epsilon == epsilon_target and ess_val < n_particles):
            key, key_resample = random.split(key)
            thetas, zs, distance_values, ys_index, zs_index = resampling(key_resample, weights, [thetas, zs, distance_values, ys_index, zs_index])
            weights, ess_val = np.ones(n_particles) / n_particles, n_particles
            alive = np.where(weights > 0.0)[0]
            if verbose>0:print(f"Resampling... {len(np.unique(distance_values[alive]))} unique particles left")
        elif verbose >1:
            print("No resampling")
        key, key_move = random.split(key)
        if verbose >1: print("d) Move particles:", end = " ")

        if num_blocks_gibbs == 0:
            thetas[alive], zs[alive], distance_values[alive], ys_index[alive], zs_index[alive], n_lsa, acc_rate, n_sim = move_smc(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], ys_index = ys_index[alive], zs_index = zs_index[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, verbose = verbose, perm = True)
        else:
            thetas[alive], zs[alive], distance_values[alive], ys_index[alive], zs_index[alive], n_lsa, acc_rate, n_sim = move_smc_gibbs_blocks(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], ys_index = ys_index[alive], zs_index = zs_index[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, H = num_blocks_gibbs, verbose = verbose, both_loc_glob= both_loc_glob, perm = True)
            
        if update_weights_distance: 
            if verbose >1: print("e) Update weights distance:", end = " ")
            zs_permuted = zs[alive[:, None], zs_index[alive]]
            model.update_weights_distance(zs_permuted, verbose = verbose)
            distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(zs = zs[alive], y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, zs_index = zs_index[alive], ys_index = ys_index[alive])
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive]> epsilon)
            if verbose>1:print("After update weights update: {} particles alive ({} particles killed)".format(len(alive), n_killed))
            
        Acc_rate.append(acc_rate)
        Thetas.append(thetas)
        Zs.append(zs)
        Weights.append(weights)
        Ys_index.append(ys_index)
        Zs_index.append(zs_index)
        Ess.append(ess_val)
        Epsilon.append(epsilon)
        Dist.append(distance_values)
        Nsim.append(n_sim)
        Nlsa.append(n_lsa)
        Unique_p.append(len(np.unique(thetas.reshape_2d(),axis=0))/n_particles)
        Unique_c.append(len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape))
        Time.append(time.time()-time_it)
        t += 1
        if epsilon == epsilon_target:
            Final_iteration -= 1
            
        if verbose>0:
            print(f"Iteration {t}: Espilon = {epsilon:0.4f}, ESS = {ess_val:0.0f} Acc. rate = {acc_rate:.2%} Uniqueness rate particules = {len(np.unique(thetas.reshape_2d(),axis=0))/n_particles:.1%} Uniqueness rate components = {len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape):.1%} Global parameters uniqueness rate = {len(np.unique(thetas.glob))/n_particles:.1%}")
        if verbose > 1 and N_sim_max<np.inf:print("NUMBER OF SIMULATIONS {}/{}".format(np.sum(Nsim), N_sim_max))
        if acc_rate < stopping_accept_rate or epsilon == epsilon_target or np.sum(Nsim) >= N_sim_max:
            epsilon_target = epsilon
            apply_permutation = True
            
        if acc_rate < stopping_accept_rate and verbose>0:
            print("Acceptance rate is too low, stopping the algorithm for epsilon = ", epsilon)
            
        if apply_permutation and Final_iteration <=0 : 
            print("Apply permutation")
            thetas, zs = thetas.apply_permutation(zs_index),np.concatenate([zs[np.arange(n_particles)[:, np.newaxis], zs_index], zs[:,K:]], axis = 1)
            zs_index = np.repeat([np.arange(model.K)], n_particles, axis = 0) 
        if verbose>0: print()
    out = {"Thetas": Thetas, "Zs": Zs, "Weights": Weights, "Ys_index": Ys_index, "Zs_index": Zs_index, "Ess": Ess, "Eps_values": Epsilon, "Dist": Dist, "N_sim": Nsim, "Time": Time, "Acc_rate": Acc_rate, "N_lsa": Nlsa, "unique_part": Unique_p, "unique_comp": Unique_c, "time_final": time.time()-time_0}
    return out
       
 