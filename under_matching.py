from utils import ess, resampling
from distances import optimal_index_distance
from moves import move_smc, move_smc_gibbs_blocks
from smc import update_weights
import numpy as np
from jax import random
import time
from scipy.special import gammaln   



def init_perm_under_matching(key, model, n_particles, y_obs, epsilon, L_0, alpha_epsilon, verbose = 1, update_weight_distance = True):
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    thetas = model.prior_generator(key_thetas, n_particles, K)
    zs = model.data_generator(key_zs, thetas)
    if verbose > 1:print("a) Simulation of the first particles:")
    if update_weight_distance: 
        model.update_weights_distance(zs, verbose)
    if verbose > 1:print("b) Computing the first distances with L = {}:".format(L_0), end = " ")
    distance_values, ys_index, zs_index, n_lsa = optimal_index_distance(zs = zs, y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, L = L_0)
    if verbose > 1:print("min = {:.2} max = {:.2} mean = {:.2}".format(np.min(distance_values), np.max(distance_values), np.mean(distance_values)))
    if epsilon == np.inf: 
        epsilon = np.quantile(distance_values, alpha_epsilon)
        print("Epsilon =", epsilon)

    weights = np.where(distance_values <= epsilon, 1., 0.)
    if np.sum(weights) == 0.0:  
        print("All weights are null, stopping the algorithm")
        return None, None, None, None, None, None, None, None, None
    weights /= np.sum(weights)
    ess_val = ess(weights)
    if verbose > 1: print("d) Setting the first weights: ESS =", round(ess_val))
    alive = np.where(weights > 0.0)[0]
    
    if update_weight_distance: 
        if verbose >1: print("h) Update weights distance:", end = " ")
        zs_match = [[] for _ in range(K)]
        for i in alive:
            i = int(i)
            for k in range(L_0):
                zs_match[int(ys_index[i,k])].append(zs[i,int(zs_index[i,k])])
        len_K_match = [len(z) for z in zs_match]
        if np.min(len_K_match)>25.:
            model.update_weights_distance((zs_match), verbose)
        distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(zs = zs[alive], y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, L = L_0, zs_index = zs_index[alive], ys_index = ys_index[alive])
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        n_killed = np.sum(distance_values[alive]> epsilon)
        if verbose>1:print("After update weights update: {} particles alive ({} particles killed)".format(len(alive), n_killed))
    return thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon


    
       
def perm_abc_smc_um(key, model, n_particles, y_obs, kernel, L_0, epsilon = np.inf, alpha_epsilon = .95, alpha_L = .95,  Final_iteration = 0, alpha_resample = 0.5, num_blocks_gibbs = 0, update_weights_distance = True, verbose = 1):
    time_0 = time.time()
    if update_weights_distance:
        model.reset_weights_distance()
    K = model.K
    if y_obs.ndim == 1: y_obs = y_obs.reshape(1,-1)
        
    thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon = init_perm_under_matching(key, model, n_particles, y_obs, epsilon, L_0, verbose = verbose, alpha_epsilon=alpha_epsilon, update_weight_distance = update_weights_distance)
    if thetas is None: return None
    alive = np.where(weights > 0.0)[0]
    
    Acc_rate, Thetas, Zs, Weights, Ys_index, Zs_index, Ess, Epsilon, L_values, Dist, Nsim, Nlsa, Unique_p, Unique_c, Time = [1.], [thetas], [zs], [weights], [ys_index], [zs_index], [ess_val], [epsilon], [L_0], [distance_values], [n_particles*K], [n_lsa], [1], [1], [time.time()-time_0]
    Prop_killed = [(n_particles - ess_val)/n_particles]
    L = L_0
    
    if verbose>0:print(f"Iteration 0: L = {L_0} Epsilon = {epsilon}, ESS = {ess_val:0.0f} Acc. rate = 100% Numb. unique particles = {len(np.unique(thetas.reshape_2d(),axis=0)):0.0f}\n")
    t = 1
    n_accept = n_particles
    
    apply_permutation = False
    failure = False
    while L < K or Final_iteration >=0:
        old_L = L
        time_it = time.time()
        L = min(max(int(K-(alpha_L*(K-old_L))), old_L+1),K)
        if verbose>1: print("a) Update L: new L = {} old L = {}".format(L, old_L))
        
            
        alive = np.where(weights > 0.0)[0]
        if verbose >1: print("b) Compute optimal distances: {} unique particles".format(len(np.unique(distance_values[alive]))), end = " ")
        ys_index, zs_index = -np.ones((n_particles, L)),-np.ones((n_particles, L))
        
        distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(zs = zs[alive], y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, L = L)
        old_ess = ess(weights)
        weights = update_weights(weights, distance_values, epsilon)
        
        if np.sum(weights) == 0.0:
            failure = True
            print("All weights are null, stopping the algorithm")
            break
        
        alive = np.where(weights > 0.0)[0]
        ess_val = np.round(ess(weights))
        
        if verbose >1: print("c) Update weights: Old ESS = {:} New ESS = {:} ({:.2%} of the particles killed)".format(round(old_ess), round(ess_val), (old_ess-ess_val)/old_ess))
        if verbose >1: print("f) Resampling:", end = " ")
        prop_killed = (old_ess - ess_val) / old_ess
        if ess_val < n_particles * alpha_resample or (L==K and ess_val < n_particles) :
            key, key_resample = random.split(key)
            thetas, zs, distance_values, ys_index, zs_index = resampling(key_resample, weights, [thetas, zs, distance_values, ys_index, zs_index], n_particles)
            weights, ess_val = np.ones(n_particles) / n_particles, n_particles
            alive = np.where(weights > 0.0)[0]
            if verbose>0:print(f"Resampling... {len(np.unique(distance_values[alive]))} unique particles left")
        else: 
            if verbose > 1: print("No resampling")
        if verbose > 1: print("d) Move particles:")
        key, key_move = random.split(key)
        if num_blocks_gibbs == 0:
            thetas[alive], zs[alive], distance_values[alive], ys_index[alive], zs_index[alive], n_lsa, n_accept, n_sim = move_smc(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], ys_index = ys_index[alive], zs_index = zs_index[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, verbose = verbose, perm = True, L = L) 
        else:
            thetas[alive], zs[alive], distance_values[alive], ys_index[alive], zs_index[alive], n_accept, n_sim, n_lsa = move_smc_gibbs(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], ys_index = ys_index[alive], zs_index = zs_index[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, num_blocks_gibbs = num_blocks_gibbs, verbose = verbose, perm = True, L = L)   
        
        if update_weights_distance: 
            if verbose >1: print("h) Update weights distance:", end = " ")
            zs_match = [[] for _ in range(K)]
            for i in alive:
                i = int(i)
                for k in range(L):
                    zs_match[int(ys_index[i,k])].append(zs[i,int(zs_index[i,k])])
            len_K_match = [len(z) for z in zs_match]
            if np.min(len_K_match)>25.:
                model.update_weights_distance((zs_match), verbose)
            distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(zs = zs[alive], y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, L = L, zs_index = zs_index[alive], ys_index = ys_index[alive])
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive]> epsilon)
            if verbose>1:print("After update weights update: {} particles alive ({} particles killed)".format(len(alive), n_killed))
            
            
        Acc_rate.append(n_accept/len(alive))
        Thetas.append(thetas)
        Zs.append(zs)
        Weights.append(weights)
        Ys_index.append(ys_index)
        Zs_index.append(zs_index)
        Ess.append(ess_val)
        Epsilon.append(epsilon)
        L_values.append(L)
        Dist.append(distance_values)
        Nsim.append(n_sim)
        Nlsa.append(n_lsa)
        Unique_p.append(len(np.unique(thetas.reshape_2d(),axis=0))/n_particles)
        Unique_c.append(len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape))
        Time.append(time.time()-time_it)
        Prop_killed.append(prop_killed)
        t += 1
        if L == K:
            Final_iteration -= 1
        
        if verbose>0:
            print(f"Iteration {t}: L = {L} Espilon = {epsilon:0.4f}, ESS = {ess_val:0.0f} Acc. rate = {n_accept/len(alive):.2%} Uniqueness rate particules = {len(np.unique(thetas.reshape_2d(),axis=0))/n_particles:.1%} Uniqueness rate components = {len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape):.1%} Global parameters uniqueness rate = {len(np.unique(thetas.glob))/n_particles:.1%}")
            
        if n_accept/len(alive) < .0 or L == K:
            apply_permutation = True
            
        if n_accept/len(alive) <= .0 and verbose>0:
            print("Acceptance rate is too low, stopping the algorithm for epsilon = ", epsilon)
            
        if apply_permutation and Final_iteration <=0 : 
            zs_index = np.array(zs_index, dtype= np.int32)
            thetas, zs = thetas.apply_permutation(zs_index),np.concatenate([zs[np.arange(n_particles)[:, np.newaxis], np.array(zs_index, dtype = np.int32)], zs[:,K:]], axis = 1)
            zs_index = np.repeat([np.arange(model.K)], n_particles, axis = 0) 
        if verbose>0: print()
    if failure: 
        return None
    out = {"Thetas": Thetas, "Zs": Zs, "Weights": Weights, "Ys_index": Ys_index, "Zs_index": Zs_index, "Ess": Ess, "Eps_values": Epsilon, "L_values": L_values, "Dist": Dist, "N_sim": Nsim, "Time": Time, "Acc_rate": Acc_rate, "N_lsa": Nlsa, "unique_part": Unique_p, "unique_comp": Unique_c, "time_final": time.time()-time_0, "Prop_killed": Prop_killed}
    return out
    
