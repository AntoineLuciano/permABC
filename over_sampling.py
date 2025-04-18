from utils_functions import ess, resampling
from distances import optimal_index_distance
from moves import move_smc, move_smc_gibbs_blocks
from smc import update_weights
import numpy as np
from jax import random
import time
from scipy.special import gammaln   



def init_perm_over_sampling(key, model, n_particles, y_obs, epsilon, M_0, alpha_epsilon, verbose = 1, update_weight_distance = True):
    K = model.K
    key, key_thetas, key_zs = random.split(key, 3)
    thetas = model.prior_generator(key_thetas, n_particles, M_0)
    zs = model.data_generator(key_zs, thetas)
    if verbose > 1:print("a) Simulation of the first particles:")
    if update_weight_distance: 
        model.update_weights_distance(np.concatenate([zs[:,:K], zs[:,M_0:]], axis = 1), verbose)
    if verbose > 1:print("b) Computing the first distances:", end = " ")
    distance_values, ys_index, zs_index, n_lsa = optimal_index_distance(zs = zs, y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, M = M_0)
    if verbose > 1:print("min = {:.2} max = {:.2} mean = {:.2}".format(np.min(distance_values), np.max(distance_values), np.mean(distance_values)))
    if epsilon == np.inf: 
        epsilon = np.quantile(distance_values, alpha_epsilon)
        print("Epsilon =", epsilon)

    if update_weight_distance: 
        zs_permuted = np.concatenate([zs[np.arange(n_particles)[:, np.newaxis], zs_index][:,:K], zs[:,M_0:]], axis = 1)
        model.update_weights_distance(zs_permuted, verbose)
    
    weights = np.where(distance_values <= epsilon, 1., 0.)
    weights /= np.sum(weights)
    ess_val = ess(weights)
    if verbose > 1: print("d) Setting the first weights: ESS =", round(ess_val))
    return thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon


def duplicate_particles(key, model, weights, thetas, zs, ys_index, zs_index, distance_values, old_M, new_M, alpha_M, verbose, n_duplicate = 0):
    
    alive = np.where(weights > 0.0)[0]
    K = model.K
    n_particles = len(thetas[alive])
    if n_duplicate == 0:
        proba_survive  = np.exp(gammaln(old_M-K+1)+gammaln(new_M+1)-gammaln(new_M-K+1)-gammaln(old_M+1))
        n_duplicate = int(alpha_M/(proba_survive))
    if n_duplicate ==1:
        if verbose >1: print("No duplication because n_duplicate = 1")
        return thetas, zs, distance_values, ys_index, zs_index, weights
    
    if verbose >1:print(f"Duplicating the {n_particles} alive particles in {n_duplicate} copies...")

    permutation_duplicates = random.permutation(key, np.repeat([np.arange(old_M)], n_duplicate*n_particles, axis=0), axis = 1, independent = True)
    
    new_thetas = thetas[alive].copy()
    new_thetas = new_thetas.duplicate(n_duplicate, permutation_duplicates)
    
    new_zs = np.repeat(zs[alive], n_duplicate, axis = 0)
    new_zs = np.concatenate([new_zs[np.arange(n_particles*n_duplicate)[:, np.newaxis], permutation_duplicates], new_zs[:, old_M:]], axis = 1)
    
    new_ys_index = np.repeat(ys_index[alive], n_duplicate, axis = 0)
    new_zs_index = np.repeat(zs_index[alive], n_duplicate, axis = 0)

    new_distance_values = np.repeat(distance_values[alive], n_duplicate, axis = 0)
    
    new_weights = np.repeat(weights[alive], n_duplicate, axis = 0)
    
    thetas.append(new_thetas)
    zs = np.append(zs, new_zs, axis = 0)
    distance_values = np.append(distance_values, new_distance_values, axis = 0)
    ys_index = np.append(ys_index, new_ys_index, axis = 0)
    zs_index = np.append(zs_index, new_zs_index, axis = 0)
    
    weights = np.append(weights, new_weights, axis = 0)
    weights = np.where(weights > 0.0, 1., 0.)
    weights /= np.sum(weights)
    if verbose >1 : print(f'Now particles of shape {thetas.loc.shape} and {zs.shape} with {len(np.unique(thetas.reshape_2d(),axis=0)):0.0f} unique particles')    
    return thetas, zs, distance_values, ys_index, zs_index, weights

    

def truncate_particles(thetas, zs, new_M, old_M):
    new_thetas = thetas.copy()
    new_thetas.truncating(new_M, old_M)
    new_zs = np.concatenate([zs[:,:new_M], zs[:, old_M:]], axis = 1)
    return new_thetas, new_zs
    
       
def perm_abc_smc_os(key, model, n_particles, y_obs, kernel, M_0, epsilon = np.inf, alpha_epsilon = .95, alpha_M = .95,  Final_iteration = 0, alpha_resample = 0.5, num_blocks_gibbs = 0, update_weights_distance = True, verbose = 1, duplicate = False, n_duplicate = 0):
    time_0 = time.time()
    K = model.K
    if y_obs.ndim == 1: y_obs = y_obs.reshape(1,-1)
    if update_weights_distance:
        model.reset_weights_distance()
        
    thetas, zs, distance_values, ys_index, zs_index, weights, ess_val, n_lsa, epsilon = init_perm_over_sampling(key, model, n_particles, y_obs, epsilon, M_0, verbose = verbose, alpha_epsilon=alpha_epsilon, update_weight_distance = update_weights_distance)
    alive = np.where(weights > 0.0)[0]
    
    Acc_rate, Thetas, Zs, Weights, Ys_index, Zs_index, Ess, Epsilon, M_values, Dist, Nsim, Nlsa, Unique_p, Unique_c, Time = [1.], [thetas], [zs], [weights], [ys_index], [zs_index], [ess_val], [epsilon], [M_0], [distance_values], [n_particles*M_0], [n_lsa], [1], [1], [time.time()-time_0]
    
    Prop_killed = [ess_val/n_particles]
    M = M_0
    
    if verbose>0:print(f"Iteration 0: M = {M_0} Epsilon = {epsilon}, ESS = {ess_val:0.0f} Acc. rate = 100% Numb. unique particles = {len(np.unique(thetas.reshape_2d(),axis=0)):0.0f}\n")
    t = 1
    n_accept = n_particles
    
    apply_permutation = False
    
    while M > K or Final_iteration >=0:
        old_M = M
        time_it = time.time()
        M = max(min(int(K+(alpha_M*(old_M-K))), old_M-1),K)
        if verbose>1: print("a) Update M: new M = {} old M = {}".format(M, old_M))
        if duplicate:
            if verbose>1: print("b) Duplicate particles")
            thetas, zs, distance_values, ys_index, zs_index, weights = duplicate_particles(key = key, model = model, weights = weights, thetas = thetas, zs = zs, ys_index = ys_index, zs_index = zs_index, distance_values = distance_values, old_M = old_M, new_M = M, alpha_M = alpha_M, verbose = verbose, n_duplicate = n_duplicate)
      
        alive = np.where(weights > 0.0)[0]

        if old_M >K:
            old_thetas, old_zs = thetas.copy(), zs.copy()
            
            thetas, zs = truncate_particles(thetas, zs, M, old_M)

            if verbose >1: print("c) Truncate particles: before {},{} after {},{}".format(old_thetas.loc.shape, old_zs.shape, thetas.loc.shape, zs.shape))
        if verbose >1: print("d) Compute optimal distances: {} unique alive particles".format(len(np.unique(distance_values[alive]))), end = " ")
        
        distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(zs = zs[alive], y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, M = M)
    
        old_ess = ess(weights)
        weights = update_weights(weights, distance_values, epsilon)
        alive = np.where(weights > 0.0)[0]
        ess_val = (ess(weights))
        
        if verbose >1: print("e) Weights update/Killing of the particles: Old ESS = {:} New ESS = {:} ({:.2%} of the particles killed)".format(round(old_ess), round(ess_val), (old_ess-ess_val)/old_ess))
        prop_killed = (old_ess-ess_val)/old_ess
        if verbose >1: print("f) Resampling:", end = " ")
        if ess_val < n_particles * alpha_resample or (M==K and ess_val < n_particles) or (len(zs)>n_particles):
            key, key_resample = random.split(key)
            thetas, zs, distance_values, ys_index, zs_index = resampling(key_resample, weights, [thetas, zs, distance_values, ys_index, zs_index], n_particles)
            weights, ess_val = np.ones(n_particles) / n_particles, n_particles
            alive = np.where(weights > 0.0)[0]
            if verbose>0:print(f"Resampling... {len(np.unique(distance_values[alive]))} unique particles left")
        else: 
            if verbose > 1: print("No resampling")
        if verbose > 1: print("g) Move particles:")
        key, key_move = random.split(key)
        if num_blocks_gibbs == 0:
            thetas[alive], zs[alive], distance_values[alive], ys_index[alive], zs_index[alive], n_lsa, n_accept, n_sim = move_smc(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], ys_index = ys_index[alive], zs_index = zs_index[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, verbose = verbose, perm = True, M = M) 
        else:
            thetas[alive], zs[alive], distance_values[alive], ys_index[alive], zs_index[alive], n_accept, n_sim, n_lsa = move_smc_gibbs(key = key_move, model = model, thetas = thetas[alive], zs = zs[alive], weights = weights[alive], ys_index = ys_index[alive], zs_index = zs_index[alive], epsilon = epsilon, y_obs = y_obs, distance_values = distance_values[alive], kernel = kernel, num_blocks_gibbs = num_blocks_gibbs, verbose = verbose, perm = True, M = M)   
        

        if update_weights_distance: 
            if verbose >1: print("h) Update weights distance:", end = " ")
            zs_permuted = zs[alive[:,None], zs_index[alive]]
            model.update_weights_distance(zs_permuted, verbose)
            distance_values[alive], ys_index[alive], zs_index[alive], n_lsa = optimal_index_distance(zs = zs[alive], y_obs = y_obs, model = model, verbose = verbose, epsilon = epsilon, M = M, zs_index = zs_index[alive], ys_index = ys_index[alive])
            weights = update_weights(weights, distance_values, epsilon)
            alive = np.where(weights > 0.0)[0]
            n_killed = np.sum(distance_values[alive]> epsilon)
            if verbose>1:print("After update weights update: {} particles alive ({} particles killed)".format(len(alive), n_killed))
            
        
        if M == K:
            Final_iteration -= 1
        
        if verbose>0:
            print(f"Iteration {t}: M = {M} Espilon = {epsilon:0.4f}, ESS = {ess_val:0.0f} Acc. rate = {n_accept/len(alive):.2%} Uniqueness rate particules = {len(np.unique(thetas.reshape_2d(),axis=0))/n_particles:.1%} Uniqueness rate components = {len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape):.1%} Global parameters uniqueness rate = {len(np.unique(thetas.glob))/n_particles:.1%}")
            
        if n_accept/len(alive) < .0 or M == K:
            apply_permutation = True
            duplicate = False
            
        if n_accept/len(alive) < .0 and verbose>0:
            print("Acceptance rate is too low, stopping the algorithm for epsilon = ", epsilon)
            
        if apply_permutation and Final_iteration <=0 : 
            thetas, zs = thetas.apply_permutation(zs_index),np.concatenate([zs[np.arange(n_particles)[:, np.newaxis], zs_index], zs[:,K:]], axis = 1)
            zs_index = np.repeat([np.arange(model.K)], n_particles, axis = 0) 
        Acc_rate.append(n_accept/len(alive))
        Thetas.append(thetas)
        Zs.append(zs)
        Weights.append(weights)
        Ys_index.append(ys_index)
        Zs_index.append(zs_index)
        Ess.append(ess_val)
        Epsilon.append(epsilon)
        M_values.append(M)
        Dist.append(distance_values)
        Nsim.append(n_sim)
        Nlsa.append(n_lsa)
        Unique_p.append(len(np.unique(thetas.reshape_2d(),axis=0))/n_particles)
        Unique_c.append(len(np.unique(thetas.reshape_2d()))/np.prod(thetas.reshape_2d().shape))
        Time.append(time.time()-time_it)
        Prop_killed.append(prop_killed)
        t += 1
        if verbose>0: print()

    out = {"Thetas": Thetas, "Zs": Zs, "Weights": Weights, "Ys_index": Ys_index, "Zs_index": Zs_index, "Ess": Ess, "Eps_values": Epsilon, "M_values": M_values, "Dist": Dist, "N_sim": Nsim, "Time": Time, "Acc_rate": Acc_rate, "N_lsa": Nlsa, "unique_part": Unique_p, "unique_comp": Unique_c, "time_final": time.time()-time_0, "Prop_killed": Prop_killed, "Final_iteration": Final_iteration}
    return out
    
