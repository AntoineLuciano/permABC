import jax.numpy as jnp
import numpy as np
from jax import random
from utils_functions import Theta
from distances import optimal_index_distance
from jax import vmap, random

def move_smc(key, model, thetas, zs, weights, epsilon, y_obs, distance_values, kernel, ys_index = None, zs_index = None, verbose=1, perm = False, M = 0, L = 0):
    """
    Standard Random Walk with Metropolis-Hastings acceptance move for ABC-SMC.

    Parameters:
    - key: JAX PRNG key.
    - model: Bayesian model.
    - thetas: Current particles (Theta).
    - zs: Simulated observations.
    - weights: Particle weights.
    - zs_index: Current permutation index.
    - epsilon: ABC acceptance threshold.
    - y_obs: Observed data.
    - distance_values: Current distances.
    - kernel: Proposal kernel.
    - verbose: Verbosity level.

    Returns:
    - Updated thetas, zs, distance_values, zs_index
    """
    n_particles = thetas.loc.shape[0]
    if M == 0: M = model.K
    if L == 0: L = model.K
    
    old_thetas = thetas.copy()

    key, key_kernel, key_data, key_uniform = random.split(key, 4)
    
    # Propose new parameters
    if verbose > 1: print("1. Forward kernel: ")
    forward_kernel = kernel(model=model, thetas=thetas, weights=weights, ys_index=ys_index, zs_index=zs_index, verbose=verbose, M = M, L = L)
    proposed_thetas = forward_kernel.sample(key_kernel)
    proposed_zs = model.data_generator(key_data, proposed_thetas)
    # Compute optimal distances
    if perm: 
        proposed_distances, proposed_ys_index, proposed_zs_index, n_lsa = optimal_index_distance(model = model, zs = proposed_zs, y_obs = y_obs, epsilon = epsilon,  ys_index = ys_index, zs_index = zs_index, verbose = verbose, M = M, L = L)
    else: 
        proposed_distances = np.array(model.distance(proposed_zs, y_obs))
        proposed_ys_index = None
        proposed_zs_index = None
        n_lsa = 0
    # Compute acceptance probability
    
    backward_kernel = kernel(model=model, thetas=proposed_thetas, weights=weights, ys_index= proposed_ys_index, zs_index=proposed_zs_index, verbose=verbose, tau_loc_glob = forward_kernel.get_tau_loc_glob(), M = M, L = L)
    prior_forward = model.prior_logpdf(proposed_thetas)
    prior_backward = model.prior_logpdf(thetas)
    prior_logratio = jnp.minimum(prior_forward - prior_backward, 703)
    # prior_logratio = jnp.minimum(model.prior_logpdf(proposed_thetas) - model.prior_logpdf(thetas), 703)
    kernel_logratio = backward_kernel.logpdf(thetas) - forward_kernel.logpdf(proposed_thetas)
    
    accept_prob = (proposed_distances <= epsilon) * jnp.exp(prior_logratio + kernel_logratio)
    accept_prob = jnp.nan_to_num(jnp.minimum(accept_prob, 1))
    
    # Accept/reject
    uniform_samples = random.uniform(key_uniform, shape=(n_particles,))
    accept = np.array(uniform_samples <= accept_prob)
    thetas[accept] = proposed_thetas[accept]
    zs[accept] = proposed_zs[accept]
    distance_values = np.array(distance_values)
    distance_values[accept] = proposed_distances[accept]
    if perm and zs_index is not None and ys_index is not None:
        zs_index[accept] = proposed_zs_index[accept]
        ys_index[accept] = proposed_ys_index[accept]
        
    n_accept = np.sum(accept)
    
    
    if verbose > 1:
        print("Sample proposal: loc min = {:.2} max {:.2} mean = {:.2} glob min = {:.2} max = {:.2} mean = {:.2}".format(jnp.min(proposed_thetas.loc), jnp.max(proposed_thetas.loc), jnp.mean(proposed_thetas.loc), jnp.min(proposed_thetas.glob), jnp.max(proposed_thetas.glob), jnp.mean(proposed_thetas.glob)))
        
        print("2. MH acceptance: ratio min = {:.3} max = {:.3} mean {:.3}\nPrior ratio: min = {:.3}, max = {:.3}, mean = {:.3}\nKernel ratio: min = {:.3}, max = {:.3}, mean = {:.3}\n3. Results: Acceptance = {:.2%} Rejection ABC = {:.2%} Rejection MH = {:.2%}".format(jnp.min(accept_prob), jnp.max(accept_prob), jnp.mean(accept_prob), jnp.min(prior_logratio), jnp.max(prior_logratio), jnp.mean(prior_logratio), jnp.min(kernel_logratio), jnp.max(kernel_logratio), jnp.mean(kernel_logratio), jnp.mean(accept), jnp.mean(proposed_distances >= epsilon), jnp.mean(jnp.logical_and(proposed_distances<=epsilon, accept_prob < uniform_samples))))
        if verbose >2: 
            print("Prior forward: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(prior_forward), jnp.max(prior_forward), jnp.mean(prior_forward)))
            print("Forward thetas loc: min = {:.3}, max = {:.3}, mean = {:.3} global: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(proposed_thetas.loc), jnp.max(proposed_thetas.loc), jnp.mean(proposed_thetas.loc), jnp.min(proposed_thetas.glob), jnp.max(proposed_thetas.glob), jnp.mean(proposed_thetas.glob)))
            
            print("Prior backward: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(prior_backward), jnp.max(prior_backward), jnp.mean(prior_backward)))
            print("Backward thetas loc: min = {:.3}, max = {:.3}, mean = {:.3} global: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(thetas.loc), jnp.max(thetas.loc), jnp.mean(thetas.loc), jnp.min(thetas.glob), jnp.max(thetas.glob), jnp.mean(thetas.glob)))
    accept_rate = np.sum(accept) / n_particles
    return thetas, zs, distance_values, ys_index, zs_index, n_lsa, accept_rate, n_particles*M




def create_block(key, matched_index, H, K, M=0, L=0):
    """
    Create random blocks of indices for block-wise updates in Gibbs sampling.

    Parameters:
    - index: array of shape (n_particles, K) used for permutation.
    - M: total number of silos (may be > K).
    - H: number of blocks.
    - K: number of observed silos.
    - key: JAX PRNG key.
    - L: number of matched silos (not used here, optional).

    Returns:
    - blocks: array of shape (n_particles, H+1, block_size) with index blocks.
              blocks[:, 0, :] corresponds to unmatched silos if M > K.
              blocks[:, 1:, :] are the H blocks of matched silos.
    """
    if L == 0: L = K
    if M == 0: M = K
    n_particles = matched_index.shape[0]
    matched_perms = random.permutation(key, matched_index, axis = 1, independent= True)  # shape (n_particles, K)
    bins = jnp.linspace(0, L, H + 1).astype(jnp.int32)
    matched_blocks = jnp.split(matched_perms, bins[1:-1], axis=1)  # shape (n_particles, H+1, block_size)
    if L < M: 
        # print("Unmatched silos: ", M - L)
        indexes = jnp.repeat(jnp.arange(M)[None,:], n_particles, axis = 0)
        mask = jnp.isin(indexes, matched_index, invert=True)
        unmatched_index = jnp.where(mask, indexes, -1)  # Replace non-difference elements with -1
        unmatched_index = unmatched_index[unmatched_index != -1].reshape(n_particles, -1)
        # print("Unmatched index: ", unmatched_index.shape)
        matched_blocks.append(unmatched_index)  # shape (n_particles, H+1, block_size)
    return matched_blocks  # shape (n_particles, H+1, block_size)

    
def move_smc_gibbs_blocks(key, model, thetas, zs, weights, epsilon, y_obs, distance_values, kernel, ys_index=None, zs_index=None, verbose=1, perm=True, M=0, L=0, H=0, both_loc_glob=True):
    """
    Gibbs block-wise Metropolis-Hastings move for ABC-SMC.

    Parameters are consistent with move_smc.
    
    Returns:
    - Updated thetas, zs, distance_values, ys_index, zs_index, n_lsa, n_accept, n_sim
    """
    n_particles = thetas.loc.shape[0]
    if M == 0: M = model.K
    if L == 0: L = model.K
    K = model.K
    if H == 0: H = 1
    
    key, key_kernel, key_data, key_blocks, key_uniform = random.split(key, 5)
    n_lsa, n_accept = 0, 0
    old_thetas = thetas.copy()
    old_zs = zs.copy()
    # Propose global + local updates via Gibbs blocks
    forward_kernel = kernel(model=model, thetas=thetas, weights=weights, ys_index=ys_index, zs_index=zs_index, verbose=verbose, M=M, L=L)
    proposed_thetas = forward_kernel.sample(key_kernel)
    # proposed_zs = model.data_generator(key_data, proposed_thetas)

    # Identify which particles are globally vs locally updated
    if both_loc_glob:
        glob_update = loc_update = jnp.arange(n_particles)
        block_choice_glob = block_choice_loc = np.full((n_particles,), True)
    else:
        block_choice_glob = random.uniform(key_blocks, shape=(n_particles,)) < 0.5
        block_choice_loc = ~block_choice_glob
        
        glob_update = jnp.where(block_choice_glob)[0]
        loc_update = jnp.where(block_choice_loc)[0]

    # Global update
    if verbose > 1: print("Global update")
    forward_kernel_glob = kernel(model=model, thetas=thetas[glob_update], weights=weights[glob_update], ys_index=ys_index[glob_update], zs_index=zs_index[glob_update], verbose=verbose, M=M, L=L, tau_loc_glob=forward_kernel.get_tau_loc_glob())
    proposed_thetas_glob = thetas.copy()
    proposed_thetas_glob.glob[glob_update] = proposed_thetas.glob[glob_update]
    
    proposed_zs_glob = zs.copy()
    # print("Sampling {} zs...".format(np.sum(block_choice_glob)))
    proposed_zs_glob[glob_update] = model.data_generator(key_data, proposed_thetas_glob[glob_update])
    # print("Computing {} distances...".format(np.sum(block_choice_glob)))
    if perm:
        proposed_distances_glob, proposed_ys_index_glob, proposed_zs_index_glob, n_lsa_glob = optimal_index_distance(
            model=model,
            zs=proposed_zs_glob[glob_update],
            y_obs=y_obs,
            epsilon=epsilon,
            ys_index=None if ys_index is None else ys_index[glob_update],
            zs_index=None if zs_index is None else zs_index[glob_update],
            verbose=verbose,
            M=M,
            L=L
        )
        n_lsa += n_lsa_glob
    else:
        proposed_distances_glob = np.array(model.distance(proposed_zs_glob[glob_update], y_obs))
        proposed_ys_index_glob = ys_index[glob_update] if ys_index is not None else None
        proposed_zs_index_glob = zs_index[glob_update] if zs_index is not None else None
    backward_kernel_glob = kernel(model=model, thetas=proposed_thetas_glob[glob_update], weights=weights, ys_index=proposed_ys_index_glob, zs_index=proposed_zs_index_glob, verbose=verbose, M=M, L=L, tau_loc_glob=forward_kernel.get_tau_loc_glob())
    
    prior_forward = model.prior_logpdf(proposed_thetas_glob[glob_update])
    prior_backward = model.prior_logpdf(thetas[glob_update])
    prior_logratio = jnp.minimum(prior_forward - prior_backward, 703)
    kernel_logratio = backward_kernel_glob.logpdf(thetas[glob_update]) - forward_kernel_glob.logpdf(proposed_thetas_glob[glob_update])
    # print("Kernel ratio: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(kernel_logratio), jnp.max(kernel_logratio), jnp.mean(kernel_logratio)))
    # print("Prior ratio: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(prior_logratio), jnp.max(prior_logratio), jnp.mean(prior_logratio)))
    accept_prob = (proposed_distances_glob <= epsilon) * jnp.exp(prior_logratio + kernel_logratio)
    accept_prob = jnp.nan_to_num(jnp.minimum(accept_prob, 1))

    uniform_samples = random.uniform(key_uniform, shape=(np.sum(block_choice_glob),))
                                
    accept = np.array(uniform_samples <= accept_prob)
    # accept = accept * np.array(block_choice_glob)
    # old_thetas_glob_accept = thetas[glob_update[accept]].glob.copy()
    # old_zs_glob_accept = zs[glob_update[accept]].copy()
    
    accept_thetas_glob = np.copy(proposed_thetas_glob[glob_update[accept]].glob)
    accept_zs_glob = np.copy(proposed_zs_glob[glob_update[accept]])
    
    thetas.glob[glob_update[accept]] = accept_thetas_glob
    zs[glob_update[accept]] = accept_zs_glob

    
    # print("Accepting {} global updates...".format(np.sum(accept)))
    # print("Global thetas accept = old thetas: ", np.allclose(old_thetas_glob_accept, thetas[glob_update[accept]].glob), 
    #       "= accept thetas:", np.allclose(accept_thetas_glob, thetas[glob_update[accept]].glob))
    # print("Global zs accept = old zs: ", np.allclose(old_zs_glob_accept, zs[glob_update[accept]]), 
    #       "= accept zs:", np.allclose(accept_zs_glob, zs[glob_update[accept]]))
    distance_values = np.array(distance_values)
    distance_values[glob_update[accept]] = proposed_distances_glob[accept]
    if perm and zs_index is not None and ys_index is not None:
        zs_index[glob_update[accept]] = proposed_zs_index_glob[accept]
        ys_index[glob_update[accept]] = proposed_ys_index_glob[accept]
    n_accept += np.sum(accept)*M

    if verbose > 1: 
        print("Global move: acceptance rate = {:.2%} (rejection because ABC : {:.2%} rejection because MH : {:.2%})".format(np.sum(accept)/np.sum(block_choice_glob), np.mean(proposed_distances_glob > epsilon), np.mean(np.logical_and(proposed_distances_glob <= epsilon, accept_prob < uniform_samples))))
    # Local block updates
    blocks = create_block(key_blocks, zs_index[loc_update], K = K, H = H, M = M, L = L)
    proposed_thetas_loc = thetas.copy()
    proposed_thetas_loc.loc[loc_update] = proposed_thetas.loc[loc_update]
    proposed_zs_loc = zs.copy()
    # print("Sampling {} zs...".format(np.sum(block_choice_loc)))
    proposed_zs_loc[loc_update] = model.data_generator(key_data, proposed_thetas[loc_update])
    
    for h, block_h in enumerate(blocks):
        if verbose > 1: print(f"Local update: block {h+1}/{H}")
        
        key, key_h = random.split(key)
        proposed_thetas_loc_h = thetas.copy()
        proposed_thetas_loc_h.loc[loc_update[:, None], block_h] = proposed_thetas_loc.loc[loc_update[:, None], block_h]
        proposed_zs_loc_h = zs.copy()
        proposed_zs_loc_h[loc_update[:, None], block_h] = proposed_zs_loc[loc_update[:, None], block_h]
        # print("Computing {} distances...".format(np.sum(block_choice_loc)))
        if perm:
            proposed_distances_loc_h, ys_index_loc_h, zs_index_loc_h, n_lsa_loc_h = optimal_index_distance(
                model=model,
                zs=proposed_zs_loc_h[loc_update],
                y_obs=y_obs,
                epsilon=epsilon,
                ys_index=ys_index[loc_update] if ys_index is not None else None,
                zs_index=zs_index[loc_update] if zs_index is not None else None,
                verbose=verbose,
                M=M,
                L=L
            )
            n_lsa += n_lsa_loc_h
        else:
            proposed_distances_loc_h = np.array(model.distance(proposed_zs_loc_h[loc_update], y_obs))
            ys_index_loc_h = ys_index[loc_update] if ys_index is not None else None
            zs_index_loc_h = zs_index[loc_update] if zs_index is not None else None
            
        forward_kernel_loc_h = kernel(model=model, thetas=thetas[loc_update], weights=weights[loc_update], ys_index=ys_index[loc_update], zs_index=zs_index[loc_update], verbose=verbose, M=M, L=L, tau_loc_glob=forward_kernel.get_tau_loc_glob())
        backward_kernel_loc_h = kernel(model=model, thetas=proposed_thetas_loc_h[loc_update], weights=weights[loc_update], ys_index=ys_index_loc_h, zs_index=zs_index_loc_h, verbose=verbose, M=M, L=L, tau_loc_glob=forward_kernel.get_tau_loc_glob())

        prior_logratio_h = jnp.minimum(model.prior_logpdf(proposed_thetas_loc_h[loc_update]) - model.prior_logpdf(thetas[loc_update]), 703)
        kernel_logratio_h = backward_kernel_loc_h.logpdf(thetas[loc_update]) - forward_kernel_loc_h.logpdf(proposed_thetas_loc_h[loc_update])
        # print("Kernel ratio: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(kernel_logratio_h), jnp.max(kernel_logratio_h), jnp.mean(kernel_logratio_h)))
        # print("Prior ratio: min = {:.3}, max = {:.3}, mean = {:.3}".format(jnp.min(prior_logratio_h), jnp.max(prior_logratio_h), jnp.mean(prior_logratio_h)))
        
        accept_prob_h = (proposed_distances_loc_h <= epsilon) * jnp.exp(prior_logratio_h + kernel_logratio_h)
        accept_prob_h = jnp.nan_to_num(jnp.minimum(accept_prob_h, 1))
        uniform_h = random.uniform(key_h, shape=(np.sum(block_choice_loc),))
        
        accept_h = np.array(uniform_h <= accept_prob_h)
        # accept_h = accept_h * np.array(block_choice_loc)
        # old_thetas_loc_h_accept = thetas.loc[loc_update[accept_h]].copy()
        # old_zs_loc_h_accept = zs[loc_update[accept_h]].copy()
        
        accept_thetas_loc_h = proposed_thetas_loc_h.loc[loc_update[accept_h]].copy()
        accept_zs_loc_h = proposed_zs_loc_h[loc_update[accept_h]].copy()

        
        # print("Accepting {} local updates...".format(np.sum(accept_h)))
        thetas.loc[loc_update[accept_h]] = accept_thetas_loc_h.copy()
        zs[loc_update[accept_h]] = accept_zs_loc_h.copy()
        
        
        # print("Local thetas accept = old thetas: ", np.all(old_thetas_loc_h_accept == thetas[loc_update[accept_h]].loc), "= accept thetas:", np.all(accept_thetas_loc_h == thetas[loc_update[accept_h]].loc))
        # print("Local zs accept: = old zs: ", np.all(old_zs_loc_h_accept == zs[loc_update[accept_h]]), "= accept zs:", np.all(accept_zs_loc_h == zs[loc_update[accept_h]]))
        
        distance_values[loc_update[accept_h]] = proposed_distances_loc_h[accept_h]
        if perm and zs_index is not None and ys_index is not None:
            zs_index[loc_update[accept_h]] = zs_index_loc_h[accept_h]
            ys_index[loc_update[accept_h]] = ys_index_loc_h[accept_h]
        n_accept += np.sum(accept_h)*block_h.shape[1]
        if verbose > 1:
            print(f"Block {h+1}/{len(blocks)}: acceptance rate = {np.sum(accept_h)/np.sum(block_choice_loc):.2%} (rejection because ABC : {np.mean(proposed_distances_loc_h > epsilon):.2%} rejection because MH : {np.mean(np.logical_and(proposed_distances_loc_h <= epsilon, accept_prob_h < uniform_h)):.2%})")
    
    n_sim = n_particles * M * (2 if both_loc_glob else 1)
    if verbose > 1:
        print(f"Total accepted: {n_accept}/{n_sim} ({n_accept / n_sim:.1%})")
    accept_rate = n_accept / n_sim
    return thetas, zs, distance_values, ys_index, zs_index, n_lsa, accept_rate, n_sim
