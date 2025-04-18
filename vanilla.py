import jax.numpy as jnp
from jax import random, jit, vmap
from jax.lax import while_loop
from utils_functions import Theta
from distances import optimal_index_distance  
import numpy as np

def vanilla_single(key, prior_simulator, data_simulator, discrepancy, epsilon, true_data):
    def cond_fun(val):
        _, _, _, _, dist, _ = val
        return dist >= epsilon
    
    def body_fun(val):
        key, z, theta_loc, theta_glob, dist, n_sim = val
        key, key_theta, key_data = random.split(key, 3)
        theta_prop_loc, thetas_prop_glob = prior_simulator(key_theta, 1)
        data_prop = data_simulator(key_data, theta_prop_loc, thetas_prop_glob)
        dist = jnp.reshape(discrepancy(data_prop, true_data), ())
        return key, data_prop, theta_prop_loc, thetas_prop_glob, dist, n_sim+1
    
    
    key, key_theta = random.split(key)
    fake_theta_loc, fake_theta_glob = prior_simulator(key_theta, 1)
    fake_data = jnp.zeros_like(true_data).astype(float).reshape(true_data.shape)
    n_sim = 0
    key, data, theta_loc, theta_glob, dist, n_sim = while_loop(cond_fun, body_fun, (key, fake_data, fake_theta_loc, fake_theta_glob, epsilon+1, n_sim))
    
    return data, theta_loc, theta_glob, dist, n_sim

vanilla_single = jit(vanilla_single, static_argnums=(1,2,3))

def abc_vanilla(key, model, n_points, epsilon, y_obs):
    """Approximate Bayesian Computation (ABC) Vanilla Algorithm optimized with JAX.
    
    Args:
    - key: JAX PRNG key.
    - model: Bayesian model.
    - n_points: Number of particles to generate.
    - epsilon: ABC acceptance threshold.
    - y_obs: Observed data.
    
    Returns:
    - datas: Simulated data.
    - thetas: Simulated parameters.
    - dists: Distances between simulated and observed data.
    - key: Updated PRNG key.
    """
    def prior_predictive(key, prior_simulator, data_simulator, discrepancy, true_data):
        key, key_theta, key_data = random.split(key, 3)
        theta_loc, theta_glob = prior_simulator(key_theta, 1)
        data = data_simulator(key_data, theta_loc, theta_glob)
 
        dist = discrepancy(data, true_data)
        return data[0], theta_loc[0], theta_glob, dist, 1
    
    # prior_predictive = jit(prior_predictive, static_argnums=(1,2,3))
    
    keys = random.split(key, n_points)   
    if epsilon!=np.inf: datas, thetas_loc, thetas_glob, dists, n_sim = vmap(vanilla_single, in_axes=(0, None, None, None, None, None))(keys, model.prior_generator_jax, model.data_generator_jax, model.distance, epsilon, y_obs)
    else: 
        datas, thetas_loc, thetas_glob, dists, n_sim = vmap(prior_predictive, in_axes=(0, None, None, None, None))(keys, model.prior_generator_jax, model.data_generator_jax, model.distance, y_obs)
        
    thetas = Theta(loc=thetas_loc, glob=thetas_glob)
    return datas, thetas, dists, np.sum(n_sim)




def perm_abc_vanilla(key, model, n_points, epsilon, y_obs):
    """Approximate Bayesian Computation (ABC) Vanilla Algorithm optimized with JAX.
    
    Args:
    - key: JAX PRNG key.
    - model: Bayesian model.
    - n_points: Number of particles to generate.
    - epsilon: ABC acceptance threshold.
    - y_obs: Observed data.
    
    Returns:
    - datas: Simulated data.
    - thetas: Simulated parameters.
    - dists: Distances between simulated and observed data.
    - key: Updated PRNG key.
    """

    key, subkey = random.split(key)
    datas, thetas, _, n_sim= abc_vanilla(subkey, model, n_points, epsilon, y_obs)
    # model.update_weights_distance(datas)
    dists_perm, _, zs_index, _ = optimal_index_distance(model, datas, y_obs, epsilon = 0, verbose= 2)
    datas_perm = datas[np.arange(n_points)[:,None], zs_index]
    thetas_perm = thetas.apply_permutation(zs_index)
    
    return datas_perm, thetas_perm, dists_perm, np.sum(n_sim)
