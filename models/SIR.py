import jax.numpy as jnp
from jax import random
import jax
# Enable 64-bit precision in JAX
jax.config.update("jax_enable_x64", True)
import numba as nb  # Import Numba for acceleration
from models import model
from jax import jit, vmap  
from utils_functions import Theta  # Import Theta from utils_functions
import jax.numpy as jnp
import numpy as np


@nb.jit(nopython=True)
def simulate_sir_numba(S0, I0, R0, beta, gamma, n_pop, n_obs):
    """
    Simulate the SIR model using Numba for acceleration.

    Parameters:
    - S0, I0, R0: Initial conditions for S, I, R.
    - beta, gamma: Infection and recovery rates.
    - n_pop: Total population.
    - n_obs: Number of observations.

    Returns:
    - I_trajectories: Array of infected individuals over time.
    """
    n_particles, n_silos = S0.shape
    I_trajectories = np.zeros((n_particles, n_silos, n_obs))

    for p in range(n_particles):
        for s in range(n_silos):
            S, I, R = S0[p, s], I0[p, s], R0[p, s]
            for t in range(n_obs):
                new_inf = min(beta[p, s] * S * I / n_pop, S)
                new_rec = min(gamma[p, s] * I, I)

                I += new_inf - new_rec
                R += new_rec
                S = n_pop - I - R  # Conservation of population

                I_trajectories[p, s, t] = I

    return I_trajectories

def simulate_sir(S0, I0, R0, betas, gammas, n_pop, n_obs):
    def sir_step(carry, _):
        S, I, R, beta, gamma = carry

        new_inf = jnp.minimum(beta * S * I / n_pop, S)
        new_rec = jnp.minimum(gamma * I, I)

        I_next = I + new_inf - new_rec
        R_next = R + new_rec
        S_next = n_pop - I_next - R_next  # Conservation stricte

        return (S_next, I_next, R_next, beta, gamma), I_next

    def simulate_single_particle(S0_i, I0_i, R0_i, beta_i, gamma_i):
        carry = (S0_i, I0_i, R0_i, beta_i, gamma_i)
        # Create a dummy array of zeros with length n_obs
        dummy_steps = jnp.zeros(n_obs)
        _, I_traj = jax.lax.scan(sir_step, carry, dummy_steps)
        return I_traj  # (n_obs, n_silos)

    simulate_all = vmap(simulate_single_particle, in_axes=(0, 0, 0, 0, 0))
    I_trajectories = simulate_all(S0, I0, R0, betas, gammas)
    return jnp.transpose(I_trajectories, (0, 2, 1))  # (n_particles, n_silos, n_obs)

simulate_sir_jit = jit(simulate_sir, static_argnums=(5,6,))

class SIRWithKnownInit(model):
    def __init__(self, K, weights_distance=None, n_obs=100, n_pop=1000, low_beta=1e-8, high_beta=5, low_r0=1e-8, high_r0=5, I0=100, R0=100):
        """
        SIR model with known initial conditions.
        """
        super().__init__(K, weights_distance)
        self.n_obs = n_obs
        self.n_pop = n_pop
        self.I0 = I0
        self.R0 = R0

        # Parameter support
        self.support_par_loc = jnp.array([[low_beta, high_beta]])
        self.support_par_glob = jnp.array([[low_r0, high_r0]])
        self.loc_name = ["$\beta_{$"]
        self.glob_name = ["$R_0$"]
        self.dim_loc = 1  # β is a scalar
        self.dim_glob = 1  # R0 is a scalar
        

    def prior_generator(self, key, n_particles, n_silos=0):
        """Generate prior samples for β and R0."""
        if n_silos == 0:
            n_silos = self.K
        key, key_beta, key_r0 = random.split(key, 3)

        betas = random.uniform(key_beta, shape=(n_particles, n_silos, 1), minval=self.support_par_loc[0, 0], maxval=self.support_par_loc[0, 1])
        r0 = random.uniform(key_r0, shape=(n_particles, 1), minval=self.support_par_glob[0, 0], maxval=self.support_par_glob[0, 1])

        return Theta(loc=betas, glob=r0)

    def data_generator(self, key, thetas: Theta):
        """Generate epidemic simulations using the optimized SIR model."""
        betas = thetas.loc[:, :, 0]
        gammas = betas / thetas.glob

        return np.array(simulate_sir_numba(self.n_pop - self.I0 - self.R0, self.I0, self.R0, betas, gammas, self.n_pop, self.n_obs))


class SIRWithUnknownInit(model):
    def __init__(self, K, weights_distance = None, n_obs=100, n_pop=1000, low_I=1e-8, high_I=1000, low_R=1e-8, high_R=1000, low_beta=1e-8, high_beta=5, low_r0=1e-8, high_r0=5):
        """
        SIR model with unknown initial conditions.

        Parameters:
        - K (int): Number of silos.
        - n_obs (int): Number of observations per silo.
        - n_pop (int): Total population.
        - low_I, high_I (float): Bounds for initial infected population (I0).
        - low_R, high_R (float): Bounds for initial recovered population (R0).
        - low_beta, high_beta (float): Bounds for β (infection rate).
        - low_r0, high_r0 (float): Bounds for R0 (reproduction number).
        """
        super().__init__(K, weights_distance)
        self.n_obs = n_obs
        self.n_pop = n_pop

        # Parameter support
        self.support_par_loc = jnp.array([[low_I, high_I], [low_R, high_R], [low_beta, high_beta]])
        self.support_par_glob = jnp.array([[low_r0, high_r0]])
        self.loc_name = ["$I^0_{", "$R^0_{", "$\\beta_{"]
        self.glob_name = ["$R_0$"]
        self.dim_loc = 3
        self.dim_glob = 1
        

    def prior_generator(self, key, n_particles, n_silos=0):
        if n_silos == 0: 
            n_silos = self.K
        """Generate prior samples for I0, R0, β, and R0."""
        key, key_loc, key_glob = random.split(key, 3)
        loc = random.uniform(key_loc, shape=(n_particles, n_silos, self.support_par_loc.shape[0]), 
                             minval=self.support_par_loc[:, 0], maxval=self.support_par_loc[:, 1])
        
        glob = random.uniform(key_glob, shape=(n_particles, self.support_par_glob.shape[0]), 
                              minval=self.support_par_glob[:, 0], maxval=self.support_par_glob[:, 1])
        return Theta(loc=loc, glob=glob)

    def data_generator(self, key, thetas: Theta):
        """Generate epidemic simulations using the optimized SIR model."""
        I0 = thetas.loc[:, :, 0]
        R0 = thetas.loc[:, :, 1]
        betas = thetas.loc[:, :, 2]
        gammas = betas / thetas.glob
        S0 = self.n_pop - I0 - R0
        return np.array(simulate_sir_numba(S0, I0, R0, betas, gammas, self.n_pop, self.n_obs))
    
    def prior_logpdf(self, thetas):
        return 0.
    