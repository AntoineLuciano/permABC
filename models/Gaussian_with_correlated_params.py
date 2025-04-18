import jax.numpy as jnp
from jax import random
import numba as nb  # Import Numba for acceleration
from models import model
from utils_functions import Theta  # Import Theta from utils_functions
import numpy as np
from jax.scipy.stats import norm

class GaussianWithCorrelatedParams(model):
    def __init__(self, K, n_obs=1, sigma_mu=1, sigma_alpha=1):
        """
        Gaussian model with correlated local (μ) and global (α) parameters.

        Parameters:
        - K (int): Number of silos.
        - n_obs (int): Number of observations per silo.
        - sigma_mu (float): Standard deviation of μ.
        - sigma_alpha (float): Standard deviation of α.
        """
        super().__init__(K)
        self.n_obs = n_obs
        self.sigma_mu = sigma_mu
        self.sigma_alpha = sigma_alpha

        # Parameter support
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])
        self.support_par_glob = jnp.array([[-jnp.inf, jnp.inf]])

        self.dim_loc = 1  # μ_k is a scalar
        self.dim_glob = 1  # α is a scalar
        self.loc_name = ["$\\mu_{"]
        self.glob_name = ["$\\alpha$"]

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate prior samples for α (global) and μ_k (local) parameters.

        Returns:
        - Theta dataclass containing:
          - "loc": Sampled μ_k values (n_particles, n_silos, 1).
          - "glob": Sampled α values (n_particles, 1).
        """
        if n_silos == 0:
            n_silos = self.K
        key, key_alpha, key_mu = random.split(key, 3)

        # α ~ Normal(0, sigma_alpha)
        alphas = random.normal(key_alpha, shape=(n_particles, 1)) * self.sigma_alpha

        # μ_k ~ Normal(0, sigma_mu)
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_mu 

        return Theta(loc=mus, glob=alphas)

    def prior_logpdf(self, thetas: Theta):
        """
        Compute the log-density of the prior.

        Returns:
        - Log-density values of the prior.
        """
        log_pdf_mu = norm.logpdf(thetas.loc, loc=0, scale=self.sigma_mu)
        log_pdf_alpha = norm.logpdf(thetas.glob, loc=0, scale=self.sigma_alpha)
        

        return jnp.sum(log_pdf_mu, axis=(1,2)).squeeze() + log_pdf_alpha.reshape(-1)


    def data_generator(self, key, thetas: Theta):
        """
        Generate simulated observations.

        Returns:
        - Simulated observations.
        """
        key, key_data = random.split(key)
        
        mus = thetas.loc[:,:,0]
        alphas = thetas.glob[:,0]

        return np.array(random.normal(key_data, shape=(mus.shape[0], mus.shape[1], self.n_obs)) + mus[:,:,None] + alphas[:, None, None])
    

    def prior_generator_jax(self, key, n_particles, n_silos=0):
        """
        Generate prior samples for α (global) and μ_k (local) parameters.

        Returns:
        - Tuple containing:
          - Sampled μ_k values (n_particles, n_silos, 1).
          - Sampled α values (n_particles, 1).
        """
        if n_silos == 0:
            n_silos = self.K
        key, key_mu, key_alpha = random.split(key, 3)

        # α ~ Normal(0, sigma_alpha)
        alphas = random.normal(key_alpha, shape=(n_particles, 1)) * self.sigma_alpha

        # μ_k ~ Normal(α, sigma_mu)
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_mu 

        return mus, alphas
    
    def data_generator_jax(self, key, thetas_loc, thetas_glob):
        """
        Generate simulated observations.

        Returns:
        - Simulated observations.
        """
        key, key_data = random.split(key)
        
        mus = thetas_loc[:,:,0]
        alphas = thetas_glob[:,0]

        return random.normal(key_data, shape=(mus.shape[0], mus.shape[1], self.n_obs)) + mus[:,:,None] + alphas[:, None, None]
