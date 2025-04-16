import jax.numpy as jnp
from jax import random, vmap
from scipy.stats import invgamma, norm
from models import model
from utils import Theta  # Import Theta from utils
import numpy as np

class GaussianWithNoSummaryStats(model):
    def __init__(self, K, n_obs=1, mu_0=0, sigma_0=5, alpha=2, beta=2):
        """
        Gaussian model without summary statistics.

        Parameters:
        - K (int): Number of silos.
        - n_obs (int): Number of observations per silo.
        - mu_0 (float): Prior mean of μ_k.
        - sigma_0 (float): Prior standard deviation of μ_k.
        - alpha, beta (float): Hyperparameters of the inverse-gamma prior for σ².
        """
        super().__init__(K)
        self.n_obs = n_obs
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.alpha = alpha
        self.beta = beta

        # Parameter support ranges (for potential truncation)
        self.support_par_loc = jnp.array([[-jnp.inf, jnp.inf]])
        self.support_par_glob = jnp.array([[0, jnp.inf]])

        self.dim_loc = 1  # μ_k is a scalar
        self.dim_glob = 1  # σ² is a scalar
        self.loc_name = ["$\mu_{"]
        self.glob_name = ["$\\sigma^2$"]

    def prior_generator(self, key, n_particles, n_silos=0):
        """
        Generate prior samples for μ and σ².

        Parameters:
        - key: PRNG key for randomness.
        - n_particles (int): Number of particles to generate.
        - n_silos (int): Number of silos (if 0, defaults to K).

        Returns:
        - Theta dataclass containing:
          - "loc": Sampled μ_k values (n_particles, n_silos, 1).
          - "glob": Sampled σ² values (n_particles, 1).
        """
        if n_silos == 0:
            n_silos = self.K
        key, key_mu, key_sigma = random.split(key, 3)

        # μ_k ~ Normal(mu_0, sigma_0^2)
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_0 + self.mu_0

        # σ² ~ Inverse-Gamma(alpha, beta)
        sigma2 = 1 / random.gamma(key_sigma, self.alpha, shape=(n_particles, 1)) * self.beta
        
        # return Theta(loc = mus, glob = sigma2)
        return Theta(loc=np.array(mus), glob=np.array(sigma2))
    

    def prior_logpdf(self, thetas: Theta):
        """
        Compute the log-density of the prior.

        Parameters:
        - thetas: Theta dataclass containing prior samples.

        Returns:
        - Log-density values of the prior.
        """
        log_pdf_mu = norm.logpdf(thetas.loc, loc=self.mu_0, scale=self.sigma_0)
        log_pdf_sigma2 = invgamma.logpdf(thetas.glob, a=self.alpha, scale=self.beta)
        return jnp.sum(log_pdf_mu, axis=1).squeeze() + log_pdf_sigma2.reshape(-1)

    def data_generator(self, key, thetas: Theta):
        """
        Generate simulated observations.

        Parameters:
        - key: PRNG key for randomness.
        - thetas: Theta dataclass containing prior samples.

        Returns:
        - Sorted raw generated data (no summary stats).
        """
        n_particles = thetas.loc.shape[0]
        n_silos = thetas.loc.shape[1]

        mus = thetas.loc
        sigmas = jnp.sqrt(thetas.glob)[:, :, jnp.newaxis]

        key, key_data = random.split(key)
        zs = random.normal(key_data, shape=(n_particles, n_silos, self.n_obs)) * sigmas + mus

        return np.array(self.summary(zs))  # Sorting the raw data

    def summary(self, z):
        """
        Returns sorted raw data (no transformation).

        Parameters:
        - z: Simulated observations.

        Returns:
        - Sorted values along the last axis.
        """
        return jnp.sort(z, axis=2)
    
    def prior_generator_jax(self, key, n_particles, n_silos=0):
        """
        Generate prior samples for μ and σ² using JAX.

        Parameters:
        - key: PRNG key for randomness.
        - n_particles (int): Number of particles to generate.
        - n_silos (int): Number of silos (if 0, defaults to K).

        Returns:
        - Theta dataclass containing:
          - "loc": Sampled μ_k values (n_particles, n_silos, 1).
          - "glob": Sampled σ² values (n_particles, 1).
        """
        if n_silos == 0:
            n_silos = self.K
        key, key_mu, key_sigma = random.split(key, 3)

        # μ_k ~ Normal(mu_0, sigma_0^2)
        mus = random.normal(key_mu, shape=(n_particles, n_silos, 1)) * self.sigma_0 + self.mu_0

        # σ² ~ Inverse-Gamma(alpha, beta)
        sigma2 = 1 / random.gamma(key_sigma, self.alpha, shape=(n_particles, 1)) * self.beta

        return mus , sigma2
    
    def data_generator_jax(self, key, thetas_loc, thetas_glob):
        """
        Generate simulated observations using JAX.

        Parameters:
        - key: PRNG key for randomness.
        - thetas_loc: Local parameters (μ_k).
        - thetas_glob: Global parameters (σ²).

        Returns:
        - Sorted raw generated data (no summary stats).
        """
        n_particles = thetas_loc.shape[0]
        n_silos = thetas_loc.shape[1]

        mus = thetas_loc
        sigmas = jnp.sqrt(thetas_glob)[:, :, jnp.newaxis]

        key, key_data = random.split(key)
        zs = random.normal(key_data, shape=(n_particles, n_silos, self.n_obs)) * sigmas + mus

        return jnp.sort(zs, axis=2)