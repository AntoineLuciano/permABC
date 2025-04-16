import jax.numpy as jnp
from jax import random
from jax.scipy.stats import truncnorm, multivariate_normal
from utils import Theta
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



class Kernel:
    """
    Base class for all proposal kernels in ABC-SMC.
    """
    def __init__(self, model, thetas, weights, ys_index, zs_index, tau_loc_glob=([], []), L = 0, M = 0, verbose=0):
        self.model = model
        self.K = model.K
        self.thetas = thetas
        self.weights = jnp.array(weights)
        self.verbose = verbose
        
        if L == 0: self.L = self.K
        else: self.L = L
        
        if M == 0: self.M = self.K
        else: self.M = M
    
        if ys_index is None:  self.ys_index = np.repeat([np.arange(model.K)], thetas.loc.shape[0], axis = 0)
        else: self.ys_index = np.array(ys_index, dtype = np.int32)
        
        if zs_index is None: self.zs_index = np.repeat([np.arange(model.K)], thetas.loc.shape[0], axis = 0)
        else: self.zs_index = np.array(zs_index, dtype = np.int32)
        
        self.thetas = thetas
        
        if len(tau_loc_glob[0]) == 0 and len(tau_loc_glob[1]) == 0:
            self.tau_loc, self.tau_glob = self.get_rw_variance()
        else: 
            self.tau_loc, self.tau_glob = tau_loc_glob
        # print("LOC =", self.tau_loc.reshape(-1), "GLOB =", self.tau_glob.reshape(-1))
        if self.verbose>201:print(f"Tau loc: min = {np.min(self.tau_loc):.2} max = {np.max(self.tau_loc):.2} std = {np.std(self.tau_loc):.2}\nTau glob: min = {np.min(self.tau_glob):.2} max = {np.max(self.tau_glob):.2} std = {np.std(self.tau_glob):.2}")
        self.tau = self.set_rw_variance_by_particle() 

    
    def get_rw_variance(self):
        """
        Compute the variance for the Random Walk (RW) proposal distribution.

        Returns:
        - (tau_loc, tau_glob): Computed or provided variance values.
        """
        
        if self.L < self.K:
            tau_loc = self.get_tau_loc_under_matching()
        else: 
            permuted_thetas_loc = self.thetas.copy().apply_permutation(self.zs_index).loc
            # for k in range(self.K):
            #     sns.kdeplot(permuted_thetas_loc[:,k,0])
            # plt.show()
            tau_loc = jnp.sqrt(2 * jnp.var(permuted_thetas_loc, axis=0)) # Compute variance for local params
        tau_glob = jnp.sqrt(2 * jnp.var(self.thetas.glob, axis=0))  # Compute variance for global params
        return tau_loc, tau_glob
    
    def get_tau_loc_glob(self):
        return self.tau_loc, self.tau_glob
    
    def get_tau_loc_under_matching(self):
        n_particles = self.thetas.loc.shape[0]
        thetas_match_K = [[] for _ in range(self.K)]
        for i in range(n_particles):
            for k in range(self.L):
                thetas_match_K[self.ys_index[i,k]].append(self.thetas.loc[i,self.zs_index[i,k]])
        tau_loc = np.zeros((self.thetas.loc.shape[1],self.thetas.loc.shape[2]))
        for k in range(self.K):
            if len(thetas_match_K[k])>25:
                tau_loc[k]=np.sqrt(2*np.var(thetas_match_K[k],axis=0))
        if self.verbose > 1:
            len_match = [len(theta_k)/n_particles for theta_k in thetas_match_K]
            for k in range(self.K):
                if len_match[k]>0:
                    print("Silo {}: {:.2%} matched!".format(k,len_match[k]), "Thetas_match: min = {:.3}, max = {:.3} mean = {:.3}".format(np.min(thetas_match_K[k]), np.max(thetas_match_K[k]), np.mean(thetas_match_K[k])))
                    #sns.kdeplot(np.array(thetas_match_K[k])[:,0])
                
                else: 
                    print("Silo {} Unmatch!!!".format(k))
            #plt.show()
        return tau_loc

    def set_rw_variance_by_particle(self):
        """
        Set the Random Walk (RW) variance for each particle individually.

        Parameters:
        - thetas: Current particles.
        - tau_loc: Variance for local parameters.
        - tau_glob: Variance for global parameters.

        Returns:
        - (tau_loc, tau_glob): Updated variance values.
        """
        n_particles = self.thetas.loc.shape[0]
        tau_loc_not_match = np.min(self.tau_loc, axis = 0)
        out_loc = np.zeros(self.thetas.loc.shape,dtype=np.float64) 
        out_loc[np.arange(n_particles)[:, None], self.zs_index[:, :self.L]] = self.tau_loc[self.ys_index]
        out_loc = np.where(out_loc == 0, tau_loc_not_match, out_loc)
        out_glob = np.repeat([self.tau_glob], n_particles, axis=0)
        return  Theta(loc = out_loc, glob = out_glob)
    

    def sample(self, key, thetas):
        """Propose new parameter values (to be implemented in subclasses)."""
        raise NotImplementedError("Kernel subclasses must implement `sample()` method.")

    def logpdf(self, theta, theta_prop):
        """Compute log-density of proposal distribution (to be implemented in subclasses)."""
        raise NotImplementedError("Kernel subclasses must implement `logpdf()` method.")

class KernelRW(Kernel):
    """
    Random Walk (RW) Kernel for parameter proposals.
    """
    def sample(self, key):
        """Propose new parameter values using Random Walk."""
        key, key_loc, key_glob = random.split(key, 3)
        proposed_loc = self.thetas.loc + random.normal(key_loc, shape=self.thetas.loc.shape) * self.tau.loc
        proposed_glob = self.thetas.glob + random.normal(key_glob, shape=self.thetas.glob.shape) * self.tau.glob
        return Theta(loc=proposed_loc, glob=proposed_glob)

    def logpdf(self, theta, theta_prop):
        """Compute log-density of proposal distribution."""
        logpdf_loc = -0.5 * ((theta_prop.loc - theta.loc) / self.tau_loc) ** 2
        logpdf_glob = -0.5 * ((theta_prop.glob - theta.glob) / self.tau_glob) ** 2
        return jnp.sum(logpdf_loc, axis=(1, 2)) + jnp.sum(logpdf_glob, axis=1)

class KernelTruncatedRW(Kernel):
    """
    Truncated Random Walk (TRW) Kernel to ensure parameter constraints.
    Uses a truncated normal distribution instead of clipping values.
    """
    def sample(self, key):
        """Propose new parameter values while enforcing constraints with truncated normal sampling."""
        key, key_loc, key_glob = random.split(key, 3)

        # Extract model bounds
        loc_min, loc_max = self.model.support_par_loc[:, 0], self.model.support_par_loc[:, 1]
        glob_min, glob_max = self.model.support_par_glob[:, 0], self.model.support_par_glob[:, 1]

        # Compute normalized bounds for truncation
        a_loc = (loc_min - self.thetas.loc) / self.tau.loc
        b_loc = (loc_max - self.thetas.loc) / self.tau.loc
        a_glob = (glob_min - self.thetas.glob) / self.tau.glob
        b_glob = (glob_max - self.thetas.glob) / self.tau.glob

        # Sample from truncated normal distributions using jax.random.truncated_normal
        proposed_loc = self.thetas.loc + random.truncated_normal(key_loc, lower=a_loc, upper=b_loc) * self.tau.loc
        proposed_glob = self.thetas.glob + random.truncated_normal(key_glob, lower=a_glob, upper=b_glob) * self.tau.glob
        # return Theta(loc=proposed_loc, glob=proposed_glob)
        return Theta(loc=np.array(proposed_loc), glob=np.array(proposed_glob))

    def logpdf(self, thetas_prop):
        """Compute log-density of truncated proposal distribution."""
        a_loc = (self.model.support_par_loc[:, 0] - self.thetas.loc) / self.tau.loc
        b_loc = (self.model.support_par_loc[:, 1] - self.thetas.loc) / self.tau.loc
        a_glob = (self.model.support_par_glob[:, 0] - self.thetas.glob) / self.tau.glob
        b_glob = (self.model.support_par_glob[:, 1] - self.thetas.glob) / self.tau.glob

        logpdf_loc = truncnorm.logpdf(thetas_prop.loc, a=a_loc, b=b_loc, loc=self.thetas.loc, scale=self.tau.loc)
        logpdf_glob = truncnorm.logpdf(thetas_prop.glob, a=a_glob, b=b_glob, loc=self.thetas.glob, scale=self.tau.glob)

        return jnp.sum(logpdf_loc, axis=(1, 2)) + jnp.sum(logpdf_glob, axis=1)

# class KernelCorrelatedRW(Kernel):
#     """
#     Correlated Random Walk Kernel using a Multivariate Normal distribution.
#     """
#     def __init__(self, model, thetas, weights, covariance_matrix, tau_k=([], []), verbose=0):
#         super().__init__(model, thetas, weights, tau_k, verbose)
#         self.covariance_matrix = covariance_matrix  # Full covariance matrix (not diagonal)

#     def sample(self, key, thetas):
#         """Propose new parameter values using a Multivariate Normal distribution."""
#         key, key_loc, key_glob = random.split(key, 3)

#         # Sample multivariate normal perturbations
#         mvn_perturbations = multivariate_normal.rvs(mean=jnp.zeros_like(thetas.glob), cov=self.covariance_matrix, key=key_glob)

#         proposed_loc = thetas.loc + random.normal(key_loc, shape=thetas.loc.shape) * self.tau_loc
#         proposed_glob = thetas.glob + mvn_perturbations  # Apply correlated noise

#         return Theta(loc=proposed_loc, glob=proposed_glob)

#     def logpdf(self, theta, theta_prop):
#         """Compute log-density of the correlated RW proposal distribution."""
#         logpdf_loc = -0.5 * ((theta_prop.loc - theta.loc) / self.tau_loc) ** 2
#         logpdf_glob = multivariate_normal.logpdf(theta_prop.glob, mean=theta.glob, cov=self.covariance_matrix)
        
#         return jnp.sum(logpdf_loc, axis=(1, 2)) + logpdf_glob




