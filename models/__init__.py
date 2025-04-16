import jax.numpy as jnp
import numpy as np
from jax import vmap

class model:
    def __init__(self, K, weights_distance=None):
        self.K = K
        self.weights_distance = np.ones(K) / K if weights_distance is None else np.array(weights_distance) / np.sum(weights_distance)

    def prior_generator(self, key, n_particles, n_silos=0):
        """Generate samples from the prior distribution."""
        raise NotImplementedError("Each model must implement its own prior generator.")

    def prior_logpdf(self, thetas):
        """Compute log PDF of the prior distribution."""
        raise NotImplementedError("Each model must implement its own prior logpdf.")

    def data_generator(self, key, thetas):
        """Generate simulated data given the model parameters."""
        raise NotImplementedError("Each model must implement its own data generator.")

    def distance(self, zs, y_obs):
        """Compute the distance metric between simulated and observed data."""
        if zs.shape[0]==1: 
            return jnp.sqrt((jnp.sum(((y_obs[0] - zs) * self.weights_distance[None, :, None]) ** 2, axis=(1, 2)))).astype(np.float32)
        return np.array(jnp.sqrt(jnp.sum(((y_obs[0] - zs) * self.weights_distance[None, :, None]) ** 2, axis=(1, 2))))

    def distance_global(self, zs, y_obs):
        """Compute the global distance metric between simulated and observed data."""
        
        if y_obs.shape[1] > self.K:
            return jnp.sum((self.weights_distance[self.K:] * (y_obs[0, self.K:] - zs[:, self.K:])) ** 2, axis=1)
        return np.zeros(zs.shape[0])
  
    def distance_matrices_loc(self, zs, y_obs, M=0, L=0):
        """Compute the local pairwise distance matrices if y_obs.shape[1] > K, otherwise return 0."""

        if M == 0:
            M = self.K
        if L == 0:
            L = self.K

        def distance_matrix_no_summary(z, y, w, L):
            """Compute the pairwise distance matrix for a single particle."""
            K = y.shape[0]
            M = z.shape[0]
            matrix = jnp.zeros((2 * K - L, K + M - L))
            dist_matrix = jnp.sum(((y[:, None, :] - z[None, :, :]) * w[:, None, None]) ** 2, axis = 2) 
            matrix = matrix.at[:K, :M].set(dist_matrix)
            
            return matrix
        res = vmap(distance_matrix_no_summary, in_axes=(0, None, None, None))(zs[:, :M], y_obs[0, :self.K], self.weights_distance[:self.K], L)
        return res

    def update_weights_distance(self, zs, verbose=0):
        """Update distance weights dynamically based on median absolute deviation (MAD)."""
        def mad(x):
            return jnp.median(jnp.abs(x - jnp.median(x)))
        if type(zs)==list:
            mad_values = np.array([mad(np.array(z)) for z in zs])
        else:
            new_zs = zs.swapaxes(0, 1).reshape(zs.shape[1], -1) # (K, n_particles * n_silos)
            mad_values = vmap(mad, in_axes=0)(new_zs) # (K,)  
              
        # Normalisation des poids pour qu'ils forment une distribution de probabilité
        weights = 1/mad_values
        self.weights_distance = weights / jnp.sum(weights)
        if verbose > 1: print("Weights distance: min = {:.3}, max = {:.3}".format(self.weights_distance.min(), self.weights_distance.max()))
        
    def reset_weights_distance(self):
        """Reset distance weights to uniform."""
        self.weights_distance = jnp.ones(self.weights_distance.shape) / self.weights_distance.shape[0]
        