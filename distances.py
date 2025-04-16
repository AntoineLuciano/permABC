import jax.numpy as jnp
from jax import vmap, jit
from scipy.optimize import linear_sum_assignment
import numpy as np

def remove_under_matching(ys_index, zs_index, M, L, K):
    new_zs_index, new_ys_index = np.zeros((zs_index.shape[0],L),dtype=np.int64), np.zeros((ys_index.shape[0],L),dtype=np.int64)
    for i in range(ys_index.shape[0]):
        l = 0
        for j in range(ys_index.shape[1]):
            if zs_index[i,j] < M and ys_index[i,j]<K:
                new_zs_index[i,l] = zs_index[i,j]
                new_ys_index[i,l] = ys_index[i,j]
                l+=1
    return np.array(new_ys_index, dtype=np.int32), np.array(new_zs_index, dtype = np.int32)



def compute_total_distance(zs_index, ys_index, local_dist_matrices, global_distances):
    """
    Compute the total ABC distance for each particle using precomputed local distance matrices 
    and corresponding assigned indices.

    Parameters:
    - zs_index: Array of shape (n_particles, K), column indices for simulated data assignment.
    - ys_index: Array of shape (n_particles, K), row indices for observed data assignment.
    - local_dist_matrices: Array of shape (n_particles, K, K), local distance matrices.
    - global_distances: Array of shape (n_particles,), precomputed global distances.

    Returns:
    - total_distances: Array of shape (n_particles,), computed total distances for each particle.
    """
    
    return np.sqrt(vmap(jit(lambda matrix, zs_idx, ys_idx, glob: matrix[ys_idx, zs_idx].sum() + glob), in_axes=(0, 0, 0, 0))(local_dist_matrices, zs_index, ys_index,global_distances))


def optimal_index_distance(model, zs, y_obs, epsilon=0, ys_index=None, zs_index=None, verbose=0, M = 0, L = 0):
    """
    Compute the optimal assignment of observed and simulated data to minimize ABC distance.
    Uses a smart acceptance mechanism to avoid unnecessary recomputation.

    Parameters:
    - model: Bayesian model containing `distance_matrices_loc` and `distance_global` methods.
    - zs: Simulated data of shape (n_particles, K, ...).
    - y_obs: Observed data of shape (1, K, ...).
    - epsilon: ABC acceptance threshold.
    - ys_index: Optional, precomputed row indices for observed data.
    - zs_index: Optional, precomputed column indices for simulated data.
    - verbose: Verbosity level for debugging.

    Returns:
    - optimal_distances: Array of total optimal distances for each particle.
    - updated_ys_index: Updated row indices after assignment.
    - updated_zs_index: Updated column indices after assignment.
    - num_lsa: Number of particles that required a new `linear_sum_assignment`.
    """
    num_particles = zs.shape[0]  # Number of particles (samples)
    K = model.K  # Number of observed samples (K)

    if M == 0: M = K
    if L == 0: L = K
    
    # Compute local and global distance matrices
    local_dist_matrices = model.distance_matrices_loc(zs, y_obs, M, L)
    global_distances = model.distance_global(zs, y_obs)
    zs_index, ys_index = None, None
    if zs_index is None or ys_index is None:
        if verbose > 1:
            print("Performing full optimal assignment (no prior indices).")
        # Solve linear sum assignment for all particles
        local_dist_matrices = np.where(np.isinf(local_dist_matrices), 1e12, local_dist_matrices)
        new_ys_index, new_zs_index = np.array([
            linear_sum_assignment(local_dist_matrices[i]) for i in range(local_dist_matrices.shape[0])
        ]).swapaxes(0, 1)
        if L < K:
            new_ys_index, new_zs_index = remove_under_matching(new_ys_index, new_zs_index, M, L, K)
        # Compute optimal distances
        optimal_distances = compute_total_distance(new_zs_index, new_ys_index, local_dist_matrices, global_distances)

        return optimal_distances, new_ys_index, new_zs_index, num_particles  # Full recomputation

    # Copy old indices to avoid modifying input data
    previous_ys_index = ys_index.copy()
    previous_zs_index = zs_index.copy()

    ys_index = np.array(ys_index, dtype=np.int16)
    zs_index = np.array(zs_index, dtype=np.int16)

    # Compute current distances with existing indices
    current_distances = compute_total_distance(zs_index, ys_index, local_dist_matrices, global_distances)

    # Identify which particles exceed epsilon (requiring reassignment)
    particles_to_reassign = jnp.where(current_distances > epsilon)[0]
    num_lsa = len(particles_to_reassign)

    # Identify which particles already satisfy epsilon (smart acceptance)
    smart_accepted_particles = jnp.where(current_distances <= epsilon)[0]

    # Store optimal distances
    optimal_distances = current_distances.copy()

    if num_lsa > 0:
        if verbose > 1:
            print(f"Performing linear sum assignment for {num_lsa} particles.")

        # Solve linear sum assignment only for selected particles
        new_ys_index, new_zs_index = np.array([linear_sum_assignment(local_dist_matrices[i]) for i in particles_to_reassign]).swapaxes(0, 1)

        # Compute new distances for reassigned particles
        reassigned_distances = compute_total_distance(
            new_ys_index, new_zs_index, 
            local_dist_matrices[particles_to_reassign], 
            global_distances[particles_to_reassign]
        )
        if L < K:
            new_ys_index, new_zs_index = remove_under_matching(new_ys_index, new_zs_index, M, L, K)
        # Update only for reassigned particles
        optimal_distances[particles_to_reassign] = reassigned_distances
        ys_index[particles_to_reassign] = new_ys_index
        zs_index[particles_to_reassign] = new_zs_index

        # Identify accepted/rejected particles
        newly_accepted = np.where(optimal_distances[particles_to_reassign] <= epsilon)[0]
        
        changed_but_rejected = np.where(
            (optimal_distances[particles_to_reassign] > epsilon) & 
            np.any(previous_zs_index[particles_to_reassign] != zs_index[particles_to_reassign], axis=1)
        )[0]
        
        unchanged_but_rejected = np.where(
            (optimal_distances[particles_to_reassign] > epsilon) & 
            np.all(previous_zs_index[particles_to_reassign] == zs_index[particles_to_reassign], axis=1)
        )[0]
    

        if verbose > 1:
            print(f"Smart Acceptance: {len(smart_accepted_particles)/num_particles:.2%} ({len(smart_accepted_particles)}/{num_particles} particles).")
            print(f"Newly accepted after reassignment: {len(newly_accepted)/num_particles:.2%} ({len(newly_accepted)}/{num_particles} particles).")
            print(f"Changed but rejected after reassignment: {len(changed_but_rejected)/num_particles:.2%} ({len(changed_but_rejected)}/{num_particles} particles).")
            print(f"Unchanged but rejected after reassignment: {len(unchanged_but_rejected)/num_particles:.2%} ({len(unchanged_but_rejected)}/{num_particles} particles).")
            
    else:
        if verbose > 1:
            print("All particles accepted via smart acceptance (no reassignment needed).")
    if L < K:
            ys_index, zs_index = remove_under_matching(ys_index, zs_index, M, L, K)
    return optimal_distances, ys_index, zs_index, num_lsa  # Return updated distances and indices