"""
Linear Sum Assignment (LSA) solver with parallel processing.

Uses scipy's linear_sum_assignment (Jonker-Volgenant algorithm, O(K³)).
"""

from scipy.optimize import linear_sum_assignment as _scipy_lsa
import numpy as np
import multiprocessing as mp
import concurrent.futures


# ===================================================================
# scipy-based solver (unchanged legacy path)
# ===================================================================

def solve_lsa(dist_matrices, indices=None, parallel=True, n_jobs=-1):
    """Solve N LSA problems using scipy (no warm-start)."""

    def solve_chunk(chunk):
        return [_scipy_lsa(matrix) for matrix in chunk]

    if indices is None:
        indices = np.arange(dist_matrices.shape[0])
    n_matrices = len(indices)

    mat_dim = np.max(dist_matrices.shape[1:])
    if parallel and (mat_dim > 100 or (n_matrices >= 200 and mat_dim >= 20)):
        n_cpu = mp.cpu_count() if n_jobs == -1 else n_jobs
        if n_matrices <= 1000:
            chunk_size = max(20, n_matrices // (n_cpu * 2))
        else:
            chunk_size = max(50, n_matrices // (n_cpu * 3))
        chunks = [dist_matrices[i:i+chunk_size]
                  for i in range(0, len(dist_matrices), chunk_size)]
        if len(chunks) == 1:
            return solve_chunk(chunks[0])
        max_workers = min(n_cpu, len(chunks))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            chunk_results = list(executor.map(solve_chunk, chunks))
        results = []
        for cr in chunk_results:
            results.extend(cr)
    else:
        results = solve_chunk(dist_matrices)

    ys_idx, zs_idx = zip(*results)
    return np.array(ys_idx, dtype=np.int32), np.array(zs_idx, dtype=np.int32)
