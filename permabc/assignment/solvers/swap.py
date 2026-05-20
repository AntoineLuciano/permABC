"""
Pairwise swap refinement for assignment improvement.

Given an existing assignment (ys_idx, zs_idx), iteratively swaps pairs of
assignments when the swap reduces the total cost. O(K^2) per sweep.

Two backends: Numba (preferred), pure NumPy (fallback).
"""

import numpy as np

try:
    import numba as nb
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ===================================================================
# Swap refinement — Numba (preferred)
# ===================================================================

if _HAS_NUMBA:
    @nb.njit(cache=True)
    def _swap_one_numba(c, y, z, K, max_sweeps):
        for _ in range(max_sweeps):
            changed = False
            for i in range(K - 1):
                yi = y[i]
                zi = z[i]
                for j in range(i + 1, K):
                    yj = y[j]
                    zj = z[j]
                    if c[yi, zj] + c[yj, zi] < c[yi, zi] + c[yj, zj]:
                        z[i], z[j] = zj, zi
                        zi = z[i]
                        changed = True
            if not changed:
                break

    @nb.njit(parallel=True, cache=True)
    def _swap_batch_numba(local_mats, ys_idx, zs_idx, max_sweeps):
        n = zs_idx.shape[0]
        K = zs_idx.shape[1]
        out = zs_idx.copy()
        for p in nb.prange(n):
            _swap_one_numba(local_mats[p], ys_idx[p], out[p], K, max_sweeps)
        return out

    def swap_refine_numba(local_mats, ys_idx, zs_idx, max_sweeps=2):
        """Pairwise swap refinement using Numba (parallel over particles)."""
        zs_out = _swap_batch_numba(
            np.ascontiguousarray(local_mats, dtype=np.float64),
            np.ascontiguousarray(ys_idx, dtype=np.int64),
            np.ascontiguousarray(zs_idx, dtype=np.int64),
            max_sweeps,
        )
        return ys_idx.astype(np.int32), zs_out.astype(np.int32)
else:
    def swap_refine_numba(*args, **kwargs):
        raise ImportError("Numba is not installed")


# ===================================================================
# Swap refinement — pure NumPy fallback
# ===================================================================

def _swap_refine_numpy(local_mats, ys_idx, zs_idx, max_sweeps=2):
    """Pure NumPy pairwise swap (fallback when Numba is unavailable)."""
    ys_out = np.array(ys_idx, copy=True)
    zs_out = np.array(zs_idx, copy=True)
    n = ys_out.shape[0]
    for p in range(n):
        y = ys_out[p]
        z = zs_out[p]
        c = local_mats[p]
        Lp = min(len(y), len(z))
        if Lp <= 1:
            continue
        for _ in range(max_sweeps):
            changed = False
            for i in range(Lp - 1):
                yi, zi = int(y[i]), int(z[i])
                for j in range(i + 1, Lp):
                    yj, zj = int(y[j]), int(z[j])
                    if c[yi, zj] + c[yj, zi] < c[yi, zi] + c[yj, zj]:
                        z[i], z[j] = z[j], z[i]
                        zi = int(z[i])
                        changed = True
            if not changed:
                break
        zs_out[p] = z
    return ys_out.astype(np.int32), zs_out.astype(np.int32)


# ===================================================================
# Dispatcher: always prefer Numba swap
# ===================================================================

def do_swap(local_mats, ys_idx, zs_idx, max_sweeps=2):
    """Swap refinement — Numba if available, else pure NumPy."""
    if _HAS_NUMBA:
        return swap_refine_numba(local_mats, ys_idx, zs_idx, max_sweeps)
    return _swap_refine_numpy(local_mats, ys_idx, zs_idx, max_sweeps)
