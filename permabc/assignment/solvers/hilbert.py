"""
Hilbert curve sort as a replacement for the Linear Sum Assignment solver.

Uses CGAL (via cgal_hilbert pybind11 module) for exact d-dimensional Hilbert sort.

Implements the Hilbert distance H_p described in:
    Bernton, Jacob, Gerber & Robert (2019) — "Approximate Bayesian Computation
    with the Wasserstein Distance", JRSS-B 81(2), section 2.3.2.

Cost: O(K log K)  vs  O(K³) for the Hungarian algorithm.
Property: H_p >= W_p  (Hilbert coupling is feasible for the transport problem).

Requires: CGAL headers + pybind11. Build with:  bash permabc/core/build_cgal.sh
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Load the CGAL backend (compiled .so lives in permabc/core/)
# ---------------------------------------------------------------------------
_HAS_CGAL = False
_cgal = None

try:
    from pathlib import Path
    import sys
    _core_dir = Path(__file__).resolve().parent.parent.parent / "core"
    _old_path = sys.path[:]
    sys.path.insert(0, str(_core_dir))
    try:
        import cgal_hilbert as _cgal
        _HAS_CGAL = True
    except ImportError:
        pass
    finally:
        sys.path[:] = _old_path
except Exception:
    pass


# ===================================================================
# CGAL backend — exact Hilbert sort in any dimension
# ===================================================================

def _cgal_sort(points: np.ndarray) -> np.ndarray:
    """Dispatch to the correct CGAL function based on dimensionality."""
    if not _HAS_CGAL:
        raise ImportError(
            "CGAL Hilbert module not available. "
            "Run:  bash permabc/core/build_cgal.sh"
        )
    pts = np.ascontiguousarray(points, dtype=np.float64)
    d = pts.shape[1]
    if d == 2:
        return np.asarray(_cgal.hilbert_sort_2d(pts), dtype=np.int64)
    elif d == 3:
        return np.asarray(_cgal.hilbert_sort_3d(pts), dtype=np.int64)
    else:
        return np.asarray(_cgal.hilbert_sort_nd(pts), dtype=np.int64)


def solve_hilbert_cgal(
    zs: np.ndarray,
    y_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """CGAL-based Hilbert sort assignment (exact d-dimensional)."""
    N, K, d = zs.shape

    ys_order = _cgal_sort(y_ref).astype(np.int32)
    ys_idx = np.tile(ys_order, (N, 1))
    zs_idx = np.empty((N, K), dtype=np.int32)

    for i in range(N):
        zs_idx[i] = _cgal_sort(zs[i]).astype(np.int32)

    return ys_idx, zs_idx


def hilbert_distance_cgal(
    zs: np.ndarray,
    y_ref: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CGAL-based Hilbert distance + assignment."""
    ys_idx, zs_idx = solve_hilbert_cgal(zs, y_ref)

    N, K, d = zs.shape
    particle_idx = np.arange(N)[:, None]
    y_assigned = y_ref[ys_idx]
    z_assigned = zs[particle_idx, zs_idx]
    w = weights[ys_idx]

    diff = y_assigned - z_assigned
    distances = np.sqrt(
        np.sum(w[:, :, None] ** 2 * diff ** 2, axis=(1, 2))
    )
    return distances, ys_idx, zs_idx
