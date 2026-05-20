"""
Individual assignment solvers for permABC.

Each solver provides a different trade-off between speed and optimality:
- lsa: Exact O(K^3) via scipy (Jonker-Volgenant)
- hilbert: Fast O(K log K) via CGAL Hilbert curve sorting
- sinkhorn: Approximate O(K^2 * iters) via entropic regularization
- swap: Pairwise refinement O(K^2) on an existing assignment
"""

from .lsa import solve_lsa
from .hilbert import (
    hilbert_distance_cgal, solve_hilbert_cgal, _HAS_CGAL,
)
from .sinkhorn import sinkhorn_assignment
from .swap import do_swap, swap_refine_numba, _HAS_NUMBA
