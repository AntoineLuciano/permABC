"""
Backward-compatibility shim — real code is in permabc.assignment.solvers.hilbert.
"""
from ..assignment.solvers.hilbert import (
    hilbert_distance_cgal,
    solve_hilbert_cgal,
    _HAS_CGAL,
)
