"""OLL (CFOP stage 3): orient the whole last layer with one algorithm.

57 algorithms (generated and verified by build_oll.py). For any orientation
state the solver tries all four pre-turns of the top layer (none, U, U2, U')
and takes the shortest total.
"""

from ll_common import LL_SLOTS, oll_done, oll_key, simplify
from oll_table import OLL_ALGS
from cube import SLOTS


def _bits(key):
    return "".join("1" if b else "0" for b in key)


_CASE_NUMBER = {k: i for i, k in enumerate(OLL_ALGS, 1)}


def oll_shape(cube):
    """Classic recognition shape from the four top-face edges."""
    key = oll_key(cube)
    edges = [key[i] for i in range(8) if sum(1 for p in SLOTS[LL_SLOTS[i]][0] if p) == 2]
    good = [SLOTS[LL_SLOTS[i]][0] for i in range(8)
            if sum(1 for p in SLOTS[LL_SLOTS[i]][0] if p) == 2 and key[i]]
    n = sum(edges)
    if n == 4:
        return "cross"
    if n == 0:
        return "dot"
    if n == 2:
        (a, b) = good
        # opposite edges: x and z both mirrored (height y is the same)
        return "line" if a[0] == -b[0] and a[2] == -b[2] else "L-shape"
    return "other"


def solve_oll(cube):
    """Moves that orient the last layer, or [] if it is already oriented.
    Returns (moves, info) where info has the case number and edge shape."""
    if oll_done(cube):
        return [], {"case": 0, "shape": "solved"}
    best = None
    for pre in ([], ["U"], ["U2"], ["U'"]):
        c = cube.copy().apply(pre)
        key = _bits(oll_key(c))
        alg = OLL_ALGS.get(key)
        if alg is not None:
            moves = simplify(pre + alg.split())
            if best is None or len(moves) < len(best[0]):
                best = (moves, _CASE_NUMBER[key])
    if best is None:
        raise RuntimeError("OLL case not found (is F2L solved?)")
    return best[0], {"case": best[1], "shape": oll_shape(cube)}
