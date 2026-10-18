"""PLL (CFOP stage 4): permute the last layer with one algorithm.

21 standard algorithms. The table of all 288 last-layer permutation states is
built by simulation: for every algorithm and every combination of pre-turn and
post-turn (AUF) of the top layer, we record the shortest way to solve that
state. Every algorithm is verified when the table is built.
"""

from itertools import product

from ll_common import f2l_intact, ll_key, oll_done, simplify
from cube import Cube, invert_moves, parse_moves

PLL_ALGS = {
    "Aa": "R' F R' B2 R F' R' B2 R2",
    "Ab": "R B' R F2 R' B R F2 R2",
    "E":  "x' R U' R' D R U R' D' R U R' D R U' R' D' x",
    "F":  "R' U' F' R U R' U' R' F R2 U' R' U' R U R' U R",
    "Ga": "R2 U R' U R' U' R U' R2 U' D R' U R D'",
    "Gb": "R' U' R U D' R2 U R' U R U' R U' R2 D",
    "Gc": "R2 U' R U' R U R' U R2 D' U R U' R' D",
    "Gd": "R U R' U' D R2 U' R U' R' U R' U R2 D'",
    "H":  "R2 U2 R U2 R2 U2 R2 U2 R U2 R2",
    "Ja": "R' U L' U2 R U' R' U2 R L",
    "Jb": "R U R' F' R U R' U' R' F R2 U' R'",
    "Na": "R U R' U R U R' F' R U R' U' R' F R2 U' R' U2 R U' R'",
    "Nb": "R' U R U' R' F' U' F R U R' F R' F' R U' R",
    "Ra": "R U' R' U' R U R D R' U' R D' R' U2 R'",
    "Rb": "R' U2 R U2 R' F R U R' U' R' F' R2",
    "T":  "R U R' U' R' F R2 U' R' U' R U R' F'",
    "Ua": "R U' R U R U R U' R' U' R2",
    "Ub": "R2 U R U R' U' R' U' R' U R'",
    "V":  "R' U R' U' y R' F' R2 U' R' U R' F R F y'",
    "Y":  "F R U' R' U' R U R' F' R U R' U' R' F R F'",
    "Z":  "M' U' M2 U' M2 U' M' U2 M2",
}

AUF = [[], ["U"], ["U2"], ["U'"]]


def solved_cube():
    return Cube().apply("x2")  # solving orientation


def check_alg(name):
    """Return None if the algorithm is a valid PLL, else a reason string."""
    alg = parse_moves(PLL_ALGS[name])
    c = solved_cube()
    centers = c.centers()
    c.apply(invert_moves(alg))  # the case this algorithm solves
    if c.centers() != centers:
        return "moves the centres"
    if not f2l_intact(c):
        return "disturbs F2L"
    if not oll_done(c):
        return "changes orientation"
    c.apply(alg)
    return None if c.is_solved() else "does not solve its own case"


def build_table():
    table = {}
    for name, text in PLL_ALGS.items():
        alg = parse_moves(text)
        for pre, post in product(AUF, AUF):
            seq = simplify(pre + alg + post) if not any(m[0] in "xyz" for m in alg) \
                else pre + alg + post
            c = solved_cube().apply(invert_moves(pre + alg + post))
            key = ll_key(c)
            if key not in table or len(seq) < len(table[key][0]):
                table[key] = (seq, name)
    for u in AUF:  # only a top-layer turn is needed
        c = solved_cube().apply(invert_moves(u))
        table.setdefault(ll_key(c), (list(u), "AUF" if u else "solved"))
    return table


_TABLE = None


def solve_pll(cube):
    """Moves that finish the cube (after OLL). Returns (moves, case name)."""
    global _TABLE
    if _TABLE is None:
        _TABLE = build_table()
    moves, name = _TABLE[ll_key(cube)]
    return list(moves), name
