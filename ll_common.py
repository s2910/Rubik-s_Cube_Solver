"""Shared last-layer helpers: sticker sets, keys, state enumeration, move merging.

Everything here assumes the SOLVING ORIENTATION: yellow top (U), white bottom (D),
with the cross and F2L already solved.
"""

from itertools import product

from cross import _adjacent_faces
from cube import FACES, SLOT_INDEX, SLOTS, Cube
from f2l import _CUBIES, _cross_pieces, _current_slot, _home_slot, _pairs

# The 20 last-layer stickers: 8 on the top face (no centre) + 12 on the sides.
LL_SLOTS = [i for i, (pos, _) in enumerate(SLOTS) if pos[1] == 1 and pos != (0, 1, 0)]
assert len(LL_SLOTS) == 20


def top_color(cube):
    return cube.center_color("U")


def oll_key(cube):
    """Which of the 20 last-layer stickers are the top colour (orientation only)."""
    y = top_color(cube)
    return tuple(cube.state[s] == y for s in LL_SLOTS)


def ll_key(cube):
    """Colours of all 20 last-layer stickers (orientation + permutation)."""
    return tuple(cube.state[s] for s in LL_SLOTS)


def f2l_intact(cube):
    """Bottom two layers match their centres (independent sticker check)."""
    for pos, normal in SLOTS:
        if pos[1] <= 0:
            face = next(f for f, n in FACES.items() if n == normal)
            if cube.state[SLOT_INDEX[(pos, normal)]] != cube.center_color(face):
                return False
    return True


def oll_done(cube):
    return all(oll_key(cube)[i] for i in range(8))  # first 8 LL slots = top face


def f2l_pieces(cube):
    """Cross + all 4 pairs as tracked pieces (F2L must already be solved)."""
    pieces = _cross_pieces(cube, "D")
    for pair in _pairs(cube, "D").values():
        pieces += list(pair)
    return pieces


def ll_pieces(cube):
    """The 8 last-layer pieces (4 edges, 4 corners), each tracked by its top sticker."""
    y = top_color(cube)
    sides = _adjacent_faces("U")
    pieces = [((y, cube.center_color(g)), y) for g in sides]
    for a in sides:
        for b in sides:
            if a < b and FACES[a] != tuple(-x for x in FACES[b]):
                pieces.append(((y, cube.center_color(a), cube.center_color(b)), y))
    return pieces


# ---------------------------------------------------------- move merging ----

_TURNS = {"": 1, "2": 2, "'": 3}
_SUFFIX = {1: "", 2: "2", 3: "'"}


def simplify(moves):
    """Merge/cancel adjacent turns of the same face (R R' -> nothing, U U -> U2)."""
    out = []
    for m in moves:
        face, n = m[0], _TURNS[m[1:]]
        if out and out[-1][0] == face:
            n = (out.pop()[1] + n) % 4
            if n == 0:
                continue
        out.append((face, n))
    return [f + _SUFFIX[n] for f, n in out]


# ------------------------------------------------- orientation enumeration --

def _ll_cubies():
    corners, edges = [], []
    for slots in _CUBIES:
        if SLOTS[slots[0]][0][1] != 1:
            continue
        if len(slots) == 3:
            # order the 3 stickers so their normals are right-handed
            ns = [SLOTS[s][1] for s in slots]
            det = (ns[0][0] * (ns[1][1] * ns[2][2] - ns[1][2] * ns[2][1])
                   - ns[0][1] * (ns[1][0] * ns[2][2] - ns[1][2] * ns[2][0])
                   + ns[0][2] * (ns[1][0] * ns[2][1] - ns[1][1] * ns[2][0]))
            corners.append(slots if det > 0 else [slots[0], slots[2], slots[1]])
        else:
            edges.append(slots)
    return corners, edges


def all_oll_states():
    """All 216 valid last-layer orientation states, as cubes (F2L intact).
    Index 0 of the result is the solved state."""
    corners, edges = _ll_cubies()
    out = []
    for ct in product(range(3), repeat=4):
        if sum(ct) % 3:
            continue
        for ef in product(range(2), repeat=4):
            if sum(ef) % 2:
                continue
            c = Cube().apply("x2")  # solving orientation (yellow top)
            st = c.state[:]
            for slots, t in zip(corners, ct):
                col = [c.state[s] for s in slots]
                for k, s in enumerate(slots):
                    st[s] = col[(k - t) % 3]
            for slots, f in zip(edges, ef):
                if f:
                    st[slots[0]], st[slots[1]] = c.state[slots[1]], c.state[slots[0]]
            out.append(Cube(st))
    assert len(out) == 216
    return out
