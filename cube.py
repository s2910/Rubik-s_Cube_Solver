"""Rubik's cube model: 54 stickers, moves generated from 3D geometry.

Generating moves by rotating sticker coordinates avoids hand-typed
permutation tables (a classic source of bugs).
"""

import random
from itertools import product

# Face name -> outward normal vector
FACES = {
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
}

# Solved-state colour for each face
FACE_COLOR = {"U": "W", "D": "Y", "F": "G", "B": "B", "R": "R", "L": "O"}

NORMAL_TO_FACE = {v: k for k, v in FACES.items()}


def _build_slots():
    """Every sticker = (cubie position, outward normal). Fixed order = index."""
    slots = []
    for face, n in FACES.items():
        for pos in product((-1, 0, 1), repeat=3):
            # sticker lies on this face if the cubie is in the outer layer along n
            if sum(p * q for p, q in zip(pos, n)) == 1:
                slots.append((pos, n))
    return slots


SLOTS = _build_slots()
SLOT_INDEX = {s: i for i, s in enumerate(SLOTS)}
assert len(SLOTS) == 54


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _rotate_cw(v, axis):
    """Rotate v by 90 degrees clockwise when viewed from the tip of `axis`."""
    c = _cross(axis, v)
    d = _dot(axis, v)
    return tuple(-c[i] + axis[i] * d for i in range(3))


def _build_move(face):
    axis = FACES[face]
    dest = list(range(54))  # dest[i] = where sticker i goes
    for i, (pos, normal) in enumerate(SLOTS):
        if _dot(pos, axis) == 1:  # sticker is in the turning layer
            dest[i] = SLOT_INDEX[(_rotate_cw(pos, axis), _rotate_cw(normal, axis))]
    return dest


_BASE = {f: _build_move(f) for f in FACES}


def _compose(p, q):
    """Apply permutation p, then q."""
    return [q[p[i]] for i in range(54)]


# All 18 moves: R, R', R2 ...
MOVES = {}
for _f, _p in _BASE.items():
    _p2 = _compose(_p, _p)
    _p3 = _compose(_p2, _p)
    MOVES[_f] = _p
    MOVES[_f + "2"] = _p2
    MOVES[_f + "'"] = _p3


def _build_rotation(face):
    """Whole-cube rotation about the axis of `face` (x=R, y=U, z=F direction)."""
    axis = FACES[face]
    return [
        SLOT_INDEX[(_rotate_cw(pos, axis), _rotate_cw(normal, axis))]
        for pos, normal in SLOTS
    ]


# Kept separate from MOVES so searches over the 18 face moves stay correct.
ROTATIONS = {}
for _name, _face in (("x", "R"), ("y", "U"), ("z", "F")):
    _r = _build_rotation(_face)
    _r2 = _compose(_r, _r)
    ROTATIONS[_name] = _r
    ROTATIONS[_name + "2"] = _r2
    ROTATIONS[_name + "'"] = _compose(_r2, _r)


def _build_layer_move(axis, depths):
    """Turn every layer whose coordinate along `axis` is in `depths`."""
    dest = list(range(54))
    for i, (pos, normal) in enumerate(SLOTS):
        if _dot(pos, axis) in depths:
            dest[i] = SLOT_INDEX[(_rotate_cw(pos, axis), _rotate_cw(normal, axis))]
    return dest


# Slice moves (M follows L, E follows D, S follows F) and wide moves
# (r = R + M', l = L + M, u = U + E', d = D + E, f = F + S, b = B + S').
_EXTRA_SPEC = {
    "M": ((-1, 0, 0), {0}), "E": ((0, -1, 0), {0}), "S": ((0, 0, 1), {0}),
    "r": ((1, 0, 0), {0, 1}), "l": ((-1, 0, 0), {0, 1}),
    "u": ((0, 1, 0), {0, 1}), "d": ((0, -1, 0), {0, 1}),
    "f": ((0, 0, 1), {0, 1}), "b": ((0, 0, -1), {0, 1}),
}
EXTRA = {}
for _name, (_axis, _depths) in _EXTRA_SPEC.items():
    _e = _build_layer_move(_axis, _depths)
    _e2 = _compose(_e, _e)
    EXTRA[_name] = _e
    EXTRA[_name + "2"] = _e2
    EXTRA[_name + "'"] = _compose(_e2, _e)

# Everything apply() understands. MOVES alone is the 18 face moves that the
# solvers search over; rotations and slice/wide moves are kept out of it.
ALL_MOVES = {**MOVES, **ROTATIONS, **EXTRA}


def parse_moves(text):
    """'R U R2 U'' -> ['R', 'U', 'R2', "U'"]. Accepts ’ as well as '."""
    tokens = text.replace("\u2019", "'").split()
    for t in tokens:
        if t not in ALL_MOVES:
            raise ValueError(f"Unknown move: {t!r}")
    return tokens


def invert_moves(moves):
    inv = []
    for m in reversed(moves):
        if m.endswith("2"):
            inv.append(m)
        elif m.endswith("'"):
            inv.append(m[0])
        else:
            inv.append(m + "'")
    return inv


OPPOSITE = {"U": "D", "D": "U", "F": "B", "B": "F", "R": "L", "L": "R"}


def random_scramble(length=20, rng=None):
    """Random scramble with no wasted moves.

    Rules (same idea as official WCA scramblers):
      * never turn the same face twice in a row (no R R', R R2, R R)
      * never repeat a face across its opposite face (no R L R, since
        R and L commute and R L R is just R2 L)
    Returns a list of moves like ["R", "U'", "F2", ...].
    """
    rng = rng or random.Random()
    seq = []
    while len(seq) < length:
        face = rng.choice(list(FACES))
        if seq:
            last = seq[-1][0]
            if face == last:
                continue
            if len(seq) >= 2 and seq[-2][0] == face and OPPOSITE[face] == last:
                continue
        seq.append(face + rng.choice(["", "'", "2"]))
    return seq


class Cube:
    def __init__(self, state=None):
        if state is None:
            state = [FACE_COLOR[NORMAL_TO_FACE[n]] for _, n in SLOTS]
        self.state = list(state)

    def copy(self):
        return Cube(self.state)

    def apply(self, moves):
        """Apply a move string or list of moves. Returns self for chaining."""
        if isinstance(moves, str):
            moves = parse_moves(moves)
        for m in moves:
            perm = ALL_MOVES[m]
            new = self.state[:]
            for i, d in enumerate(perm):
                new[d] = self.state[i]
            self.state = new
        return self

    def center_color(self, face):
        """Colour of the centre piece currently sitting on `face`."""
        n = FACES[face]
        return self.state[SLOT_INDEX[(n, n)]]

    def centers(self):
        return {f: self.center_color(f) for f in FACES}

    def is_solved(self):
        for face, n in FACES.items():
            colors = {
                self.state[i] for i, (_, nn) in enumerate(SLOTS) if nn == n
            }
            if len(colors) != 1:
                return False
        return True

    def face_grid(self, face):
        """3x3 grid of colours for a face, as seen looking at that face."""
        n = FACES[face]
        # Choose 'right' and 'down' directions per face for a standard net
        orient = {
            "U": ((1, 0, 0), (0, 0, 1)),
            "D": ((1, 0, 0), (0, 0, -1)),
            "F": ((1, 0, 0), (0, -1, 0)),
            "B": ((-1, 0, 0), (0, -1, 0)),
            "R": ((0, 0, -1), (0, -1, 0)),
            "L": ((0, 0, 1), (0, -1, 0)),
        }[face]
        right, down = orient
        grid = [[None] * 3 for _ in range(3)]
        for i, (pos, nn) in enumerate(SLOTS):
            if nn == n:
                col = _dot(pos, right) + 1
                row = _dot(pos, down) + 1
                grid[row][col] = self.state[i]
        return grid

    def __str__(self):
        def rows(face):
            return ["".join(r) for r in self.face_grid(face)]

        pad = " " * 4
        out = [pad + r for r in rows("U")]
        for a, b, c, d in zip(rows("L"), rows("F"), rows("R"), rows("B")):
            out.append(f"{a} {b} {c} {d}")
        out += [pad + r for r in rows("D")]
        return "\n".join(out)


def orient_for_solving(cube):
    """Flip the cube from scramble orientation (white top, green front, red
    right, orange left, yellow bottom, blue back) to solving orientation
    (yellow top, blue front, red right, orange left, white bottom, green
    back). That is a whole-cube rotation x2, not a face turn."""
    if cube.centers() != FACE_COLOR:
        raise ValueError("cube must be in scramble orientation (white top, green front)")
    return cube.apply("x2")


if __name__ == "__main__":
    cube = Cube()
    print("Solved (white top, green front):\n", cube, "\n", sep="")
    scramble = " ".join(random_scramble(20))
    print("Scramble:", scramble)
    cube.apply(scramble)
    print(cube)
    print("Solved?", cube.is_solved())
    cube.apply(invert_moves(parse_moves(scramble)))
    print("After applying inverse, solved?", cube.is_solved())
