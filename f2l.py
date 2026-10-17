"""F2L solver (CFOP stage 2).

Starts from a cube whose cross is solved on the white-centre face (in the
solving orientation that is the bottom). Solves the 4 corner+edge pairs.

How it works
------------
* One sticker per piece is enough to know a piece's position AND orientation,
  so the search state is a short tuple of slot indices: the 4 cross edges,
  every pair already solved, and the 2 pieces of the pair being solved.
* Each pair is solved by *bidirectional BFS*: one search grows from the
  current state, one grows backwards from the goal, until they meet. That
  finds a shortest solution that keeps the cross and earlier pairs intact.
* All remaining pairs are searched in lock-step, one level at a time, and
  the first to finish wins. So at every step the cheapest pair is chosen,
  which is what good human solvers do.
"""

from itertools import combinations

from cross import _adjacent_faces, is_cross_solved, white_face
from cube import FACES, MOVES, SLOT_INDEX, SLOTS, Cube, invert_moves

_MOVE_LIST = [(n, p, invert_moves([n])[0]) for n, p in MOVES.items()]


# ---------------------------------------------------------------- pieces ----

def _build_cubies():
    d = {}
    for i, (pos, _) in enumerate(SLOTS):
        if sum(1 for p in pos if p) >= 2:  # edges (2 stickers) and corners (3)
            d.setdefault(pos, []).append(i)
    return list(d.values())


_CUBIES = _build_cubies()


def _face_of_color(cube, color):
    return next(f for f in FACES if cube.center_color(f) == color)


def _home_slot(cube, colors, ref):
    """Slot where the sticker of colour `ref` sits when the piece is solved."""
    pos = [0, 0, 0]
    for c in colors:
        n = FACES[_face_of_color(cube, c)]
        for i in range(3):
            pos[i] += n[i]
    return SLOT_INDEX[(tuple(pos), FACES[_face_of_color(cube, ref)])]


def _current_slot(cube, colors, ref):
    want = set(colors)
    for slots in _CUBIES:
        if len(slots) == len(want) and {cube.state[s] for s in slots} == want:
            return next(s for s in slots if cube.state[s] == ref)
    raise ValueError(f"piece {colors} not found")


def _cross_pieces(cube, d_face):
    w = cube.center_color(d_face)
    return [((w, cube.center_color(g)), w) for g in _adjacent_faces(d_face)]


def _pairs(cube, d_face):
    """The 4 F2L pairs as {name: (corner_piece, edge_piece)}."""
    w = cube.center_color(d_face)
    sides = _adjacent_faces(d_face)
    out = {}
    for a, b in combinations(sides, 2):
        if sum(x * y for x, y in zip(FACES[a], FACES[b])) != 0:
            continue  # opposite faces never form a slot
        ca, cb = cube.center_color(a), cube.center_color(b)
        out[a + b] = (((w, ca, cb), w), ((ca, cb), ca))
    return out


def is_f2l_solved(cube, d_face):
    """Independent sticker check: both bottom layers match their centres."""
    nd = FACES[d_face]
    for pos, normal in SLOTS:
        if sum(p * q for p, q in zip(pos, nd)) in (0, 1):
            face = next(f for f, n in FACES.items() if n == normal)
            if cube.state[SLOT_INDEX[(pos, normal)]] != cube.center_color(face):
                return False
    return True


# ---------------------------------------------------------------- search ----

class _BiSearch:
    """Bidirectional BFS that advances one level per step()."""

    def __init__(self, start, goal):
        self.f = {start: (None, None, 0)}
        self.b = {goal: (None, None, 0)}
        self.ff, self.bf = [start], [goal]
        self.df = self.db = 0
        self.solution = [] if start == goal else None

    @property
    def depth(self):
        return self.df + self.db

    def step(self):
        forward = len(self.ff) <= len(self.bf)
        seen, other, frontier = (
            (self.f, self.b, self.ff) if forward else (self.b, self.f, self.bf)
        )
        d = (self.df if forward else self.db) + 1
        new, best = [], None
        for s in frontier:
            for name, perm, inv in _MOVE_LIST:
                t = tuple([perm[x] for x in s])
                if t in seen:
                    continue
                seen[t] = (s, name if forward else inv, d)
                new.append(t)
                if t in other:
                    total = d + other[t][2]
                    if best is None or total < best[0]:
                        best = (total, t)
        if forward:
            self.ff, self.df = new, d
        else:
            self.bf, self.db = new, d
        if best is not None:
            self.solution = self._build(best[1])

    def _build(self, meet):
        fwd, s = [], meet
        while self.f[s][0] is not None:
            parent, mv, _ = self.f[s]
            fwd.append(mv)
            s = parent
        fwd.reverse()
        bwd, s = [], meet
        while self.b[s][0] is not None:
            parent, mv, _ = self.b[s]
            bwd.append(mv)
            s = parent
        return fwd + bwd


def _state_and_goal(cube, pieces):
    start = tuple(_current_slot(cube, c, r) for c, r in pieces)
    goal = tuple(_home_slot(cube, c, r) for c, r in pieces)
    return start, goal


def _solve_next_pair(cube, d_face, solved, remaining, max_depth):
    """Search every remaining pair in lock-step; return (name, moves) of the
    first to finish (= the cheapest)."""
    base = _cross_pieces(cube, d_face)
    for name in solved:
        base += list(_pairs(cube, d_face)[name])
    all_pairs = _pairs(cube, d_face)
    searches = {}
    for name in remaining:
        start, goal = _state_and_goal(cube, base + list(all_pairs[name]))
        searches[name] = _BiSearch(start, goal)

    while True:
        for name, s in searches.items():
            if s.solution is not None:
                return name, s.solution
        if next(iter(searches.values())).depth >= max_depth:
            raise RuntimeError(f"no F2L pair found within {max_depth} moves")
        for s in searches.values():
            s.step()


def solve_f2l(cube, max_depth=10, greedy=True):
    """Solve F2L. `cube` must have its cross solved on the white-centre face.

    Returns a list of (pair_name, moves) in solving order. The input cube is
    not modified. With greedy=True the cheapest pair is always taken next;
    with greedy=False pairs are taken in a fixed order.
    """
    d_face = white_face(cube)
    if not is_cross_solved(cube, d_face):
        raise ValueError("solve the cross first")
    work = cube.copy()
    remaining = list(_pairs(work, d_face))
    solved, steps = [], []
    while remaining:
        candidates = remaining if greedy else remaining[:1]
        name, moves = _solve_next_pair(work, d_face, solved, candidates, max_depth)
        work.apply(moves)
        steps.append((name, moves))
        solved.append(name)
        remaining.remove(name)
    return steps


def flatten(steps):
    return [m for _, moves in steps for m in moves]


if __name__ == "__main__":
    import time

    from cross import solve_white_cross
    from cube import orient_for_solving, random_scramble

    scramble = random_scramble(20)
    cube = Cube().apply(scramble)
    print("Scramble (white top, green front):", " ".join(scramble))
    orient_for_solving(cube)
    print("Flipped to solving orientation: yellow top, blue front, white bottom\n")

    cross = solve_white_cross(cube)
    cube.apply(cross)
    print(f"Cross ({len(cross)} moves): {' '.join(cross)}")

    t0 = time.time()
    steps = solve_f2l(cube)
    for name, moves in steps:
        cube.apply(moves)
        print(f"Pair {name} ({len(moves)} moves): {' '.join(moves) or '(already solved)'}")
    print(f"\nF2L took {time.time() - t0:.1f}s, {len(flatten(steps))} moves")
    print(cube)
    print("F2L solved?", is_f2l_solved(cube, white_face(cube)))