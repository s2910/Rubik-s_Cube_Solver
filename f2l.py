"""Cross + first pair, and humanised F2L (CFOP stages 1b and 2).

Piece tracking
--------------
One sticker per piece is enough to know a piece's position AND orientation,
so a search state is a short tuple of slot indices: the 4 cross edges, every
pair already solved, and the 2 pieces of the pair being worked on.

Humanised F2L
-------------
Humans solve F2L "intuitively": they pair up a corner and an edge using only
U turns and the two faces of the target slot (e.g. U, R, F for the front-right
slot), and avoid touching the cross face. The solver searches in tiers:

  tier 1  moves of U + the slot's own two faces         (the classic way)
  tier 2  U + any side face, still never the cross face (e.g. to pull a
          piece out of another slot first)
  tier 3  everything, including the cross face          (last resort)

Each tier has a small penalty (0, 1, 3), and the cheapest (length + penalty)
over all four remaining pairs wins. So an intuitive 7-move solution beats a
6-move solution that needs awkward turns of other faces only when the
extra cost is small, exactly the trade-off a human makes.
"""

from itertools import combinations

from cross import _adjacent_faces, is_cross_solved, solve_cross, white_face
from cube import FACES, MOVES, OPPOSITE, SLOT_INDEX, SLOTS, Cube, invert_moves

_MOVE_LIST = [(n, p, invert_moves([n])[0]) for n, p in MOVES.items()]
TIER_PENALTY = {1: 0, 2: 1, 3: 3}


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
    """The 4 F2L pairs as {name: (corner_piece, edge_piece)}; name = slot faces."""
    w = cube.center_color(d_face)
    out = {}
    for a, b in combinations(_adjacent_faces(d_face), 2):
        if OPPOSITE[a] == b:
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

    def __init__(self, start, goal, moves=_MOVE_LIST, sort_tail=0):
        self.moves = moves
        self.tail = sort_tail  # last `tail` entries are interchangeable
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
            for name, perm, inv in self.moves:
                t = tuple([perm[x] for x in s])
                if self.tail:
                    t = t[:-self.tail] + tuple(sorted(t[-self.tail:]))
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


def _cheapest(searches, max_cost):
    """Advance all (key, penalty, search) triples in lock-step by total cost
    (length + penalty). Returns (key, solution, penalty) of the cheapest, or
    None. Because every level is completed before checking, the first hit at
    each cost is optimal for its search."""
    for cost in range(max_cost + 1):
        hits = []
        for key, penalty, s in searches:
            while s.solution is None and s.depth < cost - penalty:
                s.step()
            if s.solution is not None and len(s.solution) + penalty <= cost:
                hits.append((key, s.solution, penalty))
        if hits:
            return hits[0]
    return None


# ------------------------------------------------------ cross + first pair --

def solve_cross_and_pair(cube, max_moves=8):
    """Solve the white cross AND one F2L pair together if that takes at most
    `max_moves` moves in total; otherwise fall back to the plain optimal cross.

    Returns {"moves": [...], "pair": slot name or None}.
    """
    d_face = white_face(cube)
    base = _cross_pieces(cube, d_face)
    searches = []
    for name, (corner, edge) in _pairs(cube, d_face).items():
        start, goal = _state_and_goal(cube, base + [corner, edge])
        searches.append((name, 0, _BiSearch(start, goal)))
    hit = _cheapest(searches, max_moves)
    if hit is not None:
        return {"moves": hit[1], "pair": hit[0]}
    return {"moves": solve_cross(cube, d_face), "pair": None}


# ------------------------------------------------------------------- F2L ----

def _tier_moves(tier, d_face, slot):
    top = OPPOSITE[d_face]
    if tier == 1:
        faces = {top, slot[0], slot[1]}
    elif tier == 2:
        faces = {top, *_adjacent_faces(d_face)}
    else:
        faces = set(FACES)
    return [m for m in _MOVE_LIST if m[0][0] in faces]


def describe_pair(cube, d_face, name):
    """Plain-English description of where a pair's pieces are right now."""
    nd = FACES[d_face]
    corner, edge = _pairs(cube, d_face)[name]

    def where(piece, solved_layer):
        colors, ref = piece
        slot = _current_slot(cube, colors, ref)
        layer = sum(p * q for p, q in zip(SLOTS[slot][0], nd))
        home = SLOTS[_home_slot(cube, colors, ref)][0]
        if layer == -1:
            return "in the top layer"
        if SLOTS[slot][0] == home:
            return "in its slot" if slot != _home_slot(cube, colors, ref) else "solved"
        return "in another slot"

    return f"corner {where(corner, 1)}, edge {where(edge, 0)}"


def solve_f2l(cube, humanize=True, solved=(), max_cost=14):
    """Solve F2L. `cube` must have its cross solved on the white-centre face.

    `solved` lists pair names already finished (e.g. by solve_cross_and_pair).
    Returns a list of steps {"pair", "moves", "tier", "case"} in solving
    order. The input cube is not modified.

    humanize=True  : tiered intuitive search (see module docstring).
    humanize=False : shortest-possible per pair using all 18 moves.
    """
    d_face = white_face(cube)
    if not is_cross_solved(cube, d_face):
        raise ValueError("solve the cross first")
    work = cube.copy()
    done = list(solved)
    remaining = [n for n in _pairs(work, d_face) if n not in done]
    steps = []
    while remaining:
        pairs = _pairs(work, d_face)
        base = _cross_pieces(work, d_face)
        for n in done:
            base += list(pairs[n])
        searches = []
        for name in remaining:
            start, goal = _state_and_goal(work, base + list(pairs[name]))
            tiers = (1, 2, 3) if humanize else (3,)
            for tier in tiers:
                penalty = TIER_PENALTY[tier] if humanize else 0
                moves = _tier_moves(tier, d_face, name)
                searches.append(((name, tier), penalty, _BiSearch(start, goal, moves)))
        hit = _cheapest(searches, max_cost)
        if hit is None:
            raise RuntimeError("no F2L pair found within the cost limit")
        (name, tier), moves, _ = hit
        case = describe_pair(work, d_face, name)
        work.apply(moves)
        steps.append({"pair": name, "moves": moves, "tier": tier if humanize else 3, "case": case})
        done.append(name)
        remaining.remove(name)
    return steps


def flatten(steps):
    return [m for s in steps for m in (s["moves"] if isinstance(s, dict) else s[1])]


if __name__ == "__main__":
    import time

    from cube import orient_for_solving, random_scramble

    scramble = random_scramble(20)
    cube = Cube().apply(scramble)
    print("Scramble (white top, green front):", " ".join(scramble))
    orient_for_solving(cube)
    print("Flipped: yellow top, blue front, white bottom\n")

    t0 = time.time()
    first = solve_cross_and_pair(cube)
    cube.apply(first["moves"])
    label = f" + pair {first['pair']}" if first["pair"] else ""
    print(f"Cross{label} ({len(first['moves'])} moves): {' '.join(first['moves'])}")

    done = [first["pair"]] if first["pair"] else []
    steps = solve_f2l(cube, solved=done)
    for s in steps:
        cube.apply(s["moves"])
        print(f"Pair {s['pair']} ({len(s['moves'])} moves, tier {s['tier']}): "
              f"{' '.join(s['moves']) or '-'}   [{s['case']}]")
    total = len(first["moves"]) + len(flatten(steps))
    print(f"\nCross + F2L: {total} moves, {time.time() - t0:.1f}s")
    print(cube)
    print("F2L solved?", is_f2l_solved(cube, white_face(cube)))
