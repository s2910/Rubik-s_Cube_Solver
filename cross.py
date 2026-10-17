"""Optimal cross solver (CFOP stage 1).

Only the 4 cross edges matter, so the state is the slot index of each of
their 8 stickers. That is 12*11*10*9*2^4 = 190,080 states, small enough to
BFS completely from the solved cross. The BFS distance table then gives an
optimal solution (max 8 moves) by greedy descent.
"""

from collections import deque

from cube import FACES, MOVES, SLOT_INDEX, SLOTS, Cube


def _adjacent_faces(face):
    n = FACES[face]
    return [g for g, m in FACES.items() if sum(a * b for a, b in zip(n, m)) == 0]


def _goal_slots(face):
    """Slots of (cross sticker, side sticker) for each of the 4 cross edges."""
    n = FACES[face]
    out = []
    for g in _adjacent_faces(face):
        m = FACES[g]
        pos = tuple(a + b for a, b in zip(n, m))
        out.append(SLOT_INDEX[(pos, n)])
        out.append(SLOT_INDEX[(pos, m)])
    return tuple(out)


def _build_edge_slots():
    d = {}
    for i, (pos, _) in enumerate(SLOTS):
        if sum(1 for p in pos if p) == 2:  # edge cubie
            d.setdefault(pos, []).append(i)
    return list(d.values())


_EDGE_SLOTS = _build_edge_slots()


def _current_slots(cube, face):
    """Where the 8 cross stickers currently are on `cube`."""
    cross_color = cube.center_color(face)
    out = []
    for g in _adjacent_faces(face):
        side_color = cube.center_color(g)
        for slots in _EDGE_SLOTS:
            by_color = {cube.state[s]: s for s in slots}
            if set(by_color) == {cross_color, side_color}:
                out += [by_color[cross_color], by_color[side_color]]
                break
    return tuple(out)


_TABLES = {}


def _table(face):
    """BFS distance-to-solved for every cross state on `face` (cached)."""
    if face not in _TABLES:
        goal = _goal_slots(face)
        dist = {goal: 0}
        queue = deque([goal])
        perms = list(MOVES.values())
        while queue:
            s = queue.popleft()
            d = dist[s] + 1
            for p in perms:
                t = tuple(p[x] for x in s)
                if t not in dist:
                    dist[t] = d
                    queue.append(t)
        _TABLES[face] = dist
    return _TABLES[face]


def is_cross_solved(cube, face):
    """Independent check straight from stickers (does not use the table)."""
    n = FACES[face]
    for g in _adjacent_faces(face):
        m = FACES[g]
        pos = tuple(a + b for a, b in zip(n, m))
        if cube.state[SLOT_INDEX[(pos, n)]] != cube.center_color(face):
            return False
        if cube.state[SLOT_INDEX[(pos, m)]] != cube.center_color(g):
            return False
    return True


def solve_cross(cube, face="D"):
    """Shortest move list that solves the cross on `face` (any of U D F B R L)."""
    dist = _table(face)
    s = _current_slots(cube, face)
    solution = []
    while dist[s] > 0:
        for name, p in MOVES.items():
            t = tuple(p[x] for x in s)
            if dist[t] == dist[s] - 1:
                solution.append(name)
                s = t
                break
    return solution


def white_face(cube):
    """The face that currently holds the white centre."""
    return next(f for f in FACES if cube.center_color(f) == "W")


def solve_white_cross(cube):
    """Solve the white cross wherever the white centre currently is.
    In solving orientation (yellow top, white bottom) that is the D face."""
    return solve_cross(cube, white_face(cube))


def solve_best_cross(cube):
    """Try all 6 faces; return (face, moves) with the fewest moves."""
    best = None
    for face in FACES:
        moves = solve_cross(cube, face)
        if best is None or len(moves) < len(best[1]):
            best = (face, moves)
    return best


if __name__ == "__main__":
    from cube import orient_for_solving, random_scramble

    scramble = random_scramble(20)
    cube = Cube().apply(scramble)
    print("SCRAMBLE ORIENTATION: white top, green front, red right, orange left, yellow bottom, blue back")
    print("Scramble:", " ".join(scramble))
    print(cube, "\n")

    orient_for_solving(cube)
    print("SOLVING ORIENTATION (after whole-cube rotation x2): yellow top, blue front, red right, orange left, white bottom, green back")
    print(cube, "\n")

    moves = solve_white_cross(cube)
    print(f"White cross on the bottom ({len(moves)} moves): {' '.join(moves)}")
    cube.apply(moves)
    print(cube)
    print("White cross solved?", is_cross_solved(cube, white_face(cube)))