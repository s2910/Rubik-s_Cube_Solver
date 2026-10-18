import random

import pytest

from cross import (
    _table,
    is_cross_solved,
    solve_best_cross,
    solve_cross,
)
from cube import FACES, MOVES, Cube


@pytest.mark.parametrize("face", list(FACES))
def test_solved_cube_needs_zero_moves(face):
    assert solve_cross(Cube(), face) == []
    assert is_cross_solved(Cube(), face)


@pytest.mark.parametrize("face", list(FACES))
def test_state_space_size_and_max_depth(face):
    dist = _table(face)
    assert len(dist) == 12 * 11 * 10 * 9 * 16  # 190,080
    assert max(dist.values()) == 8  # known: cross is solvable in <= 8 moves


@pytest.mark.parametrize("face", list(FACES))
def test_random_scrambles_get_solved(face):
    rng = random.Random(7)
    for _ in range(100):
        scramble = [rng.choice(list(MOVES)) for _ in range(25)]
        cube = Cube().apply(scramble)
        moves = solve_cross(cube, face)
        assert len(moves) <= 8
        cube.apply(moves)
        assert is_cross_solved(cube, face)


def test_cross_after_one_move_is_solved_by_its_inverse_length():
    # One scramble move can never need more than one move to undo on the cross.
    for m in MOVES:
        for face in FACES:
            moves = solve_cross(Cube().apply(m), face)
            assert len(moves) <= 1


def test_best_cross_is_no_longer_than_any_face():
    rng = random.Random(3)
    for _ in range(20):
        scramble = [rng.choice(list(MOVES)) for _ in range(25)]
        cube = Cube().apply(scramble)
        face, best = solve_best_cross(cube)
        assert all(len(best) <= len(solve_cross(cube, f)) for f in FACES)
        cube.apply(best)
        assert is_cross_solved(cube, face)


def test_solver_does_not_mutate_input_cube():
    cube = Cube().apply("R U F' D2")
    before = cube.state[:]
    solve_cross(cube, "D")
    assert cube.state == before


# ---- white cross (white top, green front) -------------------------------

from cube import random_scramble  # noqa: E402
from cross import solve_white_cross  # noqa: E402


def test_white_cross_on_top_for_random_scrambles():
    rng = random.Random(11)
    for _ in range(200):
        cube = Cube().apply(random_scramble(20, rng))
        moves = solve_white_cross(cube)
        assert len(moves) <= 8
        cube.apply(moves)
        assert is_cross_solved(cube, "U")
        # white stickers on top edges, matching the side centers
        assert cube.face_grid("U")[0][1] == "W"
        assert cube.face_grid("U")[1][0] == "W"
        assert cube.face_grid("U")[1][2] == "W"
        assert cube.face_grid("U")[2][1] == "W"
        assert cube.face_grid("F")[0][1] == "G"
        assert cube.face_grid("R")[0][1] == "R"
        assert cube.face_grid("L")[0][1] == "O"
        assert cube.face_grid("B")[0][1] == "B"


# ---- scramble in white-top orientation, solve in yellow-top orientation --

from cube import orient_for_solving  # noqa: E402
from cross import white_face  # noqa: E402


def test_white_cross_after_flip_is_on_bottom_with_correct_sides():
    rng = random.Random(21)
    for _ in range(200):
        cube = Cube().apply(random_scramble(20, rng))
        orient_for_solving(cube)
        assert cube.centers() == {
            "U": "Y", "F": "B", "R": "R", "L": "O", "D": "W", "B": "G",
        }
        assert white_face(cube) == "D"

        moves = solve_white_cross(cube)
        assert len(moves) <= 8
        cube.apply(moves)

        d = cube.face_grid("D")
        assert d[0][1] == d[1][0] == d[1][1] == d[1][2] == d[2][1] == "W"
        # bottom-row middle sticker of each side face matches its centre
        assert cube.face_grid("F")[2][1] == "B"
        assert cube.face_grid("R")[2][1] == "R"
        assert cube.face_grid("L")[2][1] == "O"
        assert cube.face_grid("B")[2][1] == "G"
        assert is_cross_solved(cube, "D")


def test_flip_does_not_change_solution_length():
    # Same physical cube, so the optimal white cross length must be identical
    # whether we solve it white-top or after flipping to white-bottom.
    rng = random.Random(33)
    for _ in range(50):
        scramble = random_scramble(20, rng)
        top = len(solve_white_cross(Cube().apply(scramble)))
        flipped = orient_for_solving(Cube().apply(scramble))
        assert len(solve_white_cross(flipped)) == top
