import random

import pytest

from cross import is_cross_solved, solve_white_cross, white_face
from cube import Cube, orient_for_solving, random_scramble
from f2l import (
    _cross_pieces,
    _current_slot,
    _home_slot,
    _pairs,
    flatten,
    is_f2l_solved,
    solve_f2l,
)


def cube_with_cross(seed):
    """Random scramble -> solving orientation -> white cross solved on bottom."""
    c = Cube().apply(random_scramble(20, random.Random(seed)))
    orient_for_solving(c)
    c.apply(solve_white_cross(c))
    return c


def piece_is_home(cube, piece):
    colors, ref = piece
    return _current_slot(cube, colors, ref) == _home_slot(cube, colors, ref)


def test_solved_cube_needs_no_f2l_moves():
    c = orient_for_solving(Cube())
    steps = solve_f2l(c)
    assert len(steps) == 4
    assert flatten(steps) == []
    assert is_f2l_solved(c, "D")


def test_is_f2l_solved_checker_is_not_fooled():
    assert is_f2l_solved(orient_for_solving(Cube()), "D")
    assert not is_f2l_solved(Cube().apply("R U R' U'"), "U")
    # cross alone is not F2L
    assert not is_f2l_solved(cube_with_cross(1), "D")


@pytest.mark.parametrize("seed", range(60))
def test_f2l_solves_bottom_two_layers_and_keeps_cross(seed):
    c = cube_with_cross(seed)
    d = white_face(c)
    assert d == "D"
    steps = solve_f2l(c)
    c.apply(flatten(steps))
    assert is_f2l_solved(c, d)
    assert is_cross_solved(c, d)
    assert sorted(n for n, _ in steps) == ["BL", "BR", "FL", "FR"]


def test_each_step_keeps_cross_and_earlier_pairs_intact():
    for seed in range(15):
        c = cube_with_cross(100 + seed)
        pairs = _pairs(c, "D")
        done = []
        for name, moves in solve_f2l(c):
            c.apply(moves)
            done.append(name)
            assert is_cross_solved(c, "D")
            for n in done:
                corner, edge = pairs[n]
                assert piece_is_home(c, corner), (seed, n)
                assert piece_is_home(c, edge), (seed, n)


def test_pair_lengths_are_reasonable():
    # Measured max over 600 scrambles was 9 per pair and 28 total.
    for seed in range(40):
        steps = solve_f2l(cube_with_cross(200 + seed))
        assert all(len(m) <= 10 for _, m in steps)
        assert len(flatten(steps)) <= 36


def test_solver_does_not_mutate_input():
    c = cube_with_cross(5)
    before = c.state[:]
    solve_f2l(c)
    assert c.state == before


def test_requires_solved_cross():
    c = orient_for_solving(Cube().apply(random_scramble(20, random.Random(3))))
    with pytest.raises(ValueError):
        solve_f2l(c)


def test_non_greedy_mode_also_solves():
    for seed in range(5):
        c = cube_with_cross(300 + seed)
        steps = solve_f2l(c, greedy=False)
        c.apply(flatten(steps))
        assert is_f2l_solved(c, "D")


def test_greedy_does_not_lose_on_average():
    greedy = fixed = 0
    for seed in range(25):
        c = cube_with_cross(400 + seed)
        greedy += len(flatten(solve_f2l(c)))
        fixed += len(flatten(solve_f2l(c, greedy=False)))
    assert greedy <= fixed


def test_cross_pieces_start_home_after_cross_solve():
    c = cube_with_cross(9)
    assert all(piece_is_home(c, p) for p in _cross_pieces(c, "D"))