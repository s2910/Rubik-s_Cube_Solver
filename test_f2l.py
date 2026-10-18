import random

import pytest

from cross import is_cross_solved, solve_white_cross, white_face
from cube import OPPOSITE, Cube, orient_for_solving, random_scramble
from f2l import (
    _cross_pieces,
    _current_slot,
    _home_slot,
    _pairs,
    flatten,
    is_f2l_solved,
    solve_cross_and_pair,
    solve_f2l,
)


def scrambled(seed):
    return orient_for_solving(Cube().apply(random_scramble(20, random.Random(seed))))


def cube_with_cross(seed):
    c = scrambled(seed)
    c.apply(solve_white_cross(c))
    return c


def piece_is_home(cube, piece):
    colors, ref = piece
    return _current_slot(cube, colors, ref) == _home_slot(cube, colors, ref)


# ----------------------------------------------------------- F2L basics ----

def test_solved_cube_needs_no_f2l_moves():
    c = orient_for_solving(Cube())
    steps = solve_f2l(c)
    assert len(steps) == 4
    assert flatten(steps) == []
    assert is_f2l_solved(c, "D")


def test_is_f2l_solved_checker_is_not_fooled():
    assert is_f2l_solved(orient_for_solving(Cube()), "D")
    assert not is_f2l_solved(Cube().apply("R U R' U'"), "U")
    assert not is_f2l_solved(cube_with_cross(1), "D")  # cross alone is not F2L


@pytest.mark.parametrize("seed", range(40))
def test_humanised_f2l_solves_bottom_two_layers_and_keeps_cross(seed):
    c = cube_with_cross(seed)
    steps = solve_f2l(c)
    c.apply(flatten(steps))
    assert is_f2l_solved(c, "D") and is_cross_solved(c, "D")
    assert sorted(s["pair"] for s in steps) == ["BL", "BR", "FL", "FR"]


def test_each_step_keeps_cross_and_earlier_pairs_intact():
    for seed in range(15):
        c = cube_with_cross(100 + seed)
        pairs = _pairs(c, "D")
        done = []
        for step in solve_f2l(c):
            c.apply(step["moves"])
            done.append(step["pair"])
            assert is_cross_solved(c, "D")
            for n in done:
                assert all(piece_is_home(c, p) for p in pairs[n]), (seed, n)


# ------------------------------------------------------- humanised style ----

def test_tier1_uses_only_U_and_the_slots_own_faces():
    seen_tier1 = 0
    for seed in range(30):
        for s in solve_f2l(cube_with_cross(200 + seed)):
            if s["tier"] == 1:
                seen_tier1 += 1
                allowed = {"U", s["pair"][0], s["pair"][1]}
                assert {m[0] for m in s["moves"]} <= allowed, s
    assert seen_tier1 > 60


def test_humanised_f2l_never_turns_the_cross_face_unless_forced():
    for seed in range(30):
        for s in solve_f2l(cube_with_cross(300 + seed)):
            if s["tier"] < 3:
                assert "D" not in {m[0] for m in s["moves"]}


def test_steps_carry_a_readable_case_description():
    steps = solve_f2l(cube_with_cross(5))
    for s in steps:
        assert s["case"].startswith("corner ") and ", edge " in s["case"]


def test_unhumanised_mode_also_solves():
    for seed in range(5):
        c = cube_with_cross(400 + seed)
        steps = solve_f2l(c, humanize=False)
        c.apply(flatten(steps))
        assert is_f2l_solved(c, "D")


def test_pair_lengths_are_reasonable():
    for seed in range(30):
        steps = solve_f2l(cube_with_cross(500 + seed))
        assert all(len(s["moves"]) <= 10 for s in steps)
        assert len(flatten(steps)) <= 40


def test_solver_does_not_mutate_input_and_requires_cross():
    c = cube_with_cross(5)
    before = c.state[:]
    solve_f2l(c)
    assert c.state == before
    with pytest.raises(ValueError):
        solve_f2l(scrambled(3))


# --------------------------------------------------- cross + first pair ----

@pytest.mark.parametrize("seed", range(40))
def test_cross_and_pair_is_within_8_moves_and_correct(seed):
    c = scrambled(seed)
    result = solve_cross_and_pair(c)
    assert len(result["moves"]) <= 8
    c.apply(result["moves"])
    assert is_cross_solved(c, "D")
    if result["pair"]:
        assert all(piece_is_home(c, p) for p in _pairs(c, "D")[result["pair"]])
    # and F2L can continue from there
    steps = solve_f2l(c, solved=[result["pair"]] if result["pair"] else [])
    c.apply(flatten(steps))
    assert is_f2l_solved(c, "D")


def test_cross_and_pair_is_found_for_most_scrambles():
    hits = sum(bool(solve_cross_and_pair(scrambled(600 + i))["pair"]) for i in range(40))
    assert hits >= 30  # measured 98% over 300 scrambles


def test_cross_and_pair_falls_back_to_plain_cross_when_too_tight():
    c = scrambled(1)
    result = solve_cross_and_pair(c, max_moves=3)
    assert result["pair"] is None
    c.apply(result["moves"])
    assert is_cross_solved(c, "D")


def test_cross_and_pair_on_solved_cube_is_empty():
    result = solve_cross_and_pair(orient_for_solving(Cube()))
    assert result["moves"] == []


def test_cross_pieces_start_home_after_cross_solve():
    c = cube_with_cross(9)
    assert all(piece_is_home(c, p) for p in _cross_pieces(c, "D"))
