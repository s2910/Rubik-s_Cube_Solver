import random

import pytest

from cube import Cube, parse_moves, random_scramble
from solver import format_solution, solve_scramble


@pytest.mark.parametrize("seed", range(40))
def test_full_solve_for_random_scrambles(seed):
    scramble = random_scramble(20, random.Random(seed))
    result = solve_scramble(scramble)
    assert result["cube"].is_solved()
    names = [s["stage"] for s in result["stages"]]
    assert names[0].startswith("Cross") and names[-2:] == ["OLL", "PLL"]
    assert sum(n.startswith("F2L") for n in names) in (3, 4)
    assert len(result["moves"]) <= 75


def test_cross_stage_never_exceeds_8_moves_when_it_includes_a_pair():
    for seed in range(40):
        stage = solve_scramble(random_scramble(20, random.Random(100 + seed)))["stages"][0]
        assert len(stage["moves"]) <= 8


def test_solution_replays_from_the_scramble():
    # Replaying scramble + flip + all printed moves on a fresh cube solves it.
    scramble = random_scramble(25, random.Random(77))
    result = solve_scramble(scramble)
    fresh = Cube().apply(scramble).apply("x2")
    assert not fresh.is_solved()
    fresh.apply([m for s in result["stages"] for m in s["moves"]])
    assert fresh.is_solved()


def test_hand_written_scramble_and_already_solved_cube():
    result = solve_scramble(parse_moves("R U R' U' F2 D L' B"))
    assert result["cube"].is_solved()
    assert solve_scramble([])["moves"] == []


def test_format_solution_lists_every_stage_and_total():
    text = format_solution(solve_scramble(random_scramble(20, random.Random(5))))
    for word in ("Cross", "F2L", "OLL", "PLL", "TOTAL"):
        assert word in text
