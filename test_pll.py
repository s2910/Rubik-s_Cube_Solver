import pytest

from cube import Cube, invert_moves, parse_moves
from ll_common import f2l_intact, ll_key, oll_done
from pll import AUF, PLL_ALGS, build_table, check_alg, solve_pll, solved_cube


def test_there_are_21_pll_algorithms():
    assert len(PLL_ALGS) == 21


@pytest.mark.parametrize("name", list(PLL_ALGS))
def test_each_pll_algorithm_is_valid(name):
    assert check_alg(name) is None


def test_table_covers_all_288_permutation_states():
    assert len(build_table()) == 288


def test_pll_solves_every_state_reachable_by_an_algorithm_and_auf():
    for name, text in PLL_ALGS.items():
        alg = parse_moves(text)
        for pre in AUF:
            for post in AUF:
                cube = solved_cube().apply(invert_moves(pre + alg + post))
                assert oll_done(cube) and f2l_intact(cube)
                moves, case = solve_pll(cube)
                cube.apply(moves)
                assert cube.is_solved(), (name, pre, post)


def test_pure_top_turns_need_only_an_auf():
    for u in (["U"], ["U2"], ["U'"]):
        cube = solved_cube().apply(invert_moves(u))
        moves, case = solve_pll(cube)
        assert len(moves) == 1 and case == "AUF"
        assert cube.apply(moves).is_solved()


def test_solved_cube_needs_no_moves():
    assert solve_pll(solved_cube()) == ([], "solved")


def test_auf_variants_never_cost_more_than_alg_plus_two_turns():
    for name, text in PLL_ALGS.items():
        n = len(parse_moves(text))
        cube = solved_cube().apply(invert_moves(parse_moves(text)))
        assert len(solve_pll(cube)[0]) <= n + 2
