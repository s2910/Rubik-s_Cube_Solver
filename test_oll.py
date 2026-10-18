import pytest

from cube import Cube
from ll_common import all_oll_states, f2l_intact, oll_done, oll_key
from oll import OLL_ALGS, oll_shape, solve_oll

STATES = all_oll_states()


def test_there_are_57_oll_algorithms():
    assert len(OLL_ALGS) == 57


def test_every_orientation_state_is_covered():
    assert len(STATES) == 216
    assert len({oll_key(c) for c in STATES}) == 216


@pytest.mark.parametrize("i", range(216))
def test_solve_oll_orients_the_last_layer_and_keeps_f2l(i):
    cube = STATES[i].copy()
    centers = cube.centers()
    moves, info = solve_oll(cube)
    cube.apply(moves)
    assert oll_done(cube)
    assert f2l_intact(cube)
    assert cube.centers() == centers
    if i == 0:
        assert moves == [] and info["case"] == 0
    else:
        assert 1 <= info["case"] <= 57
        assert len(moves) <= 14


def test_algorithms_are_short_and_use_only_face_turns():
    for alg in OLL_ALGS.values():
        moves = alg.split()
        assert 5 <= len(moves) <= 12
        assert all(m[0] in "RUFLB" for m in moves)


def test_shapes_cover_the_classic_groups():
    shapes = {oll_shape(c) for c in STATES[1:]}
    assert shapes == {"dot", "line", "L-shape", "cross"}
