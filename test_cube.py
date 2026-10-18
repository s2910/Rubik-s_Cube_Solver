import random

import pytest

from cube import MOVES, FACES, Cube, invert_moves, parse_moves


def test_starts_solved():
    assert Cube().is_solved()


@pytest.mark.parametrize("face", list(FACES))
def test_quarter_turn_four_times_is_identity(face):
    c = Cube().apply([face] * 4)
    assert c.is_solved()


@pytest.mark.parametrize("move", list(MOVES))
def test_move_then_inverse_is_identity(move):
    c = Cube().apply([move] + invert_moves([move]))
    assert c.is_solved()


@pytest.mark.parametrize("face", list(FACES))
def test_single_turn_unsolves(face):
    assert not Cube().apply(face).is_solved()


def test_sexy_move_has_order_6():
    c = Cube()
    for _ in range(6):
        c.apply("R U R' U'")
    assert c.is_solved()


def test_sune_has_order_6():
    c = Cube()
    for _ in range(6):
        c.apply("R U R' U R U2 R'")
    assert c.is_solved()


def test_known_u_move_cycles_front_to_left():
    # After U, the top row of the F face shows what was on the R face.
    c = Cube().apply("U")
    assert c.face_grid("F")[0] == ["R", "R", "R"]
    assert c.face_grid("L")[0] == ["G", "G", "G"]


def test_random_scrambles_are_reversible():
    rng = random.Random(42)
    for _ in range(200):
        moves = [rng.choice(list(MOVES)) for _ in range(30)]
        c = Cube().apply(moves).apply(invert_moves(moves))
        assert c.is_solved()


def test_parse_rejects_bad_move():
    with pytest.raises(ValueError):
        parse_moves("R X")


def test_sticker_counts_preserved():
    rng = random.Random(1)
    moves = [rng.choice(list(MOVES)) for _ in range(50)]
    c = Cube().apply(moves)
    for color in "WYGBRO":
        assert c.state.count(color) == 9


# ---- scramble generator -------------------------------------------------

from cube import OPPOSITE, random_scramble  # noqa: E402


def test_scramble_has_requested_length():
    for n in (1, 5, 20, 40):
        assert len(random_scramble(n)) == n


def test_scramble_is_reproducible_with_seed():
    a = random_scramble(25, random.Random(123))
    b = random_scramble(25, random.Random(123))
    assert a == b


def test_scramble_never_turns_same_face_twice_in_a_row():
    rng = random.Random(0)
    for _ in range(2000):
        s = random_scramble(25, rng)
        assert all(s[i][0] != s[i + 1][0] for i in range(len(s) - 1)), s


def test_scramble_never_has_face_opposite_face_same_face():
    rng = random.Random(1)
    for _ in range(2000):
        s = random_scramble(25, rng)
        for i in range(len(s) - 2):
            bad = s[i][0] == s[i + 2][0] and OPPOSITE[s[i][0]] == s[i + 1][0]
            assert not bad, s


def test_scramble_moves_are_all_valid_and_use_every_suffix():
    rng = random.Random(2)
    seen = set()
    for _ in range(200):
        for m in random_scramble(20, rng):
            assert m in MOVES
            seen.add(m[1:])
    assert seen == {"", "'", "2"}


def test_scramble_actually_scrambles():
    rng = random.Random(5)
    assert not Cube().apply(random_scramble(20, rng)).is_solved()


# ---- whole-cube rotations and solving orientation -----------------------

from cube import FACE_COLOR, ROTATIONS, orient_for_solving  # noqa: E402


@pytest.mark.parametrize("rot", ["x", "y", "z"])
def test_rotation_four_times_is_identity(rot):
    start = Cube().apply(random_scramble(20, random.Random(9)))
    assert Cube(start.state).apply([rot] * 4).state == start.state


@pytest.mark.parametrize("rot", list(ROTATIONS))
def test_rotation_then_inverse_is_identity(rot):
    start = Cube().apply(random_scramble(20, random.Random(4)))
    c = Cube(start.state).apply([rot] + invert_moves([rot]))
    assert c.state == start.state


def test_rotation_keeps_solved_cube_solved():
    for rot in ROTATIONS:
        assert Cube().apply(rot).is_solved()


def test_x2_gives_requested_solving_orientation():
    c = orient_for_solving(Cube())
    assert c.centers() == {
        "U": "Y", "F": "B", "R": "R", "L": "O", "D": "W", "B": "G",
    }


def test_orient_for_solving_rejects_wrong_orientation():
    with pytest.raises(ValueError):
        orient_for_solving(Cube().apply("y"))


def test_rotation_does_not_change_the_scramble_itself():
    # Rotating the whole cube and rotating back must restore every sticker.
    scrambled = Cube().apply(random_scramble(25, random.Random(8)))
    rotated = Cube(scrambled.state)
    orient_for_solving(rotated)
    assert rotated.state != scrambled.state
    rotated.apply("x2")
    assert rotated.state == scrambled.state


def test_face_moves_after_rotation_follow_the_new_orientation():
    # After x2 the top layer is the old bottom layer, so "x2 U" == "D x2".
    assert Cube().apply("R U F x2 U").state == Cube().apply("R U F D x2").state
    # y turns the cube like U: front -> left, left -> back, back -> right.
    # So the layer now on the right is the old back layer: "y R" == "B y".
    assert Cube().apply("R U F y R").state == Cube().apply("R U F B y").state


# ---- slice and wide moves ------------------------------------------------

from cube import ALL_MOVES, EXTRA  # noqa: E402


@pytest.mark.parametrize("wide,parts", [
    ("x", "R M' L'"), ("y", "U E' D'"), ("z", "F S B'"),
    ("r", "R M'"), ("l", "L M"), ("u", "U E'"),
    ("d", "D E"), ("f", "F S"), ("b", "B S'"),
])
def test_slice_and_wide_moves_match_their_definitions(wide, parts):
    base = Cube().apply("R U F")
    a = Cube(base.state).apply(wide)
    b = Cube(base.state).apply(parts)
    assert a.state == b.state


@pytest.mark.parametrize("move", list(EXTRA))
def test_extra_moves_have_inverses(move):
    c = Cube().apply([move] + invert_moves([move]))
    assert c.is_solved()


def test_all_moves_contains_every_family():
    assert len(ALL_MOVES) == 18 + 9 + 27
