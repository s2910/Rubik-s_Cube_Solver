"""Full CFOP solver: scramble -> flip -> cross(+pair) -> F2L -> OLL -> PLL.

Usage:
    python solver.py                       # random 20-move scramble
    python solver.py "R U R' F2 D' L B"    # your own scramble
Scrambles are applied with white on top and green in front. The cube is then
flipped (x2) to yellow top / blue front / white bottom / green back, and every
move printed after that is in the flipped orientation.
"""

import sys
import time

from cube import Cube, orient_for_solving, parse_moves, random_scramble
from f2l import flatten, solve_cross_and_pair, solve_f2l
from ll_common import simplify
from oll import solve_oll
from pll import solve_pll


def solve_scramble(scramble):
    """Solve a scramble (list of moves). Returns a dict describing every stage."""
    cube = Cube().apply(scramble)
    orient_for_solving(cube)
    stages = []

    first = solve_cross_and_pair(cube)
    cube.apply(first["moves"])
    name = "Cross + pair " + first["pair"] if first["pair"] else "Cross"
    stages.append({"stage": name, "moves": first["moves"], "note": ""})

    done = [first["pair"]] if first["pair"] else []
    for step in solve_f2l(cube, solved=done):
        cube.apply(step["moves"])
        stages.append({"stage": f"F2L pair {step['pair']}", "moves": step["moves"],
                       "note": step["case"]})

    oll, info = solve_oll(cube)
    cube.apply(oll)
    stages.append({"stage": "OLL", "moves": oll,
                   "note": f"case #{info['case']}, {info['shape']}" if oll else "skip"})

    pll, case = solve_pll(cube)
    cube.apply(pll)
    stages.append({"stage": "PLL", "moves": pll,
                   "note": case if pll else "skip"})

    assert cube.is_solved(), "solver bug: cube not solved"
    full = [m for s in stages for m in s["moves"]]
    return {"stages": stages, "moves": simplify(full), "cube": cube}


def format_solution(result):
    lines = []
    for s in result["stages"]:
        mv = " ".join(s["moves"]) or "-"
        note = f"   [{s['note']}]" if s["note"] else ""
        lines.append(f"{s['stage']:<20} {len(s['moves']):>2} moves  {mv}{note}")
    lines.append(f"{'TOTAL':<20} {len(result['moves']):>2} moves")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scramble = parse_moves(" ".join(sys.argv[1:]))
    else:
        scramble = random_scramble(20)
    print("Scramble (white top, green front):", " ".join(scramble))
    print("Flipped for solving (x2): yellow top, blue front, white bottom\n")
    t0 = time.time()
    result = solve_scramble(scramble)
    print(format_solution(result))
    print(f"\nSolved in {time.time() - t0:.1f}s. Final cube:\n{result['cube']}")
