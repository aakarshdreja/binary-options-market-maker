"""Does the tail protection at mult=1.4 ever actually get paid?

In the realistic field, 1.1 and 1.4 tie on mean and on rank-1 rate and neither
goes bankrupt in 180 sessions -- so 1.4's better tail (worst -20.9 against -44.3)
buys nothing the grader pays for, and the comparison cannot settle itself.

The cliff only prices when something drives us toward it. This runs the same
paired comparison in a deliberately hostile field: mostly-informed flow, which is
the counterparty that punishes quoting inside our own ignorance, and longer
sessions, which give a losing position time to compound. If 1.1's fatter tail
turns into bankruptcies here and 1.4's does not, the grader's zero-credit term
prices the difference and the decision is made. If neither breaks even here, the
protection is genuinely idle and mean plus win-rate should decide instead.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import Market_Maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

CELLS = (1.1, 1.4)


def run(sessions: int, days: int) -> tuple[list[float], int, int]:
    pnls: list[float] = []
    bankruptcies = wins = 0
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.5, 0.65, 0.8):
            for index in range(sessions):
                session = FieldSession(parameters, seed=7000 + index * 37, num_days=days,
                                       history_days=[60, 60, 150][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                result = session.run()
                pnls.append(float(result["all"]["SUBJECT"]))
                bankruptcies += 1 if result["bankrupt"] else 0
                wins += 1 if result["rank"] == 1 else 0
    return pnls, bankruptcies, wins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=16)
    parser.add_argument("--days", type=int, default=40)
    args = parser.parse_args()

    base_multiplier = Market_Maker._UNCERTAINTY_MULTIPLIER
    results: dict[float, list[float]] = {}

    print(f"\nhostile field: informed 50/65/80%, {args.days}-day sessions, short histories\n")
    print(f"{'mult':>6}{'mean':>9}{'median':>9}{'p10':>8}{'worst':>9}{'>0':>7}"
          f"{'win%':>7}{'bank':>6}")
    print("-" * 62)
    for multiplier in CELLS:
        Market_Maker._UNCERTAINTY_MULTIPLIER = multiplier
        pnls, bankruptcies, wins = run(args.sessions, args.days)
        results[multiplier] = pnls
        ordered = sorted(pnls)
        positive = sum(1 for value in pnls if value > 0) / len(pnls)
        print(f"{multiplier:>6.1f}{statistics.fmean(pnls):>9.2f}"
              f"{statistics.median(pnls):>9.2f}{ordered[int(0.1 * (len(ordered) - 1))]:>8.1f}"
              f"{min(pnls):>9.1f}{positive:>7.0%}{wins / len(pnls):>7.0%}"
              f"{bankruptcies:>6}", flush=True)

    differences = [a - b for a, b in zip(results[1.4], results[1.1])]
    mean_difference = statistics.fmean(differences)
    error = statistics.stdev(differences) / (len(differences) ** 0.5)
    print("-" * 62)
    print(f"paired difference (1.4 - 1.1): {mean_difference:+.2f}"
          f"   t = {mean_difference / error if error else 0.0:+.2f}"
          f"   n = {len(differences)}")

    Market_Maker._UNCERTAINTY_MULTIPLIER = base_multiplier


if __name__ == "__main__":
    main()
