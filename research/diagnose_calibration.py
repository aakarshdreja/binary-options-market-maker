"""Is `_price_uncertainty` calibrated on the options we ACTUALLY quote?

This replaces `diagnose_uncertainty.py`, which had a measurement bug that made it
systematically optimistic. It sampled prices only after `session.run()` had
finished, by which point the book has decayed to expiry -- a typical post-run
book looks like [0, 1, 1, 1, 0, 5, 2, ...] steps remaining, and an option at zero
steps is priced exactly right by everyone, because the payoff is already
determined. So it graded us almost entirely on the easiest options in the game
and reported near-zero error (spreads showed |err| = 0.0000 off a single expired
contract) exactly where the real errors turn out to be largest.

Here the book is sampled at QUOTE time: the session is advanced day by day and
every live option is scored each day, weighting the measurement the way real
order flow weights it. Results are additionally broken out by time to expiry,
because that is the axis the old diagnostic was silently collapsing.

What we want to see is the estimated uncertainty covering the realised error --
ratio >= 1 -- in EVERY cell, not just on average. A ratio below 1 in a populated
cell means we quote tighter than our own ignorance justifies there, which is how
a market maker loses money while looking profitable.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics
from collections import defaultdict

from Market_Maker import FED_FUNDS_RATE_UNDERLYING_ID, BinaryOption
from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS


def classify(option: BinaryOption) -> str:
    company_legs = [leg for leg in option.legs if leg.underlying_id != FED_FUNDS_RATE_UNDERLYING_ID]
    if not company_legs:
        return "rate"
    return "single" if len(company_legs) == 1 else "spread"


def expiry_bucket(steps: int) -> str:
    if steps <= 1:
        return "1"
    if steps <= 3:
        return "2-3"
    if steps <= 5:
        return "4-5"
    return "6+"


def report(title: str, rows: dict[tuple[str, str], list[tuple[float, float]]],
           keys: list[str], buckets: list[str]) -> list[str]:
    print(f"\n{title}")
    print(f"{'kind':>8}{'expiry':>8}{'n':>7}{'|err|':>9}{'est unc':>9}{'ratio':>8}"
          f"{'p90 err':>9}{'p90 unc':>9}")
    print("-" * 67)
    warnings: list[str] = []
    for kind in keys:
        for bucket in buckets:
            data = rows.get((kind, bucket))
            if not data or len(data) < 20:
                continue
            errors = sorted(error for error, _ in data)
            uncertainties = sorted(unc for _, unc in data)
            mean_error = statistics.fmean(errors)
            mean_unc = statistics.fmean(uncertainties)
            ratio = mean_unc / mean_error if mean_error else float("inf")
            index = int(0.9 * (len(errors) - 1))
            flag = "  <-- UNDER-PADDED" if ratio < 1.0 else ""
            if flag:
                warnings.append(f"{kind}/{bucket} ratio {ratio:.2f}")
            print(f"{kind:>8}{bucket:>8}{len(data):>7}{mean_error:>9.4f}{mean_unc:>9.4f}"
                  f"{ratio:>8.2f}{errors[index]:>9.4f}{uncertainties[index]:>9.4f}{flag}")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--days", type=int, default=20)
    args = parser.parse_args()

    by_cell: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    by_history: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)

    for parameters in SCENARIOS.values():
        for index in range(args.sessions):
            history_days = [60, 150, 400][index % 3]
            for day in range(0, args.days, 2):
                session = Session(parameters, seed=1000 + index * 37, num_days=day,
                                  history_days=history_days, informed_fraction=0.3,
                                  competitors=COMPETITOR_SETS["hard"])
                session.run()
                for option in session.options:
                    if option.steps_until_expiry <= 0:
                        continue  # payoff already determined; nobody can misprice it
                    ours = session.subject.price_option(option)
                    truth = session.oracle.price_option_from_parameters(session.parameters, option)
                    sample = (abs(ours - truth), session.subject._price_uncertainty(option))
                    by_cell[(classify(option), expiry_bucket(option.steps_until_expiry))].append(sample)
                    by_history[(classify(option), str(history_days))].append(sample)

    warnings = report("BY OPTION TYPE AND TIME TO EXPIRY (live options only)", by_cell,
                      ["rate", "single", "spread"], ["1", "2-3", "4-5", "6+"])
    warnings += report("BY OPTION TYPE AND HISTORY LENGTH", by_history,
                       ["rate", "single", "spread"], ["60", "150", "400"])

    print()
    if warnings:
        print(f"UNDER-PADDED CELLS: {', '.join(warnings)}")
    else:
        print("every populated cell has ratio >= 1.0")


if __name__ == "__main__":
    main()
