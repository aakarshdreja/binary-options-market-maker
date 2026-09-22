"""Split the uncertainty budget correctly between rate legs and company legs.

`validate_estimation.py` now fails with ratios of 2.1 and coverage of 99-100%,
which is a real signal, not a threshold to widen again: in AGGREGATE we quote
wider than our ignorance justifies, and coverage that high means we are paying
for protection we do not use.

But the aggregate is exactly what hid the original rate bug. That test pools rate
and company options together, so heavily over-padded company legs mask
under-padded rate legs and the average looks healthy. Broken out at quote time,
the two leg types were never in balance:

    at _RATE_ERROR_SCALE = 1.0    rate 0.82-1.03    company 1.6-2.4
    at _RATE_ERROR_SCALE = 2.0    rate 1.5 -2.1     company 1.6-2.4

So raising the rate scale fixed the dangerous half and left the wasteful half
alone -- which is why the aggregate went UP rather than into balance. The missing
move is the other one: raise the rate scale AND cut the global multiplier, so
both leg types land on the same target and the aggregate comes back down.

Earlier sweeps found that cutting `_UNCERTAINTY_MULTIPLIER` below 1.4 lost money,
but every one of those ran while rate legs were still under-padded, where cutting
the global multiplier made the genuinely dangerous cells worse. That result does
not transfer, so the pair has to be swept jointly rather than one at a time.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import market_maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

# Re-run on top of the committed size and FOK settings, because those roughly
# doubled the flow we take and the uncertainty budget has to be judged against
# the book we will actually be carrying, not the smaller one it was tuned on.
#
# The open question is 2.0 vs 3.0. In the first pass 2.0 took the better mean but
# 3.0 halved the worst session (-8.6 against -16.9) and lifted the 10th
# percentile above zero, at a mean cost that was not statistically distinguishable
# from noise (t=-1.17). Under a grader that pays FULL credit for finishing top and
# ZERO for a bankruptcy, that trade is not obviously bad, so it gets decided on
# tail behaviour with the real book rather than on mean alone.
RATE_SCALES = (2.0, 2.5, 3.0)
MULTIPLIERS = (1.1, 1.4)


def run(sessions: int, days: int) -> tuple[list[float], int]:
    pnls: list[float] = []
    bankruptcies = 0
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.35, 0.5):
            for index in range(sessions):
                session = FieldSession(parameters, seed=1000 + index * 37, num_days=days,
                                       history_days=[60, 150, 400][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                result = session.run()
                pnls.append(float(result["pnl"]))
                bankruptcies += 1 if result["bankrupt"] else 0
    return pnls, bankruptcies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    base_rate = market_maker._RATE_ERROR_SCALE
    base_multiplier = market_maker._UNCERTAINTY_MULTIPLIER
    base, _ = run(args.sessions, args.days)

    print(f"\n{len(base)} paired sessions per cell; baseline is "
          f"rate={base_rate} mult={base_multiplier}\n")
    print(f"{'rate':>6}{'mult':>7}{'mean':>9}{'median':>9}{'worst':>9}{'p10':>8}"
          f"{'d(base)':>9}{'t':>7}{'bank':>6}")
    print("-" * 70)

    for rate_scale in RATE_SCALES:
        for multiplier in MULTIPLIERS:
            market_maker._RATE_ERROR_SCALE = rate_scale
            market_maker._UNCERTAINTY_MULTIPLIER = multiplier
            pnls, bankruptcies = run(args.sessions, args.days)
            differences = [a - b for a, b in zip(pnls, base)]
            mean_difference = statistics.fmean(differences)
            error = (statistics.stdev(differences) / (len(differences) ** 0.5)
                     if len(differences) > 1 else 0.0)
            ordered = sorted(pnls)
            marker = "  <-- incumbent" if (rate_scale == base_rate
                                           and multiplier == base_multiplier) else ""
            print(f"{rate_scale:>6.1f}{multiplier:>7.1f}{statistics.fmean(pnls):>9.2f}"
                  f"{statistics.median(pnls):>9.2f}{min(pnls):>9.1f}"
                  f"{ordered[int(0.1 * (len(ordered) - 1))]:>8.1f}"
                  f"{mean_difference:>+9.2f}"
                  f"{(mean_difference / error if error else 0.0):>7.2f}"
                  f"{bankruptcies:>6}{marker}", flush=True)
        print()

    market_maker._RATE_ERROR_SCALE = base_rate
    market_maker._UNCERTAINTY_MULTIPLIER = base_multiplier


if __name__ == "__main__":
    main()
