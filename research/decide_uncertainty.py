"""Settle `_UNCERTAINTY_MULTIPLIER` on the metric the grader actually pays.

`_UNCERTAINTY_MULTIPLIER` was cut 1.4 -> 1.1 on a sweep that measured +0.90
(t=2.79). That sweep was internally paired and so was sound on its own terms, but
every run of the field drew a DIFFERENT set of competitors: `simulate_field.bind`
seeded rivals with `hash(self.name)`, which Python randomises per process. Each
sweep therefore generalised only to the one field draw it happened to sample. With
the seeding made stable (crc32), the same comparison comes back a dead heat on
mean (+0.03, t=0.05) while 1.4 holds a visibly better tail.

Mean is the wrong tiebreak for that. The grader pays FULL credit for finishing top
of the session, ZERO for going bankrupt, and PARTIAL for merely surviving -- so the
downside is bounded by a cliff, not by the size of the loss, and a config that
matches on mean while losing less often is strictly better. This reports the three
quantities that map onto those terms: rank-1 rate (full credit), bankruptcies and
p10/worst (the cliff), alongside the paired mean.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import market_maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

CELLS = ((2.0, 1.1), (2.0, 1.4), (2.5, 1.1), (2.5, 1.4))


def run(sessions: int, days: int) -> tuple[list[float], int, int]:
    pnls: list[float] = []
    bankruptcies = wins = 0
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.35, 0.5):
            for index in range(sessions):
                session = FieldSession(parameters, seed=1000 + index * 37, num_days=days,
                                       history_days=[60, 150, 400][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                result = session.run()
                pnls.append(float(result["all"]["SUBJECT"]))
                bankruptcies += 1 if result["bankrupt"] else 0
                wins += 1 if result["rank"] == 1 else 0
    return pnls, bankruptcies, wins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    base_rate = market_maker._RATE_ERROR_SCALE
    base_multiplier = market_maker._UNCERTAINTY_MULTIPLIER
    base, _, _ = run(args.sessions, args.days)

    print(f"\n{len(base)} paired sessions per cell; baseline rate={base_rate} "
          f"mult={base_multiplier}\n")
    print(f"{'rate':>6}{'mult':>7}{'mean':>9}{'median':>9}{'p10':>8}{'worst':>9}"
          f"{'>0':>7}{'win%':>7}{'bank':>6}{'d(base)':>9}{'t':>7}")
    print("-" * 84)
    for rate_scale, multiplier in CELLS:
        market_maker._RATE_ERROR_SCALE = rate_scale
        market_maker._UNCERTAINTY_MULTIPLIER = multiplier
        pnls, bankruptcies, wins = run(args.sessions, args.days)
        differences = [a - b for a, b in zip(pnls, base)]
        mean_difference = statistics.fmean(differences)
        error = (statistics.stdev(differences) / (len(differences) ** 0.5)
                 if len(differences) > 1 else 0.0)
        ordered = sorted(pnls)
        positive = sum(1 for value in pnls if value > 0) / len(pnls)
        marker = "  <-- incumbent" if (rate_scale == base_rate
                                       and multiplier == base_multiplier) else ""
        print(f"{rate_scale:>6.1f}{multiplier:>7.1f}{statistics.fmean(pnls):>9.2f}"
              f"{statistics.median(pnls):>9.2f}{ordered[int(0.1 * (len(ordered) - 1))]:>8.1f}"
              f"{min(pnls):>9.1f}{positive:>7.0%}{wins / len(pnls):>7.0%}{bankruptcies:>6}"
              f"{mean_difference:>+9.2f}"
              f"{(mean_difference / error if error else 0.0):>7.2f}{marker}", flush=True)

    market_maker._RATE_ERROR_SCALE = base_rate
    market_maker._UNCERTAINTY_MULTIPLIER = base_multiplier


if __name__ == "__main__":
    main()
