"""Re-verify every tuned constant against a field that no longer moves under us.

Almost all of these were chosen while `simulate_field.bind` seeded rivals with
`hash(self.name)`, which Python randomises per process. Cells inside a single
sweep stayed paired against each other, so each result was honest about its own
run -- but every run drew a DIFFERENT set of competitors, so each conclusion
generalised only to the one field it happened to sample. Two have already been
caught that way: `_UNCERTAINTY_MULTIPLIER` (a "+0.90, t=2.79" that evaporated to
+0.10, t=0.19) and the FOK counterparty filter.

Only three constants have been re-checked since the seeding was fixed. This
re-runs the rest, one at a time, paired against the incumbent on identical seeds.

Reported alongside the paired mean difference are the terms the grader actually
pays: rank-1 rate (full credit), bankruptcies (zero credit), and p10/worst (how
close we come to that cliff). A change is only worth making on a clear,
significant, tail-safe improvement -- with a day left, the cost of a false
positive is much higher than the cost of leaving a marginal gain on the table.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import market_maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

# name -> values to try. Incumbent is read live and always included.
SWEEPS: dict[str, tuple[float, ...]] = {
    "_RATE_ERROR_SCALE": (1.5, 2.0, 2.5, 3.0),
    "_MAXIMUM_HALF_SPREAD": (0.15, 0.25, 0.35),
    "_MAXIMUM_PRICE_UNCERTAINTY": (0.10, 0.15, 0.22),
    "_INVENTORY_SKEW": (0.0, 0.02, 0.035, 0.06, 0.10),
    "_MAXIMUM_POSITION_PER_OPTION": (120, 180, 260),
    "_MAXIMUM_QUOTE_SIZE": (80, 120, 180),
    "_TRADE_RISK_FRACTION": (0.020, 0.030, 0.045, 0.060),
    "_SAFETY_BUFFER_FRACTION": (0.05, 0.15, 0.25),
    "_EXPIRY_CAPITAL_PENALTY": (0.10, 0.20, 0.35),
    "_FOK_BASE_EDGE": (0.003, 0.006, 0.012, 0.020),
    "_FOK_RISK_FRACTION": (0.15, 0.25, 0.40),
    "_FOK_UNCERTAINTY_MULTIPLIER": (0.5, 1.0, 1.5),
    "_FOK_UNPROVEN_RISK_SCALE": (0.06, 0.12, 0.25),
}


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
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    base, base_bankrupt, base_wins = run(args.sessions, args.days)
    ordered = sorted(base)
    print(f"\nincumbent over {len(base)} sessions: mean {statistics.fmean(base):.2f}  "
          f"median {statistics.median(base):.2f}  p10 {ordered[int(0.1 * (len(ordered) - 1))]:.1f}  "
          f"worst {min(base):.1f}  win {base_wins / len(base):.0%}  bank {base_bankrupt}")
    print("a change is only worth making on a clear, significant, tail-safe gain\n")

    flagged: list[str] = []
    for name, values in SWEEPS.items():
        incumbent = getattr(market_maker, name)
        print(f"{name}  (incumbent {incumbent})")
        print(f"{'value':>10}{'mean':>9}{'median':>9}{'p10':>8}{'worst':>9}"
              f"{'>0':>7}{'win%':>7}{'bank':>6}{'d(base)':>9}{'t':>7}")
        for value in values:
            setattr(market_maker, name, value)
            pnls, bankruptcies, wins = run(args.sessions, args.days)
            differences = [a - b for a, b in zip(pnls, base)]
            mean_difference = statistics.fmean(differences)
            error = (statistics.stdev(differences) / (len(differences) ** 0.5)
                     if len(differences) > 1 else 0.0)
            t_statistic = mean_difference / error if error else 0.0
            ordered = sorted(pnls)
            p10 = ordered[int(0.1 * (len(ordered) - 1))]
            positive = sum(1 for v in pnls if v > 0) / len(pnls)
            marker = "  <-- incumbent" if value == incumbent else ""
            if t_statistic > 2.0 and bankruptcies == 0 and value != incumbent:
                marker = "  ** CANDIDATE **"
                flagged.append(f"{name}={value}  d={mean_difference:+.2f} t={t_statistic:+.2f}")
            print(f"{value:>10}{statistics.fmean(pnls):>9.2f}{statistics.median(pnls):>9.2f}"
                  f"{p10:>8.1f}{min(pnls):>9.1f}{positive:>7.0%}{wins / len(pnls):>7.0%}"
                  f"{bankruptcies:>6}{mean_difference:>+9.2f}{t_statistic:>7.2f}{marker}",
                  flush=True)
        setattr(market_maker, name, incumbent)
        print()

    print("-" * 84)
    if flagged:
        print("candidates worth a closer look (t > 2, no bankruptcies):")
        for entry in flagged:
            print("  ", entry)
    else:
        print("no constant beat its incumbent significantly -- the committed set survives")


if __name__ == "__main__":
    main()
