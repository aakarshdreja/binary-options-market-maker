"""Final head-to-head standings against the realistic field.

`simulate_field.py` sweeps configs and reports only our own row. This reports the
whole scoreboard -- every player's mean, median and worst -- because the prize is
awarded on relative profitability, so what matters is not our absolute PnL but
where we sit in the ordering and how often we finish above the strongest rival.

`sharp` is the one to watch: it is a rival running the same lattice pricer we do,
handicapped only by persistent parameter error. If we cannot beat that, our edge
is the estimator rather than the strategy.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import statistics

from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

SESSIONS = 12
DAYS = 25


def main() -> None:
    names = ["SUBJECT"] + [entry[0] for entry in FIELD]
    pnls: dict[str, list[float]] = {name: [] for name in names}
    ranks: list[float] = []
    wins = beat_sharp = bankruptcies = 0

    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.35, 0.5):
            for index in range(SESSIONS):
                session = FieldSession(parameters, seed=1000 + index * 37, num_days=DAYS,
                                       history_days=[60, 150, 400][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                result = session.run()
                for name, value in result["all"].items():
                    pnls[name].append(float(value))
                ranks.append(float(result["rank"]))
                wins += 1 if result["rank"] == 1 else 0
                beat_sharp += 1 if result["all"]["SUBJECT"] > result["all"]["sharp"] else 0
                bankruptcies += 1 if result["bankrupt"] else 0

    total = len(ranks)
    print(f"\n{total} sessions per player, identical seeds and price paths for all\n")
    print(f"{'player':<14}{'mean':>9}{'median':>9}{'worst':>9}{'best':>9}{'>0':>7}")
    print("-" * 57)
    for name in sorted(names, key=lambda key: -statistics.fmean(pnls[key])):
        values = pnls[name]
        positive = sum(1 for value in values if value > 0) / len(values)
        marker = "  <-- us" if name == "SUBJECT" else ""
        print(f"{name:<14}{statistics.fmean(values):>9.2f}{statistics.median(values):>9.2f}"
              f"{min(values):>9.1f}{max(values):>9.1f}{positive:>6.0%}{marker}")
    print("-" * 57)
    print(f"our mean rank {statistics.fmean(ranks):.2f} of {len(names)}   "
          f"outright wins {wins / total:.0%}   beat 'sharp' {beat_sharp / total:.0%}   "
          f"bankrupt {bankruptcies}/{total}")


if __name__ == "__main__":
    main()
