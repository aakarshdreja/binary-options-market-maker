"""Final config decision, on top of the corrected rate calibration.

Two things forced this re-run:

  1. `_RATE_ERROR_SCALE` was added after the corrected calibration diagnostic
     showed rate options were UNDER-padded (ratio 0.82) while company options sat
     near 2.0. That changes the quoted spread on ~30% of the book, so every size
     and FOK result measured before it is stale.
  2. The win-rate sweep returned d(win) = 0.000 for all ten candidates. Our
     per-session win/loss is decided by whether a rival detonates, not by our own
     marginal aggression -- the margin is lopsided in both directions. Win rate is
     therefore not tunable through these levers, which leaves mean PnL subject to
     solvency as the only thing we can actually move.

So this sweeps the rate scale itself against the size and FOK levers, paired on
identical seeds. `worst` and `bank` are reported alongside the mean because a
config that buys mean PnL with tail risk is a bad trade under a grader that pays
ZERO for a bankruptcy.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import market_maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS


def run(sessions: int, days: int) -> list[float]:
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
    run.bankruptcies = bankruptcies
    return pnls


# Round 2. The first round found the size lever to be the strongest and most
# consistent effect in the whole exercise (+0.46 at t=6.9), and -- against
# intuition -- it improved the worst session and the p10 rather than degrading
# them. That is the signature of a cap that binds on OUR side while the true
# risk limit is the per-trade collateral check, so raising it lets us take more
# of an edge we were already measuring as positive. An effect that large has to
# be walked out to where it stops paying, rather than accepted at the first
# value that happened to be on the grid.
CONFIGS: dict[str, dict[str, float]] = {
    "size 60/90 (base)": {"_MAXIMUM_QUOTE_SIZE": 60, "_MAXIMUM_POSITION_PER_OPTION": 90},
    "size 80/120": {"_MAXIMUM_QUOTE_SIZE": 80, "_MAXIMUM_POSITION_PER_OPTION": 120},
    "size 120/180": {"_MAXIMUM_QUOTE_SIZE": 120, "_MAXIMUM_POSITION_PER_OPTION": 180},
    "size 200/300": {"_MAXIMUM_QUOTE_SIZE": 200, "_MAXIMUM_POSITION_PER_OPTION": 300},
    "60/90 + fok": {"_MAXIMUM_QUOTE_SIZE": 60, "_MAXIMUM_POSITION_PER_OPTION": 90,
                    "_FOK_RISK_FRACTION": 0.250, "_FOK_BASE_EDGE": 0.006},
    "120/180 + fok": {"_MAXIMUM_QUOTE_SIZE": 120, "_MAXIMUM_POSITION_PER_OPTION": 180,
                      "_FOK_RISK_FRACTION": 0.250, "_FOK_BASE_EDGE": 0.006},
    "120/180 + fok + risk": {"_MAXIMUM_QUOTE_SIZE": 120, "_MAXIMUM_POSITION_PER_OPTION": 180,
                             "_FOK_RISK_FRACTION": 0.250, "_FOK_BASE_EDGE": 0.006,
                             "_TRADE_RISK_FRACTION": 0.050},
}

DEFAULTS = {key: getattr(market_maker, key)
            for config in CONFIGS.values() for key in config}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    for key, value in DEFAULTS.items():
        setattr(market_maker, key, value)
    base = run(args.sessions, args.days)

    print(f"\n{len(base)} paired sessions per config\n")
    print(f"{'config':<22}{'mean':>8}{'median':>8}{'worst':>9}{'p10':>8}"
          f"{'d(base)':>9}{'se':>7}{'t':>7}{'bank':>6}")
    print("-" * 84)

    for name, overrides in CONFIGS.items():
        for key, value in DEFAULTS.items():
            setattr(market_maker, key, value)
        for key, value in overrides.items():
            setattr(market_maker, key, value)

        pnls = run(args.sessions, args.days)
        differences = [a - b for a, b in zip(pnls, base)]
        mean_difference = statistics.fmean(differences)
        error = statistics.stdev(differences) / (len(differences) ** 0.5) if len(differences) > 1 else 0.0
        ordered = sorted(pnls)
        print(f"{name:<22}{statistics.fmean(pnls):>8.2f}{statistics.median(pnls):>8.2f}"
              f"{min(pnls):>9.1f}{ordered[int(0.1 * (len(ordered) - 1))]:>8.1f}"
              f"{mean_difference:>+9.2f}{error:>7.2f}"
              f"{(mean_difference / error if error else 0.0):>7.2f}"
              f"{run.bankruptcies:>6}", flush=True)

    for key, value in DEFAULTS.items():
        setattr(market_maker, key, value)


if __name__ == "__main__":
    main()
