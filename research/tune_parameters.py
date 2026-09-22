"""Coordinate-wise tuning of the quoting/risk constants against the simulator.

Diagnostics showed FOK flow earning ~+0.054/contract against ~+0.0015 for RFQ in
the hardest field. That is structural, not luck: an RFQ is blind (we only win it
when we are the most aggressive, i.e. most likely wrong) whereas a FOK shows us
the price and lets us decline. So the sweep leans on the RFQ/FOK size split as
well as raw spread width.

Bankruptcy scores zero, so any config with a single bankruptcy is rejected
outright regardless of PnL.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import Market_Maker
from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS

TUNABLE = [
    "_BASE_HALF_SPREAD",
    "_UNCERTAINTY_MULTIPLIER",
    "_MAXIMUM_QUOTE_SIZE",
    "_TRADE_RISK_FRACTION",
    "_INVENTORY_SKEW",
    "_FOK_BASE_EDGE",
    "_FOK_RISK_FRACTION",
    "_MAXIMUM_POSITION_PER_OPTION",
]

CANDIDATES = {
    "_BASE_HALF_SPREAD": [0.010, 0.015, 0.020, 0.030, 0.040],
    "_UNCERTAINTY_MULTIPLIER": [0.8, 1.1, 1.4, 1.8, 2.2],
    "_MAXIMUM_QUOTE_SIZE": [8, 12, 18, 25],
    "_TRADE_RISK_FRACTION": [0.010, 0.020, 0.030, 0.050],
    "_INVENTORY_SKEW": [0.010, 0.025, 0.035, 0.060],
    "_FOK_BASE_EDGE": [0.005, 0.010, 0.020, 0.035],
    "_FOK_RISK_FRACTION": [0.030, 0.060, 0.100, 0.150],
    "_MAXIMUM_POSITION_PER_OPTION": [20, 40, 70],
}


def evaluate(sessions: int, days: int) -> dict[str, float]:
    per_field: dict[str, list[float]] = {name: [] for name in COMPETITOR_SETS}
    bankruptcies = 0
    total = 0
    for parameters in SCENARIOS.values():
        for field_name, competitors in COMPETITOR_SETS.items():
            for informed_fraction in (0.2, 0.5):
                for index in range(sessions):
                    session = Session(parameters, seed=1000 + index * 37, num_days=days,
                                      history_days=[60, 150, 400][index % 3],
                                      informed_fraction=informed_fraction, competitors=competitors)
                    result = session.run()
                    per_field[field_name].append(float(result["pnl"]))
                    bankruptcies += 1 if result["bankrupt"] else 0
                    total += 1
    everything = [pnl for values in per_field.values() for pnl in values]
    return {
        "mean": statistics.fmean(everything),
        "median": statistics.median(everything),
        "worst": min(everything),
        "hard": statistics.fmean(per_field["hard"]),
        "easy": statistics.fmean(per_field["easy"]),
        "bankrupt": bankruptcies,
        "total": total,
    }


def apply(config: dict[str, float]) -> None:
    for key, value in config.items():
        setattr(Market_Maker, key, value)


def score(metrics: dict[str, float]) -> float:
    if metrics["bankrupt"]:
        return -1e9  # a single bankruptcy is a zero on that test; never worth it
    # Weight the hard field: that is where competing market makers are strongest.
    return metrics["mean"] + metrics["hard"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--days", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    config = {name: getattr(Market_Maker, name) for name in TUNABLE}
    apply(config)
    best_metrics = evaluate(args.sessions, args.days)
    best_score = score(best_metrics)
    print(f"baseline  mean={best_metrics['mean']:+7.1f} hard={best_metrics['hard']:+7.1f} "
          f"easy={best_metrics['easy']:+7.1f} worst={best_metrics['worst']:+7.1f} "
          f"bankrupt={best_metrics['bankrupt']}/{best_metrics['total']}")
    print(f"          {config}")
    print("-" * 110)

    for round_index in range(args.rounds):
        improved = False
        for name in TUNABLE:
            for value in CANDIDATES[name]:
                if value == config[name]:
                    continue
                trial = dict(config)
                trial[name] = value
                apply(trial)
                metrics = evaluate(args.sessions, args.days)
                trial_score = score(metrics)
                marker = ""
                if trial_score > best_score:
                    best_score, config, best_metrics = trial_score, trial, metrics
                    improved = True
                    marker = "  <-- best"
                print(f"r{round_index} {name:<30}={value:<8} mean={metrics['mean']:+7.1f} "
                      f"hard={metrics['hard']:+7.1f} worst={metrics['worst']:+7.1f} "
                      f"bankrupt={metrics['bankrupt']}{marker}")
            apply(config)
        if not improved:
            break

    print("-" * 110)
    print(f"BEST  mean={best_metrics['mean']:+7.1f} hard={best_metrics['hard']:+7.1f} "
          f"easy={best_metrics['easy']:+7.1f} worst={best_metrics['worst']:+7.1f} "
          f"bankrupt={best_metrics['bankrupt']}/{best_metrics['total']}")
    for name in TUNABLE:
        print(f"  {name:<32} = {config[name]}")


if __name__ == "__main__":
    main()
