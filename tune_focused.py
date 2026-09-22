"""Focused A/B of quoting configs, driven by the diagnostic finding.

Measured edge per contract in the hardest field: FOK +0.054, RFQ +0.0015. That
asymmetry is structural. An RFQ is blind -- we cannot decline it, and we only win
it by being the most aggressive quote, which is precisely when we are most likely
wrong (winner's curse). A FOK shows us side, price and size up front and lets us
say no. So the hypothesis under test is: shift capital from RFQ to FOK.

Rather than brute-force every knob, each config below encodes one idea.
"""

import statistics
import sys

import Market_Maker
from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS

CONFIGS: dict[str, dict[str, float]] = {
    "baseline": {},
    "shift to FOK": {"_MAXIMUM_QUOTE_SIZE": 12, "_TRADE_RISK_FRACTION": 0.015, "_FOK_RISK_FRACTION": 0.120},
    "shift to FOK hard": {"_MAXIMUM_QUOTE_SIZE": 8, "_TRADE_RISK_FRACTION": 0.010, "_FOK_RISK_FRACTION": 0.150},
    "wider rfq": {"_BASE_HALF_SPREAD": 0.035},
    "tighter rfq": {"_BASE_HALF_SPREAD": 0.012},
    "less uncertainty": {"_UNCERTAINTY_MULTIPLIER": 1.0},
    "more uncertainty": {"_UNCERTAINTY_MULTIPLIER": 2.0},
    "picky fok": {"_FOK_BASE_EDGE": 0.030},
    "greedy fok": {"_FOK_BASE_EDGE": 0.005},
    "more skew": {"_INVENTORY_SKEW": 0.060},
    "combo A": {"_MAXIMUM_QUOTE_SIZE": 12, "_TRADE_RISK_FRACTION": 0.015, "_FOK_RISK_FRACTION": 0.120,
                "_BASE_HALF_SPREAD": 0.030},
    "combo B": {"_MAXIMUM_QUOTE_SIZE": 12, "_TRADE_RISK_FRACTION": 0.015, "_FOK_RISK_FRACTION": 0.120,
                "_BASE_HALF_SPREAD": 0.030, "_UNCERTAINTY_MULTIPLIER": 1.8, "_INVENTORY_SKEW": 0.060},
}

DEFAULTS = {key: getattr(Market_Maker, key) for config in CONFIGS.values() for key in config}


def evaluate(sessions: int, days: int) -> dict[str, float]:
    per_field: dict[str, list[float]] = {name: [] for name in COMPETITOR_SETS}
    bankruptcies = 0
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
    everything = [pnl for values in per_field.values() for pnl in values]
    return {
        "mean": statistics.fmean(everything),
        "median": statistics.median(everything),
        "worst": min(everything),
        "easy": statistics.fmean(per_field["easy"]),
        "medium": statistics.fmean(per_field["medium"]),
        "hard": statistics.fmean(per_field["hard"]),
        "bankrupt": bankruptcies,
        "total": len(everything),
    }


def main() -> None:
    sessions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    print(f"{'config':<20}{'mean':>8}{'median':>8}{'easy':>8}{'medium':>8}{'hard':>8}{'worst':>9}{'bankrupt':>10}")
    print("-" * 79)
    for name, overrides in CONFIGS.items():
        for key, value in DEFAULTS.items():
            setattr(Market_Maker, key, value)
        for key, value in overrides.items():
            setattr(Market_Maker, key, value)
        metrics = evaluate(sessions, days)
        print(f"{name:<20}{metrics['mean']:>8.1f}{metrics['median']:>8.1f}{metrics['easy']:>8.1f}"
              f"{metrics['medium']:>8.1f}{metrics['hard']:>8.1f}{metrics['worst']:>9.1f}"
              f"{metrics['bankrupt']:>6}/{metrics['total']:<4}", flush=True)


if __name__ == "__main__":
    main()
