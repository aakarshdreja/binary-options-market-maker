"""Round three: re-tune on top of the now-calibrated uncertainty estimate.

Rounds one and two were run against a `_price_uncertainty` that perturbed company
drift and volatility and nothing else. A pure rate option has no company leg, so
every one of those perturbations was a no-op and the function reported ~zero
uncertainty -- on 30% of the book, at any history length. `diagnose_uncertainty`
caught it: the estimate ran at 0.15x realised error on rate options at 60 days of
history, against 2.6x on single-company options.

Perturbing the rate lattice as well moved short-history RFQ edge from -0.0203 to
-0.0039 per contract at 60 days, and left the long-history number untouched at
+0.0073. That matters for this round: rounds one and two both crowned blanket
spread widening, but widening was only ever a crude proxy for "we are blind on
rate options". Now that the blindness is priced directly, a fat constant spread
may be pure cost, so `_BASE_HALF_SPREAD` is back on trial rather than assumed.

The remaining miscalibration is a uniform 2-3x over-conservatism across every
bucket, which is a single scalar, so `_UNCERTAINTY_MULTIPLIER` is swept too.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import math
import statistics
import sys

import Market_Maker
from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS

CONFIGS: dict[str, dict[str, float]] = {
    "baseline": {},
    # Is the fat spread still earning its keep now the blindness is priced?
    "wide 0.035": {"_BASE_HALF_SPREAD": 0.035},
    "narrow 0.012": {"_BASE_HALF_SPREAD": 0.012},
    # The uncertainty estimate is honest but ~2.5x conservative; how much of that
    # margin do we actually want to keep?
    "unc x1.0": {"_UNCERTAINTY_MULTIPLIER": 1.0},
    "unc x0.7": {"_UNCERTAINTY_MULTIPLIER": 0.7},
    "unc x1.0 + narrow": {"_UNCERTAINTY_MULTIPLIER": 1.0, "_BASE_HALF_SPREAD": 0.012},
    # Size and channel mix, the two robust findings from round two.
    "small size 8": {"_MAXIMUM_QUOTE_SIZE": 8},
    "fok heavy": {"_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120},
    "small + fok heavy": {"_MAXIMUM_QUOTE_SIZE": 8,
                          "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120},
    "round2 winner": {"_BASE_HALF_SPREAD": 0.035, "_MAXIMUM_QUOTE_SIZE": 8,
                      "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120,
                      "_INVENTORY_SKEW": 0.060, "_MAXIMUM_POSITION_PER_OPTION": 25},
    "r2 winner, unc x1.0": {"_BASE_HALF_SPREAD": 0.035, "_MAXIMUM_QUOTE_SIZE": 8,
                            "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120,
                            "_INVENTORY_SKEW": 0.060, "_MAXIMUM_POSITION_PER_OPTION": 25,
                            "_UNCERTAINTY_MULTIPLIER": 1.0},
    "r2 winner, base 0.020": {"_MAXIMUM_QUOTE_SIZE": 8,
                              "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120,
                              "_INVENTORY_SKEW": 0.060, "_MAXIMUM_POSITION_PER_OPTION": 25},
    "skew only": {"_INVENTORY_SKEW": 0.060, "_MAXIMUM_POSITION_PER_OPTION": 25},
}

DEFAULTS = {key: getattr(Market_Maker, key) for config in CONFIGS.values() for key in config}


def evaluate(sessions: int, days: int) -> tuple[list[float], list[str], int]:
    """Returns per-session PnL in a FIXED order so configs can be compared pairwise.

    Every config sees the identical set of seeds, scenarios and competitor fields,
    so the same underlying price path drives every one of them. Differencing
    against the baseline therefore cancels the path, which is the dominant term:
    session PnL has a standard deviation around 33, while the effects we are
    hunting are worth 1-3. Judging configs on unpaired means would be reading
    tea leaves.
    """
    pnls: list[float] = []
    labels: list[str] = []
    bankruptcies = 0
    for scenario_name, parameters in SCENARIOS.items():
        for field_name, competitors in COMPETITOR_SETS.items():
            for informed_fraction in (0.2, 0.5):
                for index in range(sessions):
                    session = Session(parameters, seed=1000 + index * 37, num_days=days,
                                      history_days=[60, 150, 400][index % 3],
                                      informed_fraction=informed_fraction, competitors=competitors)
                    result = session.run()
                    pnls.append(float(result["pnl"]))
                    labels.append(field_name)
                    bankruptcies += 1 if result["bankrupt"] else 0
    return pnls, labels, bankruptcies


def field_mean(pnls: list[float], labels: list[str], field: str) -> float:
    subset = [pnl for pnl, label in zip(pnls, labels) if label == field]
    return statistics.fmean(subset)


def main() -> None:
    sessions = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    baseline_pnls: list[float] | None = None
    print(f"{'config':<24}{'mean':>8}{'easy':>8}{'medium':>8}{'hard':>8}{'worst':>9}"
          f"{'d(base)':>10}{'se':>7}{'t':>7}{'bankrupt':>10}")
    print("-" * 101)
    for name, overrides in CONFIGS.items():
        for key, value in DEFAULTS.items():
            setattr(Market_Maker, key, value)
        for key, value in overrides.items():
            setattr(Market_Maker, key, value)

        pnls, labels, bankruptcies = evaluate(sessions, days)
        if baseline_pnls is None:
            baseline_pnls = pnls

        differences = [a - b for a, b in zip(pnls, baseline_pnls)]
        delta = statistics.fmean(differences)
        # Sessions repeat the same paths across fields and informed fractions, so
        # they are not independent draws; treat the count as an optimistic bound
        # and read `t` as an upper limit on the true significance.
        spread = statistics.pstdev(differences) if len(differences) > 1 else 0.0
        standard_error = spread / math.sqrt(len(differences)) if spread else 0.0
        t_statistic = delta / standard_error if standard_error else 0.0

        print(f"{name:<24}{statistics.fmean(pnls):>8.1f}"
              f"{field_mean(pnls, labels, 'easy'):>8.1f}"
              f"{field_mean(pnls, labels, 'medium'):>8.1f}"
              f"{field_mean(pnls, labels, 'hard'):>8.1f}"
              f"{min(pnls):>9.1f}{delta:>10.2f}{standard_error:>7.2f}{t_statistic:>7.2f}"
              f"{bankruptcies:>6}/{len(pnls):<4}", flush=True)


if __name__ == "__main__":
    main()
