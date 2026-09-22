"""Round two: the RFQ channel is a variance pump, so shrink it and feed the FOK.

`diagnose_pnl` measured, in the hard field, RFQ edge of -0.0053/contract over ~692
contracts a session against FOK edge of +0.0467/contract over ~97. The RFQ number
also degrades monotonically with the informed fraction (-0.0022 at 0.2, -0.0098 at
0.5), which is the fingerprint of adverse selection rather than noise: we only win
a blind auction when we are the most aggressive quote, and against competitors who
price perfectly that means we win exactly when we are wrong.

Round one confirmed the direction -- "wider rfq" was the single best knob and
"picky fok" was one of the worst, i.e. show less on the blind channel and take
more of the one we get to inspect. This round pushes both further and, crucially,
attacks SIZE as well as price. Seven hundred contracts a session at zero edge is
not a business, it is a lottery ticket, and the session standard deviation (~33)
dwarfs the total expected edge (~+0.9).
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import math
import statistics
import sys

import market_maker
from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS

CONFIGS: dict[str, dict[str, float]] = {
    "baseline": {},
    "wide 0.035": {"_BASE_HALF_SPREAD": 0.035},
    "wide 0.050": {"_BASE_HALF_SPREAD": 0.050},
    "small size 10": {"_MAXIMUM_QUOTE_SIZE": 10},
    "small size 6": {"_MAXIMUM_QUOTE_SIZE": 6},
    "wide + small": {"_BASE_HALF_SPREAD": 0.035, "_MAXIMUM_QUOTE_SIZE": 8},
    "wide + tiny": {"_BASE_HALF_SPREAD": 0.050, "_MAXIMUM_QUOTE_SIZE": 5},
    "fok heavy": {"_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120},
    "wide + fok heavy": {"_BASE_HALF_SPREAD": 0.035, "_MAXIMUM_QUOTE_SIZE": 8,
                         "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120},
    "wide + fok heavy 2": {"_BASE_HALF_SPREAD": 0.050, "_MAXIMUM_QUOTE_SIZE": 5,
                           "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120},
    "wide + fok + skew": {"_BASE_HALF_SPREAD": 0.035, "_MAXIMUM_QUOTE_SIZE": 8,
                          "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120,
                          "_INVENTORY_SKEW": 0.060, "_MAXIMUM_POSITION_PER_OPTION": 25},
    "wide + fok + tight unc": {"_BASE_HALF_SPREAD": 0.035, "_MAXIMUM_QUOTE_SIZE": 8,
                               "_FOK_BASE_EDGE": 0.010, "_FOK_RISK_FRACTION": 0.120,
                               "_UNCERTAINTY_MULTIPLIER": 1.0},
}

DEFAULTS = {key: getattr(market_maker, key) for config in CONFIGS.values() for key in config}


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
            setattr(market_maker, key, value)
        for key, value in overrides.items():
            setattr(market_maker, key, value)

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
