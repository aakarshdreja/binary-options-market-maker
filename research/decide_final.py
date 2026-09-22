"""Final decision: few candidates, many sessions, paired, on BOTH fields.

The field sweeps spread configs over about 0.7 of mean PnL, which is inside the
noise at 36 sessions -- the same trap that made rounds one and two crown settings
that later measured as harmful. So this narrows to a handful of candidates and
spends the compute on precision instead of breadth.

Two things make the answer trustworthy rather than merely favourable:

  1. PAIRING. Every candidate sees identical seeds, scenarios and rival draws, so
     differencing cancels the price path, which is the dominant variance term.
  2. BOTH FIELDS. We do not know what the real competition looks like. The
     "pessimistic" field hands rivals the TRUE parameters, so we are the only
     player who can be wrong. The "realistic" field gives rivals persistent
     estimation error, which is what actual participants will have. A setting
     that only wins in the friendly field is a bet on the field being friendly,
     and that is not a bet worth making for a single graded run.

Reported per field: mean PnL, the paired difference against the incumbent with
its standard error, mean rank, and the worst single session. Rank matters because
the prize is awarded on relative profitability, and the worst session matters
because a bankruptcy scores zero no matter how good the mean was.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import math
import statistics

import market_maker
from simulate_field import FIELD, FieldSession
from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS

CANDIDATES: dict[str, dict[str, float]] = {
    "incumbent": {"_BASE_HALF_SPREAD": 0.020, "_FOK_UNCERTAINTY_MULTIPLIER": 1.0},
    "tight rfq": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 1.0},
    "loose fok": {"_BASE_HALF_SPREAD": 0.020, "_FOK_UNCERTAINTY_MULTIPLIER": 0.4},
    "tight + loose fok": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 0.4},
    "tight + no fok unc": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 0.0},
}

DEFAULTS = {key: getattr(market_maker, key) for config in CANDIDATES.values() for key in config}


def evaluate(sessions: int, days: int, realistic: bool) -> tuple[list[float], list[float], int]:
    session_class = FieldSession if realistic else Session
    competitors = FIELD if realistic else COMPETITOR_SETS["hard"]
    pnls: list[float] = []
    ranks: list[float] = []
    bankruptcies = 0
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.35, 0.5):
            for index in range(sessions):
                session = session_class(parameters, seed=1000 + index * 37, num_days=days,
                                        history_days=[60, 150, 400][index % 3],
                                        informed_fraction=informed_fraction,
                                        competitors=competitors)
                result = session.run()
                pnls.append(float(result["pnl"]))
                ranks.append(float(result["rank"]))
                bankruptcies += 1 if result["bankrupt"] else 0
    return pnls, ranks, bankruptcies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    for realistic in (True, False):
        label = "REALISTIC field (rivals estimate)" if realistic else "PESSIMISTIC field (rivals omniscient)"
        print(f"\n{label}")
        print(f"{'candidate':<22}{'mean':>8}{'median':>8}{'worst':>9}{'rank':>7}"
              f"{'d(inc)':>9}{'se':>7}{'t':>7}{'bankrupt':>10}")
        print("-" * 87)
        incumbent: list[float] | None = None
        for name, overrides in CANDIDATES.items():
            for key, value in DEFAULTS.items():
                setattr(market_maker, key, value)
            for key, value in overrides.items():
                setattr(market_maker, key, value)

            pnls, ranks, bankruptcies = evaluate(args.sessions, args.days, realistic)
            if incumbent is None:
                incumbent = pnls
            differences = [a - b for a, b in zip(pnls, incumbent)]
            delta = statistics.fmean(differences)
            spread = statistics.pstdev(differences) if len(differences) > 1 else 0.0
            standard_error = spread / math.sqrt(len(differences)) if spread else 0.0
            t_statistic = delta / standard_error if standard_error else 0.0
            print(f"{name:<22}{statistics.fmean(pnls):>8.2f}{statistics.median(pnls):>8.2f}"
                  f"{min(pnls):>9.1f}{statistics.fmean(ranks):>7.2f}"
                  f"{delta:>9.2f}{standard_error:>7.2f}{t_statistic:>7.2f}"
                  f"{bankruptcies:>6}/{len(pnls):<4}", flush=True)


if __name__ == "__main__":
    main()
