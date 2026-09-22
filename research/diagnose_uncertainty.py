"""Is `_price_uncertainty` calibrated per option TYPE, not just on average?

`diagnose_history` showed the estimate running at 0.88x realised error on short
histories and 2.3x on long ones. An average that swings that far usually means two
populations are being blended, so this splits the book three ways:

    rate    -- the FED leg only
    single  -- one company
    spread  -- AJR vs THR

The suspicion is specific. `_price_uncertainty` perturbs company drift and
volatility, and nothing else. A pure rate option has no company leg, so every one
of those perturbations is a no-op and the function returns ~0 -- we quote rate
options at the bare base spread with no padding at all, however little history we
have seen. Meanwhile the rate lattice depends on `rate_up_probability`,
`rate_down_probability`, `rate_reversion_strength` and `rate_target`, all of which
are estimated from a handful of observed moves.

A `ratio` near 1.0 means honest. Well under 1.0 means we are quoting tighter than
our own ignorance justifies, which is how you lose money while looking profitable.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

from Market_Maker import FED_FUNDS_RATE_UNDERLYING_ID, BinaryOption
from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS


def classify(option: BinaryOption) -> str:
    company_legs = [leg for leg in option.legs if leg.underlying_id != FED_FUNDS_RATE_UNDERLYING_ID]
    if not company_legs:
        return "rate"
    return "single" if len(company_legs) == 1 else "spread"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--days", type=int, default=15)
    args = parser.parse_args()

    print(f"{'history':>8}{'kind':>9}{'n':>7}{'|err|':>9}{'est unc':>9}{'ratio':>8}{'p90 err':>9}")
    print("-" * 59)
    for history_days in (40, 60, 120, 250, 600):
        buckets: dict[str, list[tuple[float, float]]] = {"rate": [], "single": [], "spread": []}
        for parameters in SCENARIOS.values():
            for index in range(args.sessions):
                session = Session(parameters, seed=1000 + index * 37, num_days=args.days,
                                  history_days=history_days, informed_fraction=0.3,
                                  competitors=COMPETITOR_SETS["hard"])
                session.run()
                for option in session.options:
                    ours = session.subject.price_option(option)
                    truth = session.oracle.price_option_from_parameters(session.parameters, option)
                    buckets[classify(option)].append(
                        (abs(ours - truth), session.subject._price_uncertainty(option))
                    )
        for kind, rows in buckets.items():
            if not rows:
                continue
            errors = sorted(error for error, _ in rows)
            mean_error = statistics.fmean(errors)
            mean_uncertainty = statistics.fmean([unc for _, unc in rows])
            ratio = mean_uncertainty / mean_error if mean_error else float("inf")
            print(f"{history_days:>8}{kind:>9}{len(rows):>7}{mean_error:>9.4f}"
                  f"{mean_uncertainty:>9.4f}{ratio:>8.2f}{errors[int(0.9 * (len(errors) - 1))]:>9.4f}",
                  flush=True)
        print()


if __name__ == "__main__":
    main()
