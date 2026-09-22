"""Is the negative RFQ edge an estimation-error story?

`diagnose_pnl` showed RFQ edge at -0.0053/contract. Our quoted half-spread is at
least 2c, so if our theo were exact every fill would earn that 2c. Earning -0.5c
instead means pricing error is eating the whole spread and then some.

Pricing error is driven by how much history we got. The competitors in the sim are
handed the TRUE parameters, so they have zero error by construction; we have to
estimate. If edge improves monotonically with `history_days`, the story is
confirmed -- and the fix is not a fatter constant spread but a spread that scales
with how little we know, which is exactly what `_price_uncertainty` is supposed to
do. Flat edge across history lengths would instead point at a structural flaw in
the quoting logic.

Also reported is realised absolute pricing error against the oracle, measured on
the options we actually traded, which pins down whether `_price_uncertainty` is
calibrated rather than merely directionally correct.
"""

import argparse
import statistics

from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS


def measure(history_days: int, sessions: int, days: int, field: str) -> dict[str, float]:
    competitors = COMPETITOR_SETS[field]
    rfq_edge = 0.0
    rfq_contracts = 0.0
    fok_edge = 0.0
    fok_contracts = 0.0
    pnls: list[float] = []
    errors: list[float] = []
    uncertainties: list[float] = []

    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.5):
            for index in range(sessions):
                session = Session(parameters, seed=1000 + index * 37, num_days=days,
                                  history_days=history_days, informed_fraction=informed_fraction,
                                  competitors=competitors)
                result = session.run()
                pnls.append(float(result["pnl"]))
                rfq_edge += session.diagnostics.get("rfq_edge", 0.0)
                rfq_contracts += session.diagnostics.get("rfq_contracts", 0.0)
                fok_edge += session.diagnostics.get("fok_edge", 0.0)
                fok_contracts += session.diagnostics.get("fok_contracts", 0.0)

                # Compare our theo against the oracle on the live book, and check
                # our own uncertainty estimate against that realised error.
                for option in session.options:
                    ours = session.subject.price_option(option)
                    truth = session.oracle.price_option_from_parameters(session.parameters, option)
                    errors.append(abs(ours - truth))
                    uncertainties.append(session.subject._price_uncertainty(option))

    return {
        "rfq_per_contract": rfq_edge / max(rfq_contracts, 1.0),
        "rfq_contracts": rfq_contracts / len(pnls),
        "fok_per_contract": fok_edge / max(fok_contracts, 1.0),
        "fok_contracts": fok_contracts / len(pnls),
        "pnl": statistics.fmean(pnls),
        "abs_error": statistics.fmean(errors) if errors else 0.0,
        "uncertainty": statistics.fmean(uncertainties) if uncertainties else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--days", type=int, default=20)
    parser.add_argument("--field", default="hard")
    args = parser.parse_args()

    print(f"field={args.field}")
    print(f"{'history':>9}{'pnl':>9}{'rfq/ct':>10}{'rfq ct':>9}{'fok/ct':>10}{'fok ct':>9}"
          f"{'|err|':>9}{'est unc':>9}{'ratio':>8}")
    print("-" * 82)
    for history_days in (40, 60, 120, 250, 600, 1500):
        metrics = measure(history_days, args.sessions, args.days, args.field)
        ratio = metrics["uncertainty"] / metrics["abs_error"] if metrics["abs_error"] else 0.0
        print(f"{history_days:>9}{metrics['pnl']:>9.1f}{metrics['rfq_per_contract']:>10.4f}"
              f"{metrics['rfq_contracts']:>9.0f}{metrics['fok_per_contract']:>10.4f}"
              f"{metrics['fok_contracts']:>9.0f}{metrics['abs_error']:>9.4f}"
              f"{metrics['uncertainty']:>9.4f}{ratio:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
