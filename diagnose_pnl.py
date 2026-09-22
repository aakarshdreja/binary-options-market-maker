"""Where does the money actually go in the hard field?

`tune_focused` reports PnL but not its provenance. Measured edge per contract was
positive on both channels while session PnL was negative, and a gap that large is
either a ledger bug (already found and fixed once) or a genuine adverse-selection
cost that the edge statistic is blind to.

The edge statistic marks each fill against the TRUE price at the moment of the
fill, so it captures mispricing but not path risk. This splits the two apart:

    expected  = sum over fills of (true_price - traded_price) * signed_quantity
    realised  = terminal economic value - starting cash
    residual  = realised - expected      <- pure settlement variance

If `residual` averages near zero, our theo is the problem. If `expected` is
positive but small while `residual` is a persistent drag, we are being run over
by variance and need to trade smaller, not smarter.
"""

import argparse
import statistics

from simulate_session import COMPETITOR_SETS, Session
from validate_estimation import SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--days", type=int, default=25)
    parser.add_argument("--field", default="hard")
    args = parser.parse_args()

    competitors = COMPETITOR_SETS[args.field]
    rows: list[dict[str, float]] = []
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.5):
            for index in range(args.sessions):
                session = Session(parameters, seed=1000 + index * 37, num_days=args.days,
                                  history_days=[60, 150, 400][index % 3],
                                  informed_fraction=informed_fraction, competitors=competitors)
                result = session.run()
                diagnostics = session.diagnostics
                rfq_edge = diagnostics.get("rfq_edge", 0.0)
                fok_edge = diagnostics.get("fok_edge", 0.0)
                rows.append({
                    "informed": informed_fraction,
                    "pnl": float(result["pnl"]),
                    "rfq_edge": rfq_edge,
                    "fok_edge": fok_edge,
                    "rfq_contracts": diagnostics.get("rfq_contracts", 0.0),
                    "fok_contracts": diagnostics.get("fok_contracts", 0.0),
                    "residual": float(result["pnl"]) - rfq_edge - fok_edge,
                })

    def summarise(label: str, subset: list[dict[str, float]]) -> None:
        rfq_contracts = sum(r["rfq_contracts"] for r in subset)
        fok_contracts = sum(r["fok_contracts"] for r in subset)
        rfq_edge = sum(r["rfq_edge"] for r in subset)
        fok_edge = sum(r["fok_edge"] for r in subset)
        print(f"{label:<12}"
              f"pnl={statistics.fmean([r['pnl'] for r in subset]):+8.2f}  "
              f"rfq_edge={rfq_edge / len(subset):+8.2f} ({rfq_edge / max(rfq_contracts, 1):+.4f}/ct "
              f"over {rfq_contracts / len(subset):5.0f} ct)  "
              f"fok_edge={fok_edge / len(subset):+8.2f} ({fok_edge / max(fok_contracts, 1):+.4f}/ct "
              f"over {fok_contracts / len(subset):5.0f} ct)  "
              f"residual={statistics.fmean([r['residual'] for r in subset]):+8.2f} "
              f"(sd {statistics.pstdev([r['residual'] for r in subset]):6.2f})")

    print(f"field={args.field}  sessions={len(rows)}")
    print("-" * 150)
    summarise("informed .2", [r for r in rows if r["informed"] == 0.2])
    summarise("informed .5", [r for r in rows if r["informed"] == 0.5])
    summarise("all", rows)


if __name__ == "__main__":
    main()
