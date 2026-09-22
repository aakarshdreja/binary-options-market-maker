"""Does dropping inventory skew survive an adversary that hunts one side?

`_INVENTORY_SKEW = 0.0` was the one candidate that improved mean, p10 AND worst
at every flow scale tested (x1, x4, x10). That is a real result and it deserves a
real hearing rather than a reflex "never remove a risk control".

But the flow-scale sweep varies SIZE, not INTENT, and inventory skew does not
exist to protect against large trades -- it exists to protect against a
counterparty that keeps hitting the same side. Leaning quotes against inventory is
what stops a maker being filled long, long, long by someone who knows the option
is going the other way. A field of mostly-uninformed flow buys from us as often as
it sells, so inventory mean-reverts on its own and the skew is pure cost. That is
the world the sweep measured, at all three scales.

So vary intent instead. This runs mostly-informed flow (50-80%), where fills are
adverse by construction and one-sided accumulation is the whole risk, and it
reports the concentration directly -- the largest absolute position held in any
single option -- alongside PnL. If skew=0 holds up on tail and concentration here
too, it is a genuine improvement and the incumbent was costing us. If the tail or
the concentration blows out, the sweep was measuring a field too friendly to
price it, and 0.035 stays.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import market_maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS
from large_flow_check import ScaledFlow

VALUES = (0.0, 0.02, 0.035, 0.06)


def run(scale: int, sessions: int, days: int) -> tuple[list[float], int, int, int]:
    pnls: list[float] = []
    bankruptcies = wins = 0
    peak_concentration = 0
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.5, 0.65, 0.8):
            for index in range(sessions):
                session = FieldSession(parameters, seed=4200 + index * 37, num_days=days,
                                       history_days=[60, 60, 150][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                if scale != 1:
                    session.rng = ScaledFlow(session.rng, scale)
                result = session.run()
                pnls.append(float(result["all"]["SUBJECT"]))
                bankruptcies += 1 if result["bankrupt"] else 0
                wins += 1 if result["rank"] == 1 else 0
                quantities = session.subject.position.option_quantity_by_option_id.values()
                peak_concentration = max(peak_concentration,
                                         max((abs(q) for q in quantities), default=0))
    return pnls, bankruptcies, wins, peak_concentration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    incumbent = market_maker._INVENTORY_SKEW
    for scale in (1, 4):
        print(f"\n=== adversarial field (50-80% informed), flow x{scale}, "
              f"{args.days}-day sessions ===")
        print(f"{'skew':>7}{'mean':>9}{'median':>9}{'p10':>9}{'worst':>9}"
              f"{'>0':>7}{'win%':>7}{'bank':>6}{'peak pos':>10}{'d':>8}{'t':>7}")
        base: list[float] | None = None
        for value in VALUES:
            market_maker._INVENTORY_SKEW = value
            pnls, bankruptcies, wins, concentration = run(scale, args.sessions, args.days)
            if value == incumbent:
                pass
            if base is None and value == VALUES[0]:
                pass
            ordered = sorted(pnls)
            positive = sum(1 for v in pnls if v > 0) / len(pnls)
            if value == incumbent:
                base = pnls
            print(f"{value:>7}{statistics.fmean(pnls):>9.2f}{statistics.median(pnls):>9.2f}"
                  f"{ordered[int(0.1 * (len(ordered) - 1))]:>9.1f}{min(pnls):>9.1f}"
                  f"{positive:>7.0%}{wins / len(pnls):>7.0%}{bankruptcies:>6}"
                  f"{concentration:>10}"
                  + ("      --     --  <-- incumbent" if value == incumbent else ""),
                  flush=True)
            if value != incumbent:
                globals().setdefault("_pending", {})[value] = pnls
        # paired stats against the incumbent, now that we have it
        assert base is not None
        for value, pnls in globals().get("_pending", {}).items():
            differences = [a - b for a, b in zip(pnls, base)]
            mean_difference = statistics.fmean(differences)
            error = statistics.stdev(differences) / (len(differences) ** 0.5)
            print(f"   skew={value:<6} vs incumbent: d={mean_difference:+.2f}  "
                  f"t={mean_difference / error if error else 0.0:+.2f}")
        globals()["_pending"] = {}

    market_maker._INVENTORY_SKEW = incumbent


if __name__ == "__main__":
    main()
