"""Do the sweep's "candidates" survive flow big enough to actually hurt us?

`reverify_constants.py` flagged seven changes at t > 2 with no bankruptcies, and
every one of them loosens a risk control: bigger positions, a thinner safety
buffer, a smaller tenor penalty, more capital per trade, and no inventory skew at
all. That unanimity is the tell. A sweep does not usually agree that every guard
rail is too tight -- unless the environment never tests one.

It does not. The session simulator draws RFQ clips from randint(1, 15) and FOK
clips from randint(1, 12) against a $1,000 book, so `_MAXIMUM_QUOTE_SIZE`,
`_FOK_RISK_FRACTION` and `_FOK_UNPROVEN_RISK_SCALE` measured EXACTLY +0.00 across
every value tried -- they never bind even once. In a world where no single trade
can hurt you, protection is pure cost and the sweep correctly reports that
removing it pays. That tells us nothing about a venue whose clip sizes we do not
know, and the real grader never states them.

So scale the flow and re-ask. The rng stream is untouched -- only the drawn
quantity is multiplied -- so price paths and rival behaviour stay identical and
the comparison remains properly paired. If the gains hold at 4x and 10x, they are
real and worth taking. If they inverta, the incumbents are doing exactly the job
they were put there for, and the sweep was measuring an environment too gentle to
price them.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import market_maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

# The flagged candidates, plus the incumbent value for each.
CANDIDATES: tuple[tuple[str, float, float], ...] = (
    ("_INVENTORY_SKEW", 0.035, 0.0),
    ("_INVENTORY_SKEW", 0.035, 0.02),
    ("_MAXIMUM_POSITION_PER_OPTION", 180, 260),
    ("_TRADE_RISK_FRACTION", 0.030, 0.045),
    ("_SAFETY_BUFFER_FRACTION", 0.150, 0.05),
    ("_EXPIRY_CAPITAL_PENALTY", 0.200, 0.10),
)


class ScaledFlow:
    """Proxy rng that multiplies only the clip-size draws.

    Scaling the RETURN value rather than consuming extra randomness leaves the
    stream identical, so every price path, rival quote and order arrival is
    unchanged between arms. Only the size of each trade moves.
    """

    def __init__(self, inner, scale: int) -> None:
        self._inner = inner
        self._scale = scale

    def randint(self, low: int, high: int) -> int:
        value = self._inner.randint(low, high)
        # (1, 15) is the RFQ clip and (1, 12) the FOK clip in `Session`; the
        # day-loop counts use other ranges and must not be touched.
        if (low, high) in ((1, 15), (1, 12)):
            return value * self._scale
        return value

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run(scale: int, sessions: int, days: int) -> tuple[list[float], int, int]:
    pnls: list[float] = []
    bankruptcies = wins = 0
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.35, 0.5):
            for index in range(sessions):
                session = FieldSession(parameters, seed=1000 + index * 37, num_days=days,
                                       history_days=[60, 150, 400][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                if scale != 1:
                    session.rng = ScaledFlow(session.rng, scale)
                result = session.run()
                pnls.append(float(result["all"]["SUBJECT"]))
                bankruptcies += 1 if result["bankrupt"] else 0
                wins += 1 if result["rank"] == 1 else 0
    return pnls, bankruptcies, wins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    for scale in (1, 4, 10):
        base, base_bankrupt, base_wins = run(scale, args.sessions, args.days)
        ordered = sorted(base)
        print(f"\n=== flow x{scale}  (RFQ clips 1-{15 * scale}, FOK clips 1-{12 * scale}) ===")
        print(f"incumbent: mean {statistics.fmean(base):.2f}  p10 "
              f"{ordered[int(0.1 * (len(ordered) - 1))]:.1f}  worst {min(base):.1f}  "
              f"win {base_wins / len(base):.0%}  bankrupt {base_bankrupt}/{len(base)}")
        print(f"{'constant':<30}{'value':>8}{'mean':>9}{'p10':>8}{'worst':>9}"
              f"{'win%':>7}{'bank':>6}{'d':>8}{'t':>7}")
        for name, incumbent, candidate in CANDIDATES:
            setattr(market_maker, name, candidate)
            pnls, bankruptcies, wins = run(scale, args.sessions, args.days)
            setattr(market_maker, name, incumbent)
            differences = [a - b for a, b in zip(pnls, base)]
            mean_difference = statistics.fmean(differences)
            error = (statistics.stdev(differences) / (len(differences) ** 0.5)
                     if len(differences) > 1 else 0.0)
            ordered = sorted(pnls)
            verdict = ""
            if bankruptcies > base_bankrupt:
                verdict = "  <-- ADDS BANKRUPTCIES"
            elif mean_difference < 0:
                verdict = "  <-- inverts"
            print(f"{name:<30}{candidate:>8}{statistics.fmean(pnls):>9.2f}"
                  f"{ordered[int(0.1 * (len(ordered) - 1))]:>8.1f}{min(pnls):>9.1f}"
                  f"{wins / len(pnls):>7.0%}{bankruptcies:>6}{mean_difference:>+8.2f}"
                  f"{(mean_difference / error if error else 0.0):>7.2f}{verdict}", flush=True)


if __name__ == "__main__":
    main()
