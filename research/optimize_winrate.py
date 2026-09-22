"""Optimise for the objective the grader actually uses.

The problem statement is explicit: "You will receive full credit for being the
top market maker in the session by PnL, zero credit for going bankrupt (or code
errors), and partial credit for avoiding bankruptcy."

Every sweep before this one ranked configs by MEAN PnL. That is the wrong
objective, and not by a subtle margin. Mean PnL rewards being reliably a bit
better than the field; the grader rewards finishing FIRST in a given session and
pays nothing extra for winning by a lot. Those come apart badly when the rivals
are high variance: we currently carry the best mean in the field by a wide margin
(+6.6 against -3.8 for the next best) yet win outright only ~27% of sessions,
because a wild rival's good draw beats our steady one even though the same rival
detonates in the sessions we do win.

So this scores three things per config:

  win        P(strictly highest PnL in the session)  -- full credit
  solvent    P(not bankrupt)                          -- partial credit
  score      a blended proxy, win + PARTIAL * (solvent - win)

PARTIAL is unknown, so the sweep reports win and solvency separately and the
blend at a plausible 0.5; any config that improves both is safe regardless of the
true weight, and one that trades solvency for wins is flagged rather than
crowned.

Comparisons are PAIRED -- every config sees identical seeds, price paths, order
flow and rival parameter draws -- because unpaired session noise (sd ~33) buries
effects of size 1-3. The paired standard error on the win-rate difference is
computed from the per-session indicator differences, so `t` is meaningful.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import statistics

import Market_Maker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

PARTIAL = 0.5


def run(sessions: int, days: int) -> dict[str, object]:
    """Returns per-session outcome vectors so configs can be differenced pairwise."""
    wins: list[float] = []
    solvent: list[float] = []
    pnls: list[float] = []
    ranks: list[float] = []

    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.35, 0.5):
            for index in range(sessions):
                session = FieldSession(parameters, seed=1000 + index * 37, num_days=days,
                                       history_days=[60, 150, 400][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                result = session.run()
                wins.append(1.0 if result["rank"] == 1 else 0.0)
                solvent.append(0.0 if result["bankrupt"] else 1.0)
                pnls.append(float(result["pnl"]))
                ranks.append(float(result["rank"]))

    return {"wins": wins, "solvent": solvent, "pnls": pnls, "ranks": ranks}


def blended(outcome: dict[str, object]) -> list[float]:
    return [win + PARTIAL * (alive - win)
            for win, alive in zip(outcome["wins"], outcome["solvent"])]


CONFIGS: dict[str, dict[str, float]] = {
    "incumbent": {},
    # More size on the same positive edge: scales both our lead and our variance.
    "size 40/60": {"_MAXIMUM_QUOTE_SIZE": 40, "_MAXIMUM_POSITION_PER_OPTION": 60},
    "size 60/90": {"_MAXIMUM_QUOTE_SIZE": 60, "_MAXIMUM_POSITION_PER_OPTION": 90},
    # FOK carries ~7x the edge per contract of RFQ (0.051 vs 0.007) and can be
    # declined, so it is the cheapest place to buy variance.
    "fok risk 0.18": {"_FOK_RISK_FRACTION": 0.180},
    "fok risk 0.25": {"_FOK_RISK_FRACTION": 0.250},
    "fok edge 0.006": {"_FOK_BASE_EDGE": 0.006},
    "fok heavy": {"_FOK_RISK_FRACTION": 0.250, "_FOK_BASE_EDGE": 0.006},
    "trade risk 0.05": {"_TRADE_RISK_FRACTION": 0.050},
    "size + fok": {"_MAXIMUM_QUOTE_SIZE": 40, "_MAXIMUM_POSITION_PER_OPTION": 60,
                   "_FOK_RISK_FRACTION": 0.250, "_FOK_BASE_EDGE": 0.006},
    "all in": {"_MAXIMUM_QUOTE_SIZE": 60, "_MAXIMUM_POSITION_PER_OPTION": 90,
               "_FOK_RISK_FRACTION": 0.250, "_FOK_BASE_EDGE": 0.006,
               "_TRADE_RISK_FRACTION": 0.050},
}

DEFAULTS = {key: getattr(Market_Maker, key) for config in CONFIGS.values() for key in config}


def paired(current: list[float], base: list[float]) -> tuple[float, float, float]:
    differences = [a - b for a, b in zip(current, base)]
    mean = statistics.fmean(differences)
    if len(differences) < 2:
        return mean, 0.0, 0.0
    error = statistics.stdev(differences) / (len(differences) ** 0.5)
    return mean, error, (mean / error if error > 0 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    for key, value in DEFAULTS.items():
        setattr(Market_Maker, key, value)
    base = run(args.sessions, args.days)
    base_blend = blended(base)
    total = len(base["wins"])

    print(f"\n{total} paired sessions per config   "
          f"(blend = win + {PARTIAL} x (solvent - win))\n")
    print(f"{'config':<18}{'win%':>7}{'d(win)':>9}{'t':>7}{'mean':>8}{'d(mean)':>9}"
          f"{'t':>7}{'rank':>7}{'bank':>6}{'blend':>8}{'t':>7}")
    print("-" * 93)

    for name, overrides in CONFIGS.items():
        for key, value in DEFAULTS.items():
            setattr(Market_Maker, key, value)
        for key, value in overrides.items():
            setattr(Market_Maker, key, value)

        outcome = run(args.sessions, args.days)
        win_rate = statistics.fmean(outcome["wins"])
        mean_pnl = statistics.fmean(outcome["pnls"])
        bankrupt = int(total - sum(outcome["solvent"]))
        d_win, _, t_win = paired(outcome["wins"], base["wins"])
        d_pnl, _, t_pnl = paired(outcome["pnls"], base["pnls"])
        d_blend, _, t_blend = paired(blended(outcome), base_blend)

        print(f"{name:<18}{win_rate:>6.0%}{d_win:>+9.3f}{t_win:>7.2f}"
              f"{mean_pnl:>8.2f}{d_pnl:>+9.2f}{t_pnl:>7.2f}"
              f"{statistics.fmean(outcome['ranks']):>7.2f}{bankrupt:>6}"
              f"{d_blend:>+8.3f}{t_blend:>7.2f}", flush=True)

    for key, value in DEFAULTS.items():
        setattr(Market_Maker, key, value)


if __name__ == "__main__":
    main()
