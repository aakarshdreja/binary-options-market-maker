"""Was the hard counterparty filter really worth 1.6 of mean PnL?

That number came from comparing two SEPARATE process invocations, back when
`simulate_field.bind` seeded competitors with `hash(self.name)`. Python randomises
string hashing per process, so every rival's parameter perturbation was redrawn on
each run and the two numbers were never measured against the same field. The
conclusion may still be right, but the evidence for it was not.

So measure it the way it should have been measured: both variants in ONE process,
on identical seeds, paired session by session, and report the paired t-statistic.

The two arms differ only in `_fok_counterparty_id_is_reported`. Forcing it True at
construction reproduces the hard filter exactly -- identity is enforced from the
first fill, so in a venue that stamps its own id (which our simulator does, always
reporting counterparty_id=1) every genuine FOK fill is discarded and the side
convention can never confirm.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import statistics

from Market_Maker import MarketMaker
from simulate_field import FIELD, FieldSession
from validate_estimation import SCENARIOS

SESSIONS = 12
DAYS = 25

_original_init = MarketMaker.__init__


def _hard_filter_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self._fok_counterparty_id_is_reported = True


def run() -> tuple[list[float], int, int]:
    pnls: list[float] = []
    bankruptcies = wins = 0
    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.35, 0.5):
            for index in range(SESSIONS):
                session = FieldSession(parameters, seed=1000 + index * 37, num_days=DAYS,
                                       history_days=[60, 150, 400][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                result = session.run()
                pnls.append(float(result["all"]["SUBJECT"]))
                bankruptcies += 1 if result["bankrupt"] else 0
                wins += 1 if result["rank"] == 1 else 0
    return pnls, bankruptcies, wins


def main() -> None:
    MarketMaker.__init__ = _hard_filter_init
    hard, hard_bankrupt, hard_wins = run()
    MarketMaker.__init__ = _original_init
    soft, soft_bankrupt, soft_wins = run()

    differences = [a - b for a, b in zip(soft, hard)]
    mean_difference = statistics.fmean(differences)
    error = statistics.stdev(differences) / (len(differences) ** 0.5)

    print(f"\n{len(soft)} paired sessions, identical seeds, one process\n")
    print(f"{'arm':<26}{'mean':>9}{'median':>9}{'worst':>9}{'>0':>7}{'wins':>7}{'bank':>6}")
    print("-" * 64)
    for label, pnls, bankrupt, wins in (
        ("hard counterparty filter", hard, hard_bankrupt, hard_wins),
        ("self-calibrating (live)", soft, soft_bankrupt, soft_wins),
    ):
        positive = sum(1 for value in pnls if value > 0) / len(pnls)
        print(f"{label:<26}{statistics.fmean(pnls):>9.2f}{statistics.median(pnls):>9.2f}"
              f"{min(pnls):>9.1f}{positive:>7.0%}{wins:>7}{bankrupt:>6}")
    print("-" * 64)
    print(f"paired difference (self-calibrating - hard): {mean_difference:+.2f}"
          f"   t = {mean_difference / error if error else 0.0:+.2f}")


if __name__ == "__main__":
    main()
