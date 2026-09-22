"""A realistic competitor field: rivals who ESTIMATE, and are wrong in sticky ways.

`simulate_session` models a competitor's pricing error as gaussian noise added per
quote. That is unexploitable by construction. Fresh noise on every quote averages
to zero, so a rival is randomly wide or tight but never systematically wrong, and
there is no way to lean on them. Tuning against that field -- and against a "hard"
field where rivals are handed the TRUE parameters and never err at all -- pushed
our configuration toward conservatism, because in that world we are the only
player who can be wrong.

The real challenge is the opposite. Every rival is another participant estimating
the same hidden parameters from the same history, and estimation error is STICKY:
if someone's AJR drift is 3 sigma high, every AJR option they quote is too high,
all session, in the same direction. That is exploitable, and it is the actual
source of edge in this competition.

So competitors here draw a perturbed parameter set ONCE per session and price off
it consistently. The archetypes are guesses at what people will really submit:

  sharp     a strong rival, small errors, tight spread -- the one to beat
  montecarlo   simulation-based pricing, decent but vol slightly off
  norate    models companies fine, treats the rate as a driftless random walk
            (skips reversion, tilt and the floor at zero) -- a very common
            simplification, and one we fixed in ourselves late
  nocorr    ignores the shared sector shock, so AJR-vs-THR spreads are wrong
  crude     large errors, wide spread to compensate

The question this is built to answer: against rivals who are persistently wrong
rather than omnisciently right, should we be trading MORE aggressively than the
settings we tuned in the pessimistic field?
"""

import argparse
import math
import random
import zlib
import statistics
from dataclasses import replace

import Market_Maker
from Market_Maker import (
    BinaryOption,
    FokOrder,
    MarketMaker,
    MarketParameters,
    OrderType,
    Quote,
)
from simulate_session import INITIAL_CASH, Ledger, Session, worst_case_loss
from validate_estimation import SCENARIOS


def perturb(parameters: MarketParameters, rng: random.Random, scale: float,
            archetype: str) -> MarketParameters:
    """Draw a plausible wrong-but-coherent parameter set, once, for the session."""
    def jitter(value: float, relative: float, floor: float | None = None) -> float:
        moved = value * (1.0 + rng.gauss(0.0, relative * scale))
        return max(floor, moved) if floor is not None else moved

    up = min(0.45, max(0.02, jitter(parameters.rate_up_probability, 0.35)))
    down = min(0.45, max(0.02, jitter(parameters.rate_down_probability, 0.35)))
    reversion = parameters.rate_reversion_strength
    target = parameters.rate_target

    if archetype == "norate":
        # Treats the rate as a driftless random walk: symmetric moves, no pull
        # toward target. Cheap to implement, and wrong in a very specific way.
        average = 0.5 * (up + down)
        up = down = average
        reversion = 0.0
    else:
        reversion = min(0.9, max(0.0, jitter(reversion, 0.40)))
        target = max(0.0, jitter(target, 0.25))

    ajarai_sector = jitter(parameters.ajarai_sector_beta, 0.30)
    theriodic_sector = jitter(parameters.theriodic_sector_beta, 0.30)
    if archetype == "nocorr":
        # Believes the two names are independent, and folds the sector variance
        # into idiosyncratic risk so total vol stays roughly right.
        ajarai_sector = 1e-6
        theriodic_sector = 1e-6

    ajarai_idio = jitter(parameters.ajarai_idio_std_dev, 0.20, floor=1e-4)
    theriodic_idio = jitter(parameters.theriodic_idio_std_dev, 0.20, floor=1e-4)
    if archetype == "nocorr":
        ajarai_idio = math.sqrt(
            ajarai_idio ** 2 + (parameters.ajarai_sector_beta * parameters.sector_std_dev) ** 2
        )
        theriodic_idio = math.sqrt(
            theriodic_idio ** 2 + (parameters.theriodic_sector_beta * parameters.sector_std_dev) ** 2
        )

    volatility_bias = 1.0 + (0.12 * scale if archetype == "montecarlo" else 0.0)

    return MarketParameters(
        ajarai_drift=parameters.ajarai_drift + rng.gauss(0.0, 0.0035 * scale),
        ajarai_idio_std_dev=max(1e-4, ajarai_idio * volatility_bias),
        ajarai_rate_beta=jitter(parameters.ajarai_rate_beta, 0.30),
        ajarai_sector_beta=ajarai_sector,
        rate_down_probability=down,
        rate_reversion_strength=reversion,
        rate_up_probability=up,
        sector_std_dev=max(1e-6, jitter(parameters.sector_std_dev, 0.20, floor=1e-6)),
        theriodic_drift=parameters.theriodic_drift + rng.gauss(0.0, 0.0035 * scale),
        theriodic_idio_std_dev=max(1e-4, theriodic_idio * volatility_bias),
        theriodic_rate_beta=jitter(parameters.theriodic_rate_beta, 0.30),
        theriodic_sector_beta=theriodic_sector,
        rate_step=parameters.rate_step,
        rate_target=target,
    )


class ParametricCompetitor:
    """Rival whose pricing error is persistent, because it lives in the params."""

    def __init__(self, name: str, archetype: str, half_spread: float, size: int,
                 error_scale: float, pricer: MarketMaker) -> None:
        self.name = name
        self.archetype = archetype
        self.half_spread = half_spread
        self.size = size
        self.error_scale = error_scale
        self._pricer = pricer
        self._parameters: MarketParameters | None = None

    def bind(self, true_parameters: MarketParameters, seed: int) -> None:
        # crc32, NOT hash(). Python randomises string hashing per process, so
        # hash(self.name) redrew every competitor's parameter perturbation on each
        # run -- rival skill, and therefore the flow reaching us, changed run to
        # run. Cells swept inside ONE process stayed comparable, which is why the
        # paired sweeps survived, but any A/B measured across two invocations was
        # partly noise. crc32 is stable across processes and versions.
        rng = random.Random(seed ^ (zlib.crc32(self.name.encode()) & 0xFFFF))
        self._parameters = perturb(true_parameters, rng, self.error_scale, self.archetype)

    def price(self, option: BinaryOption) -> float:
        assert self._parameters is not None
        return min(max(self._pricer.price_option_from_parameters(self._parameters, option), 0.0), 1.0)

    def quote(self, option: BinaryOption, ledger: Ledger) -> Quote | None:
        theo = self.price(option)
        bid_ticks = max(0, min(99, math.floor((theo - self.half_spread) * 100)))
        offer_ticks = max(1, min(100, math.ceil((theo + self.half_spread) * 100)))
        if offer_ticks <= bid_ticks:
            offer_ticks = bid_ticks + 1
        bid, offer = bid_ticks / 100.0, offer_ticks / 100.0

        available = max(0.0, ledger.cash - 0.15 * INITIAL_CASH)
        bid_quantity = self.size if bid <= 0 else min(self.size, int(0.05 * available / bid))
        offer_quantity = self.size if offer >= 1.0 else min(self.size, int(0.05 * available / (1.0 - offer)))
        if bid_quantity <= 0:
            bid, bid_quantity = 0.0, 1
        if offer_quantity <= 0:
            offer, offer_quantity = 1.0, 1
        if bid >= offer:
            bid = max(0.0, offer - 0.01)
        return Quote(bid_price=round(bid, 2), bid_quantity=bid_quantity,
                     offer_price=round(offer, 2), offer_quantity=offer_quantity)

    def respond_to_fok(self, option: BinaryOption, order: FokOrder, ledger: Ledger) -> bool:
        theo = self.price(option)
        if order.order_type == OrderType.SELL:
            edge, signed = theo - order.price, order.quantity
        else:
            edge, signed = order.price - theo, -order.quantity
        if edge < self.half_spread:
            return False
        return worst_case_loss(order.price, signed) <= 0.06 * max(0.0, ledger.cash - 0.15 * INITIAL_CASH)


class FieldSession(Session):
    """`Session`, but with rivals that estimate instead of rivals that know."""

    def __init__(self, true_parameters, seed, num_days, history_days, informed_fraction,
                 competitors) -> None:
        super().__init__(true_parameters, seed, num_days, history_days, informed_fraction,
                         competitors=[])
        self.competitors = [
            ParametricCompetitor(name, archetype, half_spread, size, error_scale, self.oracle)
            for name, archetype, half_spread, size, error_scale in competitors
        ]
        for competitor in self.competitors:
            competitor.bind(true_parameters, seed)
        self.ledgers = {"SUBJECT": Ledger("SUBJECT")}
        for competitor in self.competitors:
            self.ledgers[competitor.name] = Ledger(competitor.name)


# (name, archetype, half_spread, size, error_scale)
FIELD: list[tuple[str, str, float, int, float]] = [
    ("sharp", "sharp", 0.020, 20, 0.5),
    ("montecarlo", "montecarlo", 0.030, 15, 1.0),
    ("norate", "norate", 0.025, 15, 1.0),
    ("nocorr", "nocorr", 0.030, 15, 1.0),
    ("crude", "crude", 0.060, 10, 2.0),
]


def run_field(sessions: int, days: int) -> dict[str, float]:
    pnls: list[float] = []
    ranks: list[float] = []
    beat_sharp = 0
    wins = 0
    bankruptcies = 0
    rfq_edge = rfq_contracts = fok_edge = fok_contracts = 0.0

    for parameters in SCENARIOS.values():
        for informed_fraction in (0.2, 0.5):
            for index in range(sessions):
                session = FieldSession(parameters, seed=1000 + index * 37, num_days=days,
                                       history_days=[60, 150, 400][index % 3],
                                       informed_fraction=informed_fraction, competitors=FIELD)
                result = session.run()
                values = result["all"]
                pnls.append(float(result["pnl"]))
                ranks.append(float(result["rank"]))
                wins += 1 if result["rank"] == 1 else 0
                beat_sharp += 1 if values["SUBJECT"] > values["sharp"] else 0
                bankruptcies += 1 if result["bankrupt"] else 0
                rfq_edge += session.diagnostics.get("rfq_edge", 0.0)
                rfq_contracts += session.diagnostics.get("rfq_contracts", 0.0)
                fok_edge += session.diagnostics.get("fok_edge", 0.0)
                fok_contracts += session.diagnostics.get("fok_contracts", 0.0)

    return {
        "mean": statistics.fmean(pnls),
        "median": statistics.median(pnls),
        "worst": min(pnls),
        "rank": statistics.fmean(ranks),
        "win_rate": wins / len(pnls),
        "beat_sharp": beat_sharp / len(pnls),
        "bankrupt": bankruptcies,
        "total": len(pnls),
        "rfq_per_contract": rfq_edge / max(rfq_contracts, 1.0),
        "fok_per_contract": fok_edge / max(fok_contracts, 1.0),
        "rfq_contracts": rfq_contracts / len(pnls),
        "fok_contracts": fok_contracts / len(pnls),
    }


CONFIGS: dict[str, dict[str, float]] = {
    "current": {},
    "tight rfq": {"_BASE_HALF_SPREAD": 0.012},
    "fok unc x0.7": {"_FOK_UNCERTAINTY_MULTIPLIER": 0.7},
    "fok unc x0.4": {"_FOK_UNCERTAINTY_MULTIPLIER": 0.4},
    "fok unc x0.0": {"_FOK_UNCERTAINTY_MULTIPLIER": 0.0},
    "tight + fok0.7": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 0.7},
    "tight + fok0.4": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 0.4},
    "tight + fok0.4 + risk": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 0.4,
                              "_FOK_RISK_FRACTION": 0.180},
    "tight + fok0.4 + pos": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 0.4,
                             "_FOK_RISK_FRACTION": 0.180, "_MAXIMUM_POSITION_PER_OPTION": 60},
    "all in": {"_BASE_HALF_SPREAD": 0.012, "_FOK_UNCERTAINTY_MULTIPLIER": 0.4,
               "_FOK_RISK_FRACTION": 0.180, "_MAXIMUM_POSITION_PER_OPTION": 60,
               "_FOK_BASE_EDGE": 0.006, "_TRADE_RISK_FRACTION": 0.050},
}

DEFAULTS = {key: getattr(Market_Maker, key) for config in CONFIGS.values() for key in config}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--days", type=int, default=25)
    args = parser.parse_args()

    print(f"{'config':<20}{'mean':>8}{'median':>8}{'worst':>9}{'rank':>7}{'win%':>7}"
          f"{'>sharp':>8}{'rfq/ct':>9}{'fok/ct':>9}{'rfq ct':>8}{'bankrupt':>10}")
    print("-" * 105)
    for name, overrides in CONFIGS.items():
        for key, value in DEFAULTS.items():
            setattr(Market_Maker, key, value)
        for key, value in overrides.items():
            setattr(Market_Maker, key, value)
        m = run_field(args.sessions, args.days)
        print(f"{name:<20}{m['mean']:>8.1f}{m['median']:>8.1f}{m['worst']:>9.1f}{m['rank']:>7.2f}"
              f"{m['win_rate']:>7.0%}{m['beat_sharp']:>8.0%}{m['rfq_per_contract']:>9.4f}"
              f"{m['fok_per_contract']:>9.4f}{m['rfq_contracts']:>8.0f}"
              f"{m['bankrupt']:>6}/{m['total']:<4}", flush=True)


if __name__ == "__main__":
    main()
