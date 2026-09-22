"""Last gate before submission: solvency and crash safety, on the grader's terms.

Everything else in this suite grades the strategy. This grades the two things that
score ZERO no matter how good the strategy is -- a code error, or a bankruptcy --
and it does so against the autograder's rules as written in the problem statement,
not against our own model of them:

    "Every time you do a trade, your balance will decrease by the maximum loss of
     your trade. So if you buy 5 contracts for $0.20 each, your cash balance will
     go down by a dollar; if you sell 5 contracts at $0.20 apiece, your cash
     balance will go down by $4. Cash solvency is checked at the end of each day.
     First, any payoffs associated with expired options will be credited -- note
     that this process can only increase your balance. At this point if your cash
     balance is below zero, you will be marked as bankrupt."

`GraderLedger` below is transcribed from that paragraph and from nothing else. It
deliberately does NOT import our helpers, because a bug we share with our own
model is exactly the bug this audit exists to catch: if `_worst_case_loss` had the
short leg backwards, our internal balance and our simulator would agree with each
other and both be wrong. The two figures agreeing here is only evidence because
they were derived independently.

Section 3 then fuzzes every method the grader calls, because an uncaught exception
is scored identically to a blow-up.
"""

import math
import random
import traceback

from Market_Maker import (
    AJARAI_NAME,
    AJARAI_UNDERLYING_ID,
    FED_FUNDS_RATE_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID,
    THERIODIC_NAME,
    THERIODIC_UNDERLYING_ID,
    BinaryOption,
    FokOrder,
    MarketHistory,
    MarketMaker,
    OptionLeg,
    OrderType,
    Underlying,
)
from validate_estimation import SCENARIOS, generate_history

FAILURES: list[str] = []
INITIAL_CASH = 1_000.0


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} {detail}")
    else:
        print(f"  ok   {label}")


class GraderLedger:
    """The autograder's cash rule, transcribed from the problem statement."""

    def __init__(self) -> None:
        self.cash = INITIAL_CASH
        self.longs: dict[int, int] = {}
        self.shorts: dict[int, int] = {}
        self.bankrupt = False
        self.low_water = INITIAL_CASH

    def trade(self, option_id: int, price: float, quantity: int) -> None:
        # "your balance will decrease by the maximum loss of your trade"
        if quantity > 0:  # buy 5 @ 0.20 -> down $1.00
            self.cash -= quantity * price
            self.longs[option_id] = self.longs.get(option_id, 0) + quantity
        else:  # sell 5 @ 0.20 -> down $4.00
            self.cash -= (-quantity) * (1.0 - price)
            self.shorts[option_id] = self.shorts.get(option_id, 0) + (-quantity)
        self.low_water = min(self.low_water, self.cash)

    def end_of_day(self, payoff_by_option_id: dict[int, float]) -> None:
        # "any payoffs associated with expired options will be credited"
        for option_id, payoff in payoff_by_option_id.items():
            self.cash += self.longs.pop(option_id, 0) * payoff
            self.cash += self.shorts.pop(option_id, 0) * (1.0 - payoff)
        # "at this point if your cash balance is below zero, you will be bankrupt"
        if self.cash < 0.0:
            self.bankrupt = True


def build_options(values: dict[int, float], day: int, first_id: int) -> list[BinaryOption]:
    """Build a day's board. `first_id` selects the id regime.

    Passing 0 every day RECYCLES ids: yesterday's expired option_id is handed to a
    brand new contract. Passing a running counter never reuses one. The venue's
    convention is not documented, and the template ships `contract_matches` -- a
    helper that compares options while ignoring their ids -- which only makes
    sense if ids and contracts can come apart. So both regimes get tested.
    """
    rate = OptionLeg(underlying_id=FED_FUNDS_RATE_UNDERLYING_ID, weight=1.0)
    ajarai = OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=1.0)
    theriodic = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=1.0)
    short_theriodic = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=-1.0)
    options: list[BinaryOption] = []
    option_id = first_id
    for steps in (1, 2, 4):
        option_id += 1
        options.append(BinaryOption(legs=(rate,), option_id=option_id, steps_until_expiry=steps,
                                    strike=max(0.0, round(values[FED_FUNDS_RATE_UNDERLYING_ID]
                                                          + 0.25 * ((day % 3) - 1), 2))))
        for leg, key in ((ajarai, AJARAI_UNDERLYING_ID), (theriodic, THERIODIC_UNDERLYING_ID)):
            option_id += 1
            options.append(BinaryOption(legs=(leg,), option_id=option_id, steps_until_expiry=steps,
                                        strike=round(values[key] * (0.96 + 0.02 * (day % 4)), 2)))
        option_id += 1
        options.append(BinaryOption(legs=(ajarai, short_theriodic), option_id=option_id,
                                    steps_until_expiry=steps, strike=0.0))
    return options


def run_session(parameters, seed: int, num_days: int, aggressive: bool,
                recycle_ids: bool = False) -> tuple[MarketMaker, GraderLedger]:
    """Drive a full session and shadow it with the independent ledger.

    `aggressive` fills the maker at every quote and lifts every FOK it accepts, at
    the largest size it showed -- a counterparty that always trades the maximum we
    are willing to show is the worst case for solvency, and the one most likely to
    walk us into the bankruptcy cliff.
    """
    rng = random.Random(seed)
    history, values = generate_history(parameters, 150, seed=seed)
    values = dict(values)
    next_id = 0
    options = build_options(values, 0, next_id)
    underlyings = [
        Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, values[FED_FUNDS_RATE_UNDERLYING_ID]),
        Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, values[AJARAI_UNDERLYING_ID]),
        Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, values[THERIODIC_UNDERLYING_ID]),
    ]
    maker = MarketMaker(underlyings, options, INITIAL_CASH)
    maker.warm_up(history)
    ledger = GraderLedger()

    for day in range(num_days):
        for option in list(options):
            quote = maker.quote(option, counterparty_id=rng.randint(1, 5))
            # Counterparty takes whichever side it likes, at our full shown size.
            if rng.random() < (0.9 if aggressive else 0.4):
                if rng.random() < 0.5:
                    price, quantity = quote.bid_price, quote.bid_quantity
                else:
                    price, quantity = quote.offer_price, -quote.offer_quantity
                if quantity:
                    maker.on_trade(option, price, quantity, counterparty_id=1)
                    ledger.trade(option.option_id, price, quantity)
            if rng.random() < (0.7 if aggressive else 0.3):
                theo = maker.price_option(option)
                fok_price = round(min(0.99, max(0.01, theo + rng.uniform(-0.3, 0.3))), 2)
                quantity = rng.randint(1, 12)
                order = FokOrder(counterparty_id=2, option_id=option.option_id,
                                 order_type=rng.choice([OrderType.BUY, OrderType.SELL]),
                                 price=fok_price, quantity=quantity)
                if maker.respond_to_fok(option, order):
                    signed = -quantity if order.order_type == OrderType.BUY else quantity
                    maker.on_trade(option, fok_price, signed, counterparty_id=2)
                    ledger.trade(option.option_id, fok_price, signed)

        # End of day: advance the world, settle whatever expired.
        payoffs = {option.option_id: option.expiry_valuation(values)
                   for option in options if option.steps_until_expiry == 0}
        values = parameters.advance_step(values)
        survivors = [option.advance_step() for option in options
                     if option.steps_until_expiry > 0]
        if not recycle_ids:
            next_id = max(next_id, max((o.option_id for o in options), default=0))
        replacements = build_options(values, day + 1, next_id)
        used = {option.option_id for option in survivors}
        options = survivors + [option for option in replacements if option.option_id not in used]
        underlyings = [
            Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, values[FED_FUNDS_RATE_UNDERLYING_ID]),
            Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, values[AJARAI_UNDERLYING_ID]),
            Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, values[THERIODIC_UNDERLYING_ID]),
        ]
        maker.on_step_advance(underlyings, options)
        ledger.end_of_day(payoffs)

    return maker, ledger


def main() -> None:
    print("1. our internal balance vs an independently transcribed grader ledger")
    for regime, recycle in (("unique option ids", False), ("RECYCLED option ids", True)):
        worst_gap = 0.0
        bankruptcies = 0
        lowest = INITIAL_CASH
        sessions = 0
        for parameters in SCENARIOS.values():
            for seed in (11, 29, 53):
                for aggressive in (False, True):
                    maker, ledger = run_session(parameters, seed, num_days=20,
                                                aggressive=aggressive, recycle_ids=recycle)
                    worst_gap = max(worst_gap, abs(maker.cash_balance - ledger.cash))
                    bankruptcies += 1 if ledger.bankrupt else 0
                    lowest = min(lowest, ledger.cash)
                    sessions += 1
        check(f"[{regime}] balances agree across {sessions} sessions", worst_gap < 1e-6,
              f"worst divergence {worst_gap:.6f}")
        check(f"[{regime}] no bankruptcy under maximally aggressive flow", bankruptcies == 0,
              f"{bankruptcies}/{sessions} sessions")
        print(f"       lowest end-of-day cash: {lowest:.2f} of {INITIAL_CASH:.0f}")

    print("2. the collateral rule matches the worked example in the statement")
    ledger = GraderLedger()
    ledger.trade(1, 0.20, 5)
    check("buy 5 @ 0.20 costs exactly $1", abs((INITIAL_CASH - ledger.cash) - 1.0) < 1e-9,
          f"cost {INITIAL_CASH - ledger.cash}")
    ledger = GraderLedger()
    ledger.trade(1, 0.20, -5)
    check("sell 5 @ 0.20 costs exactly $4", abs((INITIAL_CASH - ledger.cash) - 4.0) < 1e-9,
          f"cost {INITIAL_CASH - ledger.cash}")
    maker, _ = run_session(list(SCENARIOS.values())[0], 11, num_days=1, aggressive=False)
    probe = MarketMaker(maker.underlying_state, maker.active_option_state, INITIAL_CASH)
    option = maker.active_option_state[0]
    probe.on_trade(option, 0.20, -5, counterparty_id=1)
    check("our on_trade charges the same $4 for the short",
          abs((INITIAL_CASH - probe.cash_balance) - 4.0) < 1e-9,
          f"charged {INITIAL_CASH - probe.cash_balance}")

    print("3. fuzz every entry point the grader calls -- an exception scores zero")
    rng = random.Random(4242)
    parameters = list(SCENARIOS.values())[0]
    history, values = generate_history(parameters, 150, seed=5)
    underlyings = [
        Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, values[FED_FUNDS_RATE_UNDERLYING_ID]),
        Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, values[AJARAI_UNDERLYING_ID]),
        Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, values[THERIODIC_UNDERLYING_ID]),
    ]
    options = build_options(values, 0, 0)
    errors: list[str] = []
    calls = 0
    legs = {
        "rate": (OptionLeg(underlying_id=FED_FUNDS_RATE_UNDERLYING_ID, weight=1.0),),
        "ajr": (OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=1.0),),
        "spread": (OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=1.0),
                   OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=-1.0)),
        "weighted": (OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=2.5),
                     OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=-1.75)),
    }
    for cash in (INITIAL_CASH, 1.0, 0.0, -50.0):
        maker = MarketMaker(underlyings, options, cash)
        maker.warm_up(history)
        for _ in range(400):
            kind = rng.choice(list(legs))
            # Strikes span the absurd on purpose: the grader promises single legs
            # and AJR/THR spreads, but a price that overflows or divides by zero
            # on a strike we did not anticipate scores the same as bankruptcy.
            strike = rng.choice([0.0, -1e9, 1e9, -0.01, 1e-12,
                                 values[AJARAI_UNDERLYING_ID] * rng.uniform(0.1, 3.0)])
            option = BinaryOption(legs=legs[kind], option_id=rng.randint(1, 40),
                                  steps_until_expiry=rng.choice([0, 1, 2, 5, 30, 250]),
                                  strike=strike)
            try:
                calls += 1
                theo = maker.price_option(option)
                if not (0.0 <= theo <= 1.0) or math.isnan(theo):
                    errors.append(f"price out of range {theo} ({kind}, K={strike})")
                quote = maker.quote(option, counterparty_id=rng.randint(1, 9))
                if not (0.0 <= quote.bid_price < quote.offer_price <= 1.0):
                    errors.append(f"bad quote {quote.bid_price}/{quote.offer_price}")
                if quote.bid_quantity <= 0 or quote.offer_quantity <= 0:
                    errors.append(f"non-positive size {quote.bid_quantity}/{quote.offer_quantity}")
                order = FokOrder(counterparty_id=rng.randint(1, 9), option_id=option.option_id,
                                 order_type=rng.choice([OrderType.BUY, OrderType.SELL]),
                                 price=round(rng.uniform(0.01, 0.99), 2),
                                 quantity=rng.randint(1, 20))
                if maker.respond_to_fok(option, order):
                    maker.on_trade(option, order.price,
                                   rng.choice([1, -1]) * order.quantity,
                                   counterparty_id=order.counterparty_id)
                maker.on_step_advance(underlyings, options)
            except Exception as exception:
                errors.append(f"{type(exception).__name__} in {kind} K={strike}: {exception}")
    check(f"{calls} fuzzed call cycles raised nothing and stayed in range",
          not errors, "; ".join(errors[:3]))

    print("4. warm_up must survive whatever history it is handed")
    # NOTE: empty tuples and ragged lengths are NOT tested. `MarketHistory`
    # rejects both in its own __post_init__, which is provided template code above
    # the banner -- the grader would crash constructing such a history before it
    # ever reached us, so a maker that "survived" them would be guarding an
    # unreachable state. Everything below is a history the grader can actually hand us.
    bad_histories = {
        "one point": {FED_FUNDS_RATE_UNDERLYING_ID: (2.25,), AJARAI_UNDERLYING_ID: (900.0,),
                      THERIODIC_UNDERLYING_ID: (850.0,)},
        "two points": {FED_FUNDS_RATE_UNDERLYING_ID: (2.0, 2.25),
                       AJARAI_UNDERLYING_ID: (900.0, 905.0),
                       THERIODIC_UNDERLYING_ID: (850.0, 848.0)},
        "rate stuck at zero": {FED_FUNDS_RATE_UNDERLYING_ID: tuple([0.0] * 60),
                               AJARAI_UNDERLYING_ID: tuple([900.0] * 60),
                               THERIODIC_UNDERLYING_ID: tuple([850.0] * 60)},
        "huge values": {FED_FUNDS_RATE_UNDERLYING_ID: tuple([25.0] * 60),
                        AJARAI_UNDERLYING_ID: tuple([1e9] * 60),
                        THERIODIC_UNDERLYING_ID: tuple([1e-9] * 60)},
    }
    for label, series in bad_histories.items():
        try:
            maker = MarketMaker(underlyings, options, INITIAL_CASH)
            maker.warm_up(MarketHistory(values_by_underlying_id=series))
            prices = [maker.price_option(option) for option in options]
            quotes = [maker.quote(option, counterparty_id=1) for option in options]
            fine = (all(0.0 <= p <= 1.0 and not math.isnan(p) for p in prices)
                    and all(0.0 <= q.bid_price < q.offer_price <= 1.0 for q in quotes))
            check(f"{label}", fine, "produced an unusable price or quote")
        except Exception as exception:
            check(f"{label}", False, f"{type(exception).__name__}: {exception}")

    print("-" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for failure in FAILURES:
            print(" ", failure)
    else:
        print("submission gate clear: solvent, faithful to the grader ledger, no crashes")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("EXCEPTION RAISED -- this would score zero")
