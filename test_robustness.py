"""Edge-case robustness suite.

A crash or an invalid `Quote` scores zero on a test, which is the same as going
bankrupt. This hammers the degenerate inputs that a live session can plausibly
produce: no warm-up, near-empty history, an underlying that never moves, extreme
strikes, long tenors, and a market maker that has already burned through its cash.
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
from validate_estimation import SCENARIOS

RATE_LEG = OptionLeg(underlying_id=FED_FUNDS_RATE_UNDERLYING_ID, weight=1.0)
AJARAI_LEG = OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=1.0)
THERIODIC_LEG = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=1.0)
SHORT_THERIODIC_LEG = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=-1.0)

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} {detail}")


def make_market_maker(rate=2.25, ajarai=900.0, theriodic=850.0, cash=1_000.0, options=None):
    underlyings = [
        Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, rate),
        Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, ajarai),
        Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, theriodic),
    ]
    return MarketMaker(underlyings, options or [], cash)


def option_zoo(rate=2.25, ajarai=900.0, theriodic=850.0) -> list[BinaryOption]:
    options: list[BinaryOption] = []
    option_id = 0
    for steps in (0, 1, 2, 7, 15, 40):
        for legs, strike in (
            ((RATE_LEG,), 0.0),
            ((RATE_LEG,), rate),
            ((RATE_LEG,), 99.0),
            ((AJARAI_LEG,), 0.01),
            ((AJARAI_LEG,), ajarai),
            ((AJARAI_LEG,), ajarai * 1_000.0),
            ((THERIODIC_LEG,), theriodic),
            ((AJARAI_LEG, SHORT_THERIODIC_LEG), 0.0),
            ((AJARAI_LEG, SHORT_THERIODIC_LEG), 250.0),
            ((AJARAI_LEG, THERIODIC_LEG), ajarai + theriodic),
            ((AJARAI_LEG, SHORT_THERIODIC_LEG, RATE_LEG), 10.0),
        ):
            option_id += 1
            options.append(BinaryOption(legs=legs, option_id=option_id,
                                        steps_until_expiry=steps, strike=strike))
    return options


def exercise(label: str, market_maker: MarketMaker, options: list[BinaryOption]) -> None:
    for option in options:
        price = market_maker.price_option(option)
        check(f"{label}/price-range", 0.0 <= price <= 1.0 and math.isfinite(price), f"{option} -> {price}")

        quote = market_maker.quote(option, counterparty_id=1)  # Quote validates itself
        check(f"{label}/quote-order", quote.bid_price < quote.offer_price, str(quote))
        check(f"{label}/quote-size", quote.bid_quantity > 0 and quote.offer_quantity > 0, str(quote))

        for order_type in (OrderType.BUY, OrderType.SELL):
            for order_price in (0.0, 0.01, 0.5, 0.99, 1.0):
                order = FokOrder(counterparty_id=2, option_id=option.option_id,
                                 order_type=order_type, price=order_price, quantity=5)
                result = market_maker.respond_to_fok(option, order)
                check(f"{label}/fok-bool", isinstance(result, bool), str(result))


def constant_history(num_days: int, rate: float = 2.0) -> MarketHistory:
    return MarketHistory(values_by_underlying_id={
        FED_FUNDS_RATE_UNDERLYING_ID: tuple([rate] * num_days),
        AJARAI_UNDERLYING_ID: tuple([900.0] * num_days),
        THERIODIC_UNDERLYING_ID: tuple([850.0] * num_days),
    })


def main() -> None:
    print("1. no warm_up at all")
    exercise("no-warmup", make_market_maker(), option_zoo())

    print("2. tiny / degenerate histories")
    for num_days in (1, 2, 5, 11, 12, 13):
        market_maker = make_market_maker()
        market_maker.warm_up(constant_history(num_days))
        exercise(f"const-{num_days}d", market_maker, option_zoo())

    print("3. rate pinned at zero (floor) with flat companies")
    market_maker = make_market_maker(rate=0.0)
    market_maker.warm_up(constant_history(80, rate=0.0))
    exercise("zero-rate", market_maker, option_zoo(rate=0.0))

    print("4. realistic histories across scenarios")
    for name, parameters in SCENARIOS.items():
        random.seed(7)
        values = {FED_FUNDS_RATE_UNDERLYING_ID: 2.25,
                  AJARAI_UNDERLYING_ID: 900.0, THERIODIC_UNDERLYING_ID: 850.0}
        series = {k: [v] for k, v in values.items()}
        for _ in range(199):
            values = parameters.advance_step(values)
            for key, value in values.items():
                series[key].append(value)
        market_maker = make_market_maker(values[FED_FUNDS_RATE_UNDERLYING_ID],
                                         values[AJARAI_UNDERLYING_ID], values[THERIODIC_UNDERLYING_ID])
        market_maker.warm_up(MarketHistory(values_by_underlying_id={k: tuple(v) for k, v in series.items()}))
        exercise(f"real-{name[:8]}", market_maker,
                 option_zoo(values[FED_FUNDS_RATE_UNDERLYING_ID],
                            values[AJARAI_UNDERLYING_ID], values[THERIODIC_UNDERLYING_ID]))

    print("5. exhausted / negative cash")
    for cash in (100.0, 1.0, 0.0, -50.0):
        market_maker = make_market_maker(cash=1_000.0)
        market_maker.warm_up(constant_history(80))
        market_maker.cash_balance = cash
        exercise(f"cash-{cash}", market_maker, option_zoo())

    print("6. tiny and huge underlying values")
    for ajarai, theriodic in ((0.01, 0.02), (1e9, 1e9), (1.0, 1e6)):
        market_maker = make_market_maker(ajarai=ajarai, theriodic=theriodic)
        market_maker.warm_up(constant_history(80))
        exercise(f"scale-{ajarai}", market_maker, option_zoo(2.25, ajarai, theriodic))

    print("7. step advance + settlement bookkeeping")
    options = option_zoo()
    market_maker = make_market_maker(options=options)
    market_maker.warm_up(constant_history(80))
    starting_cash = market_maker.cash_balance
    for option in options[:8]:
        market_maker.on_trade(option, 0.30, 5, 1)
        market_maker.on_trade(option, 0.30, -5, 1)  # offsetting round trip
    round_trip_cash = market_maker.cash_balance
    check("gross-collateral", round_trip_cash < starting_cash, "round trip should post collateral twice")
    # Expire everything; gross settlement must return the full collateral.
    market_maker.on_step_advance(market_maker.underlying_state, [])
    check("gross-refund", abs(market_maker.cash_balance - starting_cash) < 1e-6,
          f"expected {starting_cash}, got {market_maker.cash_balance}")

    print("-" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for failure in FAILURES[:20]:
            print(" ", failure)
    else:
        print("all robustness checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("EXCEPTION RAISED -- this would score zero")
