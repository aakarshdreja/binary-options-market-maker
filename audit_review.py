"""Adjudicate an external code review against the actual behaviour of the code.

A review raised several claimed bugs. Some are worth taking seriously and some
rest on a misreading, but the way to tell them apart is to execute the disputed
cases rather than to argue about them, so each claim below is turned into a test
that can only pass if the claim is false.

The claims, and what each test settles:

  C3  "_two_company_probability drops the leg weights from the variance: it uses
       var1 + var2 - 2*cov, which is only valid for weights +1/-1."
      -> Priced against Monte Carlo at deliberately non-unit weights.

  C4  "_single_company_probability handles signs inconsistently for zero or
       negative strikes."
      -> Every sign combination of weight and strike, against Monte Carlo, plus
         the degenerate cases where the answer must be exactly 0 or 1.

  B1/B2 "Quote can emit non-penny prices, or bid >= offer, and raise ValueError."
      -> Brute force over adversarial states. `Quote.__post_init__` already
         enforces all four invariants, so constructing one IS the assertion; any
         violation raises and fails the test.

  C5  "Shrinkage can produce NaN on degenerate histories."
      -> Warm up on constant, single-value and zero-variance histories and assert
         every emitted price is finite.

  B3  "An RFQ that fills at exactly the pending FOK price casts a false vote and
       can flip the side convention."
      -> This one is real. Tested explicitly below.
"""

import math
import random
import traceback

import Market_Maker
from Market_Maker import (
    AJARAI_NAME,
    AJARAI_UNDERLYING_ID,
    FED_FUNDS_RATE_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID,
    THERIODIC_NAME,
    THERIODIC_UNDERLYING_ID,
    BinaryOption,
    MarketHistory,
    MarketMaker,
    OptionLeg,
    OrderType,
    FokOrder,
    Underlying,
)
from validate_pricing import BASE_PARAMETERS, STRESS_PARAMETERS, build_market_maker, monte_carlo_price

FAILURES: list[str] = []
TRIALS = 200_000


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} {detail}")
    else:
        print(f"  ok   {label}")


def price_case(parameters, legs, strike, steps, values):
    option = BinaryOption(legs=tuple(legs), option_id=1, steps_until_expiry=steps, strike=strike)
    maker = build_market_maker(values[FED_FUNDS_RATE_UNDERLYING_ID],
                               values[AJARAI_UNDERLYING_ID], values[THERIODIC_UNDERLYING_ID])
    analytic = maker.price_option_from_parameters(parameters, option)
    empirical = monte_carlo_price(parameters, option, values, TRIALS)
    error = math.sqrt(max(empirical * (1.0 - empirical), 1e-9) / TRIALS)
    return analytic, empirical, (analytic - empirical) / error if error > 0 else 0.0


def main() -> None:
    values = {FED_FUNDS_RATE_UNDERLYING_ID: 2.25,
              AJARAI_UNDERLYING_ID: 900.0, THERIODIC_UNDERLYING_ID: 850.0}

    print("C3. weighted two-company spreads (claim: variance ignores weights)")
    # The disputed fast path needs opposite-signed weights and a zero strike. If
    # the weights were really missing from the variance, a 2:-1 spread would be
    # mispriced by far more than Monte Carlo noise.
    weighted_cases = [
        ("2*AJR - 1*THR >= 0", [OptionLeg(AJARAI_UNDERLYING_ID, 2.0),
                                OptionLeg(THERIODIC_UNDERLYING_ID, -1.0)], 0.0, 5, BASE_PARAMETERS),
        ("1*AJR - 3*THR >= 0", [OptionLeg(AJARAI_UNDERLYING_ID, 1.0),
                                OptionLeg(THERIODIC_UNDERLYING_ID, -3.0)], 0.0, 4, BASE_PARAMETERS),
        ("0.5*AJR - 0.4*THR >= 0", [OptionLeg(AJARAI_UNDERLYING_ID, 0.5),
                                    OptionLeg(THERIODIC_UNDERLYING_ID, -0.4)], 0.0, 6, STRESS_PARAMETERS),
        ("-2*AJR + 1.5*THR >= 0", [OptionLeg(AJARAI_UNDERLYING_ID, -2.0),
                                   OptionLeg(THERIODIC_UNDERLYING_ID, 1.5)], 0.0, 3, STRESS_PARAMETERS),
    ]
    for label, legs, strike, steps, parameters in weighted_cases:
        analytic, empirical, z = price_case(parameters, legs, strike, steps, values)
        check(f"{label:<26} {analytic:.5f} vs {empirical:.5f}", abs(z) < 4.0, f"z={z:+.2f}")

    print("\nC4. single-leg sign handling (claim: inconsistent for K <= 0)")
    sign_cases = [
        ("-1*THR >= -900", [OptionLeg(THERIODIC_UNDERLYING_ID, -1.0)], -900.0, 4, BASE_PARAMETERS),
        ("-2*AJR >= -1800", [OptionLeg(AJARAI_UNDERLYING_ID, -2.0)], -1800.0, 5, BASE_PARAMETERS),
        ("-1*AJR >= -1000", [OptionLeg(AJARAI_UNDERLYING_ID, -1.0)], -1000.0, 3, STRESS_PARAMETERS),
        ("0.5*AJR >= 400", [OptionLeg(AJARAI_UNDERLYING_ID, 0.5)], 400.0, 6, BASE_PARAMETERS),
    ]
    for label, legs, strike, steps, parameters in sign_cases:
        analytic, empirical, z = price_case(parameters, legs, strike, steps, values)
        check(f"{label:<26} {analytic:.5f} vs {empirical:.5f}", abs(z) < 4.0, f"z={z:+.2f}")

    # Degenerate strikes where the answer is forced by sign alone.
    forced = [
        ("+1*AJR >= 0 is certain", [OptionLeg(AJARAI_UNDERLYING_ID, 1.0)], 0.0, 3, 1.0),
        ("+1*AJR >= -50 is certain", [OptionLeg(AJARAI_UNDERLYING_ID, 1.0)], -50.0, 3, 1.0),
        ("-1*AJR >= 0 is impossible", [OptionLeg(AJARAI_UNDERLYING_ID, -1.0)], 0.0, 3, 0.0),
        ("-1*AJR >= 10 is impossible", [OptionLeg(AJARAI_UNDERLYING_ID, -1.0)], 10.0, 3, 0.0),
    ]
    for label, legs, strike, steps, expected in forced:
        option = BinaryOption(legs=tuple(legs), option_id=1, steps_until_expiry=steps, strike=strike)
        maker = build_market_maker(2.25, 900.0, 850.0)
        got = maker.price_option_from_parameters(BASE_PARAMETERS, option)
        check(f"{label:<30} -> {got:.4f}", abs(got - expected) < 1e-12)

    print("\nB1/B2. Quote invariants under adversarial state (penny grid, bid < offer)")
    # Quote.__post_init__ enforces positive sizes, [0,1] prices, bid < offer and a
    # whole-penny grid, so if any of these states could produce a bad quote this
    # loop would raise rather than return.
    rng = random.Random(7)
    options = [
        BinaryOption(legs=(OptionLeg(AJARAI_UNDERLYING_ID, 1.0),), option_id=1, steps_until_expiry=5, strike=900.0),
        BinaryOption(legs=(OptionLeg(FED_FUNDS_RATE_UNDERLYING_ID, 1.0),), option_id=2, steps_until_expiry=3, strike=2.25),
        BinaryOption(legs=(OptionLeg(AJARAI_UNDERLYING_ID, 1.0), OptionLeg(THERIODIC_UNDERLYING_ID, -1.0)),
                     option_id=3, steps_until_expiry=7, strike=0.0),
        # Pinned deep ITM / OTM, where the centre is jammed against 0.00 and 1.00.
        BinaryOption(legs=(OptionLeg(AJARAI_UNDERLYING_ID, 1.0),), option_id=4, steps_until_expiry=1, strike=100.0),
        BinaryOption(legs=(OptionLeg(AJARAI_UNDERLYING_ID, 1.0),), option_id=5, steps_until_expiry=1, strike=9_000.0),
    ]
    emitted = 0
    bad = 0
    for cash in (1_000.0, 200.0, 20.0, 1.0, 0.5, 0.0, -5.0):
        for inventory in (-400, -180, -40, 0, 40, 180, 400):
            maker = MarketMaker(
                [Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, 2.25),
                 Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, 900.0),
                 Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, 850.0)],
                options, cash)
            maker.warm_up(MarketHistory(values_by_underlying_id={
                FED_FUNDS_RATE_UNDERLYING_ID: tuple(2.0 + 0.25 * ((i // 5) % 3) for i in range(120)),
                AJARAI_UNDERLYING_ID: tuple(900.0 * (1.0 + 0.002 * ((i * 7) % 11 - 5)) for i in range(120)),
                THERIODIC_UNDERLYING_ID: tuple(850.0 * (1.0 + 0.002 * ((i * 5) % 13 - 6)) for i in range(120)),
            }))
            for option in options:
                maker.position.option_quantity_by_option_id[option.option_id] = inventory
                try:
                    quote = maker.quote(option, counterparty_id=rng.randint(1, 5))
                    emitted += 1
                    if quote.bid_price >= quote.offer_price:
                        bad += 1
                except Exception as error:  # noqa: BLE001 - the point is to catch any
                    bad += 1
                    FAILURES.append(f"quote raised {type(error).__name__}: {error}")
                    print(f"  FAIL quote raised {type(error).__name__}: {error}")
    check(f"{emitted} quotes across cash x inventory states all valid", bad == 0, f"{bad} bad")

    print("\nC5. degenerate histories produce finite prices (claim: NaN from shrinkage)")
    degenerate = {
        "constant everything": {FED_FUNDS_RATE_UNDERLYING_ID: (2.0,) * 40,
                                AJARAI_UNDERLYING_ID: (900.0,) * 40,
                                THERIODIC_UNDERLYING_ID: (850.0,) * 40},
        "single day": {FED_FUNDS_RATE_UNDERLYING_ID: (2.0,),
                       AJARAI_UNDERLYING_ID: (900.0,),
                       THERIODIC_UNDERLYING_ID: (850.0,)},
        "rate pinned at zero": {FED_FUNDS_RATE_UNDERLYING_ID: (0.0,) * 30,
                                AJARAI_UNDERLYING_ID: tuple(900.0 + i for i in range(30)),
                                THERIODIC_UNDERLYING_ID: (850.0,) * 30},
        "two days only": {FED_FUNDS_RATE_UNDERLYING_ID: (2.0, 2.25),
                          AJARAI_UNDERLYING_ID: (900.0, 901.0),
                          THERIODIC_UNDERLYING_ID: (850.0, 849.0)},
    }
    for label, history in degenerate.items():
        maker = MarketMaker(
            [Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, 2.25),
             Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, 900.0),
             Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, 850.0)],
            options, 1_000.0)
        maker.warm_up(MarketHistory(values_by_underlying_id=history))
        prices = [maker.price_option(option) for option in options]
        uncertainties = [maker._price_uncertainty(option) for option in options]
        finite = all(math.isfinite(v) for v in prices + uncertainties)
        ranged = all(0.0 <= p <= 1.0 for p in prices)
        check(f"{label:<22} prices finite and in [0,1]", finite and ranged, f"{prices}")

    print("\nB3. an RFQ filling at exactly the pending FOK price must not vote")
    maker = MarketMaker(
        [Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, 2.25),
         Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, 900.0),
         Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, 850.0)],
        options, 1_000.0)
    maker.warm_up(MarketHistory(values_by_underlying_id={
        FED_FUNDS_RATE_UNDERLYING_ID: tuple(2.0 + 0.25 * ((i // 5) % 3) for i in range(120)),
        AJARAI_UNDERLYING_ID: tuple(900.0 * (1.0 + 0.002 * ((i * 7) % 11 - 5)) for i in range(120)),
        THERIODIC_UNDERLYING_ID: tuple(850.0 * (1.0 + 0.002 * ((i * 5) % 13 - 6)) for i in range(120)),
    }))
    option = options[0]
    theo = maker.price_option(option)
    price = round(max(0.02, theo - 0.20), 2)
    order = FokOrder(counterparty_id=2, option_id=option.option_id,
                     order_type=OrderType.SELL, price=price, quantity=3)
    accepted = maker.respond_to_fok(option, order)
    check("seed FOK accepted", accepted)
    before = (maker._fok_sign_agreements, maker._fok_sign_disagreements)
    # An RFQ fill on the same option, at the SAME penny price, on the OPPOSITE
    # side. This is the collision the review describes.
    maker.on_trade(option, price, -5, counterparty_id=9)
    after = (maker._fok_sign_agreements, maker._fok_sign_disagreements)
    check("RFQ at the FOK price did not cast a vote", before == after,
          f"{before} -> {after}  <-- the collision is REAL if this fails")

    print("-" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for failure in FAILURES:
            print(" ", failure)
    else:
        print("every disputed claim tested; none reproduced")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
