"""Does the FOK side-convention detector actually work?

`FokOrder.order_type` is not documented as to whose side it names. We assume the
counterparty's -- they sent the order, so BUY means they buy and we sell -- and
verify that assumption against fills, because reading it backwards means taking
the wrong side of every FOK we believed had edge.

Nothing else in the suite exercises the backwards world, so this builds it
explicitly: a fake exchange with a switchable convention, which reports fills the
way a real one would (a signed quantity) and lets us assert that the market maker
notices, flips, and then stays flipped.

Also pinned here are the two failure modes that made the original detector unsafe:

  1. RFQ contamination. `on_trade` fires for RFQ fills too. Keyed only by
     option_id, an RFQ fill would consume the pending FOK expectation and cast a
     vote about a trade that was never a FOK -- able to flip a CORRECT convention
     into a wrong one. RFQ volume is roughly 10x FOK volume over ~14 options, so
     this is the common case, not a corner.
  2. Unbounded tuition. While the convention is unproven we should be risking a
     fraction of the usual budget, so that being wrong is survivable.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import traceback

import market_maker
from market_maker import (
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

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL {label} {detail}")
    else:
        print(f"  ok   {label}")


def build_market_maker(options: list[BinaryOption]) -> MarketMaker:
    underlyings = [
        Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, 2.25),
        Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, 900.0),
        Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, 850.0),
    ]
    market_maker = MarketMaker(underlyings, options, 1_000.0)
    market_maker.warm_up(MarketHistory(values_by_underlying_id={
        FED_FUNDS_RATE_UNDERLYING_ID: tuple(2.0 + 0.25 * ((i // 7) % 3) for i in range(200)),
        AJARAI_UNDERLYING_ID: tuple(900.0 * (1.0 + 0.001 * ((i * 7) % 11 - 5)) for i in range(200)),
        THERIODIC_UNDERLYING_ID: tuple(850.0 * (1.0 + 0.001 * ((i * 5) % 13 - 6)) for i in range(200)),
    }))
    return market_maker


def make_options() -> list[BinaryOption]:
    ajarai_leg = OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=1.0)
    return [
        BinaryOption(legs=(ajarai_leg,), option_id=i + 1, steps_until_expiry=4,
                     strike=round(900.0 * (0.90 + 0.05 * i), 2))
        for i in range(6)
    ]


def offer_fok(market_maker: MarketMaker, option: BinaryOption, counterparty_buys: bool,
              price: float, quantity: int, exchange_matches_our_reading: bool) -> int:
    """Send one FOK and, if accepted, report the fill the way an exchange would.

    `exchange_matches_our_reading` selects the world: True means order_type names
    the counterparty's side (our default), False means it names ours. Either way
    the fill is reported as a signed quantity from OUR point of view, which is
    the ground truth the detector is supposed to recover.
    """
    order_type = OrderType.BUY if counterparty_buys else OrderType.SELL
    if not exchange_matches_our_reading:
        order_type = OrderType.SELL if counterparty_buys else OrderType.BUY

    order = FokOrder(counterparty_id=2, option_id=option.option_id,
                     order_type=order_type, price=price, quantity=quantity)
    if not market_maker.respond_to_fok(option, order):
        return 0
    # Counterparty buys => we are short.
    signed = -quantity if counterparty_buys else quantity
    market_maker.on_trade(option, price, signed, counterparty_id=2)
    return signed


def drive(market_maker: MarketMaker, options: list[BinaryOption],
          exchange_matches_our_reading: bool, rounds: int = 24) -> int:
    """Feed a two-sided book and count the fills.

    Both sides AND both directions of mispricing are needed. A backwards reader
    computes edge with the sign inverted, so it rejects everything that is truly
    profitable and accepts only what destroys value -- feed it profitable flow
    alone and it simply declines all of it, learning nothing. The orders it will
    bite on are the bad ones, so those have to be in the mix for the detector to
    ever see a fill and notice the sign is wrong.
    """
    accepted = 0
    for index in range(rounds):
        option = options[index % len(options)]
        theo = market_maker.price_option(option)
        counterparty_buys = (index // 2) % 2 == 0
        above_theo = index % 2 == 0
        price = round(min(0.98, theo + 0.25), 2) if above_theo else round(max(0.02, theo - 0.25), 2)
        if offer_fok(market_maker, option, counterparty_buys, price, 3, exchange_matches_our_reading):
            accepted += 1
    return accepted


def main() -> None:
    print("1. exchange agrees with our default reading -> never flips")
    options = make_options()
    market_maker = build_market_maker(options)
    accepted = drive(market_maker, options, exchange_matches_our_reading=True)
    check("accepts profitable flow", accepted > 0, f"accepted {accepted}")
    check("stays on default convention", market_maker._fok_describes_counterparty_side is True)
    check("records agreements", market_maker._fok_sign_agreements > 0,
          f"agreements={market_maker._fok_sign_agreements}")
    check("no disagreements", market_maker._fok_sign_disagreements == 0,
          f"disagreements={market_maker._fok_sign_disagreements}")

    print("2. exchange uses the OPPOSITE reading -> detects and flips")
    options = make_options()
    market_maker = build_market_maker(options)
    drive(market_maker, options, exchange_matches_our_reading=False)
    check("flipped to the other convention", market_maker._fok_describes_counterparty_side is False)

    print("3. after flipping, it is stable in the world it flipped into")
    before = market_maker._fok_describes_counterparty_side
    drive(market_maker, options, exchange_matches_our_reading=False, rounds=20)
    check("no second flip", market_maker._fok_describes_counterparty_side is before)

    print("4. RFQ fills must not vote (the contamination bug)")
    options = make_options()
    market_maker = build_market_maker(options)
    option = options[0]
    theo = market_maker.price_option(option)
    fok_price = round(max(0.02, theo - 0.25), 2)
    order = FokOrder(counterparty_id=2, option_id=option.option_id,
                     order_type=OrderType.SELL, price=fok_price, quantity=3)
    check("accepts the seed order", market_maker.respond_to_fok(option, order))
    # An RFQ prints on the SAME option, at a different price, on the wrong side.
    rfq_price = round(min(0.99, fok_price + 0.10), 2)
    market_maker.on_trade(option, rfq_price, -4, counterparty_id=9)
    check("RFQ cast no vote",
          market_maker._fok_sign_agreements == 0 and market_maker._fok_sign_disagreements == 0,
          f"agree={market_maker._fok_sign_agreements} disagree={market_maker._fok_sign_disagreements}")
    check("expectation survives the RFQ", option.option_id in market_maker._pending_fok_by_option_id)
    # Now the genuine FOK fill arrives and should be the one that counts.
    market_maker.on_trade(option, fok_price, 3, counterparty_id=2)
    check("real FOK fill voted", market_maker._fok_sign_agreements == 1,
          f"agree={market_maker._fok_sign_agreements}")

    print("5. unproven convention risks less than a confirmed one")
    options = make_options()
    cautious = build_market_maker(options)
    confident = build_market_maker(options)
    confident._fok_sign_agreements = 99  # pretend the convention is established

    # Deep in the money, so a buy locks a lot of collateral per contract and the
    # BUDGET is what binds -- not the position cap, and not the edge hurdle.
    option = options[0]
    theo = confident.price_option(option)
    price = round(max(0.02, theo - 0.12), 2)
    # Read the budgets off the LIVE constants instead of hard-coding them. As
    # literals, this test kept passing its own setup assertion while the
    # behaviour it guards drifted underneath: raising _FOK_RISK_FRACTION from
    # 0.120 to 0.250 doubled the risk taken on an unproven convention, and the
    # literals disguised which of the two budgets had actually moved.
    confident_budget = market_maker._FOK_RISK_FRACTION * confident._available_capital()
    cautious_budget = market_maker._FOK_UNPROVEN_RISK_SCALE * confident_budget
    # Size the clip to sit midway between the two budgets, so the ONLY thing that
    # can separate the two makers is the unproven-convention discount.
    quantity = min(
        int(((cautious_budget + confident_budget) / 2.0) / price),
        market_maker._MAXIMUM_POSITION_PER_OPTION,
    )
    big = FokOrder(counterparty_id=2, option_id=option.option_id,
                   order_type=OrderType.SELL, price=price, quantity=quantity)
    worst_case = quantity * price
    check("test is set up to isolate the budget",
          cautious_budget < worst_case <= confident_budget,
          f"worst_case={worst_case:.1f} cautious={cautious_budget:.1f} confident={confident_budget:.1f}")
    cautious_takes_it = cautious.respond_to_fok(option, big)
    confident_takes_it = confident.respond_to_fok(option, big)
    check("confident maker takes the large clip", confident_takes_it)
    check("unproven maker declines the large clip", not cautious_takes_it,
          "an unverified side convention should not be tested at full size")

    print("6. the venue may not report the FOK sender's id -- detector must survive it")
    # `on_trade`'s counterparty_id is undocumented as to WHOSE id it is. Our own
    # session simulator reports a fixed venue id on every fill, FOK included, so a
    # detector that requires the sender's id to match discards every genuine FOK
    # fill, never confirms, and spends the session on the reduced unproven budget
    # -- worth 1.6 of mean PnL. Nothing else in this suite covers that world,
    # which is exactly how the hard filter got committed. So: same flow as case 2,
    # but the venue stamps a constant id that never equals the sender's.
    options = make_options()
    market_maker = build_market_maker(options)
    for index in range(24):
        option = options[index % len(options)]
        theo = market_maker.price_option(option)
        counterparty_buys = (index // 2) % 2 == 0
        price = (round(min(0.98, theo + 0.25), 2) if index % 2 == 0
                 else round(max(0.02, theo - 0.25), 2))
        order = FokOrder(counterparty_id=2, option_id=option.option_id,
                         order_type=OrderType.SELL if counterparty_buys else OrderType.BUY,
                         price=price, quantity=3)
        if market_maker.respond_to_fok(option, order):
            signed = -3 if counterparty_buys else 3
            market_maker.on_trade(option, price, signed, counterparty_id=1)  # venue id, not sender
    check("votes were still cast",
          market_maker._fok_sign_agreements + market_maker._fok_sign_disagreements > 0,
          f"agree={market_maker._fok_sign_agreements} "
          f"disagree={market_maker._fok_sign_disagreements}")
    check("still flips in a backwards venue",
          market_maker._fok_describes_counterparty_side is False)

    print("7. once the venue proves it reports senders, the id is enforced again")
    options = make_options()
    market_maker = build_market_maker(options)
    option = options[0]
    theo = market_maker.price_option(option)
    price = round(max(0.02, theo - 0.25), 2)
    order = FokOrder(counterparty_id=2, option_id=option.option_id,
                     order_type=OrderType.SELL, price=price, quantity=3)
    check("accepts the first order", market_maker.respond_to_fok(option, order))
    market_maker.on_trade(option, price, 3, counterparty_id=2)  # sender id passed through
    check("learned that ids are reported", market_maker._fok_counterparty_id_is_reported)
    # Now a same-price, same-size RFQ from a stranger must be rejected on identity
    # alone -- the one case price and size cannot separate.
    order = FokOrder(counterparty_id=2, option_id=option.option_id,
                     order_type=OrderType.SELL, price=price, quantity=3)
    check("accepts the second order", market_maker.respond_to_fok(option, order))
    votes = (market_maker._fok_sign_agreements, market_maker._fok_sign_disagreements)
    market_maker.on_trade(option, price, -3, counterparty_id=9)
    check("stranger at same price and size cast no vote",
          (market_maker._fok_sign_agreements, market_maker._fok_sign_disagreements) == votes,
          f"{votes} -> ({market_maker._fok_sign_agreements}, "
          f"{market_maker._fok_sign_disagreements})")

    print("-" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S)")
        for failure in FAILURES:
            print(" ", failure)
    else:
        print("all FOK convention checks passed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("EXCEPTION RAISED -- this would score zero")
