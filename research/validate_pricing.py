"""Validate the analytic pricer against Monte Carlo run through the real simulator.

Not part of the submission -- this only exists to prove `price_option_from_parameters`
matches `MarketParameters.advance_step`, which is the ground truth the grader uses.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import math
import random

from market_maker import (
    AJARAI_NAME,
    AJARAI_UNDERLYING_ID,
    FED_FUNDS_RATE_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID,
    THERIODIC_NAME,
    THERIODIC_UNDERLYING_ID,
    BinaryOption,
    MarketMaker,
    MarketParameters,
    OptionLeg,
    Underlying,
)

NUM_TRIALS = 200_000

BASE_PARAMETERS = MarketParameters(
    ajarai_drift=0.004,
    ajarai_idio_std_dev=0.030,
    ajarai_rate_beta=-0.08,
    ajarai_sector_beta=1.10,
    rate_down_probability=0.20,
    rate_reversion_strength=0.15,
    rate_up_probability=0.25,
    sector_std_dev=0.020,
    theriodic_drift=-0.002,
    theriodic_idio_std_dev=0.045,
    theriodic_rate_beta=-0.12,
    theriodic_sector_beta=0.80,
)

# Deliberately awkward: high reversion + a starting rate near the zero floor, to
# exercise the probability clamping and the `max(..., 0.0)` barrier.
STRESS_PARAMETERS = MarketParameters(
    ajarai_drift=0.010,
    ajarai_idio_std_dev=0.060,
    ajarai_rate_beta=-0.25,
    ajarai_sector_beta=1.50,
    rate_down_probability=0.45,
    rate_reversion_strength=0.55,
    rate_up_probability=0.45,
    sector_std_dev=0.035,
    theriodic_drift=0.006,
    theriodic_idio_std_dev=0.025,
    theriodic_rate_beta=0.05,
    theriodic_sector_beta=-0.60,  # negative sector beta => negative correlation
)


def build_market_maker(rate: float, ajarai: float, theriodic: float) -> MarketMaker:
    underlyings = [
        Underlying(name=FED_FUNDS_RATE_NAME, underlying_id=FED_FUNDS_RATE_UNDERLYING_ID, value=rate),
        Underlying(name=AJARAI_NAME, underlying_id=AJARAI_UNDERLYING_ID, value=ajarai),
        Underlying(name=THERIODIC_NAME, underlying_id=THERIODIC_UNDERLYING_ID, value=theriodic),
    ]
    return MarketMaker(underlying_initial_state=underlyings, option_initial_state=[], cash_balance=1_000.0)


def monte_carlo_price(
    parameters: MarketParameters, option: BinaryOption, initial_values: dict[int, float], num_trials: int
) -> float:
    payoff_total = 0.0
    for _ in range(num_trials):
        values = dict(initial_values)
        for _ in range(option.steps_until_expiry):
            values = parameters.advance_step(values)
        payoff_total += option.expiry_valuation(values)
    return payoff_total / num_trials


def run_case(
    label: str, parameters: MarketParameters, initial_values: dict[int, float], option: BinaryOption
) -> tuple[str, float, float, float, float, bool]:
    market_maker = build_market_maker(
        initial_values[FED_FUNDS_RATE_UNDERLYING_ID],
        initial_values[AJARAI_UNDERLYING_ID],
        initial_values[THERIODIC_UNDERLYING_ID],
    )
    analytic = market_maker.price_option_from_parameters(parameters, option)

    random.seed(abs(hash(label)) % (2**31))
    simulated = monte_carlo_price(parameters, option, initial_values, NUM_TRIALS)

    standard_error = math.sqrt(max(simulated * (1.0 - simulated), 1e-12) / NUM_TRIALS)
    difference = analytic - simulated
    z_score = difference / standard_error if standard_error > 0 else 0.0
    # Allow 4 sigma, plus a small floor for the tiny bias from the simulator's
    # per-step `round(value, 2)`.
    passed = abs(difference) <= max(4.0 * standard_error, 0.002)
    return label, analytic, simulated, difference, z_score, passed


def main() -> None:
    base_values = {
        FED_FUNDS_RATE_UNDERLYING_ID: 2.50,
        AJARAI_UNDERLYING_ID: 900.0,
        THERIODIC_UNDERLYING_ID: 850.0,
    }
    stress_values = {
        FED_FUNDS_RATE_UNDERLYING_ID: 0.25,  # right next to the zero floor
        AJARAI_UNDERLYING_ID: 1_200.0,
        THERIODIC_UNDERLYING_ID: 1_150.0,
    }

    rate_leg = OptionLeg(underlying_id=FED_FUNDS_RATE_UNDERLYING_ID, weight=1.0)
    ajarai_leg = OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=1.0)
    theriodic_leg = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=1.0)
    short_theriodic_leg = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=-1.0)

    cases: list[tuple[str, MarketParameters, dict[int, float], BinaryOption]] = [
        # --- rate only: exercises the lattice, including exact-tie strikes ---
        ("FED >= 2.50 in 1d (exact tie)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(rate_leg,), option_id=1, steps_until_expiry=1, strike=2.50)),
        ("FED >= 2.75 in 3d", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(rate_leg,), option_id=2, steps_until_expiry=3, strike=2.75)),
        ("FED >= 3.00 in 5d", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(rate_leg,), option_id=3, steps_until_expiry=5, strike=3.00)),
        ("FED >= 0.25 in 4d (zero floor)", STRESS_PARAMETERS, stress_values,
         BinaryOption(legs=(rate_leg,), option_id=4, steps_until_expiry=4, strike=0.25)),
        # --- single company: rate-conditioned lognormal ---
        ("AJR >= 900 in 1d (atm)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(ajarai_leg,), option_id=5, steps_until_expiry=1, strike=900.0)),
        ("AJR >= 1000 in 5d (otm)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(ajarai_leg,), option_id=6, steps_until_expiry=5, strike=1_000.0)),
        ("THR >= 700 in 4d (itm)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(theriodic_leg,), option_id=7, steps_until_expiry=4, strike=700.0)),
        ("AJR >= 1500 in 6d (deep otm)", STRESS_PARAMETERS, stress_values,
         BinaryOption(legs=(ajarai_leg,), option_id=8, steps_until_expiry=6, strike=1_500.0)),
        ("-THR >= -900 in 3d (negative weight)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(short_theriodic_leg,), option_id=9, steps_until_expiry=3, strike=-900.0)),
        # --- spreads: exercises the shared sector shock / correlation ---
        ("AJR - THR >= 0 in 2d", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(ajarai_leg, short_theriodic_leg), option_id=10, steps_until_expiry=2, strike=0.0)),
        ("AJR - THR >= 0 in 7d", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(ajarai_leg, short_theriodic_leg), option_id=11, steps_until_expiry=7, strike=0.0)),
        ("AJR - THR >= 0 in 5d (neg corr)", STRESS_PARAMETERS, stress_values,
         BinaryOption(legs=(ajarai_leg, short_theriodic_leg), option_id=12, steps_until_expiry=5, strike=0.0)),
        # --- general two-company strike: exercises the Simpson fallback ---
        ("AJR - THR >= 100 in 4d (quadrature)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(ajarai_leg, short_theriodic_leg), option_id=13, steps_until_expiry=4, strike=100.0)),
        ("AJR + THR >= 1800 in 3d (quadrature)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(ajarai_leg, theriodic_leg), option_id=14, steps_until_expiry=3, strike=1_800.0)),
        # --- mixed rate + company legs ---
        ("AJR + 100*FED >= 1150 in 3d", BASE_PARAMETERS, base_values,
         BinaryOption(
             legs=(ajarai_leg, OptionLeg(underlying_id=FED_FUNDS_RATE_UNDERLYING_ID, weight=100.0)),
             option_id=15, steps_until_expiry=3, strike=1_150.0)),
        # --- degenerate: already expired ---
        ("AJR >= 800 in 0d (expired)", BASE_PARAMETERS, base_values,
         BinaryOption(legs=(ajarai_leg,), option_id=16, steps_until_expiry=0, strike=800.0)),
    ]

    print(f"{'case':<40} {'analytic':>10} {'monte carlo':>12} {'diff':>9} {'z':>7}  result")
    print("-" * 90)
    failures = 0
    for label, parameters, initial_values, option in cases:
        label, analytic, simulated, difference, z_score, passed = run_case(label, parameters, initial_values, option)
        if not passed:
            failures += 1
        print(
            f"{label:<40} {analytic:>10.5f} {simulated:>12.5f} {difference:>+9.5f} "
            f"{z_score:>+7.2f}  {'ok' if passed else 'FAIL'}"
        )
    print("-" * 90)
    print(f"{len(cases) - failures}/{len(cases)} passed ({NUM_TRIALS:,} trials each)")


if __name__ == "__main__":
    main()
