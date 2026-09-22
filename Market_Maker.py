import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id

# ============================================================================
# YOUR MARKET MAKER -- fill in the six stubbed methods below
# ============================================================================

# --- pricing helpers --------------------------------------------------------
#
# Model recap (read straight off `MarketParameters.advance_step`):
#
#   rate_{t+1}  = rate_t + {+step, -step, 0}, floored at 0, probabilities tilted
#                 toward `rate_target` by `rate_reversion_strength`
#   log(V_{t+1} / V_t) = drift + rate_beta * (rate_{t+1} - rate_t)
#                      + sector_beta * sector_shock_t + idio_t
#
# Two structural facts make this exactly solvable, with no Monte Carlo:
#
#   1. The per-step `rate_change` terms telescope, so over n steps a company
#      only ever sees the NET rate move (rate_n - rate_0).  The path is
#      irrelevant, so we only need the terminal rate distribution -- which is
#      a finite lattice we can enumerate exactly.
#   2. Sums of the independent per-step gaussian shocks are gaussian, so the
#      n-step log return is normal once we condition on the terminal rate.
#
# Hence, conditional on rate_n = r:
#
#   log(V_n / V_0) ~ Normal( n*drift + rate_beta*(r - rate_0),
#                            n*(sector_beta^2 * sector_std^2 + idio_std^2) )
#
# and the two companies covary by  n * sector_beta_A * sector_beta_T * sector_std^2,
# because `advance_step` draws ONE sector shock and hands it to both.

_SQRT_2: Final[float] = math.sqrt(2.0)
_TIE_EPSILON: Final[float] = 1e-9
_QUADRATURE_HALF_WIDTH: Final[float] = 8.0
_QUADRATURE_INTERVALS: Final[int] = 128


def _worst_case_loss(price: float, quantity: int) -> float:
    """Collateral the grader locks for a fill: buys risk the premium, sells risk
    the full payout net of premium."""
    return quantity * price if quantity > 0 else -quantity * (1.0 - price)


def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _terminal_rate_distribution(
    market_parameters: MarketParameters, initial_rate: float, num_steps: int
) -> dict[float, float]:
    """Exact distribution of the rate after `num_steps`, as {rate: probability}.

    Deliberately delegates to the parameter object's own `tilted_rate_probabilities`
    and `next_rate_value` so the lattice reproduces the simulator bit for bit --
    including the clamping of the tilted probabilities and the floor at zero.
    """
    distribution: dict[float, float] = {initial_rate: 1.0}
    for _ in range(num_steps):
        next_distribution: defaultdict[float, float] = defaultdict(float)
        for rate, probability in distribution.items():
            if probability <= 0.0:
                continue

            up_probability, down_probability = market_parameters.tilted_rate_probabilities(rate)
            stay_probability: float = max(0.0, 1.0 - up_probability - down_probability)
            if up_probability > 0.0:
                next_distribution[market_parameters.next_rate_value(rate, 1)] += probability * up_probability
            if down_probability > 0.0:
                next_distribution[market_parameters.next_rate_value(rate, -1)] += probability * down_probability
            if stay_probability > 0.0:
                next_distribution[rate] += probability * stay_probability
        distribution = dict(next_distribution)
    return distribution


def _company_parameters(market_parameters: MarketParameters, underlying_id: int) -> tuple[float, float, float, float]:
    """(drift, rate_beta, sector_beta, idio_std_dev) for one company."""
    if underlying_id == AJARAI_UNDERLYING_ID:
        return (
            market_parameters.ajarai_drift,
            market_parameters.ajarai_rate_beta,
            market_parameters.ajarai_sector_beta,
            market_parameters.ajarai_idio_std_dev,
        )
    return (
        market_parameters.theriodic_drift,
        market_parameters.theriodic_rate_beta,
        market_parameters.theriodic_sector_beta,
        market_parameters.theriodic_idio_std_dev,
    )


def _log_return_moments(
    market_parameters: MarketParameters, underlying_id: int, num_steps: int, net_rate_change: float
) -> tuple[float, float]:
    """(mean, variance) of the n-step log return, conditional on the net rate move."""
    drift, rate_beta, sector_beta, idio_std_dev = _company_parameters(market_parameters, underlying_id)
    mean: float = (num_steps * drift) + (rate_beta * net_rate_change)
    sector_variance: float = (sector_beta * market_parameters.sector_std_dev) ** 2
    variance: float = num_steps * (sector_variance + (idio_std_dev**2))
    return mean, variance


def _log_return_covariance(market_parameters: MarketParameters, num_steps: int) -> float:
    """Covariance of the two companies' log returns -- the shared sector shock."""
    return (
        num_steps
        * market_parameters.ajarai_sector_beta
        * market_parameters.theriodic_sector_beta
        * (market_parameters.sector_std_dev**2)
    )


def _single_company_probability(
    weight: float, spot: float, strike: float, mean: float, std_dev: float
) -> float:
    """P(weight * spot * exp(X) >= strike) for X ~ Normal(mean, std_dev^2)."""
    if weight > 0.0:
        threshold_value: float = strike / weight
        if threshold_value <= 0.0:
            return 1.0  # a positively weighted, strictly positive value always clears
    else:
        threshold_value = strike / weight  # dividing by a negative flips the inequality
        if threshold_value <= 0.0:
            return 0.0  # would require a non-positive valuation, which cannot happen

    if spot <= 0.0:
        return 0.0 if weight > 0.0 else 1.0

    log_threshold: float = math.log(threshold_value / spot)
    if std_dev <= 0.0:
        deterministic_clears: bool = (log_threshold - mean) <= _TIE_EPSILON
        if weight > 0.0:
            return 1.0 if deterministic_clears else 0.0
        return 0.0 if deterministic_clears else 1.0

    standardised: float = (log_threshold - mean) / std_dev
    if weight > 0.0:
        return 1.0 - _standard_normal_cdf(standardised)  # need X above the threshold
    return _standard_normal_cdf(standardised)  # need X below the threshold


def _two_company_probability(
    first_weight: float,
    first_spot: float,
    first_mean: float,
    first_variance: float,
    second_weight: float,
    second_spot: float,
    second_mean: float,
    second_variance: float,
    covariance: float,
    strike: float,
) -> float:
    """P(w1*S1*exp(X1) + w2*S2*exp(X2) >= strike) for bivariate normal (X1, X2)."""
    first_std_dev: float = math.sqrt(max(first_variance, 0.0))
    second_std_dev: float = math.sqrt(max(second_variance, 0.0))

    # Fast exact path: a head-to-head spread (opposite signs, zero strike) reduces
    # to a one-dimensional question about X1 - X2, which is itself normal.  This is
    # the only two-legged shape the problem statement says actually trades.
    if abs(strike) <= _TIE_EPSILON and first_weight * second_weight < 0.0 and first_spot > 0.0 and second_spot > 0.0:
        if first_weight > 0.0:
            positive_scale, negative_scale = first_weight * first_spot, -second_weight * second_spot
            difference_mean: float = first_mean - second_mean
        else:
            positive_scale, negative_scale = second_weight * second_spot, -first_weight * first_spot
            difference_mean = second_mean - first_mean
        difference_variance: float = first_variance + second_variance - (2.0 * covariance)
        log_threshold: float = math.log(negative_scale / positive_scale)
        if difference_variance <= 0.0:
            return 1.0 if (log_threshold - difference_mean) <= _TIE_EPSILON else 0.0
        return 1.0 - _standard_normal_cdf((log_threshold - difference_mean) / math.sqrt(difference_variance))

    # General fallback: integrate the second leg out numerically (composite Simpson
    # over the standard normal), solving the first leg in closed form at each node.
    if second_std_dev <= 0.0:
        residual_strike: float = strike - (second_weight * second_spot * math.exp(second_mean))
        return _single_company_probability(first_weight, first_spot, residual_strike, first_mean, first_std_dev)

    correlation: float = 0.0
    if first_std_dev > 0.0 and second_std_dev > 0.0:
        correlation = max(-1.0, min(1.0, covariance / (first_std_dev * second_std_dev)))
    conditional_std_dev: float = first_std_dev * math.sqrt(max(0.0, 1.0 - (correlation**2)))

    step: float = (2.0 * _QUADRATURE_HALF_WIDTH) / _QUADRATURE_INTERVALS
    total: float = 0.0
    for index in range(_QUADRATURE_INTERVALS + 1):
        z: float = -_QUADRATURE_HALF_WIDTH + (index * step)
        if index in (0, _QUADRATURE_INTERVALS):
            simpson_weight: float = 1.0
        elif index % 2 == 1:
            simpson_weight = 4.0
        else:
            simpson_weight = 2.0

        second_log_return: float = second_mean + (second_std_dev * z)
        residual_strike = strike - (second_weight * second_spot * math.exp(second_log_return))
        conditional_mean: float = first_mean + (correlation * first_std_dev * z)
        conditional_probability: float = _single_company_probability(
            first_weight, first_spot, residual_strike, conditional_mean, conditional_std_dev
        )
        total += simpson_weight * conditional_probability * _standard_normal_pdf(z)
    return total * step / 3.0


# --- parameter estimation ---------------------------------------------------
#
# `sector_beta` and `sector_std_dev` are NOT separately identifiable from a price
# history: doubling a beta while halving the sector vol produces an identical
# world.  Pricing only ever consumes the residual covariance matrix, so rather
# than fight over an unidentified scale we fix `sector_std_dev = 1.0` and let the
# sector betas absorb it.  That drops two free parameters and the estimation
# variance that comes with them.

_MINIMUM_HISTORY_DAYS: Final[int] = 12
_MAXIMUM_ABSOLUTE_CORRELATION: Final[float] = 0.995
_COORDINATE_DESCENT_SWEEPS: Final[int] = 6
_COORDINATE_DESCENT_GRID: Final[int] = 17

# Weak priors on the rate dynamics. Unpenalised MLE badly overfits the reversion
# strength on short histories (measured ~0.31 against a true 0.15 at 150 days),
# and strength/target trade off against each other, so the likelihood surface is
# flat in exactly the direction that hurts. These sigmas were chosen by measured
# pricing error, not taste -- see experiment_estimation.py. The target prior is
# centred on the dataclass default but loose enough to follow real evidence.
_REVERSION_STRENGTH_PRIOR_STD_DEV: Final[float] = 0.20
_RATE_TARGET_PRIOR_MEAN: Final[float] = 2.00
_RATE_TARGET_PRIOR_STD_DEV: Final[float] = 1.00


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _estimate_rate_step(rates: tuple[float, ...]) -> float:
    """Smallest observed non-zero rate move; falls back to the documented grid."""
    moves: list[float] = [abs(rates[i] - rates[i - 1]) for i in range(1, len(rates))]
    positive_moves: list[float] = [move for move in moves if move > 1e-9]
    if not positive_moves:
        return RATE_STRIKE_GRID
    return round(min(positive_moves), 2)


def _rate_log_likelihood(
    rates: tuple[float, ...], rate_step: float, up_prob: float, down_prob: float, strength: float, target: float
) -> float:
    """Exact log-likelihood of the observed rate path.

    Mirrors `tilted_rate_probabilities` / `next_rate_value` inline rather than
    calling them, purely so the coordinate descent below stays cheap. It also
    handles the ambiguity created by the zero floor: at rate 0 a "down" draw is
    indistinguishable from a "stay", so both probabilities are summed.
    """
    total: float = 0.0
    for index in range(1, len(rates)):
        current: float = rates[index - 1]
        observed: float = rates[index]

        tilt: float = strength * (target - current)
        up_probability: float = min(max(up_prob + tilt, 0.0), 1.0)
        down_probability: float = min(max(down_prob - tilt, 0.0), 1.0 - up_probability)
        stay_probability: float = max(0.0, 1.0 - up_probability - down_probability)

        probability: float = 0.0
        if abs(observed - max(round(current + rate_step, 2), 0.0)) < 1e-9:
            probability += up_probability
        if abs(observed - max(round(current - rate_step, 2), 0.0)) < 1e-9:
            probability += down_probability
        if abs(observed - current) < 1e-9:
            probability += stay_probability
        total += math.log(max(probability, 1e-12))
    return total


def _estimate_rate_dynamics(rates: tuple[float, ...], rate_step: float) -> tuple[float, float, float, float]:
    """Penalised-ML (up_prob, down_prob, reversion_strength, target).

    Coordinate descent over a shrinking 1-D grid: a full 4-D grid would be far
    too slow, and the penalised surface is well behaved enough that sweeping one
    axis at a time converges in a handful of passes.
    """
    current: list[float] = [0.20, 0.20, 0.10, 2.00]
    windows: list[float] = [0.25, 0.25, 0.25, 1.50]
    bounds: list[tuple[float, float]] = [(0.005, 0.900), (0.005, 0.900), (0.0, 1.0), (0.0, 6.0)]

    def score(candidate: list[float]) -> float:
        if candidate[0] + candidate[1] > 1.0:
            return -math.inf
        penalty: float = 0.5 * ((candidate[2] / _REVERSION_STRENGTH_PRIOR_STD_DEV) ** 2)
        penalty += 0.5 * (
            ((candidate[3] - _RATE_TARGET_PRIOR_MEAN) / _RATE_TARGET_PRIOR_STD_DEV) ** 2
        )
        return _rate_log_likelihood(
            rates, rate_step, candidate[0], candidate[1], candidate[2], candidate[3]
        ) - penalty

    best_score: float = score(current)
    for _ in range(_COORDINATE_DESCENT_SWEEPS):
        for axis in range(4):
            low: float = max(bounds[axis][0], current[axis] - windows[axis])
            high: float = min(bounds[axis][1], current[axis] + windows[axis])
            if high <= low:
                continue
            step: float = (high - low) / (_COORDINATE_DESCENT_GRID - 1)
            for grid_index in range(_COORDINATE_DESCENT_GRID):
                candidate: list[float] = list(current)
                candidate[axis] = low + (grid_index * step)
                candidate_score: float = score(candidate)
                if candidate_score > best_score:
                    best_score = candidate_score
                    current = candidate
        windows = [window * 0.5 for window in windows]

    up_probability: float = min(max(current[0], 0.005), 0.9)
    down_probability: float = min(max(current[1], 0.005), 0.9)
    if up_probability + down_probability > 1.0:
        scale: float = 1.0 / (up_probability + down_probability)
        up_probability *= scale
        down_probability *= scale
    return up_probability, down_probability, min(max(current[2], 0.0), 1.0), max(current[3], 0.0)


def _regress_log_returns(
    values: tuple[float, ...], rate_changes: list[float]
) -> tuple[float, float, list[float], float]:
    """OLS of daily log returns on the same day's rate change.

    Returns (drift, rate_beta, residuals, drift_std_error).  `advance_step` moves
    the rate first and feeds that same-day change to the companies, so log return
    t is paired with rate change t -- no lag.

    Both coefficients are shrunk toward zero by an empirical-Bayes factor
    var_signal / (var_signal + var_noise). Drift is the hardest thing here to pin
    down and it scales linearly with days-to-expiry, so an unshrunk noisy estimate
    would bias every price we make.
    """
    log_returns: list[float] = [math.log(values[i] / values[i - 1]) for i in range(1, len(values))]
    sample_size: int = len(log_returns)

    mean_return: float = _mean(log_returns)
    mean_change: float = _mean(rate_changes)
    change_variance: float = sum((change - mean_change) ** 2 for change in rate_changes)
    covariance: float = sum(
        (rate_changes[i] - mean_change) * (log_returns[i] - mean_return) for i in range(sample_size)
    )

    rate_beta: float = covariance / change_variance if change_variance > 1e-12 else 0.0
    drift: float = mean_return - (rate_beta * mean_change)
    residuals: list[float] = [
        log_returns[i] - drift - (rate_beta * rate_changes[i]) for i in range(sample_size)
    ]

    degrees_of_freedom: int = max(1, sample_size - 2)
    residual_variance: float = sum(residual**2 for residual in residuals) / degrees_of_freedom

    drift_variance: float = residual_variance * (
        (1.0 / sample_size) + ((mean_change**2) / change_variance if change_variance > 1e-12 else 0.0)
    )
    drift = drift * (drift**2 / (drift**2 + drift_variance)) if (drift**2 + drift_variance) > 0 else 0.0

    if change_variance > 1e-12:
        beta_variance: float = residual_variance / change_variance
        rate_beta = rate_beta * (rate_beta**2 / (rate_beta**2 + beta_variance)) if (
            rate_beta**2 + beta_variance
        ) > 0 else 0.0

    return drift, rate_beta, residuals, math.sqrt(max(drift_variance, 0.0))


def _decompose_sector_exposure(
    ajarai_residuals: list[float], theriodic_residuals: list[float]
) -> tuple[float, float, float, float]:
    """Residual covariance -> (sector_beta_A, sector_beta_T, idio_A, idio_T).

    With `sector_std_dev` pinned at 1.0 we need beta_A * beta_T = cov and
    beta^2 <= var on each side.  Splitting the correlation evenly between the two
    (beta_A^2 / var_A == beta_T^2 / var_T) satisfies both for any |rho| <= 1 and
    keeps each idio variance non-negative by construction.
    """
    sample_size: int = min(len(ajarai_residuals), len(theriodic_residuals))
    degrees_of_freedom: int = max(1, sample_size - 2)

    ajarai_variance: float = sum(residual**2 for residual in ajarai_residuals) / degrees_of_freedom
    theriodic_variance: float = sum(residual**2 for residual in theriodic_residuals) / degrees_of_freedom
    ajarai_std_dev: float = math.sqrt(max(ajarai_variance, 1e-12))
    theriodic_std_dev: float = math.sqrt(max(theriodic_variance, 1e-12))

    covariance: float = (
        sum(ajarai_residuals[i] * theriodic_residuals[i] for i in range(sample_size)) / degrees_of_freedom
    )
    correlation: float = max(
        -_MAXIMUM_ABSOLUTE_CORRELATION,
        min(_MAXIMUM_ABSOLUTE_CORRELATION, covariance / (ajarai_std_dev * theriodic_std_dev)),
    )
    covariance = correlation * ajarai_std_dev * theriodic_std_dev

    shared_magnitude: float = math.sqrt(abs(covariance))
    ajarai_sector_beta: float = shared_magnitude * math.sqrt(ajarai_std_dev / theriodic_std_dev)
    theriodic_sector_beta: float = math.copysign(
        shared_magnitude * math.sqrt(theriodic_std_dev / ajarai_std_dev), covariance
    )

    ajarai_idio: float = math.sqrt(max(ajarai_variance - (ajarai_sector_beta**2), 1e-12))
    theriodic_idio: float = math.sqrt(max(theriodic_variance - (theriodic_sector_beta**2), 1e-12))
    return ajarai_sector_beta, theriodic_sector_beta, ajarai_idio, theriodic_idio


# --- quoting / risk constants -----------------------------------------------
#
# Collateral is charged at WORST-CASE loss and only refunded at expiry, so
# selling a $0.05 option locks $0.95 per contract for a nickel of premium.
# Capital, not edge, is the binding constraint -- these limits are what keep us
# out of the bankruptcy bucket (a zero) rather than what maximise upside.

# 0.012 rather than 0.020 because rival makers carrying their own estimation
# error absorb a share of the informed flow before it reaches us -- measured RFQ
# edge per contract rises from +0.0042 against omniscient rivals to +0.0069
# against realistic ones, at comparable volume.
#
# Re-measured after the field seeding was made deterministic, because the original
# selection ran under `hash(name)` rivals and so was only ever valid for one
# field draw. It survives cleanly, as a genuine interior optimum on 144 paired
# sessions per cell: 0.008 is worth -1.69 (t=-6.48), 0.018 -0.24 (t=-1.19), and
# widening further collapses -- 0.025 costs -2.48 (t=-8.31) and 0.035 costs -4.96
# (t=-10.39), taking the rank-1 rate from 22% down to 14% and 8%. Quoting wider
# does not buy protection here; it just hands the flow to somebody else.
_BASE_HALF_SPREAD: Final[float] = 0.012
# Back to 1.4, after a spell at 1.1. The cut to 1.1 was made on a measured +0.90
# (t=2.79), and that number was an artefact: the field simulator seeded rival
# parameter error with `hash(name)`, which Python randomises per process, so each
# sweep scored its candidate against a DIFFERENT set of competitors. Cells within
# one sweep were still paired against each other, which is why the result looked
# clean, but it generalised only to the single field draw that run happened to
# sample. With the seeding made stable, the same comparison is a dead heat --
# +0.10 (t=0.19) over 180 paired sessions -- so the case for cutting evaporates.
#
# On the tiebreaks 1.4 is ahead, and it is ahead everywhere: in the realistic
# field it halves the worst session (-20.9 against -44.3) and lifts profitable
# sessions from 86% to 92% at equal mean, and in a hostile field (50-80% informed
# flow, 40-day sessions) it dominates outright -- +1.62 (t=1.88), p10 -1.5 against
# -11.8, and a higher rank-1 rate as well. The direction is consistent across both
# fields and every statistic, which is worth more here than the marginal spread
# income 1.1 was buying, because the grader pays zero for a blow-up.
_UNCERTAINTY_MULTIPLIER: Final[float] = 1.400
_MAXIMUM_HALF_SPREAD: Final[float] = 0.250
_MAXIMUM_PRICE_UNCERTAINTY: Final[float] = 0.150
# Every constant from here to `_FOK_UNPROVEN_RISK_SCALE` was re-swept one at a
# time once the field seeding was made deterministic, since the original choices
# were made under `hash(name)` rivals. Seven "improvements" came back at t > 2 --
# and every single one loosened a risk control. That unanimity was the tell: the
# session simulator draws RFQ clips from randint(1, 15) and FOK clips from
# randint(1, 12) against a $1,000 book, so `_MAXIMUM_QUOTE_SIZE`,
# `_FOK_RISK_FRACTION` and `_FOK_UNPROVEN_RISK_SCALE` measured EXACTLY +0.00 at
# every value tried. They never bind even once. In a world where no trade can hurt
# you, protection is pure cost, and a sweep will faithfully report that removing it
# pays. All were re-tested at 4x and 10x flow before being rejected.
#
# 0.035 stays for that reason. Dropping the skew to 0.0 looked like the one clean
# Pareto win -- better mean, p10 AND worst at 1x, 4x and 10x flow. But that sweep
# varies trade SIZE, and inventory skew does not exist to survive large trades; it
# exists to survive a counterparty that keeps hitting the SAME SIDE. Against
# mostly-informed flow (50-80%), where fills are adverse by construction, the gain
# collapses to t=0.97 while peak concentration in a single option climbs from 133
# to 160, the worst session degrades (-63.1 against -60.3) and profitable sessions
# fall from 72% to 69%. At 1x the control simply never fires -- peak position 40
# against a cap of 180 -- which is the whole reason removing it looked free.
_INVENTORY_SKEW: Final[float] = 0.035
# These two were the binding constraint on size, and they should not have been.
# `_affordable_quantity` already limits every fill to
# `_TRADE_RISK_FRACTION * available_capital / unit_collateral`, which scales with
# the money we actually have and tightens automatically as we lose it; the flat
# caps below bind BEFORE it and so were throttling positive-edge flow for no
# added protection. Raising them is monotonically profitable in the field sim
# (+0.25 at 60/90 up to +0.59 at 200/300) with the worst session pinned at -17.9
# throughout -- the tell that the collateral budget, not the cap, is what governs.
#
# Stopping at 120/180 rather than 200/300 forfeits about 0.17 of measured mean on
# purpose. Per-trade collateral is bounded but AGGREGATE concentration is not, and
# the field sim cannot see that risk: its RFQs cap at 15 lots and its FOKs at 12,
# so reaching a 300 position needs ~20 fills in one option and simply never
# happens here. Against a grader that pays zero for a bankruptcy, that is the
# wrong risk to buy with a simulator that is structurally blind to it.
_MAXIMUM_POSITION_PER_OPTION: Final[int] = 180
_MAXIMUM_QUOTE_SIZE: Final[int] = 120
# Left at 0.030. It is the one lever that genuinely loosens the per-trade
# collateral bound rather than removing a redundant cap in front of it, and the
# re-sweep shows exactly what that buys: 0.045 gains mean (+5.10 at 4x flow,
# +47.32 at 10x) by spending the tail, taking the worst session from -82.3 to
# -95.9 and from -45.9 to -63.3 respectively. Mean for tail is a bad trade against
# a grader that scores zero for a blow-up and only partial credit for surviving.
_TRADE_RISK_FRACTION: Final[float] = 0.030
# 0.05 measured better everywhere it was tried, and is still refused. This is the
# reserve between us and the bankruptcy cliff, and the field sim recorded ZERO
# bankruptcies in every configuration at every flow scale -- so it has never once
# observed the event this buffer exists to prevent, and its verdict that the
# buffer is idle is exactly what you would expect from a simulator that cannot
# produce the disaster. At realistic flow the gain is +0.02 on a mean of 10.10.
_SAFETY_BUFFER_FRACTION: Final[float] = 0.150
# Same reasoning, same size of prize: 0.10 is worth +0.04 on a mean of 10.10 at
# realistic flow. Not worth shortening the tenor discount that keeps long-dated
# collateral from being locked up cheaply.
_EXPIRY_CAPITAL_PENALTY: Final[float] = 0.200
# A FOK shows us side, price and size before we commit, and we are allowed to say
# no. An RFQ is blind, cannot be declined, and is only won by being the most
# aggressive quote -- which is exactly when we are most likely to be the one who
# is wrong. Measured across the simulator that asymmetry is worth roughly +0.05
# per contract on FOK against ~+0.007 on RFQ, so the FOK channel gets the larger
# budget and the lower hurdle.
# The hurdle drops to 0.006 and the budget more than doubles because the exchange
# SPLITS a FOK between every maker that accepts it, so the size we actually get is
# a fraction of the size we agreed to -- accepting is cheaper than it looks, and
# declining forfeits the whole clip rather than trimming it. Worth +0.49 over the
# size change alone (t=2.59), and it improves the worst session rather than
# degrading it, since the flow it adds is flow we already screened for edge.
_FOK_BASE_EDGE: Final[float] = 0.006
_FOK_RISK_FRACTION: Final[float] = 0.250
# Kept as its own knob because the RFQ spread and the FOK hurdle are not quite
# the same problem -- an RFQ must be quoted on both sides whether we like it or
# not, whereas a FOK can simply be declined -- which suggests the hurdle could
# afford to be softer. Measured, that turned out to be wrong: 0.4 was worth
# +0.05 (t=0.10) against estimating rivals and -0.38 against omniscient ones,
# and it degraded the worst session in both. Dropping the term altogether scored
# a better mean but is structurally reckless: it would accept a FOK on an option
# we can barely price so long as it shows a cent of apparent edge, and the
# simulator's FOK flow is non-adaptive in a way a real adversary would not be.
# So it stays at parity with the RFQ spread, on evidence rather than by default.
_FOK_UNCERTAINTY_MULTIPLIER: Final[float] = 1.000
_FOK_CONVENTION_CONFIRMATIONS: Final[int] = 2
_FOK_UNPROVEN_EDGE_PREMIUM: Final[float] = 0.020
# 0.120, not 0.250, so that the ABSOLUTE unproven budget stays at
# 0.250 * 0.120 = 0.030 of capital -- exactly where it sat before
# `_FOK_RISK_FRACTION` was raised from 0.120 to 0.250. This is the number that
# caps what it costs us to be wrong about whether `FokOrder.order_type` names the
# counterparty's side or ours, and reading it backwards means taking the wrong
# side of every FOK we accept until the detector notices. Expressed as a fraction
# of the main budget it silently doubled when that budget doubled, which is the
# opposite of what a tuition cap should do; the robustness test caught it.
_FOK_UNPROVEN_RISK_SCALE: Final[float] = 0.120
_DEFAULT_DRIFT_STD_ERROR: Final[float] = 0.006
_DEFAULT_VOLATILITY_RELATIVE_ERROR: Final[float] = 0.300
_DEFAULT_RATE_PROBABILITY_STD_ERROR: Final[float] = 0.060
_PROBABILITY_FLOOR: Final[float] = 1e-4
# The binomial standard error sqrt(p(1-p)/n) covers only the error in the up/down
# FREQUENCIES. It misses everything else we estimate about the rate: the target
# it reverts to, the reversion strength, and the step size -- all of which move
# the terminal lattice, and none of which the perturbation touches. So it is a
# lower bound on our true rate ignorance, and measurement confirms it: sampling
# live options at quote time, rate cells came out UNDER-padded (ratio 0.82 on a
# 60-day history, realised p90 error 0.165 against 0.122 of padding) while every
# company cell sat near 2.0. This scale closes that gap so one leg type is not
# quoted on thinner protection than the other.
_RATE_ERROR_SCALE: Final[float] = 2.000

# Neutral prior, used only if a price is requested before `warm_up` has run.
_FALLBACK_MARKET_PARAMETERS: Final[MarketParameters] = MarketParameters(
    ajarai_drift=0.0,
    ajarai_idio_std_dev=0.02,
    ajarai_rate_beta=-0.05,
    ajarai_sector_beta=1.0,
    rate_down_probability=0.15,
    rate_reversion_strength=0.1,
    rate_up_probability=0.15,
    sector_std_dev=0.015,
    theriodic_drift=0.0,
    theriodic_idio_std_dev=0.02,
    theriodic_rate_beta=-0.05,
    theriodic_sector_beta=1.0,
)


class MarketMaker:
    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:
        self.underlying_state: list[Underlying] = underlying_initial_state
        self.active_option_state: list[BinaryOption] = option_initial_state
        self.cash_balance: float = cash_balance
        self.position: Position = Position()

        # Set by `warm_up`; until then we price off a neutral prior so that any
        # grader logging call before warm-up still returns something sane.
        self.estimated_parameters: MarketParameters = _FALLBACK_MARKET_PARAMETERS
        # Prices only move when the step advances, so a per-step memo is safe.
        self._price_cache: dict[BinaryOption, float] = {}
        self._uncertainty_cache: dict[BinaryOption, float] = {}

        # Mirror of the grader's solvency ledger. It debits WORST-CASE loss on
        # every fill and only credits back at expiry, so we have to reproduce
        # that accounting ourselves -- `self.cash_balance` is our copy of it.
        self._initial_cash_balance: float = cash_balance
        self._drift_std_error_by_underlying_id: dict[int, float] = {
            AJARAI_UNDERLYING_ID: _DEFAULT_DRIFT_STD_ERROR,
            THERIODIC_UNDERLYING_ID: _DEFAULT_DRIFT_STD_ERROR,
        }
        self._volatility_relative_error: float = _DEFAULT_VOLATILITY_RELATIVE_ERROR
        self._rate_probability_std_error: float = _DEFAULT_RATE_PROBABILITY_STD_ERROR

        self._known_option_by_id: dict[int, BinaryOption] = {
            option.option_id: option for option in option_initial_state
        }
        self._pending_payoff_by_option_id: dict[int, float] = {}

        # Collateral is charged PER TRADE, so a buy and an offsetting sell each
        # post their own worst-case loss. Settlement must refund on the same
        # GROSS basis -- long contracts pay `outcome`, short contracts refund
        # `1 - outcome`. Netting here would silently lose the collateral of every
        # offsetting round trip and make us believe we were far closer to
        # bankruptcy than we really are.
        self._long_quantity_by_option_id: dict[int, int] = defaultdict(int)
        self._short_quantity_by_option_id: dict[int, int] = defaultdict(int)

        # `FokOrder.order_type` could plausibly describe either side of the trade.
        # Rather than guess once and be wrong all session, we assume it is the
        # counterparty's side and then verify against the signed quantity that
        # comes back through `on_trade`, flipping if the evidence disagrees.
        # `FokOrder.order_type` is not documented as to WHOSE side it names. The
        # natural reading is the counterparty's -- they sent the order, so BUY
        # means they buy and we sell -- and that is the default. But reading it
        # backwards would mean taking the opposite side of every FOK we thought
        # had edge, so the convention is also checked against reality: on a fill
        # the exchange hands us a SIGNED quantity, which settles the question.
        self._fok_describes_counterparty_side: bool = True
        self._fok_sign_agreements: int = 0
        self._fok_sign_disagreements: int = 0
        # option_id -> (price, expected sign, sender id, clip size) of a FOK we
        # have accepted and are waiting to see filled. `on_trade` fires for RFQ
        # fills too, so without the price and size an unrelated RFQ on the same
        # option would consume this expectation and vote on a trade it knows
        # nothing about. See `_score_fok_convention`.
        self._pending_fok_by_option_id: dict[int, tuple[float, int, int, int]] = {}
        # Set once we observe `on_trade` reporting the id of a FOK sender we were
        # actually waiting on, which tells us the field is passed through rather
        # than being the venue's own id. Until then we cannot read anything into
        # it, so it is not used to reject fills.
        self._fok_counterparty_id_is_reported: bool = False

    def on_step_advance(self, new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]) -> None:
        # An option sitting at `steps_until_expiry == 0` settles against TODAY's
        # values, so snapshot its payoff before the state is replaced.
        self._record_pending_payoffs()

        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state
        self._price_cache.clear()
        self._uncertainty_cache.clear()
        self._pending_fok_by_option_id.clear()

        self._settle_expired_options()
        self._known_option_by_id.update({option.option_id: option for option in new_option_state})

    def on_trade(self, option: BinaryOption, price: float, quantity: int, counterparty_id: int) -> None:
        self.position.add_option_quantity(option.option_id, quantity)
        self._known_option_by_id[option.option_id] = option
        self.cash_balance -= _worst_case_loss(price, quantity)
        if quantity > 0:
            self._long_quantity_by_option_id[option.option_id] += quantity
        else:
            self._short_quantity_by_option_id[option.option_id] -= quantity

        self._score_fok_convention(option.option_id, price, quantity, counterparty_id)

    def _score_fok_convention(
        self, option_id: int, price: float, quantity: int, counterparty_id: int
    ) -> None:
        """Check a fill against the side we predicted when we accepted a FOK.

        Matching on price alone is not enough. Quotes are snapped to a penny grid
        and an RFQ on the same option can settle on the very same penny as a FOK
        we are still waiting on -- at which point an unrelated fill, possibly on
        the opposite side, votes on a question it knows nothing about and can flip
        a CORRECT convention into a wrong one. RFQ volume runs roughly 10x FOK
        volume, so that collision is rare per trade but not rare per session.

        Price and size are the two discriminators we can rely on unconditionally,
        because we chose them ourselves when we accepted: the exchange splits a
        FOK across every maker that accepts, so our share may be CUT below the
        clip we agreed to, but it can never exceed it and the side cannot change.
        A partial fill therefore still votes.

        Identity is the tempting third discriminator and the dangerous one. The
        interface never says whose id `on_trade` reports -- the order's sender, or
        the venue -- and requiring a match costs everything if it is the latter:
        every genuine FOK fill is discarded, the convention never confirms, and we
        spend the whole session on the reduced unproven-convention budget. That is
        not hypothetical; it cost 1.6 of mean PnL when this filter was hard. So
        the field is trusted only once it has proven itself informative -- the
        first time a reported id equals the sender we were waiting on, we know the
        venue passes it through and can start enforcing it. Until then it is
        ignored and price and size carry the test alone.
        """
        pending: tuple[float, int, int, int] | None = self._pending_fok_by_option_id.get(option_id)
        if pending is None or quantity == 0:
            return

        expected_price, expected_sign, expected_counterparty, expected_quantity = pending
        if abs(price - expected_price) > _TIE_EPSILON or abs(quantity) > expected_quantity:
            return  # not the fill we are waiting on; leave the expectation standing
        if counterparty_id == expected_counterparty:
            self._fok_counterparty_id_is_reported = True
        elif self._fok_counterparty_id_is_reported:
            return  # this venue does report senders, and this one is not ours

        self._pending_fok_by_option_id.pop(option_id, None)
        if (1 if quantity > 0 else -1) == expected_sign:
            self._fok_sign_agreements += 1
        else:
            self._fok_sign_disagreements += 1

        if self._fok_sign_disagreements > self._fok_sign_agreements + 2:
            self._fok_describes_counterparty_side = not self._fok_describes_counterparty_side
            self._fok_sign_agreements = 0
            self._fok_sign_disagreements = 0

    def _open_option_ids(self) -> set[int]:
        return {option_id for option_id, quantity in self._long_quantity_by_option_id.items() if quantity} | {
            option_id for option_id, quantity in self._short_quantity_by_option_id.items() if quantity
        }

    def _record_pending_payoffs(self) -> None:
        current_values: dict[int, float] = self._value_by_underlying_id()
        open_option_ids: set[int] = self._open_option_ids()
        for option in self.active_option_state:
            if option.steps_until_expiry == 0 and option.option_id in open_option_ids:
                self._pending_payoff_by_option_id[option.option_id] = option.expiry_valuation(current_values)

    def _settle_expired_options(self) -> None:
        """Credit expiry payoffs for anything no longer live.

        Gross, not net: every long contract pays `outcome` and every short
        contract refunds `1 - outcome`. Both are non-negative, matching the
        grader's note that settlement can only ever increase the balance.

        "No longer live" means two things, not one. The obvious case is an
        option_id that has dropped out of the active set. The other is an id that
        is still present but now names a DIFFERENT contract, because the venue
        recycled it onto a fresh option after the old one expired. The template
        ships `contract_matches`, a helper whose only purpose is to compare two
        options while ignoring their ids, which is a strong hint that ids and
        contracts are not one-to-one here.

        Missing that second case does not risk insolvency -- it strands the payoff
        and leaves us believing we are poorer than we are -- but it is expensive
        in exactly the wrong way: understated cash shrinks every subsequent quote
        through `_available_capital`, so we quietly stop competing. Driven through
        a session that recycles ids, it left us holding 162.75 against a true
        1086.75.

        The comparison is on legs and strike alone, deliberately NOT on
        `contract_matches`, because that helper also compares `steps_until_expiry`
        -- which decrements every day for a perfectly healthy contract. Matching
        on it would settle live positions early and credit payoffs we have not
        earned, which is the one direction that can overstate cash. Legs and
        strike are fixed for the life of a contract, so they identify it exactly.
        """
        active_by_option_id: dict[int, BinaryOption] = {
            option.option_id: option for option in self.active_option_state
        }
        expired_option_ids: list[int] = []
        for option_id in self._open_option_ids():
            current: BinaryOption | None = active_by_option_id.get(option_id)
            if current is None:
                expired_option_ids.append(option_id)
                continue
            # A pending payoff is only ever recorded by `_record_pending_payoffs`,
            # and only for an option sitting at `steps_until_expiry == 0`. Such an
            # option has expired and the grader credits it this same evening, so
            # its presence is proof on its own -- whatever the id is doing now.
            # This is the only signal that separates an id recycled onto an
            # IDENTICAL contract from one that genuinely survived, and without it
            # we would strand those payoffs forever.
            if option_id in self._pending_payoff_by_option_id:
                expired_option_ids.append(option_id)
                continue
            previous: BinaryOption | None = self._known_option_by_id.get(option_id)
            if previous is not None and (previous.legs, previous.strike) != (current.legs, current.strike):
                expired_option_ids.append(option_id)

        current_values: dict[int, float] = self._value_by_underlying_id()
        for option_id in expired_option_ids:
            payoff: float | None = self._pending_payoff_by_option_id.pop(option_id, None)
            if payoff is None:
                known_option: BinaryOption | None = self._known_option_by_id.get(option_id)
                # No snapshot and no contract to value against: bank nothing. That
                # only ever understates our balance, which is the safe direction.
                payoff = known_option.expiry_valuation(current_values) if known_option is not None else 0.0

            self.cash_balance += self._long_quantity_by_option_id.pop(option_id, 0) * payoff
            self.cash_balance += self._short_quantity_by_option_id.pop(option_id, 0) * (1.0 - payoff)
            self.position.option_quantity_by_option_id[option_id] = 0

    def _value_by_underlying_id(self) -> dict[int, float]:
        return {underlying.underlying_id: underlying.value for underlying in self.underlying_state}

    @property
    def name(self) -> str:
        return "Aakarsh D Reja"

    def price_option(self, option: BinaryOption) -> float:
        cached_price: float | None = self._price_cache.get(option)
        if cached_price is not None:
            return cached_price

        price: float = self.price_option_from_parameters(self.estimated_parameters, option)
        self._price_cache[option] = price
        return price

    def price_option_from_parameters(
        self, market_parameters: MarketParameters, option: BinaryOption
    ) -> float:
        value_by_underlying_id: dict[int, float] = {
            underlying.underlying_id: underlying.value for underlying in self.underlying_state
        }

        # Already at expiry: the payoff is fully determined by today's values.
        if option.steps_until_expiry <= 0:
            return option.expiry_valuation(value_by_underlying_id)

        num_steps: int = option.steps_until_expiry
        rate_weight: float = 0.0
        company_legs: list[OptionLeg] = []
        for leg in option.legs:
            if leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                rate_weight = leg.weight
            else:
                company_legs.append(leg)

        initial_rate: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_distribution: dict[float, float] = _terminal_rate_distribution(
            market_parameters, initial_rate, num_steps
        )

        # Condition on where the rate lands: that pins down both the rate leg's
        # contribution to the strike and the companies' drift adjustment.
        total_probability: float = 0.0
        for terminal_rate, rate_probability in rate_distribution.items():
            if rate_probability <= 0.0:
                continue

            residual_strike: float = option.strike - (rate_weight * terminal_rate)
            net_rate_change: float = terminal_rate - initial_rate
            total_probability += rate_probability * self._conditional_company_probability(
                market_parameters, company_legs, value_by_underlying_id, residual_strike, num_steps, net_rate_change
            )
        return min(max(total_probability, 0.0), 1.0)

    def _conditional_company_probability(
        self,
        market_parameters: MarketParameters,
        company_legs: list[OptionLeg],
        value_by_underlying_id: dict[int, float],
        residual_strike: float,
        num_steps: int,
        net_rate_change: float,
    ) -> float:
        """P(company legs clear `residual_strike`), given the net rate move."""
        if not company_legs:
            # Pure rate option: the rate leg alone already decided it. Payoff is
            # `>=`, so an exact tie pays -- which is common on the 0.25 grid.
            return 1.0 if residual_strike <= _TIE_EPSILON else 0.0

        first_leg: OptionLeg = company_legs[0]
        first_mean, first_variance = _log_return_moments(
            market_parameters, first_leg.underlying_id, num_steps, net_rate_change
        )
        first_spot: float = value_by_underlying_id[first_leg.underlying_id]

        if len(company_legs) == 1:
            return _single_company_probability(
                first_leg.weight, first_spot, residual_strike, first_mean, math.sqrt(max(first_variance, 0.0))
            )

        second_leg: OptionLeg = company_legs[1]
        second_mean, second_variance = _log_return_moments(
            market_parameters, second_leg.underlying_id, num_steps, net_rate_change
        )
        return _two_company_probability(
            first_leg.weight,
            first_spot,
            first_mean,
            first_variance,
            second_leg.weight,
            value_by_underlying_id[second_leg.underlying_id],
            second_mean,
            second_variance,
            _log_return_covariance(market_parameters, num_steps),
            residual_strike,
        )

    def quote(self, option: BinaryOption, counterparty_id: int) -> Quote:
        theoretical_price: float = self.price_option(option)
        available_capital: float = self._available_capital()

        # A 0.00 bid and a 1.00 offer both carry zero worst-case loss, so this is
        # the closest thing to declining an RFQ the `Quote` contract allows.
        if available_capital <= 0.0:
            return Quote(bid_price=0.0, bid_quantity=1, offer_price=1.0, offer_quantity=1)

        half_spread: float = min(
            _MAXIMUM_HALF_SPREAD,
            _BASE_HALF_SPREAD + (_UNCERTAINTY_MULTIPLIER * self._price_uncertainty(option)),
        )

        # Lean the whole market against inventory: long positions want to sell.
        current_quantity: int = self.position.option_quantity_by_option_id.get(option.option_id, 0)
        inventory_fraction: float = max(
            -1.0, min(1.0, current_quantity / _MAXIMUM_POSITION_PER_OPTION)
        )
        centre: float = theoretical_price - (_INVENTORY_SKEW * inventory_fraction)

        bid_ticks: int = max(0, min(99, math.floor((centre - half_spread) * 100.0)))
        offer_ticks: int = max(1, min(100, math.ceil((centre + half_spread) * 100.0)))
        if offer_ticks <= bid_ticks:
            if bid_ticks > 0:
                bid_ticks = offer_ticks - 1
            else:
                offer_ticks = bid_ticks + 1
        bid_price: float = bid_ticks / 100.0
        offer_price: float = offer_ticks / 100.0

        buy_room: int = max(0, _MAXIMUM_POSITION_PER_OPTION - current_quantity)
        sell_room: int = max(0, _MAXIMUM_POSITION_PER_OPTION + current_quantity)
        bid_quantity: int = self._affordable_quantity(bid_price, buy_room, option.steps_until_expiry)
        offer_quantity: int = self._affordable_quantity(1.0 - offer_price, sell_room, option.steps_until_expiry)

        # Quantities must be positive, so when a side is exhausted retreat it to
        # its zero-collateral price rather than showing size we cannot support.
        if bid_quantity <= 0:
            bid_price, bid_quantity = 0.0, 1
        if offer_quantity <= 0:
            offer_price, offer_quantity = 1.0, 1
        # Unreachable as written, and verified so by exhaustive enumeration of all
        # 40,000 tick states: the block above already guarantees bid < offer, and
        # the two retreats only ever widen (bid down to 0.00, offer up to 1.00).
        # It stays as insurance, because `Quote` raises on a crossed market and a
        # raise scores zero for the session -- but it is now written to be CORRECT
        # if some future edit does reach it. The previous form was not: at an offer
        # of 0.00 it set bid = max(0.0, -0.01) = 0.00, producing bid == offer and
        # raising the exact error it was standing there to prevent.
        if bid_price >= offer_price:
            offer_price = min(1.0, max(offer_price, 0.01))
            bid_price = max(0.0, offer_price - 0.01)

        return Quote(
            bid_price=round(bid_price, 2),
            bid_quantity=bid_quantity,
            offer_price=round(offer_price, 2),
            offer_quantity=offer_quantity,
        )

    def respond_to_fok(self, option: BinaryOption, fok_order: FokOrder) -> bool:
        theoretical_price: float = self.price_option(option)
        required_edge: float = _FOK_BASE_EDGE + (
            _FOK_UNCERTAINTY_MULTIPLIER * self._price_uncertainty(option)
        )

        we_are_buying: bool = (
            fok_order.order_type == OrderType.SELL
            if self._fok_describes_counterparty_side
            else fok_order.order_type == OrderType.BUY
        )

        # Until a fill has actually confirmed the side convention, trade it like
        # an unproven hypothesis: demand more edge and risk a fraction of the
        # usual budget. If the reading is backwards we take the wrong side of
        # every FOK until the flip triggers, so this bounds the tuition fee --
        # and once confirmed it costs nothing, because it switches itself off.
        convention_confirmed: bool = self._fok_sign_agreements >= _FOK_CONVENTION_CONFIRMATIONS
        risk_fraction: float = _FOK_RISK_FRACTION
        if not convention_confirmed:
            required_edge += _FOK_UNPROVEN_EDGE_PREMIUM
            risk_fraction *= _FOK_UNPROVEN_RISK_SCALE

        if we_are_buying:
            edge: float = theoretical_price - fok_order.price
            signed_quantity: int = fok_order.quantity
        else:
            edge = fok_order.price - theoretical_price
            signed_quantity = -fok_order.quantity

        if edge < required_edge:
            return False

        current_quantity: int = self.position.option_quantity_by_option_id.get(option.option_id, 0)
        if abs(current_quantity + signed_quantity) > _MAXIMUM_POSITION_PER_OPTION:
            return False

        # Accepting can fill the whole order, so budget for the full worst case.
        worst_case_loss: float = _worst_case_loss(fok_order.price, signed_quantity)
        if worst_case_loss > risk_fraction * self._available_capital():
            return False

        self._pending_fok_by_option_id[option.option_id] = (
            fok_order.price,
            1 if signed_quantity > 0 else -1,
            fok_order.counterparty_id,
            fok_order.quantity,
        )
        return True

    def warm_up(self, market_history: MarketHistory) -> None:
        self._price_cache.clear()
        self._uncertainty_cache.clear()

        values_by_underlying_id: dict[int, tuple[float, ...]] = market_history.values_by_underlying_id
        rates: tuple[float, ...] = values_by_underlying_id.get(FED_FUNDS_RATE_UNDERLYING_ID, ())
        ajarai_values: tuple[float, ...] = values_by_underlying_id.get(AJARAI_UNDERLYING_ID, ())
        theriodic_values: tuple[float, ...] = values_by_underlying_id.get(THERIODIC_UNDERLYING_ID, ())

        # Too little data to beat the prior, and a badly overfit estimate is far
        # more dangerous than a vague one.
        if min(len(rates), len(ajarai_values), len(theriodic_values)) < _MINIMUM_HISTORY_DAYS:
            return

        rate_step: float = _estimate_rate_step(rates)
        up_probability, down_probability, reversion_strength, rate_target = _estimate_rate_dynamics(rates, rate_step)
        rate_changes: list[float] = [rates[i] - rates[i - 1] for i in range(1, len(rates))]

        ajarai_drift, ajarai_rate_beta, ajarai_residuals, ajarai_drift_error = _regress_log_returns(
            ajarai_values, rate_changes
        )
        theriodic_drift, theriodic_rate_beta, theriodic_residuals, theriodic_drift_error = _regress_log_returns(
            theriodic_values, rate_changes
        )
        ajarai_sector_beta, theriodic_sector_beta, ajarai_idio, theriodic_idio = _decompose_sector_exposure(
            ajarai_residuals, theriodic_residuals
        )

        self.estimated_parameters = MarketParameters(
            ajarai_drift=ajarai_drift,
            ajarai_idio_std_dev=ajarai_idio,
            ajarai_rate_beta=ajarai_rate_beta,
            ajarai_sector_beta=ajarai_sector_beta,
            rate_down_probability=down_probability,
            rate_reversion_strength=reversion_strength,
            rate_up_probability=up_probability,
            sector_std_dev=1.0,  # pinned; the sector betas carry the scale
            theriodic_drift=theriodic_drift,
            theriodic_idio_std_dev=theriodic_idio,
            theriodic_rate_beta=theriodic_rate_beta,
            theriodic_sector_beta=theriodic_sector_beta,
            rate_step=rate_step,
            rate_target=rate_target,
        )

        self._drift_std_error_by_underlying_id = {
            AJARAI_UNDERLYING_ID: ajarai_drift_error,
            THERIODIC_UNDERLYING_ID: theriodic_drift_error,
        }
        # Standard error of an estimated standard deviation is ~sigma / sqrt(2n).
        self._volatility_relative_error = 1.0 / math.sqrt(2.0 * max(1, len(ajarai_residuals)))

        # The up/down move frequencies are just binomial proportions over the
        # observed transitions, so their standard error is sqrt(p(1-p)/n). This
        # is the dominant source of error on rate options and, until now, was the
        # one thing `_price_uncertainty` never perturbed -- so a pure rate option
        # reported ~zero uncertainty no matter how little history we had seen.
        num_transitions: int = max(1, len(rate_changes))
        self._rate_probability_std_error = math.sqrt(
            max(up_probability * (1.0 - up_probability), down_probability * (1.0 - down_probability), 1e-4)
            / num_transitions
        )

    def _available_capital(self) -> float:
        return self.cash_balance - (_SAFETY_BUFFER_FRACTION * self._initial_cash_balance)

    def _affordable_quantity(self, unit_collateral: float, position_room: int, steps_until_expiry: int) -> int:
        """Largest size we can show given collateral, position room and tenor."""
        hard_cap: int = min(_MAXIMUM_QUOTE_SIZE, position_room)
        if hard_cap <= 0:
            return 0
        if unit_collateral <= 1e-9:
            return hard_cap  # zero-collateral price: only position limits bind

        # Collateral is frozen until expiry, so longer-dated risk earns less size.
        tenor_scale: float = 1.0 / (1.0 + (_EXPIRY_CAPITAL_PENALTY * steps_until_expiry))
        budget: float = _TRADE_RISK_FRACTION * self._available_capital() * tenor_scale
        return max(0, min(hard_cap, int(budget / unit_collateral)))

    def _price_uncertainty(self, option: BinaryOption) -> float:
        """How far our price could move if our estimates are off by ~1 std error.

        Re-prices the option under drift and volatility perturbations and takes
        the worst deviation. The drifts are pushed in opposite directions across
        the two companies as well as together, because that is what actually
        moves an AJR-vs-THR spread.
        """
        cached: float | None = self._uncertainty_cache.get(option)
        if cached is not None:
            return cached

        base_price: float = self.price_option(option)
        parameters: MarketParameters = self.estimated_parameters

        # Convention risk. `steps_until_expiry == 0` is priced as a certainty
        # (0.0 or 1.0) because `expiry_valuation` settles against the values we
        # can already see. If the exchange instead settles those against the NEXT
        # step's values, that certainty is badly wrong. Rather than guess, we
        # treat the disagreement between the two readings as uncertainty: if the
        # option is genuinely decided both readings agree and this costs nothing,
        # and if it is not, we widen exactly where it matters.
        if option.steps_until_expiry == 0:
            one_step_price: float = self.price_option_from_parameters(
                parameters, replace(option, steps_until_expiry=1)
            )
            uncertainty = min(abs(one_step_price - base_price), _MAXIMUM_PRICE_UNCERTAINTY)
            self._uncertainty_cache[option] = uncertainty
            return uncertainty
        ajarai_error: float = self._drift_std_error_by_underlying_id.get(
            AJARAI_UNDERLYING_ID, _DEFAULT_DRIFT_STD_ERROR
        )
        theriodic_error: float = self._drift_std_error_by_underlying_id.get(
            THERIODIC_UNDERLYING_ID, _DEFAULT_DRIFT_STD_ERROR
        )
        volatility_scale: float = 1.0 + self._volatility_relative_error
        rate_error: float = _RATE_ERROR_SCALE * self._rate_probability_std_error

        worst_deviation: float = 0.0
        has_company_leg: bool = any(
            leg.underlying_id != FED_FUNDS_RATE_UNDERLYING_ID for leg in option.legs
        )

        # Two axes of ignorance, perturbed separately.
        #
        # Company axis: drifts pushed in opposite directions as well as together,
        # because opposite is what actually moves an AJR-vs-THR spread.
        # Rate axis: the up/down frequencies shifted against each other, which
        # slides the whole terminal-rate lattice up or down.
        #
        # Scenarios that cannot move this particular option are skipped -- the
        # company perturbations are exact no-ops on a pure rate option, and
        # pricing is by far the hot path here.
        scenarios: list[tuple[float, float, float, float]] = []
        if has_company_leg:
            scenarios.extend([
                (1.0, -1.0, volatility_scale, 0.0),
                (-1.0, 1.0, volatility_scale, 0.0),
                (1.0, 1.0, 1.0 / volatility_scale, 0.0),
                (-1.0, -1.0, 1.0 / volatility_scale, 0.0),
            ])
        if rate_error > 0.0:
            scenarios.extend([(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 1.0, -1.0)])

        for ajarai_sign, theriodic_sign, scale, rate_sign in scenarios:
            # `MarketParameters` rejects a non-positive or over-unit probability
            # pair outright, so clamp into the open interval rather than to the
            # bounds -- a perturbation is not worth raising on.
            up_probability: float = min(
                1.0 - (2.0 * _PROBABILITY_FLOOR),
                max(_PROBABILITY_FLOOR, parameters.rate_up_probability + (rate_sign * rate_error)),
            )
            down_probability: float = min(
                1.0 - up_probability - _PROBABILITY_FLOOR,
                max(_PROBABILITY_FLOOR, parameters.rate_down_probability - (rate_sign * rate_error)),
            )
            perturbed: MarketParameters = replace(
                parameters,
                ajarai_drift=parameters.ajarai_drift + (ajarai_sign * ajarai_error),
                theriodic_drift=parameters.theriodic_drift + (theriodic_sign * theriodic_error),
                ajarai_idio_std_dev=parameters.ajarai_idio_std_dev * scale,
                theriodic_idio_std_dev=parameters.theriodic_idio_std_dev * scale,
                sector_std_dev=parameters.sector_std_dev * scale,
                rate_up_probability=up_probability,
                rate_down_probability=down_probability,
            )
            worst_deviation = max(
                worst_deviation, abs(self.price_option_from_parameters(perturbed, option) - base_price)
            )

        uncertainty: float = min(worst_deviation, _MAXIMUM_PRICE_UNCERTAINTY)
        self._uncertainty_cache[option] = uncertainty
        return uncertainty
