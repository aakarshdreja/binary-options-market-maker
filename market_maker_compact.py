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

_SQRT_2: Final[float] = math.sqrt(2.0)
_TIE_EPSILON: Final[float] = 1e-9
_QUADRATURE_HALF_WIDTH: Final[float] = 8.0
_QUADRATURE_INTERVALS: Final[int] = 128

def _worst_case_loss(price: float, quantity: int) -> float:
    return quantity * price if quantity > 0 else -quantity * (1.0 - price)

def _standard_normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))

def _standard_normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _terminal_rate_distribution(
    market_parameters: MarketParameters, initial_rate: float, num_steps: int
) -> dict[float, float]:
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
    drift, rate_beta, sector_beta, idio_std_dev = _company_parameters(market_parameters, underlying_id)
    mean: float = (num_steps * drift) + (rate_beta * net_rate_change)
    sector_variance: float = (sector_beta * market_parameters.sector_std_dev) ** 2
    variance: float = num_steps * (sector_variance + (idio_std_dev**2))
    return mean, variance

def _log_return_covariance(market_parameters: MarketParameters, num_steps: int) -> float:
    return (
        num_steps
        * market_parameters.ajarai_sector_beta
        * market_parameters.theriodic_sector_beta
        * (market_parameters.sector_std_dev**2)
    )

def _single_company_probability(
    weight: float, spot: float, strike: float, mean: float, std_dev: float
) -> float:
    if weight > 0.0:
        threshold_value: float = strike / weight
        if threshold_value <= 0.0:
            return 1.0
    else:
        threshold_value = strike / weight
        if threshold_value <= 0.0:
            return 0.0

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
        return 1.0 - _standard_normal_cdf(standardised)
    return _standard_normal_cdf(standardised)

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
    first_std_dev: float = math.sqrt(max(first_variance, 0.0))
    second_std_dev: float = math.sqrt(max(second_variance, 0.0))

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

_MINIMUM_HISTORY_DAYS: Final[int] = 12
_MAXIMUM_ABSOLUTE_CORRELATION: Final[float] = 0.995
_COORDINATE_DESCENT_SWEEPS: Final[int] = 6
_COORDINATE_DESCENT_GRID: Final[int] = 17

_REVERSION_STRENGTH_PRIOR_STD_DEV: Final[float] = 0.20
_RATE_TARGET_PRIOR_MEAN: Final[float] = 2.00
_RATE_TARGET_PRIOR_STD_DEV: Final[float] = 1.00

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def _estimate_rate_step(rates: tuple[float, ...]) -> float:
    moves: list[float] = [abs(rates[i] - rates[i - 1]) for i in range(1, len(rates))]
    positive_moves: list[float] = [move for move in moves if move > 1e-9]
    if not positive_moves:
        return RATE_STRIKE_GRID
    return round(min(positive_moves), 2)

def _rate_log_likelihood(
    rates: tuple[float, ...], rate_step: float, up_prob: float, down_prob: float, strength: float, target: float
) -> float:
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

_BASE_HALF_SPREAD: Final[float] = 0.012
_UNCERTAINTY_MULTIPLIER: Final[float] = 1.400
_MAXIMUM_HALF_SPREAD: Final[float] = 0.250
_MAXIMUM_PRICE_UNCERTAINTY: Final[float] = 0.150
_INVENTORY_SKEW: Final[float] = 0.035
_MAXIMUM_POSITION_PER_OPTION: Final[int] = 180
_MAXIMUM_QUOTE_SIZE: Final[int] = 120
_TRADE_RISK_FRACTION: Final[float] = 0.030
_SAFETY_BUFFER_FRACTION: Final[float] = 0.150
_EXPIRY_CAPITAL_PENALTY: Final[float] = 0.200
_FOK_BASE_EDGE: Final[float] = 0.006
_FOK_RISK_FRACTION: Final[float] = 0.250
_FOK_UNCERTAINTY_MULTIPLIER: Final[float] = 1.000
_FOK_CONVENTION_CONFIRMATIONS: Final[int] = 2
_FOK_UNPROVEN_EDGE_PREMIUM: Final[float] = 0.020
_FOK_UNPROVEN_RISK_SCALE: Final[float] = 0.120
_DEFAULT_DRIFT_STD_ERROR: Final[float] = 0.006
_DEFAULT_VOLATILITY_RELATIVE_ERROR: Final[float] = 0.300
_DEFAULT_RATE_PROBABILITY_STD_ERROR: Final[float] = 0.060
_PROBABILITY_FLOOR: Final[float] = 1e-4
_RATE_ERROR_SCALE: Final[float] = 2.000

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

        self.estimated_parameters: MarketParameters = _FALLBACK_MARKET_PARAMETERS
        self._price_cache: dict[BinaryOption, float] = {}
        self._uncertainty_cache: dict[BinaryOption, float] = {}

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

        self._long_quantity_by_option_id: dict[int, int] = defaultdict(int)
        self._short_quantity_by_option_id: dict[int, int] = defaultdict(int)

        self._fok_describes_counterparty_side: bool = True
        self._fok_sign_agreements: int = 0
        self._fok_sign_disagreements: int = 0
        self._pending_fok_by_option_id: dict[int, tuple[float, int, int, int]] = {}
        self._fok_counterparty_id_is_reported: bool = False

    def on_step_advance(self, new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]) -> None:
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
        pending: tuple[float, int, int, int] | None = self._pending_fok_by_option_id.get(option_id)
        if pending is None or quantity == 0:
            return

        expected_price, expected_sign, expected_counterparty, expected_quantity = pending
        if abs(price - expected_price) > _TIE_EPSILON or abs(quantity) > expected_quantity:
            return
        if counterparty_id == expected_counterparty:
            self._fok_counterparty_id_is_reported = True
        elif self._fok_counterparty_id_is_reported:
            return

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
        active_by_option_id: dict[int, BinaryOption] = {
            option.option_id: option for option in self.active_option_state
        }
        expired_option_ids: list[int] = []
        for option_id in self._open_option_ids():
            current: BinaryOption | None = active_by_option_id.get(option_id)
            if current is None:
                expired_option_ids.append(option_id)
                continue
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
        if not company_legs:
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

        if available_capital <= 0.0:
            return Quote(bid_price=0.0, bid_quantity=1, offer_price=1.0, offer_quantity=1)

        half_spread: float = min(
            _MAXIMUM_HALF_SPREAD,
            _BASE_HALF_SPREAD + (_UNCERTAINTY_MULTIPLIER * self._price_uncertainty(option)),
        )

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

        if bid_quantity <= 0:
            bid_price, bid_quantity = 0.0, 1
        if offer_quantity <= 0:
            offer_price, offer_quantity = 1.0, 1
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
            sector_std_dev=1.0,
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
        self._volatility_relative_error = 1.0 / math.sqrt(2.0 * max(1, len(ajarai_residuals)))

        num_transitions: int = max(1, len(rate_changes))
        self._rate_probability_std_error = math.sqrt(
            max(up_probability * (1.0 - up_probability), down_probability * (1.0 - down_probability), 1e-4)
            / num_transitions
        )

    def _available_capital(self) -> float:
        return self.cash_balance - (_SAFETY_BUFFER_FRACTION * self._initial_cash_balance)

    def _affordable_quantity(self, unit_collateral: float, position_room: int, steps_until_expiry: int) -> int:
        hard_cap: int = min(_MAXIMUM_QUOTE_SIZE, position_room)
        if hard_cap <= 0:
            return 0
        if unit_collateral <= 1e-9:
            return hard_cap

        tenor_scale: float = 1.0 / (1.0 + (_EXPIRY_CAPITAL_PENALTY * steps_until_expiry))
        budget: float = _TRADE_RISK_FRACTION * self._available_capital() * tenor_scale
        return max(0, min(hard_cap, int(budget / unit_collateral)))

    def _price_uncertainty(self, option: BinaryOption) -> float:
        cached: float | None = self._uncertainty_cache.get(option)
        if cached is not None:
            return cached

        base_price: float = self.price_option(option)
        parameters: MarketParameters = self.estimated_parameters

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
