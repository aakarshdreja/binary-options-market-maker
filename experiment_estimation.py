"""Throwaway harness: compare estimator variants by realised pricing error.

The MLE overfits `rate_reversion_strength` on short histories, and the drift
shrinkage may be over-tuned. This measures variants head to head so the choice
is made on evidence rather than taste.
"""

import math
import statistics

from Market_Maker import (
    AJARAI_NAME,
    AJARAI_UNDERLYING_ID,
    FED_FUNDS_RATE_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID,
    THERIODIC_NAME,
    THERIODIC_UNDERLYING_ID,
    MarketMaker,
    MarketParameters,
    Underlying,
    _decompose_sector_exposure,
    _estimate_rate_step,
    _mean,
    _rate_log_likelihood,
)
from validate_estimation import SCENARIOS, build_options, generate_history

SWEEPS = 6
GRID = 17


def estimate_rate_dynamics(rates, rate_step, strength_prior_sigma, target_prior_sigma, fixed_target):
    current = [0.20, 0.20, 0.10, 2.00]
    windows = [0.25, 0.25, 0.25, 1.50]
    bounds = [(0.005, 0.9), (0.005, 0.9), (0.0, 1.0), (0.0, 6.0)]
    num_axes = 3 if fixed_target else 4

    def score(candidate):
        if candidate[0] + candidate[1] > 1.0:
            return -math.inf
        value = _rate_log_likelihood(rates, rate_step, candidate[0], candidate[1], candidate[2], candidate[3])
        if strength_prior_sigma:
            value -= 0.5 * (candidate[2] / strength_prior_sigma) ** 2
        if target_prior_sigma:
            value -= 0.5 * ((candidate[3] - 2.0) / target_prior_sigma) ** 2
        return value

    best = score(current)
    for _ in range(SWEEPS):
        for axis in range(num_axes):
            low = max(bounds[axis][0], current[axis] - windows[axis])
            high = min(bounds[axis][1], current[axis] + windows[axis])
            if high <= low:
                continue
            step = (high - low) / (GRID - 1)
            for index in range(GRID):
                candidate = list(current)
                candidate[axis] = low + index * step
                value = score(candidate)
                if value > best:
                    best, current = value, candidate
        windows = [w * 0.5 for w in windows]

    up = min(max(current[0], 0.005), 0.9)
    down = min(max(current[1], 0.005), 0.9)
    if up + down > 1.0:
        scale = 1.0 / (up + down)
        up, down = up * scale, down * scale
    return up, down, min(max(current[2], 0.0), 1.0), max(current[3], 0.0)


def regress(values, rate_changes, shrink_drift, shrink_beta):
    log_returns = [math.log(values[i] / values[i - 1]) for i in range(1, len(values))]
    n = len(log_returns)
    mean_r, mean_x = _mean(log_returns), _mean(rate_changes)
    var_x = sum((x - mean_x) ** 2 for x in rate_changes)
    cov = sum((rate_changes[i] - mean_x) * (log_returns[i] - mean_r) for i in range(n))
    beta = cov / var_x if var_x > 1e-12 else 0.0
    drift = mean_r - beta * mean_x
    residuals = [log_returns[i] - drift - beta * rate_changes[i] for i in range(n)]
    dof = max(1, n - 2)
    resid_var = sum(r * r for r in residuals) / dof
    drift_var = resid_var * (1.0 / n + (mean_x**2 / var_x if var_x > 1e-12 else 0.0))
    if shrink_drift and (drift**2 + drift_var) > 0:
        drift *= drift**2 / (drift**2 + drift_var)
    if shrink_beta and var_x > 1e-12:
        beta_var = resid_var / var_x
        if (beta**2 + beta_var) > 0:
            beta *= beta**2 / (beta**2 + beta_var)
    residuals = [log_returns[i] - drift - beta * rate_changes[i] for i in range(n)]
    return drift, beta, residuals


def build_parameters(history, variant):
    values = history.values_by_underlying_id
    rates = values[FED_FUNDS_RATE_UNDERLYING_ID]
    step = _estimate_rate_step(rates)
    up, down, strength, target = estimate_rate_dynamics(
        rates, step, variant["strength_prior"], variant["target_prior"], variant["fixed_target"]
    )
    if variant["fixed_target"]:
        target = 2.0
    changes = [rates[i] - rates[i - 1] for i in range(1, len(rates))]
    a_drift, a_beta, a_res = regress(values[AJARAI_UNDERLYING_ID], changes,
                                     variant["shrink_drift"], variant["shrink_beta"])
    t_drift, t_beta, t_res = regress(values[THERIODIC_UNDERLYING_ID], changes,
                                     variant["shrink_drift"], variant["shrink_beta"])
    a_sec, t_sec, a_idio, t_idio = _decompose_sector_exposure(a_res, t_res)
    return MarketParameters(
        ajarai_drift=a_drift, ajarai_idio_std_dev=a_idio, ajarai_rate_beta=a_beta, ajarai_sector_beta=a_sec,
        rate_down_probability=down, rate_reversion_strength=strength, rate_up_probability=up,
        sector_std_dev=1.0,
        theriodic_drift=t_drift, theriodic_idio_std_dev=t_idio, theriodic_rate_beta=t_beta,
        theriodic_sector_beta=t_sec,
        rate_step=step, rate_target=target,
    )


VARIANTS = {
    "A current":            dict(strength_prior=0.0,  target_prior=0.0, fixed_target=False, shrink_drift=True,  shrink_beta=True),
    "B no drift shrink":    dict(strength_prior=0.0,  target_prior=0.0, fixed_target=False, shrink_drift=False, shrink_beta=True),
    "C fixed target":       dict(strength_prior=0.0,  target_prior=0.0, fixed_target=True,  shrink_drift=True,  shrink_beta=True),
    "D priors":             dict(strength_prior=0.20, target_prior=1.0, fixed_target=False, shrink_drift=True,  shrink_beta=True),
    "E priors+fixed tgt":   dict(strength_prior=0.20, target_prior=0.0, fixed_target=True,  shrink_drift=True,  shrink_beta=True),
    "F tight priors":       dict(strength_prior=0.10, target_prior=0.5, fixed_target=False, shrink_drift=True,  shrink_beta=True),
    "G priors, no dshrink": dict(strength_prior=0.20, target_prior=1.0, fixed_target=False, shrink_drift=False, shrink_beta=True),
}


def main() -> None:
    results = {name: {} for name in VARIANTS}
    for scenario_name, true_parameters in SCENARIOS.items():
        for days in (60, 150, 400):
            per_variant = {name: [] for name in VARIANTS}
            for seed in range(6):
                history, final = generate_history(true_parameters, days, seed=seed * 977 + 13)
                underlyings = [
                    Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, final[FED_FUNDS_RATE_UNDERLYING_ID]),
                    Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, final[AJARAI_UNDERLYING_ID]),
                    Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, final[THERIODIC_UNDERLYING_ID]),
                ]
                options = build_options(final)
                market_maker = MarketMaker(underlyings, options, cash_balance=1_000.0)
                truths = [market_maker.price_option_from_parameters(true_parameters, o) for o in options]
                for name, variant in VARIANTS.items():
                    parameters = build_parameters(history, variant)
                    for option, truth in zip(options, truths):
                        per_variant[name].append(
                            abs(market_maker.price_option_from_parameters(parameters, option) - truth)
                        )
            for name in VARIANTS:
                results[name][(scenario_name, days)] = statistics.fmean(per_variant[name])

    keys = sorted({k for name in VARIANTS for k in results[name]}, key=lambda k: (k[0], k[1]))
    print(f"{'variant':<22}" + "".join(f"{k[0][:9]}/{k[1]:<4}" for k in keys) + "  overall")
    print("-" * (22 + 14 * len(keys) + 9))
    for name in VARIANTS:
        row = "".join(f"{results[name][k]:<14.4f}" for k in keys)
        overall = statistics.fmean([results[name][k] for k in keys])
        print(f"{name:<22}{row}  {overall:.4f}")


if __name__ == "__main__":
    main()
