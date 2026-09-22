# Binary Options Market Maker

A market making trading bot for binary options (event contracts) on three correlated, uncertain underlyings. It estimates the hidden dynamics of the market from a short price history, prices any contract in closed form, and quotes and trades against other automated market makers while managing risk against worst case loss.

The project is inspired by Akuna Capital's Virtual Quant Trading Challenge, which posed the underlying problem: three daily evolving underlyings whose joint dynamics are never told to you, event contracts on them and on spreads between them, two different order flows (quoted markets and take it or leave it orders), and a scoring rule that punishes bankruptcy far more than it rewards being merely profitable. Everything in this repository, the estimation method, the analytic pricer, the quoting and risk logic, and the backtesting used to tune and stress test it, is my own design and implementation. A copy of the original brief is kept at [`docs/challenge_context.md`](docs/challenge_context.md) for reference.

## What it does

- **Learns the market from a short history.** No model parameters are given; the bot fits a mean reverting rate process and a two factor (correlated) valuation model for the two companies from a warm up price history, using maximum likelihood and shrinkage regression rather than guesswork.
- **Prices contracts exactly, not by simulation.** Because the rate lives on a discrete grid, its full outcome distribution can be computed exactly, and each contract's fair value follows from closed form normal and bivariate normal probabilities rather than a slow Monte Carlo loop.
- **Quotes two sided markets that know what they don't know.** The bid/offer spread widens automatically when the underlying parameter estimates are less certain, and narrows as more history becomes available.
- **Manages inventory.** Quotes skew away from the theoretical price to pull an open position back toward flat rather than letting it run.
- **Figures out an undocumented trading convention on its own.** One order type never states whose side of the trade its label refers to; the bot treats this as unknown, watches how its own fills come back, and only trades confidently once it has statistically confirmed which convention is in effect.
- **Manages risk the way it's actually measured.** Every trade is booked against a worst case collateral rule, with a cash safety buffer, per contract position limits, and size that shrinks automatically for longer dated, riskier trades.

## Results

Since there was no live grader to test against, I built a local backtesting simulator that reproduces the exact same solvency and settlement rules, and used it to tune and stress test the bot before relying on it. These are backtest figures, not an official score: across 90 paired sessions against a simulated field of five rival market makers of varying sophistication, the final configuration averaged a PnL rank of about 2.3 out of 6 (1st is best) with zero bankruptcies, in both a realistic field (rivals estimate the market same as the bot does) and a pessimistic one (rivals are handed the true parameters and never err).

## How it works

- **Estimation** (`warm_up`, in [`market_maker.py`](market_maker.py)): the rate's step size, up/down/stay probabilities, mean reversion strength and target are fit by penalized maximum likelihood over a shrinking coordinate grid. Each company's drift and rate sensitivity come from a regression of its daily returns on the rate's daily change, shrunk toward zero by how uncertain the fit is. What's left over after that regression is split into a shared factor common to both companies and each one's own idiosyncratic noise.
- **Pricing** (`price_option_from_parameters`): the rate's exact terminal distribution is computed by propagating its transition probabilities forward day by day. Conditional on each rate outcome, a single leg contract's fair value is a standard normal CDF; a two leg spread contract (e.g. one company's valuation against the other's) is a bivariate normal calculation. Averaging over the rate distribution gives the exact model price.
- **Quoting** (`quote`): spread width comes from re-pricing the same contract after nudging each estimated parameter by its statistical error and taking the largest resulting price move; the quote is then skewed by current inventory and sized by remaining risk budget, scaled down for longer dated contracts.
- **Take it or leave it orders** (`respond_to_fok`): only accepted when the edge over a minimum threshold is large enough, where that threshold itself shrinks as the bot statistically confirms which side of the order convention it's dealing with.
- **Risk** (`_available_capital` and the position caps around it): every fill is booked at worst case collateral, with a cash buffer that never gets traded through and a hard position cap per contract.

## Repository layout

```
.
├── market_maker.py            the bot, fully documented
├── market_maker_compact.py    same logic, comment-stripped for a size-limited submission target
├── build_compact.py           generates the file above from the one above it
├── docs/
│   └── challenge_context.md   the original challenge brief, kept for reference
└── research/                  the backtesting simulator, and the validation, tuning and stress tests built on it
```

Every file in `research/` imports `market_maker` straight from the repository root (each one carries a short `sys.path` snippet at the top for that), so they can all still be run directly, individually, exactly as they were during development. Grouped by purpose:

| Files in `research/` | What they are |
|---|---|
| `validate_*.py`, `test_*.py` | Correctness checks: the analytic pricer against Monte Carlo, the estimator against known ground truth, edge case robustness. |
| `diagnose_*.py`, `experiment_estimation.py` | Where profit and loss actually comes from, and whether the quoted spread is honestly calibrated to real pricing error. |
| `tune_*.py`, `sweep_*.py`, `optimize_winrate.py` | Searches over the quoting and risk constants, optimized for the metric that's actually scored (rank and bankruptcy avoidance, not raw mean PnL). |
| `simulate_*.py`, `sweep_fok_uncertainty.py` | The backtesting simulator itself, and the field of rival archetypes it plays against. |
| `skew_adversarial.py`, `stress_uncertainty.py`, `large_flow_check.py`, `ab_counterparty.py`, `reverify_constants.py` | Whether a tuned setting survives a rival that specifically targets it, or flow much larger than it was tuned on. |
| `decide_final.py`, `decide_uncertainty.py`, `final_sweep.py`, `final_standings.py`, `final_audit.py`, `audit_review.py` | The final decision process and a pre-launch audit against edge case inputs. |
| `*.txt` files | Saved console output from the corresponding script, kept as a record of what was actually measured. |

## Running it yourself

Requirements: Python 3.12 or newer, standard library only, no external dependencies. Run scripts from the repository root:

```bash
python research/validate_pricing.py
python research/validate_estimation.py
python research/final_audit.py
```

Each script is self-contained and prints its own results to the console; most also carry a docstring at the top explaining what question they were written to answer.
