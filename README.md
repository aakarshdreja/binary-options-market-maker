# Binary Options Market Maker

An automated market maker for the Akuna Capital Virtual Quant Trading Challenge 2026, a competition where you write a trading bot that prices and trades binary options on uncertain, daily evolving quantities, competing against other bots for profit.

This repository holds the solution itself plus the full research process behind it: parameter estimation, an analytic pricer, a simulator built to test the strategy before submitting it, and the diagnostics, tuning and stress tests that shaped the final constants.

No prior finance background is assumed below. Every term is defined before it is used.

## Table of contents

1. [The problem, explained from scratch](#the-problem-explained-from-scratch)
2. [The hidden model I had to work out](#the-hidden-model-i-had-to-work-out)
3. [My approach](#my-approach)
4. [How I validated it before submitting](#how-i-validated-it-before-submitting)
5. [Repository layout](#repository-layout)
6. [Running it yourself](#running-it-yourself)

## The problem, explained from scratch

My own condensed restatement of the official brief is at [`docs/problem_statement.md`](docs/problem_statement.md); this section expands on it for a reader with no prior context.

### What a binary option is

A binary option (also called an event contract) is a bet on whether some future statement is true or false. It pays out exactly **1.0** if the statement turns out true, and **0.0** otherwise. For example: "will the Fed funds rate be at least 3.0% in 3 days." Because the payout is capped at 1.0, the fair price of the contract is simply the probability that the statement is true. A contract that is 70% likely to happen is worth about $0.70 today.

In this challenge, the statements are about three underlying quantities that evolve day by day:

- **FED**: the Fed funds interest rate.
- **AJR**: the valuation of a fictional private AI lab, AjarAI.
- **THR**: the valuation of a second fictional private AI lab, Theriodic.

Most contracts reference a single underlying (e.g. "AJR ≥ $1tn in 5 days"), but a few are spreads across two of them (e.g. "AJR ≥ THR in 2 days").

### How trades arrive

There are two distinct order flows, and they require different logic:

- **RFQ (request for quote)**: the exchange asks every market maker to name a two sided market, a price at which you'd buy and a price at which you'd sell, and it routes the trade to whoever offered the best price, splitting the order across makers if needed. You do not know in advance whether the customer wants to buy or sell, so both sides of your quote have to be genuinely tradeable.
- **FOK (fill or kill)**: you are shown the full order up front, price, size and side, and you simply decide yes or no. If several market makers say yes, the order is split between them.

### How you're scored

The grader runs a sequence of tests:

- **THEO**: checks that your pricing formula, given the *true* underlying model parameters, produces the correct probability.
- **VERBOSE** (a handful of short tests): full credit as long as your code does not crash and you don't go bankrupt. These exist to help you debug.
- **SCORED** (the bulk of the tests): you are not given the true parameters here, only a warm-up history of past values for each underlying. You must estimate the model from that history, then trade a full session. Your score depends on your profit and loss (PnL) relative to the other market makers in that session: full credit for finishing on top, partial credit for surviving without going bankrupt, and zero credit for going bankrupt or erroring.

### What "going bankrupt" actually means

The grader tracks a separate cash ledger from anything you track yourself. Every time you trade, your cash balance is immediately debited by the **worst case loss** of that trade (if you buy 5 contracts at $0.20, you lose at most $1.00 if they expire worthless, so $1.00 is locked away immediately; if you sell 5 at $0.20, your worst case is paying out $1.00 per contract, so $4.00 is locked away). That cash only comes back when a contract actually expires and pays off. At the end of each day, if your cash balance is negative, you are bankrupt and the session ends early for you. This means risk has to be managed against your *worst case*, not your *expected*, exposure.

## The hidden model I had to work out

None of the three underlyings' dynamics are told to you directly, only their [`MarketParameters`](Market_Maker.py) shape. From the problem statement and the template code, the generative model is:

- **FED** moves on a fixed grid (steps of a fixed size, e.g. 0.25) each day: it goes up with some probability, down with some probability, or stays, and those probabilities are tilted to pull the rate back toward a long run target the further away it drifts (mean reversion). It is also floored at zero.
- **AJR** and **THR** each move as a multiplicative (percentage) random walk each day. Each day's percentage move is the sum of four things: a constant drift, a sensitivity to that day's rate move (rate beta), a shared shock common to both companies (a "sector" factor, capturing something like industry wide sentiment that moves both AI labs together), and its own idiosyncratic noise.

None of the drift, betas, volatilities or rate transition probabilities are given. You only receive a short history of past daily values and have to estimate all of it before you can price anything.

## My approach

The full implementation is in [`Market_Maker.py`](Market_Maker.py) (documented; used for development) with a comment-stripped, submission-sized twin in [`Market_Maker_compact.py`](Market_Maker_compact.py) (some grading platforms rejected the full file on size). Both implement the exact same logic.

### 1. Estimating the market from history (`warm_up`)

- **Rate step size**: the smallest nonzero move observed in the rate history.
- **Rate dynamics** (up/down probabilities, mean reversion strength, target level): fit by penalized maximum likelihood. A full four dimensional grid search over these would be too slow, so I use coordinate descent, optimizing one dimension at a time over a grid that shrinks each pass, with a mild prior toward "no reversion, target near the initial rate" so a short history doesn't overfit noise into a strong reversion story.
- **Company drift and rate sensitivity**: a linear regression of each company's daily log return against that day's rate change. The resulting estimate is shrunk toward zero in proportion to how uncertain it is, so a handful of noisy days doesn't get read as a confident trend.
- **Shared sector exposure**: after regressing out the rate effect, whatever correlation is left between AJR's and THR's daily noise is attributed to the shared sector factor; the rest is each company's own idiosyncratic volatility.

### 2. Pricing a contract exactly, given a parameter set (`price_option_from_parameters`)

Because the rate lives on a small discrete grid, its full probability distribution by expiry can be computed exactly by propagating the up/down/stay transition probabilities forward, one day at a time, rather than simulated. For each possible rate outcome, AJR and THR are each conditionally lognormal (their log values are normally distributed), so:

- A single-company contract's probability of clearing its strike has a closed form using the standard normal CDF.
- A two-company spread contract (like "AJR ≥ THR") reduces to a bivariate normal calculation using the two companies' means, variances and their shared sector covariance.

Averaging the conditional probability over every possible rate outcome gives the exact probability under the model, which doubles as the theoretical price. This is what `THEO` tests directly, and it is also what the estimated model from step 1 is fed into for every other test.

### 3. Turning a price into a two-sided quote (`quote`)

- Start from the theoretical price.
- Widen the spread around it based on how uncertain the underlying parameter estimates are: I re-price the same option after nudging each estimated parameter by its statistical standard error, one at a time, and use the largest resulting price change as an uncertainty measure. A short warm-up history means wide spreads; a longer one narrows them automatically.
- Skew the quote away from the theoretical price in whichever direction reduces my current position in that contract, so the book is nudged back toward flat rather than letting a position run.
- Size the quote from how much risk capital remains, scaled down for longer dated contracts, since more can happen to the price before they expire.

### 4. Deciding whether to take a FOK order (`respond_to_fok`)

I only accept a FOK order if my price versus their price gives me more edge than a minimum threshold. That threshold starts wide, because the problem statement never states which side of the order `"buy"` or `"sell"` describes, the requester's side or mine. I treat this as unknown at first, track whether my fills come back consistent with one interpretation or the other, and only narrow the threshold once I'm statistically confident which convention is in effect. Every accepted trade is still capped by remaining risk budget and by a maximum position size per contract.

### 5. Risk management

- Every fill is booked against cash using the exact worst case loss rule described above, so my own solvency tracking matches the grader's ledger.
- A cash safety buffer I never trade below.
- A maximum position per contract.
- Capital sizing that shrinks automatically for larger and longer dated trades.

## How I validated it before submitting

Since the real grader isn't available to test against ahead of time, I built my own local simulator that reproduces its rules exactly (worst case cash debits, expiry driven credits, end of day solvency checks) and used it to drive the whole research process, which lives in [`research/`](research):

- **Correctness checks**: [`validate_pricing.py`](research/validate_pricing.py) compares the closed form pricer against brute force Monte Carlo simulation of the real dynamics. [`validate_estimation.py`](research/validate_estimation.py) checks that `warm_up` recovers enough of the true model to price accurately from a range of history lengths. [`test_fok_convention.py`](research/test_fok_convention.py) verifies the buy/sell side detector described above actually works, in both possible conventions.
- **Diagnostics**: [`diagnose_pnl.py`](research/diagnose_pnl.py), [`diagnose_history.py`](research/diagnose_history.py), [`diagnose_uncertainty.py`](research/diagnose_uncertainty.py) and [`diagnose_calibration.py`](research/diagnose_calibration.py) trace exactly where profit and loss come from and whether the quoted spread is honestly calibrated to realised pricing error. This is how I caught, for example, that rate-only contracts were getting an uncertainty estimate of nearly zero, because the calculation only perturbed company parameters and never the rate parameters, until it was fixed.
- **Tuning**: [`tune_parameters.py`](research/tune_parameters.py), [`tune_round2.py`](research/tune_round2.py), [`tune_round3.py`](research/tune_round3.py), [`tune_focused.py`](research/tune_focused.py), [`sweep_uncertainty.py`](research/sweep_uncertainty.py), [`sweep_spread.py`](research/sweep_spread.py) and [`optimize_winrate.py`](research/optimize_winrate.py) search the quoting and risk constants against the simulator, optimizing for the metric the grader actually pays (relative rank and bankruptcy avoidance, not raw mean PnL).
- **Adversarial and stress testing**: [`skew_adversarial.py`](research/skew_adversarial.py), [`stress_uncertainty.py`](research/stress_uncertainty.py), [`large_flow_check.py`](research/large_flow_check.py) and [`ab_counterparty.py`](research/ab_counterparty.py) check whether a tuned setting still holds up against a counterparty that specifically targets it, or against much larger order flow than the tuning sessions used.
- **Statistical discipline**: comparisons are paired (same random seed for every candidate config, so differences reflect the config and not luck) and judged by significance (t-statistics), not by eyeballing an average. This is also what caught a real bug: [`reverify_constants.py`](research/reverify_constants.py) found that several earlier "winning" tuning decisions had been measured against *different, randomly reseeded* competitor pools across runs, making the comparisons invalid, and re-ran them properly.
- **Final gate**: [`final_audit.py`](research/final_audit.py) replays the grader's exact solvency rule against deliberately hostile inputs (a one point history, a rate stuck at zero, extreme values) to make sure nothing crashes or goes bankrupt before submission. [`decide_final.py`](research/decide_final.py) and [`final_standings.py`](research/final_standings.py) run the fully tuned strategy through many paired sessions against a simulated field of five rival archetypes (`sharp`, `montecarlo`, `norate`, `nocorr`, `crude`, described in [`simulate_field.py`](research/simulate_field.py)) for a last check before submitting.

These are figures from my own local simulator, not an official grade: across 90 paired sessions against the simulated field, the final configuration averaged a PnL rank of about 2.3 out of 6 players (1st is best) with zero bankruptcies in either a realistic field (rivals estimate the model, same as I do) or a pessimistic one (rivals are handed the true parameters).

## Repository layout

```
.
├── Market_Maker.py            the solution, fully documented
├── Market_Maker_compact.py    same logic, comment-stripped for size-limited submission
├── make_compact.py            generates the file above from the one above it
├── docs/
│   └── problem_statement.md   my own condensed restatement of the official brief
└── research/                  the simulator, validation, tuning and audit scripts
```

Every file in `research/` imports `Market_Maker` straight from the repository root (each one carries a short `sys.path` snippet at the top for that), so they can all still be run directly, individually, exactly as they were during development. Grouped by purpose:

| Files in `research/` | What they are |
|---|---|
| `validate_*.py`, `test_*.py` | Correctness checks against the true simulator dynamics. |
| `diagnose_*.py`, `experiment_estimation.py` | Investigations into where PnL comes from and whether estimates are well calibrated. |
| `tune_*.py`, `sweep_*.py`, `optimize_winrate.py` | Searches over the quoting and risk constants. |
| `simulate_*.py` | The local simulator itself, and the competitor field it plays against. |
| `skew_adversarial.py`, `stress_uncertainty.py`, `large_flow_check.py`, `ab_counterparty.py`, `reverify_constants.py` | Adversarial and robustness checks on the tuned settings. |
| `decide_final.py`, `decide_uncertainty.py`, `final_sweep.py`, `final_standings.py`, `final_audit.py`, `audit_review.py` | The final decision process and pre-submission audit. |
| `*.txt` files | Saved console output from the corresponding script, kept as a record of what was actually measured. |

## Running it yourself

Requirements: Python 3.12 or newer, standard library only, no external dependencies. Run scripts from the repository root:

```bash
python research/validate_pricing.py
python research/validate_estimation.py
python research/final_audit.py
```

Each script is self-contained and prints its own results to the console; most also carry a docstring at the top explaining what question they were written to answer.

---

This was built for the Akuna Capital Virtual Quant Trading Challenge 2026. FED, AJR and THR are fictional instruments defined by the competition; the market model, estimation, pricing and trading logic in this repository are my own work.
