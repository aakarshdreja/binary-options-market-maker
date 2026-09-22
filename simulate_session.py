"""Local session simulator: does our market maker survive, and does it win?

Reproduces the grader's solvency ledger exactly as described in the brief --
every fill debits WORST-CASE loss, payoffs are credited only at expiry, and the
balance is checked at the end of each day. Bankruptcy scores zero, so the primary
output here is the bankruptcy rate; PnL rank is secondary.

Competitors are given the TRUE parameters, which is deliberately pessimistic:
they never mis-price, so any edge we show comes from spread discipline and risk
management rather than from opponents being dumb.
"""

import argparse
import math
import random
import statistics
from dataclasses import dataclass, field

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
    MarketParameters,
    OptionLeg,
    OrderType,
    Quote,
    Underlying,
)
from validate_estimation import SCENARIOS

INITIAL_CASH = 1_000.0
TARGET_ACTIVE_OPTIONS = 14


def worst_case_loss(price: float, quantity: int) -> float:
    return quantity * price if quantity > 0 else -quantity * (1.0 - price)


@dataclass
class Ledger:
    name: str
    cash: float = INITIAL_CASH
    bankrupt: bool = False
    # Collateral is charged PER TRADE, so a buy and an offsetting sell each post
    # their own worst-case loss. Settlement must therefore refund on a GROSS
    # basis -- long contracts pay `outcome`, short contracts refund `1 - outcome`
    # -- otherwise an offsetting round trip posts collateral twice and refunds
    # once, and cash silently disappears.
    long_by_option_id: dict[int, int] = field(default_factory=dict)
    short_by_option_id: dict[int, int] = field(default_factory=dict)
    num_trades: int = 0
    contracts_traded: int = 0

    def apply_fill(self, option_id: int, price: float, quantity: int) -> None:
        self.cash -= worst_case_loss(price, quantity)
        if quantity > 0:
            self.long_by_option_id[option_id] = self.long_by_option_id.get(option_id, 0) + quantity
        else:
            self.short_by_option_id[option_id] = self.short_by_option_id.get(option_id, 0) - quantity
        self.num_trades += 1
        self.contracts_traded += abs(quantity)

    def net_quantity(self, option_id: int) -> int:
        return self.long_by_option_id.get(option_id, 0) - self.short_by_option_id.get(option_id, 0)

    def settle(self, option_id: int, payoff: float) -> None:
        self.cash += self.long_by_option_id.pop(option_id, 0) * payoff
        self.cash += self.short_by_option_id.pop(option_id, 0) * (1.0 - payoff)

    def economic_value(self, true_price_by_option_id: dict[int, float]) -> float:
        total = self.cash
        for option_id, quantity in self.long_by_option_id.items():
            total += quantity * true_price_by_option_id.get(option_id, 0.0)
        for option_id, quantity in self.short_by_option_id.items():
            total += quantity * (1.0 - true_price_by_option_id.get(option_id, 0.0))
        return total


class ReferenceMarketMaker:
    """Competitor that prices perfectly and quotes a fixed spread."""

    def __init__(self, name: str, half_spread: float, size: int, noise: float, pricer: MarketMaker) -> None:
        self.name = name
        self.half_spread = half_spread
        self.size = size
        self.noise = noise
        self._pricer = pricer
        self._true_parameters: MarketParameters | None = None
        self._rng = random.Random(hash(name) & 0xFFFF)

    def bind(self, true_parameters: MarketParameters) -> None:
        self._true_parameters = true_parameters

    def price(self, option: BinaryOption) -> float:
        assert self._true_parameters is not None
        value = self._pricer.price_option_from_parameters(self._true_parameters, option)
        if self.noise:
            value += self._rng.gauss(0.0, self.noise)
        return min(max(value, 0.0), 1.0)

    def quote(self, option: BinaryOption, ledger: Ledger) -> Quote | None:
        theo = self.price(option)
        bid_ticks = max(0, min(99, math.floor((theo - self.half_spread) * 100)))
        offer_ticks = max(1, min(100, math.ceil((theo + self.half_spread) * 100)))
        if offer_ticks <= bid_ticks:
            offer_ticks = bid_ticks + 1
        bid, offer = bid_ticks / 100.0, offer_ticks / 100.0

        available = max(0.0, ledger.cash - 0.15 * INITIAL_CASH)
        bid_quantity = self.size if bid <= 0 else min(self.size, int(0.05 * available / bid))
        offer_quantity = self.size if offer >= 1.0 else min(self.size, int(0.05 * available / (1.0 - offer)))
        if bid_quantity <= 0:
            bid, bid_quantity = 0.0, 1
        if offer_quantity <= 0:
            offer, offer_quantity = 1.0, 1
        if bid >= offer:
            bid = max(0.0, offer - 0.01)
        return Quote(bid_price=round(bid, 2), bid_quantity=bid_quantity,
                     offer_price=round(offer, 2), offer_quantity=offer_quantity)

    def respond_to_fok(self, option: BinaryOption, order: FokOrder, ledger: Ledger) -> bool:
        theo = self.price(option)
        if order.order_type == OrderType.SELL:
            edge, signed = theo - order.price, order.quantity
        else:
            edge, signed = order.price - theo, -order.quantity
        if edge < self.half_spread:
            return False
        return worst_case_loss(order.price, signed) <= 0.06 * max(0.0, ledger.cash - 0.15 * INITIAL_CASH)


class Session:
    def __init__(self, true_parameters: MarketParameters, seed: int, num_days: int,
                 history_days: int, informed_fraction: float, competitors: list[tuple[str, float, int, float]]) -> None:
        self.parameters = true_parameters
        self.rng = random.Random(seed)
        self.num_days = num_days
        self.informed_fraction = informed_fraction

        random.seed(seed)
        self.values = {FED_FUNDS_RATE_UNDERLYING_ID: 2.25,
                       AJARAI_UNDERLYING_ID: 900.0,
                       THERIODIC_UNDERLYING_ID: 850.0}
        series: dict[int, list[float]] = {k: [v] for k, v in self.values.items()}
        for _ in range(history_days - 1):
            self.values = true_parameters.advance_step(self.values)
            for key, value in self.values.items():
                series[key].append(value)
        self.history = MarketHistory(values_by_underlying_id={k: tuple(v) for k, v in series.items()})

        self.next_option_id = 1
        self.options: list[BinaryOption] = []
        self._list_new_options()

        self.oracle = MarketMaker(self._underlyings(), list(self.options), INITIAL_CASH)
        self.subject = MarketMaker(self._underlyings(), list(self.options), INITIAL_CASH)
        self.subject.warm_up(self.history)

        self.competitors = [ReferenceMarketMaker(n, hs, sz, nz, self.oracle) for n, hs, sz, nz in competitors]
        for competitor in self.competitors:
            competitor.bind(true_parameters)

        self.diagnostics: dict[str, float] = {}
        self.ledgers = {"SUBJECT": Ledger("SUBJECT")}
        for competitor in self.competitors:
            self.ledgers[competitor.name] = Ledger(competitor.name)

    def _underlyings(self) -> list[Underlying]:
        return [
            Underlying(FED_FUNDS_RATE_NAME, FED_FUNDS_RATE_UNDERLYING_ID, self.values[FED_FUNDS_RATE_UNDERLYING_ID]),
            Underlying(AJARAI_NAME, AJARAI_UNDERLYING_ID, self.values[AJARAI_UNDERLYING_ID]),
            Underlying(THERIODIC_NAME, THERIODIC_UNDERLYING_ID, self.values[THERIODIC_UNDERLYING_ID]),
        ]

    def _list_new_options(self) -> None:
        rate_leg = OptionLeg(underlying_id=FED_FUNDS_RATE_UNDERLYING_ID, weight=1.0)
        ajarai_leg = OptionLeg(underlying_id=AJARAI_UNDERLYING_ID, weight=1.0)
        theriodic_leg = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=1.0)
        short_theriodic_leg = OptionLeg(underlying_id=THERIODIC_UNDERLYING_ID, weight=-1.0)

        while len(self.options) < TARGET_ACTIVE_OPTIONS:
            steps = self.rng.randint(1, 7)
            kind = self.rng.random()
            if kind < 0.30:
                base = self.values[FED_FUNDS_RATE_UNDERLYING_ID]
                strike = max(0.0, round(base + self.rng.choice([-0.5, -0.25, 0.0, 0.25, 0.5]), 2))
                legs = (rate_leg,)
            elif kind < 0.85:
                use_ajarai = self.rng.random() < 0.5
                underlying_id = AJARAI_UNDERLYING_ID if use_ajarai else THERIODIC_UNDERLYING_ID
                base = self.values[underlying_id]
                strike = round(base * self.rng.choice([0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10]), 2)
                legs = (ajarai_leg if use_ajarai else theriodic_leg,)
            else:
                legs = (ajarai_leg, short_theriodic_leg)
                strike = 0.0
            self.options.append(BinaryOption(legs=legs, option_id=self.next_option_id,
                                             steps_until_expiry=steps, strike=strike))
            self.next_option_id += 1

    def _true_price(self, option: BinaryOption) -> float:
        return self.oracle.price_option_from_parameters(self.parameters, option)

    def _collect_quotes(self, option: BinaryOption) -> list[tuple[str, Quote]]:
        quotes: list[tuple[str, Quote]] = []
        subject_ledger = self.ledgers["SUBJECT"]
        if not subject_ledger.bankrupt:
            quotes.append(("SUBJECT", self.subject.quote(option, counterparty_id=1)))
        for competitor in self.competitors:
            ledger = self.ledgers[competitor.name]
            if ledger.bankrupt:
                continue
            quote = competitor.quote(option, ledger)
            if quote is not None:
                quotes.append((competitor.name, quote))
        return quotes

    def _record_fill(self, name: str, option: BinaryOption, price: float, quantity: int,
                     channel: str = "rfq") -> None:
        self.ledgers[name].apply_fill(option.option_id, price, quantity)
        if name == "SUBJECT":
            self.subject.on_trade(option, price, quantity, counterparty_id=1)
            true_price = self._true_price(option)
            edge = quantity * (true_price - price) if quantity > 0 else -quantity * (price - true_price)
            self.diagnostics[f"{channel}_edge"] = self.diagnostics.get(f"{channel}_edge", 0.0) + edge
            self.diagnostics[f"{channel}_contracts"] = (
                self.diagnostics.get(f"{channel}_contracts", 0.0) + abs(quantity)
            )
            self.diagnostics[f"{channel}_fills"] = self.diagnostics.get(f"{channel}_fills", 0.0) + 1

    def _run_rfq(self) -> None:
        option = self.rng.choice(self.options)
        quantity = self.rng.randint(1, 15)
        quotes = self._collect_quotes(option)
        if not quotes:
            return

        true_price = self._true_price(option)
        informed = self.rng.random() < self.informed_fraction
        best_bid = max(q.bid_price for _, q in quotes)
        best_offer = min(q.offer_price for _, q in quotes)

        if informed:
            # Only lifts/hits when it is genuinely profitable -- this is the flow
            # that punishes a mispriced or too-tight market.
            buy_edge, sell_edge = true_price - best_offer, best_bid - true_price
            if max(buy_edge, sell_edge) <= 0.0:
                return
            counterparty_buys = buy_edge >= sell_edge
        else:
            counterparty_buys = self.rng.random() < 0.5

        if counterparty_buys:
            book = sorted(quotes, key=lambda item: item[1].offer_price)
            for name, quote in book:
                if quantity <= 0:
                    break
                filled = min(quantity, quote.offer_quantity)
                self._record_fill(name, option, quote.offer_price, -filled)
                quantity -= filled
        else:
            book = sorted(quotes, key=lambda item: -item[1].bid_price)
            for name, quote in book:
                if quantity <= 0:
                    break
                filled = min(quantity, quote.bid_quantity)
                self._record_fill(name, option, quote.bid_price, filled)
                quantity -= filled

    def _run_fok(self) -> None:
        option = self.rng.choice(self.options)
        quantity = self.rng.randint(1, 12)
        true_price = self._true_price(option)
        informed = self.rng.random() < self.informed_fraction
        counterparty_buys = self.rng.random() < 0.5

        # Informed flow prices the order in its own favour; uninformed flow pays away.
        offset = abs(self.rng.gauss(0.0, 0.05)) + 0.01
        if counterparty_buys:
            price = true_price - offset if informed else true_price + offset
        else:
            price = true_price + offset if informed else true_price - offset
        price = round(min(max(price, 0.0), 1.0), 2)

        order = FokOrder(counterparty_id=2, option_id=option.option_id,
                         order_type=OrderType.BUY if counterparty_buys else OrderType.SELL,
                         price=price, quantity=quantity)

        acceptors: list[str] = []
        subject_ledger = self.ledgers["SUBJECT"]
        if not subject_ledger.bankrupt and self.subject.respond_to_fok(option, order):
            acceptors.append("SUBJECT")
        for competitor in self.competitors:
            ledger = self.ledgers[competitor.name]
            if not ledger.bankrupt and competitor.respond_to_fok(option, order, ledger):
                acceptors.append(competitor.name)
        if not acceptors:
            return

        share = quantity // len(acceptors)
        remainder = quantity % len(acceptors)
        for index, name in enumerate(acceptors):
            allocation = share + (1 if index < remainder else 0)
            if allocation <= 0:
                continue
            self._record_fill(name, option, price, -allocation if counterparty_buys else allocation, "fok")

    def run(self) -> dict[str, object]:
        for _ in range(self.num_days):
            for _ in range(self.rng.randint(8, 16)):
                self._run_rfq()
            for _ in range(self.rng.randint(3, 8)):
                self._run_fok()

            # End of day: settle expiries, then check solvency.
            expiring = [o for o in self.options if o.steps_until_expiry == 0]
            for option in expiring:
                payoff = option.expiry_valuation(self.values)
                for ledger in self.ledgers.values():
                    ledger.settle(option.option_id, payoff)
            self.options = [o for o in self.options if o.steps_until_expiry > 0]

            for ledger in self.ledgers.values():
                if not ledger.bankrupt and ledger.cash < 0.0:
                    ledger.bankrupt = True
            if self.ledgers["SUBJECT"].bankrupt:
                break

            self.values = self.parameters.advance_step(self.values)
            self.options = [o.advance_step() for o in self.options]
            self._list_new_options()
            self.oracle.on_step_advance(self._underlyings(), list(self.options))
            self.subject.on_step_advance(self._underlyings(), list(self.options))

        true_price_by_option_id = {o.option_id: self._true_price(o) for o in self.options}
        values = {name: ledger.economic_value(true_price_by_option_id) - INITIAL_CASH
                  for name, ledger in self.ledgers.items()}
        subject = self.ledgers["SUBJECT"]
        ranked = sorted(values.items(), key=lambda item: -item[1])
        return {
            "bankrupt": subject.bankrupt,
            "pnl": values["SUBJECT"],
            "rank": [n for n, _ in ranked].index("SUBJECT") + 1,
            "num_participants": len(values),
            "trades": subject.num_trades,
            "contracts": subject.contracts_traded,
            "min_cash": subject.cash,
            "all": values,
            "diagnostics": self.diagnostics,
        }


COMPETITOR_SETS: dict[str, list[tuple[str, float, int, float]]] = {
    "easy":   [("wide", 0.08, 10, 0.03), ("noisy", 0.05, 10, 0.05)],
    "medium": [("wide", 0.05, 15, 0.02), ("noisy", 0.04, 15, 0.03), ("tight", 0.03, 10, 0.0)],
    "hard":   [("tight", 0.02, 20, 0.0), ("tighter", 0.015, 20, 0.0), ("wide", 0.04, 20, 0.01)],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=12)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    print(f"{'scenario':<22}{'field':<8}{'informed':>9}{'bankrupt':>10}{'mean pnl':>10}"
          f"{'med pnl':>9}{'worst':>9}{'rank':>7}{'trades':>8}")
    print("-" * 92)

    overall_bankrupt = 0
    overall_sessions = 0
    for scenario_name, parameters in SCENARIOS.items():
        for field_name, competitors in COMPETITOR_SETS.items():
            for informed_fraction in (0.2, 0.5):
                results = []
                for index in range(args.sessions):
                    session = Session(parameters, seed=1000 + index * 37, num_days=args.days,
                                      history_days=[60, 150, 400][index % 3],
                                      informed_fraction=informed_fraction, competitors=competitors)
                    results.append(session.run())
                bankruptcies = sum(1 for r in results if r["bankrupt"])
                pnls = [float(r["pnl"]) for r in results]
                overall_bankrupt += bankruptcies
                overall_sessions += len(results)
                print(f"{scenario_name[:21]:<22}{field_name:<8}{informed_fraction:>9.1f}"
                      f"{bankruptcies:>4}/{len(results):<5}{statistics.fmean(pnls):>10.1f}"
                      f"{statistics.median(pnls):>9.1f}{min(pnls):>9.1f}"
                      f"{statistics.fmean([float(r['rank']) for r in results]):>7.2f}"
                      f"{statistics.fmean([float(r['trades']) for r in results]):>8.0f}")
    print("-" * 92)
    print(f"total bankruptcies: {overall_bankrupt}/{overall_sessions}")


if __name__ == "__main__":
    main()
