# -*- coding: utf-8 -*-
# (c) Nano Nano Ltd 2019
# pylint: disable=bad-option-value, unnecessary-dunder-call

import copy
import itertools
import sys
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterator, List, Optional, Tuple, Union

import requests
from datetime import datetime, date, time, timezone, timedelta
from colorama import Fore
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

from .bt_types import (
    TRANSFER_TYPES,
    AssetName,
    AssetSymbol,
    Date,
    DisposalType,
    Note,
    TrType,
    Wallet,
    Year,
)
from .config import config
from .constants import TAX_RULES_UK_COMPANY, WARNING
from .holdings import Holdings
from .price.valueasset import ValueAsset
from .tax_event import (
    TaxEvent,
    TaxEventCapitalGains,
    TaxEventIncome,
    TaxEventMarginTrade,
    TaxEventNoGainNoLoss,
)
from .transactions import Buy, Sell

PRECISION = Decimal("0.00")


class TaxReportRecord(TypedDict):  # pylint: disable=too-few-public-methods
    CapitalGains: "CalculateCapitalGains"
    Income: NotRequired["CalculateIncome"]
    MarginTrading: NotRequired["CalculateMarginTrading"]


class HoldingsReportAsset(TypedDict):  # pylint: disable=too-few-public-methods
    name: AssetName
    quantity: Decimal
    cost: Decimal
    value: Optional[Decimal]
    gain: NotRequired[Decimal]


class HoldingsReportTotal(TypedDict):  # pylint: disable=too-few-public-methods
    cost: Decimal
    value: Decimal
    gain: Decimal


class HoldingsReportRecord(TypedDict):  # pylint: disable=too-few-public-methods
    holdings: Dict[AssetSymbol, HoldingsReportAsset]
    totals: HoldingsReportTotal

class YearlyReportAsset(TypedDict):
    quantity_start_of_year: Decimal
    value_in_fiat_start_of_year: Decimal
    quantity_end_of_year: Decimal
    average_balance: Decimal
    value_in_fiat_at_end_of_year: Decimal
    days_held: int 

class YearlyReportTotal(TypedDict):
    total_value_in_fiat_at_end_of_year: Decimal
    total_value_in_fiat_start_of_year: Decimal
    penalty_due_rw: Decimal

class YearlyReportRecord(TypedDict):
    assets: Dict[AssetSymbol, YearlyReportAsset]
    totals: YearlyReportTotal

class DailyReportRecord(TypedDict):
    btc_balance: Decimal
    eur_value: Decimal
    missing_price: bool
    quantity: Decimal

class CapitalGainsIndividual(TypedDict):  # pylint: disable=too-few-public-methods
    allowance: Decimal
    basic_rate: Decimal
    higher_rate: Decimal
    proceeds_limit: NotRequired[Decimal]


class ChargableGainsCompany(TypedDict):  # pylint: disable=too-few-public-methods
    small_rate: Optional[Decimal]
    main_rate: Decimal


class CapitalGainsReportTotal(TypedDict):  # pylint: disable=too-few-public-methods
    cost: Decimal
    fees: Decimal
    proceeds: Decimal
    gain: Decimal


class CapitalGainsReportSummary(TypedDict):  # pylint: disable=too-few-public-methods
    disposals: Decimal
    total_gain: Decimal
    total_loss: Decimal


class CapitalGainsReportEstimate(TypedDict):  # pylint: disable=too-few-public-methods
    allowance: Decimal
    cgt_basic_rate: Decimal
    cgt_higher_rate: Decimal
    allowance_used: Decimal
    taxable_gain: Decimal
    cgt_basic: Decimal
    cgt_higher: Decimal
    proceeds_limit: Decimal
    proceeds_warning: bool


class ChargableGainsReportEstimate(TypedDict):  # pylint: disable=too-few-public-methods
    ct_small_rates: List[Optional[Decimal]]
    ct_main_rates: List[Decimal]
    taxable_gain: Decimal
    ct_small: NotRequired[Decimal]
    ct_main: Decimal


class IncomeReportTotal(TypedDict):  # pylint: disable=too-few-public-methods
    amount: Decimal
    fees: Decimal


@dataclass
class BuyAccumulator:
    quantity: Decimal = Decimal(0)
    cost: Decimal = Decimal(0)
    fee_value: Decimal = Decimal(0)
    dates: List[Date] = field(default_factory=list)

class YearlyThresholdReport(TypedDict):
    exceeded_threshold: bool
    days_above_threshold: List[date]
    consecutive_working_days: int
    missing_prices: bool

class MarginReportTotal(TypedDict):  # pylint: disable=too-few-public-methods
    gains: Decimal
    losses: Decimal
    fees: Decimal

class TaxCalculator:  # pylint: disable=too-many-instance-attributes
    INCOME_TYPES = (
        TrType.MINING,
        TrType.STAKING_REWARD,
        TrType.STAKING,
        TrType.DIVIDEND,
        TrType.INTEREST,
        TrType.INCOME,
        TrType.FORK,
        TrType.AIRDROP,
        TrType.REFERRAL,
        TrType.CASHBACK,
        TrType.FEE_REBATE,
    )

    NO_GAIN_NO_LOSS_TYPES = (
        TrType.GIFT_SENT,
        TrType.GIFT_SPOUSE,
        TrType.CHARITY_SENT,
        TrType.LOST,
        TrType.SWAP,
        TrType.CRYPTO_CRYPTO,
    )

    MARGIN_TYPES = (TrType.MARGIN_GAIN, TrType.MARGIN_LOSS, TrType.MARGIN_FEE)

    def __init__(self, transactions: List[Union[Buy, Sell]], tax_rules: str) -> None:
        self.transactions = transactions
        self.tax_rules = tax_rules
        self.buys_ordered: Dict[AssetSymbol, List[Buy]] = {}
        self.sells_ordered: List[Sell] = []
        self.other_transactions: List[Union[Buy, Sell]] = []

        self.match_missing = False
        self.tax_events: Dict[Year, List[TaxEvent]] = {}
        self.holdings: Dict[AssetSymbol, Holdings] = {}

        self.tax_report: Dict[Year, TaxReportRecord] = {}
        self.holdings_report: Optional[HoldingsReportRecord] = None
        self.yearly_holdings_report: Optional[Dict[Year, YearlyReportRecord]] = None
        self.daily_holdings_report: Optional[Dict[Year, Dict[date, DailyReportRecord]]] = {}

    def order_transactions(self) -> None:
        if config.debug:
            print(f"{Fore.CYAN}order transactions")

        for t in tqdm(
            self.transactions,
            unit="t",
            desc=f"{Fore.CYAN}order transactions{Fore.GREEN}",
            disable=bool(config.debug or not sys.stdout.isatty()),
        ):
            if isinstance(t, Buy) and t.is_crypto() and t.acquisition:
                if t.asset not in self.buys_ordered:
                    self.buys_ordered[t.asset] = []

                self.buys_ordered[t.asset].append(t)
            elif isinstance(t, Sell) and t.is_crypto() and t.disposal:
                self.sells_ordered.append(t)
            else:
                self.other_transactions.append(t)

        if config.debug:
            for a in sorted(self.buys_ordered):
                for t in self.buys_ordered[a]:
                    print(f"{Fore.GREEN}buys: {t}")

            for t in self.sells_ordered:
                print(f"{Fore.GREEN}sells: {t}")

    def fifo_match(self) -> None:
        # Keep copy for income tax processing
        self.transactions = copy.deepcopy(self.transactions)

        if config.debug:
            print(f"{Fore.CYAN}fifo match transactions")

        buy_index = {}

        for s in tqdm(
            self.sells_ordered,
            unit="t",
            desc=f"{Fore.CYAN}fifo match transactions{Fore.GREEN}",
            disable=bool(config.debug or not sys.stdout.isatty()),
        ):
            matches = []
            s_quantity_remaining = s.quantity

            if config.debug:
                print(f"{Fore.GREEN}fifo: {s}")

            if s.asset not in buy_index:
                buy_index[s.asset] = 0

            while (
                s.asset in self.buys_ordered
                and buy_index[s.asset] < len(self.buys_ordered[s.asset])
                and s_quantity_remaining > 0
            ):
                b = self.buys_ordered[s.asset][buy_index[s.asset]]

                if b.date() > s.date():
                    break

                if b.quantity > s_quantity_remaining:
                    b_remainder = b.split_buy(s_quantity_remaining)
                    self.buys_ordered[s.asset].insert(buy_index[s.asset] + 1, b_remainder)

                    if config.debug:
                        print(f"{Fore.GREEN}fifo: {b} <-- split")
                        print(f"{Fore.YELLOW}fifo:   {b_remainder} <-- split")
                else:
                    if config.debug:
                        print(f"{Fore.GREEN}fifo: {b}")

                b.matched = True
                matches.append(b)
                s_quantity_remaining -= b.quantity

                buy_index[s.asset] += 1

            if s_quantity_remaining > 0:
                tqdm.write(
                    f"{WARNING} No matching Buy of {s_quantity_remaining.normalize():0,f} "
                    f"{s.asset} for Sell of {s.format_quantity()} {s.asset}"
                )
                if config.cost_basis_zero_if_missing:
                    buy_match = Buy(TrType.TRADE, s_quantity_remaining, s.asset, Decimal(0))
                    buy_match.wallet = s.wallet
                    buy_match.timestamp = s.timestamp
                    buy_match.note = Note("Added as cost basis zero")
                    tqdm.write(f"{Fore.GREEN}fifo: {buy_match}")
                    matches.append(buy_match)
                    s.matched = True
                    self._create_disposal(s, matches)
                else:
                    self.match_missing = True
            else:
                s.matched = True
                self._create_disposal(s, matches)

    def lifo_match(self) -> None:
        # Keep copy for income tax processing
        self.transactions = copy.deepcopy(self.transactions)

        if config.debug:
            print(f"{Fore.CYAN}lifo match transactions")

        buy_index = {}

        for s in tqdm(
            self.sells_ordered,
            unit="t",
            desc=f"{Fore.CYAN}lifo match transactions{Fore.GREEN}",
            disable=bool(config.debug or not sys.stdout.isatty()),
        ):
            matches = []
            s_quantity_remaining = s.quantity

            if config.debug:
                print(f"{Fore.GREEN}lifo: {s}")

            if s.asset in self.buys_ordered:
                for i in reversed(range(len(self.buys_ordered[s.asset]))):
                    if self.buys_ordered[s.asset][i].date() <= s.date() and not self.buys_ordered[s.asset][i].matched:
                        buy_index[s.asset] = i
                        break
                else:
                    buy_index[s.asset] = -1

            while (
                s.asset in self.buys_ordered
                and buy_index[s.asset] >= 0
                and s_quantity_remaining > 0
            ):
                b = self.buys_ordered[s.asset][buy_index[s.asset]]

                if b.date() > s.date():
                    break

                if b.matched:
                    buy_index[s.asset] -= 1
                    continue

                if b.quantity > s_quantity_remaining:
                    b_remainder = b.split_buy(s_quantity_remaining)
                    self.buys_ordered[s.asset].insert(buy_index[s.asset] + 1, b_remainder)

                    if config.debug:
                        print(f"{Fore.GREEN}lifo: {b} <-- split")
                        print(f"{Fore.YELLOW}lifo:   {b_remainder} <-- split")
                else:
                    if config.debug:
                        print(f"{Fore.GREEN}lifo: {b}")

                b.matched = True
                matches.append(b)
                s_quantity_remaining -= b.quantity

                buy_index[s.asset] -= 1

            if s_quantity_remaining > 0:
                tqdm.write(
                    f"{WARNING} No matching Buy of {s_quantity_remaining.normalize():0,f} "
                    f"{s.asset} for Sell of {s.format_quantity()} {s.asset}"
                )
                if config.cost_basis_zero_if_missing:
                    buy_match = Buy(TrType.TRADE, s_quantity_remaining, s.asset, Decimal(0))
                    buy_match.wallet = s.wallet
                    buy_match.timestamp = s.timestamp
                    buy_match.note =  Note("Added as cost basis zero")
                    tqdm.write(f"{Fore.GREEN}lifo: {buy_match}")
                    matches.append(buy_match)
                    s.matched = True
                    self._create_disposal(s, matches)
                else:
                    self.match_missing = True
            else:
                s.matched = True
                self._create_disposal(s, matches)

    def _create_disposal(self, sell: Sell, matches: List[Buy]) -> None:
        short_term = BuyAccumulator()
        long_term = BuyAccumulator()

        for buy in matches:
            if buy.cost is None:
                raise RuntimeError("Missing cost")

            if sell.date() <= buy.date() + relativedelta(years=1):
                short_term.quantity += buy.quantity
                short_term.cost += buy.cost
                if buy.fee_value:
                    short_term.fee_value += buy.fee_value
                short_term.dates.append(buy.date())
            else:
                long_term.quantity += buy.quantity
                long_term.cost += buy.cost
                if buy.fee_value:
                    long_term.fee_value += buy.fee_value
                long_term.dates.append(buy.date())

        if short_term.dates:
            sell_adjusted = copy.deepcopy(sell)
            if sell.proceeds is None:
                raise RuntimeError("Missing sell.proceeds")

            if sell.fee_value:
                sell_adjusted.proceeds = sell.proceeds.quantize(
                    PRECISION
                ) - sell.fee_value.quantize(PRECISION)

                if sell_adjusted.proceeds < 0:
                    sell_adjusted.proceeds = Decimal(0)

            if sell_adjusted.proceeds is None:
                raise RuntimeError("Missing sell_adjusted.proceeds")

            sell_adjusted.proceeds = sell_adjusted.proceeds * (short_term.quantity / sell.quantity)

            if sell.t_type in self.NO_GAIN_NO_LOSS_TYPES:
                tax_event: Union[TaxEventCapitalGains, TaxEventNoGainNoLoss] = TaxEventNoGainNoLoss(
                    DisposalType.SHORT_TERM,
                    sell_adjusted,
                    short_term.cost.quantize(PRECISION) + short_term.fee_value.quantize(PRECISION),
                    Decimal(0),
                    short_term.dates,
                )
            else:
                tax_event = TaxEventCapitalGains(
                    DisposalType.SHORT_TERM,
                    sell_adjusted,
                    short_term.cost.quantize(PRECISION) + short_term.fee_value.quantize(PRECISION),
                    Decimal(0),
                    short_term.dates,
                )
            self.tax_events[self._which_tax_year(tax_event.date)].append(tax_event)

            if config.debug:
                print(f"{Fore.CYAN}fifo: {tax_event}")

        if long_term.dates:
            sell_adjusted = copy.deepcopy(sell)
            if sell.proceeds is None:
                raise RuntimeError("Missing sell.proceeds")

            if sell.fee_value:
                sell_adjusted.proceeds = sell.proceeds.quantize(
                    PRECISION
                ) - sell.fee_value.quantize(PRECISION)

                if sell_adjusted.proceeds < 0:
                    sell_adjusted.proceeds = Decimal(0)

            if sell_adjusted.proceeds is None:
                raise RuntimeError("Missing sell_adjusted.proceeds")

            sell_adjusted.proceeds = sell_adjusted.proceeds * (long_term.quantity / sell.quantity)

            if sell.t_type in self.NO_GAIN_NO_LOSS_TYPES:
                tax_event = TaxEventNoGainNoLoss(
                    DisposalType.LONG_TERM,
                    sell_adjusted,
                    long_term.cost.quantize(PRECISION) + long_term.fee_value.quantize(PRECISION),
                    Decimal(0),
                    long_term.dates,
                )
            else:
                tax_event = TaxEventCapitalGains(
                    DisposalType.LONG_TERM,
                    sell_adjusted,
                    long_term.cost.quantize(PRECISION) + long_term.fee_value.quantize(PRECISION),
                    Decimal(0),
                    long_term.dates,
                )
            self.tax_events[self._which_tax_year(tax_event.date)].append(tax_event)

            if config.debug:
                print(f"{Fore.CYAN}fifo: {tax_event}")

        if sell.t_type in (TrType.SWAP, TrType.CRYPTO_CRYPTO):
            # Cost basis copied to "Buy" asset
            if sell.t_record and sell.t_record.buy:
                sell.t_record.buy.cost = short_term.cost + long_term.cost
                if short_term.fee_value + long_term.fee_value:
                    sell.t_record.buy.fee_value = short_term.fee_value + long_term.fee_value
            else:
                raise RuntimeError("Missing t_record.buy for SWAP/CRYPTO_CRYPTO")

    def process_holdings(self, tax_year: Optional[Year] = None) -> None:
        if config.debug:
            print(f"{Fore.CYAN}process holdings")

        transactions = sorted(self._all_transactions())
        if tax_year:
            transactions = [t for t in transactions if t.year <= tax_year]

        for t in tqdm(
            transactions,
            unit="t",
            desc=f"{Fore.CYAN}process holdings{Fore.GREEN}",
            disable=bool(config.debug or not sys.stdout.isatty()),
        ):
            if t.asset not in self.holdings:
                self.holdings[t.asset] = Holdings(t.asset)

            if t.matched:
                if config.debug:
                    print(f"{Fore.BLUE}holdings: //{t} <- matched")
                self._process_transaction(t)
                continue

            if not config.transfers_include and t.t_type in TRANSFER_TYPES:
                if config.debug:
                    print(f"{Fore.BLUE}holdings: //{t} <- transfer {'crypto' if t.is_crypto() else 'fiat'}")

                if isinstance(t, Buy):
                    self._process_transaction(t)
                elif isinstance(t, Sell):
                    self._process_transaction(t)
                continue

            if t.is_fiat():
                if config.debug:
                    print(f"{Fore.BLUE}holdings: //{t} <- fiat")
                self._process_transaction(t)
                continue

            if config.debug:
                print(f"{Fore.GREEN}holdings: {t}")

            if isinstance(t, Buy):
                self._add_tokens(t)
            elif isinstance(t, Sell):
                self._subtract_tokens(t)

    def _update_balance(self, t, operation):
        if config.debug:
            print(f"{Fore.GREEN}Processing {operation.__name__.capitalize()} for {t.asset}, Quantity: {t.quantity}, Timestamp: {t.timestamp}")
        operation(t.quantity, t.timestamp)

    def _update_cost(self, t, operation, cost):
        if config.debug:
            print(f"{Fore.GREEN}Updating cost for {t.asset}, Cost: {cost}, Timestamp: {t.timestamp}")
        operation(cost, t.timestamp)  # Chiamiamo l'operazione (add o subtract cost history)

    def _process_transaction(self, t):
        if isinstance(t, Buy):
            self._update_balance(t, self.holdings[t.asset]._addto_balance_history)
            self._update_cost(t, self.holdings[t.asset]._addto_cost_history, t.cost)
        elif isinstance(t, Sell):
            self._update_balance(t, self.holdings[t.asset]._subctractto_balance_history)
        if config.debug:
            print(f"{Fore.YELLOW}Updated Holdings for {t.asset}: {self.holdings[t.asset]}")

    def _add_tokens(self, t: Buy) -> None:
        if not t.acquisition:
            cost = fees = Decimal(0)
        else:
            if t.cost is None:
                raise RuntimeError("Missing cost")

            cost = t.cost
            fees = t.fee_value or Decimal(0)

        self.holdings[t.asset].add_tokens(t.quantity, cost, fees, t.t_type is TrType.DEPOSIT, t.timestamp)

    def _subtract_tokens(self, t: Sell) -> None:
        if not t.disposal:
            self.holdings[t.asset].subtract_tokens(
                t.quantity, Decimal(0), Decimal(0), t.t_type is TrType.WITHDRAWAL, t.timestamp
            )
        else:
            raise RuntimeError("Unmatched disposal in holdings")

    def process_income(self) -> None:
        if config.debug:
            print(f"{Fore.CYAN}process income")

        for t in tqdm(
            self.transactions,
            unit="t",
            desc=f"{Fore.CYAN}process income{Fore.GREEN}",
            disable=bool(config.debug or not sys.stdout.isatty()),
        ):
            if t.t_type in self.INCOME_TYPES and (t.is_crypto() or config.fiat_income):
                tax_event = TaxEventIncome(t)
                self.tax_events[self._which_tax_year(tax_event.date)].append(tax_event)

    def _all_transactions(self) -> Iterator[Union[Buy, Sell]]:
        return itertools.chain(
            *self.buys_ordered.values(), self.sells_ordered, self.other_transactions
        )

    def process_margin_trades(self) -> None:
        if config.debug:
            print(f"{Fore.CYAN}process margin trades")

        for t in tqdm(
            self.transactions,
            unit="t",
            desc=f"{Fore.CYAN}process margin trades{Fore.GREEN}",
            disable=bool(config.debug or not sys.stdout.isatty()),
        ):

            if t.t_type in self.MARGIN_TYPES:
                tax_event = TaxEventMarginTrade(t)
                self.tax_events[self._which_tax_year(tax_event.date)].append(tax_event)

    def calculate_capital_gains(self, tax_year: Year) -> "CalculateCapitalGains":
        calc_cgt = CalculateCapitalGains(tax_year)

        if tax_year in self.tax_events:
            for te in sorted(self.tax_events[tax_year]):
                if isinstance(te, TaxEventCapitalGains):
                    calc_cgt.tax_summary(te)
                elif isinstance(te, TaxEventNoGainNoLoss):
                    calc_cgt.non_tax_summary(te)

        if self.tax_rules in TAX_RULES_UK_COMPANY:
            calc_cgt.tax_estimate_ct(tax_year)
        else:
            calc_cgt.tax_estimate_cgt(tax_year)

        return calc_cgt

    def calculate_income(self, tax_year: Year) -> "CalculateIncome":
        calc_income = CalculateIncome()

        if tax_year in self.tax_events:
            for te in sorted(self.tax_events[tax_year]):
                if isinstance(te, TaxEventIncome):
                    calc_income.totalise(te)

        calc_income.totals_by_type()
        return calc_income

    def calculate_margin_trading(self, tax_year: Year) -> "CalculateMarginTrading":
        calc_margin_trading = CalculateMarginTrading()

        if tax_year in self.tax_events:
            for te in sorted(self.tax_events[tax_year]):
                if isinstance(te, TaxEventMarginTrade):
                    calc_margin_trading.totalise(te)

        calc_margin_trading.totals_by_contract()
        return calc_margin_trading

    def calculate_holdings(self, value_asset: ValueAsset) -> None:
        holdings: Dict[AssetSymbol, HoldingsReportAsset] = {}
        totals: HoldingsReportTotal = {"cost": Decimal(0), "value": Decimal(0), "gain": Decimal(0)}

        if config.debug:
            print(f"{Fore.CYAN}calculating holdings")

        for h in tqdm(
            self.holdings,
            unit="h",
            desc=f"{Fore.CYAN}calculating holdings{Fore.GREEN}",
            disable=bool(config.debug or not sys.stdout.isatty()),
        ):
            if self.holdings[h].quantity > 0 or config.show_empty_wallets:
                try:
                    value, name, _ = value_asset.get_current_value(
                        self.holdings[h].asset, self.holdings[h].quantity
                    )
                except requests.exceptions.HTTPError as e:
                    tqdm.write(
                        f"{WARNING} Skipping valuation of {self.holdings[h].asset} "
                        f"due to API failure ({e.response.status_code})"
                    )
                    value = None
                    name = AssetName("")

                value = value.quantize(PRECISION) if value is not None else None
                cost = (self.holdings[h].cost + self.holdings[h].fees).quantize(PRECISION)

                if value is not None:
                    holdings[h] = {
                        "name": name,
                        "quantity": self.holdings[h].quantity,
                        "cost": cost,
                        "value": value,
                        "gain": value - cost,
                    }

                    totals["value"] += value
                    totals["gain"] += value - cost
                else:
                    holdings[h] = {
                        "name": name,
                        "quantity": self.holdings[h].quantity,
                        "cost": cost,
                        "value": None,
                    }

                totals["cost"] += holdings[h]["cost"]

        self.holdings_report = {"holdings": holdings, "totals": totals}

    def _which_tax_year(self, date: Date) -> Year:
        if date > config.get_tax_year_end(date.year):
            tax_year = Year(date.year + 1)
        else:
            tax_year = Year(date.year)

        if tax_year not in self.tax_events:
            self.tax_events[tax_year] = []

        return tax_year

    def _end_of_year(self, year: int) -> Date:
        return config.get_tax_year_end(year)

    def _start_of_year(self, year: int) -> Date:
        return config.get_tax_year_start(year)

    def _get_first_tax_year(self) -> Optional[Year]:
        """
        Identify the first fiscal year based on the processed transactions and sorted in chronological order.
        Search all sorted transactions to find the first transaction with a valid date.
        """
        # Check the ordered transactions
        all_ordered_transactions = list(self._all_transactions())

        if not all_ordered_transactions:
            raise ValueError("No transactions available to determine the starting fiscal year.")
    
        # Find the transaction with the oldest date
        first_transaction = min(all_ordered_transactions, key=lambda t: t.timestamp)
        first_year = first_transaction.timestamp.year

        if config.debug:
            print(f"First year identified by processed transactions: {first_year}")

        return first_year

    def calculate_yearly_holdings(self, value_asset: ValueAsset, tax_year: Optional[int] = None) -> None:
        """
        Calculate the annual holdings for each asset held with a fiat value and quantity.
        If a tax_year is given, generate the report for that year only.
        Save the report in self.yearly_holdings_report.

        :param value_asset: The ValueAsset instance to get historical prices.
        :param tax_year: (optional) The tax year to generate the report for.
        """
        if config.debug:
            print(f"{Fore.CYAN}Calculating yearly holdings")

        current_year = datetime.now().year
        current_date = datetime.now().date()

        # If a tax_year is not provided, generate the report for all years excluding the current year.
        if tax_year is None:
            first_year = self._get_first_tax_year()
            years = range(first_year, current_year + 1)
            if config.debug:
                print(f"Generating report for all years from {first_year} to {current_year}")
        else:
            # If tax_year is in the future, raise an exception
            if tax_year > current_year:
                raise ValueError("Future years cannot be selected.")
            years = [tax_year]  # Only the specified fiscal year
            if config.debug:
                print(f"Generating report for the year {tax_year}")

        yearly_holdings_report = {}

        if self.yearly_holdings_report is None:
            self.yearly_holdings_report = {}

        # Cycle through each fiscal year with progress bar
        for year in tqdm(years, desc=f"{Fore.CYAN}Generating yearly report{Fore.GREEN}"):
            if year == current_year:
                # If it's the current year, we use the current date instead of the end of the year
                end_of_year_date = current_date
            else:
                # Get end-of-year date (just the date part)
                end_of_year_date = self._end_of_year(year)

            # Get end-of-year datetime (date + time)
            end_of_year_datetime = datetime.combine(end_of_year_date, time.min)  # This is a datetime (with time)

            # Convert end_of_year_datetime in UTC
            end_of_year_datetime_utc = end_of_year_datetime.replace(tzinfo=timezone.utc)

            # Get the start-of-year date
            start_of_year_date = self._start_of_year(year)
            start_of_year_datetime = datetime.combine(start_of_year_date, time.min)
            start_of_year_datetime_utc = start_of_year_datetime.replace(tzinfo=timezone.utc)

            assets_report = {}
            total_value_in_fiat = Decimal(0)  # Initialize the sum to 0 for the year
            total_value_in_fiat_start_of_year = Decimal(0)  # Initialize the sum to 0 for the year

            if config.debug:
                print(f"Processing year {year}: from {start_of_year_date} to {end_of_year_date}")

            # Cycle on each asset in holdings with progress bar
            for h in tqdm(self.holdings, unit="asset", desc=f"Processing year {year}", leave=False):
                holdings = self.holdings[h]

                # Check if the asset had a positive balance on any day of the year
                had_positive_balance_during_year = False
                for day in (start_of_year_date + timedelta(days=n) for n in range((end_of_year_date - start_of_year_date).days + 1)):
                    quantity_at_day = holdings.get_balance_at_date(day)
                    if quantity_at_day > 0:
                        had_positive_balance_during_year = True
                        break

                # Quantity check (exclude if no positive balance, unless config.show_empty_wallets is enabled)
                if had_positive_balance_during_year:
                    if config.debug:
                        print(f"Processing asset {h} for year {year}")

                    # Quantità a inizio anno
                    quantity_start_of_year = holdings.get_balance_at_date(start_of_year_date)

                    # Valore in fiat a inizio anno
                    if holdings.is_crypto():
                        try:
                            price_at_start_of_year, _, _ = value_asset.get_historical_price(h, start_of_year_datetime_utc, no_cache=False)
                            if price_at_start_of_year is not None:
                                value_in_fiat_start_of_year = quantity_start_of_year * price_at_start_of_year
                            else:
                                value_in_fiat_start_of_year = Decimal(0)
                        except requests.exceptions.HTTPError as e:
                            tqdm.write(f"Warning: Unable to get historical price for {h} on {start_of_year_datetime} due to HTTP error: {e}")
                            value_in_fiat_start_of_year = Decimal(0)
                    elif holdings.is_fiat():
                        value_in_fiat_start_of_year = quantity_start_of_year
                    else:
                        value_in_fiat_start_of_year = Decimal(0)

                    # Add to total value in fiat at start of year
                    total_value_in_fiat_start_of_year += value_in_fiat_start_of_year

                    # Get quantity at the end of the year (or current date if it's the current year)
                    quantity_end_of_year = holdings.get_balance_at_date(end_of_year_date)

                    # Calculate average balance and days held
                    average_balance = holdings.calculate_average_balance(start_of_year_date, end_of_year_date)
                    days_held = holdings.calculate_days_held(start_of_year_date, end_of_year_date)

                    if config.debug:
                        print(f"Asset {h}: Quantity at end of year = {quantity_end_of_year}, Average balance = {average_balance}")

                    if holdings.is_crypto():
                        # If it's a crypto asset, get the historical price
                        try:
                            if year == current_year:
                                # For the current year, use the current price
                                value_in_fiat, _, _ = value_asset.get_current_value(h, quantity_end_of_year)
                            else:
                                # For previous years, get the historical price at the end of the year
                                price_at_end_of_year, _, _ = value_asset.get_historical_price(h, end_of_year_datetime_utc, no_cache=False)

                                if price_at_end_of_year is not None:
                                    value_in_fiat = quantity_end_of_year * price_at_end_of_year
                                    if config.debug:
                                        print(f"Asset {h}: Price at end of year = {price_at_end_of_year}, Value in fiat = {value_in_fiat}")
                                else:
                                    value_in_fiat = Decimal(0)  # If no price is found, set it to 0
                                    if config.debug:
                                        print(f"Asset {h}: Price not found for end of year, setting value in fiat to 0")
                            if value_in_fiat is None:
                                value_in_fiat = Decimal(0)
                                if config.debug:
                                    print(f"Asset {h}: Value in fiat is None, setting to 0")
                        except requests.exceptions.HTTPError as e:
                            tqdm.write(f"Warning: Unable to get historical price for {h} on {end_of_year_datetime} due to HTTP error: {e}")
                            value_in_fiat = Decimal(0)
                            if config.debug:
                                print(f"Asset {h}: HTTP error, setting value in fiat to 0")
                    if holdings.is_fiat():
                        # If it is not a cryptocurrency, use the amount directly as fiat value
                        value_in_fiat = quantity_end_of_year
                        if config.debug:
                            print(f"Asset {h} is not a cryptocurrency, using quantity as value in fiat: {value_in_fiat}")

                    # Add the fiat value to the annual sum
                    total_value_in_fiat += value_in_fiat

                    # Save details in report
                    assets_report[h] = YearlyReportAsset(
                        quantity_start_of_year=quantity_start_of_year,
                        value_in_fiat_start_of_year=value_in_fiat_start_of_year,
                        quantity_end_of_year=quantity_end_of_year,
                        average_balance=average_balance,
                        value_in_fiat_at_end_of_year=value_in_fiat,
                        days_held=days_held
                    )

            # Calcola la data di scadenza per l'anno fiscale
            scadenza = date(year, 6, 30)  # Data di scadenza impostata al 30 giugno

            # Calcola la sanzione utilizzando la data di scadenza corretta
            if config.debug:
                print(f"Calcolo sanzione RW per anno {year}, data di scadenza: {scadenza}, valore_attivita_estere={total_value_in_fiat} ")

            penalty_due_rw = self.calcola_sanzione_valore_attivita_estere(
	            valore_attivita_estere=total_value_in_fiat,
	            data_scadenza=scadenza,
	            paese_black_list=False,  # oppure True se pertinente
            )

            # Save the total amount in the report
            yearly_holdings_report[year] = YearlyReportRecord(
                assets=assets_report,
                totals=YearlyReportTotal(
                    total_value_in_fiat_start_of_year=total_value_in_fiat_start_of_year,
                    total_value_in_fiat_at_end_of_year=total_value_in_fiat,
                    penalty_due_rw=penalty_due_rw
                )
            )

            if config.debug:
                print(f"Year {year}: Total value in fiat at end of year = {total_value_in_fiat}")

        # Save the report in the self.yearly_holdings_report instance field
        self.yearly_holdings_report = yearly_holdings_report
        tqdm.write("Yearly report saved to self.yearly_holdings_report.")

    def calcola_interessi_mora(self, importo_dovuto, data_scadenza, data_pagamento) -> dict:
        tassi_interessi = [
            (date(2016, 1, 1), date(2016, 12, 31), Decimal('0.002')),
            (date(2017, 1, 1), date(2017, 12, 31), Decimal('0.001')),
            (date(2018, 1, 1), date(2018, 12, 31), Decimal('0.003')),
            (date(2019, 1, 1), date(2019, 12, 31), Decimal('0.008')),
            (date(2020, 1, 1), date(2020, 12, 31), Decimal('0.0005')),
            (date(2021, 1, 1), date(2021, 12, 31), Decimal('0.0001')),
            (date(2022, 1, 1), date(2022, 12, 31), Decimal('0.0125')),
            (date(2023, 1, 1), date(2023, 12, 31), Decimal('0.05')),
            (date(2024, 1, 1), date(9999, 12, 31), Decimal('0.025')),
        ]

        interessi = Decimal('0.00')
        data_inizio = data_scadenza
        data_fine = data_pagamento
        dettagli_calcolo = []

        if config.debug:
            print(f"Calcolo interessi per importo dovuto: {importo_dovuto}")
            print(f"Data scadenza: {data_scadenza}, Data pagamento: {data_pagamento}")

        for inizio, fine, tasso in tassi_interessi:
            if data_inizio >= data_fine:
                break
            periodo_inizio = max(data_inizio, inizio)
            periodo_fine = min(data_fine, fine)
            if periodo_inizio < periodo_fine:
                giorni = (periodo_fine - periodo_inizio).days
                interessi_annui = (importo_dovuto * tasso * giorni) / Decimal('365')
                interessi += interessi_annui
                descrizione = (
                    f"Periodo: {periodo_inizio} - {periodo_fine}, Giorni: {giorni}, "
                    f"Tasso: {tasso}, Interessi: {interessi_annui}"
                )
                dettagli_calcolo.append(descrizione)

                if config.debug:
                    print(descrizione)

        interessi = interessi.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        if config.debug:
            print(f"Interessi totali calcolati: {interessi}")

        return {
            'interessi': interessi,
            'dettagli': dettagli_calcolo
        }

    def calcola_sanzione_imposta_dovuta(self, imposta_dovuta: Decimal, tax_year: int, data_pagamento: date = None) -> dict:
        if not isinstance(imposta_dovuta, Decimal):
            raise TypeError(f"Expected imposta_dovuta to be of type Decimal, but got {type(imposta_dovuta)}")
        if imposta_dovuta <= 0:
            return {
                'sanzione': Decimal('0.00'),
                'interessi_di_mora': Decimal('0.00'),
                'totale': Decimal('0.00'),
                'dettagli_calcolo_interessi': [],
                'dettagli_calcolo_sanzione': []
            }

        data_scadenza = date(tax_year + 1, 6, 30)
        if data_pagamento is None:
            data_pagamento = datetime.now().date()

        giorni_ritardo = max((data_pagamento - data_scadenza).days, 0)
        dettagli_calcolo_sanzione = []

        if config.debug:
            print(f"Data scadenza: {data_scadenza}, Data pagamento: {data_pagamento}, Giorni ritardo: {giorni_ritardo}")

        sanzione_base = imposta_dovuta * Decimal('0.30')
        if giorni_ritardo <= 14:
            sanzione = imposta_dovuta * Decimal('0.001') * giorni_ritardo
            descrizione = f"Importo: Imposta dovuta {imposta_dovuta}, Giorni: {giorni_ritardo}, Tasso giornaliero: 0.1%, Sanzione calcolata: {sanzione}"
        elif giorni_ritardo <= 30:
            sanzione = sanzione_base * Decimal('0.1')
            descrizione = f"Importo: Imposta dovuta {imposta_dovuta}, Giorni: {giorni_ritardo}, Tasso base: 10%, Sanzione calcolata: {sanzione}"
        elif giorni_ritardo <= 90:
            sanzione = sanzione_base * (Decimal('1') / Decimal('9'))
            descrizione = f"Importo: Imposta dovuta {imposta_dovuta}, Giorni: {giorni_ritardo}, Tasso base: 1/9, Sanzione calcolata: {sanzione}"
        elif giorni_ritardo <= 365:
            sanzione = sanzione_base * (Decimal('1') / Decimal('8'))
            descrizione = f"Importo: Imposta dovuta {imposta_dovuta}, Giorni: {giorni_ritardo}, Tasso base: 1/8, Sanzione calcolata: {sanzione}"
        elif giorni_ritardo <= 730:
            sanzione = sanzione_base * (Decimal('1') / Decimal('7'))
            descrizione = f"Importo: Imposta dovuta {imposta_dovuta}, Giorni: {giorni_ritardo}, Tasso base: 1/7, Sanzione calcolata: {sanzione}"
        else:
            sanzione = sanzione_base * (Decimal('1') / Decimal('6'))
            descrizione = f"Importo: Imposta dovuta {imposta_dovuta}, Giorni: {giorni_ritardo}, Tasso base: 1/6, Sanzione calcolata: {sanzione}"

        dettagli_calcolo_sanzione.append(descrizione)

        if config.debug:
            print(f"Sanzione base: {sanzione_base}, Sanzione calcolata: {sanzione}")
            print(descrizione)

        interessi_mora = self.calcola_interessi_mora(imposta_dovuta, data_scadenza, data_pagamento)

        if config.debug:
            print(f"Interessi di mora calcolati: {interessi_mora['interessi']}")
            for dettaglio in interessi_mora['dettagli']:
                print(dettaglio)

        totale = sanzione + interessi_mora['interessi']

        return {
            'sanzione': sanzione.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'interessi_di_mora': interessi_mora['interessi'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'totale': totale.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'dettagli_calcolo_interessi': interessi_mora['dettagli'],
            'dettagli_calcolo_sanzione': dettagli_calcolo_sanzione
        }

    def calcola_sanzione_valore_attivita_estere(
        self,
        valore_attivita_estere: Decimal,
        data_scadenza: date,
        paese_black_list: bool,
        data_pagamento: date = None,
    ) -> dict:
        if not isinstance(valore_attivita_estere, Decimal):
            raise TypeError(f"Expected valore_attivita_estere to be of type Decimal, but got {type(valore_attivita_estere)}")
        if valore_attivita_estere <= 0:
            return {
                'sanzione': Decimal('0.00'),
                'interessi_di_mora': Decimal('0.00'),
                'totale': Decimal('0.00'),
                'dettagli_calcolo_interessi': [],
                'dettagli_calcolo_sanzione': []
            }
        if data_pagamento is None:
            data_pagamento = datetime.now().date()

        giorni_ritardo = max((data_pagamento - data_scadenza).days, 0)
        dettagli_calcolo_sanzione = []

        if config.debug:
            print(f"Data scadenza: {data_scadenza}, Data pagamento: {data_pagamento}, Giorni ritardo: {giorni_ritardo}")

        percentuale_sanzione = Decimal('0.06') if paese_black_list else Decimal('0.03')
        sanzione_base = valore_attivita_estere * percentuale_sanzione
        if giorni_ritardo <= 90:
            sanzione = sanzione_base * (Decimal('1') / Decimal('6'))
            descrizione = f"Importo: Valore attività estere {valore_attivita_estere}, Giorni: {giorni_ritardo}, Tasso base: 1/6, Sanzione calcolata: {sanzione}"
        elif giorni_ritardo <= 365:
            sanzione = sanzione_base * (Decimal('1') / Decimal('5'))
            descrizione = f"Importo: Valore attività estere {valore_attivita_estere}, Giorni: {giorni_ritardo}, Tasso base: 1/5, Sanzione calcolata: {sanzione}"
        elif giorni_ritardo <= 730:
            sanzione = sanzione_base * (Decimal('1') / Decimal('4'))
            descrizione = f"Importo: Valore attività estere {valore_attivita_estere}, Giorni: {giorni_ritardo}, Tasso base: 1/4, Sanzione calcolata: {sanzione}"
        else:
            sanzione = sanzione_base * (Decimal('1') / Decimal('3'))
            descrizione = f"Importo: Valore attività estere {valore_attivita_estere}, Giorni: {giorni_ritardo}, Tasso base: 1/3, Sanzione calcolata: {sanzione}"

        dettagli_calcolo_sanzione.append(descrizione)

        if config.debug:
            print(f"Sanzione base: {sanzione_base}, Sanzione calcolata: {sanzione}")
            print(descrizione)

        interessi_mora = self.calcola_interessi_mora(valore_attivita_estere, data_scadenza, data_pagamento)

        if config.debug:
            print(f"Interessi di mora calcolati: {interessi_mora['interessi']}")
            for dettaglio in interessi_mora['dettagli']:
                print(dettaglio)

        totale = sanzione + interessi_mora['interessi']

        return {
            'sanzione': sanzione.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'interessi_di_mora': interessi_mora['interessi'].quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'totale': totale.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            'dettagli_calcolo_interessi': interessi_mora['dettagli'],
            'dettagli_calcolo_sanzione': dettagli_calcolo_sanzione
        }


    def check_holding_threshold(self, value_asset: ValueAsset, tax_year: int, threshold: Decimal = Decimal("51645.69")) -> bool:
        """
        Check if the euro value of the cryptocurrencies held has exceeded the threshold of €51,645.69
        for at least seven consecutive working days, only for tax years up to 2022 (inclusive).
        """
        if config.debug:
            print(f"{Fore.CYAN}Checking holding threshold for the year {tax_year}")

        # Skip calculation if the year is greater than 2022
        if tax_year > 2022:
            return False  # Skip calculation for years after 2022

        # Retrieve the start and end date of the fiscal year
        start_date = self._start_of_year(tax_year)
        end_date = self._end_of_year(tax_year)

        current_value = Decimal(0)
        consecutive_working_days = 0
        days_above_threshold = []
        exceeded_threshold = False
        missing_prices = False

        current_datetime = datetime.combine(start_date, time.min)
        current_datetime = current_datetime.replace(tzinfo=timezone.utc)
        current_date = current_datetime.date()

        end_datetime = datetime.combine(end_date, time.min)
        end_datetime = end_datetime.replace(tzinfo=timezone.utc)

        while current_date <= end_date:
            # Consider only working days (Mon-Fri)
            if current_date.weekday() < 5:  # Monday = 0, ..., Friday = 4
                # Calculate the value on that day
                current_value = Decimal(0)
                for asset, holding in self.holdings.items():
                    if holding.is_crypto():
                        # Get the balance for that date
                        quantity = holding.get_balance_at_date(current_date)
                        price_at_date, _, _ = value_asset.get_historical_price(asset, current_datetime, no_cache=False)

                        # Add the total value
                        if price_at_date is not None:
                            current_value += quantity * price_at_date
                        else:
                            print(f"Warning: price_at_date is None for asset {asset} on {current_date}")
                            missing_prices = True
                    else:
                        continue  # Ignore fiat currencies (they are not considered for exceeding the threshold)
                # Check if the value exceeds the threshold
                if current_value >= threshold:
                    consecutive_working_days += 1
                    days_above_threshold.append((current_date, current_value))  # Save the date above the threshold
                else:
                    consecutive_working_days = 0
                    days_above_threshold = []  # Reset the list if the threshold is not exceeded

                # If we exceed 7 consecutive working days, save the result and stop
                if consecutive_working_days >= 7:
                    exceeded_threshold = True
                    break

            # Skip to the next day
            current_datetime += timedelta(days=1)
            current_date = current_datetime.date()

        # Save the report in yearly holdings report directly
        threshold_report = YearlyThresholdReport(
            exceeded_threshold=exceeded_threshold,
            days_above_threshold=days_above_threshold,
            consecutive_working_days=consecutive_working_days,
            missing_prices=missing_prices
        )

        if tax_year not in self.yearly_holdings_report:
            self.yearly_holdings_report[tax_year] = {}

        self.yearly_holdings_report[tax_year]['threshold_report'] = threshold_report

        print(f"Threshold report calculated for the year {tax_year}.")
    
    def calculate_daily_holdings_and_average(self, value_asset: ValueAsset, tax_year: int) -> Dict[str, Decimal]:
        """
        Calcola il saldo giornaliero di tutte le criptovalute in BTC e usa il prezzo storico di BTC in EUR per calcolare il valore in EUR.
        Salva i risultati nel report giornaliero e calcola la giacenza media.

        :param value_asset: Il servizio per ottenere i prezzi storici.
        :param tax_year: L'anno fiscale per il quale calcolare il saldo giornaliero e la giacenza media.
        :return: Un dizionario con la giacenza media in BTC e EUR.
        """
        if config.debug:
            print(f"{Fore.CYAN}Calcolo saldi giornalieri e giacenza media per l'anno fiscale {tax_year}")

        current_year = datetime.now().year
        current_date = datetime.now().date() - timedelta(days=1)

        # Recupera le date di inizio e fine dell'anno fiscale
        start_date = self._start_of_year(tax_year)

        if tax_year == current_year:
            end_date = current_date # Se è l'anno corrente, usa la data di ieri
        else:
            end_date = self._end_of_year(tax_year)  # Altrimenti, usa la fine dell'anno fiscale

        current_datetime = datetime.combine(start_date, time.min).replace(tzinfo=timezone.utc)
        current_report_date = current_datetime.date()
        end_datetime = datetime.combine(end_date, time.min).replace(tzinfo=timezone.utc)

        total_btc = Decimal(0)
        total_eur = Decimal(0)
        total_btc_crypto_only = Decimal(0)
        total_eur_crypto_only = Decimal(0)
        days_count = (end_date - start_date).days + 1

        if config.debug:
            print(f"Start date: {start_date}, End date: {end_date}, Days count: {days_count}")

        if tax_year not in self.daily_holdings_report:
            self.daily_holdings_report[tax_year] = {}

        while current_datetime <= end_datetime:
            daily_btc_total = Decimal(0)
            daily_btc_crypto_only = Decimal(0)

            if config.debug:
                print(f"Processing date: {current_report_date}")

            for asset_symbol, holdings in self.holdings.items():
                quantity = holdings.get_balance_at_date(current_report_date)
                if quantity > 0:
                    if config.debug:
                        print(f"Asset: {asset_symbol}, Quantity: {quantity}")

                    if asset_symbol == 'BTC':
                        btc_value = quantity
                        btc_crypto_only_value = btc_value
                    elif asset_symbol == 'EUR':
                        btc_to_eur_price, _, _ = value_asset.get_historical_price('BTC', current_datetime)
                        if btc_to_eur_price is None:
                            btc_to_eur_price = Decimal(0)
                            if config.debug:
                                print(f"Missing BTC to EUR price for date: {current_report_date}")
                            self.daily_holdings_report[tax_year][current_report_date] = {
                                'btc_balance': daily_btc_total,
                                'missing_price': True
                            }
                        else:
                            btc_value = quantity / btc_to_eur_price
                            btc_crypto_only_value = 0
                            if config.debug:
                                print(f"EUR converted to BTC: {btc_value} at price {btc_to_eur_price}")
                    else:
                        asset_to_btc_price, _, _ = value_asset.get_historical_price(asset_symbol, current_datetime, return_in_btc=True)
                        if config.debug:
                            print(f"Price for {asset_symbol} on {current_report_date}: {asset_to_btc_price}")
                        if asset_to_btc_price is None:
                            asset_to_btc_price = Decimal(0)
                            if config.debug:
                                print(f"Missing price for {asset_symbol} on {current_report_date}")
                            self.daily_holdings_report[tax_year][current_report_date] = {
                                'btc_balance': daily_btc_total,
                                'missing_price': True
                            }
                        else:
                            btc_value = quantity * asset_to_btc_price
                            btc_crypto_only_value = btc_value
                            if config.debug:
                                print(f"{asset_symbol} quantity: {quantity}, price in BTC: {asset_to_btc_price}")
                                print(f"{asset_symbol} converted to BTC: {btc_value} at price {asset_to_btc_price}")

                    daily_btc_total += btc_value
                    daily_btc_crypto_only += btc_crypto_only_value

            btc_to_eur_price, _, _ = value_asset.get_historical_price('BTC', current_datetime)
            if btc_to_eur_price is None:
                btc_to_eur_price = Decimal(0)
                if config.debug:
                    print(f"Missing BTC to EUR price on {current_report_date}")

            daily_eur_total = daily_btc_total * btc_to_eur_price
            daily_eur_crypto_only = daily_btc_crypto_only * (btc_to_eur_price or Decimal(0))

            if config.debug:
                print(f"Daily BTC total: {daily_btc_total}, Daily EUR total: {daily_eur_total}")

            self.daily_holdings_report[tax_year][current_report_date] = DailyReportRecord(
                btc_balance=daily_btc_total,
                eur_value=daily_eur_total,
                btc_balance_only_crypto=daily_btc_crypto_only,
                eur_value_only_crypto=daily_eur_crypto_only
            )

            total_btc += daily_btc_total
            total_eur += daily_eur_total
            total_btc_crypto_only += daily_btc_crypto_only
            total_eur_crypto_only += daily_eur_crypto_only

            current_datetime += timedelta(days=1)
            current_report_date = current_datetime.date()

        average_btc = total_btc / days_count
        average_eur = total_eur / days_count
        average_btc_crypto_only = total_btc_crypto_only / days_count
        average_eur_crypto_only = total_eur_crypto_only / days_count

        self.daily_holdings_report[tax_year]['average'] = {
            'average_btc_balance': average_btc,
            'average_eur_value': average_eur,
            'average_btc_balance_only_crypto': average_btc_crypto_only,
            'average_eur_value_only_crypto': average_eur_crypto_only
        }

        if config.debug:
            print(f"Average BTC: {average_btc}, Average EUR: {average_eur}")
            print(f"Average BTC (Crypto Only): {average_btc_crypto_only}, Average EUR (Crypto Only): {average_eur_crypto_only}")

        print(f"Giacenza media calcolata per l'anno fiscale {tax_year}: {average_btc} BTC, {average_eur} EUR")


class CalculateCapitalGains:
    # Rate changes start from 6th April in previous year, i.e. 2022 is for tax year 2021/22
    CG_DATA_INDIVIDUAL: Dict[Year, CapitalGainsIndividual] = {
        Year(2009): {
            "allowance": Decimal(9600),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(18),
        },
        Year(2010): {
            "allowance": Decimal(10100),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(18),
        },
        Year(2011): {
            "allowance": Decimal(10100),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(28),
        },
        Year(2012): {
            "allowance": Decimal(10600),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(28),
        },
        Year(2013): {
            "allowance": Decimal(10600),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(28),
        },
        Year(2014): {
            "allowance": Decimal(10900),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(28),
        },
        Year(2015): {
            "allowance": Decimal(11000),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(28),
        },
        Year(2016): {
            "allowance": Decimal(11100),
            "basic_rate": Decimal(18),
            "higher_rate": Decimal(28),
        },
        Year(2017): {
            "allowance": Decimal(11100),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
        },
        Year(2018): {
            "allowance": Decimal(11300),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
        },
        Year(2019): {
            "allowance": Decimal(11700),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
        },
        Year(2020): {
            "allowance": Decimal(12000),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
        },
        Year(2021): {
            "allowance": Decimal(12300),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
        },
        Year(2022): {
            "allowance": Decimal(12300),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
        },
        Year(2023): {
            "allowance": Decimal(12300),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
        },
        Year(2024): {
            "allowance": Decimal(6000),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
            "proceeds_limit": Decimal(50000),
        },
        Year(2025): {
            "allowance": Decimal(3000),
            "basic_rate": Decimal(10),
            "higher_rate": Decimal(20),
            "proceeds_limit": Decimal(50000),
        },
    }

    # Rate changes start from 1st April
    CG_DATA_COMPANY: Dict[Year, ChargableGainsCompany] = {
        Year(2009): {"small_rate": Decimal(21), "main_rate": Decimal(28)},
        Year(2010): {"small_rate": Decimal(21), "main_rate": Decimal(28)},
        Year(2011): {"small_rate": Decimal(20), "main_rate": Decimal(23)},
        Year(2012): {"small_rate": Decimal(20), "main_rate": Decimal(24)},
        Year(2013): {"small_rate": Decimal(20), "main_rate": Decimal(23)},
        Year(2014): {"small_rate": Decimal(20), "main_rate": Decimal(21)},
        Year(2015): {"small_rate": None, "main_rate": Decimal(20)},
        Year(2016): {"small_rate": None, "main_rate": Decimal(20)},
        Year(2017): {"small_rate": None, "main_rate": Decimal(19)},
        Year(2018): {"small_rate": None, "main_rate": Decimal(19)},
        Year(2019): {"small_rate": None, "main_rate": Decimal(19)},
        Year(2020): {"small_rate": None, "main_rate": Decimal(19)},
        Year(2021): {"small_rate": None, "main_rate": Decimal(19)},
        Year(2022): {"small_rate": None, "main_rate": Decimal(19)},
        Year(2023): {"small_rate": Decimal(19), "main_rate": Decimal(25)},
        Year(2024): {"small_rate": Decimal(19), "main_rate": Decimal(25)},
    }

    def __init__(self, tax_year: Year) -> None:
        self.totals: CapitalGainsReportTotal = {
            "cost": Decimal(0),
            "fees": Decimal(0),
            "proceeds": Decimal(0),
            "gain": Decimal(0),
        }
        self.summary: CapitalGainsReportSummary = {
            "disposals": Decimal(0),
            "total_gain": Decimal(0),
            "total_loss": Decimal(0),
        }

        self.cgt_estimate: CapitalGainsReportEstimate = {
            "allowance": Decimal(self.CG_DATA_INDIVIDUAL[tax_year]["allowance"]),
            "cgt_basic_rate": self.CG_DATA_INDIVIDUAL[tax_year]["basic_rate"],
            "cgt_higher_rate": self.CG_DATA_INDIVIDUAL[tax_year]["higher_rate"],
            "allowance_used": Decimal(0),
            "taxable_gain": Decimal(0),
            "cgt_basic": Decimal(0),
            "cgt_higher": Decimal(0),
            "proceeds_limit": self.get_proceeds_limit(tax_year),
            "proceeds_warning": False,
        }

        self.ct_estimate: ChargableGainsReportEstimate = {
            "ct_small_rates": [],
            "ct_main_rates": [],
            "taxable_gain": Decimal(0),
            "ct_small": Decimal(0),
            "ct_main": Decimal(0),
        }

        self.short_term: Dict[AssetSymbol, List[TaxEventCapitalGains]] = {}
        self.short_term_totals: CapitalGainsReportTotal = {
            "cost": Decimal(0),
            "fees": Decimal(0),
            "proceeds": Decimal(0),
            "gain": Decimal(0),
        }
        self.long_term: Dict[AssetSymbol, List[TaxEventCapitalGains]] = {}
        self.long_term_totals: CapitalGainsReportTotal = {
            "cost": Decimal(0),
            "fees": Decimal(0),
            "proceeds": Decimal(0),
            "gain": Decimal(0),
        }
        self.non_tax_by_type: Dict[str, List[TaxEventNoGainNoLoss]] = {}
        self.non_tax_by_type_total: Dict[str, CapitalGainsReportTotal] = {}
        self.total_proceeds = Decimal(0)
        self.total_cost = Decimal(0)
        self.total_gain = Decimal(0)

    def get_proceeds_limit(self, tax_year: Year) -> Decimal:
        if "proceeds_limit" in self.CG_DATA_INDIVIDUAL[tax_year]:
            # For 2023 HMRC has introduced a fixed CGT reporting proceeds limit
            return Decimal(self.CG_DATA_INDIVIDUAL[tax_year]["proceeds_limit"])
        return Decimal(self.CG_DATA_INDIVIDUAL[tax_year]["allowance"]) * 4

    def get_ct_rate(self, date: Date) -> Tuple[Optional[Decimal], Decimal]:
        if date < datetime.date(date.year, 4, 1):
            # Use rate from previous year
            year = Year(date.year - 1)
            return (
                self.CG_DATA_COMPANY[year]["small_rate"],
                self.CG_DATA_COMPANY[year]["main_rate"],
            )
        year = Year(date.year)
        return (
            self.CG_DATA_COMPANY[year]["small_rate"],
            self.CG_DATA_COMPANY[year]["main_rate"],
        )

    def tax_summary(self, te: TaxEventCapitalGains) -> None:
        if te.disposal_type is DisposalType.SHORT_TERM:
            if te.asset not in self.short_term:
                self.short_term[te.asset] = []

            self.short_term[te.asset].append(te)

            self.short_term_totals["cost"] += te.cost
            self.short_term_totals["fees"] += te.fees
            self.short_term_totals["proceeds"] += te.proceeds
            self.short_term_totals["gain"] += te.gain
        elif te.disposal_type is DisposalType.LONG_TERM:
            if te.asset not in self.long_term:
                self.long_term[te.asset] = []

            self.long_term[te.asset].append(te)

            self.long_term_totals["cost"] += te.cost
            self.long_term_totals["fees"] += te.fees
            self.long_term_totals["proceeds"] += te.proceeds
            self.long_term_totals["gain"] += te.gain
        else:
            raise RuntimeError("Unexpected disposal_type")

        self.total_proceeds = self.short_term_totals["proceeds"] + self.long_term_totals["proceeds"]
        self.total_cost = self.short_term_totals["cost"] + self.long_term_totals["cost"]
        self.total_gain = self.short_term_totals["gain"] + self.long_term_totals["gain"]
        if config.debug:
            print (f"Total Proceeds: {self.total_proceeds}, Total Cost: {self.total_cost}, Total Gain: {self.total_gain}")

    def non_tax_summary(self, te: TaxEventNoGainNoLoss) -> None:
        if te.t_type.value not in self.non_tax_by_type:
            self.non_tax_by_type[te.t_type.value] = []

        self.non_tax_by_type[te.t_type.value].append(te)

        if te.t_type.value not in self.non_tax_by_type_total:
            self.non_tax_by_type_total[te.t_type.value] = {
                "cost": te.cost,
                "fees": te.fees,
                "proceeds": te.market_value,
                "gain": Decimal(),
            }
        else:
            self.non_tax_by_type_total[te.t_type.value]["cost"] += te.cost
            self.non_tax_by_type_total[te.t_type.value]["fees"] += te.fees
            self.non_tax_by_type_total[te.t_type.value]["proceeds"] += te.market_value

    def tax_estimate_cgt(self, tax_year: Year) -> None:
        if self.totals["gain"] > self.cgt_estimate["allowance"]:
            self.cgt_estimate["allowance_used"] = self.cgt_estimate["allowance"]
            self.cgt_estimate["taxable_gain"] = self.totals["gain"] - self.cgt_estimate["allowance"]
            self.cgt_estimate["cgt_basic"] = (
                self.cgt_estimate["taxable_gain"]
                * self.CG_DATA_INDIVIDUAL[tax_year]["basic_rate"]
                / 100
            )
            self.cgt_estimate["cgt_higher"] = (
                self.cgt_estimate["taxable_gain"]
                * self.CG_DATA_INDIVIDUAL[tax_year]["higher_rate"]
                / 100
            )
        elif self.totals["gain"] > 0:
            self.cgt_estimate["allowance_used"] = self.totals["gain"]

        if self.totals["proceeds"] >= self.cgt_estimate["proceeds_limit"]:
            self.cgt_estimate["proceeds_warning"] = True

    def tax_estimate_ct(self, tax_year: Year) -> None:
        if self.totals["gain"] > 0:
            self.ct_estimate["taxable_gain"] = self.totals["gain"]

        start_date = config.get_tax_year_start(tax_year)
        end_date = config.get_tax_year_end(tax_year)
        day_count = (end_date - start_date).days + 1

        for date in (start_date + datetime.timedelta(n) for n in range(day_count)):
            small_rate, main_rate = self.get_ct_rate(Date(date))

            if small_rate not in self.ct_estimate["ct_small_rates"]:
                self.ct_estimate["ct_small_rates"].append(small_rate)

            if main_rate not in self.ct_estimate["ct_main_rates"]:
                self.ct_estimate["ct_main_rates"].append(main_rate)

            if self.ct_estimate["taxable_gain"] > 0:
                if small_rate is None:
                    # Use main rate if there isn't a small rate
                    self.ct_estimate["ct_small"] += (
                        self.ct_estimate["taxable_gain"] / day_count * main_rate / 100
                    )
                else:
                    self.ct_estimate["ct_small"] += (
                        self.ct_estimate["taxable_gain"] / day_count * small_rate / 100
                    )

                self.ct_estimate["ct_main"] += (
                    self.ct_estimate["taxable_gain"] / day_count * main_rate / 100
                )

        if self.ct_estimate["ct_small_rates"] == [None]:
            # No small rate so remove estimate
            self.ct_estimate.pop("ct_small")
            self.ct_estimate["ct_small_rates"] = []

class CalculateIncome:
    def __init__(self) -> None:
        self.totals: IncomeReportTotal = {"amount": Decimal(0), "fees": Decimal(0)}
        self.assets: Dict[AssetSymbol, List[TaxEventIncome]] = {}
        self.types: Dict[str, List[TaxEventIncome]] = {}
        self.type_totals: Dict[str, IncomeReportTotal] = {}

    def totalise(self, te: TaxEventIncome) -> None:
        self.totals["amount"] += te.amount
        self.totals["fees"] += te.fees

        if te.asset not in self.assets:
            self.assets[te.asset] = []

        self.assets[te.asset].append(te)

        if te.type.value not in self.types:
            self.types[te.type.value] = []

        self.types[te.type.value].append(te)

    def totals_by_type(self) -> None:
        for income_type, te_list in self.types.items():
            for te in te_list:
                if income_type not in self.type_totals:
                    self.type_totals[income_type] = {"amount": te.amount, "fees": te.fees}
                else:
                    self.type_totals[income_type]["amount"] += te.amount
                    self.type_totals[income_type]["fees"] += te.fees


class CalculateMarginTrading:
    def __init__(self) -> None:
        self.totals: MarginReportTotal = {
            "gains": Decimal(0),
            "losses": Decimal(0),
            "fees": Decimal(0),
            "net_total": Decimal(0) 
        }
        self.contracts: Dict[Tuple[Wallet, Note], List[TaxEventMarginTrade]] = {}
        self.contract_totals: Dict[Tuple[Wallet, Note], MarginReportTotal] = {}

    def totalise(self, te: TaxEventMarginTrade) -> None:
        self.totals["gains"] += te.gain
        self.totals["losses"] += te.loss
        self.totals["fees"] += te.fee
        self.totals["net_total"] += te.gain - te.loss

        if (te.wallet, te.note) not in self.contracts:
            self.contracts[(te.wallet, te.note)] = []

        self.contracts[(te.wallet, te.note)].append(te)

    def totals_by_contract(self) -> None:
        for (wallet, note), te_list in self.contracts.items():
            for te in te_list:
                if (wallet, note) not in self.contract_totals:
                    self.contract_totals[(wallet, note)] = {
                        "gains": te.gain,
                        "losses": te.loss,
                        "fees": te.fee,
                        "net_total": te.gain - te.loss
                    }
                else:
                    self.contract_totals[(wallet, note)]["gains"] += te.gain
                    self.contract_totals[(wallet, note)]["losses"] += te.loss
                    self.contract_totals[(wallet, note)]["fees"] += te.fee
                    self.contract_totals[(wallet, note)]["net_total"] += te.gain - te.loss

                if config.debug:
                    print(f"{Fore.GREEN}margin: {te.t}")
                    print(f"{Fore.YELLOW}margin:   {self.totals_str(wallet, note)}")

    def totals_str(self, wallet: Wallet, note: Note) -> str:
        return (
            f'{wallet} {note}: gains={config.sym()}{self.contract_totals[(wallet, note)]["gains"]} '
            f'losses={config.sym()}{self.contract_totals[(wallet, note)]["losses"]} '
            f'fess={config.sym()}{self.contract_totals[(wallet, note)]["fees"]} '
            f'net_total={config.sym()}{self.totals["net_total"]} '
    )