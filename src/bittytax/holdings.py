# -*- coding: utf-8 -*-
# (c) Nano Nano Ltd 2019

from decimal import Decimal

from datetime import datetime, timedelta
from colorama import Fore
from tqdm import tqdm

from .bt_types import AssetSymbol
from .config import config
from .constants import WARNING



class Holdings:
    def __init__(self, asset: AssetSymbol) -> None:
        self.asset = asset
        self.quantity = Decimal(0)
        self.cost = Decimal(0)
        self.fees = Decimal(0)
        self.withdrawals = 0
        self.deposits = 0
        self.mismatches = 0
        self.balance_history = [(None, Decimal(0))]  # (timestamp, saldo)

    def _update_balance_history(self, new_quantity: Decimal, transaction_date: datetime):
        if self.balance_history[-1][0] is None:
            self.balance_history[-1] = (transaction_date.date(), new_quantity)
        else:
            last_quantity = self.balance_history[-1][1]
            if new_quantity != last_quantity:
                self.balance_history.append((transaction_date.date(), new_quantity))

    def add_tokens(self, quantity: Decimal, cost: Decimal, fees: Decimal, is_deposit: bool, transaction_date: datetime) -> None:
        self.quantity += quantity
        self.cost += cost
        self.fees += fees
        self._update_balance_history(self.quantity, transaction_date)

        if is_deposit:
            self.deposits += 1

        if config.debug:
            print(
                f"{Fore.YELLOW}holdings:   "
                f"{self.asset}={self.quantity.normalize():0,f} (+{quantity.normalize():0,f}) "
                f"cost={config.sym()}{self.cost:0,.2f} {config.ccy} "
                f"(+{config.sym()}{cost:0,.2f} {config.ccy}) "
                f"fees={config.sym()}{self.fees:0,.2f} {config.ccy} "
                f"(+{config.sym()}{fees:0,.2f} {config.ccy})"
            )

    def subtract_tokens(
        self, quantity: Decimal, cost: Decimal, fees: Decimal, is_withdrawal: bool, transaction_date: datetime
    ) -> None:
        self.quantity -= quantity
        self.cost -= cost
        self.fees -= fees
        self._update_balance_history(self.quantity, transaction_date)

        if is_withdrawal:
            self.withdrawals += 1

        if config.debug:
            print(
                f"{Fore.YELLOW}holdings:   "
                f"{self.asset}={self.quantity.normalize():0,f} (-{quantity.normalize():0,f}) "
                f"cost={config.sym()}{self.cost:0,.2f} {config.ccy} "
                f"(-{config.sym()}{cost:0,.2f} {config.ccy}) "
                f"fees={config.sym()}{self.fees:0,.2f} {config.ccy} "
                f"(-{config.sym()}{fees:0,.2f} {config.ccy})"
            )

    def check_transfer_mismatch(self) -> None:
        if self.withdrawals > 0 and self.withdrawals != self.deposits:
            tqdm.write(
                f"{WARNING} Disposal detected between a Withdrawal and a Deposit "
                f"({self.withdrawals}:{self.deposits}) for {self.asset}, cost basis will be wrong"
            )
            self.mismatches += 1

    def get_balance_at_date(self, target_date: datetime.date) -> Decimal:
        # Returns the balance as of the most recent date before or equal to `target_date`
        for date, balance in reversed(self.balance_history):
            if date <= target_date:
                return balance
        return Decimal(0)  # If there are no previous dates

    def calculate_average_balance(self, start_date: datetime.date, end_date: datetime.date) -> Decimal:
        current_date = start_date
        total_balance = Decimal(0)
        day_count = 0

        while current_date <= end_date:
            total_balance += self.get_balance_at_date(current_date)
            current_date += timedelta(days=1)
            day_count += 1

        # Average annual asset balance
        if day_count > 0:
            return total_balance / Decimal(day_count)
        return Decimal(0)