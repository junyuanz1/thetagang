from dataclasses import dataclass
from typing import List, Dict

from ib_insync import IB, AccountValue, Position


@dataclass
class AccountPortfolio:
    account_number: str
    base_currency: str
    total: float
    available: float
    # currency -> Position type
    positions: Dict[str, Position]


class Portfolio:
    _account_numbers: List[str]
    _ib: IB
    _account_summary: Dict[str, List[AccountValue]]
    _account_position: Dict[str, List[Position]]

    def __init__(self, _ib: IB, account_numbers: List[str]):
        self._account_numbers = account_numbers
        self._ib = _ib
        self._account_summary = {}

    def pull_account_summary(self):
        for account_number in self._account_numbers:
            individual_account_summary = self._ib.accountSummary(account=account_number)
            self._account_summary[account_number] = individual_account_summary

    def pull_positions(self):
        for account_number in self._account_numbers:
            positions = self._ib.positions(account=account_number)
            self._account_position[account_number] = positions


    def build_account_portfolio(self):
        for account_number in self._account_numbers:
            account_summary = self._account_summary[account_number]
            account_position = self._account_position[account_number]
