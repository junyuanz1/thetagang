import math
from typing import Optional, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from ib_async import Contract, Option, Ticker

from thetagang.options import (
    _delta_is_valid,
    _nearest_strikes,
    _open_interest_is_valid_sort_by_delta,
    _price_is_valid,
    _select_ticker,
    _valid_strike,
)
from thetagang.types import OptionRight


@patch("thetagang.options.midpoint_or_market_price", return_value=1.0)
def test_price_is_valid_call(_: MagicMock) -> None:
    ticker = Ticker(
        contract=Option(strike=100),
    )
    assert _price_is_valid(OptionRight.CALL, ticker, 100, 0.05) is True


@patch("thetagang.options.midpoint_or_market_price", return_value=1.0)
def test_price_is_valid_put_valid_price(_: MagicMock) -> None:
    ticker = Ticker(
        contract=Option(strike=90),
    )
    assert _price_is_valid(OptionRight.PUT, ticker, 100, 0.05) is True


@patch("thetagang.options.midpoint_or_market_price", return_value=0.1)
def test_price_is_valid_put_invalid_price(_: MagicMock) -> None:
    ticker = Ticker(
        contract=Option(strike=110),
    )
    assert _price_is_valid(OptionRight.PUT, ticker, 100, 0.05) is False


@patch("thetagang.options.midpoint_or_market_price", return_value=0.01)
def test_price_is_valid_below_minimum_price(_: MagicMock) -> None:
    ticker = Ticker(
        contract=Option(strike=90),
    )
    assert _price_is_valid(OptionRight.CALL, ticker, 100, 0.05) is False


@patch("thetagang.options.midpoint_or_market_price", return_value=0.1)
def test_price_is_valid_put_exceeds_market_price(_: MagicMock) -> None:
    ticker = Ticker(
        contract=Option(strike=110),
    )
    assert _price_is_valid(OptionRight.PUT, ticker, 100, 0.05) is False


def test_nearest_strikes_put() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    strikes_per_chain = 3
    result = _nearest_strikes(OptionRight.PUT, strikes_per_chain, strikes)
    assert result == [100, 105, 110]


def test_nearest_strikes_call() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    strikes_per_chain = 3
    result = _nearest_strikes(OptionRight.CALL, strikes_per_chain, strikes)
    assert result == [90, 95, 100]


def test_nearest_strikes_put_less_than_chain() -> None:
    strikes = [90.0, 95.0]
    strikes_per_chain = 3
    result = _nearest_strikes(OptionRight.PUT, strikes_per_chain, strikes)
    assert result == [90, 95]


def test_nearest_strikes_call_less_than_chain() -> None:
    strikes = [90.0, 95.0]
    strikes_per_chain = 3
    result = _nearest_strikes(OptionRight.CALL, strikes_per_chain, strikes)
    assert result == [90, 95]


def test_valid_strike_put_within_limit() -> None:
    assert _valid_strike(OptionRight.PUT, 90, 100, 95) is True


def test_valid_strike_put_exceeds_limit() -> None:
    assert _valid_strike(OptionRight.PUT, 100, 100, 95) is False


def test_valid_strike_put_within_price_range() -> None:
    assert _valid_strike(OptionRight.PUT, 105, 100, None) is True


def test_valid_strike_put_exceeds_price_range() -> None:
    assert _valid_strike(OptionRight.PUT, 110, 100, None) is False


def test_valid_strike_call_within_limit() -> None:
    assert _valid_strike(OptionRight.CALL, 110, 100, 105) is True


def test_valid_strike_call_below_limit() -> None:
    assert _valid_strike(OptionRight.CALL, 100, 100, 105) is False


def test_valid_strike_call_within_price_range() -> None:
    assert _valid_strike(OptionRight.CALL, 95, 100, None) is True


def test_valid_strike_call_below_price_range() -> None:
    assert _valid_strike(OptionRight.CALL, 90, 100, None) is False


def test_delta_is_valid() -> None:
    # Create mocked objects that will be cast to Ticker type
    def create_mock_ticker(delta: Optional[float] = None) -> Ticker:
        ticker = Mock()
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.delta = delta
        return cast(Ticker, ticker)  # Cast Mock to Ticker type for type checking

    # Test cases
    test_cases = [
        # Valid cases
        (create_mock_ticker(delta=0.3), 0.5, True),  # Valid delta within target
        (create_mock_ticker(delta=-0.2), 0.5, True),  # Valid negative delta
        (create_mock_ticker(delta=0.5), 0.5, True),  # Delta equals target
        # Invalid cases
        (cast(Ticker, Mock(modelGreeks=None)), 0.5, False),  # No modelGreeks
        (create_mock_ticker(delta=None), 0.5, False),  # Delta is None
        (create_mock_ticker(delta=math.nan), 0.5, False),  # Delta is NaN
        (create_mock_ticker(delta=0.6), 0.5, False),  # Delta exceeds target
        (create_mock_ticker(delta=-0.7), 0.5, False),  # Negative delta exceeds target
    ]

    # Run all test cases
    for ticker, target_delta, expected_result in test_cases:
        result = _delta_is_valid(ticker, target_delta)
        assert result == expected_result, (
            f"Failed for delta={ticker.modelGreeks.delta if hasattr(ticker, 'modelGreeks') and ticker.modelGreeks else None}, "
            f"target={target_delta}, expected={expected_result}, got={result}"
        )


def create_mock_ticker(
    delta: Optional[float] = 0.3,
    put_oi: int = 100,
    call_oi: int = 100,
    last_trade_date: Optional[str] = "20241231",
    has_model_greeks: bool = True,
    has_contract: bool = True,
) -> Ticker:
    ticker = Mock()

    # Set up modelGreeks (can be None)
    if has_model_greeks:
        ticker.modelGreeks = Mock()
        ticker.modelGreeks.delta = delta
    else:
        ticker.modelGreeks = None

    # Set up open interest
    ticker.putOpenInterest = put_oi
    ticker.callOpenInterest = call_oi

    # Set up contract (can be None)
    if has_contract:
        ticker.contract = Mock()
        ticker.contract.lastTradeDateOrContractMonth = last_trade_date
    else:
        ticker.contract = None

    return cast(Ticker, ticker)


class TestOptionFiltering:
    @pytest.fixture
    def mock_underlying(self) -> Mock:
        contract = Mock(spec=Contract)
        contract.symbol = "AAPL"
        return contract

    def test_none_contract(self, mock_underlying: Mock) -> None:
        ticker1 = create_mock_ticker(delta=0.3, has_contract=True)
        ticker2 = create_mock_ticker(delta=0.5, has_contract=False)  # None contract
        ticker3 = create_mock_ticker(delta=0.4, has_contract=True)

        tickers = [ticker1, ticker2, ticker3]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        # Verify correct order: None contract with highest delta first, then by delta
        assert result == [ticker2, ticker3, ticker1]

    def test_mixed_none_values(self, mock_underlying: Mock) -> None:
        ticker1 = create_mock_ticker(
            delta=0.3, has_contract=True, has_model_greeks=True
        )
        ticker2 = create_mock_ticker(
            delta=None, has_contract=False, has_model_greeks=False
        )
        ticker3 = create_mock_ticker(
            delta=0.5, has_contract=True, has_model_greeks=True
        )
        ticker4 = create_mock_ticker(
            delta=0.5, has_contract=False, has_model_greeks=True
        )

        tickers = [ticker1, ticker2, ticker3, ticker4]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        assert result == [ticker4, ticker2, ticker3, ticker1]

    def test_none_contract_grouping(self, mock_underlying: Mock) -> None:
        ticker1 = create_mock_ticker(
            delta=0.5, has_contract=True, has_model_greeks=True
        )
        ticker2 = create_mock_ticker(
            delta=None, has_contract=False, has_model_greeks=False
        )
        ticker3 = create_mock_ticker(
            delta=0.5, has_contract=False, has_model_greeks=True
        )

        tickers = [ticker1, ticker2, ticker3]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        assert result == [ticker3, ticker2, ticker1]

    def test_all_none_values(self, mock_underlying: Mock) -> None:
        ticker1 = create_mock_ticker(
            delta=None, has_model_greeks=False, has_contract=False
        )
        ticker2 = create_mock_ticker(
            delta=None, has_model_greeks=False, has_contract=False
        )

        tickers = [ticker1, ticker2]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        assert result == [ticker1, ticker2]  # Should maintain original order

    def test_open_interest_filtering_puts(self, mock_underlying: Mock) -> None:
        ticker1 = create_mock_ticker(put_oi=50, has_contract=True)
        ticker2 = create_mock_ticker(put_oi=150, has_contract=True)
        ticker3 = create_mock_ticker(put_oi=100, has_contract=True)

        tickers = [ticker1, ticker2, ticker3]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.PUT,
            minimum_open_interest=100,
            delta_ord_desc=True,
        )

        # Only tickers with OI >= 100 should remain
        assert len(result) == 2
        assert ticker2 in result
        assert ticker3 in result
        assert ticker1 not in result

    def test_delta_sorting_descending(self, mock_underlying: Mock) -> None:
        ticker1 = create_mock_ticker(
            delta=0.2, has_contract=True, has_model_greeks=True
        )
        ticker2 = create_mock_ticker(
            delta=0.5, has_contract=True, has_model_greeks=True
        )
        ticker3 = create_mock_ticker(
            delta=None, has_contract=True, has_model_greeks=False
        )
        ticker4 = create_mock_ticker(
            delta=0.3, has_contract=True, has_model_greeks=True
        )
        ticker5 = create_mock_ticker(
            delta=None, has_contract=True, has_model_greeks=False
        )

        tickers = [ticker1, ticker2, ticker3, ticker4, ticker5]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        assert result == [ticker2, ticker4, ticker1, ticker3, ticker5]

    def test_delta_sorting_ascending(self, mock_underlying: Mock) -> None:
        ticker1 = create_mock_ticker(
            delta=0.2, has_contract=True, has_model_greeks=True
        )
        ticker2 = create_mock_ticker(
            delta=0.5, has_contract=True, has_model_greeks=True
        )
        ticker3 = create_mock_ticker(
            delta=None, has_contract=True, has_model_greeks=False
        )
        ticker4 = create_mock_ticker(
            delta=0.3, has_contract=True, has_model_greeks=True
        )
        ticker5 = create_mock_ticker(
            delta=None, has_contract=True, has_model_greeks=False
        )

        tickers = [ticker1, ticker2, ticker3, ticker4, ticker5]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=False,
        )

        assert result == [ticker3, ticker5, ticker1, ticker4, ticker2]


class TestSelectTicker:
    @pytest.fixture
    def mock_underlying(self) -> Mock:
        contract = Mock(spec=Contract)
        contract.symbol = "AAPL"
        return contract

    def create_ticker_with_price(
        self,
        delta: float,
        price: float,
        strike: float = 100.0,
        dte: str = "20241231",
        has_contract: bool = True,
    ) -> Ticker:
        ticker = create_mock_ticker(delta=delta, has_contract=has_contract)

        # Set up price mocking
        ticker.bid = price - 0.05
        ticker.ask = price + 0.05
        ticker.last = price

        # Set up contract details only if has_contract is True
        if has_contract:
            assert ticker.contract is not None  # Type assertion for mypy
            ticker.contract.strike = strike
            ticker.contract.lastTradeDateOrContractMonth = dte

        return ticker

    @patch("thetagang.options.midpoint_or_market_price")
    def test_normal_selection(self, mock_price_fn: Mock, mock_underlying: Mock) -> None:
        # Create tickers with different deltas and prices
        ticker1 = self.create_ticker_with_price(delta=0.3, price=2.5)
        ticker2 = self.create_ticker_with_price(delta=0.5, price=3.0)
        ticker3 = self.create_ticker_with_price(delta=0.4, price=2.8)

        # Set up price function mock to return actual prices
        mock_price_fn.side_effect = lambda t: {
            ticker1: 2.5,
            ticker2: 3.0,
            ticker3: 2.8,
        }[t]

        valid_tickers = [ticker1, ticker2, ticker3]
        delta_reject_tickers = []

        result = _select_ticker(
            underlying=mock_underlying,
            right=OptionRight.CALL,
            valid_tickers=iter(valid_tickers),
            delta_reject_tickers=iter(delta_reject_tickers),
            minimum_open_interest=0,
            minimum_price=0.0,
            fallback_minimum_price=None,
        )

        # Should select highest delta ticker
        assert result == ticker2

    @patch("thetagang.options.midpoint_or_market_price")
    def test_fallback_to_delta_reject(
        self, mock_price_fn: Mock, mock_underlying: Mock
    ) -> None:
        # Create valid tickers with low open interest
        ticker1 = create_mock_ticker(
            delta=0.3,
            has_contract=True,
            put_oi=500,  # Below minimum
            call_oi=500,  # Below minimum
        )

        # Create delta reject tickers with valid open interest
        ticker2 = create_mock_ticker(
            delta=0.6,
            has_contract=True,
            put_oi=1500,  # Above minimum
            call_oi=1500,  # Above minimum
        )
        ticker3 = create_mock_ticker(
            delta=0.7,
            has_contract=True,
            put_oi=2000,  # Above minimum
            call_oi=2000,  # Above minimum
        )

        # Set up price function mock
        mock_price_fn.side_effect = lambda t: {
            ticker1: 2.5,
            ticker2: 3.0,
            ticker3: 3.5,
        }[t]

        valid_tickers = [ticker1]
        delta_reject_tickers = [ticker2, ticker3]

        result = _select_ticker(
            underlying=mock_underlying,
            right=OptionRight.CALL,
            valid_tickers=iter(valid_tickers),
            delta_reject_tickers=iter(delta_reject_tickers),
            minimum_open_interest=1000,  # High threshold to force fallback
            minimum_price=1.0,
            fallback_minimum_price=None,
        )

        # Should select from delta_reject_tickers
        assert result == ticker3, "Should select the ticker with highest delta"
        assert result.modelGreeks is not None
        assert result.modelGreeks.delta == 0.7

    @patch("thetagang.options.midpoint_or_market_price")
    def test_fallback_minimum_price(
        self, mock_price_fn: Mock, mock_underlying: Mock
    ) -> None:
        ticker1 = self.create_ticker_with_price(delta=0.3, price=1.5)
        ticker2 = self.create_ticker_with_price(delta=0.5, price=2.5)
        ticker3 = self.create_ticker_with_price(delta=0.4, price=2.0)

        # Set up price function mock
        mock_price_fn.side_effect = lambda t: {
            ticker1: 1.5,
            ticker2: 2.5,
            ticker3: 2.0,
        }[t]

        valid_tickers = [ticker1, ticker2, ticker3]
        delta_reject_tickers = []

        result = _select_ticker(
            underlying=mock_underlying,
            right=OptionRight.CALL,
            valid_tickers=iter(valid_tickers),
            delta_reject_tickers=iter(delta_reject_tickers),
            minimum_open_interest=0,
            minimum_price=0.0,
            fallback_minimum_price=2.2,  # Only ticker2 meets this
        )

        assert result == ticker2

    @patch("thetagang.options.midpoint_or_market_price")
    def test_price_sorting_fallback(
        self, mock_price_fn: Mock, mock_underlying: Mock
    ) -> None:
        ticker1 = self.create_ticker_with_price(delta=0.3, price=3.0)
        ticker2 = self.create_ticker_with_price(delta=0.5, price=1.0)
        ticker3 = self.create_ticker_with_price(delta=0.4, price=2.0)

        # Set up price function mock
        mock_price_fn.side_effect = lambda t: {
            ticker1: 3.0,
            ticker2: 1.0,
            ticker3: 2.0,
        }[t]

        valid_tickers = [ticker1, ticker2, ticker3]
        delta_reject_tickers = []

        result = _select_ticker(
            underlying=mock_underlying,
            right=OptionRight.CALL,
            valid_tickers=iter(valid_tickers),
            delta_reject_tickers=iter(delta_reject_tickers),
            minimum_open_interest=0,
            minimum_price=0.0,
            fallback_minimum_price=5.0,  # Force price sorting as none meet this
        )

        assert result == ticker1
