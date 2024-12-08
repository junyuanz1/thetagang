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
        tickers = [
            create_mock_ticker(delta=0.3, has_contract=True),
            create_mock_ticker(delta=0.5, has_contract=False),  # None contract
            create_mock_ticker(delta=0.4, has_contract=True),
        ]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        # Verify that None contract are treated as DTE=0 in sorting
        result_pairs = [
            (
                t.modelGreeks.delta if t.modelGreeks else None,
                t.contract.lastTradeDateOrContractMonth if t.contract else None,
            )
            for t in result
        ]
        # Tickers should be sorted by delta first, then by date with None contract treated as DTE=0
        expected_pairs = [
            (0.5, None),  # Highest delta, None contract
            (0.4, "20241231"),
            (0.3, "20241231"),
        ]
        assert result_pairs == expected_pairs

    def test_mixed_none_values(self, mock_underlying: Mock) -> None:
        tickers = [
            create_mock_ticker(delta=0.3, has_contract=True, has_model_greeks=True),
            create_mock_ticker(delta=None, has_contract=False, has_model_greeks=False),
            create_mock_ticker(delta=0.5, has_contract=True, has_model_greeks=True),
            create_mock_ticker(delta=0.5, has_contract=False, has_model_greeks=True),
        ]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        # Verify sorting with mixed None values
        result_pairs = [
            (
                t.modelGreeks.delta if t.modelGreeks else None,
                True if t.contract else False,  # Just check contract existence
            )
            for t in result
        ]
        expected_pairs = [
            (0.5, False),  # Highest delta, no contract
            (None, False),  # None modelGreeks gets grouped with similar delta value
            (0.5, True),  # Same delta with contract
            (0.3, True),  # Lower delta with contract
        ]
        assert result_pairs == expected_pairs

    # Add a more focused test for None contract grouping
    def test_none_contract_grouping(self, mock_underlying: Mock) -> None:
        tickers = [
            create_mock_ticker(delta=0.5, has_contract=True, has_model_greeks=True),
            create_mock_ticker(delta=None, has_contract=False, has_model_greeks=False),
            create_mock_ticker(delta=0.5, has_contract=False, has_model_greeks=True),
        ]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        result_pairs = [
            (
                t.modelGreeks.delta if t.modelGreeks else None,
                True if t.contract else False,
            )
            for t in result
        ]
        expected_pairs = [
            (0.5, False),  # No contract
            (None, False),  # None modelGreeks grouped with no contract
            (0.5, True),  # With contract
        ]
        assert result_pairs == expected_pairs

    def test_all_none_values(self, mock_underlying: Mock) -> None:
        tickers = [
            create_mock_ticker(delta=None, has_model_greeks=False, has_contract=False),
            create_mock_ticker(delta=None, has_model_greeks=False, has_contract=False),
        ]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        # Should not raise any errors and should maintain order
        assert len(result) == 2
        assert all(t.modelGreeks is None for t in result)
        assert all(t.contract is None for t in result)

    # Previous test cases remain the same, just updating their create_mock_ticker calls
    def test_open_interest_filtering_puts(self, mock_underlying: Mock) -> None:
        tickers = [
            create_mock_ticker(put_oi=50, has_contract=True),
            create_mock_ticker(put_oi=150, has_contract=True),
            create_mock_ticker(put_oi=100, has_contract=True),
        ]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.PUT,
            minimum_open_interest=100,
            delta_ord_desc=True,
        )

        assert len(result) == 2
        assert all(t.putOpenInterest >= 100 for t in result)

    def test_delta_sorting_descending(self, mock_underlying: Mock) -> None:
        tickers = [
            create_mock_ticker(delta=0.2, has_contract=True, has_model_greeks=True),
            create_mock_ticker(delta=0.5, has_contract=True, has_model_greeks=True),
            create_mock_ticker(
                delta=None, has_contract=True, has_model_greeks=False
            ),  # None modelGreeks
            create_mock_ticker(delta=0.3, has_contract=True, has_model_greeks=True),
            create_mock_ticker(
                delta=None, has_contract=True, has_model_greeks=False
            ),  # Another None modelGreeks
        ]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=True,
        )

        # Extract deltas safely, handling None modelGreeks
        deltas = [t.modelGreeks.delta if t.modelGreeks else None for t in result]
        # None modelGreeks should be treated as delta=0 and sorted to the end
        assert deltas == [0.5, 0.3, 0.2, None, None]

    def test_delta_sorting_ascending(self, mock_underlying: Mock) -> None:
        tickers = [
            create_mock_ticker(delta=0.2, has_contract=True, has_model_greeks=True),
            create_mock_ticker(delta=0.5, has_contract=True, has_model_greeks=True),
            create_mock_ticker(
                delta=None, has_contract=True, has_model_greeks=False
            ),  # None modelGreeks
            create_mock_ticker(delta=0.3, has_contract=True, has_model_greeks=True),
            create_mock_ticker(
                delta=None, has_contract=True, has_model_greeks=False
            ),  # Another None modelGreeks
        ]

        result = _open_interest_is_valid_sort_by_delta(
            underlying=mock_underlying,
            tickers=tickers,
            right=OptionRight.CALL,
            minimum_open_interest=0,
            delta_ord_desc=False,
        )

        # Extract deltas safely, handling None modelGreeks
        deltas = [t.modelGreeks.delta if t.modelGreeks else None for t in result]
        # None modelGreeks should be treated as delta=0 and sorted to the beginning when ascending
        assert deltas == [None, None, 0.2, 0.3, 0.5]
