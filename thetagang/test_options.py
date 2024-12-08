import math
from typing import Optional, cast
from unittest.mock import MagicMock, Mock, patch

from ib_async import Option, Ticker

from thetagang.options import (
    _delta_is_valid,
    _nearest_strikes,
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
