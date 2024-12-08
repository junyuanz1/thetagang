from unittest.mock import MagicMock, patch

from ib_async import Option, Ticker

from thetagang.options import _nearest_strikes, _price_is_valid
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
