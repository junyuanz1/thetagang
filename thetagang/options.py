import math
from typing import Iterator, List, Optional

from ib_async import Contract, Option, Ticker, util
from more_itertools import partition

from thetagang import log
from thetagang.config import Config
from thetagang.fmt import dfmt
from thetagang.ibkr import IBKR, TickerField
from thetagang.types import OptionRight
from thetagang.util import (
    get_max_dte,
    get_target_delta,
    get_target_dte,
    midpoint_or_market_price,
    option_dte,
)


class NoValidContractsError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


async def find_eligible_contracts(
    ibkr: IBKR,
    config: Config,
    underlying: Contract,
    right: OptionRight,
    minimum_price: float,
    min_dte: int = 0,
    strike_limit: Optional[float] = None,
    fallback_minimum_price: Optional[float] = None,
    exclude_strick: Optional[float] = None,
    exclude_expiration: Optional[str] = None,
    contract_target_dte_override: Optional[int] = None,
    contract_target_delta_override: Optional[float] = None,
) -> Ticker:
    contract_target_dte: int = (
        contract_target_dte_override
        if contract_target_dte_override
        else get_target_dte(config.target, config.symbol_config(underlying.symbol))
    )
    contract_target_delta: float = (
        contract_target_delta_override
        if contract_target_delta_override
        else get_target_delta(
            config.target, config.symbol_config(underlying.symbol), right
        )
    )
    contract_max_dte = get_max_dte(
        config.target,
        config.symbol_config(underlying.symbol),
    )
    strikes_per_chain = config.option_chains.strikes
    expirations_to_display = config.option_chains.expirations
    minimum_open_interest = config.target.minimum_open_interest
    order_exchange = config.get_order_exchange()

    log.notice(
        f"{underlying.symbol}: Searching option chain for "
        f"right={right} strike_limit={strike_limit} minimum_price={dfmt(minimum_price,3)} "
        f"fallback_minimum_price={dfmt(fallback_minimum_price if fallback_minimum_price else 0,3)} "
        f"contract_target_dte={contract_target_dte} contract_max_dte={contract_max_dte} "
        f"contract_target_delta={contract_target_delta}, "
        "this can take a while...",
    )

    underlying_ticker = await ibkr.get_ticker_for_contract(underlying)

    underlying_price = midpoint_or_market_price(underlying_ticker)

    chains = await ibkr.get_chains_for_contract(underlying)

    chain = next(c for c in chains if c.exchange == underlying.exchange)

    strikes = sorted(
        strike
        for strike in chain.strikes
        if _valid_strike(right, strike, underlying_price, strike_limit)
    )

    expirations = sorted(
        exp
        for exp in chain.expirations
        if option_dte(exp) >= contract_target_dte
        and option_dte(exp) >= min_dte
        and (not contract_max_dte or option_dte(exp) <= contract_max_dte)
    )[:expirations_to_display]
    if len(expirations) < 1:
        raise NoValidContractsError(
            f"No valid contract expirations found for {underlying.symbol}. Continuing anyway...",
        )

    strikes = _nearest_strikes(right, strikes_per_chain, strikes)
    if len(strikes) < 1:
        raise NoValidContractsError(
            f"No valid contract strikes found for {underlying.symbol}. Continuing anyway...",
        )
    log.info(
        f"{underlying.symbol}: Scanning between strikes {strikes[0]} and {strikes[-1]},"
        f" from expirations {expirations[0]} to {expirations[-1]}"
    )

    contracts = [
        Option(
            underlying.symbol,
            expiration,
            strike,
            right.value,
            order_exchange,
        )
        for expiration in expirations
        for strike in strikes
    ]

    contracts = await ibkr.qualify_contracts(*contracts)

    log.info(f"{underlying.symbol}: Found {len(contracts)} contracts...")

    # exclude strike, but only for the first exp
    if exclude_strick is not None or exclude_expiration is not None:
        log.info(
            f"{underlying.symbol}: Excluding strike {exclude_strick} and expiration {exclude_expiration}..."
        )
        contracts = [
            c
            for c in contracts
            if (
                (
                    exclude_expiration is None
                    or c.lastTradeDateOrContractMonth != exclude_expiration
                )
                and (exclude_strick is None or c.strike != exclude_strick)
            )
        ]

    log.info(f"{underlying.symbol}: Processing {len(contracts)} contracts...")

    tickers = await ibkr.get_tickers_for_contracts(
        underlying.symbol,
        contracts,
        generic_tick_list="101",
        required_fields=[],
        optional_fields=[
            TickerField.MARKET_PRICE,
            TickerField.GREEKS,
            TickerField.OPEN_INTEREST,
            TickerField.MIDPOINT,
        ],
    )

    log.info(
        f"{underlying.symbol}: Filtering invalid prices for {len(tickers)} tickers..."
    )
    tickers = [
        ticker
        for ticker in tickers
        if _price_is_valid(right, ticker, underlying_price, minimum_price)
    ]

    log.info(
        f"{underlying.symbol}: Filtering invalid deltas for {len(tickers)} tickers..."
    )
    delta_reject_tickers, tickers = partition(
        lambda x: _delta_is_valid(x, contract_target_delta), tickers
    )

    return _select_ticker(
        underlying,
        right,
        tickers,
        delta_reject_tickers,
        minimum_open_interest,
        minimum_price,
        fallback_minimum_price,
    )


def _valid_strike(
    right: OptionRight,
    strike: float,
    underlying_price: float,
    strike_limit: Optional[float],
) -> bool:
    if right == OptionRight.PUT and strike_limit:
        return strike <= strike_limit
    elif right == OptionRight.PUT:
        return strike <= underlying_price + 0.05 * underlying_price
    elif right == OptionRight.CALL and strike_limit:
        return strike >= strike_limit
    elif right == OptionRight.CALL:
        return strike >= underlying_price - 0.05 * underlying_price


def _delta_is_valid(ticker: Ticker, target_delta: float) -> bool:
    return (
        ticker.modelGreeks is not None
        and ticker.modelGreeks
        and ticker.modelGreeks.delta is not None
        and not util.isNan(ticker.modelGreeks.delta)
        and abs(ticker.modelGreeks.delta) <= target_delta
    )


def _nearest_strikes(
    right: OptionRight, strikes_per_chain: int, strikes: List[float]
) -> List[float]:
    if right == OptionRight.PUT:
        return strikes[-strikes_per_chain:]
    else:
        return strikes[:strikes_per_chain]


def _price_is_valid(
    right: OptionRight, ticker: Ticker, underlying_price: float, minimum_price: float
) -> bool:
    def cost_doesnt_exceed_market_price(ticker: Ticker) -> bool:
        # when writing puts, we need to be sure that the strike +
        # credit is less than or equal to the current market price, so
        # that we don't exceed the target capital allocation for this
        # position
        return (
            right == OptionRight.CALL
            or isinstance(ticker.contract, Option)
            and ticker.contract.strike
            <= midpoint_or_market_price(ticker) + underlying_price
        )

    return midpoint_or_market_price(
        ticker
    ) > minimum_price and cost_doesnt_exceed_market_price(ticker)


def _open_interest_is_valid_sort_by_delta(
    underlying: Contract,
    tickers: List[Ticker],
    right: OptionRight,
    minimum_open_interest: int,
    delta_ord_desc: bool,
) -> List[Ticker]:
    def sort_tickers(tickers: List[Ticker]) -> List[Ticker]:
        # sort by delta first, then expiry date
        tickers = sorted(
            sorted(
                tickers,
                key=lambda t: (
                    abs(t.modelGreeks.delta)
                    if t.modelGreeks and t.modelGreeks.delta
                    else 0
                ),
                reverse=delta_ord_desc,
            ),
            key=lambda t: (
                option_dte(t.contract.lastTradeDateOrContractMonth) if t.contract else 0
            ),
        )
        return tickers

    def open_interest_is_valid(ticker: Ticker) -> bool:
        if minimum_open_interest > 0:
            # The open interest value is never present when using historical
            # data, so just ignore it when the value is None
            if right == OptionRight.PUT:
                return ticker.putOpenInterest >= minimum_open_interest
            if right == OptionRight.CALL:
                return ticker.callOpenInterest >= minimum_open_interest
        return True

    log.info(
        f"{underlying.symbol}: Filtering invalid open interest for {len(list(tickers))} tickers..."
    )
    tickers = [ticker for ticker in tickers if open_interest_is_valid(ticker)]
    log.info(
        f"{underlying.symbol}: Sorting {len(list(tickers))} tickers with desending delta ..."
    )
    tickers = sort_tickers(tickers)

    return tickers


def _select_ticker(
    underlying: Contract,
    right: OptionRight,
    valid_tickers: Iterator[Ticker],
    delta_reject_tickers: Iterator[Ticker],
    minimum_open_interest: int,
    minimum_price: float,
    fallback_minimum_price: Optional[float],
) -> Ticker:
    tickers = _open_interest_is_valid_sort_by_delta(
        underlying, list(valid_tickers), right, minimum_open_interest, True
    )

    the_chosen_ticker = None

    # some final processing to ensure we have a valid contract
    if len(tickers) == 0:
        if not math.isclose(minimum_price, 0.0):
            log.warning(
                f"{underlying.symbol}: No valid contracts found with valid open interest, falling back to search for contracts with higher delta..."
            )
            # if we arrive here, it means that 1) we expect to roll for a
            # credit only, but 2) we didn't find any suitable contracts,
            # most likely because we can't roll out and up/down to the
            # target delta
            #
            # because of this, we'll allow rolling to a less-than-optimal
            # strike, provided it's still a credit
            tickers = _open_interest_is_valid_sort_by_delta(
                underlying,
                list(delta_reject_tickers),
                right,
                minimum_open_interest,
                False,
            )
        if len(tickers) < 1:
            # if there are _still_ no tickers remaining, there's nothing
            # more we can do
            raise NoValidContractsError(
                f"No valid contracts found for {underlying.symbol}. Continuing anyway...",
            )
    elif fallback_minimum_price is not None:
        # if there's a fallback minimum price specified, try to find
        # contracts that are at least that price first
        for ticker in tickers:
            if midpoint_or_market_price(ticker) > fallback_minimum_price:
                the_chosen_ticker = ticker
                break
        if the_chosen_ticker is None:
            # uh of, if we make it here then all of these options are
            # net debits, so let's at least choose the ticker that will
            # result in the smallest debit (i.e., minimize the max loss)
            tickers = sorted(tickers, key=midpoint_or_market_price, reverse=True)

    if the_chosen_ticker is None:
        # fall back to the first suitable result
        the_chosen_ticker = tickers[0]

    if not the_chosen_ticker or not the_chosen_ticker.contract:
        raise RuntimeError(
            f"{underlying.symbol}: Something went wrong, the_chosen_ticker={the_chosen_ticker}"
        )

    log.notice(
        f"{underlying.symbol}: Found suitable contract at "
        f"strike={the_chosen_ticker.contract.strike} "
        f"dte={option_dte(the_chosen_ticker.contract.lastTradeDateOrContractMonth)} "
        f"price={dfmt(midpoint_or_market_price(the_chosen_ticker),3)}"
    )

    return the_chosen_ticker
