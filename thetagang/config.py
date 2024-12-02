import math
from dataclasses import field
from typing import Any, Dict, List, Literal, Optional, Self

from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from thetagang.fmt import dfmt, ffmt, pfmt

error_console = Console(stderr=True, style="bold red")
console = Console()


class DisplayMixin:
    def add_to_table(self, table: Table, section: str = "") -> None:
        raise NotImplementedError


@dataclass
class Account(DisplayMixin):
    number: str = Field(...)
    cancel_orders: bool = Field(...)
    margin_usage: float = Field(..., ge=0.0)
    market_data_type: int = Field(..., ge=1, le=4)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_row("[spring_green1]Account details")
        table.add_row("", "Account number", "=", self.number)
        table.add_row("", "Cancel existing orders", "=", f"{self.cancel_orders}")
        table.add_row(
            "",
            "Margin usage",
            "=",
            f"{self.margin_usage} ({pfmt(self.margin_usage,0)})",
        )
        table.add_row("", "Market data type", "=", f"{self.market_data_type}")


@dataclass
class Constants(DisplayMixin):
    @dataclass
    class ConstantsWriteThreshold:
        write_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
        write_threshold_sigma: Optional[float] = Field(None, ge=0.0)

    write_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    write_threshold_sigma: Optional[float] = Field(None, ge=0.0)
    daily_stddev_window: str = Field("30 D")
    calls: Optional[ConstantsWriteThreshold] = Field(None)
    puts: Optional[ConstantsWriteThreshold] = Field(None)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Constants")
        table.add_row("", "Daily stddev window", "=", self.daily_stddev_window)

        c_write_thresh = (
            f"{ffmt(self.calls.write_threshold_sigma)}σ"
            if self.calls and self.calls.write_threshold_sigma
            else pfmt(self.calls.write_threshold if self.calls else None)
        )
        p_write_thresh = (
            f"{ffmt(self.puts.write_threshold_sigma)}σ"
            if self.puts and self.puts.write_threshold_sigma
            else pfmt(self.puts.write_threshold if self.puts else None)
        )

        table.add_row("", "Write threshold for puts", "=", p_write_thresh)
        table.add_row("", "Write threshold for calls", "=", c_write_thresh)


@dataclass
class OptionChains:
    expirations: int = Field(..., ge=1)
    strikes: int = Field(..., ge=1)


@dataclass
class AlgoSettings:
    strategy: str = Field("Adaptive")
    params: List[str] = field(default_factory=lambda: ["adaptivePriority", "Patient"])


@dataclass
class Orders(DisplayMixin):
    minimum_credit: float = Field(0.0, ge=0.0)
    exchange: str = Field("SMART")
    algo: AlgoSettings = Field(
        AlgoSettings("Adaptive", ["adaptivePriority", "Patient"])
    )
    price_update_delay: List[int] = field(default_factory=lambda: [30, 60])

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Order settings")
        table.add_row("", "Exchange", "=", self.exchange)
        table.add_row("", "Params", "=", f"{self.algo.params}")
        table.add_row("", "Price update delay", "=", f"{self.price_update_delay}")
        table.add_row("", "Minimum credit", "=", f"{dfmt(self.minimum_credit)}")


@dataclass
class IBAsync:
    api_response_wait_time: int = Field(60, ge=0)
    logfile: Optional[str] = Field(None)


@dataclass
class IBC:
    tradingMode: Literal["live", "paper"] = Field("paper")
    password: Optional[str] = Field(None)
    userid: Optional[str] = Field(None)
    gateway: bool = Field(True)
    RaiseRequestErrors: bool = Field(False)
    ibcPath: str = Field("/opt/ibc")
    ibcIni: str = Field("/etc/thetagang/config.ini")
    twsPath: Optional[str] = Field(None)
    twsSettingsPath: Optional[str] = Field(None)
    javaPath: str = Field("/opt/java/openjdk/bin")
    fixuserid: Optional[str] = Field(None)
    fixpassword: Optional[str] = Field(None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tradingMode": self.tradingMode,
            "password": self.password,
            "userid": self.userid,
            "gateway": self.gateway,
            "ibcPath": self.ibcPath,
            "ibcIni": self.ibcIni,
            "twsPath": self.twsPath,
            "twsSettingsPath": self.twsSettingsPath,
            "javaPath": self.javaPath,
            "fixuserid": self.fixuserid,
            "fixpassword": self.fixpassword,
        }


@dataclass
class Watchdog:
    @dataclass
    class ProbeContract:
        currency: str = Field("USD")
        exchange: str = Field("SMART")
        secType: str = Field("STK")
        symbol: str = Field("SPY")

    appStartupTime: int = Field(30)
    appTimeout: int = Field(20)
    clientId: int = Field(1)
    connectTimeout: int = Field(2)
    host: str = Field("127.0.0.1")
    port: int = Field(7497)
    probeTimeout: int = Field(4)
    readonly: bool = Field(False)
    retryDelay: int = Field(2)
    probeContract: ProbeContract = Field(default_factory=ProbeContract)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "appStartupTime": self.appStartupTime,
            "appTimeout": self.appTimeout,
            "clientId": self.clientId,
            "connectTimeout": self.connectTimeout,
            "host": self.host,
            "port": self.port,
            "probeTimeout": self.probeTimeout,
            "readonly": self.readonly,
            "retryDelay": self.retryDelay,
        }


@dataclass
class CashManagement(DisplayMixin):
    @dataclass
    class Orders:
        exchange: str = Field("SMART")
        algo: AlgoSettings = Field(AlgoSettings("Vwap", []))

    enabled: bool = Field(False)
    cash_fund: str = Field("SGOV")
    target_cash_balance: int = Field(0, ge=0)
    buy_threshold: int = Field(10000, ge=0)
    sell_threshold: int = Field(10000, ge=0)
    primary_exchange: str = Field("")
    orders: Orders = field(default_factory=Orders)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Cash management")
        table.add_row("", "Enabled", "=", f"{self.enabled}")
        table.add_row("", "Cash fund", "=", f"{self.cash_fund}")
        table.add_row("", "Target cash", "=", f"{dfmt(self.target_cash_balance)}")
        table.add_row("", "Buy threshold", "=", f"{dfmt(self.buy_threshold)}")
        table.add_row("", "Sell threshold", "=", f"{dfmt(self.sell_threshold)}")


@dataclass
class Allocation:
    lower_bound: Optional[float] = Field(None, ge=0.0)
    upper_bound: Optional[float] = Field(None, ge=0.0)
    weight: float = Field(..., ge=0.0)


@dataclass
class VIXCallHedge(DisplayMixin):
    enabled: bool = Field(False)
    delta: float = Field(0.3, ge=0.0, le=1.0)
    target_dte: int = Field(30, gt=0)
    ignore_dte: int = Field(0, ge=0)
    max_dte: Optional[int] = Field(None, ge=1)
    close_hedges_when_vix_exceeds: Optional[float] = Field(None)
    allocation: List[Allocation] = Field(
        default_factory=lambda: [
            Allocation(lower_bound=None, upper_bound=15.0, weight=0.0),
            Allocation(lower_bound=15.0, upper_bound=30.0, weight=0.01),
            Allocation(lower_bound=30.0, upper_bound=50.0, weight=0.005),
            Allocation(lower_bound=50.0, upper_bound=None, weight=0.0),
        ]
    )

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Hedging with VIX calls")
        table.add_row("", "Enabled", "=", f"{self.enabled}")
        table.add_row("", "Target delta", "<=", f"{self.delta}")
        table.add_row("", "Target DTE", ">=", f"{self.target_dte}")
        table.add_row("", "Ignore DTE", "<=", f"{self.ignore_dte}")
        if self.close_hedges_when_vix_exceeds:
            table.add_row(
                "",
                "Close hedges when VIX",
                ">=",
                f"{self.close_hedges_when_vix_exceeds}",
            )

        for alloc in self.allocation:
            if alloc.lower_bound or alloc.upper_bound:
                table.add_row()
                if alloc.lower_bound:
                    table.add_row(
                        "",
                        f"Allocate {pfmt(alloc.weight)} when VIXMO",
                        ">=",
                        f"{alloc.lower_bound}",
                    )
                if alloc.upper_bound:
                    table.add_row(
                        "",
                        f"Allocate {pfmt(alloc.weight)} when VIXMO",
                        "<=",
                        f"{alloc.upper_bound}",
                    )


@dataclass
class WriteWhen(DisplayMixin):
    @dataclass
    class Puts:
        green: bool = Field(False)
        red: bool = Field(True)

    @dataclass
    class Calls:
        green: bool = Field(True)
        red: bool = Field(False)
        cap_factor: float = Field(1.0, ge=0.0, le=1.0)
        cap_target_floor: float = Field(0.0, ge=0.0, le=1.0)
        excess_only: bool = Field(False)

    calculate_net_contracts: bool = Field(False)
    calls: Calls = Field(default_factory=Calls)
    puts: Puts = Field(default_factory=Puts)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]When writing new contracts")
        table.add_row(
            "",
            "Calculate net contract positions",
            "=",
            f"{self.calculate_net_contracts}",
        )
        table.add_row("", "Puts, write when red", "=", f"{self.puts.red}")
        table.add_row("", "Puts, write when green", "=", f"{self.puts.green}")
        table.add_row("", "Calls, write when green", "=", f"{self.calls.green}")
        table.add_row("", "Calls, write when red", "=", f"{self.calls.red}")
        table.add_row("", "Call cap factor", "=", f"{pfmt(self.calls.cap_factor)}")
        table.add_row(
            "", "Call cap target floor", "=", f"{pfmt(self.calls.cap_target_floor)}"
        )
        table.add_row("", "Excess only", "=", f"{self.calls.excess_only}")


@dataclass
class RollWhen(DisplayMixin):
    @dataclass
    class Calls:
        itm: bool = Field(True)
        always_when_itm: bool = Field(False)
        credit_only: bool = Field(False)
        has_excess: bool = Field(True)
        maintain_high_water_mark: bool = Field(False)

    @dataclass
    class Puts:
        itm: bool = Field(False)
        always_when_itm: bool = Field(False)
        credit_only: bool = Field(False)
        has_excess: bool = Field(True)

    pnl: float = Field(0.0, ge=0.0, le=1.0)
    dte: int = Field(..., ge=0)
    min_pnl: float = Field(0.0)
    close_at_pnl: float = Field(1.0)
    close_if_unable_to_roll: bool = Field(False)
    max_dte: Optional[int] = Field(None, ge=1)
    calls: Calls = Field(default_factory=Calls)
    puts: Puts = Field(default_factory=Puts)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Close option positions")
        table.add_row("", "When P&L", ">=", f"{pfmt(self.close_at_pnl,0)}")
        table.add_row(
            "", "Close if unable to roll", "=", f"{self.close_if_unable_to_roll}"
        )

        table.add_section()
        table.add_row("[spring_green1]Roll options when either condition is true")
        table.add_row(
            "",
            "Days to expiry",
            "<=",
            f"{self.dte} and P&L >= {self.min_pnl} ({pfmt(self.min_pnl,0)})",
        )

        if self.max_dte:
            table.add_row(
                "",
                "P&L",
                ">=",
                f"{self.pnl} ({pfmt(self.pnl,0)}) and DTE <= {self.max_dte}",
            )
        else:
            table.add_row("", "P&L", ">=", f"{self.pnl} ({pfmt(self.pnl,0)})")

        table.add_row("", "Puts: credit only", "=", f"{self.puts.credit_only}")
        table.add_row("", "Puts: roll excess", "=", f"{self.puts.has_excess}")
        table.add_row("", "Calls: credit only", "=", f"{self.calls.credit_only}")
        table.add_row("", "Calls: roll excess", "=", f"{self.calls.has_excess}")
        table.add_row(
            "",
            "Calls: maintain high water mark",
            "=",
            f"{self.calls.maintain_high_water_mark}",
        )


@dataclass
class Target(DisplayMixin):
    @dataclass
    class Puts:
        delta: Optional[float] = Field(None, ge=0.0, le=1.0)

    @dataclass
    class Calls:
        delta: Optional[float] = Field(None, ge=0.0, le=1.0)

    dte: int = Field(..., ge=0)
    delta: float = Field(0.3, ge=0.0, le=1.0)
    minimum_open_interest: int = Field(..., ge=0)
    maximum_new_contracts_percent: float = Field(..., ge=0.0, le=1.0)
    max_dte: Optional[int] = Field(None, ge=1)
    maximum_new_contracts: Optional[int] = Field(None, ge=1)
    calls: Puts = Field(default_factory=Puts)
    puts: Calls = Field(default_factory=Calls)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Write options with targets of")
        table.add_row("", "Days to expiry", ">=", f"{self.dte}")
        if self.max_dte:
            table.add_row("", "Days to expiry", "<=", f"{self.max_dte}")
        table.add_row("", "Default delta", "<=", f"{self.delta}")
        if self.puts and self.puts.delta:
            table.add_row("", "Delta for puts", "<=", f"{self.puts.delta}")
        if self.calls and self.calls.delta:
            table.add_row("", "Delta for calls", "<=", f"{self.calls.delta}")
        table.add_row(
            "",
            "Maximum new contracts",
            "=",
            f"{pfmt(self.maximum_new_contracts_percent,0)} of buying power",
        )
        table.add_row("", "Minimum open interest", "=", f"{self.minimum_open_interest}")


@dataclass
class SymbolConfig:
    @dataclass
    class WriteWhen:
        green: Optional[bool] = None
        red: Optional[bool] = None

    @dataclass
    class CallsConfig:
        cap_factor: Optional[float] = Field(None, ge=0, le=1)
        cap_target_floor: Optional[float] = Field(None, ge=0, le=1)
        excess_only: Optional[bool] = Field(None)
        delta: Optional[float] = Field(None, ge=0, le=1)
        write_threshold: Optional[float] = Field(None, ge=0, le=1)
        write_threshold_sigma: Optional[float] = Field(None, gt=0)
        strike_limit: Optional[float] = Field(None, gt=0)
        maintain_high_water_mark: Optional[bool] = None
        write_when: Optional[WriteWhen] = field(default_factory=WriteWhen)

    @dataclass
    class PutsConfig:
        delta: Optional[float] = Field(None, ge=0, le=1)
        write_threshold: Optional[float] = Field(None, ge=0, le=1)
        write_threshold_sigma: Optional[float] = Field(None, gt=0)
        strike_limit: Optional[float] = Field(None, gt=0)
        write_when: Optional[WriteWhen] = field(default_factory=WriteWhen)

    weight: float = Field(..., ge=0, le=1)
    primary_exchange: str = Field("", min_length=1)
    delta: Optional[float] = Field(None, ge=0, le=1)
    write_threshold: Optional[float] = Field(None, ge=0, le=1)
    write_threshold_sigma: Optional[float] = Field(None, gt=0)
    max_dte: Optional[int] = Field(None, ge=1)
    dte: Optional[int] = Field(None, ge=0)
    close_if_unable_to_roll: Optional[bool] = Field(None)
    calls: Optional[CallsConfig] = Field(None)
    puts: Optional[PutsConfig] = Field(None)
    adjust_price_after_delay: bool = Field(False)
    no_trading: Optional[bool] = Field(None)


@dataclass
class Config(DisplayMixin):
    account: Account
    orders: Orders
    option_chains: OptionChains
    ib_async: IBAsync
    ibc: IBC
    watchdog: Watchdog
    cash_management: CashManagement
    vix_call_hedge: VIXCallHedge
    write_when: WriteWhen
    roll_when: RollWhen
    target: Target
    symbols: Dict[str, SymbolConfig]
    constants: Constants = field(default_factory=Constants)

    @model_validator(mode="after")
    def check_symbols(self) -> Self:
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")
        return self

    @model_validator(mode="after")
    def check_symbol_weights(self) -> Self:
        assert math.isclose(
            1, sum([s.weight or 0.0 for s in self.symbols.values()]), rel_tol=1e-5
        )

    def create_symbols_table(self) -> Table:
        table = Table(
            title="Configured symbols and target weights",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("Symbol")
        table.add_column("Weight", justify="right")
        table.add_column("Call delta", justify="right")
        table.add_column("Call strike limit", justify="right")
        table.add_column("Call threshold", justify="right")
        table.add_column("HWM", justify="right")
        table.add_column("Put delta", justify="right")
        table.add_column("Put strike limit", justify="right")
        table.add_column("Put threshold", justify="right")

        for symbol, sconfig in self.symbols.items():
            call_thresh = (
                f"{ffmt(sconfig.calls.write_threshold_sigma)}σ"
                if sconfig.calls and sconfig.calls.write_threshold_sigma
                else pfmt(sconfig.calls.write_threshold if sconfig.calls else None)
            )
            put_thresh = (
                f"{ffmt(sconfig.puts.write_threshold_sigma)}σ"
                if sconfig.puts and sconfig.puts.write_threshold_sigma
                else pfmt(sconfig.puts.write_threshold if sconfig.puts else None)
            )

            table.add_row(
                symbol,
                pfmt(sconfig.weight or 0.0),
                ffmt(sconfig.calls.delta if sconfig.calls else None),
                dfmt(sconfig.calls.strike_limit if sconfig.calls else None),
                call_thresh,
                str(sconfig.calls.maintain_high_water_mark if sconfig.calls else False),
                ffmt(sconfig.puts.delta if sconfig.puts else None),
                dfmt(sconfig.puts.strike_limit if sconfig.puts else None),
                put_thresh,
            )
        return table

    def display(self, config_path: str) -> None:
        console = Console()
        config_table = Table(box=box.SIMPLE_HEAVY)
        config_table.add_column("Section")
        config_table.add_column("Setting")
        config_table.add_column("")
        config_table.add_column("Value")

        # Add all component tables
        self.account.add_to_table(config_table)
        if self.constants:
            self.constants.add_to_table(config_table)
        self.orders.add_to_table(config_table)
        self.roll_when.add_to_table(config_table)
        self.write_when.add_to_table(config_table)
        self.target.add_to_table(config_table)
        self.cash_management.add_to_table(config_table)
        self.vix_call_hedge.add_to_table(config_table)

        # Create tree and add tables
        tree = Tree(":control_knobs:")
        tree.add(Group(f":file_cabinet: Loaded from {config_path}", config_table))
        tree.add(Group(":yin_yang: Symbology", self.create_symbols_table()))

        console.print(Panel(tree, title="Config"))


def normalize_config(config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # Do any pre-processing necessary to the config here, such as handling
    # defaults, deprecated values, config changes, etc.
    if "minimum_cushion" in config["account"]:
        raise RuntimeError(
            "Config error: minimum_cushion is deprecated and replaced with margin_usage. See sample config for details."
        )

    if "ib_insync" in config:
        error_console.print(
            "WARNING: config param `ib_insync` is deprecated, please rename it to the equivalent `ib_async`.",
        )

        if "ib_async" not in config:
            # swap the old ib_insync key to the new ib_async key
            config["ib_async"] = config["ib_insync"]
        del config["ib_insync"]

    if "twsVersion" in config["ibc"]:
        error_console.print(
            "WARNING: config param ibc.twsVersion is deprecated, please remove it from your config.",
        )

        # TWS version is pinned to latest stable, delete any existing config if it's present
        del config["ibc"]["twsVersion"]

    if "maximum_new_contracts" in config["target"]:
        error_console.print(
            "WARNING: config param target.maximum_new_contracts is deprecated, please remove it from your config.",
        )

        del config["target"]["maximum_new_contracts"]

    # xor: should have weight OR parts, but not both
    if any(["weight" in s for s in config["symbols"].values()]) == any(
        ["parts" in s for s in config["symbols"].values()]
    ):
        raise RuntimeError(
            "ERROR: all symbols should have either a weight or parts specified, but parts and weights cannot be mixed."
        )

    if "parts" in list(config["symbols"].values())[0]:
        # If using "parts" instead of "weight", convert parts into weights
        total_parts = float(sum([s["parts"] for s in config["symbols"].values()]))
        for k in config["symbols"].keys():
            config["symbols"][k]["weight"] = config["symbols"][k]["parts"] / total_parts
        for s in config["symbols"].values():
            del s["parts"]

    if (
        "close_at_pnl" in config["roll_when"]
        and config["roll_when"]["close_at_pnl"]
        and config["roll_when"]["close_at_pnl"] <= config["roll_when"]["min_pnl"]
    ):
        raise RuntimeError(
            "ERROR: roll_when.close_at_pnl needs to be greater than roll_when.min_pnl."
        )

    return config
