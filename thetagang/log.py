from typing import Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme(
    {
        "notice": "green",
        "warning": "yellow",
        "error": "red",
    }
)

console: Console = Console(theme=custom_theme)


def info(text: str) -> None:
    console.print(text)


def notice(text: str) -> None:
    console.print(text, style="notice")


def warning(text: str) -> None:
    console.print(text, style="warning")


def error(text: str) -> None:
    console.print_exception()
    console.print(text, style="red")


def print(content: Union[Panel, Table]) -> None:
    console.print(content)
