import asyncio
from typing import Any, Coroutine, List, Union

from annotated_types import T
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
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


async def tasks_progress(
    tasks: List[Coroutine[Any, Any, T]], description: str
) -> List[T]:
    results = []
    total_tasks = len(tasks)

    progress = Progress(
        TextColumn("{task.description: <50}"),
        BarColumn(),
        TaskProgressColumn(),
    )

    with progress:
        progress_task = progress.add_task(description, total=total_tasks)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            progress.advance(progress_task)

    return results
