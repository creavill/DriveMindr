"""
DriveMindr CLI — the entry point.

Usage::

    drivemindr scan C:\\
    drivemindr classify --db ./drivemindr.db
    drivemindr scan C:\\ --db ./my.db --verbose
    drivemindr info --db ./my.db

No network calls. Everything local.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from drivemindr import __version__
from drivemindr.config import setup_logging
from drivemindr.database import Database
from drivemindr.scanner import FileScanner
from drivemindr.utils import format_bytes, format_count

app = typer.Typer(
    name="drivemindr",
    help="AI-Powered Windows Storage Manager — Privacy-First, Fully Offline.",
    add_completion=False,
)
console = Console()
logger = logging.getLogger("drivemindr.cli")

# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------
_db_option = typer.Option("drivemindr.db", "--db", help="Path to the SQLite database.")
_verbose_option = typer.Option(False, "--verbose", "-v", help="Enable verbose/debug logging.")


def _init(db_path: str, verbose: bool) -> Database:
    """Bootstrap logging and database for every command."""
    setup_logging(verbose=verbose, log_dir=Path(db_path).parent)
    db = Database(db_path)
    db.connect()
    return db


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def scan(
    root: str = typer.Argument(..., help="Drive or directory to scan (e.g. C:\\)"),
    db: str = _db_option,
    verbose: bool = _verbose_option,
) -> None:
    """Scan a drive/directory and store file metadata in the local database."""
    database = _init(db, verbose)

    console.print(f"\n[bold]DriveMindr v{__version__}[/bold] — scanning [cyan]{root}[/cyan]\n")

    def _progress(scanned: int, errors: int) -> None:
        console.print(
            f"  Scanned [green]{format_count(scanned)}[/green] files"
            f"  |  Errors: [yellow]{errors}[/yellow]",
            end="\r",
        )

    try:
        scanner = FileScanner(database)
        summary = scanner.scan(root, progress_callback=_progress)
        console.print()  # newline after progress

        # Show summary table
        table = Table(title="Scan Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Files", format_count(summary["files"]))
        table.add_row("Directories", format_count(summary["dirs"]))
        table.add_row("Total size", format_bytes(summary["total_bytes"]))
        table.add_row("Errors", format_count(summary["errors"]))
        table.add_row("Scan ID", summary["scan_id"])
        console.print(table)

        # Registry scan (Windows only)
        console.print("\n[dim]Scanning installed applications (Windows Registry)...[/dim]")
        app_count = scanner.scan_installed_apps()
        console.print(f"  Found [green]{app_count}[/green] installed applications.\n")

        logger.info("Scan command completed successfully — scan_id=%s", summary["scan_id"])
    except FileNotFoundError as exc:
        console.print(f"\n[red]Error:[/red] {exc}")
        logger.error("Scan failed: %s", exc)
        raise typer.Exit(code=1)
    except Exception:
        logger.exception("Unexpected error during scan")
        console.print("\n[red]Unexpected error — see drivemindr_debug.log for details.[/red]")
        raise typer.Exit(code=1)
    finally:
        database.close()


@app.command()
def classify(
    db: str = _db_option,
    verbose: bool = _verbose_option,
    model: str = typer.Option("", "--model", "-m", help="Override the Ollama model name."),
) -> None:
    """Classify scanned files using the local Ollama AI model.

    Requires Ollama running on localhost:11434. Run 'drivemindr scan' first.
    """
    from drivemindr.classifier import FileClassifier, OllamaClient

    database = _init(db, verbose)

    console.print(f"\n[bold]DriveMindr v{__version__}[/bold] — AI classification\n")

    try:
        file_count = database.file_count()
        if file_count == 0:
            console.print("[yellow]No files in database. Run 'drivemindr scan' first.[/yellow]")
            raise typer.Exit(code=1)

        console.print(f"  Files in DB: [green]{format_count(file_count)}[/green]\n")

        # Build client with optional model override
        client_kwargs: dict = {}
        if model:
            client_kwargs["model"] = model
        ollama = OllamaClient(**client_kwargs)

        classifier = FileClassifier(database, ollama_client=ollama)

        # Preflight — verify Ollama is up
        console.print("[dim]Checking Ollama availability...[/dim]")
        status = classifier.preflight_check()
        if not status["ollama_up"]:
            console.print(
                "\n[red]Ollama is not running.[/red]\n"
                "  Start it with: [cyan]ollama serve[/cyan]\n"
                "  Download from: [cyan]ollama.com[/cyan] (install offline)\n"
            )
            logger.error("Classify aborted — Ollama not available")
            raise typer.Exit(code=1)

        if not status["model_ready"]:
            target = model or "llama3.1:8b"
            console.print(
                f"\n[red]Model not found.[/red]\n"
                f"  Pull it with: [cyan]ollama pull {target}[/cyan]\n"
            )
            logger.error("Classify aborted — model not available")
            raise typer.Exit(code=1)

        console.print("[green]Ollama is ready.[/green]\n")

        # Run classification
        def _progress(classified: int, overridden: int, errors: int) -> None:
            console.print(
                f"  Classified [green]{format_count(classified)}[/green]"
                f"  |  Safety overrides: [yellow]{overridden}[/yellow]"
                f"  |  Errors: [red]{errors}[/red]",
                end="\r",
            )

        summary = classifier.classify_all(progress_callback=_progress)
        console.print()  # newline after progress

        table = Table(title="Classification Summary")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Classified", format_count(summary["classified"]))
        table.add_row("Safety overrides", format_count(summary["overridden"]))
        table.add_row("Errors", format_count(summary["errors"]))
        table.add_row("Batches", format_count(summary["batches"]))
        console.print(table)

        logger.info("Classify command completed: %s", summary)
    except typer.Exit:
        raise
    except Exception:
        logger.exception("Unexpected error during classification")
        console.print("\n[red]Unexpected error — see drivemindr_debug.log for details.[/red]")
        raise typer.Exit(code=1)
    finally:
        database.close()


@app.command()
def info(
    db: str = _db_option,
    verbose: bool = _verbose_option,
) -> None:
    """Show a summary of the current database contents."""
    database = _init(db, verbose)

    try:
        file_count = database.file_count()
        total_size = database.total_size()

        console.print(f"\n[bold]DriveMindr v{__version__}[/bold] — database info\n")
        console.print(f"  Database : [cyan]{db}[/cyan]")
        console.print(f"  Files    : [green]{format_count(file_count)}[/green]")
        console.print(f"  Total    : [green]{format_bytes(total_size)}[/green]\n")

        if file_count > 0:
            console.print("[bold]Top 20 largest files:[/bold]\n")
            table = Table()
            table.add_column("Size", justify="right", style="green")
            table.add_column("Path")
            for row in database.get_top_largest(20):
                table.add_row(format_bytes(row["size_bytes"]), row["path"])
            console.print(table)

        apps = database.get_installed_apps()
        if apps:
            console.print(f"\n[bold]Installed applications:[/bold] {len(apps)}\n")

    finally:
        database.close()


@app.command()
def version() -> None:
    """Print the DriveMindr version."""
    console.print(f"DriveMindr v{__version__}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
