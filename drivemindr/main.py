"""
DriveMindr CLI — the entry point.

Usage::

    drivemindr scan C:\\
    drivemindr classify --db ./drivemindr.db
    drivemindr dashboard --db ./drivemindr.db
    drivemindr execute --dry-run --db ./drivemindr.db
    drivemindr execute --db ./drivemindr.db
    drivemindr undo <batch_id> --db ./drivemindr.db
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
def dashboard(
    db: str = _db_option,
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit server port."),
) -> None:
    """Launch the review dashboard (Streamlit).

    Opens a browser with Drive Overview, Action Review, and Execution Plan views.
    """
    import subprocess
    import shutil

    streamlit_path = shutil.which("streamlit")
    if streamlit_path is None:
        console.print(
            "\n[red]Streamlit is not installed.[/red]\n"
            "  Install it with: [cyan]pip install streamlit[/cyan]\n"
        )
        raise typer.Exit(code=1)

    dashboard_file = Path(__file__).parent / "dashboard.py"
    if not dashboard_file.exists():
        console.print("[red]dashboard.py not found.[/red]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]DriveMindr v{__version__}[/bold] — launching dashboard\n")
    console.print(f"  Database: [cyan]{db}[/cyan]")
    console.print(f"  URL: [cyan]http://localhost:{port}[/cyan]\n")

    try:
        subprocess.run(
            [
                streamlit_path, "run", str(dashboard_file),
                "--server.port", str(port),
                "--server.headless", "true",
                "--", "--db", db,
            ],
            check=True,
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")
    except subprocess.CalledProcessError as exc:
        logger.error("Streamlit exited with code %d", exc.returncode)
        raise typer.Exit(code=1)


@app.command()
def execute(
    db: str = _db_option,
    verbose: bool = _verbose_option,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview actions without making any changes.",
    ),
) -> None:
    """Execute user-approved actions (move, delete, archive).

    Run 'drivemindr scan', 'drivemindr classify', and review in the dashboard
    first. Use --dry-run to preview what would happen.
    """
    from drivemindr.executor import ExecutionEngine

    database = _init(db, verbose)

    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[bold red]LIVE[/bold red]"
    console.print(f"\n[bold]DriveMindr v{__version__}[/bold] — execution engine ({mode})\n")

    try:
        stats = database.get_review_stats()
        if stats["approved"] == 0:
            console.print(
                "[yellow]No approved actions. Review files in the dashboard first.[/yellow]"
            )
            raise typer.Exit(code=1)

        console.print(f"  Approved actions: [green]{stats['approved']:,}[/green]\n")

        if not dry_run:
            console.print(
                "[bold red]WARNING:[/bold red] This will move and delete files. "
                "All operations are logged and reversible via undo.\n"
            )
            confirm = typer.confirm("Proceed with execution?")
            if not confirm:
                console.print("[dim]Aborted.[/dim]")
                raise typer.Exit(code=0)

        engine = ExecutionEngine(database)

        def _progress(moved: int, deleted: int, archived: int, errors: int) -> None:
            console.print(
                f"  Moved [green]{moved}[/green]"
                f"  |  Deleted [yellow]{deleted}[/yellow]"
                f"  |  Archived [blue]{archived}[/blue]"
                f"  |  Errors [red]{errors}[/red]",
                end="\r",
            )

        summary = engine.execute_plan(dry_run=dry_run, progress_callback=_progress)
        console.print()  # newline after progress

        table = Table(title="Execution Summary" + (" (DRY RUN)" if dry_run else ""))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Moved", format_count(summary["moved"]))
        table.add_row("Deleted", format_count(summary["deleted"]))
        table.add_row("Archived", format_count(summary["archived"]))
        table.add_row("Symlinked", format_count(summary["symlinked"]))
        table.add_row("Skipped", format_count(summary["skipped"]))
        table.add_row("Errors", format_count(summary["errors"]))
        if summary["batch_id"]:
            table.add_row("Batch ID", summary["batch_id"])
        console.print(table)

        if not dry_run and summary["batch_id"]:
            console.print(
                f"\n  To undo: [cyan]drivemindr undo {summary['batch_id']}[/cyan]\n"
            )

        logger.info("Execute command completed: %s", summary)
    except typer.Exit:
        raise
    except Exception:
        logger.exception("Unexpected error during execution")
        console.print("\n[red]Unexpected error — see drivemindr_debug.log for details.[/red]")
        raise typer.Exit(code=1)
    finally:
        database.close()


@app.command()
def undo(
    batch_id: str = typer.Argument(..., help="Batch ID to undo (from execute output)."),
    db: str = _db_option,
    verbose: bool = _verbose_option,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview undo without making changes.",
    ),
) -> None:
    """Undo a batch of executed actions."""
    from drivemindr.undo import UndoManager

    database = _init(db, verbose)

    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[bold]LIVE[/bold]"
    console.print(f"\n[bold]DriveMindr v{__version__}[/bold] — undo ({mode})\n")

    try:
        undo_mgr = UndoManager(database)
        summary = undo_mgr.undo_batch(batch_id, dry_run=dry_run)

        table = Table(title="Undo Summary" + (" (DRY RUN)" if dry_run else ""))
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Undone", format_count(summary["undone"]))
        table.add_row("Skipped", format_count(summary["skipped"]))
        table.add_row("Failed", format_count(summary["failed"]))
        console.print(table)
    except Exception:
        logger.exception("Unexpected error during undo")
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
