#!/usr/bin/env python3
"""
Data Synchronization Script for DagsHub/DVC

Exports production database to versioned datasets:
- Metadata tables (predictions, features, devices, labels) → CSV
- Raw signals (labeled + 10% sampled unlabeled) → JSON with UUID sharding

Usage:
    python scripts/sync_production_data.py --db-path data/database/mlops.db
    python scripts/sync_production_data.py --sample-rate 0.1 --dvc-add
    python scripts/sync_production_data.py --dry-run  # Preview only

Airflow Integration:
    Daily DAG exports production data, versions with DVC, pushes to DagsHub
"""

import argparse
import logging
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


class DataSyncManager:
    """Manages export of production database to DVC-tracked datasets."""

    def __init__(
        self,
        db_path: str | Path = "data/database/mlops.db",
        output_dir: str | Path = "data/sync",
        signals_dir: str | Path = "data/raw_signals",
        sample_rate: float = 0.1,
        dry_run: bool = False,
    ):
        """
        Initialize data sync manager.

        Args:
            db_path: Path to SQLite database (ignored when DATABASE_URL env var is set
                     and starts with 'postgresql' — PostgreSQL is used instead).
            output_dir: Directory for CSV exports (default: data/sync)
            signals_dir: Directory for JSON signal exports (default: data/raw_signals)
            sample_rate: Fraction of unlabeled signals to export (0.0-1.0, default: 0.1)
            dry_run: If True, preview actions without executing
        """
        # Database() reads DATABASE_URL from the environment automatically.
        # When DATABASE_URL starts with 'postgresql', db_path is ignored.
        self.db = Database(str(db_path))
        self.output_dir = Path(output_dir)
        self.signals_dir = Path(signals_dir)
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self.dry_run = dry_run

        # Create output directories
        if not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.signals_dir.mkdir(parents=True, exist_ok=True)

        self.stats: dict[str, Any] = {}

    def export_metadata_tables(self, since: datetime | None = None) -> dict[str, int]:
        """
        Export metadata tables to CSV files.

        Args:
            since: Optional datetime to export only records after this time

        Returns:
            Dict with counts: {"predictions": N, "features": N, ...}
        """
        counts = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Predictions
            progress.add_task("Exporting predictions...", total=None)
            if not self.dry_run:
                counts["predictions"] = self.db.export_predictions_to_csv(
                    self.output_dir / "predictions.csv", since=since
                )
            else:
                counts["predictions"] = 0

            # Features
            progress.add_task("Exporting features...", total=None)
            if not self.dry_run:
                counts["features"] = self.db.export_features_to_csv(
                    self.output_dir / "features.csv", since=since
                )
            else:
                counts["features"] = 0

            # Devices
            progress.add_task("Exporting devices...", total=None)
            if not self.dry_run:
                counts["devices"] = self.db.export_devices_to_csv(self.output_dir / "devices.csv")
            else:
                counts["devices"] = 0

            # Sparse labels
            progress.add_task("Exporting sparse labels...", total=None)
            if not self.dry_run:
                counts["sparse_labels"] = self.db.export_sparse_labels_to_csv(
                    self.output_dir / "sparse_labels.csv", since=since
                )
            else:
                counts["sparse_labels"] = 0

        return counts

    def export_labeled_signals(self) -> int:
        """
        Export all labeled signals to JSON files with UUID sharding.

        Returns:
            Number of signals exported
        """
        labeled_ids = self.db.get_labeled_signal_ids()

        if self.dry_run:
            console.print(f"[yellow]Would export {len(labeled_ids)} labeled signals")
            return len(labeled_ids)

        with console.status(f"[bold green]Exporting {len(labeled_ids)} labeled signals..."):
            count = self.db.export_signals_to_json(self.signals_dir, labeled_ids)

        return count

    def export_sampled_unlabeled_signals(self) -> int:
        """
        Export random sample of unlabeled signals for drift monitoring.

        Returns:
            Number of signals exported
        """
        unlabeled_ids = self.db.get_unlabeled_signal_ids()

        # Calculate sample size (minimum 1 if any exist)
        if not unlabeled_ids:
            console.print("[yellow]No unlabeled signals found")
            return 0

        sample_size = max(1, int(len(unlabeled_ids) * self.sample_rate))
        sample_ids = random.sample(unlabeled_ids, k=sample_size)

        if self.dry_run:
            console.print(
                f"[yellow]Would sample {sample_size}/{len(unlabeled_ids)} "
                f"unlabeled signals ({self.sample_rate * 100:.1f}%)"
            )
            return sample_size

        with console.status(
            f"[bold green]Exporting {sample_size}/{len(unlabeled_ids)} "
            f"unlabeled signals ({self.sample_rate * 100:.1f}%)..."
        ):
            count = self.db.export_signals_to_json(self.signals_dir, sample_ids)

        return count

    def add_to_dvc(self) -> bool:
        """
        Add exported data to DVC tracking.

        Returns:
            True if successful, False otherwise
        """
        if self.dry_run:
            console.print("[yellow]Would run: dvc add data/sync/ data/raw_signals/")
            return True

        try:
            with console.status("[bold green]Adding files to DVC..."):
                # Add sync directory (CSVs)
                subprocess.run(
                    ["dvc", "add", str(self.output_dir)],
                    check=True,
                    cwd=Path(__file__).parent.parent,
                    capture_output=True,
                )

                # Add raw_signals directory (JSONs)
                subprocess.run(
                    ["dvc", "add", str(self.signals_dir)],
                    check=True,
                    cwd=Path(__file__).parent.parent,
                    capture_output=True,
                )

            console.print("[green]✓[/green] Files added to DVC")
            return True

        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗[/red] DVC add failed: {e}")
            logger.error(f"DVC add failed: {e.stderr.decode() if e.stderr else e}")
            return False
        except FileNotFoundError:
            console.print("[red]✗[/red] DVC not found. Install with: pip install dvc")
            return False

    def sync_all(self, since: datetime | None = None, add_to_dvc: bool = False) -> dict[str, Any]:
        """
        Execute full sync workflow.

        Args:
            since: Optional datetime to export only recent records
            add_to_dvc: If True, add exported files to DVC tracking

        Returns:
            Dict with export statistics
        """
        console.rule("[bold blue]Data Synchronization", align="left")

        if self.dry_run:
            console.print("[bold yellow]DRY RUN MODE - No files will be modified\n")

        # Export metadata
        console.print("[bold cyan]1. Exporting metadata tables...[/bold cyan]")
        metadata_counts = self.export_metadata_tables(since=since)

        # Export labeled signals
        console.print("\n[bold cyan]2. Exporting labeled signals...[/bold cyan]")
        labeled_count = self.export_labeled_signals()

        # Export sampled unlabeled signals
        console.print("\n[bold cyan]3. Sampling unlabeled signals...[/bold cyan]")
        unlabeled_sample_count = self.export_sampled_unlabeled_signals()

        # Summary stats
        stats = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata_counts,
            "labeled_signals": labeled_count,
            "unlabeled_sample": unlabeled_sample_count,
            "sample_rate": self.sample_rate,
            "dry_run": self.dry_run,
        }

        # Display summary table
        self._display_summary(stats)

        # Add to DVC if requested
        if add_to_dvc and not self.dry_run:
            console.print("\n[bold cyan]4. Adding to DVC tracking...[/bold cyan]")
            dvc_success = self.add_to_dvc()
            stats["dvc_added"] = dvc_success

            if dvc_success:
                console.print(
                    "\n[bold green]Next steps:[/bold green]\n"
                    "  1. dvc push          # Push to DagsHub remote\n"
                    "  2. git add *.dvc     # Stage DVC metadata\n"
                    "  3. git commit -m 'Data sync {timestamp}'\n"
                    "  4. git push          # Push to GitHub"
                )

        return stats

    def _display_summary(self, stats: dict[str, Any]) -> None:
        """Display summary table of export statistics."""
        table = Table(title="Export Summary", show_header=True, header_style="bold magenta")
        table.add_column("Category", style="cyan", width=20)
        table.add_column("Count", justify="right", style="green", width=10)

        # Metadata rows
        for name, count in stats["metadata"].items():
            table.add_row(f"{name.replace('_', ' ').title()}", f"{count:,}")

        table.add_section()

        # Signal rows
        table.add_row("Labeled Signals", f"{stats['labeled_signals']:,}")
        table.add_row("Unlabeled Sample", f"{stats['unlabeled_sample']:,}")
        table.add_row("Sample Rate", f"{stats['sample_rate'] * 100:.1f}%")

        console.print("\n", table)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Export production database to DVC-tracked datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full sync with DVC tracking
  python scripts/sync_production_data.py --dvc-add

  # Sync only recent data (last 7 days)
  python scripts/sync_production_data.py --since-days 7

  # Custom sample rate (20% of unlabeled)
  python scripts/sync_production_data.py --sample-rate 0.2

  # Preview without executing
  python scripts/sync_production_data.py --dry-run
        """,
    )

    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help=(
            "Path to SQLite database (default: data/database/mlops.db). "
            "Ignored when DATABASE_URL env var is set to a PostgreSQL URL."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sync",
        help="Output directory for CSV files (default: data/sync)",
    )
    parser.add_argument(
        "--signals-dir",
        type=str,
        default="data/raw_signals",
        help="Output directory for JSON signals (default: data/raw_signals)",
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.1,
        help="Fraction of unlabeled signals to export (default: 0.1)",
    )
    parser.add_argument(
        "--since-days",
        type=int,
        help="Export only records from last N days",
    )
    parser.add_argument(
        "--dvc-add",
        action="store_true",
        help="Add exported files to DVC tracking",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview actions without executing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve database source: DATABASE_URL env var takes priority over --db-path.
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url.startswith("postgresql"):
        console.print("[dim]Database: PostgreSQL (from DATABASE_URL)[/dim]\n")
        db_path_resolved = args.db_path or "data/database/mlops.db"
    else:
        db_path_resolved = args.db_path or "data/database/mlops.db"
        console.print(f"[dim]Database: SQLite ({db_path_resolved})[/dim]\n")
        if not args.dry_run and not Path(db_path_resolved).exists():
            console.print(
                f"[red]Error: SQLite database not found: {db_path_resolved}\n"
                "Set DATABASE_URL=postgresql://... to use PostgreSQL instead.[/red]"
            )
            sys.exit(1)

    # Calculate since datetime if specified
    since = None
    if args.since_days:
        from datetime import timedelta

        since = datetime.now() - timedelta(days=args.since_days)
        console.print(f"[dim]Exporting records since: {since.isoformat()}[/dim]\n")

    try:
        # Initialize sync manager
        sync_manager = DataSyncManager(
            db_path=db_path_resolved,
            output_dir=args.output_dir,
            signals_dir=args.signals_dir,
            sample_rate=args.sample_rate,
            dry_run=args.dry_run,
        )

        # Execute sync
        sync_manager.sync_all(since=since, add_to_dvc=args.dvc_add)

        # Exit with success
        console.print("\n[bold green]✓ Sync completed successfully[/bold green]")
        sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n[yellow]Sync cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Sync failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
