#!/usr/bin/env python3
"""
Demo script for Phase 2.4: Data Synchronization & Versioning

Demonstrates:
1. Creating sample predictions in database
2. Injecting sparse labels
3. Exporting to DVC-tracked datasets
4. UUID sharding for raw signals

Run: python scripts/demo_data_sync.py
"""

import sys
import tempfile
from pathlib import Path

# Add project root for src/ imports, and scripts/ for sibling-module imports
# (e.g. sync_production_data).  Both inserts are needed when running from any
# working directory.  Pattern matches other scripts in this project.
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from sync_production_data import DataSyncManager

from src.database.database import Database

console = Console()


def main():
    """Run data sync demonstration."""
    console.print(
        Panel.fit(
            "[bold cyan]Phase 2.4: Data Synchronization Demo[/bold cyan]\n"
            "Production DB → DVC-tracked datasets",
            border_style="blue",
        )
    )

    # Create temporary database with sample data
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "demo.db"
        console.print(f"\n[dim]Creating demo database: {db_path}[/dim]")

        db = Database(str(db_path))

        # Step 1: Create sample devices and predictions
        console.print("\n[bold]Step 1: Creating sample data...[/bold]")

        devices = []
        for i in range(3):
            device_id = db.register_device(
                device_name=f"Device-{chr(65 + i)}-{i + 1:03d}",
                device_type=f"Sensor-Type-{i + 1}",
                location=f"Building-{i + 1}-Floor-{(i % 3) + 1}",
            )
            devices.append(device_id)
            console.print(f"  ✓ Registered device: {device_id[:8]}...")

        # Create predictions for each device
        prediction_count = 0
        for device_id in devices:
            for j in range(10):
                features = {
                    "peak_height": 2.5 + j * 0.1,
                    "fwhm": 3.0 + j * 0.05,
                    "peak_center": 50.0,
                    "peak_area": 100.0 + j * 5,
                    "noise_level": 0.05,
                    "snr": 50.0 + j,
                }

                db.store_prediction(
                    device_id=device_id,
                    time_values=[float(x) for x in range(101)],
                    amplitude_values=[2.5 + 0.01 * x for x in range(101)],
                    predicted_label=j % 2,
                    model_version="demo_v1.0",
                    features=features,
                    prediction_confidence=0.85 + j * 0.01,
                    shape_type="gaussian" if j % 2 == 0 else "lorentzian",
                )
                prediction_count += 1

        console.print(f"\n  ✓ Created {prediction_count} predictions across {len(devices)} devices")

        # Step 2: Inject sparse labels (50% labeled)
        console.print("\n[bold]Step 2: Injecting sparse labels (50%)...[/bold]")

        cursor = db.conn.cursor()
        cursor.execute("SELECT prediction_id FROM predictions ORDER BY prediction_id")
        all_predictions = [row["prediction_id"] for row in cursor.fetchall()]

        labeled_count = 0
        for i, pred_id in enumerate(all_predictions):
            if i % 2 == 0:  # Label 50%
                db.inject_sparse_label(pred_id, i % 2, "demo_labeler", "demo_user")
                labeled_count += 1

        console.print(
            f"  ✓ Injected {labeled_count} labels ({labeled_count / len(all_predictions) * 100:.0f}%)"
        )

        # Step 3: Export to DVC-tracked datasets
        console.print("\n[bold]Step 3: Exporting to DVC-tracked datasets...[/bold]")

        sync_manager = DataSyncManager(
            db_path=db_path,
            output_dir=Path(tmpdir) / "sync",
            signals_dir=Path(tmpdir) / "raw_signals",
            sample_rate=0.3,  # 30% of unlabeled
            dry_run=False,
        )

        stats = sync_manager.sync_all(add_to_dvc=False)

        # Step 4: Verify outputs
        console.print("\n[bold]Step 4: Verifying outputs...[/bold]")

        # Check CSV files
        csv_files = list((Path(tmpdir) / "sync").glob("*.csv"))
        console.print("\n  [cyan]Metadata CSVs:[/cyan]")
        for csv_file in sorted(csv_files):
            lines = len(csv_file.read_text().split("\n"))
            console.print(f"    • {csv_file.name}: {lines - 1} rows")

        # Check JSON files (UUID sharded)
        json_files = list((Path(tmpdir) / "raw_signals").rglob("*.json"))
        console.print("\n  [cyan]Raw Signals (JSON with UUID sharding):[/cyan]")
        console.print(f"    • Total files: {len(json_files)}")

        # Show shard distribution
        shard_dirs = set()
        for json_file in json_files:
            parts = json_file.parts
            # Extract shard structure: .../prefix1/prefix2/device_id/
            if len(parts) >= 3:
                shard_dir = f"{parts[-4]}/{parts[-3]}"
                shard_dirs.add(shard_dir)

        console.print(f"    • Shards used: {len(shard_dirs)} (UUID prefix distribution)")
        console.print(f"    • Example: {sorted(shard_dirs)[0] if shard_dirs else 'N/A'}")

        # Show sample JSON structure
        if json_files:
            import json

            sample_file = json_files[0]
            with open(sample_file) as f:
                sample_data = json.load(f)

            console.print("\n  [cyan]Sample JSON structure:[/cyan]")
            console.print(f"    • signal_id: {sample_data['signal_id']}")
            console.print(f"    • prediction_id: {sample_data['prediction_id']}")
            console.print(f"    • device_id: {sample_data['device_id'][:16]}...")
            console.print(f"    • n_points: {sample_data['n_points']}")
            console.print(f"    • shape_type: {sample_data.get('shape_type', 'N/A')}")

        # Summary
        console.print(
            Panel.fit(
                f"[bold green]✓ Demo Complete![/bold green]\n\n"
                f"Exported:\n"
                f"  • {stats['metadata']['predictions']:,} predictions\n"
                f"  • {stats['metadata']['features']:,} feature vectors\n"
                f"  • {stats['metadata']['devices']:,} devices\n"
                f"  • {stats['labeled_signals']:,} labeled signals (100%)\n"
                f"  • {stats['unlabeled_sample']:,} unlabeled signals ({stats['sample_rate'] * 100:.0f}% sample)\n\n"
                f"[dim]Next: Add to DVC with --dvc-add flag[/dim]",
                border_style="green",
                title="Export Summary",
            )
        )

        console.print(
            "\n[dim]In production:[/dim]\n"
            "  1. python scripts/sync_production_data.py --dvc-add\n"
            "  2. dvc push\n"
            "  3. git add *.dvc && git commit -m 'Data sync'\n"
            "  4. git push\n"
        )


if __name__ == "__main__":
    main()
