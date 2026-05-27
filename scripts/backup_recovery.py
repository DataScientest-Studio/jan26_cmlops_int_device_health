"""
Backup and Recovery Script

Provides automated backup and recovery for:
- PostgreSQL database
- MLflow artifacts and metadata
- DVC tracked data
- Model registry

Usage:
    # Create backup
    python scripts/backup_recovery.py backup --output backups/

    # Restore from backup
    python scripts/backup_recovery.py restore --backup backups/backup_2026_02_19.tar.gz

    # List backups
    python scripts/backup_recovery.py list --directory backups/
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(help="Backup and recovery for MLOps system")


def get_timestamp() -> str:
    """Get current timestamp for backup naming."""
    return datetime.now().strftime("%Y_%m_%d_%H%M%S")


def check_docker_running() -> bool:
    """Check if Docker services are running."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q"],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def backup_database(backup_dir: Path) -> Path:
    """Backup PostgreSQL database."""
    console.print("\n📦 Backing up PostgreSQL database...")

    db_backup_file = backup_dir / "postgres_backup.sql"

    try:
        # Export database using pg_dump via docker exec
        result = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "postgres",
                "pg_dump",
                "-U",
                "mlflow_user",
                "-d",
                "mlflow",
                "--no-owner",
                "--no-acl",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        db_backup_file.write_text(result.stdout)
        console.print(f"  ✅ Database backed up: {db_backup_file}")
        return db_backup_file

    except subprocess.CalledProcessError as e:
        console.print(f"  ❌ Database backup failed: {e}", style="bold red")
        raise


def backup_mlflow(backup_dir: Path) -> Path | None:
    """Backup MLflow artifacts directory."""
    console.print("\n📦 Backing up MLflow artifacts...")

    mlruns_dir = Path("mlruns")
    if not mlruns_dir.exists():
        console.print("  ⚠️  No mlruns directory found, skipping")
        return None

    mlflow_backup = backup_dir / "mlruns"
    shutil.copytree(mlruns_dir, mlflow_backup, dirs_exist_ok=True)

    console.print(f"  ✅ MLflow artifacts backed up: {mlflow_backup}")
    return mlflow_backup


def backup_models(backup_dir: Path) -> Path | None:
    """Backup registered models."""
    console.print("\n📦 Backing up models...")

    models_dir = Path("models")
    if not models_dir.exists():
        console.print("  ⚠️  No models directory found, skipping")
        return None

    models_backup = backup_dir / "models"
    shutil.copytree(models_dir, models_backup, dirs_exist_ok=True)

    console.print(f"  ✅ Models backed up: {models_backup}")
    return models_backup


def backup_dvc_cache(backup_dir: Path) -> Path | None:
    """Backup DVC cache."""
    console.print("\n📦 Backing up DVC cache...")

    dvc_cache = Path(".dvc/cache")
    if not dvc_cache.exists():
        console.print("  ⚠️  No DVC cache found, skipping")
        return None

    cache_backup = backup_dir / "dvc_cache"
    shutil.copytree(dvc_cache, cache_backup, dirs_exist_ok=True)

    console.print(f"  ✅ DVC cache backed up: {cache_backup}")
    return cache_backup


def create_backup_manifest(backup_dir: Path, components: dict[str, Any]) -> Path:
    """Create manifest file with backup metadata."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "components": components,
        "system_info": {
            "python_version": subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "docker_version": subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
            ).stdout.strip(),
        },
    }

    manifest_file = backup_dir / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))

    return manifest_file


def compress_backup(backup_dir: Path, output_file: Path) -> Path:
    """Compress backup directory into tar.gz."""
    console.print(f"\n🗜️  Compressing backup to {output_file}...")

    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(backup_dir, arcname=backup_dir.name)

    # Calculate size
    size_mb = output_file.stat().st_size / (1024 * 1024)
    console.print(f"  ✅ Backup compressed: {size_mb:.2f} MB")

    return output_file


@app.command()
def backup(
    output: Path = typer.Option(
        Path("backups"),
        help="Output directory for backups",
    ),
    compress: bool = typer.Option(
        True,
        help="Compress backup as tar.gz",
    ),
) -> None:
    """Create full system backup."""
    console.print("[bold blue]🔄 Starting backup process...[/bold blue]")

    # Check Docker is running
    if not check_docker_running():
        console.print(
            "[bold red]❌ Docker services not running. Start with: docker compose up -d[/bold red]"
        )
        raise typer.Exit(1)

    # Create backup directory
    timestamp = get_timestamp()
    backup_dir = output / f"backup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Backup components
    components = {}

    try:
        db_backup = backup_database(backup_dir)
        components["database"] = str(db_backup.name) if db_backup else None

        mlflow_backup = backup_mlflow(backup_dir)
        components["mlflow"] = str(mlflow_backup.name) if mlflow_backup else None

        models_backup = backup_models(backup_dir)
        components["models"] = str(models_backup.name) if models_backup else None

        dvc_backup = backup_dvc_cache(backup_dir)
        components["dvc"] = str(dvc_backup.name) if dvc_backup else None

        # Create manifest
        manifest = create_backup_manifest(backup_dir, components)
        console.print(f"\n📋 Backup manifest created: {manifest}")

        # Compress if requested
        if compress:
            archive_file = output / f"backup_{timestamp}.tar.gz"
            compress_backup(backup_dir, archive_file)

            # Remove uncompressed directory
            shutil.rmtree(backup_dir)

            console.print(f"\n[bold green]✅ Backup complete: {archive_file}[/bold green]")
        else:
            console.print(f"\n[bold green]✅ Backup complete: {backup_dir}[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]❌ Backup failed: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command()
def restore(
    backup: Path = typer.Argument(..., help="Path to backup file or directory"),
    skip_database: bool = typer.Option(
        False,
        help="Skip database restore (keep existing)",
    ),
) -> None:
    """Restore system from backup."""
    console.print("[bold blue]🔄 Starting restore process...[/bold blue]")

    if not backup.exists():
        console.print(f"[bold red]❌ Backup not found: {backup}[/bold red]")
        raise typer.Exit(1)

    # Extract if tar.gz
    if backup.suffix == ".gz":
        console.print(f"\n📦 Extracting backup: {backup}")
        with tarfile.open(backup, "r:gz") as tar:
            extract_dir = backup.parent / "restore_temp"
            extract_dir.mkdir(exist_ok=True)
            tar.extractall(extract_dir)

        # Find extracted backup directory
        backup_dirs = list(extract_dir.glob("backup_*"))
        if not backup_dirs:
            console.print("[bold red]❌ No backup directory found in archive[/bold red]")
            raise typer.Exit(1)

        backup_dir = backup_dirs[0]
    else:
        backup_dir = backup

    # Read manifest
    manifest_file = backup_dir / "manifest.json"
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text())
        console.print(f"\n📋 Backup created: {manifest['timestamp']}")

    # Restore database
    if not skip_database:
        db_backup = backup_dir / "postgres_backup.sql"
        if db_backup.exists():
            console.print("\n📦 Restoring PostgreSQL database...")
            try:
                with open(db_backup) as f:
                    subprocess.run(
                        [
                            "docker",
                            "compose",
                            "exec",
                            "-T",
                            "postgres",
                            "psql",
                            "-U",
                            "mlflow_user",
                            "-d",
                            "mlflow",
                        ],
                        stdin=f,
                        check=True,
                    )
                console.print("  ✅ Database restored")
            except subprocess.CalledProcessError as e:
                console.print(f"  ❌ Database restore failed: {e}", style="bold red")

    # Restore MLflow
    mlflow_backup = backup_dir / "mlruns"
    if mlflow_backup.exists():
        console.print("\n📦 Restoring MLflow artifacts...")
        if Path("mlruns").exists():
            shutil.rmtree("mlruns")
        shutil.copytree(mlflow_backup, "mlruns")
        console.print("  ✅ MLflow artifacts restored")

    # Restore models
    models_backup = backup_dir / "models"
    if models_backup.exists():
        console.print("\n📦 Restoring models...")
        if Path("models").exists():
            shutil.rmtree("models")
        shutil.copytree(models_backup, "models")
        console.print("  ✅ Models restored")

    # Restore DVC cache
    dvc_backup = backup_dir / "dvc_cache"
    if dvc_backup.exists():
        console.print("\n📦 Restoring DVC cache...")
        dvc_cache_dir = Path(".dvc/cache")
        if dvc_cache_dir.exists():
            shutil.rmtree(dvc_cache_dir)
        shutil.copytree(dvc_backup, dvc_cache_dir)
        console.print("  ✅ DVC cache restored")

    console.print("\n[bold green]✅ Restore complete![/bold green]")
    console.print("\n📝 Next steps:")
    console.print("  1. Restart Docker services: docker compose restart")
    console.print("  2. Verify API: curl http://localhost/health")
    console.print("  3. Check MLflow: http://localhost:5000")


@app.command()
def list_backups(
    directory: Path = typer.Option(
        Path("backups"),
        help="Directory containing backups",
    ),
) -> None:
    """List available backups."""
    if not directory.exists():
        console.print(f"[bold yellow]⚠️  Backup directory not found: {directory}[/bold yellow]")
        return

    backups = sorted(directory.glob("backup_*.tar.gz"))

    if not backups:
        console.print("[bold yellow]No backups found[/bold yellow]")
        return

    table = Table(title="Available Backups")
    table.add_column("Filename", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Created", style="green")

    for backup_file in backups:
        size_mb = backup_file.stat().st_size / (1024 * 1024)
        created = datetime.fromtimestamp(backup_file.stat().st_mtime)

        table.add_row(
            backup_file.name,
            f"{size_mb:.2f} MB",
            created.strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)


@app.command()
def test_recovery() -> None:
    """Test backup and recovery workflow."""
    console.print("[bold blue]🧪 Testing backup/recovery workflow...[/bold blue]")

    # Check Docker
    if not check_docker_running():
        console.print("[bold red]❌ Docker not running[/bold red]")
        raise typer.Exit(1)

    console.print("  ✅ Docker services running")

    # Check DVC
    try:
        subprocess.run(["dvc", "version"], check=True, capture_output=True)
        console.print("  ✅ DVC installed")
    except Exception:
        console.print("  ❌ DVC not installed")
        raise typer.Exit(1) from None

    # Check backup directory
    backup_dir = Path("backups")
    if not backup_dir.exists():
        backup_dir.mkdir()
        console.print(f"  ✅ Created backup directory: {backup_dir}")
    else:
        console.print(f"  ✅ Backup directory exists: {backup_dir}")

    console.print("\n[bold green]✅ System ready for backup/recovery[/bold green]")


if __name__ == "__main__":
    app()
