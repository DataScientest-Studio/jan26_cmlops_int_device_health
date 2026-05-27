#!/usr/bin/env python3
"""
Start the FastAPI application for manual testing.

This script starts the API server with the bootstrap model and database.

Usage:
    python scripts/start_api.py

The API will be available at:
    - http://localhost:8000
    - API docs: http://localhost:8000/docs
    - Alternative docs: http://localhost:8000/redoc
"""

import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def check_prerequisites():
    """Check if model and database are set up."""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "bootstrap_model.pkl"

    if not model_path.exists():
        console.print(
            Panel(
                "[bold red]Model not found![/bold red]\n\n"
                f"Expected: {model_path}\n\n"
                "Please train the bootstrap model first:\n"
                "  [cyan]python scripts/bootstrap_model.py[/cyan]",
                border_style="red",
            )
        )
        return False

    console.print(f"[green]✓[/green] Model found: {model_path}")
    return True


def start_server():
    """Start the FastAPI application."""
    console.print(
        Panel.fit(
            "[bold cyan]Starting MLOps Device Health API[/bold cyan]\n\n"
            "API will be available at:\n"
            "  • Main: [link]http://localhost:8000[/link]\n"
            "  • Docs: [link]http://localhost:8000/docs[/link]\n"
            "  • ReDoc: [link]http://localhost:8000/redoc[/link]\n\n"
            "Press [bold red]Ctrl+C[/bold red] to stop",
            border_style="cyan",
        )
    )

    try:
        subprocess.run(
            [
                "uvicorn",
                "src.api.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        console.print(f"\n[red]Server error: {e}[/red]")
        sys.exit(1)


def main():
    """Main entry point."""
    if not check_prerequisites():
        sys.exit(1)

    start_server()


if __name__ == "__main__":
    main()
