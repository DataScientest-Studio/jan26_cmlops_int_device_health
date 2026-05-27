"""
DagsHub Integration Setup Script

Configures DVC remote storage and MLflow tracking for DagsHub.
Prerequisites:
- DAGSHUB_USER, DAGSHUB_TOKEN, DAGSHUB_REPO in .env file
- DagsHub account and repository created

Usage:
    python scripts/setup_dagshub.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


def check_environment() -> tuple[str, str, str]:
    """Validate required environment variables are set."""
    user = os.getenv("DAGSHUB_USER")
    token = os.getenv("DAGSHUB_TOKEN")
    repo = os.getenv("DAGSHUB_REPO")

    if not all([user, token, repo]):
        print("❌ Missing DagsHub credentials in environment")
        print("\nRequired environment variables:")
        print("  - DAGSHUB_USER")
        print("  - DAGSHUB_TOKEN")
        print("  - DAGSHUB_REPO")
        print("\nAdd these to your .env file or export them in your shell.")
        sys.exit(1)

    return user, token, repo  # type: ignore[return-value]  # sys.exit(1) above ensures non-None


def configure_dvc_remote(user: str, token: str, repo: str) -> None:
    """Configure DagsHub as DVC remote storage."""
    print("\n📦 Configuring DVC remote storage...")

    # DagsHub DVC remote URL format
    remote_url = f"https://dagshub.com/{user}/{repo}.dvc"

    # Check if remote already exists
    result = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True,
        text=True,
    )

    if "dagshub" in result.stdout:
        print("  ℹ️  DagsHub remote already exists, updating...")
        subprocess.run(
            ["dvc", "remote", "modify", "dagshub", "url", remote_url],
            check=True,
        )
    else:
        print("  ➕ Adding new DagsHub remote...")
        subprocess.run(
            ["dvc", "remote", "add", "-d", "dagshub", remote_url],
            check=True,
        )

    # Configure authentication — auth type and username go in shared config;
    # password goes ONLY in .dvc/config.local (git-ignored).
    print("  🔐 Configuring authentication...")
    subprocess.run(
        ["dvc", "remote", "modify", "dagshub", "auth", "basic"],
        check=True,
    )
    subprocess.run(
        ["dvc", "remote", "modify", "dagshub", "user", user],
        check=True,
    )
    # --local writes to .dvc/config.local which is git-ignored — never expose token in .dvc/config
    subprocess.run(
        ["dvc", "remote", "modify", "--local", "dagshub", "password", token],
        check=True,
    )

    print("  ✅ DVC remote configured successfully")
    print(f"     Remote URL: {remote_url}")


def configure_git_remote(user: str, repo: str) -> None:
    """Add DagsHub as git remote (optional)."""
    print("\n🔗 Configuring Git remote...")

    result = subprocess.run(
        ["git", "remote", "-v"],
        capture_output=True,
        text=True,
    )

    if "dagshub" in result.stdout:
        print("  ℹ️  DagsHub git remote already exists")
    else:
        # HTTPS URL for git remote
        git_url = f"https://dagshub.com/{user}/{repo}.git"
        print(f"  ➕ Adding DagsHub git remote: {git_url}")
        subprocess.run(
            ["git", "remote", "add", "dagshub", git_url],
            check=True,
        )
        print("  ✅ Git remote added")
        print("     You can push to DagsHub with: git push dagshub main")


def create_env_template(user: str, repo: str) -> None:
    """Update .env.example with DagsHub configuration."""
    print("\n📝 Updating environment templates...")

    env_example = Path(".env.example")
    if not env_example.exists():
        print("  ⚠️  .env.example not found, skipping")
        return

    # MLflow tracking URI for DagsHub
    mlflow_uri = f"https://dagshub.com/{user}/{repo}.mlflow"

    print(f"  ✅ MLflow tracking URI: {mlflow_uri}")
    print("\n  Add this to your .env file:")
    print(f"     MLFLOW_TRACKING_URI={mlflow_uri}")


def test_dvc_connection(user: str, repo: str) -> None:
    """Test DVC remote connectivity."""
    print("\n🧪 Testing DVC remote connection...")

    try:
        # Try to list DVC tracked files
        result = subprocess.run(
            ["dvc", "status", "-r", "dagshub"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("  ✅ DVC remote connection successful")
        else:
            print(f"  ⚠️  DVC status check returned: {result.stderr}")
            print("     This may be normal if no files are tracked yet")
    except subprocess.TimeoutExpired:
        print("  ⚠️  Connection timeout - check your credentials and network")
    except Exception as e:
        print(f"  ⚠️  Connection test failed: {e}")


def print_next_steps(user: str, repo: str) -> None:
    """Display next steps for user."""
    print("\n" + "=" * 60)
    print("🎉 DagsHub integration configured!")
    print("=" * 60)

    print("\n📋 Next Steps:")
    print("\n1. Add MLflow tracking URI to your .env file:")
    print(f"   MLFLOW_TRACKING_URI=https://dagshub.com/{user}/{repo}.mlflow")

    print("\n2. Track data files with DVC:")
    print("   dvc add data/processed/features.csv")
    print("   git add data/processed/features.csv.dvc .gitignore")
    print("   git commit -m 'Track features with DVC'")

    print("\n3. Push data to DagsHub:")
    print("   dvc push")

    print("\n4. Pull data on another machine:")
    print("   dvc pull")

    print("\n5. Update docker-compose.yml:")
    print("   Set MLFLOW_TRACKING_URI environment variable")

    print("\n6. Restart services:")
    print("   docker-compose down && docker-compose up -d")

    print("\n📖 Documentation:")
    print(f"   Repository: https://dagshub.com/{user}/{repo}")
    print(f"   MLflow: https://dagshub.com/{user}/{repo}.mlflow")
    print(f"   DVC: https://dagshub.com/{user}/{repo}.dvc")


def main() -> None:
    """Main setup workflow."""
    print("=" * 60)
    print("🚀 DagsHub Integration Setup")
    print("=" * 60)

    # Load .env.secrets and any mode-specific env file so that credentials
    # are available even when the variables are not exported in the shell.
    project_root = Path(__file__).parent.parent
    for env_file in (".env.secrets", ".env.local", ".env.cloud"):
        path = project_root / env_file
        if path.exists():
            load_dotenv(path, override=False)  # shell vars take priority

    # Check environment
    user, token, repo = check_environment()
    print("\n✅ Credentials found:")
    print(f"   User: {user}")
    print(f"   Repo: {repo}")
    print(f"   Token: {'*' * len(token)}")

    # Configure remotes
    configure_dvc_remote(user, token, repo)
    configure_git_remote(user, repo)
    create_env_template(user, repo)
    test_dvc_connection(user, repo)

    # Show next steps
    print_next_steps(user, repo)


if __name__ == "__main__":
    main()
