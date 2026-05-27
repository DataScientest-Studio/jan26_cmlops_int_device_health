# Stub for airflow.configuration.
# Apache Airflow 2.11.0 moved configuration into internal modules.
# This stub satisfies imports from installed airflow operators/models.
from __future__ import annotations

import os


class _StubConf:
    """Minimal stub for airflow.configuration.conf used by operators."""

    def get(self, section: str, key: str, fallback: str = "") -> str:
        env_key = f"AIRFLOW__{section.upper()}__{key.upper()}"
        return os.environ.get(env_key, fallback)

    def getint(self, section: str, key: str, fallback: int = 0) -> int:
        try:
            return int(self.get(section, key, str(fallback)))
        except (ValueError, TypeError):
            return fallback

    def getfloat(self, section: str, key: str, fallback: float = 0.0) -> float:
        try:
            return float(self.get(section, key, str(fallback)))
        except (ValueError, TypeError):
            return fallback

    def getboolean(self, section: str, key: str, fallback: bool = False) -> bool:
        val = self.get(section, key, str(fallback)).lower()
        return val in ("1", "true", "yes", "on")

    def has_option(self, section: str, key: str) -> bool:
        env_key = f"AIRFLOW__{section.upper()}__{key.upper()}"
        return env_key in os.environ

    def __contains__(self, item: object) -> bool:
        return False


conf = _StubConf()
