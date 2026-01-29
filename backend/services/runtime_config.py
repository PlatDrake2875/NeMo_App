# backend/services/runtime_config.py
"""
Runtime configuration manager for dynamic config updates.
Allows updating Qdrant connection settings without server restart.
"""
from typing import Optional
from urllib.parse import urlparse

from config import logger


class RuntimeConfigManager:
    """Manages runtime configuration that can be updated without restart."""

    _instance: Optional["RuntimeConfigManager"] = None

    def __init__(self):
        self._qdrant_url: Optional[str] = None
        self._qdrant_host: Optional[str] = None
        self._qdrant_port: Optional[int] = None
        self._qdrant_https: bool = False

    @classmethod
    def get_instance(cls) -> "RuntimeConfigManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = RuntimeConfigManager()
        return cls._instance

    def set_qdrant_url(self, url: str) -> dict:
        """
        Parse and set Qdrant URL.

        Args:
            url: Full URL like https://example.com:6333 or http://localhost:6333

        Returns:
            Dict with parsed host, port, and https status
        """
        parsed = urlparse(url)

        self._qdrant_url = url
        self._qdrant_host = parsed.hostname or "localhost"
        self._qdrant_port = parsed.port or (443 if parsed.scheme == "https" else 6333)
        self._qdrant_https = parsed.scheme == "https"

        logger.info(f"Runtime config: Qdrant URL updated to {url}")
        logger.info(
            f"  -> Host: {self._qdrant_host}, Port: {self._qdrant_port}, HTTPS: {self._qdrant_https}"
        )

        return {
            "host": self._qdrant_host,
            "port": self._qdrant_port,
            "https": self._qdrant_https,
            "url": self._qdrant_url,
        }

    def get_qdrant_config(self) -> dict:
        """Get current Qdrant configuration."""
        return {
            "host": self._qdrant_host,
            "port": self._qdrant_port,
            "https": self._qdrant_https,
            "url": self._qdrant_url,
        }

    @property
    def qdrant_host(self) -> Optional[str]:
        return self._qdrant_host

    @property
    def qdrant_port(self) -> Optional[int]:
        return self._qdrant_port

    @property
    def qdrant_https(self) -> bool:
        return self._qdrant_https

    def has_qdrant_override(self) -> bool:
        """Check if Qdrant URL has been overridden at runtime."""
        return self._qdrant_url is not None

    def clear_qdrant_override(self):
        """Clear the runtime Qdrant configuration override."""
        self._qdrant_url = None
        self._qdrant_host = None
        self._qdrant_port = None
        self._qdrant_https = False
        logger.info("Runtime config: Qdrant URL override cleared")


def get_runtime_config() -> RuntimeConfigManager:
    """Get the runtime config manager singleton."""
    return RuntimeConfigManager.get_instance()
