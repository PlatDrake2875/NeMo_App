"""Entry point for the benchmark CLI.

Usage:
    python -m scripts.benchmark <command> [options]
"""

from scripts.benchmark.cli import app

if __name__ == "__main__":
    app()
