"""
Migration Utilities - Migrate old evaluation data formats to new format.

This module provides utilities to:
- Migrate old evaluation results to new schema with metadata
- Add content hashes to existing datasets
- Update old metric results with version info
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default directories
EVAL_DATASETS_DIR = Path("data/evaluation_datasets")
EVAL_RUNS_DIR = Path("data/evaluation_runs")


def migrate_dataset_add_hashes(
    dataset_path: Path,
    backup: bool = True,
) -> dict[str, Any]:
    """
    Migrate a dataset to add content hashes to pairs.

    Args:
        dataset_path: Path to the dataset JSON file
        backup: Whether to create a backup before migration

    Returns:
        Migrated dataset dict
    """
    with open(dataset_path) as f:
        dataset = json.load(f)

    if backup:
        backup_path = dataset_path.with_suffix(".json.bak")
        with open(backup_path, "w") as f:
            json.dump(dataset, f, indent=2)
        logger.info(f"Created backup at {backup_path}")

    # Add content hashes to pairs
    pairs = dataset.get("pairs", [])
    for pair in pairs:
        question = pair.get("question", pair.get("query", ""))
        answer = pair.get("expected_answer", pair.get("ground_truth", ""))

        if "content_hash" not in pair:
            normalized = f"{question.lower().strip()}|||{answer.lower().strip()}"
            pair["content_hash"] = hashlib.sha256(normalized.encode()).hexdigest()

    # Add overall dataset hash
    pairs_json = json.dumps(pairs, sort_keys=True)
    dataset["content_hash"] = hashlib.sha256(pairs_json.encode()).hexdigest()

    # Add migration metadata
    if "migration_info" not in dataset:
        dataset["migration_info"] = {}
    dataset["migration_info"]["content_hashes_added"] = datetime.now().isoformat()

    # Save migrated dataset
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)

    logger.info(f"Migrated dataset {dataset_path}: added content hashes to {len(pairs)} pairs")
    return dataset


def migrate_run_add_metadata(
    run_path: Path,
    backup: bool = True,
) -> dict[str, Any]:
    """
    Migrate an evaluation run to add metadata fields.

    Args:
        run_path: Path to the run JSON file
        backup: Whether to create a backup before migration

    Returns:
        Migrated run dict
    """
    with open(run_path) as f:
        run_data = json.load(f)

    if backup:
        backup_path = run_path.with_suffix(".json.bak")
        with open(backup_path, "w") as f:
            json.dump(run_data, f, indent=2)
        logger.info(f"Created backup at {backup_path}")

    # Add metadata if not present
    if "metadata" not in run_data:
        config = run_data.get("config", {})
        run_data["metadata"] = {
            "llm_model": "unknown",  # Not tracked in old format
            "llm_model_version": None,
            "llm_temperature": config.get("temperature", 0.1),
            "llm_seed": None,
            "embedder_model": config.get("embedder", "sentence-transformers/all-MiniLM-L6-v2"),
            "embedder_model_version": None,
            "metric_versions": {
                "answer_correctness": "1.0.0",  # Old version
                "faithfulness": "1.0.0",
                "context_precision": "1.0.0",
                "relevancy": "1.0.0",
            },
            "dataset_content_hash": None,
            "determinism_guaranteed": False,
            "determinism_warnings": ["Migrated from old format - seed not tracked"],
            "run_timestamp": run_data.get("created_at", datetime.now().isoformat()),
        }

    # Add migration info
    if "migration_info" not in run_data:
        run_data["migration_info"] = {}
    run_data["migration_info"]["metadata_added"] = datetime.now().isoformat()

    # Save migrated run
    with open(run_path, "w") as f:
        json.dump(run_data, f, indent=2)

    logger.info(f"Migrated run {run_path}: added metadata")
    return run_data


def migrate_all_datasets(
    datasets_dir: Optional[Path] = None,
    backup: bool = True,
) -> list[str]:
    """
    Migrate all datasets in a directory.

    Args:
        datasets_dir: Directory containing datasets (default: EVAL_DATASETS_DIR)
        backup: Whether to create backups

    Returns:
        List of migrated dataset IDs
    """
    datasets_dir = datasets_dir or EVAL_DATASETS_DIR
    migrated = []

    for path in datasets_dir.glob("*.json"):
        if path.suffix == ".bak":
            continue
        try:
            migrate_dataset_add_hashes(path, backup)
            migrated.append(path.stem)
        except Exception as e:
            logger.error(f"Failed to migrate {path}: {e}")

    logger.info(f"Migrated {len(migrated)} datasets")
    return migrated


def migrate_all_runs(
    runs_dir: Optional[Path] = None,
    backup: bool = True,
) -> list[str]:
    """
    Migrate all evaluation runs in a directory.

    Args:
        runs_dir: Directory containing runs (default: EVAL_RUNS_DIR)
        backup: Whether to create backups

    Returns:
        List of migrated run IDs
    """
    runs_dir = runs_dir or EVAL_RUNS_DIR
    migrated = []

    for path in runs_dir.glob("*.json"):
        if path.suffix == ".bak":
            continue
        try:
            migrate_run_add_metadata(path, backup)
            migrated.append(path.stem)
        except Exception as e:
            logger.error(f"Failed to migrate {path}: {e}")

    logger.info(f"Migrated {len(migrated)} runs")
    return migrated


def check_migration_status() -> dict[str, Any]:
    """
    Check migration status of all datasets and runs.

    Returns:
        Status dict with counts of migrated/unmigrated items
    """
    datasets_needing_migration = 0
    datasets_migrated = 0
    runs_needing_migration = 0
    runs_migrated = 0

    # Check datasets
    for path in EVAL_DATASETS_DIR.glob("*.json"):
        if path.suffix == ".bak":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if "content_hash" in data and "migration_info" in data:
                datasets_migrated += 1
            else:
                datasets_needing_migration += 1
        except Exception:
            datasets_needing_migration += 1

    # Check runs
    for path in EVAL_RUNS_DIR.glob("*.json"):
        if path.suffix == ".bak":
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if "metadata" in data and "migration_info" in data:
                runs_migrated += 1
            else:
                runs_needing_migration += 1
        except Exception:
            runs_needing_migration += 1

    return {
        "datasets": {
            "migrated": datasets_migrated,
            "needs_migration": datasets_needing_migration,
            "total": datasets_migrated + datasets_needing_migration,
        },
        "runs": {
            "migrated": runs_migrated,
            "needs_migration": runs_needing_migration,
            "total": runs_migrated + runs_needing_migration,
        },
        "all_migrated": datasets_needing_migration == 0 and runs_needing_migration == 0,
    }


def run_full_migration(backup: bool = True) -> dict[str, Any]:
    """
    Run full migration of all datasets and runs.

    Args:
        backup: Whether to create backups

    Returns:
        Migration results
    """
    logger.info("Starting full migration...")

    datasets = migrate_all_datasets(backup=backup)
    runs = migrate_all_runs(backup=backup)

    status = check_migration_status()

    return {
        "datasets_migrated": datasets,
        "runs_migrated": runs,
        "status": status,
    }
