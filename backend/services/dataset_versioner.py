"""
Dataset Versioner - Track versions and changes to evaluation datasets.

For publication-grade ML research, datasets should be:
- Immutable once published
- Version-controlled with clear lineage
- Comparable across versions

This module provides versioning capabilities for evaluation datasets.
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Directory for storing versioned datasets
VERSIONED_DATASETS_DIR = Path("data/versioned_datasets")
VERSIONED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VersionInfo:
    """Information about a dataset version."""

    version: int
    created_at: str
    content_hash: str
    pair_count: int
    description: str
    parent_version: Optional[int] = None
    changes: list[str] = field(default_factory=list)
    is_published: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "content_hash": self.content_hash,
            "pair_count": self.pair_count,
            "description": self.description,
            "parent_version": self.parent_version,
            "changes": self.changes,
            "is_published": self.is_published,
        }


@dataclass
class VersionComparison:
    """Result of comparing two dataset versions."""

    version_a: int
    version_b: int
    pairs_added: int
    pairs_removed: int
    pairs_modified: int
    added_questions: list[str] = field(default_factory=list)
    removed_questions: list[str] = field(default_factory=list)
    modified_questions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_a": self.version_a,
            "version_b": self.version_b,
            "pairs_added": self.pairs_added,
            "pairs_removed": self.pairs_removed,
            "pairs_modified": self.pairs_modified,
            "added_questions": self.added_questions[:20],  # Limit output
            "removed_questions": self.removed_questions[:20],
            "modified_questions": self.modified_questions[:20],
        }


class DatasetVersioner:
    """
    Manage versions of evaluation datasets.

    Provides:
    - Version creation with change tracking
    - Version comparison
    - Publishing (marking as immutable)
    - Version history
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or VERSIONED_DATASETS_DIR

    def _get_dataset_dir(self, dataset_id: str) -> Path:
        """Get directory for a dataset's versions."""
        dataset_dir = self.base_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _get_version_path(self, dataset_id: str, version: int) -> Path:
        """Get path to a specific version file."""
        return self._get_dataset_dir(dataset_id) / f"v{version}.json"

    def _get_metadata_path(self, dataset_id: str) -> Path:
        """Get path to dataset metadata file."""
        return self._get_dataset_dir(dataset_id) / "metadata.json"

    def _load_metadata(self, dataset_id: str) -> dict[str, Any]:
        """Load dataset metadata."""
        path = self._get_metadata_path(dataset_id)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {
            "dataset_id": dataset_id,
            "current_version": 0,
            "versions": [],
            "is_published": False,
        }

    def _save_metadata(self, dataset_id: str, metadata: dict[str, Any]) -> None:
        """Save dataset metadata."""
        path = self._get_metadata_path(dataset_id)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _compute_content_hash(self, pairs: list[dict[str, Any]]) -> str:
        """Compute hash of dataset content."""
        pairs_json = json.dumps(pairs, sort_keys=True)
        return hashlib.sha256(pairs_json.encode()).hexdigest()

    def create_version(
        self,
        dataset_id: str,
        pairs: list[dict[str, Any]],
        description: str = "",
        changes: Optional[list[str]] = None,
    ) -> VersionInfo:
        """
        Create a new version of a dataset.

        Args:
            dataset_id: Dataset identifier
            pairs: Q&A pairs for this version
            description: Description of this version
            changes: List of changes from previous version

        Returns:
            VersionInfo for the new version

        Raises:
            ValueError: If dataset is published (immutable)
        """
        metadata = self._load_metadata(dataset_id)

        if metadata.get("is_published"):
            raise ValueError(
                f"Dataset {dataset_id} is published and cannot be modified. "
                "Create a new dataset or fork to make changes."
            )

        # Increment version
        current_version = metadata.get("current_version", 0)
        new_version = current_version + 1

        # Compute content hash
        content_hash = self._compute_content_hash(pairs)

        # Check if content is actually different from last version
        if current_version > 0:
            last_version_info = metadata["versions"][-1]
            if last_version_info["content_hash"] == content_hash:
                logger.info(f"Content unchanged from v{current_version}")
                return VersionInfo(**last_version_info)

        # Create version info
        version_info = VersionInfo(
            version=new_version,
            created_at=datetime.now().isoformat(),
            content_hash=content_hash,
            pair_count=len(pairs),
            description=description,
            parent_version=current_version if current_version > 0 else None,
            changes=changes or [],
            is_published=False,
        )

        # Save version data
        version_data = {
            "version_info": version_info.to_dict(),
            "pairs": pairs,
        }
        version_path = self._get_version_path(dataset_id, new_version)
        with open(version_path, "w") as f:
            json.dump(version_data, f, indent=2)

        # Update metadata
        metadata["current_version"] = new_version
        metadata["versions"].append(version_info.to_dict())
        self._save_metadata(dataset_id, metadata)

        logger.info(f"Created version {new_version} for dataset {dataset_id}")
        return version_info

    def get_version(
        self,
        dataset_id: str,
        version: Optional[int] = None,
    ) -> tuple[VersionInfo, list[dict[str, Any]]]:
        """
        Get a specific version of a dataset.

        Args:
            dataset_id: Dataset identifier
            version: Version number (None for latest)

        Returns:
            Tuple of (version_info, pairs)

        Raises:
            ValueError: If version doesn't exist
        """
        metadata = self._load_metadata(dataset_id)

        if version is None:
            version = metadata.get("current_version", 0)

        if version == 0:
            raise ValueError(f"No versions exist for dataset {dataset_id}")

        version_path = self._get_version_path(dataset_id, version)
        if not version_path.exists():
            raise ValueError(f"Version {version} not found for dataset {dataset_id}")

        with open(version_path) as f:
            data = json.load(f)

        version_info = VersionInfo(**data["version_info"])
        pairs = data["pairs"]

        return version_info, pairs

    def list_versions(self, dataset_id: str) -> list[VersionInfo]:
        """
        List all versions of a dataset.

        Args:
            dataset_id: Dataset identifier

        Returns:
            List of version info objects
        """
        metadata = self._load_metadata(dataset_id)
        return [VersionInfo(**v) for v in metadata.get("versions", [])]

    def publish(self, dataset_id: str) -> VersionInfo:
        """
        Publish a dataset, marking it as immutable.

        Once published, no new versions can be created.
        This is suitable for evaluation datasets used in papers.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Version info for the published version

        Raises:
            ValueError: If no versions exist
        """
        metadata = self._load_metadata(dataset_id)

        current_version = metadata.get("current_version", 0)
        if current_version == 0:
            raise ValueError(f"Cannot publish dataset {dataset_id}: no versions exist")

        # Mark as published
        metadata["is_published"] = True
        metadata["published_at"] = datetime.now().isoformat()
        metadata["published_version"] = current_version

        # Update the version's published flag
        for v in metadata["versions"]:
            if v["version"] == current_version:
                v["is_published"] = True

        self._save_metadata(dataset_id, metadata)

        # Also update the version file
        version_path = self._get_version_path(dataset_id, current_version)
        with open(version_path) as f:
            version_data = json.load(f)
        version_data["version_info"]["is_published"] = True
        with open(version_path, "w") as f:
            json.dump(version_data, f, indent=2)

        logger.info(f"Published dataset {dataset_id} at version {current_version}")

        return VersionInfo(**metadata["versions"][-1])

    def compare_versions(
        self,
        dataset_id: str,
        version_a: int,
        version_b: int,
    ) -> VersionComparison:
        """
        Compare two versions of a dataset.

        Args:
            dataset_id: Dataset identifier
            version_a: First version to compare
            version_b: Second version to compare

        Returns:
            Comparison result showing differences
        """
        _, pairs_a = self.get_version(dataset_id, version_a)
        _, pairs_b = self.get_version(dataset_id, version_b)

        # Build question -> answer maps
        def build_map(pairs):
            return {
                p.get("question", p.get("query", "")): p.get("expected_answer", p.get("ground_truth", ""))
                for p in pairs
            }

        map_a = build_map(pairs_a)
        map_b = build_map(pairs_b)

        questions_a = set(map_a.keys())
        questions_b = set(map_b.keys())

        # Find differences
        added = questions_b - questions_a
        removed = questions_a - questions_b
        common = questions_a & questions_b

        # Find modified (same question, different answer)
        modified = [q for q in common if map_a[q] != map_b[q]]

        return VersionComparison(
            version_a=version_a,
            version_b=version_b,
            pairs_added=len(added),
            pairs_removed=len(removed),
            pairs_modified=len(modified),
            added_questions=list(added),
            removed_questions=list(removed),
            modified_questions=modified,
        )

    def fork(
        self,
        source_dataset_id: str,
        new_dataset_id: Optional[str] = None,
        version: Optional[int] = None,
    ) -> str:
        """
        Fork a dataset to create a new independent copy.

        Useful when you want to modify a published dataset.

        Args:
            source_dataset_id: Dataset to fork from
            new_dataset_id: ID for the new dataset (auto-generated if None)
            version: Version to fork from (latest if None)

        Returns:
            ID of the new forked dataset
        """
        # Get source version
        version_info, pairs = self.get_version(source_dataset_id, version)

        # Generate new ID
        if new_dataset_id is None:
            new_dataset_id = f"{source_dataset_id}_fork_{str(uuid.uuid4())[:8]}"

        # Create first version of forked dataset
        self.create_version(
            new_dataset_id,
            pairs,
            description=f"Forked from {source_dataset_id} v{version_info.version}",
            changes=[f"Initial fork from {source_dataset_id}"],
        )

        logger.info(f"Forked {source_dataset_id} v{version_info.version} to {new_dataset_id}")
        return new_dataset_id


def add_lineage_to_pairs(
    pairs: list[dict[str, Any]],
    source_chunk_ids: Optional[list[str]] = None,
    generation_model: Optional[str] = None,
    created_from: str = "generation",
) -> list[dict[str, Any]]:
    """
    Add lineage information to Q&A pairs.

    Args:
        pairs: List of Q&A pairs
        source_chunk_ids: IDs of source chunks (if applicable)
        generation_model: Model used for generation
        created_from: How pairs were created ("generation", "import", "manual")

    Returns:
        Pairs with lineage information added
    """
    import hashlib
    from datetime import datetime

    result = []
    for i, pair in enumerate(pairs):
        pair_copy = pair.copy()

        # Compute content hash for chunk if available
        chunk_content = pair.get("_source_chunk_content")
        chunk_hash = None
        if chunk_content:
            chunk_hash = hashlib.sha256(chunk_content.encode()).hexdigest()[:16]

        lineage = {
            "source_chunk_id": source_chunk_ids[i] if source_chunk_ids and i < len(source_chunk_ids) else None,
            "generation_model": generation_model,
            "chunk_content_hash": chunk_hash,
            "created_from": created_from,
            "created_at": datetime.now().isoformat(),
        }

        pair_copy["lineage"] = lineage
        result.append(pair_copy)

    return result
