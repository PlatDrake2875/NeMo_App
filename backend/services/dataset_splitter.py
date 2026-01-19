"""
Dataset Splitter - Create train/val/test splits for evaluation datasets.

For rigorous ML evaluation, it's essential to:
- Have clear train/val/test splits
- Ensure no leakage between splits
- Use deterministic splitting for reproducibility
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitInfo:
    """Information about a single split."""

    name: str  # "train", "val", or "test"
    pairs: list[dict[str, Any]]
    pair_count: int
    content_hash: str
    indices: list[int]  # Original indices

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pair_count": self.pair_count,
            "content_hash": self.content_hash,
        }


@dataclass
class SplitResult:
    """Result of splitting a dataset."""

    train: SplitInfo
    val: SplitInfo
    test: SplitInfo
    seed: Optional[int]
    method: str
    original_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train": self.train.to_dict(),
            "val": self.val.to_dict(),
            "test": self.test.to_dict(),
            "seed": self.seed,
            "method": self.method,
            "original_count": self.original_count,
        }

    def get_split(self, name: str) -> SplitInfo:
        """Get a specific split by name."""
        if name == "train":
            return self.train
        elif name == "val":
            return self.val
        elif name == "test":
            return self.test
        else:
            raise ValueError(f"Unknown split: {name}")


class DatasetSplitter:
    """
    Split datasets into train/val/test with various strategies.

    Supports:
    - Random splitting with seed
    - Hash-based deterministic splitting
    - Stratified splitting (future)
    """

    def split_by_ratio(
        self,
        pairs: list[dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> SplitResult:
        """
        Split dataset by ratio using random shuffling.

        Args:
            pairs: List of Q&A pairs to split
            train_ratio: Proportion for training (default 0.7)
            val_ratio: Proportion for validation (default 0.15)
            test_ratio: Proportion for testing (default 0.15)
            seed: Random seed for reproducibility

        Returns:
            SplitResult with train/val/test splits
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Ratios sum to {total}, normalizing...")
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        n = len(pairs)
        indices = np.arange(n)

        # Shuffle with seed
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split indices
        train_indices = indices[:train_end].tolist()
        val_indices = indices[train_end:val_end].tolist()
        test_indices = indices[val_end:].tolist()

        # Create splits
        train_split = self._create_split_info("train", pairs, train_indices)
        val_split = self._create_split_info("val", pairs, val_indices)
        test_split = self._create_split_info("test", pairs, test_indices)

        return SplitResult(
            train=train_split,
            val=val_split,
            test=test_split,
            seed=seed,
            method="ratio_random",
            original_count=n,
        )

    def split_by_hash(
        self,
        pairs: list[dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> SplitResult:
        """
        Split dataset deterministically based on content hash.

        This method assigns each pair to a split based on its content hash,
        ensuring the same pair always goes to the same split regardless of
        dataset order. This is more robust for reproducibility.

        Args:
            pairs: List of Q&A pairs to split
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Seed combined with hash for assignment

        Returns:
            SplitResult with deterministic splits
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.01:
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

        train_indices = []
        val_indices = []
        test_indices = []

        train_threshold = train_ratio
        val_threshold = train_ratio + val_ratio

        for i, pair in enumerate(pairs):
            # Compute content hash
            question = pair.get("question", pair.get("query", ""))
            answer = pair.get("expected_answer", pair.get("ground_truth", ""))
            content = f"{question}|||{answer}|||{seed}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Convert hash to float in [0, 1)
            hash_int = int(content_hash[:16], 16)
            hash_float = hash_int / (16 ** 16)

            # Assign to split based on hash value
            if hash_float < train_threshold:
                train_indices.append(i)
            elif hash_float < val_threshold:
                val_indices.append(i)
            else:
                test_indices.append(i)

        # Create splits
        train_split = self._create_split_info("train", pairs, train_indices)
        val_split = self._create_split_info("val", pairs, val_indices)
        test_split = self._create_split_info("test", pairs, test_indices)

        return SplitResult(
            train=train_split,
            val=val_split,
            test=test_split,
            seed=seed,
            method="content_hash",
            original_count=len(pairs),
        )

    def _create_split_info(
        self,
        name: str,
        all_pairs: list[dict[str, Any]],
        indices: list[int],
    ) -> SplitInfo:
        """Create SplitInfo for a set of indices."""
        pairs = [all_pairs[i] for i in indices]

        # Compute content hash for integrity
        import json
        pairs_json = json.dumps(pairs, sort_keys=True)
        content_hash = hashlib.sha256(pairs_json.encode()).hexdigest()

        return SplitInfo(
            name=name,
            pairs=pairs,
            pair_count=len(pairs),
            content_hash=content_hash,
            indices=indices,
        )

    def assign_split_labels(
        self,
        pairs: list[dict[str, Any]],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        method: Literal["random", "hash"] = "hash",
    ) -> list[dict[str, Any]]:
        """
        Add 'split' field to each pair without separating them.

        Args:
            pairs: List of Q&A pairs
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed
            method: "random" or "hash"

        Returns:
            Pairs with 'split' field added
        """
        if method == "hash":
            result = self.split_by_hash(pairs, train_ratio, val_ratio, test_ratio, seed)
        else:
            result = self.split_by_ratio(pairs, train_ratio, val_ratio, test_ratio, seed)

        # Create index -> split mapping
        split_map = {}
        for idx in result.train.indices:
            split_map[idx] = "train"
        for idx in result.val.indices:
            split_map[idx] = "val"
        for idx in result.test.indices:
            split_map[idx] = "test"

        # Add split field to pairs
        labeled_pairs = []
        for i, pair in enumerate(pairs):
            pair_copy = pair.copy()
            pair_copy["split"] = split_map.get(i, "unknown")
            labeled_pairs.append(pair_copy)

        return labeled_pairs


def enforce_split_isolation(
    pairs: list[dict[str, Any]],
    required_split: Literal["train", "val", "test"],
) -> list[dict[str, Any]]:
    """
    Filter pairs to only include those from a specific split.

    Args:
        pairs: List of Q&A pairs with 'split' field
        required_split: The split to filter for

    Returns:
        Filtered list of pairs

    Raises:
        ValueError: If no pairs have split labels or none match required split
    """
    # Check if pairs have split labels
    labeled_pairs = [p for p in pairs if "split" in p]
    if not labeled_pairs:
        raise ValueError("Pairs do not have split labels. Run assign_split_labels first.")

    # Filter by required split
    filtered = [p for p in pairs if p.get("split") == required_split]
    if not filtered:
        available_splits = set(p.get("split") for p in pairs if "split" in p)
        raise ValueError(
            f"No pairs found for split '{required_split}'. "
            f"Available splits: {available_splits}"
        )

    return filtered
