#!/usr/bin/env python3

"""Shared helpers for offline drug prioritization scripts."""

from __future__ import annotations

import logging
from collections.abc import Hashable
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DRUGSTONE_RESULT_COLUMNS = [
    "drugId",
    "label",
    "status",
    "drugstoneType",
    "score",
    "hasEdgesTo",
    "isResult",
    "isConnector",
]


def load_module_node_ids(module_path: str | Path) -> list[str]:
    """Load module node IDs from a module node table using only the 'name' column."""
    logger.info("Loading module node table from %s", module_path)
    module_df = pd.read_csv(module_path, sep="\t", dtype=str).fillna("")

    if "name" not in module_df.columns:
        raise ValueError(
            "Module node table is missing required column 'name'. "
            f"Available columns: {list(module_df.columns)}"
        )

    module_node_ids = [node_id for node_id in module_df["name"].tolist() if node_id]
    if not module_node_ids:
        raise ValueError("Module node table does not contain any values in 'name'.")

    logger.info("Loaded %d module node IDs", len(module_node_ids))
    return module_node_ids


def normalize_scores(
    scored_drugs: list[tuple[Hashable, float]],
) -> list[tuple[Hashable, float]]:
    """Normalize drug scores to the Drugst.One-like [0, 1] scale."""
    if not scored_drugs:
        return scored_drugs

    max_score = max(score for _, score in scored_drugs)
    if max_score <= 0:
        return [(drug_id, 0.0) for drug_id, _ in scored_drugs]

    return [(drug_id, float(score) / float(max_score)) for drug_id, score in scored_drugs]


def write_results(result_df: pd.DataFrame, output_path: str | Path) -> None:
    """Write the ranked algorithm-level output table."""
    result_df.to_csv(output_path, index=False)
    logger.info("Wrote %d ranked drugs to %s", len(result_df), output_path)
