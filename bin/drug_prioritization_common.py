#!/usr/bin/env python3

"""Shared helpers for offline drug prioritization scripts."""

from __future__ import annotations

import ast
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


def write_drug_predictions(
    module_path: str | Path,
    ranking_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Write module-node annotations expanded with ranked drug predictions."""
    logger.info("Loading module node table from %s", module_path)
    module_df = pd.read_csv(module_path, sep="\t", dtype=str).fillna("")

    if "name" not in module_df.columns:
        raise ValueError(
            "Module node table is missing required column 'name'. "
            f"Available columns: {list(module_df.columns)}"
        )

    module_df["symbol"] = module_df["name"]

    drugs_by_node: dict[str, list[dict[str, object]]] = {
        node_id: [] for node_id in module_df["name"].tolist()
    }
    module_node_ids = set(drugs_by_node)

    for _, drug in ranking_df.iterrows():
        targeted_node_ids = [
            str(node_id) for node_id in ast.literal_eval(drug["hasEdgesTo"])
        ]
        drug_annotation = {
            "drug_id": drug["drugId"],
            "score": drug["score"],
            "drug_name": drug["label"],
            "status": drug["status"],
        }
        for node_id in targeted_node_ids:
            if node_id in module_node_ids:
                drugs_by_node[node_id].append(drug_annotation)

    expanded_rows = []
    for _, module_row in module_df.iterrows():
        node_id = module_row["name"]
        drug_rows = sorted(
            drugs_by_node[node_id],
            key=lambda drug: drug["score"],
            reverse=True,
        )
        if not drug_rows:
            expanded_row = module_row.copy()
            expanded_row["drug_id"] = ""
            expanded_row["score"] = ""
            expanded_row["drug_name"] = ""
            expanded_row["status"] = ""
            expanded_rows.append(expanded_row)
            continue

        for drug_row in drug_rows:
            expanded_row = module_row.copy()
            expanded_row["drug_id"] = drug_row["drug_id"]
            expanded_row["score"] = drug_row["score"]
            expanded_row["drug_name"] = drug_row["drug_name"]
            expanded_row["status"] = drug_row["status"]
            expanded_rows.append(expanded_row)

    output_df = pd.DataFrame(expanded_rows)
    output_columns = list(module_df.columns) + [
        "drug_id",
        "score",
        "drug_name",
        "status",
    ]
    output_df.to_csv(output_path, sep="\t", index=False, columns=output_columns)
    logger.info("Wrote %d drug prediction rows to %s", len(output_df), output_path)
