#!/usr/bin/env python3

"""Perform NetMedPy offline drug prioritization for one disease module."""

from __future__ import annotations

import argparse
import logging
import math
import pickle
import sys
from pathlib import Path

import netmedpy
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drug_prioritization_common import (
    DRUGSTONE_RESULT_COLUMNS,
    load_module_node_ids,
    normalize_scores,
    write_drug_predictions,
    write_results,
)

logger = logging.getLogger(__name__)

PRIORITIZATION_ALGORITHMS = (
    "network_proximity",
    "network_separation",
)
TARGET_GROUP_NAME = "module"
NETMEDPY_PROPERTIES = ["z_score"]


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Perform NetMedPy offline drug prioritization for one disease module "
            "using a prepared PPI, drug-target dictionary, and distance matrix."
        )
    )
    parser.add_argument(
        "--ppi",
        required=True,
        type=Path,
        help="Prepared NetworkX PPI pickle produced by prepare_drug_prioritization_inputs.py.",
    )
    parser.add_argument(
        "--drug-targets",
        required=True,
        type=Path,
        help="Prepared drug-target dictionary pickle produced by prepare_drug_prioritization_inputs.py.",
    )
    parser.add_argument(
        "--distances",
        required=True,
        type=Path,
        help="NetMedPy shortest-path distance matrix produced by precompute_netmedpy_distances.py.",
    )
    parser.add_argument(
        "--module",
        required=True,
        type=Path,
        help="Disease module node TSV. Only the 'name' column is used.",
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=PRIORITIZATION_ALGORITHMS,
        help="NetMedPy prioritization algorithm to use.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        type=str,
        help=(
            "Output prefix. If --ranking-output is omitted, the default ranking file is "
            "written as '<prefix>.<algorithm>.csv'."
        ),
    )
    parser.add_argument(
        "--ranking-output",
        type=Path,
        help="Output CSV path. Defaults to '<prefix>.<algorithm>.csv'.",
    )
    parser.add_argument(
        "--drug-predictions-output",
        type=Path,
        help=(
            "Output drug predictions TSV path. Defaults to "
            "'<prefix>.<algorithm>.drug_predictions.tsv'."
        ),
    )
    parser.add_argument(
        "--drug-background",
        required=True,
        type=Path,
        help="PPI-specific drug background table for labels and status.",
    )
    parser.add_argument(
        "--result-size",
        "--result_size",
        dest="result_size",
        type=int,
        default=None,
        help="Number of ranked drugs to write. If omitted, write all ranked drugs.",
    )
    parser.add_argument(
        "--n-processors",
        type=int,
        default=1,
        help="Number of processors passed to netmedpy.screening.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        default="INFO",
        help="The desired log level.",
    )
    return parser.parse_args()


def load_ppi_network(path: Path) -> nx.Graph:
    """Load the prepared NetworkX PPI."""
    logger.info("Loading prepared PPI from %s", path)
    with path.open("rb") as handle:
        network = pickle.load(handle)

    logger.info(
        "Loaded PPI with %d nodes and %d edges",
        network.number_of_nodes(),
        network.number_of_edges(),
    )
    return network


def load_drug_targets(path: Path) -> dict[str, set[str]]:
    """Load the prepared drug-target dictionary."""
    logger.info("Loading prepared drug-target dictionary from %s", path)
    with path.open("rb") as handle:
        drug_targets = pickle.load(handle)
    logger.info("Loaded drug-target dictionary for %d drugs", len(drug_targets))
    return drug_targets


def load_drug_background(path: Path) -> dict[str, dict[str, str]]:
    """Load drug labels/status from a PPI-specific background table."""
    logger.info("Loading drug background from %s", path)
    background = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "drug_id" not in background.columns:
        raise ValueError(f"Drug background is missing required column 'drug_id': {path}")

    metadata = {}
    for _, row in background.iterrows():
        drug_id = row["drug_id"]
        metadata[drug_id] = {
            "label": row.get("drug_name", "") or drug_id,
            "status": row.get("status", ""),
        }
    return metadata


def run_netmedpy_screening(
    sources: dict[str, set[str]],
    module_nodes: set[str],
    network: nx.Graph,
    distances,
    score: str,
    properties: list[str],
    n_processors: int,
) -> dict[str, pd.DataFrame]:
    """Run NetMedPy screening with one module target group."""
    if not sources:
        raise ValueError("No candidate drugs remain after filtering.")
    if n_processors < 1:
        raise ValueError("--n-processors must be at least 1")

    logger.info(
        "Running NetMedPy screening for %d drugs against %d module nodes with score '%s'",
        len(sources),
        len(module_nodes),
        score,
    )
    return netmedpy.screening(
        sources,
        {TARGET_GROUP_NAME: module_nodes},
        network,
        distances,
        score=score,
        properties=properties,
        n_procs=n_processors,
    )


def collect_metric_values(
    screen_data: dict[str, pd.DataFrame],
    properties: list[str],
) -> dict[str, dict[str, float]]:
    """Transform NetMedPy property tables into drug-keyed metric dictionaries."""
    metric_values: dict[str, dict[str, float]] = {}
    for property_name in properties:
        property_table = screen_data[property_name]
        if TARGET_GROUP_NAME not in property_table.columns:
            raise ValueError(
                f"NetMedPy output for '{property_name}' is missing target column "
                f"'{TARGET_GROUP_NAME}'. Available columns: {list(property_table.columns)}"
            )

        for drug_id, value in property_table[TARGET_GROUP_NAME].items():
            metric_values.setdefault(str(drug_id), {})[property_name] = float(value)
    return metric_values


def score_from_negative_z_score(
    metric_values: dict[str, dict[str, float]],
) -> list[tuple[str, float]]:
    """Rank drugs by ``-z_score`` so lower z-scores become higher scores."""
    scored_drugs = [
        (drug_id, -metrics["z_score"])
        for drug_id, metrics in metric_values.items()
        if "z_score" in metrics and pd.notna(metrics["z_score"])
    ]
    if not scored_drugs:
        raise ValueError("No NetMedPy z-scores found.")

    finite_scores = [score for _, score in scored_drugs if math.isfinite(score)]
    if not finite_scores:
        return [(drug_id, 0.0) for drug_id, _ in scored_drugs]

    min_score = min(finite_scores)
    return [
        (drug_id, score - min_score if math.isfinite(score) else 0.0)
        for drug_id, score in scored_drugs
    ]


def build_result_table(
    metric_values: dict[str, dict[str, float]],
    scored_drugs: list[tuple[str, float]],
    drug_targets: dict[str, set[str]],
    module_nodes: set[str],
    drug_metadata: dict[str, dict[str, str]],
    result_size: int | None,
) -> pd.DataFrame:
    """Build the Drugst.One-like NetMedPy ranked output table.

    ``hasEdgesTo`` contains only module nodes directly targeted by the drug.
    ``isConnector`` is always ``False`` because this output ranks drugs, not
    intermediate connector nodes.
    """
    top_scored_drugs = sorted(scored_drugs, key=lambda item: (-item[1], item[0]))
    if result_size is not None:
        top_scored_drugs = top_scored_drugs[:result_size]

    rows = []
    for drug_id, score in top_scored_drugs:
        metadata = drug_metadata.get(drug_id, {})
        row = {
            "drugId": drug_id,
            "label": metadata.get("label", drug_id),
            "status": metadata.get("status", ""),
            "drugstoneType": "drug",
            "score": float(score),
            "hasEdgesTo": str(sorted(drug_targets.get(drug_id, set()) & module_nodes)),
            "isResult": True,
            "isConnector": False,
        }
        row.update(metric_values.get(drug_id, {}))
        rows.append(row)

    metric_columns = sorted(
        {
            property_name
            for metrics in metric_values.values()
            for property_name in metrics
        }
    )
    return pd.DataFrame(
        rows,
        columns=DRUGSTONE_RESULT_COLUMNS + metric_columns,
    )


def main(args) -> None:
    """Coordinate argument parsing, NetMedPy screening, and output writing."""
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    ranking_output_path = args.ranking_output or Path(f"{args.prefix}.{args.algorithm}.csv")
    drug_predictions_output_path = args.drug_predictions_output or Path(
        f"{args.prefix}.{args.algorithm}.drug_predictions.tsv"
    )

    network = load_ppi_network(args.ppi)
    drug_targets = load_drug_targets(args.drug_targets)
    module_node_ids = load_module_node_ids(args.module)
    module_nodes = set(module_node_ids)
    logger.info("Loading NetMedPy distances from %s", args.distances)
    distances = netmedpy.load_distances(str(args.distances))
    drug_metadata = load_drug_background(args.drug_background)
    score = "proximity" if args.algorithm == "network_proximity" else "separation_z_score"

    screen_data = run_netmedpy_screening(
        sources=drug_targets,
        module_nodes=module_nodes,
        network=network,
        distances=distances,
        score=score,
        properties=NETMEDPY_PROPERTIES,
        n_processors=args.n_processors,
    )
    metric_values = collect_metric_values(screen_data, NETMEDPY_PROPERTIES)
    scored_drugs = score_from_negative_z_score(metric_values=metric_values)
    scored_drugs = normalize_scores(scored_drugs)

    result_df = build_result_table(
        metric_values=metric_values,
        scored_drugs=scored_drugs,
        drug_targets=drug_targets,
        module_nodes=module_nodes,
        drug_metadata=drug_metadata,
        result_size=args.result_size,
    )
    write_results(result_df, ranking_output_path)
    write_drug_predictions(
        module_path=args.module,
        ranking_df=result_df,
        output_path=drug_predictions_output_path,
    )


if __name__ == "__main__":
    # Proximity sample argv for local debugging in the script directory:
    # sys.argv = [
    #     "drug_prioritization_netmedpy.py",
    #     "--ppi",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.netmedpy_ppi.pkl",
    #     "--drug-targets",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.netmedpy_drug_targets.pkl",
    #     "--distances",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.netmedpy_shortest_paths.pkl.npz",
    #     "--drug-background",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.drug_background.tsv",
    #     "--module",
    #     "../../data/modules/ensembl/seeds_ensembl.string.human_links_v12_0_min700.Ensembl.diamond.nodes.tsv",
    #     "--algorithm",
    #     "network_proximity",
    #     "--prefix",
    #     "seeds_ensembl.string.human_links_v12_0_min700.Ensembl.diamond",
    #     "--n-processors",
    #     "6",
    #     "-l",
    #     "DEBUG",
    # ]

    # Separation sample argv for local debugging in the script directory:
    # sys.argv = [
    #     "drug_prioritization_netmedpy.py",
    #     "--ppi",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.netmedpy_ppi.pkl",
    #     "--drug-targets",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.netmedpy_drug_targets.pkl",
    #     "--distances",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.netmedpy_shortest_paths.pkl.npz",
    #     "--drug-background",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.drug_background.tsv",
    #     "--module",
    #     "../../data/modules/ensembl/seeds_ensembl.string.human_links_v12_0_min700.Ensembl.diamond.nodes.tsv",
    #     "--algorithm",
    #     "network_separation",
    #     "--prefix",
    #     "seeds_ensembl.string.human_links_v12_0_min700.Ensembl.diamond",
    #     "--n-processors",
    #     "6",
    #     "-l",
    #     "DEBUG",
    # ]
    main(parse_args())
