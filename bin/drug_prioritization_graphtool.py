#!/usr/bin/env python3

"""Perform graph-tool offline drug prioritization for one disease module."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import graph_tool.all as gt
import numpy as np
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
    "trustrank",
    "harmonic_centrality",
    "degree",
)
BIOLOGICAL_NODE_TYPE = "protein/gene"
DRUG_NODE_TYPE = "drug"
PDI_EDGE_TYPE = "protein-drug"


def parse_args():
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Perform graph-tool offline drug prioritization for one disease module by "
            "scoring drug nodes in a merged heterogeneous PPI + PDI graph."
        )
    )

    parser.add_argument(
        "--drug-prioritization-graph",
        "--merged-graph",
        "--graph",
        dest="drug_prioritization_graph",
        required=True,

        help=(
            "Merged heterogeneous PPI + PDI graph-tool .gt file produced by "
            "prepare_drug_prioritization_inputs.py."
        ),
    )
    parser.add_argument(
        "--module",
        required=True,
        help=(
            "Disease module node TSV. Only the 'name' column is used."
        ),
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=PRIORITIZATION_ALGORITHMS,
        help="Drug prioritization algorithm to use.",
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
        "--includeIndirectDrugs",
        action="store_true",
        help=(
            "Keep indirect drugs in the merged graph. If omitted, keep only "
            "drug-protein edges that touch a module protein/gene and remove "
            "drugs without any remaining drug-protein edges."
        ),
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
        "--damping-factor",
        "--damping_factor",
        dest="damping_factor",
        type=float,
        default=0.85,
        help="Damping factor for TrustRank / personalized PageRank.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        default="INFO",
        help="The desired log level.",
    )
    return parser.parse_args()


def load_prioritization_graph(graph_path: str) -> gt.Graph:
    """Load the prepared merged heterogeneous PPI + PDI graph."""
    logger.info("Loading merged prioritization graph from %s", graph_path)
    graph = gt.load_graph(graph_path)

    logger.info(
        "Merged graph contains %d vertices and %d edges",
        graph.num_vertices(),
        graph.num_edges(),
    )
    return graph


def get_module_vertex_indices(
    graph: gt.Graph,
    module_node_ids: list[str],
) -> list[int]:
    """Map module node IDs onto biological vertices of the merged graph."""
    biological_vertices = gt.find_vertex(graph, graph.vp["type"], BIOLOGICAL_NODE_TYPE)
    biological_vertex_index: dict[str, int] = {}

    for vertex in biological_vertices:
        node_id = str(graph.vp["id"][vertex]).strip()
        biological_vertex_index[node_id] = int(vertex)

    module_vertex_indices = [
        biological_vertex_index[node_id]
        for node_id in module_node_ids
    ]

    logger.info(
        "Mapped %d/%d module node IDs onto the merged prioritization graph",
        len(module_vertex_indices),
        len(module_node_ids),
    )
    return module_vertex_indices


def filter_direct_drugs(
    graph: gt.Graph,
    module_vertex_indices: list[int],
) -> list[int]:
    """Remove non-direct protein-drug edges in place and return direct drug vertices."""
    edge_keep = graph.new_edge_property("bool")
    edge_keep.a = False
    direct_drug_vertices: set[int] = set()
    pdi_edges = list(gt.find_edge(graph, graph.ep["type"], PDI_EDGE_TYPE))

    for module_vertex_index in module_vertex_indices:
        module_vertex = graph.vertex(module_vertex_index)
        for edge in module_vertex.all_edges():
            if str(graph.ep["type"][edge]).strip() == PDI_EDGE_TYPE:
                source = int(edge.source())
                target = int(edge.target())
                other_vertex_index = target if source == module_vertex_index else source
                edge_keep[edge] = True
                direct_drug_vertices.add(other_vertex_index)

    edges_to_remove = [edge for edge in pdi_edges if not edge_keep[edge]]
    graph.set_fast_edge_removal(True)
    for edge in edges_to_remove:
        graph.remove_edge(edge)
    graph.set_fast_edge_removal(False)

    logger.info(
        "Direct-only filtering kept %d protein-drug edges and removed %d protein-drug edges",
        len(pdi_edges) - len(edges_to_remove),
        len(edges_to_remove),
    )
    logger.info(
        "Direct-only filtering kept %d direct drugs",
        len(direct_drug_vertices),
    )
    return sorted(direct_drug_vertices)


def get_candidate_drug_vertices(graph: gt.Graph) -> list[int]:
    """Return all drug vertices in the prepared working graph."""
    drug_vertices = [
        int(vertex)
        for vertex in graph.vertices()
        if str(graph.vp["type"][vertex]).strip() == DRUG_NODE_TYPE
    ]
    if drug_vertices:
        logger.info("Working graph contains %d candidate drug vertices", len(drug_vertices))
    else:
        logger.warning("Working graph does not contain any candidate drug vertices")
    return drug_vertices


def run_trustrank(
    graph: gt.Graph,
    module_vertex_indices: list[int],
    drug_vertex_indices: list[int],
    damping_factor: float,
) -> list[tuple[int, float]]:
    """Run personalized PageRank / TrustRank seeded on all module nodes."""
    personalization = graph.new_vertex_property("double")
    personalization.a = 0.0

    seed_weight = 1.0 / len(module_vertex_indices)
    for vertex_index in module_vertex_indices:
        personalization[graph.vertex(vertex_index)] = seed_weight

    scores = gt.pagerank(graph, damping=damping_factor, pers=personalization)
    return [(vertex_index, float(scores[graph.vertex(vertex_index)])) for vertex_index in drug_vertex_indices]


def run_degree(
    graph: gt.Graph,
    module_vertex_indices: list[int],
    drug_vertex_indices: list[int],
) -> list[tuple[int, float]]:
    """Score each drug by the number of adjacent module proteins/genes."""
    module_vertex_index_set = set(module_vertex_indices)
    scored_drugs: list[tuple[int, float]] = []

    for drug_vertex_index in drug_vertex_indices:
        score = 0.0
        for neighbor in graph.vertex(drug_vertex_index).all_neighbors():
            if int(neighbor) in module_vertex_index_set:
                score += 1.0
        scored_drugs.append((drug_vertex_index, score))

    return scored_drugs


def run_harmonic_centrality(
    graph: gt.Graph,
    module_vertex_indices: list[int],
    drug_vertex_indices: list[int],
) -> list[tuple[int, float]]:
    """Score drugs by seed-normalized harmonic centrality to module nodes."""
    candidate_drug_vertices = [graph.vertex(vertex_index) for vertex_index in drug_vertex_indices]
    harmonic_scores = np.zeros(len(drug_vertex_indices), dtype=float)
    max_reachable_distance = graph.num_vertices() - 1

    # Iterate over each seed and add its reciprocal distance contribution.
    for seed_vertex_index in module_vertex_indices:
        drug_distances = np.asarray(
            gt.shortest_distance(
                graph,
                source=graph.vertex(seed_vertex_index),
                target=candidate_drug_vertices,
            ),
            dtype=float,
        )
        # In an unweighted graph, reachable paths have length <= N - 1.
        # graph-tool reports unreachable targets as a larger sentinel value.
        reachable = (
            np.isfinite(drug_distances)
            & (drug_distances > 0)
            & (drug_distances <= max_reachable_distance)
        )
        harmonic_scores[reachable] += 1.0 / drug_distances[reachable]

    harmonic_scores /= len(module_vertex_indices)

    return [
        (drug_vertex_index, float(score))
        for drug_vertex_index, score in zip(drug_vertex_indices, harmonic_scores)
    ]


def get_drug_neighbor_ids(
    graph: gt.Graph,
    drug_vertex_index: int,
    module_vertex_indices: set[int],
) -> list[str]:
    """Return module nodes directly targeted by one drug."""
    neighbor_ids = {
        str(graph.vp["id"][neighbor]).strip()
        for neighbor in graph.vertex(drug_vertex_index).all_neighbors()
        if int(neighbor) in module_vertex_indices
    }
    return sorted(node_id for node_id in neighbor_ids if node_id)


def get_drug_label(graph: gt.Graph, drug_vertex_index: int) -> str:
    """Return the best available drug label."""
    label = str(graph.vp["name"][graph.vertex(drug_vertex_index)]).strip()
    if label:
        return label
    return str(graph.vp["id"][graph.vertex(drug_vertex_index)]).strip()


def get_drug_status(graph: gt.Graph, drug_vertex_index: int) -> str:
    """Return the drug status string stored on the prepared merged graph."""
    if "status" not in graph.vp:
        return ""
    return str(graph.vp["status"][graph.vertex(drug_vertex_index)]).strip()


def build_result_table(
    graph: gt.Graph,
    scored_drugs: list[tuple[int, float]],
    module_vertex_indices: list[int],
    result_size: int | None,
) -> pd.DataFrame:
    """Build the algorithm-level ranked output table.

    ``hasEdgesTo`` contains only module nodes directly targeted by the drug.
    ``isConnector`` is always ``False`` because this output ranks drugs, not
    intermediate connector nodes.
    """
    top_scored_drugs = sorted(
        scored_drugs,
        key=lambda item: (-item[1], str(graph.vp["id"][graph.vertex(item[0])]).strip()),
    )
    if result_size is not None:
        top_scored_drugs = top_scored_drugs[:result_size]

    result_rows = []
    module_vertex_index_set = set(module_vertex_indices)
    for drug_vertex_index, score in top_scored_drugs:
        result_rows.append(
            {
                "drugId": str(graph.vp["id"][graph.vertex(drug_vertex_index)]).strip(),
                "label": get_drug_label(graph, drug_vertex_index),
                "status": get_drug_status(graph, drug_vertex_index),
                "drugstoneType": str(graph.vp["type"][graph.vertex(drug_vertex_index)]).strip(),
                "score": float(score),
                "hasEdgesTo": str(
                    get_drug_neighbor_ids(
                        graph=graph,
                        drug_vertex_index=drug_vertex_index,
                        module_vertex_indices=module_vertex_index_set,
                    )
                ),
                "isResult": True,
                "isConnector": False,
            }
        )

    return pd.DataFrame(
        result_rows,
        columns=DRUGSTONE_RESULT_COLUMNS,
    )


def main(args) -> None:
    """Coordinate argument parsing, graph preparation, scoring, and output writing."""
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    ranking_output_path = args.ranking_output or Path(f"{args.prefix}.{args.algorithm}.csv")
    drug_predictions_output_path = args.drug_predictions_output or Path(
        f"{args.prefix}.{args.algorithm}.drug_predictions.tsv"
    )

    module_node_ids = load_module_node_ids(args.module)
    graph = load_prioritization_graph(args.drug_prioritization_graph)

    module_vertex_indices = get_module_vertex_indices(graph, module_node_ids)

    if not args.includeIndirectDrugs:
        logger.info("Restricting to drugs directly connected to module proteins/genes")
        drug_vertex_indices = filter_direct_drugs(graph, module_vertex_indices)
    else:
        logger.info("Keeping indirect drugs in the merged graph")
        drug_vertex_indices = get_candidate_drug_vertices(graph)

    if not drug_vertex_indices:
        logger.warning("No candidate drugs remain after filtering; writing empty outputs")
        result_df = build_result_table(
            graph=graph,
            scored_drugs=[],
            module_vertex_indices=module_vertex_indices,
            result_size=args.result_size,
        )
        write_results(result_df, ranking_output_path)
        write_drug_predictions(
            module_path=args.module,
            ranking_df=result_df,
            output_path=drug_predictions_output_path,
        )
        return

    logger.info(
        "Working graph contains %d vertices and %d edges",
        graph.num_vertices(),
        graph.num_edges(),
    )

    logger.info("Scoring candidate drugs with algorithm '%s'", args.algorithm)
    if args.algorithm == "trustrank":
        scored_drugs = run_trustrank(
            graph=graph,
            module_vertex_indices=module_vertex_indices,
            drug_vertex_indices=drug_vertex_indices,
            damping_factor=args.damping_factor,
        )
    elif args.algorithm == "harmonic_centrality":
        scored_drugs = run_harmonic_centrality(
            graph=graph,
            module_vertex_indices=module_vertex_indices,
            drug_vertex_indices=drug_vertex_indices,
        )
    elif args.algorithm == "degree":
        scored_drugs = run_degree(
            graph=graph,
            module_vertex_indices=module_vertex_indices,
            drug_vertex_indices=drug_vertex_indices,
        )

    scored_drugs = normalize_scores(scored_drugs)

    result_df = build_result_table(
        graph=graph,
        scored_drugs=scored_drugs,
        module_vertex_indices=module_vertex_indices,
        result_size=args.result_size,
    )
    write_results(result_df, ranking_output_path)
    write_drug_predictions(
        module_path=args.module,
        ranking_df=result_df,
        output_path=drug_predictions_output_path,
    )

if __name__ == "__main__":
    # Sample argv for local debugging in the script directory:
    # sys.argv = [
    #     "drug_prioritization_graphtool.py",
    #     "--drug-prioritization-graph",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.drug_prioritization.gt",
    #     "--module",
    #     "../../data/modules/ensembl/seeds_ensembl.string.human_links_v12_0_min700.Ensembl.diamond.nodes.tsv",
    #     "--algorithm",
    #     "harmonic_centrality",
    #     "--prefix",
    #     "seeds_ensembl.string.human_links_v12_0_min700.Ensembl.diamond",
    #      "--includeIndirectDrugs",
    #     "-l",
    #     "DEBUG",
    # ]
    main(parse_args())
