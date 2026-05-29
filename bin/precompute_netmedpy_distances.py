#!/usr/bin/env python

"""Precompute NetMedPy shortest-path distances for one prepared PPI network."""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a prepared netmedpy PPI NetworkX pickle and precompute a "
            "NetMedPy distance matrix for network proximity/separation."
        )
    )
    parser.add_argument(
        "--ppi",
        required=True,
        type=Path,
        help="Prepared NetworkX PPI pickle, usually '<prefix>.netmedpy_ppi.pkl'.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        type=str,
        help="Output prefix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NetMedPy distance matrix. Defaults to '<prefix>.netmedpy_shortest_paths.pkl.npz'.",
    )
    parser.add_argument(
        "--n-processors",
        type=int,
        default=1,
        help="Number of processors passed to netmedpy.all_pair_distances.",
    )
    parser.add_argument(
        "--allow-disconnected",
        action="store_true",
        help="Allow disconnected PPI graphs. Prepared netmedpy PPIs should normally be LCC-only.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        default="INFO",
        help="The desired log level.",
    )
    return parser.parse_args()


def load_network(path: Path) -> nx.Graph:
    logger.info("Loading prepared netmedpy PPI from %s", path)
    with path.open("rb") as handle:
        network = pickle.load(handle)

    if not isinstance(network, nx.Graph):
        raise TypeError(f"Expected a networkx.Graph in {path}, got {type(network)!r}")
    if network.number_of_nodes() == 0:
        raise ValueError(f"Prepared PPI is empty: {path}")

    logger.info(
        "Loaded PPI with %d nodes and %d edges",
        network.number_of_nodes(),
        network.number_of_edges(),
    )
    return network


def precompute_distances(
    network: nx.Graph,
    output_path: Path,
    n_processors: int,
) -> None:
    try:
        import netmedpy
    except ImportError as error:
        raise ImportError(
            "netmedpy is required to precompute distances. Install it in the "
            "runtime environment used for this process."
        ) from error

    logger.info(
        "Computing all-pairs shortest-path distances with %d processor(s)",
        n_processors,
    )
    distances = netmedpy.all_pair_distances(
        network,
        distance="shortest_path",
        n_processors=n_processors,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    netmedpy.save_distances(distances, output_path)
    logger.info("Wrote NetMedPy shortest-path distance matrix: %s", output_path)


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    output_path = args.output or Path(f"{args.prefix}.netmedpy_shortest_paths.pkl.npz")

    if args.n_processors < 1:
        raise ValueError("--n-processors must be at least 1")

    network = load_network(args.ppi)
    if not args.allow_disconnected and not nx.is_connected(network):
        raise ValueError(
            "Prepared PPI is disconnected. Recreate it as an LCC-only netmedpy PPI, "
            "or pass --allow-disconnected if this is intentional."
        )

    precompute_distances(
        network=network,
        output_path=output_path,
        n_processors=args.n_processors,
    )


if __name__ == "__main__":
    # sys.argv = [
    #     "precompute_netmedpy_distances.py",
    #     "--ppi",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.netmedpy_ppi.pkl",
    #     "--prefix",
    #     "../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl",
    #     "--n-processors",
    #     "1",
    #     "-l",
    #     "DEBUG",
    # ]
    main(parse_args())
