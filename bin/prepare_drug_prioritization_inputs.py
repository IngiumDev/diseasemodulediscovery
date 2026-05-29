#!/usr/bin/env python

"""Prepare PPI-specific offline drug-prioritization inputs.

This script builds the network-level inputs used by offline drug prioritization:

* a PDI-only graph-tool graph
* a merged PPI + PDI graph-tool graph
* a PPI-specific drug background table
* a NetworkX PPI graph and drug-target dictionary for netmedpy

The input ``drug_has_target`` table is expected to use UniProt identifiers on the
target side. The output protein/gene identifiers are converted into the same
namespace as the selected PPI network before any PPI-aware filtering is applied.
"""

from __future__ import annotations

import ast
import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Iterable

import graph_tool.all as gt
import networkx as nx
import pandas as pd
import pyintergraph
from gprofiler import GProfiler

logger = logging.getLogger(__name__)

SUPPORTED_ID_SPACES = ("uniprot", "entrez", "ensembl", "symbol")
GPROFILER_ORGANISM = "hsapiens"
GPROFILER_TARGET_NAMESPACES = {
    "uniprot": "UNIPROTSWISSPROT",
    "entrez": "ENTREZGENE_ACC",
    "ensembl": "ENSG",
    "symbol": "HGNC",
}
BIOLOGICAL_NODE_TYPE = "protein/gene"
DRUG_NODE_TYPE = "drug"
PPI_EDGE_TYPE = "protein-protein"
PDI_EDGE_TYPE = "protein-drug"
DRUGSTONE_DRUG_NODE_TYPE = "drug"
DRUGSTONE_PROTEIN_NODE_TYPE = "protein"
DRUGSTONE_PPI_EDGE_TYPE = "protein-protein"
DRUGSTONE_PDI_EDGE_TYPE = "drug-protein"
DRUGSTONE_LABEL_FIELD = "label"
DRUGSTONE_NODE_TYPE_FIELD = "type"
DRUGSTONE_EDGE_TYPE_FIELD = "type"
DRUGSTONE_ENTREZ_FIELD = "entrez"
DRUGSTONE_ENSEMBL_FIELD = "ensembl"
DRUGSTONE_SYMBOL_FIELD = "symbol"
DRUGSTONE_UNIPROT_FIELD = "uniprot"
DRUGSTONE_ACTIONS_FIELD = "actions"
DRUGSTONE_ID_SPACE_FIELDS = {
    "uniprot": DRUGSTONE_UNIPROT_FIELD,
    "entrez": DRUGSTONE_ENTREZ_FIELD,
    "ensembl": DRUGSTONE_ENSEMBL_FIELD,
    "symbol": DRUGSTONE_SYMBOL_FIELD,
}


def parse_args():
    """Define and parse command-line arguments for the preparation step."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a UniProt-based drug_has_target table into the pipeline "
            "ID space, filter it to one PPI network, and build offline "
            "drug-prioritization graph/background inputs."
        )
    )

    required = parser.add_argument_group("required inputs")
    required.add_argument("--ppi", required=True, type=Path, help="Input PPI graph-tool .gt or Drugst.One protein-protein GraphML file.")
    required.add_argument("--pdi", required=True, type=Path, help="Input drug_has_target CSV/TSV or Drugst.One protein-drug GraphML file.")
    required.add_argument("--drugs", required=True, type=Path, help="Input drug metadata CSV/TSV file.")
    required.add_argument("-p", "--prefix", required=True, type=str, help="Output prefix.")
    required.add_argument(
        "--id-space",
        required=True,
        choices=SUPPORTED_ID_SPACES,
        default="entrez",  # This is the pipeline default
        help="Identifier namespace used by the PPI network and disease modules.",
    )
    filtering = parser.add_argument_group("filtering")
    filtering.add_argument(
        "--includeNonApprovedDrugs",
        action="store_true",
        help=(
            "Include non-approved drugs. If omitted, interactions are kept only "
            "for drugs whose drugGroups list contains --approved-match as an exact entry."
        ),
    )
    filtering.add_argument(
        "--approved-match",
        default="approved",
        help="Case-insensitive exact drugGroups entry used to mark a drug as approved.",
    )

    columns = parser.add_argument_group("less commonly changed column names")
    columns.add_argument("--pdi-drug-column", default="sourceDomainId", help="PDI column containing drug IDs.")
    columns.add_argument(
        "--pdi-protein-column",
        default="targetDomainId",
        help="PDI column containing UniProt protein IDs.",
    )
    columns.add_argument("--drug-id-column", default="primaryDomainId", help="Drug metadata ID column.")
    columns.add_argument("--drug-name-column", default="displayName", help="Drug display-name column.")
    columns.add_argument("--drug-groups-column", default="drugGroups", help="Drug status/groups column.")

    outputs = parser.add_argument_group("output files")
    outputs.add_argument(
        "--drug-background-output",
        type=Path,
        help="Output PPI-specific drug background table. Defaults to '<prefix>.drug_background.tsv'.",
    )
    outputs.add_argument(
        "--pdi-graph-output",
        type=Path,
        help="Output PDI-only graph-tool graph. Defaults to '<prefix>.pdi.gt'.",
    )
    outputs.add_argument(
        "--merged-graph-output",
        type=Path,
        help="Output merged PPI + PDI graph-tool graph. Defaults to '<prefix>.drug_prioritization.gt'.",
    )
    outputs.add_argument(
        "--netmedpy-ppi-output",
        type=Path,
        help="Output netmedpy NetworkX PPI pickle. Defaults to '<prefix>.netmedpy_ppi.pkl'.",
    )
    outputs.add_argument(
        "--netmedpy-drug-targets-output",
        type=Path,
        help=(
            "Output netmedpy drug-target dictionary pickle. "
            "Defaults to '<prefix>.netmedpy_drug_targets.pkl'."
        ),
    )

    parser.add_argument(
        "-l",
        "--log-level",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"),
        default="INFO",
        help="The desired log level.",
    )

    return parser.parse_args()


def clean_text_id(value) -> str:
    """Return a string ID with whitespace removed and empty/null values collapsed."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""
    return text


def parse_list_str(value) -> list[str]:
    """Safely parse a Python list stored as a string. Returns [] on failure."""
    text = clean_text_id(value)
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(parsed, list):
        return []
    return [clean_text_id(entry) for entry in parsed if clean_text_id(entry)]


def remove_known_prefix(value, prefix: str) -> str:
    """Remove a known NeDRex-style domain prefix from an ID.

    Examples:
    - ``drugbank.DB00001`` -> ``DB00001``
    - ``uniprot.P00734`` -> ``P00734``
    """
    text = clean_text_id(value)
    dotted_prefix = f"{prefix}."
    colon_prefix = f"{prefix}:"
    if text.lower().startswith(dotted_prefix):
        return text[len(dotted_prefix) :]
    if text.lower().startswith(colon_prefix):
        return text[len(colon_prefix) :]
    return text


def clean_drug_id(value) -> str:
    """Return the bare DrugBank ID used as the drug vertex ID."""
    return remove_known_prefix(value, "drugbank")


def clean_uniprot_id(value) -> str:
    """Return the bare UniProt accession from NeDRex PDI target IDs."""
    return remove_known_prefix(value, "uniprot")


def read_table(path: Path) -> pd.DataFrame:
    """Read a delimited table as strings while preserving empty cells as blanks."""
    logger.debug("Reading table from %s", path)
    return pd.read_csv(path, dtype=str).fillna("")


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], source_name: str) -> None:
    """Raise a readable error if a table is missing required columns."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def remove_graph_tool_self_loops(graph: gt.Graph) -> int:
    """Remove self-loop edges from a graph-tool graph and return the count."""
    self_loops = [edge for edge in graph.edges() if edge.source() == edge.target()]
    for edge in self_loops:
        graph.remove_edge(edge)
    return len(self_loops)


def load_ppi_gt_graph(ppi_path: Path) -> tuple[gt.Graph, set[str]]:
    """Load a pipeline graph-tool PPI graph and return its node IDs.

    Pipeline PPI graphs are undirected graph-tool graphs with one required
    vertex property:

    - ``name``: the node ID in the selected pipeline namespace

    They do not carry graph properties or edge properties. The merged
    prioritization graph adds explicit node and edge annotations while copying
    this PPI structure.
    """
    logger.info("Loading PPI graph from %s", ppi_path)
    graph = gt.load_graph(str(ppi_path))
    removed_self_loops = remove_graph_tool_self_loops(graph)

    if "name" not in graph.vp:
        raise ValueError(
            "PPI graph is missing required vertex property 'name'. "
            f"Available vertex properties: {list(graph.vp.keys())}"
        )

    ppi_nodes = {str(graph.vp["name"][vertex]).strip() for vertex in graph.vertices()}
    ppi_nodes.discard("")

    if graph.is_directed():
        logger.warning("Input PPI graph is directed; merged prioritization graph will still be undirected.")
    if removed_self_loops:
        logger.warning("Removed %d PPI self-loop edges from %s", removed_self_loops, ppi_path)

    logger.info("Loaded %d PPI nodes", len(ppi_nodes))
    logger.info("Loaded %d PPI edges", graph.num_edges())
    return graph, ppi_nodes


def graphml_namespace_ids(node_data: dict, id_space: str) -> list[str]:
    """Return selected namespace IDs from a Drugst.One GraphML protein node."""
    field = DRUGSTONE_ID_SPACE_FIELDS[id_space]
    value = node_data.get(field, "")
    if not value:
        return []
    return [node_id.strip() for node_id in value.split(",") if node_id.strip()]


def load_ppi_graphml(ppi_path: Path, id_space: str) -> tuple[gt.Graph, set[str]]:
    """Convert a Drugst.One protein-protein GraphML to the pipeline PPI graph shape."""
    logger.info("Loading Drugst.One GraphML PPI from %s", ppi_path)
    input_graph = nx.read_graphml(ppi_path)

    graph = gt.Graph(directed=False)
    name_prop = graph.new_vertex_property("string")
    vertex_by_id: dict[str, gt.Vertex] = {}
    seen_edges: set[tuple[str, str]] = set()
    skipped = {
        "not_protein_protein_edge": 0,
        "not_protein_endpoint_types": 0,
        "missing_namespace_id": 0,
        "self_loop": 0,
        "duplicate_edge": 0,
    }

    for source, target, edge_data in input_graph.edges(data=True):
        if edge_data.get(DRUGSTONE_EDGE_TYPE_FIELD) != DRUGSTONE_PPI_EDGE_TYPE:
            skipped["not_protein_protein_edge"] += 1
            continue
        if source == target:
            skipped["self_loop"] += 1
            continue

        source_data = input_graph.nodes[source]
        target_data = input_graph.nodes[target]
        if (
            source_data.get(DRUGSTONE_NODE_TYPE_FIELD) != DRUGSTONE_PROTEIN_NODE_TYPE
            or target_data.get(DRUGSTONE_NODE_TYPE_FIELD) != DRUGSTONE_PROTEIN_NODE_TYPE
        ):
            skipped["not_protein_endpoint_types"] += 1
            continue

        source_ids = graphml_namespace_ids(source_data, id_space)
        target_ids = graphml_namespace_ids(target_data, id_space)
        if not source_ids or not target_ids:
            skipped["missing_namespace_id"] += 1
            continue

        for source_id in source_ids:
            for target_id in target_ids:
                if source_id == target_id:
                    skipped["self_loop"] += 1
                    continue

                edge_key = tuple(sorted((source_id, target_id)))
                if edge_key in seen_edges:
                    skipped["duplicate_edge"] += 1
                    continue

                for node_id in edge_key:
                    if node_id not in vertex_by_id:
                        vertex = graph.add_vertex()
                        vertex_by_id[node_id] = vertex
                        name_prop[vertex] = node_id

                graph.add_edge(vertex_by_id[source_id], vertex_by_id[target_id])
                seen_edges.add(edge_key)

    graph.vp["name"] = name_prop
    ppi_nodes = set(vertex_by_id)

    logger.info(
        "Converted GraphML PPI to %d %s nodes and %d edges",
        graph.num_vertices(),
        id_space,
        graph.num_edges(),
    )
    logger.info("Skipped GraphML PPI edge counts: %s", skipped)
    return graph, ppi_nodes


def load_ppi_graph(ppi_path: Path, id_space: str) -> tuple[gt.Graph, set[str]]:
    """Load PPI input by extension and return a graph with vertex property ``name``."""
    suffix = ppi_path.suffix.lower()
    if suffix == ".gt":
        logger.info("Using graph-tool PPI loading logic for %s", ppi_path)
        return load_ppi_gt_graph(ppi_path)
    if suffix == ".graphml":
        logger.info("Using GraphML PPI loading logic for %s", ppi_path)
        return load_ppi_graphml(ppi_path, id_space=id_space)
    raise ValueError(
        f"Unsupported PPI input extension '{ppi_path.suffix}'. "
        "Expected .gt or .graphml."
    )


def load_drug_metadata(
    drugs_path: Path,
    drug_id_column: str,
    drug_name_column: str,
    drug_groups_column: str,
    approved_match: str,
) -> pd.DataFrame:
    """Load drug metadata and add a boolean ``approved`` column."""
    drugs = read_table(drugs_path)
    require_columns(
        drugs,
        [drug_id_column, drug_name_column, drug_groups_column],
        source_name="drug metadata",
    )

    drugs = drugs.copy()
    drugs[drug_id_column] = drugs[drug_id_column].map(clean_drug_id)
    approved_entry = clean_text_id(approved_match).casefold()
    drugs["approved"] = drugs[drug_groups_column].apply(
        lambda groups: approved_entry in {entry.casefold() for entry in parse_list_str(groups)}
    )

    logger.info("Loaded %d drugs from metadata", len(drugs))
    logger.info("Approved drugs in metadata: %d", int(drugs["approved"].sum()))
    return drugs


def load_pdi_table(
    pdi_path: Path,
    pdi_drug_column: str,
    pdi_protein_column: str,
) -> pd.DataFrame:
    """Load the CSV/TSV PDI table and normalize drug and UniProt target IDs."""
    logger.info("Loading PDI table from %s", pdi_path)
    pdi = read_table(pdi_path)
    require_columns(pdi, [pdi_drug_column, pdi_protein_column], source_name="PDI table")

    pdi = pdi.copy()
    pdi[pdi_drug_column] = pdi[pdi_drug_column].map(clean_drug_id)
    pdi[pdi_protein_column] = pdi[pdi_protein_column].map(
        clean_uniprot_id
    )
    pdi = pdi[(pdi[pdi_drug_column] != "") & (pdi[pdi_protein_column] != "")].copy()

    logger.info("Loaded %d non-empty PDI rows", len(pdi))
    return pdi


def build_drug_label_map(
    drugs: pd.DataFrame,
    drug_id_column: str,
    drug_name_column: str,
) -> dict[str, str]:
    """Map unique drug display labels to normalized primary drug IDs."""
    label_ids = (
        drugs[[drug_name_column, drug_id_column]]
        .dropna()
        .assign(_label_key=lambda df: df[drug_name_column].str.casefold())
        .loc[lambda df: (df["_label_key"] != "") & (df[drug_id_column] != "")]
        .groupby("_label_key")[drug_id_column]
        .agg(lambda ids: set(ids))
    )

    ambiguous = label_ids[label_ids.map(len) > 1]
    if not ambiguous.empty:
        examples = list(ambiguous.head(5).items())
        logger.warning(
            "Found %d ambiguous drug display labels in metadata; examples: %s",
            len(ambiguous),
            examples,
        )

    return {
        label: next(iter(ids))
        for label, ids in label_ids.items()
        if len(ids) == 1
    }


def load_pdi_graphml(
    pdi_path: Path,
    drugs: pd.DataFrame,
    pdi_drug_column: str,
    pdi_protein_column: str,
    drug_id_column: str,
    drug_name_column: str,
) -> pd.DataFrame:
    """Load a Drugst.One protein-drug GraphML and normalize it like drug_has_target."""
    logger.info("Loading Drugst.One GraphML PDI from %s", pdi_path)
    graph = nx.read_graphml(pdi_path)
    drug_label_to_id = build_drug_label_map(
        drugs=drugs,
        drug_id_column=drug_id_column,
        drug_name_column=drug_name_column,
    )

    rows: list[dict[str, str]] = []
    skipped: dict[str, int] = {}

    for source, target, edge_data in graph.edges(data=True):
        if edge_data.get(DRUGSTONE_EDGE_TYPE_FIELD) != DRUGSTONE_PDI_EDGE_TYPE:
            skipped["not_drug_protein_edge"] = skipped.get("not_drug_protein_edge", 0) + 1
            continue

        source_data = graph.nodes[source]
        target_data = graph.nodes[target]
        source_type = source_data.get(DRUGSTONE_NODE_TYPE_FIELD)
        target_type = target_data.get(DRUGSTONE_NODE_TYPE_FIELD)
        if source_type == DRUGSTONE_DRUG_NODE_TYPE and target_type == DRUGSTONE_PROTEIN_NODE_TYPE:
            drug_node = source_data
            protein_node = target_data
        elif source_type == DRUGSTONE_PROTEIN_NODE_TYPE and target_type == DRUGSTONE_DRUG_NODE_TYPE:
            drug_node = target_data
            protein_node = source_data
        else:
            skipped["not_drug_protein_endpoint_types"] = skipped.get("not_drug_protein_endpoint_types", 0) + 1
            continue

        drug_label = drug_node.get(DRUGSTONE_LABEL_FIELD, "")
        drug_id = drug_label_to_id.get(drug_label.casefold())
        if not drug_id:
            skipped["unmapped_drug_label"] = skipped.get("unmapped_drug_label", 0) + 1
            continue

        uniprot_id = protein_node.get(DRUGSTONE_UNIPROT_FIELD, "")
        if not uniprot_id:
            skipped["missing_uniprot"] = skipped.get("missing_uniprot", 0) + 1
            continue

        rows.append(
            {
                pdi_drug_column: drug_id,
                pdi_protein_column: uniprot_id,
                "actions": edge_data.get(DRUGSTONE_ACTIONS_FIELD) or "[]",
            }
        )

    pdi = pd.DataFrame(rows, columns=[pdi_drug_column, pdi_protein_column, "actions"])
    pdi = pdi[(pdi[pdi_drug_column] != "") & (pdi[pdi_protein_column] != "")].copy()

    node_type_counts = pd.Series(
        [data.get(DRUGSTONE_NODE_TYPE_FIELD) for _, data in graph.nodes(data=True)]
    ).value_counts()
    logger.info("GraphML node type counts: %s", node_type_counts.to_dict())
    logger.info("GraphML edges normalized to non-empty PDI rows: %d", len(pdi))
    if skipped:
        logger.warning("Skipped GraphML edge counts by reason: %s", skipped)

    return pdi


def load_pdi(
    pdi_path: Path,
    pdi_drug_column: str,
    pdi_protein_column: str,
    drugs: pd.DataFrame,
    drug_id_column: str,
    drug_name_column: str,
) -> pd.DataFrame:
    """Load PDI input by extension and normalize drug and UniProt target IDs."""
    suffix = pdi_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        logger.info("Using CSV/TSV PDI loading logic for %s", pdi_path)
        return load_pdi_table(
            pdi_path=pdi_path,
            pdi_drug_column=pdi_drug_column,
            pdi_protein_column=pdi_protein_column,
        )
    if suffix == ".graphml":
        logger.info("Using GraphML PDI loading logic for %s", pdi_path)
        return load_pdi_graphml(
            pdi_path=pdi_path,
            drugs=drugs,
            pdi_drug_column=pdi_drug_column,
            pdi_protein_column=pdi_protein_column,
            drug_id_column=drug_id_column,
            drug_name_column=drug_name_column,
        )
    else:
        raise ValueError(
            f"Unsupported PDI input extension '{pdi_path.suffix}'. "
            "Expected .csv, .tsv, or .graphml."
        )


def add_drug_metadata_to_pdi(
    pdi: pd.DataFrame,
    drugs: pd.DataFrame,
    pdi_drug_column: str,
    drug_id_column: str,
) -> pd.DataFrame:
    """Attach drug names, status strings, and approval flags to PDI rows."""
    merged = pdi.merge(
        drugs,
        left_on=pdi_drug_column,
        right_on=drug_id_column,
        how="left",
        suffixes=("", "_drug"),
    )
    merged["approved"] = merged["approved"].fillna(False).astype(bool)
    return merged


def filter_approved_drug_interactions(
    pdi: pd.DataFrame,
    include_non_approved_drugs: bool,
) -> pd.DataFrame:
    """Remove non-approved drug-target rows unless runtime parameters keep them."""
    if include_non_approved_drugs:
        logger.info("Keeping non-approved drug interactions")
        return pdi.copy()

    filtered = pdi[pdi["approved"]].copy()
    logger.info("PDI rows after approved-only filtering: %d", len(filtered))
    return filtered

# TODO: understand why duplicates occur
def fetch_uniprot_conversion_map(
    uniprot_ids: set[str],
    id_space: str,
) -> dict[str, set[str]]:
    """Convert UniProt IDs to the selected pipeline namespace with g:Profiler."""
    cleaned_ids = sorted(uniprot_id for uniprot_id in uniprot_ids if uniprot_id)
    if not cleaned_ids:
        return {}

    target_namespace = GPROFILER_TARGET_NAMESPACES[id_space]
    logger.info(
        "Running g:Profiler conversion for %d IDs to %s (%s)",
        len(cleaned_ids),
        target_namespace,
        GPROFILER_ORGANISM,
    )
    gp = GProfiler(return_dataframe=False)
    converted_rows = gp.convert(
        organism=GPROFILER_ORGANISM,
        query=cleaned_ids,
        target_namespace=target_namespace,
        numeric_namespace="ENTREZGENE_ACC",
    )

    conversion_map: dict[str, set[str]] = {input_id: set() for input_id in cleaned_ids}
    for row in converted_rows:
        incoming = str(row.get("incoming", "")).strip()
        converted = str(row.get("converted", "")).strip()
        if incoming and converted and converted.upper() not in {"N/A", "NONE"}:
            conversion_map.setdefault(incoming, set()).add(converted)

    mapped_count = sum(1 for converted_ids in conversion_map.values() if converted_ids)
    logger.info("g:Profiler mapped %d/%d IDs", mapped_count, len(cleaned_ids))
    return conversion_map


def convert_pdi_targets_to_id_space(
    pdi: pd.DataFrame,
    pdi_protein_column: str,
    conversion_map: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert UniProt PDI targets into the selected pipeline namespace.

    One-to-many mappings duplicate the corresponding drug-target rows, which
    means a drug targeting UniProt ``P1`` will target both genes ``A`` and ``B``
    if ``P1 -> {A, B}``. Unmapped UniProt IDs are dropped and returned in a
    diagnostic table.
    """
    mapping_rows = [
        {
            "original_uniprot": uniprot_id,
            "converted_target_id": converted_id,
            "conversion_count": len(converted_ids),
        }
        for uniprot_id, converted_ids in conversion_map.items()
        for converted_id in sorted(converted_ids)
    ]
    mapping = pd.DataFrame(
        mapping_rows,
        columns=["original_uniprot", "converted_target_id", "conversion_count"],
    )
    converted = pdi.merge(
        mapping,
        left_on=pdi_protein_column,
        right_on="original_uniprot",
        how="inner",
    )

    unmapped_ids = sorted(
        uniprot_id for uniprot_id, converted_ids in conversion_map.items() if not converted_ids
    )
    unmapped = pd.DataFrame(
        {
            "uniprot_id": unmapped_ids,
            "reason": "not_found_in_conversion_map",
        }
    )

    logger.info("Converted PDI rows: %d", len(converted))
    logger.info("UniProt IDs not converted: %d", unmapped["uniprot_id"].nunique())
    return converted, unmapped


def filter_converted_pdi_to_ppi(
    converted_pdi: pd.DataFrame,
    ppi_nodes: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only converted PDI rows whose target is present in the PPI network."""
    if converted_pdi.empty:
        logger.error(
            "No converted PDI rows are available before PPI filtering. "
            "Check the UniProt conversion step and approved-drug filtering."
        )
        raise SystemExit(1)

    in_ppi = converted_pdi["converted_target_id"].isin(ppi_nodes)
    filtered = converted_pdi[in_ppi].copy()

    missing = (
        converted_pdi.loc[~in_ppi, ["converted_target_id", "original_uniprot"]]
        .drop_duplicates()
        .assign(reason="not_found_in_ppi")
        .sort_values(["converted_target_id", "original_uniprot"])
    )

    logger.info("PDI rows after PPI-target filtering: %d", len(filtered))
    logger.info("Converted targets not present in PPI: %d", missing["converted_target_id"].nunique())

    if filtered.empty:
        logger.error(
            "No PDI rows remain after filtering converted targets to the PPI. "
            "Converted target IDs and PPI node IDs may be in different namespaces."
        )
        raise SystemExit(1)

    return filtered, missing


def _add_vertex(
    graph: gt.Graph,
    vertex_by_id: dict[str, gt.Vertex],
    node_id: str,
    vertex_properties: dict,
    values: dict[str, str | bool],
):
    """Return an existing vertex or create a new vertex with minimal node properties.

    Vertex properties created by this script:
    - ``id``: stable bare node ID used by algorithms, e.g. ``348`` or ``DB00001``
    - ``name``: display name for drugs only; empty for protein/gene nodes
    - ``namespace``: ID namespace, e.g. ``entrez``, ``ensembl``, ``symbol``, ``uniprot``, ``drugbank``
    - ``type``: node class, currently ``protein/gene`` or ``drug``
    - ``status``: drug status/groups string from metadata; empty for protein/gene nodes
    - ``approved``: whether the drug status/groups include the approved entry
    """
    if node_id in vertex_by_id:
        return vertex_by_id[node_id]
    else:
        vertex = graph.add_vertex()
        vertex_by_id[node_id] = vertex
        for property_name, value in values.items():
            vertex_properties[property_name][vertex] = value
    return vertex


def _new_prioritization_graph():
    """Create an undirected graph-tool graph with the prioritization schema.

    Kept vertex properties:
    - ``id``: algorithm-facing node ID
    - ``name``: drug display name; empty for protein/gene nodes
    - ``namespace``: source namespace for ``id``
    - ``type``: node class, ``protein/gene`` or ``drug``
    - ``status``: drug status/groups string from metadata; empty for protein/gene nodes
    - ``approved``: whether the drug status/groups include the approved entry

    Kept edge properties:
    - ``type``: interaction class, ``protein-protein`` or ``protein-drug``
    """
    graph = gt.Graph(directed=False)
    props = {
        "id": graph.new_vertex_property("string"),
        "name": graph.new_vertex_property("string"),
        "namespace": graph.new_vertex_property("string"),
        "type": graph.new_vertex_property("string"),
        "status": graph.new_vertex_property("string"),
        "approved": graph.new_vertex_property("bool"),
        "edge_type": graph.new_edge_property("string"),
    }
    return graph, props


def _install_graph_properties(graph: gt.Graph, props: dict) -> None:
    """Attach common vertex and edge properties to a graph before saving."""
    graph.vp["id"] = props["id"]
    graph.vp["name"] = props["name"]
    graph.vp["namespace"] = props["namespace"]
    graph.vp["type"] = props["type"]
    graph.vp["status"] = props["status"]
    graph.vp["approved"] = props["approved"]
    graph.ep["type"] = props["edge_type"]


def add_pdi_edges_to_graph(
    graph: gt.Graph,
    props: dict,
    vertex_by_id: dict[str, gt.Vertex],
    pdi: pd.DataFrame,
    id_space: str,
    pdi_drug_column: str,
    drug_name_column: str,
    drug_groups_column: str,
) -> None:
    """Add protein/gene, drug, and protein-drug edges from a filtered PDI table."""
    for _, row in pdi.iterrows():
        target_id = str(row["converted_target_id"])
        drug_id = str(row[pdi_drug_column])

        target_vertex = _add_vertex(
            graph,
            vertex_by_id,
            target_id,
            props,
            {
                "id": target_id,
                "name": "",
                "namespace": id_space,
                "type": BIOLOGICAL_NODE_TYPE,
                "status": "",
                "approved": False,
            },
        )
        drug_vertex = _add_vertex(
            graph,
            vertex_by_id,
            drug_id,
            props,
            {
                "id": drug_id,
                "name": str(row.get(drug_name_column, drug_id)),
                "namespace": "drugbank",
                "type": DRUG_NODE_TYPE,
                "status": str(row.get(drug_groups_column, "")),
                "approved": bool(row.get("approved", False)),
            },
        )

        edge = graph.add_edge(target_vertex, drug_vertex)
        props["edge_type"][edge] = PDI_EDGE_TYPE


def build_pdi_graph(
    pdi: pd.DataFrame,
    id_space: str,
    pdi_drug_column: str,
    drug_name_column: str,
    drug_groups_column: str,
) -> gt.Graph:
    """Build a PDI-only graph filtered to proteins/genes present in the PPI."""
    graph, props = _new_prioritization_graph()
    vertex_by_id: dict[str, gt.Vertex] = {}
    add_pdi_edges_to_graph(
        graph=graph,
        props=props,
        vertex_by_id=vertex_by_id,
        pdi=pdi,
        id_space=id_space,
        pdi_drug_column=pdi_drug_column,
        drug_name_column=drug_name_column,
        drug_groups_column=drug_groups_column,
    )
    _install_graph_properties(graph, props)
    return graph


def build_annotated_ppi_graph(
    ppi_graph: gt.Graph,
    id_space: str,
) -> tuple[gt.Graph, dict, dict[str, gt.Vertex]]:
    """Copy the PPI graph structure and add the prioritization graph properties.

    The input PPI graph only provides ``vp["name"]``. The copied graph keeps the
    PPI topology, then receives this script's graph schema:

    Vertex properties:
    - ``id``: copied from the original PPI ``name`` property
    - ``name``: empty for protein/gene nodes
    - ``namespace``: selected pipeline ID space
    - ``type``: ``protein/gene``
    - ``status``: empty for protein/gene nodes
    - ``approved``: ``False`` for protein/gene nodes

    Edge properties:
    - ``type``: ``protein-protein``
    """
    graph = gt.Graph(ppi_graph, directed=False)
    props = {
        "id": graph.new_vertex_property("string"),
        "name": graph.new_vertex_property("string"),
        "namespace": graph.new_vertex_property("string"),
        "type": graph.new_vertex_property("string"),
        "status": graph.new_vertex_property("string"),
        "approved": graph.new_vertex_property("bool"),
        "edge_type": graph.new_edge_property("string"),
    }
    vertex_by_id: dict[str, gt.Vertex] = {}

    for vertex in graph.vertices():
        node_id = str(graph.vp["name"][vertex]).strip()
        if not node_id:
            logger.error("Encountered a PPI vertex with an empty name property.")
            raise SystemExit(1)

        props["id"][vertex] = node_id
        props["name"][vertex] = ""
        props["namespace"][vertex] = id_space
        props["type"][vertex] = BIOLOGICAL_NODE_TYPE
        props["status"][vertex] = ""
        props["approved"][vertex] = False
        vertex_by_id[node_id] = vertex

    seen_edges: set[tuple[str, str]] = set()
    for edge in graph.edges():
        source_id = str(props["id"][edge.source()]).strip()
        target_id = str(props["id"][edge.target()]).strip()
        if not source_id or not target_id:
            logger.error("Encountered a PPI edge with an empty endpoint ID.")
            raise SystemExit(1)
        if source_id == target_id:
            logger.error("Encountered a PPI self-loop for node %s.", source_id)
            raise SystemExit(1)

        edge_key = tuple(sorted((source_id, target_id)))
        if edge_key in seen_edges:
            logger.error(
                "Encountered a duplicate PPI edge between %s and %s.",
                source_id,
                target_id,
            )
            raise SystemExit(1)
        seen_edges.add(edge_key)
        props["edge_type"][edge] = PPI_EDGE_TYPE

    return graph, props, vertex_by_id


def add_pdi_drugs_and_edges(
    output_graph: gt.Graph,
    props: dict,
    vertex_by_id: dict[str, gt.Vertex],
    pdi_graph: gt.Graph,
) -> None:
    """Add drug vertices and PDI edges from the already-built PDI graph."""
    drug_vertices = gt.find_vertex(pdi_graph, pdi_graph.vp["type"], DRUG_NODE_TYPE)
    for vertex in drug_vertices:
        node_id = str(pdi_graph.vp["id"][vertex]).strip()
        if not node_id:
            logger.error("Encountered a PDI drug vertex with an empty ID.")
            raise SystemExit(1)
        _add_vertex(
            output_graph,
            vertex_by_id,
            node_id,
            props,
            {
                "id": node_id,
                "name": str(pdi_graph.vp["name"][vertex]),
                "namespace": str(pdi_graph.vp["namespace"][vertex]),
                "type": str(pdi_graph.vp["type"][vertex]),
                "status": str(pdi_graph.vp["status"][vertex]),
                "approved": bool(pdi_graph.vp["approved"][vertex]),
            },
        )

    seen_pdi_edges: set[tuple[str, str]] = set()
    for edge in pdi_graph.edges():
        source_id = str(pdi_graph.vp["id"][edge.source()]).strip()
        target_id = str(pdi_graph.vp["id"][edge.target()]).strip()
        if not source_id or not target_id:
            logger.error("Encountered a PDI edge with an empty endpoint ID.")
            raise SystemExit(1)

        source_type = str(pdi_graph.vp["type"][edge.source()])
        target_type = str(pdi_graph.vp["type"][edge.target()])
        if source_type == DRUG_NODE_TYPE and target_type == BIOLOGICAL_NODE_TYPE:
            drug_id = source_id
            target_node_id = target_id
        elif source_type == BIOLOGICAL_NODE_TYPE and target_type == DRUG_NODE_TYPE:
            drug_id = target_id
            target_node_id = source_id
        else:
            logger.error(
                "Encountered a PDI edge that is not drug/protein-gene: "
                "%s (%s) -- %s (%s).",
                source_id,
                source_type,
                target_id,
                target_type,
            )
            raise SystemExit(1)

        if drug_id not in vertex_by_id:
            logger.error("PDI edge references drug %s, but the drug vertex was not added.", drug_id)
            raise SystemExit(1)
        if target_node_id not in vertex_by_id:
            logger.error(
                "PDI edge references target %s, but that target is absent from the copied PPI graph.",
                target_node_id,
            )
            raise SystemExit(1)

        edge_key = tuple(sorted((drug_id, target_node_id)))
        if edge_key in seen_pdi_edges:
            logger.error(
                "Encountered a duplicate PDI edge between drug %s and target %s.",
                drug_id,
                target_node_id,
            )
            raise SystemExit(1)
        seen_pdi_edges.add(edge_key)

        new_edge = output_graph.add_edge(vertex_by_id[drug_id], vertex_by_id[target_node_id])
        props["edge_type"][new_edge] = PDI_EDGE_TYPE


def build_merged_prioritization_graph(
    ppi_graph: gt.Graph,
    pdi_graph: gt.Graph,
    id_space: str,
) -> gt.Graph:
    """Build a heterogeneous graph by merging annotated PPI and PDI graphs."""
    graph, props, vertex_by_id = build_annotated_ppi_graph(
        ppi_graph=ppi_graph,
        id_space=id_space,
    )
    add_pdi_drugs_and_edges(
        output_graph=graph,
        props=props,
        vertex_by_id=vertex_by_id,
        pdi_graph=pdi_graph,
    )

    _install_graph_properties(graph, props)
    return graph


def build_drug_background(
    pdi: pd.DataFrame,
    pdi_drug_column: str,
    drug_name_column: str,
    drug_groups_column: str,
) -> pd.DataFrame:
    """Create the PPI-specific drug background from retained drug vertices."""

    background = (
        pdi.groupby([pdi_drug_column, drug_name_column, drug_groups_column, "approved"], dropna=False)
        .agg(
            n_targets=("converted_target_id", "nunique"),
            targets=("converted_target_id", lambda values: ";".join(sorted(set(values)))),
        )
        .reset_index()
        .rename(
            columns={
                pdi_drug_column: "drug_id",
                drug_name_column: "drug_name",
                drug_groups_column: "status",
            }
        )
        .sort_values("drug_id")
    )
    return background


def build_netmedpy_ppi_network(ppi_graph: gt.Graph) -> nx.Graph:
    """Convert the PPI graph-tool graph to an LCC-only NetworkX graph."""
    network = pyintergraph.gt2nx(ppi_graph, labelname="name")

    if network.number_of_nodes() == 0:
        logger.error("Cannot build netmedpy PPI network from an empty graph.")
        raise SystemExit(1)

    original_node_count = network.number_of_nodes()
    component_nodes = max(nx.connected_components(network), key=len)
    network.remove_nodes_from(set(network) - component_nodes)

    logger.info(
        "Built netmedpy PPI network from largest connected component: %d/%d nodes, %d edges",
        network.number_of_nodes(),
        original_node_count,
        network.number_of_edges(),
    )
    return network


def build_netmedpy_drug_targets(
    pdi: pd.DataFrame,
    ppi_network: nx.Graph,
    pdi_drug_column: str,
) -> dict[str, set[str]]:
    """Build the netmedpy source dictionary, filtered to the PPI LCC."""
    lcc_nodes = set(ppi_network.nodes)
    lcc_pdi = pdi[pdi["converted_target_id"].isin(lcc_nodes)]

    dropped_rows = len(pdi) - len(lcc_pdi)
    if dropped_rows:
        logger.info(
            "Dropped %d PDI rows from netmedpy targets because targets are outside the PPI LCC",
            dropped_rows,
        )

    drug_targets = {
        drug_id: set(group["converted_target_id"])
        for drug_id, group in lcc_pdi.groupby(pdi_drug_column, sort=True)
    }

    logger.info("Built netmedpy drug-target dictionary for %d drugs", len(drug_targets))
    return drug_targets


def save_outputs(
    drug_background: pd.DataFrame,
    pdi_graph: gt.Graph,
    merged_graph: gt.Graph,
    netmedpy_ppi: nx.Graph,
    netmedpy_drug_targets: dict[str, set[str]],
    drug_background_output: Path,
    pdi_graph_output: Path,
    merged_graph_output: Path,
    netmedpy_ppi_output: Path,
    netmedpy_drug_targets_output: Path,
) -> None:
    """Save graph and background outputs for this network."""
    output_paths = {
        "drug_background": drug_background_output,
        "pdi_graph": pdi_graph_output,
        "merged_graph": merged_graph_output,
        "netmedpy_ppi": netmedpy_ppi_output,
        "netmedpy_drug_targets": netmedpy_drug_targets_output,
    }

    drug_background.to_csv(output_paths["drug_background"], sep="\t", index=False)
    pdi_graph.save(str(output_paths["pdi_graph"]))
    merged_graph.save(str(output_paths["merged_graph"]))
    with output_paths["netmedpy_ppi"].open("wb") as handle:
        pickle.dump(netmedpy_ppi, handle)
    with output_paths["netmedpy_drug_targets"].open("wb") as handle:
        pickle.dump(netmedpy_drug_targets, handle)

    for label, path in output_paths.items():
        logger.info("Wrote %s: %s", label, path)


def main(args) -> None:
    """Run the full PDI conversion, PPI filtering, graph building, and export workflow."""
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("Arguments: %s", args)

    id_space = args.id_space
    drug_background_output = args.drug_background_output or Path(f"{args.prefix}.drug_background.tsv")
    pdi_graph_output = args.pdi_graph_output or Path(f"{args.prefix}.pdi.gt")
    merged_graph_output = args.merged_graph_output or Path(f"{args.prefix}.drug_prioritization.gt")
    netmedpy_ppi_output = args.netmedpy_ppi_output or Path(f"{args.prefix}.netmedpy_ppi.pkl")
    netmedpy_drug_targets_output = (
        args.netmedpy_drug_targets_output or Path(f"{args.prefix}.netmedpy_drug_targets.pkl")
    )

    ppi_graph, ppi_nodes = load_ppi_graph(args.ppi, id_space=id_space)

    drugs = load_drug_metadata(
        drugs_path=args.drugs,
        drug_id_column=args.drug_id_column,
        drug_name_column=args.drug_name_column,
        drug_groups_column=args.drug_groups_column,
        approved_match=args.approved_match,
    )
    raw_pdi = load_pdi(
        pdi_path=args.pdi,
        pdi_drug_column=args.pdi_drug_column,
        pdi_protein_column=args.pdi_protein_column,
        drugs=drugs,
        drug_id_column=args.drug_id_column,
        drug_name_column=args.drug_name_column,
    )

    pdi_with_drugs = add_drug_metadata_to_pdi(
        raw_pdi,
        drugs,
        pdi_drug_column=args.pdi_drug_column,
        drug_id_column=args.drug_id_column,
    )
    approved_filtered_pdi = filter_approved_drug_interactions(
        pdi_with_drugs,
        include_non_approved_drugs=args.includeNonApprovedDrugs,
    )

    input_uniprots = set(approved_filtered_pdi[args.pdi_protein_column])
    if id_space == "uniprot":
        logger.info("Pipeline ID space is UniProt; using PDI targets directly")
        converted_pdi = approved_filtered_pdi.copy()
        converted_pdi["original_uniprot"] = converted_pdi[args.pdi_protein_column]
        converted_pdi["converted_target_id"] = converted_pdi[args.pdi_protein_column]
        converted_pdi["conversion_count"] = 1
        unmapped_uniprots = pd.DataFrame(columns=["uniprot_id", "reason"])
    else:
        logger.info("Converting PDI targets from UniProt to %s", id_space)
        conversion_map = fetch_uniprot_conversion_map(
            uniprot_ids=input_uniprots,
            id_space=id_space,
        )
        converted_pdi, unmapped_uniprots = convert_pdi_targets_to_id_space(
            approved_filtered_pdi,
            pdi_protein_column=args.pdi_protein_column,
            conversion_map=conversion_map,
        )
    ppi_filtered_pdi, ppi_missing_targets = filter_converted_pdi_to_ppi(
        converted_pdi,
        ppi_nodes=ppi_nodes,
    )
    # Different UniProt targets for the same drug can collapse onto one gene
    # ID after namespace conversion, so one drug-target graph edge is enough.
    deduplicated_pdi = ppi_filtered_pdi.drop_duplicates(
        subset=[args.pdi_drug_column, "converted_target_id"],
        keep="first",
    ).copy()
    duplicate_rows_removed = len(ppi_filtered_pdi) - len(deduplicated_pdi)
    if duplicate_rows_removed:
        logger.warning(
            "Removed %d duplicate PDI rows with the same drug and converted target",
            duplicate_rows_removed,
        )

    filtered_pdi = deduplicated_pdi.sort_values(
        [args.pdi_drug_column, "converted_target_id"]
    ).reset_index(drop=True)

    pdi_graph = build_pdi_graph(
        filtered_pdi,
        id_space=id_space,
        pdi_drug_column=args.pdi_drug_column,
        drug_name_column=args.drug_name_column,
        drug_groups_column=args.drug_groups_column,
    )
    merged_graph = build_merged_prioritization_graph(
        ppi_graph=ppi_graph,
        pdi_graph=pdi_graph,
        id_space=id_space,
    )
    drug_background = build_drug_background(
        filtered_pdi,
        pdi_drug_column=args.pdi_drug_column,
        drug_name_column=args.drug_name_column,
        drug_groups_column=args.drug_groups_column,
    )
    netmedpy_ppi = build_netmedpy_ppi_network(ppi_graph)
    netmedpy_drug_targets = build_netmedpy_drug_targets(
        pdi=filtered_pdi,
        ppi_network=netmedpy_ppi,
        pdi_drug_column=args.pdi_drug_column,
    )

    unique_uniprots = len(input_uniprots)
    unmapped_count = unmapped_uniprots["uniprot_id"].nunique()
    unmapped_pct = round((unmapped_count / unique_uniprots * 100), 3) if unique_uniprots else 0.0
    missing_ppi_count = ppi_missing_targets["converted_target_id"].nunique()
    converted_target_count = converted_pdi["converted_target_id"].nunique() if not converted_pdi.empty else 0
    missing_ppi_pct = (
        round((missing_ppi_count / converted_target_count * 100), 3)
        if converted_target_count
        else 0.0
    )

    logger.info(
        "UniProt IDs not added because they could not be converted: %d/%d (%.3f%%)",
        unmapped_count,
        unique_uniprots,
        unmapped_pct,
    )
    logger.info(
        "Converted PDI targets not added because they are absent from the PPI: %d/%d (%.3f%%)",
        missing_ppi_count,
        converted_target_count,
        missing_ppi_pct,
    )

    save_outputs(
        drug_background=drug_background,
        pdi_graph=pdi_graph,
        merged_graph=merged_graph,
        netmedpy_ppi=netmedpy_ppi,
        netmedpy_drug_targets=netmedpy_drug_targets,
        drug_background_output=drug_background_output,
        pdi_graph_output=pdi_graph_output,
        merged_graph_output=merged_graph_output,
        netmedpy_ppi_output=netmedpy_ppi_output,
        netmedpy_drug_targets_output=netmedpy_drug_targets_output,
    )

    logger.info("Final PDI rows: %d", len(filtered_pdi))
    logger.info("Final drug background size: %d", len(drug_background))
    logger.info("Final netmedpy drug-target dictionary size: %d", len(netmedpy_drug_targets))


if __name__ == "__main__":
    # sys.argv = [
    #     "prepare_drug_prioritization_inputs.py",
    #     "--ppi",
    #     "../../data/input/networks/string.human_links_v12_0_min700.Ensembl.gt",
    #     "--pdi",
    #     "../../data/nedrexdb_licensed/drug_has_target.csv",
    #     "--drugs",
    #     "../../data/nedrexdb_licensed/drug.csv",
    #     "--prefix",
    #     "string.human_links_v12_0_min700.Ensembl",
    #     "--id-space",
    #     "ensembl",
    #     "-l",
    #     "DEBUG",
    # ]
    # sys.argv = [
    #     "prepare_drug_prioritization_inputs.py",
    #     "--ppi",
    #     "../../data/drugstone_unlicensed/NeDRex_2.1.52_protein-protein-interaction_download.graphml",
    #     "--pdi",
    #     "../../data/drugstone_unlicensed/NeDRex_2.1.52_protein-drug-interaction_download.graphml",
    #     "--drugs",
    #     "../../data/nedrexdb_licensed/drug.csv",
    #     "--prefix",
    #     "NeDRex_2.1.52_protein-protein-interaction_download.Ensembl",
    #     "--id-space",
    #     "ensembl",
    #     "-l",
    #     "DEBUG",
    # ]
    main(parse_args())
