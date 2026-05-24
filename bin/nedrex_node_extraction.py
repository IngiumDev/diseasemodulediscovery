#!/usr/bin/env python
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_NEDREX_API_key import get_pagination_max
from validation_utils import join_url_path

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))

logger.addHandler(_handler)


def list_node_collections(endpoint_url: str, api_key: str = None) -> list:
    """
    Fetch list of node collections from the NeDRex API.
    Parameters:
        endpoint_url (str): Base URL of the API (e.g., 'https://api.nedrex.net/').
        api_key (str): Optional API key for authentication.
    Returns:
        list: A list of collection names.
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    # Ensure proper URL joining
    url = join_url_path(endpoint_url, "list_node_collections")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def list_edge_collections(endpoint_url: str, api_key: str = None) -> list:
    """
    Fetch list of edge collections from the NeDRex API.
    Parameters:
        endpoint_url (str): Base URL of the API (e.g., 'https://api.nedrex.net/').
        api_key (str): Optional API key for authentication.
    Returns:
        list: A list of collection names.
    """
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    # Ensure proper URL joining
    url = join_url_path(endpoint_url, "list_edge_collections")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def get_collection_count(endpoint_url: str, collection: str, api_key: str = None) -> Optional[int]:
    """Fetch the number of items in a NeDRex collection, if available."""
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    url = join_url_path(endpoint_url, collection, "details")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        details = response.json()
    except requests.RequestException as exc:
        logger.debug("Could not fetch details for collection '%s': %s", collection, exc)
        return None

    if isinstance(details, dict):
        count = details.get("count")
        return count if isinstance(count, int) else None

    if isinstance(details, list) and details:
        count = details[0].get("count")
        return count if isinstance(count, int) else None

    return None


def filter_valid_collections(requested_collections: list[str], valid_collections: set[str]) -> list[str]:
    """Return requested collections that exist and log the skipped collection names."""
    valid_requested = [collection for collection in requested_collections if collection in valid_collections]
    invalid_requested = [collection for collection in requested_collections if collection not in valid_collections]

    if valid_requested:
        logger.info("Valid collections to download: %s", ", ".join(valid_requested))

    if invalid_requested:
        logger.warning("Skipping invalid collections: %s", ", ".join(invalid_requested))

    return valid_requested


# api key can be null, return is pandas dataframe
def fetch_nodes(
    endpoint_url: str,
    collection: str,
    api_key: str = None,
    pagination_max: int = None,
    progress_bar: tqdm = None,
) -> pd.DataFrame:
    """
    Fetch nodes from the NeDRex API.
    Parameters:
        endpoint_url (str): The URL of the endpoint to fetch nodes from.
        api_key (str): The API key for authentication.
        pagination_max (int): Optional pre-fetched pagination limit.
        progress_bar (tqdm): Optional shared progress bar to update per fetched batch.

    Returns:
        dict: The response data from the API.
    """

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key
    offset = 0
    all_records = []

    if pagination_max is None:
        pagination_max = get_pagination_max(api_key, endpoint_url)

    url = join_url_path(endpoint_url, collection, "all")

    logger.info("Downloading collection '%s'", collection)

    while True:
        params = {"offset": offset, "limit": pagination_max}
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        if resp.status_code in {404, 422}:
            raise RuntimeError(f"Failed to fetch nodes: {resp.status_code} {resp.text}")

        batch = resp.json()
        all_records.extend(batch)
        if progress_bar is not None:
            progress_bar.set_description(f"Downloading {collection}")
            progress_bar.update(len(batch))

        # stop when we get fewer than limit
        if len(batch) < pagination_max:
            break
        offset += pagination_max

    logger.info("Finished collection '%s' with %d records", collection, len(all_records))
    return pd.DataFrame(all_records)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch specified node collections from the NeDRex API and save to CSV files")
    parser.add_argument('--base-url', '-u', default='https://api.nedrex.net/licensed',
                        help='Base URL of the NeDRex API')
    parser.add_argument('-c', '--collections', nargs='+', required=True,
                        help='One or more collection names to download')
    parser.add_argument('--output', '-o', required=True, help='Output directory to save CSV files')
    parser.add_argument('--api-key', '-k', dest='api_key', help='API key for authentication')
    args = parser.parse_args()
    if args.api_key:
        api_key = args.api_key
        logger.info("Using API key from command line argument")
    else:
        api_key = os.getenv("NEDREX_LICENSED_KEY")
        if api_key:
            logger.info("Loaded API key from environment variable NEDREX_LICENSED_KEY")
        else:
            logger.warning(
                "No API key provided (via --api-key or environment); requests may fail if authentication is required")

    node_collections = list_node_collections(args.base_url, api_key)
    edge_collections = list_edge_collections(args.base_url, api_key)
    valid_collections = set(node_collections + edge_collections)
    collections_to_download = filter_valid_collections(args.collections, valid_collections)

    if not collections_to_download:
        logger.error("No valid collections were requested. Nothing to download.")
        return

    pagination_max = get_pagination_max(api_key, args.base_url)
    collection_counts = {
        collection: get_collection_count(args.base_url, collection, api_key)
        for collection in collections_to_download
    }
    total_records = sum(count for count in collection_counts.values() if count is not None)

    os.makedirs(args.output, exist_ok=True)

    with tqdm(total=total_records, desc="Downloading collections", unit="records") as progress_bar:
        for collection in collections_to_download:
            try:
                progress_bar.set_description(f"Downloading {collection}")
                df = fetch_nodes(args.base_url, collection, api_key, pagination_max, progress_bar)
                output_path = os.path.join(args.output, f"{collection}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved collection '{collection}' to {output_path}")
            except Exception as e:
                logger.error(f"Failed to fetch or save collection '{collection}': {e}")




if __name__ == '__main__':
    # sys.argv = ['nedrex_node_extraction.py', '--base-url', 'https://api.nedrex.net/licensed',
    #             '--output', './data', '--api-key', '<MISSING>', '--collections',
    #             'disorder', 'drug', 'gene', 'genomic_variant', 'go', 'pathway', 'phenotype', 'protein', 'side_effect',
    #             'signature', 'tissue', 'disorder_has_phenotype', 'disorder_is_subtype_of_disorder',
    #             'drug_has_contraindication', 'drug_has_indication', 'drug_has_side_effect', 'drug_has_target',
    #             'gene_associated_with_disorder', 'gene_expressed_in_tissue', 'go_is_subtype_of_go',
    #             'molecule_similarity_molecule', 'protein_encoded_by_gene', 'protein_expressed_in_tissue',
    #             'protein_has_go_annotation', 'protein_has_signature', 'protein_in_pathway',
    #             'protein_interacts_with_protein', 'side_effect_same_as_phenotype', 'variant_affects_gene',
    #             'variant_associated_with_disorder']
    # sys.argv = ['nedrex_node_extraction.py','--base-url', 'https://api.nedrex.net/open', '--collections', 'disorder', 'drug', 'gene', 'genomic_variant', 'go', 'pathway', 'phenotype', 'protein', 'side_effect',
    #                 'signature', 'tissue', 'disorder_has_phenotype', 'disorder_is_subtype_of_disorder',
    #                 'drug_has_contraindication', 'drug_has_indication', 'drug_has_side_effect', 'drug_has_target',
    #                 'gene_associated_with_disorder', 'gene_expressed_in_tissue', 'go_is_subtype_of_go',
    #                 'molecule_similarity_molecule', 'protein_encoded_by_gene', 'protein_expressed_in_tissue',
    #                 'protein_has_go_annotation', 'protein_has_signature', 'protein_in_pathway',
    #                 'protein_interacts_with_protein', 'side_effect_same_as_phenotype', 'variant_affects_gene',
    #                 'variant_associated_with_disorder', '--output', './']
    # sys.argv = ['nedrex_node_extraction.py', '--base-url', 'https://api.nedrex.net/open', '--collections', 'disorder',
    #             'drug', 'gene', 'disorder_is_subtype_of_disorder'
    #             , 'drug_has_target',
    #             'gene_associated_with_disorder', '--output', './']
    main()
