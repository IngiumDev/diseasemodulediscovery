#!/usr/bin/env python
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_true_drugs import load_csv

# Constants

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_handler)

ValidationResult = dict[str, float | int]


def load_drugs_list(file_path: str) -> pd.DataFrame:
    """
    Load a candidate drug ranking file.

    Args:
        file_path: Path to a candidate drug CSV with drugId, score, and isResult columns.

    Returns:
        DataFrame with drug_id, rank, and score columns. Rows are sorted by
        score descending and drug_id ascending before ranks are assigned.
    """
    drugs = pd.read_csv(file_path, usecols=["drugId", "score", "isResult"])
    drugs = drugs.loc[drugs["isResult"].astype(str).str.lower() == "true", ["drugId", "score"]]
    drugs = drugs.assign(
        score=drugs["score"].astype(float),
        drug_id=drugs["drugId"].astype(str).str.strip().str.removeprefix("drugbank."),
    )
    drugs = drugs.sort_values(["score", "drug_id"], ascending=[False, True])
    drugs["rank"] = range(1, len(drugs) + 1)
    drugs = drugs[["drug_id", "rank", "score"]].reset_index(drop=True)

    return drugs


def calculate_dcg(true_drugs: list[str], candidates: pd.DataFrame) -> float:
    """
    Calculate DCG from candidate drug ranks and binary true-drug relevance.

    Args:
        true_drugs: True drug IDs in unprefixed DrugBank form.
        candidates: DataFrame with drug_id, rank, and score columns.

    Returns:
        Discounted cumulative gain for the candidate ranking.
    """
    relevance = candidates["drug_id"].isin(true_drugs).to_numpy(dtype=float)
    ranks = candidates["rank"].to_numpy(dtype=float)
    discounts = 1.0 / np.log2(ranks + 1)
    return float(np.sum(relevance * discounts))


def calculate_threshold_auc_metrics(
        background_drugs: set[str],
        true_drugs: list[str],
        candidates: pd.DataFrame) -> dict[str, float]:
    """
    Calculate score-threshold AUPR and ROC AUC over the full drug background.

    Args:
        background_drugs: Drug IDs in the validation background.
        true_drugs: True drug IDs retained in the validation background.
        candidates: DataFrame with drug_id, rank, and score columns.

    Returns:
        Dictionary with observed AUPR and observed AUC.

    Candidate drugs use their prioritization score. Background drugs absent from
    the candidate ranking receive score 0.
    """
    true_drug_set = set(true_drugs)
    positive_count = len(true_drug_set)
    negative_count = len(background_drugs - true_drug_set)
    if positive_count == 0:
        return {"observed AUPR": 0.0, "observed AUC": float("nan")}

    background_scores = pd.DataFrame({"drug_id": sorted(background_drugs)})
    background_scores["score"] = 0.0

    candidate_scores = candidates.loc[candidates["drug_id"].isin(background_drugs), ["drug_id", "score"]]
    candidate_scores = candidate_scores.groupby("drug_id", as_index=False)["score"].max()
    background_scores = background_scores.drop(columns="score").merge(candidate_scores, on="drug_id", how="left")
    background_scores["score"] = background_scores["score"].fillna(0.0)
    background_scores["is_true_drug"] = background_scores["drug_id"].isin(true_drug_set).astype(int)

    aupr = average_precision_score(background_scores["is_true_drug"], background_scores["score"])
    auc = (
        roc_auc_score(background_scores["is_true_drug"], background_scores["score"])
        if negative_count > 0
        else float("nan")
    )

    return {
        "observed AUPR": aupr,
        "observed AUC": auc,
    }


def generate_random_distributions(
        drug_ids: list[str],
        true_drugs: list[str],
        length: int,
        count: int,
        seed: Optional[int] = None) -> tuple[list[float], list[int]]:
    """
    Generate random DCG and overlap distributions from sampled drug rankings.

    Args:
        drug_ids: Drug IDs to sample from.
        true_drugs: True drug IDs retained in the validation background.
        length: Number of drugs to sample per permutation.
        count: Number of permutations.
        seed: Optional NumPy random generator seed.

    Returns:
        Tuple containing random DCG values and random overlap counts.
    """
    sample_size = min(length, len(drug_ids))
    drug_ids = np.asarray(drug_ids)
    true_drugs = np.asarray(true_drugs)
    discounts = 1.0 / np.log2(np.arange(1, sample_size + 1) + 1)
    rng = np.random.default_rng(seed)

    dcg_values: list[float] = []
    overlap_values: list[int] = []
    for _ in range(count):
        sample_ids = rng.choice(drug_ids, size=sample_size, replace=False)
        relevance = np.isin(sample_ids, true_drugs)
        dcg_values.append(float(np.sum(relevance * discounts)))
        overlap_values.append(int(np.sum(relevance)))
    return dcg_values, overlap_values


def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments.

    Args:
        None.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Perform (local) statistical validation of drug prioritization results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--candidate-drugs', required=True, help="Candidate drug results from drug prioritization tool")
    parser.add_argument('--drug-list', required=True, help="Path to the complete drug list CSV file for sampling")
    parser.add_argument('--permutation-count', type=int, default=10000,
                        help="Number of random permutations to generate for DCG comparison")
    parser.add_argument('--true-drugs', required=True,
                        help="Path to the true drugs list CSV file containing ground truth relevant drug IDs")
    parser.add_argument('--out-dir', required=True, help="Directory to save the validation results")
    parser.add_argument('--output-file', default='drug_validation_results.tsv',
                        help="Output file name for the validation results in TSV format")
    parser.add_argument('--id', default=None,
                        help="ID for output row; defaults to basename of candidate-drugs file")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Run drug validation from parsed command-line arguments.

    Args:
        args: Parsed command-line arguments from parse_args().

    Returns:
        None.
    """
    # Derive output ID from candidate-drugs basename if not provided
    if args.id is None:
        id_value = os.path.splitext(os.path.basename(args.candidate_drugs))[0]
    else:
        id_value = args.id
    # Verify that the output directory exists or create it, and ensure it's a directory
    if not os.path.exists(args.out_dir):
        try:
            os.makedirs(args.out_dir)
            logger.info(f"Created output directory '{args.out_dir}'")
        except OSError as e:
            logging.error(f"Could not create output directory '{args.out_dir}': {e}")
            sys.exit(1)
    elif not os.path.isdir(args.out_dir):
        logging.error(f"Output path '{args.out_dir}' exists and is not a directory")
        sys.exit(1)

    true_drugs = load_csv(args.true_drugs)["trueDrugs"].tolist()

    # Load candidate drugs
    candidates = load_drugs_list(args.candidate_drugs)
    logger.debug(f"Loaded {len(candidates)} candidate drugs from {args.candidate_drugs}")
    drugs_df = pd.read_csv(args.drug_list, sep="\t")

    # generate distributions of DCG values and overlap counts
    permutation_count = args.permutation_count

    result = drug_list_validation(drugs_df, true_drugs, candidates, permutation_count)
    # Save results to TSV file
    output_file = os.path.join(args.out_dir, args.output_file)
    try:
        header = [
            'ID',
            'empirical_DCG_based_p_value',
            'empirical_p_value_without_considering_ranks',
            'observed_DCG',
            'observed_overlap',
            'observed_AUPR',
            'observed_AUC',
            'dcg_exceed_count',
            'overlap_exceed_count',
            'candidate_count',
            'percent_true_drugs_found',
            # TODO: Update MultiQC parsing/config to consume these true-drug background filtering columns.
            'true_drugs_original_count',
            'true_drugs_removed_count',
            'true_drugs_left_count',
            'true_drugs_file'
        ]
        with open(output_file, 'w') as f:
            f.write('\t'.join(header) + '\n')
            row = [
                id_value,
                str(result['empirical DCG-based p-value']),
                str(result['empirical p-value without considering ranks']),
                str(result['observed DCG']),
                str(result['observed overlap']),
                str(result['observed AUPR']),
                str(result['observed AUC']),
                str(result['dcg exceed count']),
                str(result['overlap exceed count']),
                str(result['candidate count']),
                str(result['percent true drugs found']),
                str(result['true drugs original count']),
                str(result['true drugs removed count']),
                str(result['true drugs left count']),
                args.true_drugs
            ]
            f.write('\t'.join(row) + '\n')
        logger.info(f"Results successfully saved to '{output_file}'")
    except IOError as e:
        logger.error(f"Failed to write results to '{output_file}': {e}")
        sys.exit(1)


def drug_list_validation(
        drugs_df: pd.DataFrame,
        true_drugs: list[str],
        candidates: pd.DataFrame,
        permutation_count: int) -> ValidationResult:
    """
    Validate a scored candidate drug ranking against true drugs and background drugs.

    Args:
        drugs_df: DataFrame with a drug_id column containing the validation background.
        true_drugs: True drug IDs, with or without a drugbank. prefix.
        candidates: DataFrame with drug_id, rank, and score columns.
        permutation_count: Number of random permutations for empirical p-values.

    Returns:
        Validation statistics and empirical p-value counts.

    """
    if "drug_id" not in drugs_df.columns:
        raise ValueError("Drug background must contain a 'drug_id' column.")

    background_drug_ids = drugs_df["drug_id"].dropna().astype(str).str.strip().tolist()
    background_drugs = set(background_drug_ids)

    cleaned_true_drugs = set()
    for drug_id in pd.Series(true_drugs).dropna().astype(str).str.strip():
        cleaned_true_drugs.add(drug_id[len("drugbank."):] if drug_id.startswith("drugbank.") else drug_id)
    true_drugs = cleaned_true_drugs

    true_drugs_original_count = len(true_drugs)
    true_drugs = sorted(true_drugs.intersection(background_drugs))
    true_drugs_left_count = len(true_drugs)
    true_drugs_removed_count = true_drugs_original_count - true_drugs_left_count
    logger.info(
        "Removed %d true drugs absent from the sampling background; %d remain",
        true_drugs_removed_count,
        true_drugs_left_count,
    )

    if true_drugs_left_count == 0:
        logger.warning("No true drugs remain after filtering to the sampling background")
        return {
            "empirical DCG-based p-value": 1.0,
            "empirical p-value without considering ranks": 1.0,
            "observed DCG": 0.0,
            "observed overlap": 0,
            "dcg exceed count": permutation_count,
            "overlap exceed count": permutation_count,
            "candidate count": len(candidates),
            "percent true drugs found": 0.0,
            "observed AUPR": 0.0,
            "observed AUC": float("nan"),
            "true drugs original count": true_drugs_original_count,
            "true drugs removed count": true_drugs_removed_count,
            "true drugs left count": true_drugs_left_count,
        }

    dcg_observed = calculate_dcg(true_drugs, candidates)
    threshold_metrics = calculate_threshold_auc_metrics(background_drugs, true_drugs, candidates)
    # Log how many drugs were observed
    logger.info(f"Observed DCG: {dcg_observed} for {len(candidates)} candidate drugs")
    dcg_random, overlap_random = generate_random_distributions(background_drug_ids, true_drugs, length=len(candidates),
                                                               count=permutation_count)
    logger.debug(f"Generated {len(dcg_random)} random DCG values and {len(overlap_random)} overlap counts")
    # empirical DCG-based p-value
    exceed_dcg = sum(1 for v in dcg_random if v >= dcg_observed)
    p_value_dcg = (exceed_dcg + 1) / (permutation_count + 1)
    # observed overlap ignoring ranks
    observed_overlap = int(candidates["drug_id"].isin(true_drugs).sum())
    # empirical overlap-based p-value
    exceed_overlap = sum(1 for o in overlap_random if o >= observed_overlap)
    p_value_overlap = (exceed_overlap + 1) / (permutation_count + 1)
    logger.info(f"Observed DCG: {dcg_observed}, DCG p-value: {p_value_dcg}")
    logger.info(f"Observed overlap: {observed_overlap}, overlap p-value: {p_value_overlap}")
    # number of exceeding cases
    logger.info(f"Number of random DCG values exceeding observed: {exceed_dcg} out of {permutation_count}")
    logger.info("Validation completed successfully")
    return {
        "empirical DCG-based p-value": p_value_dcg,
        "empirical p-value without considering ranks": p_value_overlap,
        "observed DCG": dcg_observed,
        "observed overlap": observed_overlap,
        "dcg exceed count": exceed_dcg,
        "overlap exceed count": exceed_overlap,
        "candidate count": len(candidates),
        "percent true drugs found": (observed_overlap / len(true_drugs) * 100) if candidates.shape[0] > 0 else 0.0,
        "observed AUPR": threshold_metrics["observed AUPR"],
        "observed AUC": threshold_metrics["observed AUC"],
        "true drugs original count": true_drugs_original_count,
        "true drugs removed count": true_drugs_removed_count,
        "true drugs left count": true_drugs_left_count,
    }


if __name__ == '__main__':
    # sys.argv = [
    #     'drug_validation.py',
    #     '--candidate-drugs',
    #     '../../results/dmd_mondo_0004975_alzheimer_limited/entrez/drug_prioritization/offline_drug_prioritization/seeds_entrez.nedrex.reviewed_proteins_exp.Entrez.no_tool.trustrank.csv',
    #     '--drug-list',
    #     '../../data/input/drug_prioritization_inputs/string.human_links_v12_0_min700.Ensembl.drug_background.tsv',
    #     '--true-drugs',
    #     '../../data/input/true_approved_drugs_with_targets_by_disease/mondo.0004975.csv',
    #     '--permutation-count',
    #     '1000000',
    #     '--out-dir',
    #     '../../data/drug_validation_results',
    # ]
    main(parse_args())
