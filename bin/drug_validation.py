#!/usr/bin/env python
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Any, Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_true_drugs import load_csv

# Constants

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logger.addHandler(_handler)

ValidationResult = dict[str, float | int | str]


def normalize_drugbank_id(drug_id: str) -> str:
    """
    Normalize DrugBank IDs to the unprefixed form used for validation joins.
    """
    drug_id = str(drug_id).strip()
    return drug_id[len("drugbank."):] if drug_id.startswith("drugbank.") else drug_id


def normalize_candidate_ranking(candidates: pd.DataFrame | list) -> pd.DataFrame:
    """
    Normalize and deduplicate candidate drugs before assigning evaluation ranks.

    Duplicate candidate drug IDs are collapsed by keeping their highest score so
    DCG, NDCG, overlap, and candidate counts are not inflated by repeated rows.
    """
    if not isinstance(candidates, pd.DataFrame):
        candidates = pd.DataFrame(candidates, columns=["drug_id", "rank", "score"])

    if "drug_id" not in candidates.columns:
        if "drugId" not in candidates.columns:
            raise ValueError("Candidate drugs must contain a 'drug_id' or 'drugId' column.")
        candidates = candidates.rename(columns={"drugId": "drug_id"})

    if "score" not in candidates.columns:
        raise ValueError("Candidate drugs must contain a 'score' column.")

    candidates = candidates.loc[:, ["drug_id", "score"]].dropna(subset=["drug_id", "score"]).copy()
    candidates["drug_id"] = candidates["drug_id"].map(normalize_drugbank_id)
    candidates["score"] = candidates["score"].astype(float)
    candidates = candidates.groupby("drug_id", as_index=False)["score"].max()
    candidates = candidates.sort_values(["score", "drug_id"], ascending=[False, True])
    candidates["rank"] = np.arange(1, len(candidates) + 1)
    return candidates[["drug_id", "rank", "score"]].reset_index(drop=True)


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
    drugs = drugs.rename(columns={"drugId": "drug_id"})
    return normalize_candidate_ranking(drugs)


def calculate_dcg(
        item_relevance: np.ndarray,
        item_scores: np.ndarray,
        k: int | None = None,
        log_base: float = 2.0) -> float:
    """
    Calculate DCG with sklearn-style average tie handling.

    Equal score groups share the average relevance of the group across the rank
    positions occupied by that tied block. Inputs only need to be aligned by
    item; they do not need to be pre-sorted because item_scores induce the rank
    order.

    Args:
        item_relevance: Relevance value for each item, typically 1 for true
            drugs and 0 otherwise.
        item_scores: Prediction or prioritization score for each item, aligned
            with item_relevance.
        k: Optional rank cutoff. Items below the cutoff receive zero discount.
        log_base: Logarithm base used for rank discounts.

    Returns:
        Discounted cumulative gain with average tie handling for equal scores.

    This follows the tied-score averaging method described by McSherry, F. and
    Najork, M. (2008), "Computing Information Retrieval Performance Measures
    Efficiently in the Presence of Tied Scores." Advances in Information
    Retrieval, ECIR 2008, Lecture Notes in Computer Science, vol 4956.
    """
    item_relevance = np.asarray(item_relevance, dtype=float)
    item_scores = np.asarray(item_scores, dtype=float)

    if item_relevance.ndim != 1 or item_scores.ndim != 1:
        raise ValueError("DCG relevance and score arrays must be one-dimensional.")
    if item_relevance.shape[0] != item_scores.shape[0]:
        raise ValueError("DCG relevance and score arrays must have the same length.")
    if item_relevance.size == 0:
        return 0.0

    rank_discounts = 1.0 / (np.log(np.arange(item_relevance.size) + 2) / np.log(log_base))
    if k is not None:
        rank_discounts[int(k):] = 0.0

    cumulative_rank_discounts = np.cumsum(rank_discounts)
    _, score_group_index_by_item, score_group_counts = np.unique(
        -item_scores,
        return_inverse=True,
        return_counts=True,
    )

    mean_relevance_by_score_group = np.zeros(len(score_group_counts), dtype=float)
    np.add.at(mean_relevance_by_score_group, score_group_index_by_item, item_relevance)
    mean_relevance_by_score_group /= score_group_counts

    score_group_end_ranks = np.cumsum(score_group_counts) - 1
    discount_sum_by_score_group = np.empty(len(score_group_counts), dtype=float)
    discount_sum_by_score_group[0] = cumulative_rank_discounts[score_group_end_ranks[0]]
    discount_sum_by_score_group[1:] = np.diff(cumulative_rank_discounts[score_group_end_ranks])

    return float(np.sum(mean_relevance_by_score_group * discount_sum_by_score_group))


def calculate_ndcg(true_drugs: list[str], candidates: pd.DataFrame, observed_dcg: float) -> float:
    """
    Calculate binary-relevance NDCG for the evaluated candidate ranking length.

    NDCG: Normalized effect size of rank quality, scaled relative to the ideal ranking for that disease/background.
    """
    ideal_relevant_count = min(len(true_drugs), len(candidates))
    if ideal_relevant_count == 0:
        return 0.0

    ideal_relevance = np.zeros(len(candidates), dtype=float)
    ideal_relevance[:ideal_relevant_count] = 1.0
    idcg = calculate_dcg(ideal_relevance, ideal_relevance)
    return observed_dcg / idcg if idcg > 0 else 0.0


def calculate_threshold_auc_metrics(
        background_drugs: set[str],
        true_drugs: list[str],
        candidates: pd.DataFrame) -> dict[str, float]:
    """
    Calculate observed threshold metrics over the full drug background.

    Args:
        background_drugs: Drug IDs in the validation background.
        true_drugs: True drug IDs retained in the validation background.
        candidates: DataFrame with drug_id, rank, and score columns.

    Returns:
        Dictionary with observed average precision, observed AUPR, and observed AUC.

    Candidate drugs use their prioritization score. Background drugs absent from
    the candidate ranking receive score 0.

    Average precision: sklearn stepwise precision-recall summary.
    AUPR: Trapezoidal area under the precision-recall curve.
    AUC: Global positive-vs-background ranking separation.
    """
    true_drug_set = set(true_drugs)
    positive_count = len(true_drug_set)
    negative_count = len(background_drugs - true_drug_set)
    if positive_count == 0:
        return {
            "observed average precision": 0.0,
            "observed AUPR": 0.0,
            "observed AUC": float("nan"),
        }

    background_scores = pd.DataFrame({"drug_id": sorted(background_drugs)})
    background_scores["score"] = 0.0

    candidate_scores = candidates.loc[candidates["drug_id"].isin(background_drugs), ["drug_id", "score"]]
    candidate_scores = candidate_scores.groupby("drug_id", as_index=False)["score"].max()
    background_scores = background_scores.drop(columns="score").merge(candidate_scores, on="drug_id", how="left")
    background_scores["score"] = background_scores["score"].fillna(0.0)
    background_scores["is_true_drug"] = background_scores["drug_id"].isin(true_drug_set).astype(int)

    observed_average_precision = average_precision_score(
        background_scores["is_true_drug"],
        background_scores["score"],
    )
    precision, recall, _ = precision_recall_curve(
        background_scores["is_true_drug"],
        background_scores["score"],
    )
    observed_aupr = sklearn_auc(recall, precision)
    observed_auc = (
        roc_auc_score(background_scores["is_true_drug"], background_scores["score"])
        if negative_count > 0
        else float("nan")
    )

    return {
        "observed average precision": observed_average_precision,
        "observed AUPR": observed_aupr,
        "observed AUC": observed_auc,
    }


def calculate_empirical_p_value(random_values: list[float] | np.ndarray, observed_value: float) -> tuple[float, int]:
    """
    Calculate an empirical p-value with plus-one correction.

    NaN random values are ignored. If the observed value is undefined or no valid
    random values remain, the p-value is undefined and returned as NaN.
    """
    if pd.isna(observed_value):
        return float("nan"), 0

    random_values = np.asarray(random_values, dtype=float)
    valid_values = random_values[~np.isnan(random_values)]
    if valid_values.size == 0:
        return float("nan"), 0

    exceed_count = int(np.sum(valid_values >= observed_value))
    p_value = (exceed_count + 1) / (valid_values.size + 1)
    return p_value, exceed_count


def calculate_random_ranking_metrics(
    sampled_is_true_drug: np.ndarray,
    sampled_scores: np.ndarray,
) -> tuple[float, int]:
    """
    Calculate random-ranking DCG and overlap for one permutation.

    Args:
        sampled_is_true_drug: Binary array ordered by sampled rank. 1 means the
            sampled drug at that rank is a true drug.
        sampled_scores: Score array aligned with sampled_is_true_drug.

    Returns:
        Tuple of random DCG and random overlap.
    """
    sampled_is_true_drug = sampled_is_true_drug.astype(int, copy=False)

    random_dcg = calculate_dcg(sampled_is_true_drug, sampled_scores)
    random_overlap = int(np.sum(sampled_is_true_drug))

    return random_dcg, random_overlap


def generate_random_metric_distributions(
        drug_ids: list[str],
        true_drugs: list[str],
        score_template: np.ndarray | list[float],
        count: int,
        seed: Optional[int] = None) -> dict[str, list[float] | list[int]]:
    """
    Generate random metric distributions from sampled drugs assigned to score slots.

    DCG p-value: Tests whether known drugs receive unusually high-ranking score
    slots compared with random drugs from the same drug background while
    preserving the observed ranking score structure, including ties.

    Args:
        drug_ids: Drug IDs to sample from.
        true_drugs: True drug IDs retained in the validation background.
        score_template: Observed candidate scores ordered by evaluation rank.
            These scores are reused for every random permutation so tied blocks
            in the observed ranking are preserved in the null model.
        count: Number of permutations.
        seed: Optional NumPy random generator seed.

    Returns:
        Dictionary containing random DCG and overlap values.
    """
    background_drug_ids = np.asarray(drug_ids)
    true_drug_ids = np.asarray(true_drugs)
    score_template = np.asarray(score_template, dtype=float)

    background_drug_count = len(background_drug_ids)
    sample_size = min(len(score_template), background_drug_count)

    is_true_drug_by_background_index = np.isin(background_drug_ids, true_drug_ids).astype(int)
    random_sample_scores = score_template[:sample_size]  # only use as many score slots as the background size
    rng = np.random.default_rng(seed)

    dcg_values: list[float] = []
    overlap_values: list[int] = []
    for _ in range(count):
        sampled_background_indices = rng.choice(
            background_drug_count,
            size=sample_size,
            replace=False,
        )
        sampled_is_true_drug = is_true_drug_by_background_index[sampled_background_indices]

        random_dcg, random_overlap = calculate_random_ranking_metrics(
            sampled_is_true_drug=sampled_is_true_drug,
            sampled_scores=random_sample_scores,
        )

        dcg_values.append(random_dcg)
        overlap_values.append(random_overlap)

    return {
        "dcg": dcg_values,
        "overlap": overlap_values,
    }

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
            'observed_NDCG',
            'observed_overlap',
            'observed_average_positive_rank',
            'observed_average_precision',
            'observed_AUPR',
            'observed_AUC',
            'dcg_exceed_count',
            'overlap_exceed_count',
            'candidate_count',
            'candidates_absent_from_background_count',
            'background_size',
            'positive_percentage',
            'percent_true_drugs_found',
            'true_drugs_original_count',
            'true_drugs_removed_count',
            'true_drugs_left_count',
            'true_drugs_file',
            'true_drug_ranks_json'
        ]
        with open(output_file, 'w') as f:
            f.write('\t'.join(header) + '\n')
            row = [
                id_value,
                str(result['empirical DCG-based p-value']),
                str(result['empirical p-value without considering ranks']),
                str(result['observed DCG']),
                str(result['observed NDCG']),
                str(result['observed overlap']),
                str(result['observed average positive rank']),
                str(result['observed average precision']),
                str(result['observed AUPR']),
                str(result['observed AUC']),
                str(result['dcg exceed count']),
                str(result['overlap exceed count']),
                str(result['candidate count']),
                str(result['candidates absent from background count']),
                str(result['background size']),
                str(result['positive percentage']),
                str(result['percent true drugs found']),
                str(result['true drugs original count']),
                str(result['true drugs removed count']),
                str(result['true drugs left count']),
                args.true_drugs,
                str(result['true drug ranks json'])
            ]
            f.write('\t'.join(row) + '\n')
        logger.info(f"Results successfully saved to '{output_file}'")
    except IOError as e:
        logger.error(f"Failed to write results to '{output_file}': {e}")
        sys.exit(1)


def drug_list_validation(
        drugs_df: pd.DataFrame,
        true_drugs: list[str],
        candidates: pd.DataFrame | list,
        permutation_count: int) -> ValidationResult:
    """
    Validate a scored candidate drug ranking against true drugs and background drugs.

    Args:
        drugs_df: DataFrame with a drug_id column containing the validation background.
        true_drugs: True drug IDs, with or without a drugbank. prefix.
        candidates: DataFrame or row list with drug_id, rank, and score columns.
        permutation_count: Number of random permutations for empirical p-values.

    Returns:
        Validation statistics and empirical p-value counts.

    """
    if "drug_id" not in drugs_df.columns:
        raise ValueError("Drug background must contain a 'drug_id' column.")

    background_drug_ids = [
        normalize_drugbank_id(drug_id)
        for drug_id in drugs_df["drug_id"].dropna().astype(str).str.strip()
    ]
    background_drug_ids = sorted(set(background_drug_ids))
    background_size = len(background_drug_ids)
    background_drugs = set(background_drug_ids)
    candidates = normalize_candidate_ranking(candidates)
    candidate_drugs = set(candidates["drug_id"])
    candidates_absent_from_background = candidate_drugs - background_drugs
    candidates_absent_from_background_count = len(candidates_absent_from_background)
    if candidates_absent_from_background:
        logger.warning(
            "%d candidate drugs are absent from the validation background",
            candidates_absent_from_background_count,
        )

    ordered_true_drugs = []
    seen_true_drugs = set()
    for drug_id in pd.Series(true_drugs).dropna().astype(str).str.strip():
        normalized_drug_id = normalize_drugbank_id(drug_id)
        if normalized_drug_id not in seen_true_drugs:
            ordered_true_drugs.append(normalized_drug_id)
            seen_true_drugs.add(normalized_drug_id)

    true_drugs_original_count = len(ordered_true_drugs)
    true_drugs = [drug_id for drug_id in ordered_true_drugs if drug_id in background_drugs]
    true_drugs_left_count = len(true_drugs)
    true_drugs_removed_count = true_drugs_original_count - true_drugs_left_count
    positive_percentage = (
        true_drugs_left_count / background_size * 100
        if background_size > 0
        else 0.0
    )
    logger.info(
        "Removed %d true drugs absent from the sampling background; %d remain",
        true_drugs_removed_count,
        true_drugs_left_count,
    )

    if true_drugs_left_count == 0:
        return return_zero_true_drugs_left(candidates, permutation_count, true_drugs_left_count,
                                           true_drugs_original_count, true_drugs_removed_count,
                                           candidates_absent_from_background_count, background_size,
                                           positive_percentage)

    rank_by_drug_id = dict(zip(candidates["drug_id"], candidates["rank"]))
    true_drug_rank_entries = []
    found_true_drug_ranks = []
    for drug_id in true_drugs:
        rank = rank_by_drug_id.get(drug_id)
        if rank is not None:
            rank = int(rank)
            found_true_drug_ranks.append(rank)
        true_drug_rank_entries.append({
            "drug_id": drug_id,
            "rank": rank,
        })
    true_drug_ranks_json = json.dumps(true_drug_rank_entries, separators=(",", ":"))
    observed_average_positive_rank = (
        float(np.mean(found_true_drug_ranks))
        if found_true_drug_ranks
        else float("nan")
    )

    observed_relevance = candidates["drug_id"].isin(true_drugs).to_numpy(dtype=float)
    observed_scores = candidates["score"].to_numpy(dtype=float)
    dcg_observed = calculate_dcg(observed_relevance, observed_scores)
    ndcg_observed = calculate_ndcg(true_drugs, candidates, dcg_observed)
    threshold_metrics = calculate_threshold_auc_metrics(background_drugs, true_drugs, candidates)
    # Log how many drugs were observed
    logger.info(f"Observed DCG: {dcg_observed} for {len(candidates)} candidate drugs")
    logger.info(f"Observed NDCG: {ndcg_observed}")
    logger.info(
        "Observed average precision: %s; observed AUPR: %s; observed AUC: %s",
        threshold_metrics["observed average precision"],
        threshold_metrics["observed AUPR"],
        threshold_metrics["observed AUC"],
    )
    random_metrics = generate_random_metric_distributions(
        background_drug_ids,
        true_drugs,
        score_template=candidates["score"],
        count=permutation_count,
    )
    logger.debug(
        "Generated random metric values: DCG=%d, overlap=%d",
        len(random_metrics["dcg"]),
        len(random_metrics["overlap"]),
    )
    # empirical DCG-based p-value
    p_value_dcg, exceed_dcg = calculate_empirical_p_value(random_metrics["dcg"], dcg_observed)
    # observed overlap ignoring ranks
    observed_overlap = int(candidates.loc[candidates["drug_id"].isin(true_drugs), "drug_id"].nunique())
    # empirical overlap-based p-value
    p_value_overlap, exceed_overlap = calculate_empirical_p_value(random_metrics["overlap"], observed_overlap)
    logger.info(f"Observed DCG: {dcg_observed}, DCG p-value: {p_value_dcg}")
    logger.info(f"Observed overlap: {observed_overlap}, overlap p-value: {p_value_overlap}")
    logger.info(
        "Observed average precision: %s; observed AUPR: %s; observed AUC: %s",
        threshold_metrics["observed average precision"],
        threshold_metrics["observed AUPR"],
        threshold_metrics["observed AUC"],
    )
    # number of exceeding cases
    logger.info(f"Number of random DCG values exceeding observed: {exceed_dcg} out of {permutation_count}")
    logger.info("Validation completed successfully")
    return {
        "empirical DCG-based p-value": p_value_dcg,
        "empirical p-value without considering ranks": p_value_overlap,
        "observed DCG": dcg_observed,
        "observed NDCG": ndcg_observed,
        "observed overlap": observed_overlap,
        "observed average positive rank": observed_average_positive_rank,
        "observed average precision": threshold_metrics["observed average precision"],
        "observed AUPR": threshold_metrics["observed AUPR"],
        "observed AUC": threshold_metrics["observed AUC"],
        "dcg exceed count": exceed_dcg,
        "overlap exceed count": exceed_overlap,
        "candidate count": len(candidates),
        "candidates absent from background count": candidates_absent_from_background_count,
        "background size": background_size,
        "positive percentage": positive_percentage,
        "percent true drugs found": (observed_overlap / len(true_drugs) * 100) if candidates.shape[0] > 0 else 0.0,
        "true drugs original count": true_drugs_original_count,
        "true drugs removed count": true_drugs_removed_count,
        "true drugs left count": true_drugs_left_count,
        "true drug ranks json": true_drug_ranks_json,
    }


def return_zero_true_drugs_left(candidates: DataFrame, permutation_count: int, true_drugs_left_count: Literal[0],
                                true_drugs_original_count: int, true_drugs_removed_count: int,
                                candidates_absent_from_background_count: int, background_size: int,
                                positive_percentage: float) -> dict[
    str | Any, float | int | Any]:
    logger.warning("No true drugs remain after filtering to the sampling background")
    return {
        "empirical DCG-based p-value": 1.0,
        "empirical p-value without considering ranks": 1.0,
        "observed DCG": 0.0,
        "observed NDCG": 0.0,
        "observed overlap": 0,
        "observed average positive rank": float("nan"),
        "observed average precision": 0.0,
        "observed AUPR": 0.0,
        "observed AUC": float("nan"),
        "dcg exceed count": permutation_count,
        "overlap exceed count": permutation_count,
        "candidate count": len(candidates),
        "candidates absent from background count": candidates_absent_from_background_count,
        "background size": background_size,
        "positive percentage": positive_percentage,
        "percent true drugs found": 0.0,
        "true drugs original count": true_drugs_original_count,
        "true drugs removed count": true_drugs_removed_count,
        "true drugs left count": true_drugs_left_count,
        "true drug ranks json": "[]",
    }


if __name__ == '__main__':
    # sys.argv = [
    #     'drug_validation.py',
    #     '--candidate-drugs',
    #     '../../results/dmd_mondo_0004975_alzheimer_limited/entrez/drug_prioritization/offline_drug_prioritization/seeds_entrez.nedrex.reviewed_proteins_exp.Entrez.no_tool.trustrank.csv',
    #     '--drug-list',
    #     '../../results/offline_drug_prioritization_preprocessed/entrez/nedrex.reviewed_proteins_exp.Entrez.drug_background.tsv',
    #     '--true-drugs',
    #     '../../data/input/true_approved_drugs_with_targets_by_disease/mondo.0004975.csv',
    #     '--permutation-count',
    #     '10000',
    #     '--out-dir',
    #     '../../data/drug_validation_results',
    # ]
    main(parse_args())
