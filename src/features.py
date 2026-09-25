"""
Pairwise feature engineering: given a (Source 1 record, candidate record)
pair, compute a fixed-width numeric feature vector for the matching model.

All features are computed from string/set similarity — nothing here
depends on an external service, database, or geocoder, per the
challenge's no-external-lookup rule.
"""

import numpy as np
from rapidfuzz import fuzz


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _numeric_overlap(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


FEATURE_NAMES = [
    "name_levenshtein_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_jaccard",
    "name_sorted_exact_match",
    "name_len_diff",
    "addr_levenshtein_ratio",
    "addr_token_sort_ratio",
    "addr_jaccard",
    "addr_numeric_overlap",
    "addr_len_diff",
    "landmark_present_both",
    "country_match",
    "name_first_token_match",
]


def pair_features(s1_row: dict, cand_row: dict) -> list:
    """
    s1_row / cand_row: dict-like with keys normalized_name, name_sorted,
    normalized_address, address_numbers, landmark, country (raw column).
    Returns a list of floats in FEATURE_NAMES order.
    """
    n1, n2 = s1_row["normalized_name"], cand_row["normalized_name"]
    a1, a2 = s1_row["normalized_address"], cand_row["normalized_address"]

    feats = [
        fuzz.ratio(n1, n2) / 100.0,
        fuzz.token_sort_ratio(n1, n2) / 100.0,
        fuzz.token_set_ratio(n1, n2) / 100.0,
        fuzz.partial_ratio(n1, n2) / 100.0,
        _jaccard(n1, n2),
        1.0 if s1_row["name_sorted"] and s1_row["name_sorted"] == cand_row["name_sorted"] else 0.0,
        abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1),
        fuzz.ratio(a1, a2) / 100.0,
        fuzz.token_sort_ratio(a1, a2) / 100.0,
        _jaccard(a1, a2),
        _numeric_overlap(s1_row["address_numbers"], cand_row["address_numbers"]),
        abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1),
        1.0 if (s1_row["landmark"] and cand_row["landmark"]) else 0.0,
        1.0 if (s1_row["country"] and s1_row["country"] == cand_row["country"]) else 0.0,
        1.0 if (n1.split()[:1] == n2.split()[:1] and n1) else 0.0,
    ]
    return feats


def build_feature_matrix(pairs_df, s1_lookup: dict, other_lookup: dict):
    """
    pairs_df: DataFrame with columns [source1_entity_id, candidate_entity_id
              (or matched_entity_id for training)].
    s1_lookup / other_lookup: entity_id -> row dict, precomputed once for speed.
    Returns: np.ndarray of shape (len(pairs_df), len(FEATURE_NAMES))
    """
    id_col = "candidate_entity_id" if "candidate_entity_id" in pairs_df.columns else "matched_entity_id"
    rows = []
    for s1_id, other_id in zip(pairs_df["source1_entity_id"].values, pairs_df[id_col].values):
        s1_row = s1_lookup.get(s1_id)
        other_row = other_lookup.get(other_id)
        if s1_row is None or other_row is None:
            rows.append([0.0] * len(FEATURE_NAMES))
            continue
        rows.append(pair_features(s1_row, other_row))
    return np.array(rows, dtype=float)


def frame_to_lookup(df) -> dict:
    """Build an entity_id -> row-dict lookup for fast repeated feature computation."""
    cols = ["entity_id", "normalized_name", "name_sorted", "normalized_address",
            "address_numbers", "landmark", "country"]
    return {row["entity_id"]: {c: row[c] for c in cols} for _, row in df[cols].iterrows()}
