"""
Run the full pipeline on the TEST set and write the two required files:
  - output/candidate_pairs.tsv   (blocking output, last stage before the model)
  - output/matching_results.tsv  (final thresholded matches, scored on leaderboard)

Run this AFTER training/threshold-tuning on the train split.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from io_utils import load_source, write_pairs_tsv
from normalize_parallel import add_normalized_columns_parallel as add_normalized_columns
from blocking import build_candidates, candidates_to_frame
from features import build_feature_matrix, frame_to_lookup
from train_model import predict_with_ensemble


def run_test_pipeline(
    test_dir: str,
    models: list,
    best_threshold: float,
    output_dir: str,
    top_k: int = 15,
    sim_threshold: float = 0.35,
):
    os.makedirs(output_dir, exist_ok=True)

    s1 = load_source(os.path.join(test_dir, "test_source1.tsv"))
    s2 = load_source(os.path.join(test_dir, "test_source2.tsv"))
    s3 = load_source(os.path.join(test_dir, "test_source3.tsv"))

    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)
    other = pd.concat([s2, s3], ignore_index=True)

    s1_ids = list(s1["entity_id"].values)

    # --- Blocking: this IS candidate_pairs.tsv -------------------------
    candidates = build_candidates(s1, other, top_k=top_k, sim_threshold=sim_threshold)
    write_pairs_tsv(
        {k: list(v) for k, v in candidates.items()},
        s1_ids,
        os.path.join(output_dir, "candidate_pairs.tsv"),
        id_col="candidate_entity_ids",
    )

    # --- Score every candidate with the trained ensemble ----------------
    pairs_df = candidates_to_frame(candidates, s1_ids)
    pairs_df = pairs_df.rename(columns={"candidate_entity_id": "candidate_entity_id"})

    if len(pairs_df) == 0:
        write_pairs_tsv({}, s1_ids, os.path.join(output_dir, "matching_results.tsv"),
                         id_col="matched_entity_ids")
        return

    s1_lookup = frame_to_lookup(s1)
    other_lookup = frame_to_lookup(other)
    X = build_feature_matrix(pairs_df, s1_lookup, other_lookup)
    pairs_df["prob"] = predict_with_ensemble(models, X)

    # --- Apply the tuned threshold; final matches are a strict subset of
    #     candidates by construction ------------------------------------
    matches = {}
    for s1_id, group in pairs_df.groupby("source1_entity_id"):
        kept = group.loc[group["prob"] >= best_threshold, "candidate_entity_id"].tolist()
        matches[s1_id] = kept

    write_pairs_tsv(matches, s1_ids, os.path.join(output_dir, "matching_results.tsv"),
                     id_col="matched_entity_ids")

    print(f"Wrote {os.path.join(output_dir, 'candidate_pairs.tsv')}")
    print(f"Wrote {os.path.join(output_dir, 'matching_results.tsv')}")
    print(f"Total S1 entities: {len(s1_ids)} | with >=1 candidate: "
          f"{sum(1 for v in candidates.values() if v)} | with >=1 final match: "
          f"{sum(1 for v in matches.values() if v)}")
