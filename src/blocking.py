"""
Blocking / candidate generation.

Goal: for every Source 1 entity, cheaply produce a shortlist of Source 2 /
Source 3 records that are plausibly the same business, with recall as
high as we can afford (this sets the ceiling for everything downstream —
the matching model can only ever be as good as this shortlist).

Strategy (union of three independent blocking signals, so a record only
needs to survive ONE of them to become a candidate):
  1. Character n-gram TF-IDF cosine similarity over name+address, via
     approximate nearest neighbors (works well against typos and does
     not depend on any fixed vocabulary/country list).
  2. Exact match on a normalized "sorted name tokens" key (handles exact
     dupes / pure suffix noise cheaply, no similarity search needed).
  3. Shared numeric token (house number / PIN / ZIP) as a fallback net,
     since these often survive even heavy address paraphrasing.

All three are computed per-country-agnostic; country is not used to
filter blocking (per the challenge rules), only as a downstream feature.
"""

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def _char_ngram_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)


def build_candidates(
    s1_df,
    other_df,
    top_k: int = 15,
    sim_threshold: float = 0.35,
):
    """
    s1_df, other_df: DataFrames already run through
        normalize.add_normalized_columns(), each with an 'entity_id' column.
    other_df is the concatenation of source2 + source3 records (their
    entity_id prefixes already distinguish them, so this can be a single
    combined pool for one nearest-neighbor index).

    Returns: dict {source1_entity_id: set(candidate_entity_id)}
    """
    candidates = defaultdict(set)

    # --- Signal 1: TF-IDF char n-gram nearest neighbors -----------------
    s1_text = (s1_df["normalized_name"] + " " + s1_df["normalized_address"]).values
    other_text = (other_df["normalized_name"] + " " + other_df["normalized_address"]).values

    vectorizer = _char_ngram_vectorizer()
    all_text = np.concatenate([s1_text, other_text])
    vectorizer.fit(all_text)

    s1_vecs = vectorizer.transform(s1_text)
    other_vecs = vectorizer.transform(other_text)

    k = min(top_k, len(other_df)) if len(other_df) > 0 else 0
    if k > 0:
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
        nn.fit(other_vecs)
        distances, indices = nn.kneighbors(s1_vecs)
        for row_i, s1_id in enumerate(s1_df["entity_id"].values):
            for dist, other_i in zip(distances[row_i], indices[row_i]):
                similarity = 1.0 - dist
                if similarity >= sim_threshold:
                    candidates[s1_id].add(other_df["entity_id"].values[other_i])

    # --- Signal 2: exact match on sorted normalized name tokens ---------
    name_key_index = defaultdict(list)
    for idx, key in enumerate(other_df["name_sorted"].values):
        if key:
            name_key_index[key].append(other_df["entity_id"].values[idx])
    for idx, key in enumerate(s1_df["name_sorted"].values):
        if key and key in name_key_index:
            s1_id = s1_df["entity_id"].values[idx]
            candidates[s1_id].update(name_key_index[key])

    # --- Signal 3: shared numeric address token (house #/PIN/ZIP) -------
    num_index = defaultdict(list)
    for idx, nums in enumerate(other_df["address_numbers"].values):
        for n in nums:
            num_index[n].append(other_df["entity_id"].values[idx])
    for idx, nums in enumerate(s1_df["address_numbers"].values):
        if not nums:
            continue
        s1_id = s1_df["entity_id"].values[idx]
        matched = set()
        for n in nums:
            matched.update(num_index.get(n, []))
        # Only trust a bare numeric match if the name is at least loosely
        # similar in its first token — otherwise "123 Main St" bakeries
        # and robotics shops on the same street all collide.
        s1_first_tok = s1_df["normalized_name"].values[idx].split()[:1]
        for cand_id in matched:
            cand_row = other_df[other_df["entity_id"] == cand_id]
            if cand_row.empty:
                continue
            cand_first_tok = cand_row["normalized_name"].values[0].split()[:1]
            if s1_first_tok and cand_first_tok and s1_first_tok[0][:3] == cand_first_tok[0][:3]:
                candidates[s1_id].add(cand_id)

    return candidates


def candidates_to_frame(candidates: dict, s1_ids: list):
    """Turn the {s1_id: {cand_ids}} dict into a long DataFrame of pairs, one row per (s1, cand)."""
    import pandas as pd
    rows = []
    for s1 in s1_ids:
        for cand in candidates.get(s1, []):
            rows.append((s1, cand))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
