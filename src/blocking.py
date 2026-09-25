"""
Blocking / candidate generation — SCALABLE VERSION.

The original version of this module computed a global nearest-neighbor
search (every Source 1 record vs. every Source 2/3 record) before ever
narrowing the search space. That is O(n * m) similarity computations and
does not finish in practical time once n and m are in the millions — it
is not a "let it run longer" problem, it is an algorithmic one.

This version buckets records by a cheap, selective key BEFORE doing any
TF-IDF/cosine work, so the expensive part only ever runs within a small
bucket (typically tens to low thousands of records), not against the
full pool. This is standard "blocking" in the entity-resolution sense —
the same word used in the problem statement's own tips.
"""

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from tqdm import tqdm
except ImportError:  # tqdm is in requirements.txt, but don't hard-fail without it
    def tqdm(x, **kwargs):
        return x


def _block_key(name: str, country: str, prefix_len: int = 4) -> str:
    """
    Cheap, selective blocking key: country + first few characters of the
    normalized name. Records that don't share this key never get compared
    by the TF-IDF signal below — but they can still be caught by the
    exact-name-token and numeric-address-token signals, which are
    independent of this key and run separately.
    """
    prefix = name[:prefix_len] if name else "\u2205"
    return f"{country}|{prefix}"


def build_candidates(
    s1_df,
    other_df,
    top_k: int = 15,
    sim_threshold: float = 0.35,
    prefix_len: int = 6,
    max_block_size: int = 2000,
    max_features: int = 50_000,
    show_progress: bool = True,
):
    """
    s1_df, other_df: DataFrames already run through
        normalize.add_normalized_columns(), each with an 'entity_id' column.
    other_df is the concatenation of source2 + source3 records.

    Returns: dict {source1_entity_id: set(candidate_entity_id)}

    New params vs. the original version:
      prefix_len:     how many characters of the normalized name to use
                       as part of the blocking key. Shorter = bigger
                       buckets (more recall, slower). Longer = smaller
                       buckets (faster, but can miss matches where the
                       first few characters differ, e.g. a typo in the
                       first letter). 4 is a reasonable starting point.
      max_block_size: hard cap on how many "other" records get compared
                       within one bucket. Protects against pathological
                       buckets (extremely common name prefixes) blowing
                       up runtime. Records beyond the cap are simply not
                       considered by the TF-IDF signal for that bucket —
                       they can still be caught by the other two signals.
      max_features:   caps TF-IDF vocabulary size, which bounds memory.
    """
    candidates = defaultdict(set)

    # --- Precompute TF-IDF vectorization (optimized for speed) ---
    s1_text = (s1_df["normalized_name"] + " " + s1_df["normalized_address"]).values
    other_text = (other_df["normalized_name"] + " " + other_df["normalized_address"]).values

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), min_df=3, max_features=max_features, max_df=0.8
    )
    # Fit only on other_df to save time (s1 is smaller)
    vectorizer.fit(other_text)

    s1_vecs = vectorizer.transform(s1_text)      # sparse (n_s1 x V)
    other_vecs = vectorizer.transform(other_text)  # sparse (n_other x V)

    # --- Bucket both pools by the same key ---
    s1_names = s1_df["normalized_name"].values
    s1_countries = s1_df["country"].values
    other_names = other_df["normalized_name"].values
    other_countries = other_df["country"].values

    s1_buckets = defaultdict(list)
    for idx in range(len(s1_df)):
        key = _block_key(s1_names[idx], s1_countries[idx], prefix_len)
        s1_buckets[key].append(idx)

    other_buckets = defaultdict(list)
    for idx in range(len(other_df)):
        key = _block_key(other_names[idx], other_countries[idx], prefix_len)
        other_buckets[key].append(idx)

    s1_ids = s1_df["entity_id"].values
    other_ids = other_df["entity_id"].values

    # --- TF-IDF signal, computed ONLY within each bucket ---
    iterator = tqdm(s1_buckets.items(), total=len(s1_buckets), disable=not show_progress,
                     desc="Blocking (TF-IDF within buckets)")
    for key, s1_idx_list in iterator:
        other_idx_list = other_buckets.get(key)
        if not other_idx_list:
            continue
        if len(other_idx_list) > max_block_size:
            other_idx_list = other_idx_list[:max_block_size]

        s1_block = s1_vecs[s1_idx_list]          # (b1 x V) sparse
        other_block = other_vecs[other_idx_list]  # (b2 x V) sparse
        sim_matrix = (s1_block @ other_block.T).toarray()  # (b1 x b2) small dense

        for local_i, global_i in enumerate(s1_idx_list):
            row = sim_matrix[local_i]
            n = len(row)
            top_n = min(top_k, n)
            if top_n < n:
                top_local = np.argpartition(-row, top_n - 1)[:top_n]
            else:
                top_local = np.arange(n)
            for local_j in top_local:
                if row[local_j] >= sim_threshold:
                    candidates[s1_ids[global_i]].add(other_ids[other_idx_list[local_j]])

    # --- Signal 2: exact match on sorted normalized name tokens (cheap, O(n)) ---
    name_key_index = defaultdict(list)
    for idx, key in enumerate(other_df["name_sorted"].values):
        if key:
            name_key_index[key].append(other_ids[idx])
    for idx, key in enumerate(s1_df["name_sorted"].values):
        if key and key in name_key_index:
            candidates[s1_ids[idx]].update(name_key_index[key])

    # --- Signal 3: shared numeric address token + loose name check (cheap) ---
    num_index = defaultdict(list)
    for idx, nums in enumerate(other_df["address_numbers"].values):
        for n in nums:
            num_index[n].append(idx)

    for idx, nums in enumerate(s1_df["address_numbers"].values):
        if not nums:
            continue
        matched_idx = set()
        for n in nums:
            matched_idx.update(num_index.get(n, []))
        s1_first_tok = s1_names[idx].split()[:1]
        if not s1_first_tok:
            continue
        for cand_idx in matched_idx:
            cand_first_tok = other_names[cand_idx].split()[:1]
            if cand_first_tok and s1_first_tok[0][:3] == cand_first_tok[0][:3]:
                candidates[s1_ids[idx]].add(other_ids[cand_idx])

    return candidates


def bucket_size_report(df, prefix_len: int = 4, top_n: int = 15):
    """
    Run this FIRST, on your real data, before calling build_candidates on
    the full dataset. Shows how selective your blocking key actually is —
    if a handful of buckets contain a large fraction of all records, that's
    exactly the pathological case max_block_size guards against, and you
    should raise prefix_len (or add another key component) before running
    the full build. Takes seconds even on millions of rows.
    """
    names = df["normalized_name"].values
    countries = df["country"].values
    sizes = defaultdict(int)
    for i in range(len(df)):
        sizes[_block_key(names[i], countries[i], prefix_len)] += 1
    sizes = sorted(sizes.items(), key=lambda x: -x[1])
    total = len(df)
    print(f"Total records: {total:,} | unique buckets: {len(sizes):,} | "
          f"avg bucket size: {total/len(sizes):.1f}")
    print(f"Top {top_n} largest buckets:")
    for key, count in sizes[:top_n]:
        print(f"  {key!r}: {count:,} records ({100*count/total:.2f}%)")
    return sizes


def candidates_to_frame(candidates: dict, s1_ids: list):
    """Turn the {s1_id: {cand_ids}} dict into a long DataFrame of pairs."""
    import pandas as pd
    rows = []
    for s1 in s1_ids:
        for cand in candidates.get(s1, []):
            rows.append((s1, cand))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])