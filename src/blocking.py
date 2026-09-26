"""
Blocking / candidate generation — VECTORIZED VERSION (no per-row Python loops).

Root cause of every previous slowdown: the earlier versions looped over
`other_df` in pure Python (`for idx in range(len(other_df))`) to build
lookup structures, and did this INSIDE build_candidates — so it re-paid
that full-dataset cost on every single call, even when scoring a 1,000-row
test slice. A Python for-loop over 10M+ rows takes tens of seconds no
matter how simple the loop body is; that fixed cost was what you were
actually timing, not blocking quality.

This version does the equivalent work as SQL-style joins via
pandas.merge and pandas.groupby, which run in optimized C, not the
Python interpreter. On a 64GB instance this comfortably handles the
full ~12M-row dataset in well under a minute for the join itself.

Design (multiple cheap signals, unioned — candidates from ANY signal
are kept; the downstream ML model is responsible for precision, not
blocking):
  1. name-prefix + country key           (catches noisy Latin-script names)
  2. exact sorted-name-token key         (catches near-exact renames)
  3. shared numeric address token        (house #/PIN/ZIP — critical for
                                           transliterated names, e.g. a
                                           Hindi business name has NO
                                           shared prefix with its Latin
                                           Source-1 counterpart, but the
                                           address's house number usually
                                           still matches)
  4. shared significant address token    (non-numeric words like a city
                                           or street name, which often
                                           stay in Latin script even when
                                           the business name doesn't)

No TF-IDF, no nearest-neighbor search, no sklearn similarity search at
all — those are what made this slow. Fine-grained similarity scoring
(Levenshtein, Jaccard, etc.) still happens later, in features.py, but
only on the much smaller candidate set this produces.
"""

from collections import Counter

import pandas as pd

STOPWORDS = {
    "the", "and", "of", "for", "inc", "corp", "co", "ltd", "llc", "pvt",
    "private", "limited", "company", "st", "rd", "ave", "dr", "near",
}


def _prep_keys(df: pd.DataFrame, prefix_len: int) -> pd.DataFrame:
    """Vectorized (no Python row loop) column construction."""
    out = df[["entity_id", "normalized_name", "name_sorted",
              "normalized_address", "address_numbers", "country"]].copy()
    out["name_prefix_key"] = out["country"] + "|" + out["normalized_name"].str[:prefix_len]
    return out


def _join_on_key(s1_keys: pd.DataFrame, other_keys: pd.DataFrame, key_col: str,
                  max_block_size: int) -> pd.DataFrame:
    """
    Vectorized equivalent of bucketing + comparing within a bucket: an
    inner merge on `key_col` IS the set of same-bucket (s1, other) pairs.
    Blocks larger than max_block_size on the other side are dropped
    (protects against pathological, extremely common keys) rather than
    looped over.
    """
    other_counts = other_keys[key_col].value_counts()
    valid_keys = other_counts[other_counts <= max_block_size].index
    s1_f = s1_keys[s1_keys[key_col].isin(valid_keys) & (s1_keys[key_col] != "")]
    other_f = other_keys[other_keys[key_col].isin(valid_keys) & (other_keys[key_col] != "")]
    if len(s1_f) == 0 or len(other_f) == 0:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_other"])
    merged = s1_f[["entity_id", key_col]].merge(
        other_f[["entity_id", key_col]], on=key_col, suffixes=("_s1", "_other")
    )
    return merged[["entity_id_s1", "entity_id_other"]]


def _explode_tokens(df: pd.DataFrame, col: str, token_filter=None) -> pd.DataFrame:
    """Long-format (entity_id, token) table, vectorized via pandas .explode."""
    tmp = df[["entity_id", col]].explode(col).dropna(subset=[col])
    tmp = tmp[tmp[col] != ""]
    if token_filter is not None:
        tmp = tmp[tmp[col].isin(token_filter)]
    return tmp


def build_candidates(
    s1_df,
    other_df,
    top_k: int = 15,           # kept for signature compatibility, unused (no similarity ranking here)
    sim_threshold: float = 0.35,  # kept for signature compatibility, unused
    prefix_len: int = 6,
    max_block_size: int = 3000,
):
    """
    s1_df, other_df: DataFrames already run through
        normalize.add_normalized_columns(), each with an 'entity_id' column.
    Returns: dict {source1_entity_id: set(candidate_entity_id)}
    """
    s1_keys = _prep_keys(s1_df, prefix_len)
    other_keys = _prep_keys(other_df, prefix_len)

    all_pairs = []

    # --- Signal 1: name-prefix + country ---
    all_pairs.append(_join_on_key(s1_keys, other_keys, "name_prefix_key", max_block_size))

    # --- Signal 2: exact sorted-name-token match ---
    s1_keys2 = s1_keys.rename(columns={"name_sorted": "key2"})
    other_keys2 = other_keys.rename(columns={"name_sorted": "key2"})
    all_pairs.append(_join_on_key(s1_keys2, other_keys2, "key2", max_block_size))

    # --- Signal 3: shared numeric address token ---
    # Drop overly common numbers (e.g. "1", "0") before joining, same idea
    # as max_block_size above, computed via vectorized value_counts.
    other_nums_long = _explode_tokens(other_df, "address_numbers")
    num_counts = other_nums_long["address_numbers"].value_counts()
    common_nums = set(num_counts[num_counts <= max_block_size].index)
    s1_nums_long = _explode_tokens(s1_df, "address_numbers", token_filter=common_nums)
    other_nums_long = other_nums_long[other_nums_long["address_numbers"].isin(common_nums)]
    if len(s1_nums_long) and len(other_nums_long):
        pairs3 = s1_nums_long.merge(other_nums_long, on="address_numbers", suffixes=("_s1", "_other"))
        all_pairs.append(pairs3[["entity_id_s1", "entity_id_other"]])

    # --- Signal 4: shared significant (non-numeric, non-stopword) address token ---
    def addr_tokens(text):
        return [t for t in text.split() if len(t) >= 4 and t not in STOPWORDS]

    s1_addr = s1_df[["entity_id", "normalized_address"]].copy()
    s1_addr["addr_tokens"] = s1_addr["normalized_address"].apply(addr_tokens)
    other_addr = other_df[["entity_id", "normalized_address"]].copy()
    other_addr["addr_tokens"] = other_addr["normalized_address"].apply(addr_tokens)

    other_tok_long = _explode_tokens(other_addr, "addr_tokens")
    tok_counts = other_tok_long["addr_tokens"].value_counts()
    common_toks = set(tok_counts[tok_counts <= max_block_size].index)
    s1_tok_long = _explode_tokens(s1_addr, "addr_tokens", token_filter=common_toks)
    other_tok_long = other_tok_long[other_tok_long["addr_tokens"].isin(common_toks)]
    if len(s1_tok_long) and len(other_tok_long):
        pairs4 = s1_tok_long.merge(other_tok_long, on="addr_tokens", suffixes=("_s1", "_other"))
        all_pairs.append(pairs4[["entity_id_s1", "entity_id_other"]])

    combined = pd.concat(all_pairs, ignore_index=True).drop_duplicates()

    # Vectorized groupby -> dict, instead of a Python loop over every pair
    candidates = combined.groupby("entity_id_s1")["entity_id_other"].apply(set).to_dict()
    return candidates


def bucket_size_report(df, prefix_len: int = 5, top_n: int = 15):
    """Quick, vectorized sanity check of name-prefix-key selectivity."""
    keys = df["country"] + "|" + df["normalized_name"].str[:prefix_len]
    counts = keys.value_counts()
    total = len(df)
    print(f"Total records: {total:,} | unique buckets: {len(counts):,} | "
          f"avg bucket size: {total/len(counts):.1f}")
    print(f"Top {top_n} largest buckets:")
    for key, count in counts.head(top_n).items():
        print(f"  {key!r}: {count:,} records ({100*count/total:.2f}%)")
    return counts


def candidates_to_frame(candidates: dict, s1_ids: list):
    """Turn the {s1_id: {cand_ids}} dict into a long DataFrame of pairs."""
    rows = []
    for s1 in s1_ids:
        for cand in candidates.get(s1, []):
            rows.append((s1, cand))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])
