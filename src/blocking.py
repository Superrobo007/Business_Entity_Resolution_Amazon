"""
Blocking / candidate generation.
"""

import pandas as pd

STOPWORDS = {
    "the", "and", "of", "for", "inc", "corp", "co", "ltd", "llc", "pvt",
    "private", "limited", "company", "st", "rd", "ave", "dr", "near",
}


def _prep_keys(df: pd.DataFrame, prefix_len: int) -> pd.DataFrame:
    out = df[["entity_id", "normalized_name", "name_sorted",
              "normalized_address", "address_numbers", "country"]].copy()
    out["name_prefix_key"] = out["country"] + "|" + out["normalized_name"].str[:prefix_len]
    out["name_sorted_key"] = out["country"] + "|" + out["name_sorted"]
    return out


def _join_on_key(s1_keys, other_keys, key_col, max_block_size):
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


def _explode_tokens(df, col, extra_cols=None, token_filter=None):
    extra_cols = extra_cols or []
    keep = ["entity_id"] + extra_cols + [col]
    tmp = df[keep].explode(col).dropna(subset=[col])
    tmp = tmp[tmp[col] != ""]
    if token_filter is not None:
        tmp = tmp[tmp[col].isin(token_filter)]
    return tmp


def _token_join_signal(s1_df, other_df, source_col, min_len, max_block_size,
                        require_country_match, extra_stopwords=None):
    stop = STOPWORDS | (extra_stopwords or set())

    def tokens(text):
        return [t for t in text.split() if len(t) >= min_len and t not in stop]

    extra = ["country"] if require_country_match else []
    s1_t = s1_df[["entity_id", source_col] + extra].copy()
    s1_t["tok"] = s1_t[source_col].apply(tokens)
    other_t = other_df[["entity_id", source_col] + extra].copy()
    other_t["tok"] = other_t[source_col].apply(tokens)

    s1_long = _explode_tokens(s1_t, "tok", extra_cols=extra)
    other_long = _explode_tokens(other_t, "tok", extra_cols=extra)

    if require_country_match:
        s1_long["key"] = s1_long["country"] + "|" + s1_long["tok"]
        other_long["key"] = other_long["country"] + "|" + other_long["tok"]
        key_col = "key"
    else:
        key_col = "tok"

    counts = other_long[key_col].value_counts()
    valid = set(counts[counts <= max_block_size].index)
    s1_long = s1_long[s1_long[key_col].isin(valid)]
    other_long = other_long[other_long[key_col].isin(valid)]
    if len(s1_long) == 0 or len(other_long) == 0:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_other"])

    merged = s1_long[["entity_id", key_col]].merge(
        other_long[["entity_id", key_col]], on=key_col, suffixes=("_s1", "_other")
    )
    return merged[["entity_id_s1", "entity_id_other"]]


def _ngram_signal(s1_df, other_df, n, min_shared, max_block_size, require_country_match):
    """
    Signal 6: character n-gram overlap on normalized_name.

    Why this is needed: Signals 2/5 need a whole word to match exactly.
    A single-word business name with a typo, or a transliteration
    variant ("Acme" vs "Acmee", or a differently-romanized name), shares
    ZERO whole tokens with its true match -- invisible to every signal
    so far. Character n-grams break words into overlapping chunks, so
    a typo in part of a word still leaves other chunks matching.

    Why this can't repeat the earlier crash: this is the SAME
    explode+merge join pattern as every other signal here (no dense
    O(bucket^2) similarity matrix like the earlier TF-IDF attempt).
    Requiring `min_shared` n-grams in common (via a groupby count on
    the merged pairs) keeps precision reasonable without needing any
    similarity scoring -- just counting.
    """
    def ngrams(name):
        s = f"_{name}_"
        if len(s) < n:
            return []
        return [s[i:i + n] for i in range(len(s) - n + 1)]

    extra = ["country"] if require_country_match else []
    s1_t = s1_df[["entity_id", "normalized_name"] + extra].copy()
    s1_t["ng"] = s1_t["normalized_name"].apply(ngrams)
    other_t = other_df[["entity_id", "normalized_name"] + extra].copy()
    other_t["ng"] = other_t["normalized_name"].apply(ngrams)

    s1_long = _explode_tokens(s1_t, "ng", extra_cols=extra)
    other_long = _explode_tokens(other_t, "ng", extra_cols=extra)

    if require_country_match:
        s1_long["key"] = s1_long["country"] + "|" + s1_long["ng"]
        other_long["key"] = other_long["country"] + "|" + other_long["ng"]
        key_col = "key"
    else:
        key_col = "ng"

    counts = other_long[key_col].value_counts()
    valid = set(counts[counts <= max_block_size].index)
    s1_long = s1_long[s1_long[key_col].isin(valid)]
    other_long = other_long[other_long[key_col].isin(valid)]
    if len(s1_long) == 0 or len(other_long) == 0:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_other"])

    merged = s1_long[["entity_id", key_col]].merge(
        other_long[["entity_id", key_col]], on=key_col, suffixes=("_s1", "_other")
    )
    # Require several shared n-grams, not just one -- this is what keeps
    # this signal selective without any similarity scoring.
    pair_counts = merged.groupby(["entity_id_s1", "entity_id_other"]).size()
    good_pairs = pair_counts[pair_counts >= min_shared].reset_index()[["entity_id_s1", "entity_id_other"]]
    return good_pairs


def build_candidates(
    s1_df,
    other_df,
    top_k: int = 15,
    sim_threshold: float = 0.35,
    prefix_len: int = 6,
    max_block_size: int = 300,
    require_country_match: bool = False,
    addr_token_min_len: int = 4,
    name_token_min_len: int = 4,
    use_name_token_signal: bool = True,
    use_ngram_signal: bool = False,
    ngram_n: int = 4,
    ngram_min_shared: int = 3,
):
    s1_keys = _prep_keys(s1_df, prefix_len)
    other_keys = _prep_keys(other_df, prefix_len)

    all_pairs = [
        _join_on_key(s1_keys, other_keys, "name_prefix_key", max_block_size),
        _join_on_key(s1_keys, other_keys,
                     "name_sorted_key" if require_country_match else "name_sorted",
                     max_block_size),
    ]

    extra = ["country"] if require_country_match else []
    other_nums_long = _explode_tokens(other_df, "address_numbers", extra_cols=extra)
    s1_nums_long = _explode_tokens(s1_df, "address_numbers", extra_cols=extra)
    if require_country_match:
        other_nums_long["num_key"] = other_nums_long["country"] + "|" + other_nums_long["address_numbers"]
        s1_nums_long["num_key"] = s1_nums_long["country"] + "|" + s1_nums_long["address_numbers"]
        key_col3 = "num_key"
    else:
        key_col3 = "address_numbers"
    num_counts = other_nums_long[key_col3].value_counts()
    common_nums = set(num_counts[num_counts <= max_block_size].index)
    s1_nums_long = s1_nums_long[s1_nums_long[key_col3].isin(common_nums)]
    other_nums_long = other_nums_long[other_nums_long[key_col3].isin(common_nums)]
    if len(s1_nums_long) and len(other_nums_long):
        pairs3 = s1_nums_long[["entity_id", key_col3]].merge(
            other_nums_long[["entity_id", key_col3]], on=key_col3, suffixes=("_s1", "_other")
        )
        all_pairs.append(pairs3[["entity_id_s1", "entity_id_other"]])

    all_pairs.append(_token_join_signal(
        s1_df, other_df, "normalized_address", addr_token_min_len,
        max_block_size, require_country_match
    ))

    if use_name_token_signal:
        all_pairs.append(_token_join_signal(
            s1_df, other_df, "normalized_name", name_token_min_len,
            max_block_size, require_country_match
        ))

    if use_ngram_signal:
        all_pairs.append(_ngram_signal(
            s1_df, other_df, ngram_n, ngram_min_shared,
            max_block_size, require_country_match
        ))

    combined = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
    candidates = combined.groupby("entity_id_s1")["entity_id_other"].agg(list).to_dict()
    return candidates


def bucket_size_report(df, prefix_len: int = 5, top_n: int = 15):
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
    rows = []
    for s1 in s1_ids:
        for cand in candidates.get(s1, []):
            rows.append((s1, cand))
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id"])