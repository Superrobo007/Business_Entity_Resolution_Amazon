"""
IO helpers for the Business Entity Resolution challenge.

All challenge files are TAB-separated. Always pass sep="\t" explicitly —
the problem statement warns that reading without it silently collapses
everything into a single column.
"""

import pandas as pd

REQUIRED_SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]


def load_source(path: str) -> pd.DataFrame:
    """Load a single source file (source1 / source2 / source3)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_SOURCE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}. Found: {list(df.columns)}")
    for c in ["business_name", "business_address", "country"]:
        df[c] = df[c].fillna("").astype(str).str.strip()
    return df.reset_index(drop=True)


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load train_ground_truth.tsv -> columns [source1_entity_id, matched_entity_ids]."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    expected = ["source1_entity_id", "matched_entity_ids"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns: {missing}. Found: {list(df.columns)}")
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("").astype(str).str.strip()
    return df.reset_index(drop=True)


def explode_ground_truth(gt: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the one-row-per-S1-entity ground truth into a long table of
    (source1_entity_id, matched_id) positive pairs. Rows with an empty
    matched_entity_ids (singletons) produce no rows here.

    Vectorized via pandas .str.split + .explode instead of
    .iterrows() + Python string splitting — the iterrows() version
    takes tens of seconds to over a minute at ~2M+ rows; this version
    runs in roughly a second, same output.
    """
    non_empty = gt[gt["matched_entity_ids"] != ""].copy()
    if len(non_empty) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "matched_entity_id"])

    non_empty["matched_entity_id"] = non_empty["matched_entity_ids"].str.split(",")
    long_df = non_empty.explode("matched_entity_id")
    long_df["matched_entity_id"] = long_df["matched_entity_id"].str.strip()
    long_df = long_df[long_df["matched_entity_id"] != ""]
    return long_df[["source1_entity_id", "matched_entity_id"]].reset_index(drop=True)


def write_pairs_tsv(pairs_by_s1: dict, s1_ids: list, out_path: str, id_col: str) -> None:
    """
    Write a TSV in the required output format: one row per Source 1 entity,
    even if it has no matches/candidates (empty string, no quoting).
    `pairs_by_s1` maps source1_entity_id -> list of matched/candidate ids.
    `s1_ids` is the full list of Source 1 entity ids that MUST all appear.

    Builds all lines in memory and writes once with writelines(), rather
    than one f.write() call per row — a minor but free speedup at
    millions of rows (fewer individual I/O calls).
    """
    lines = [f"source1_entity_id\t{id_col}\n"]
    for s1 in s1_ids:
        ids = pairs_by_s1.get(s1, [])
        seen = set()
        deduped = []
        for i in ids:
            if i not in seen:
                seen.add(i)
                deduped.append(i)
        lines.append(f"{s1}\t{','.join(deduped)}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
