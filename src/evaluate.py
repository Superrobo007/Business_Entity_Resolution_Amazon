"""
Macro-averaged F_0.5 scorer, computed exactly as the challenge defines it:
per Source 1 entity, then averaged across entities. Singletons (no true
matches) score 1.0 if predicted empty, 0.0 if any match is predicted.

Use this on YOUR OWN held-out validation split — there is no test ground
truth, so this is how you estimate leaderboard performance locally.
"""

import numpy as np


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + b2) * precision * recall / denom


def per_entity_f05(true_ids: set, pred_ids: set) -> float:
    """Score a single Source 1 entity."""
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids and pred_ids:
        return 0.0
    if true_ids and not pred_ids:
        return 0.0
    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    return f_beta(precision, recall, beta=0.5)


def parse_ground_truth(gt_df) -> dict:
    """
    Parse gt_df's comma-separated matched_entity_ids ONCE into
    {source1_entity_id: set(true_ids)}. Call this once and reuse the
    result — re-parsing the same strings inside a threshold sweep (18+
    times) is pure wasted work.
    """
    result = {}
    for s1, ids_str in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        result[s1] = set(x.strip() for x in ids_str.split(",") if x.strip()) if ids_str else set()
    return result


def macro_f05(gt_df, predictions: dict) -> float:
    """
    gt_df: ground truth DataFrame [source1_entity_id, matched_entity_ids].
    predictions: dict {source1_entity_id: list_or_set_of_predicted_ids}
    Returns the macro-averaged F_0.5 across every S1 entity in gt_df.

    Uses .values + zip instead of .iterrows() (iterrows() builds a
    pandas Series per row, which is much slower than plain array
    iteration at this scale — same result, meaningfully faster).
    """
    true_ids_by_s1 = parse_ground_truth(gt_df)
    return macro_f05_from_parsed(true_ids_by_s1, predictions)


def macro_f05_from_parsed(true_ids_by_s1: dict, predictions: dict) -> float:
    """
    Same computation as macro_f05, but takes an already-parsed
    {s1: set(true_ids)} dict (from parse_ground_truth) so a threshold
    sweep doesn't re-parse ground truth strings on every iteration.
    """
    scores = []
    for s1, true_ids in true_ids_by_s1.items():
        pred_ids = set(predictions.get(s1, []))
        scores.append(per_entity_f05(true_ids, pred_ids))
    return float(np.mean(scores)) if scores else 0.0


def sweep_threshold(gt_df, s1_ids_val, cand_scores: dict, thresholds=None) -> tuple:
    """
    cand_scores: dict {source1_entity_id: [(candidate_id, model_probability), ...]}
    Tries a range of thresholds and returns (best_threshold, best_score, all_results).

    Ground truth is parsed once up front and reused across every
    threshold, instead of being re-parsed inside macro_f05 on each of
    the ~18 sweep iterations.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)

    true_ids_by_s1 = parse_ground_truth(gt_df)

    results = []
    for t in thresholds:
        preds = {
            s1: [cid for cid, p in cand_scores.get(s1, []) if p >= t]
            for s1 in s1_ids_val
        }
        score = macro_f05_from_parsed(true_ids_by_s1, preds)
        results.append((t, score))
    best_t, best_score = max(results, key=lambda x: x[1])
    return best_t, best_score, results


def blocking_recall(gt_df, candidates: dict) -> float:
    """
    Recall ceiling of the blocking stage: of all true matches, what
    fraction were even present in the candidate set?

    Uses .values + zip instead of .iterrows() for the same reason as
    macro_f05 above.
    """
    hit, total = 0, 0
    for s1, ids_str in zip(gt_df["source1_entity_id"].values, gt_df["matched_entity_ids"].values):
        true_ids = set(x.strip() for x in ids_str.split(",") if x.strip()) if ids_str else set()
        if not true_ids:
            continue
        cand_ids = set(candidates.get(s1, []))
        hit += len(true_ids & cand_ids)
        total += len(true_ids)
    return hit / total if total else 1.0
