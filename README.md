# Business Entity Resolution — Pipeline

## Structure
```
business_entity_resolution/
├── src/
│   ├── io_utils.py        # TSV loading, output writing
│   ├── normalize.py       # name/address normalization
│   ├── blocking.py        # candidate generation
│   ├── features.py        # pairwise similarity features
│   ├── train_model.py     # LightGBM training + inference
│   ├── evaluate.py        # macro F0.5 scorer, threshold sweep
│   └── generate_outputs.py# writes candidate_pairs.tsv / matching_results.tsv
├── notebook/pipeline.ipynb# end-to-end orchestration notebook
├── requirements.txt
└── README.md
```

## Reproduce end-to-end
1. Place challenge data under `dataset/train/` and `dataset/test/` (sibling of this folder, or edit `DATA_DIR` in the notebook).
2. `pip install -r requirements.txt`
3. Open `notebook/pipeline.ipynb` and run all cells top to bottom. It will:
   - load and normalize all three sources
   - split train into an internal train/val by Source-1 entity (no leakage)
   - build candidates on train/val via `blocking.build_candidates`
   - train a 5-fold LightGBM ensemble on train candidates (`train_model.py`)
   - tune the decision threshold on val to maximize macro F0.5 (`evaluate.sweep_threshold`)
   - run the full pipeline on the real test set and write `output/candidate_pairs.tsv` and `output/matching_results.tsv`
4. Validate: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`

## Key tunables
- `blocking.build_candidates(top_k, sim_threshold)` — controls recall ceiling vs. candidate volume. Check `evaluate.blocking_recall` after any change.
- `train_model.train_lightgbm(params=...)` — LightGBM hyperparameters.
- Final decision threshold from `evaluate.sweep_threshold` — directly optimizes macro F0.5 rather than accuracy/F1, since the leaderboard metric is precision-weighted.

## Notes
- All normalization/blocking/feature logic is string-based only — no external APIs, geocoders, or databases are used anywhere in this pipeline, per the challenge's fair-play rules.
- Country is used only as a feature (`country_match`), never as a filter, so unseen labels (e.g. France in test) are handled gracefully.
- Model: LightGBM gradient-boosted trees (MIT-licensed, well under the 8B-parameter cap).
