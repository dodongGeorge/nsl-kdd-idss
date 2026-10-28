# Network Intrusion Detection Using Machine Learning Models (improved version)

Group 14 | The University of Mindanao | Random Forest on NSL-KDD: Normal, DoS, Probe, R2L, U2R.

## What changed from v1
1. **Preprocessing:** 7 new login/privilege features; correlation filter (r > 0.95) that protects login/privilege features; one-hot columns shared by train and test.
2. **Class balancing:** SMOTE on R2L/U2R and random under-sampling of Normal/DoS, on training data only (KDDTest+ is never resampled).
3. **Grid search:** rare-class target x large-class cap x min_samples_leaf, stratified 3-fold CV scored by macro-F1.
4. **Per-class feature analysis:** one-vs-rest forests; a feature is kept if it is top-15 or >= 0.01 important for any class (`figures/per_class_importance.csv`).
5. **R2L/U2R:** the combination above lifts their recall (see `results_summary.md`).

## R2L balancing update
The app combines two decision mechanisms: (1) a configurable R2L probability boost and (2) a separate binary Random Forest specialist for R2L versus all other classes. The specialist threshold is selected by maximizing R2L F1 on a held-out split from KDDTrain+, then the specialist is refit on all training records. KDDTest+ is not used to tune the threshold.

The specialist does not guarantee better generalization, especially for attack types absent from training. Evaluate R2L precision/recall/F1, macro-F1, and the Normal false-positive rate on the untouched KDDTest+ set after retraining. The app's saved metrics are updated to reflect the included model artifact.

## Files
`train_model.py` (full pipeline, writes everything below) | `common.py` (shared preprocessing) | `app.py` (Streamlit) |
`nsl_kdd_random_forest.joblib` | `results.json`, `results_summary.md` | `figures/` | `data/` | `sample_connections.csv`

## Run
```
pip install -r requirements.txt
python train_model.py     # optional: retrains (about 10 minutes on 1 CPU)
streamlit run app.py
```

## Limits
Offline benchmark only. Grid-search CV scores (about 0.94 macro-F1) are optimistic because folds contain only attack types seen in training; KDDTest+ includes 17 attack types that never appear in training, which caps R2L/U2R recall.

