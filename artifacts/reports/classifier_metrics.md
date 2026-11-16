# Fault detector comparison (SOFI test split)

Selected: **random_forest** (by mean grouped-CV PR-AUC)

| model | accuracy | balanced acc | fault precision | fault recall | fault F1 | macro F1 | PR-AUC | ROC-AUC | episode recall |
|---|---|---|---|---|---|---|---|---|---|
| gbdt | 0.9812 | 0.630 | 0.645 | 0.263 | 0.374 | 0.682 | 0.328 | 0.812 | 0.438 |
| random_forest | 0.9812 | 0.617 | 0.667 | 0.237 | 0.350 | 0.670 | 0.350 | 0.882 | 0.375 |
| mlp | 0.9748 | 0.678 | 0.400 | 0.368 | 0.384 | 0.685 | 0.367 | 0.858 | 0.562 |

## Grouped 5-fold cross-validation (train+val, blocked by day)

| model | PR-AUC (mean ± sd) | ROC-AUC (mean ± sd) | pooled OOF fault F1 | pooled OOF accuracy |
|---|---|---|---|---|
| gbdt | 0.453 ± 0.135 | 0.925 ± 0.039 | 0.473 | 0.9874 |
| random_forest | 0.506 ± 0.135 | 0.954 ± 0.015 | 0.523 | 0.9898 |
| mlp | 0.435 ± 0.141 | 0.937 ± 0.033 | 0.451 | 0.9876 |

Split sizes: train=18644 rows / 272 faults, val=3726 rows / 44 faults, test=3572 rows / 76 faults
