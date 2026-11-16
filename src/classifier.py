"""Stage 1: supervised fault detector trained on SOFI symptom features.

Compares a gradient-boosted decision tree, a random forest and a neural network
(MLP). Faults are ~1.5 % of samples, so every model is trained with balanced
class weights and its decision threshold is tuned on the validation split to
maximise fault-class F1. Model selection uses validation PR-AUC; the test split
is only scored once, for the final report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from . import config, data

LABELS = ["NE (no error)", "F (fault)"]


def build_models(seed: int = config.RANDOM_SEED) -> dict:
    return {
        "gbdt": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", early_stopping=True, validation_fraction=0.15,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "mlp": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=300,
                          early_stopping=True, random_state=seed),
        ),
    }


def _fit(name: str, model, X: pd.DataFrame, y: np.ndarray):
    weights = compute_sample_weight("balanced", y)
    if name == "mlp":
        model.fit(X, y, mlpclassifier__sample_weight=weights)
    else:
        model.fit(X, y)  # class weighting is configured on the estimator
    return model


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    report = classification_report(y_true, pred, labels=[0, 1], target_names=LABELS,
                                   output_dict=True, zero_division=0)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "per_class": {k: report[k] for k in LABELS},
        "macro_f1": report["macro avg"]["f1-score"],
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def episode_recall(df: pd.DataFrame, pred: np.ndarray) -> float:
    """Share of fault episodes in which at least one sample is flagged."""
    ep = data.fault_episodes(df)
    hit = pd.Series(pred, index=df.index)[ep >= 0].groupby(ep[ep >= 0]).max()
    return float(hit.mean()) if len(hit) else float("nan")


@dataclass
class FaultDetector:
    """A fitted model plus the feature list and decision threshold it was tuned with."""

    name: str
    model: object
    features: list[str]
    threshold: float
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)

    def save(self, path=config.MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path=config.MODEL_PATH) -> "FaultDetector":
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `netdiag train` first")
        return joblib.load(path)


def cross_validate(df: pd.DataFrame, features: list[str], n_splits: int = 5,
                   block_hours: int = 24, verbose: bool = True) -> dict:
    """Grouped, stratified K-fold over time blocks on the train+val partitions.

    Each held-out test split only contains a handful of fault episodes, so the
    fold-to-fold spread reported here is the more honest picture of model quality.
    """
    groups = (df["timestamp"] // (block_hours * 3600)).to_numpy()
    y = df["label"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    folds = list(cv.split(np.zeros(len(df)), y, groups))
    out = {}
    for name in build_models():
        oof = np.zeros(len(df))
        pr, roc = [], []
        for tr, te in folds:
            model = _fit(name, build_models()[name], df.iloc[tr][features], y[tr])
            oof[te] = model.predict_proba(df.iloc[te][features])[:, 1]
            pr.append(average_precision_score(y[te], oof[te]))
            roc.append(roc_auc_score(y[te], oof[te]))
        thr = best_f1_threshold(y, oof)
        pooled = evaluate(y, oof, thr)
        out[name] = {
            "pr_auc_mean": float(np.mean(pr)), "pr_auc_std": float(np.std(pr)),
            "roc_auc_mean": float(np.mean(roc)), "roc_auc_std": float(np.std(roc)),
            "pooled_oof_fault_f1": pooled["per_class"][LABELS[1]]["f1-score"],
            "pooled_oof_accuracy": pooled["accuracy"],
        }
        if verbose:
            print(f"  CV {name}: PR-AUC {np.mean(pr):.3f} ± {np.std(pr):.3f}, "
                  f"ROC-AUC {np.mean(roc):.3f} ± {np.std(roc):.3f}")
    return out


def train_and_compare(df: pd.DataFrame | None = None, verbose: bool = True,
                      run_cv: bool = True) -> tuple[FaultDetector, dict]:
    """Train all candidate models, pick the best, score it on the held-out test split."""
    df = data.load_dataset() if df is None else df
    df = df.assign(split=data.assign_splits(df))
    features = data.feature_columns(df)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    y = {s: p["label"].to_numpy() for s, p in parts.items()}

    results: dict = {
        "split_sizes": {s: {"rows": len(p), "faults": int(p["label"].sum())} for s, p in parts.items()},
        "models": {},
    }
    if run_cv:
        if verbose:
            print("Grouped 5-fold CV on train+val ...")
        results["cv"] = cross_validate(df[df["split"] != "test"], features, verbose=verbose)
    fitted = {}
    for name, model in build_models().items():
        if verbose:
            print(f"Training {name} ...")
        model = _fit(name, model, parts["train"][features], y["train"])
        val_proba = model.predict_proba(parts["val"][features])[:, 1]
        threshold = best_f1_threshold(y["val"], val_proba)
        test_proba = model.predict_proba(parts["test"][features])[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        results["models"][name] = {
            "val": evaluate(y["val"], val_proba, threshold),
            "test": evaluate(y["test"], test_proba, threshold) | {
                "episode_recall": episode_recall(parts["test"], test_pred)
            },
        }
        fitted[name] = (model, threshold)
        if verbose:
            m = results["models"][name]
            print(f"  val PR-AUC={m['val']['pr_auc']:.3f}  test acc={m['test']['accuracy']:.4f} "
                  f"fault-F1={m['test']['per_class'][LABELS[1]]['f1-score']:.3f}")

    if run_cv:
        best = max(results["cv"], key=lambda n: results["cv"][n]["pr_auc_mean"])
        results["selection_criterion"] = "mean grouped-CV PR-AUC"
    else:
        best = max(results["models"], key=lambda n: results["models"][n]["val"]["pr_auc"])
        results["selection_criterion"] = "validation PR-AUC"
    results["selected_model"] = best

    # Refit the winner on train+val so the deployed detector sees all labelled
    # history; keep the validation-tuned threshold.
    model, threshold = fitted[best]
    trainval = pd.concat([parts["train"], parts["val"]])
    final = _fit(best, build_models()[best], trainval[features], trainval["label"].to_numpy())
    detector = FaultDetector(best, final, features, threshold, results["models"][best])
    return detector, results


def write_report(results: dict, path=None) -> None:
    path = path or config.REPORTS_DIR / "classifier_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))

    lines = ["# Fault detector comparison (SOFI test split)", "",
             f"Selected: **{results['selected_model']}** (by {results['selection_criterion']})", "",
             "| model | accuracy | balanced acc | fault precision | fault recall | fault F1 | macro F1 | PR-AUC | ROC-AUC | episode recall |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in results["models"].items():
        t = m["test"]
        f = t["per_class"][LABELS[1]]
        lines.append(f"| {name} | {t['accuracy']:.4f} | {t['balanced_accuracy']:.3f} | {f['precision']:.3f} | "
                     f"{f['recall']:.3f} | {f['f1-score']:.3f} | {t['macro_f1']:.3f} | {t['pr_auc']:.3f} | "
                     f"{t['roc_auc']:.3f} | {t['episode_recall']:.3f} |")
    if "cv" in results:
        lines += ["", "## Grouped 5-fold cross-validation (train+val, blocked by day)", "",
                  "| model | PR-AUC (mean ± sd) | ROC-AUC (mean ± sd) | pooled OOF fault F1 | pooled OOF accuracy |",
                  "|---|---|---|---|---|"]
        for name, c in results["cv"].items():
            lines.append(f"| {name} | {c['pr_auc_mean']:.3f} ± {c['pr_auc_std']:.3f} | "
                         f"{c['roc_auc_mean']:.3f} ± {c['roc_auc_std']:.3f} | "
                         f"{c['pooled_oof_fault_f1']:.3f} | {c['pooled_oof_accuracy']:.4f} |")
    lines += ["", "Split sizes: " + ", ".join(
        f"{s}={v['rows']} rows / {v['faults']} faults" for s, v in results["split_sizes"].items())]
    (path.parent / "classifier_metrics.md").write_text("\n".join(lines) + "\n")
