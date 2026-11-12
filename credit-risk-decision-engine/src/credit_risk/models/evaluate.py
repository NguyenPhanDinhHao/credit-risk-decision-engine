"""Evaluation metrics and plots.

Accuracy alone is misleading for imbalanced credit data: with ~8% defaults a model that approves
everyone is ~92% accurate yet useless. We report ranking (ROC-AUC, KS), precision-recall behaviour
(PR-AUC, precision, recall, F1) and probability quality (Brier, log-loss, calibration).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def probability_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": float(ks_2samp(p[y == 1], p[y == 0]).statistic),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y, pred = np.asarray(y), (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float((tp + tn) / len(y)),
        "flagged_rate": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def full_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    return {**probability_metrics(y, p), **threshold_metrics(y, p, threshold)}


# ------------------------------------------------------------------ plots
def plot_roc_pr(curves: dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path, title_suffix: str = "") -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC {title_suffix}")
    base = float(np.mean(next(iter(curves.values()))[0]))
    axes[1].axhline(base, color="k", ls="--", lw=0.7, label=f"prevalence={base:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall {title_suffix}")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path


def plot_confusion_matrix(m: dict[str, Any], out_path: Path, title: str = "") -> Path:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["pred 0", "pred 1"], yticklabels=["actual 0", "actual 1"],
           title=f"{title}\nthreshold={m['threshold']:.3f}")
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)
    return out_path
