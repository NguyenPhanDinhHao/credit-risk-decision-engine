"""SHAP explanations.

SHAP values attribute a model's output to its inputs given the model and the background data. They
describe the MODEL, not causal effects in the world: correlated features share credit unpredictably
and proxies of sensitive attributes can carry attribution. Explanations here are in the space of the
uncalibrated model margin (log-odds for LightGBM/logistic regression, probability for random forest).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _transformed(bundle: Any, X_raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    pipe = bundle.pipeline
    X_fe = pipe.named_steps["features"].transform(bundle.prepare(X_raw))
    M = pipe.named_steps["prep"].transform(X_fe)
    return np.asarray(M, dtype=float), list(pipe.named_steps["prep"].get_feature_names_out())


def _positive_class(values: Any) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[:, :, 1] if values.ndim == 3 else values


def build_explainer(bundle: Any, background_raw: pd.DataFrame | None = None) -> Any:
    model = bundle.pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        bg = bundle.metadata.get("background_matrix")
        if bg is None:
            if background_raw is None:
                raise ValueError("A background sample is required to explain a logistic regression model")
            bg, _ = _transformed(bundle, background_raw)
        bg = np.asarray(bg, dtype=float)
        return shap.LinearExplainer(model, shap.maskers.Independent(bg, max_samples=len(bg)))
    return shap.TreeExplainer(model)


def shap_values(bundle: Any, X_raw: pd.DataFrame, explainer: Any = None, background_raw: pd.DataFrame | None = None):
    M, names = _transformed(bundle, X_raw)
    explainer = explainer or build_explainer(bundle, background_raw if background_raw is not None else X_raw)
    sv = _positive_class(explainer.shap_values(M))
    return sv, M, names


def explain_rows(bundle: Any, X_raw: pd.DataFrame, top_k: int = 5) -> list[list[dict[str, Any]]]:
    """Local explanation: top-k features by |SHAP| for each row."""
    if bundle._explainer is None:
        bundle._explainer = build_explainer(bundle, X_raw)
    sv, M, names = shap_values(bundle, X_raw, explainer=bundle._explainer)
    out = []
    for i in range(sv.shape[0]):
        idx = np.argsort(-np.abs(sv[i]))[:top_k]
        out.append([
            {"feature": names[j], "value": float(M[i, j]), "contribution": float(sv[i, j]),
             "direction": "increases_risk" if sv[i, j] > 0 else "decreases_risk"}
            for j in idx
        ])
    return out


def explain_instance(bundle: Any, row: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
    """Convenience wrapper returning a DataFrame for a single applicant."""
    return pd.DataFrame(explain_rows(bundle, row.iloc[[0]], top_k=top_k)[0])


def run_shap_analysis(bundle: Any, X_sample_raw: pd.DataFrame, out_dir: Path, top_dependence: int = 3) -> dict[str, Any]:
    """Global importance, summary (beeswarm) plot, dependence plots. Returns the importance table."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(bundle.pipeline.named_steps["model"], LogisticRegression):
        bundle.metadata["background_matrix"] = _transformed(bundle, X_sample_raw.head(200))[0]
    explainer = build_explainer(bundle, X_sample_raw)
    bundle._explainer = explainer
    sv, M, names = shap_values(bundle, X_sample_raw, explainer=explainer)
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out_dir / "shap_global_importance.csv", index=False)

    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4c78a8")
    ax.set_title("Global feature importance (mean |SHAP|)"); ax.set_xlabel("mean |SHAP value|")
    fig.tight_layout(); fig.savefig(out_dir / "shap_importance_bar.png", dpi=120); plt.close(fig)

    shap.summary_plot(sv, M, feature_names=names, max_display=20, show=False)
    plt.tight_layout(); plt.savefig(out_dir / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()

    for feat in imp["feature"].head(top_dependence):
        shap.dependence_plot(feat, sv, M, feature_names=names, show=False)
        plt.tight_layout(); plt.savefig(out_dir / f"shap_dependence_{feat}.png", dpi=120, bbox_inches="tight"); plt.close()

    # Local explanation example (first row of the sample)
    local = explain_instance(bundle, X_sample_raw.iloc[[0]])
    local.to_csv(out_dir / "shap_local_example.csv", index=False)
    logger.info("SHAP artifacts written to %s", out_dir)
    return {"importance": imp.head(20).to_dict("records")}
