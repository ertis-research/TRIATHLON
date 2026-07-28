from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Limita el paralelismo para que la ejecución sea estable en Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CSV = (
    SCRIPT_DIR
    / "huelva_par_etl_outputs_v0_1"
    / "huelva_thm_par_feature_dataset_v0_1.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "model_outputs_huelva_thm_softsensor_v0_2"

VERSION = "v0.2"
RANDOM_STATE = 42

DEFAULT_TEST_ROWS = 5
DEFAULT_WALK_FORWARD_MIN_TRAIN_ROWS = 10
MIN_DEVELOPMENT_ROWS = 14
MIN_TEST_ROWS = 3

MAX_TRAIN_MISSING_RATIO = 0.35
MIN_TRAIN_VARIANCE = 1e-12

PRIMARY_FAMILY = "hgb"
WRITE_EXCEL = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# En v0.2 se utilizan exclusivamente las diez medias de las ventanas de 24 h.
FEATURE_BASES: Tuple[str, ...] = (
    "turbidity_ntu",
    "conductivity_us_cm",
    "uv254_abs_m",
    "toc_mg_l",
    "doc_mg_l",
    "free_chlorine_mg_l",
    "ph",
    "temperature_c",
    "color_hazen",
    "flow",
)

EXPECTED_FEATURE_COLUMNS: Tuple[str, ...] = tuple(
    f"{base}_mean_24h" for base in FEATURE_BASES
)

RIDGE_ALPHAS: Tuple[float, ...] = (
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
)

# Configuraciones deliberadamente compactas y regularizadas por el pequeño n.
HGB_CONFIGS: Tuple[Dict[str, Any], ...] = (
    {
        "learning_rate": 0.03,
        "max_iter": 400,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 2,
        "l2_regularization": 1.0,
        "max_bins": 63,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 400,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 3,
        "l2_regularization": 5.0,
        "max_bins": 63,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 2,
        "l2_regularization": 5.0,
        "max_bins": 63,
    },
    {
        "learning_rate": 0.05,
        "max_iter": 300,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 2,
        "l2_regularization": 5.0,
        "max_bins": 63,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 450,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 3,
        "l2_regularization": 10.0,
        "max_bins": 63,
    },
    {
        "learning_rate": 0.02,
        "max_iter": 600,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 4,
        "l2_regularization": 10.0,
        "max_bins": 63,
    },
)


Candidate = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrenamiento temporal robusto v0.2 del soft-sensor THM de Huelva. "
            "Selecciona modelos mediante walk-forward sobre desarrollo y conserva "
            "un test final completamente aislado."
        )
    )
    parser.add_argument("--input-csv", default=None, help="Dataset generado por el ETL par.")
    parser.add_argument("--output-dir", default=None, help="Carpeta de salida.")
    parser.add_argument(
        "--test-rows",
        type=int,
        default=DEFAULT_TEST_ROWS,
        help="Número de últimas muestras reservadas como test final.",
    )
    parser.add_argument(
        "--walk-forward-min-train-rows",
        type=int,
        default=DEFAULT_WALK_FORWARD_MIN_TRAIN_ROWS,
        help="Mínimo de muestras para comenzar el walk-forward en desarrollo.",
    )
    parser.add_argument("--no-excel", action="store_true", help="No genera el reporte Excel.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return math.nan
    return float(numerator / denominator)


def pearson_correlation(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)

    if len(true_array) < 2:
        return math.nan
    if np.nanstd(true_array) <= 0 or np.nanstd(pred_array) <= 0:
        return math.nan

    return float(np.corrcoef(true_array, pred_array)[0, 1])


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    true_array = np.asarray(y_true, dtype=float)
    pred_array = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(true_array, pred_array))
    rmse = float(math.sqrt(mean_squared_error(true_array, pred_array)))
    bias = float(np.mean(pred_array - true_array))

    if len(true_array) >= 2 and np.nanstd(true_array) > 0:
        r2 = float(r2_score(true_array, pred_array))
    else:
        r2 = math.nan

    target_mean = float(np.mean(true_array)) if len(true_array) else math.nan
    target_range = float(np.max(true_array) - np.min(true_array)) if len(true_array) else math.nan

    return {
        "n": int(len(true_array)),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "abs_bias": abs(bias),
        "r2": r2,
        "pearson_r": pearson_correlation(true_array, pred_array),
        "target_mean": target_mean,
        "target_min": float(np.min(true_array)) if len(true_array) else math.nan,
        "target_max": float(np.max(true_array)) if len(true_array) else math.nan,
        "target_range": target_range,
        "nmae_by_mean": safe_divide(mae, abs(target_mean)),
        "nmae_by_range": safe_divide(mae, target_range),
    }


def load_dataset(input_csv: Path) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    required = {"sample_timestamp", "thm"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Dataset incompleto. Faltan columnas: {sorted(missing_required)}")

    out = df.copy()
    out["sample_timestamp"] = pd.to_datetime(out["sample_timestamp"], errors="coerce")
    out["thm"] = pd.to_numeric(out["thm"], errors="coerce")

    if "sample_id" not in out.columns:
        out["sample_id"] = [f"Huelva_THM_{index + 1:03d}" for index in range(len(out))]
    if "sample_point" not in out.columns:
        out["sample_point"] = ""
    if "relative_file" not in out.columns:
        out["relative_file"] = ""

    out["sample_point"] = out["sample_point"].map(normalize_text)
    out = out[out["sample_timestamp"].notna() & out["thm"].notna()].copy()
    out = out.sort_values("sample_timestamp").reset_index(drop=True)

    available_features = [col for col in EXPECTED_FEATURE_COLUMNS if col in out.columns]
    missing_features = [col for col in EXPECTED_FEATURE_COLUMNS if col not in out.columns]

    if not available_features:
        raise ValueError(
            "No se encontró ninguna de las features esperadas *_mean_24h en el dataset."
        )

    for col in available_features:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    feature_inventory = pd.DataFrame(
        [
            {
                "feature": col,
                "available": col in out.columns,
                "missing_ratio_all": (
                    float(out[col].isna().mean()) if col in out.columns else math.nan
                ),
                "variance_all": (
                    float(out[col].var(skipna=True)) if col in out.columns else math.nan
                ),
            }
            for col in EXPECTED_FEATURE_COLUMNS
        ]
    )

    if missing_features:
        print(f"[WARN] Features esperadas ausentes: {', '.join(missing_features)}")

    return out, available_features, feature_inventory


def resolve_temporal_split(n_rows: int, requested_test_rows: int) -> Tuple[int, int]:
    test_rows = max(MIN_TEST_ROWS, int(requested_test_rows))
    test_rows = min(test_rows, max(MIN_TEST_ROWS, n_rows // 3))
    development_rows = n_rows - test_rows

    if development_rows < MIN_DEVELOPMENT_ROWS:
        raise ValueError(
            f"Split temporal imposible: total={n_rows}, desarrollo={development_rows}, "
            f"test={test_rows}. Desarrollo debe tener al menos {MIN_DEVELOPMENT_ROWS}."
        )

    return development_rows, test_rows


def filter_features_on_training(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[List[str], pd.DataFrame]:
    kept: List[str] = []
    rows: List[Dict[str, Any]] = []

    for col in feature_cols:
        series = pd.to_numeric(train_df[col], errors="coerce")
        missing_ratio = float(series.isna().mean())
        variance = float(series.var(skipna=True)) if series.notna().sum() > 1 else 0.0

        keep = missing_ratio <= MAX_TRAIN_MISSING_RATIO and variance > MIN_TRAIN_VARIANCE
        if missing_ratio > MAX_TRAIN_MISSING_RATIO:
            reason = "high_missing_ratio"
        elif variance <= MIN_TRAIN_VARIANCE:
            reason = "near_constant"
        else:
            reason = "kept"

        rows.append(
            {
                "feature": col,
                "train_missing_ratio": missing_ratio,
                "train_variance": variance,
                "kept": keep,
                "reason": reason,
            }
        )
        if keep:
            kept.append(col)

    if not kept:
        raise ValueError("No queda ninguna feature válida en el fold de entrenamiento.")

    return kept, pd.DataFrame(rows)


def candidate_definitions() -> List[Candidate]:
    candidates: List[Candidate] = [
        {
            "candidate_id": "dummy_mean",
            "family": "dummy",
            "params": {"strategy": "mean"},
        },
        {
            "candidate_id": "dummy_median",
            "family": "dummy",
            "params": {"strategy": "median"},
        },
    ]

    for alpha in RIDGE_ALPHAS:
        candidates.append(
            {
                "candidate_id": f"ridge_alpha_{alpha:g}",
                "family": "ridge",
                "params": {"alpha": alpha},
            }
        )

    for index, config in enumerate(HGB_CONFIGS, start=1):
        candidates.append(
            {
                "candidate_id": f"hgb_cfg_{index}",
                "family": "hgb",
                "params": deepcopy(config),
            }
        )

    return candidates


def make_preprocessor(feature_cols: Sequence[str], scale: bool) -> ColumnTransformer:
    numeric_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), list(feature_cols)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(candidate: Candidate, feature_cols: Sequence[str]) -> Pipeline:
    family = candidate["family"]
    params = candidate["params"]

    if family == "dummy":
        estimator = DummyRegressor(strategy=str(params["strategy"]))
        scale = False
    elif family == "ridge":
        estimator = Ridge(alpha=float(params["alpha"]))
        scale = True
    elif family == "hgb":
        estimator = HistGradientBoostingRegressor(
            random_state=RANDOM_STATE,
            early_stopping=False,
            **params,
        )
        scale = False
    else:
        raise ValueError(f"Familia no soportada: {family}")

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(feature_cols, scale=scale)),
            ("model", estimator),
        ]
    )


def model_input(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    return df[list(feature_cols)].copy()


def walk_forward_candidate(
    development_df: pd.DataFrame,
    feature_cols: Sequence[str],
    candidate: Candidate,
    min_train_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: List[Dict[str, Any]] = []
    audit_rows: List[pd.DataFrame] = []

    for prediction_index in range(min_train_rows, len(development_df)):
        train_df = development_df.iloc[:prediction_index].copy()
        eval_df = development_df.iloc[[prediction_index]].copy()

        kept_features, audit = filter_features_on_training(train_df, feature_cols)
        audit = audit.copy()
        audit["candidate_id"] = candidate["candidate_id"]
        audit["prediction_index"] = prediction_index
        audit["prediction_timestamp"] = eval_df["sample_timestamp"].iloc[0]
        audit_rows.append(audit)

        pipeline = make_pipeline(candidate, kept_features)
        pipeline.fit(model_input(train_df, kept_features), train_df["thm"].to_numpy(dtype=float))
        prediction = float(pipeline.predict(model_input(eval_df, kept_features))[0])
        y_true = float(eval_df["thm"].iloc[0])

        prediction_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "family": candidate["family"],
                "params_json": json.dumps(candidate["params"], ensure_ascii=False, sort_keys=True),
                "sample_id": eval_df["sample_id"].iloc[0],
                "sample_timestamp": eval_df["sample_timestamp"].iloc[0],
                "sample_point": eval_df["sample_point"].iloc[0],
                "relative_file": eval_df["relative_file"].iloc[0],
                "train_rows": len(train_df),
                "train_start": train_df["sample_timestamp"].min(),
                "train_end": train_df["sample_timestamp"].max(),
                "train_target_mean": float(train_df["thm"].mean()),
                "train_target_min": float(train_df["thm"].min()),
                "train_target_max": float(train_df["thm"].max()),
                "n_features": len(kept_features),
                "features": "|".join(kept_features),
                "y_true": y_true,
                "y_pred": prediction,
                "error": prediction - y_true,
                "abs_error": abs(prediction - y_true),
            }
        )

    predictions_df = pd.DataFrame(prediction_rows)
    audits_df = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    return predictions_df, audits_df


def run_walk_forward_selection(
    development_df: pd.DataFrame,
    feature_cols: Sequence[str],
    min_train_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Candidate]]:
    all_predictions: List[pd.DataFrame] = []
    all_audits: List[pd.DataFrame] = []
    candidates_by_id: Dict[str, Candidate] = {}

    candidates = candidate_definitions()
    total = len(candidates)

    for index, candidate in enumerate(candidates, start=1):
        candidates_by_id[candidate["candidate_id"]] = candidate
        print(f"[walk-forward {index}/{total}] {candidate['candidate_id']}")

        predictions, audits = walk_forward_candidate(
            development_df=development_df,
            feature_cols=feature_cols,
            candidate=candidate,
            min_train_rows=min_train_rows,
        )
        all_predictions.append(predictions)
        if not audits.empty:
            all_audits.append(audits)

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    audits_df = pd.concat(all_audits, ignore_index=True) if all_audits else pd.DataFrame()

    metric_rows: List[Dict[str, Any]] = []
    for candidate_id, group in predictions_df.groupby("candidate_id", sort=False):
        candidate = candidates_by_id[candidate_id]
        metrics = regression_metrics(group["y_true"], group["y_pred"])
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "params_json": json.dumps(candidate["params"], ensure_ascii=False, sort_keys=True),
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["rmse", "mae", "abs_bias", "candidate_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    metrics_df["global_rank"] = np.arange(1, len(metrics_df) + 1)
    metrics_df["family_rank"] = (
        metrics_df.groupby("family")["rmse"].rank(method="first", ascending=True).astype(int)
    )

    best_dummy_mae = float(
        metrics_df[metrics_df["family"] == "dummy"].sort_values("mae").iloc[0]["mae"]
    )
    metrics_df["mae_improvement_vs_best_dummy"] = best_dummy_mae - metrics_df["mae"]
    metrics_df["relative_mae_improvement_vs_best_dummy"] = metrics_df[
        "mae_improvement_vs_best_dummy"
    ].map(lambda value: safe_divide(float(value), best_dummy_mae))

    return metrics_df, predictions_df, audits_df, candidates_by_id


def best_candidate_id(metrics_df: pd.DataFrame, family: Optional[str] = None) -> str:
    subset = metrics_df if family is None else metrics_df[metrics_df["family"] == family]
    if subset.empty:
        raise ValueError(f"No hay candidatos para family={family}")
    return str(
        subset.sort_values(
            ["rmse", "mae", "abs_bias", "candidate_id"],
            ascending=[True, True, True, True],
        ).iloc[0]["candidate_id"]
    )


def fit_candidate(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    candidate: Candidate,
) -> Tuple[Pipeline, List[str], pd.DataFrame]:
    kept_features, audit = filter_features_on_training(train_df, feature_cols)
    pipeline = make_pipeline(candidate, kept_features)
    pipeline.fit(model_input(train_df, kept_features), train_df["thm"].to_numpy(dtype=float))
    return pipeline, kept_features, audit


def evaluate_selected_candidates_on_test(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    selected_ids: Sequence[str],
    candidates_by_id: Dict[str, Candidate],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    fitted: Dict[str, Dict[str, Any]] = {}
    audit_frames: List[pd.DataFrame] = []

    for candidate_id in selected_ids:
        candidate = candidates_by_id[candidate_id]
        pipeline, kept_features, audit = fit_candidate(
            train_df=development_df,
            feature_cols=feature_cols,
            candidate=candidate,
        )

        audit = audit.copy()
        audit["candidate_id"] = candidate_id
        audit_frames.append(audit)

        predictions = pipeline.predict(model_input(test_df, kept_features))
        metrics = regression_metrics(test_df["thm"], predictions)

        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "params_json": json.dumps(candidate["params"], ensure_ascii=False, sort_keys=True),
                "n_features": len(kept_features),
                "features": "|".join(kept_features),
                **metrics,
            }
        )

        pred_df = test_df[
            ["sample_id", "sample_timestamp", "sample_point", "relative_file", "thm"]
        ].copy()
        pred_df = pred_df.rename(columns={"thm": "y_true"})
        pred_df["y_pred"] = predictions
        pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]
        pred_df["abs_error"] = pred_df["error"].abs()
        pred_df["candidate_id"] = candidate_id
        pred_df["family"] = candidate["family"]
        prediction_frames.append(pred_df)

        fitted[candidate_id] = {
            "pipeline": pipeline,
            "candidate": candidate,
            "features": kept_features,
            "feature_audit": audit,
        }

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["rmse", "mae", "abs_bias"], ascending=[True, True, True]
    ).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    audits_df = pd.concat(audit_frames, ignore_index=True)

    best_dummy_mae = float(
        metrics_df[metrics_df["family"] == "dummy"].sort_values("mae").iloc[0]["mae"]
    )
    metrics_df["mae_improvement_vs_dummy"] = best_dummy_mae - metrics_df["mae"]
    metrics_df["relative_mae_improvement_vs_dummy"] = metrics_df[
        "mae_improvement_vs_dummy"
    ].map(lambda value: safe_divide(float(value), best_dummy_mae))

    return metrics_df, predictions_df, fitted, audits_df


def permutation_importance_exploratory(
    fitted_info: Dict[str, Any],
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    features = fitted_info["features"]
    if len(test_df) < 3:
        return pd.DataFrame()

    result = permutation_importance(
        fitted_info["pipeline"],
        model_input(test_df, features),
        test_df["thm"].to_numpy(dtype=float),
        scoring="neg_mean_absolute_error",
        n_repeats=50,
        random_state=RANDOM_STATE,
    )

    return pd.DataFrame(
        {
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "note": "Exploratory only: final test contains very few samples.",
        }
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def development_feature_correlations(
    development_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in feature_cols:
        valid = development_df[[col, "thm"]].dropna()
        correlation = (
            float(valid[col].corr(valid["thm"]))
            if len(valid) >= 3 and valid[col].std() > 0 and valid["thm"].std() > 0
            else math.nan
        )
        rows.append(
            {
                "feature": col,
                "n_valid": len(valid),
                "pearson_with_thm_development": correlation,
                "feature_mean": float(valid[col].mean()) if len(valid) else math.nan,
                "feature_std": float(valid[col].std()) if len(valid) > 1 else math.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(
        "pearson_with_thm_development",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)


def create_bundle(
    candidate_id: str,
    candidate: Candidate,
    pipeline: Pipeline,
    features: Sequence[str],
    training_df: pd.DataFrame,
    input_csv: Path,
    training_scope: str,
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "scope": "Huelva THM soft-sensor prediction only",
        "training_scope": training_scope,
        "input_csv": str(input_csv),
        "candidate_id": candidate_id,
        "family": candidate["family"],
        "params": candidate["params"],
        "features": list(features),
        "pipeline": pipeline,
        "training_rows": len(training_df),
        "training_start": str(training_df["sample_timestamp"].min()),
        "training_end": str(training_df["sample_timestamp"].max()),
        "target_mean": float(training_df["thm"].mean()),
        "target_min": float(training_df["thm"].min()),
        "target_max": float(training_df["thm"].max()),
    }


def save_test_plot(
    test_predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    observed = (
        test_predictions[["sample_timestamp", "y_true"]]
        .drop_duplicates()
        .sort_values("sample_timestamp")
    )
    ax.plot(observed["sample_timestamp"], observed["y_true"], marker="o", label="THM real")

    for candidate_id, group in test_predictions.groupby("candidate_id"):
        ordered = group.sort_values("sample_timestamp")
        ax.plot(
            ordered["sample_timestamp"],
            ordered["y_pred"],
            marker="o",
            label=candidate_id,
        )

    ax.set_title("Huelva THM — test temporal final")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("THM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def save_walk_forward_plot(
    walk_forward_predictions: pd.DataFrame,
    candidate_id: str,
    output_path: Path,
) -> None:
    selected = walk_forward_predictions[
        walk_forward_predictions["candidate_id"] == candidate_id
    ].sort_values("sample_timestamp")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(selected["sample_timestamp"], selected["y_true"], marker="o", label="THM real")
    ax.plot(selected["sample_timestamp"], selected["y_pred"], marker="o", label="THM predicho")
    ax.set_title(f"Huelva THM — walk-forward desarrollo — {candidate_id}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("THM")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def markdown_table_from_metrics(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[str]:
    selected = df.head(max_rows) if max_rows else df
    lines = [
        "| Candidato | Familia | n | MAE | RMSE | Bias | R² | Pearson r |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | {row['family']} | {int(row['n'])} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['bias']:.4f} | "
            f"{row['r2']:.4f} | {row['pearson_r']:.4f} |"
        )
    return lines


def write_report(
    output_path: Path,
    input_csv: Path,
    full_df: pd.DataFrame,
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    walk_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    primary_hgb_id: str,
    best_overall_id: str,
    walk_min_train_rows: int,
) -> None:
    lines: List[str] = []
    lines.append("# Entrenamiento soft-sensor THM Huelva v0.2")
    lines.append("")
    lines.append(f"- Dataset: `{input_csv}`")
    lines.append("- Alcance: predicción THM mediante soft-sensor.")
    lines.append("- Fuera de alcance: propagación y alertas.")
    lines.append("")
    lines.append("## Cambios respecto a v0.1")
    lines.append("")
    lines.append("- Se elimina el bloque de validación único de cinco muestras.")
    lines.append("- La selección se realiza mediante walk-forward expanding-window.")
    lines.append("- El test final permanece completamente aislado durante la selección.")
    lines.append("- Solo se usan las diez medias de las ventanas de 24 h.")
    lines.append("- `sample_point` queda fuera porque es constante.")
    lines.append("- HGB es la familia principal; Dummy y Ridge son controles.")
    lines.append("")
    lines.append("## Datos y split")
    lines.append("")
    lines.append(f"- Muestras totales: **{len(full_df)}**")
    lines.append(f"- Desarrollo: **{len(development_df)}**")
    lines.append(f"- Test final: **{len(test_df)}**")
    lines.append(f"- Inicio walk-forward: **{walk_min_train_rows}** muestras de entrenamiento")
    lines.append(
        f"- Rango completo: **{full_df['sample_timestamp'].min()} → "
        f"{full_df['sample_timestamp'].max()}**"
    )
    lines.append("")
    lines.append("## Features")
    lines.append("")
    for col in feature_cols:
        lines.append(f"- `{col}`")
    lines.append("")
    lines.append("## Selección walk-forward")
    lines.append("")
    lines.append(f"- HGB principal seleccionado: `{primary_hgb_id}`")
    lines.append(f"- Mejor candidato global: `{best_overall_id}`")
    lines.append("")
    lines.extend(markdown_table_from_metrics(walk_metrics, max_rows=10))
    lines.append("")
    lines.append("## Evaluación final en test")
    lines.append("")
    lines.extend(markdown_table_from_metrics(test_metrics))
    lines.append("")
    lines.append("## Interpretación")
    lines.append("")
    lines.append(
        "Con 26 muestras de laboratorio, los resultados continúan siendo exploratorios. "
        "La v0.2 reduce la dependencia de un único bloque de validación y permite comprobar "
        "si el HGB supera de forma consistente a Dummy y Ridge en predicciones temporales."
    )
    lines.append("")
    lines.append(
        "El bundle `primary_hgb_evaluation_bundle.joblib` se entrena solo con desarrollo y "
        "es el asociado a las métricas del test. El bundle `primary_hgb_full_data_bundle.joblib` "
        "se reentrena después con las 26 muestras y es el candidato para futuras inferencias, "
        "sin constituir todavía una validación productiva."
    )
    lines.append("")
    lines.append("## Archivos principales")
    lines.append("")
    lines.append("- `walk_forward_candidate_metrics.csv`")
    lines.append("- `walk_forward_predictions_all.csv`")
    lines.append("- `final_test_metrics.csv`")
    lines.append("- `final_test_predictions_all.csv`")
    lines.append("- `primary_hgb_evaluation_bundle.joblib`")
    lines.append("- `primary_hgb_full_data_bundle.joblib`")
    lines.append("- `best_overall_evaluation_bundle.joblib`")
    lines.append("- `best_overall_full_data_bundle.joblib`")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve() if args.input_csv else INPUT_CSV.resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR.resolve()
    write_excel = WRITE_EXCEL and not args.no_excel

    if not input_csv.exists():
        raise FileNotFoundError(f"No existe input_csv: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    full_df, feature_cols, feature_inventory = load_dataset(input_csv)
    development_rows, test_rows = resolve_temporal_split(len(full_df), args.test_rows)

    development_df = full_df.iloc[:development_rows].copy()
    test_df = full_df.iloc[development_rows:].copy()

    walk_min_train_rows = max(6, int(args.walk_forward_min_train_rows))
    if walk_min_train_rows >= len(development_df):
        raise ValueError(
            f"walk_forward_min_train_rows={walk_min_train_rows} debe ser menor que "
            f"development_rows={len(development_df)}."
        )

    print("========================================")
    print("ENTRENAMIENTO SOFT-SENSOR THM HUELVA")
    print("========================================")
    print(f"Versión:               {VERSION}")
    print(f"Dataset:               {input_csv}")
    print(f"Salida:                {output_dir}")
    print(f"Muestras:              {len(full_df)}")
    print(f"Desarrollo/Test:       {len(development_df)}/{len(test_df)}")
    print(f"Walk-forward min train:{walk_min_train_rows}")
    print(f"Predicciones WF/cand.: {len(development_df) - walk_min_train_rows}")
    print(f"Features medias 24 h:  {len(feature_cols)}")
    print(f"Rango:                 {full_df['sample_timestamp'].min()} -> {full_df['sample_timestamp'].max()}")
    print("")
    print("Muestras por punto:")
    print(full_df["sample_point"].replace("", "(vacío)").value_counts(dropna=False).to_string())
    print("")
    print("[INFO] sample_point no se utiliza como feature.")
    print("")

    walk_metrics, walk_predictions, walk_audits, candidates_by_id = run_walk_forward_selection(
        development_df=development_df,
        feature_cols=feature_cols,
        min_train_rows=walk_min_train_rows,
    )

    best_dummy_id = best_candidate_id(walk_metrics, family="dummy")
    best_ridge_id = best_candidate_id(walk_metrics, family="ridge")
    primary_hgb_id = best_candidate_id(walk_metrics, family=PRIMARY_FAMILY)
    best_overall_id = best_candidate_id(walk_metrics)

    selected_ids = list(dict.fromkeys([best_dummy_id, best_ridge_id, primary_hgb_id, best_overall_id]))

    test_metrics, test_predictions, fitted_test, test_audits = evaluate_selected_candidates_on_test(
        development_df=development_df,
        test_df=test_df,
        feature_cols=feature_cols,
        selected_ids=selected_ids,
        candidates_by_id=candidates_by_id,
    )

    primary_importance = permutation_importance_exploratory(
        fitted_info=fitted_test[primary_hgb_id],
        test_df=test_df,
    )
    correlations = development_feature_correlations(development_df, feature_cols)

    # Bundles de evaluación: entrenados solo con desarrollo y asociados al test final.
    primary_eval_info = fitted_test[primary_hgb_id]
    primary_eval_bundle = create_bundle(
        candidate_id=primary_hgb_id,
        candidate=primary_eval_info["candidate"],
        pipeline=primary_eval_info["pipeline"],
        features=primary_eval_info["features"],
        training_df=development_df,
        input_csv=input_csv,
        training_scope="development_only_for_final_test_evaluation",
    )

    overall_eval_info = fitted_test[best_overall_id]
    overall_eval_bundle = create_bundle(
        candidate_id=best_overall_id,
        candidate=overall_eval_info["candidate"],
        pipeline=overall_eval_info["pipeline"],
        features=overall_eval_info["features"],
        training_df=development_df,
        input_csv=input_csv,
        training_scope="development_only_for_final_test_evaluation",
    )

    # Bundles finales: reentrenados con todas las muestras después de cerrar la evaluación.
    primary_full_pipeline, primary_full_features, primary_full_audit = fit_candidate(
        train_df=full_df,
        feature_cols=feature_cols,
        candidate=candidates_by_id[primary_hgb_id],
    )
    primary_full_bundle = create_bundle(
        candidate_id=primary_hgb_id,
        candidate=candidates_by_id[primary_hgb_id],
        pipeline=primary_full_pipeline,
        features=primary_full_features,
        training_df=full_df,
        input_csv=input_csv,
        training_scope="all_available_samples_after_evaluation",
    )

    overall_full_pipeline, overall_full_features, overall_full_audit = fit_candidate(
        train_df=full_df,
        feature_cols=feature_cols,
        candidate=candidates_by_id[best_overall_id],
    )
    overall_full_bundle = create_bundle(
        candidate_id=best_overall_id,
        candidate=candidates_by_id[best_overall_id],
        pipeline=overall_full_pipeline,
        features=overall_full_features,
        training_df=full_df,
        input_csv=input_csv,
        training_scope="all_available_samples_after_evaluation",
    )

    # CSV y artefactos.
    feature_inventory.to_csv(
        output_dir / "feature_inventory.csv", index=False, encoding="utf-8-sig"
    )
    walk_metrics.to_csv(
        output_dir / "walk_forward_candidate_metrics.csv", index=False, encoding="utf-8-sig"
    )
    walk_predictions.to_csv(
        output_dir / "walk_forward_predictions_all.csv", index=False, encoding="utf-8-sig"
    )
    walk_audits.to_csv(
        output_dir / "walk_forward_feature_filter_audit.csv", index=False, encoding="utf-8-sig"
    )
    test_metrics.to_csv(
        output_dir / "final_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    test_predictions.to_csv(
        output_dir / "final_test_predictions_all.csv", index=False, encoding="utf-8-sig"
    )
    test_audits.to_csv(
        output_dir / "final_test_feature_filter_audit.csv", index=False, encoding="utf-8-sig"
    )
    primary_importance.to_csv(
        output_dir / "primary_hgb_permutation_importance_test_exploratory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlations.to_csv(
        output_dir / "feature_target_correlations_development.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_full_audit.to_csv(
        output_dir / "primary_hgb_full_data_feature_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_full_audit.to_csv(
        output_dir / "best_overall_full_data_feature_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    joblib.dump(primary_eval_bundle, output_dir / "primary_hgb_evaluation_bundle.joblib")
    joblib.dump(primary_full_bundle, output_dir / "primary_hgb_full_data_bundle.joblib")
    joblib.dump(overall_eval_bundle, output_dir / "best_overall_evaluation_bundle.joblib")
    joblib.dump(overall_full_bundle, output_dir / "best_overall_full_data_bundle.joblib")

    metadata = {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "scope": "prediction_only",
        "propagation_and_alerts_in_scope": False,
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "n_samples": len(full_df),
        "development_rows": len(development_df),
        "test_rows": len(test_df),
        "walk_forward_min_train_rows": walk_min_train_rows,
        "walk_forward_predictions_per_candidate": len(development_df) - walk_min_train_rows,
        "features": feature_cols,
        "sample_point_counts": full_df["sample_point"].value_counts(dropna=False).to_dict(),
        "best_dummy_candidate_id": best_dummy_id,
        "best_ridge_candidate_id": best_ridge_id,
        "primary_hgb_candidate_id": primary_hgb_id,
        "best_overall_candidate_id": best_overall_id,
        "selection_criterion": "walk_forward_rmse_then_mae_then_abs_bias_on_development",
        "final_test_used_for_selection": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    save_test_plot(
        test_predictions=test_predictions,
        output_path=output_dir / "final_test_predictions_vs_real.png",
    )
    save_walk_forward_plot(
        walk_forward_predictions=walk_predictions,
        candidate_id=primary_hgb_id,
        output_path=output_dir / "primary_hgb_walk_forward_predictions_vs_real.png",
    )

    write_report(
        output_path=output_dir / "training_report_v0_2.md",
        input_csv=input_csv,
        full_df=full_df,
        development_df=development_df,
        test_df=test_df,
        feature_cols=feature_cols,
        walk_metrics=walk_metrics,
        test_metrics=test_metrics,
        primary_hgb_id=primary_hgb_id,
        best_overall_id=best_overall_id,
        walk_min_train_rows=walk_min_train_rows,
    )

    if write_excel:
        try:
            with pd.ExcelWriter(output_dir / "training_report_v0_2.xlsx", engine="openpyxl") as writer:
                walk_metrics.to_excel(writer, sheet_name="walk_metrics", index=False)
                walk_predictions.to_excel(writer, sheet_name="walk_predictions", index=False)
                test_metrics.to_excel(writer, sheet_name="test_metrics", index=False)
                test_predictions.to_excel(writer, sheet_name="test_predictions", index=False)
                correlations.to_excel(writer, sheet_name="feature_correlations", index=False)
                primary_importance.to_excel(writer, sheet_name="importance_exploratory", index=False)
                feature_inventory.to_excel(writer, sheet_name="feature_inventory", index=False)
        except Exception as exc:
            print(f"[WARN] No se pudo generar Excel: {type(exc).__name__}: {exc}")

    print("")
    print("========================================")
    print("RESUMEN ENTRENAMIENTO HUELVA v0.2")
    print("========================================")
    print(f"Mejor Dummy WF:       {best_dummy_id}")
    print(f"Mejor Ridge WF:       {best_ridge_id}")
    print(f"HGB principal WF:     {primary_hgb_id}")
    print(f"Mejor global WF:      {best_overall_id}")
    print("")
    print("Top 5 walk-forward:")
    print(
        walk_metrics[
            ["candidate_id", "family", "n", "mae", "rmse", "bias", "r2", "pearson_r"]
        ]
        .head(5)
        .round(4)
        .to_string(index=False)
    )
    print("")
    print("Test final por familia seleccionada:")
    print(
        test_metrics[
            [
                "candidate_id",
                "family",
                "n",
                "mae",
                "rmse",
                "bias",
                "r2",
                "pearson_r",
                "mae_improvement_vs_dummy",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("")

    primary_test = test_metrics[test_metrics["candidate_id"] == primary_hgb_id].iloc[0]
    best_dummy_test = test_metrics[test_metrics["family"] == "dummy"].sort_values("mae").iloc[0]

    if primary_test["mae"] < best_dummy_test["mae"]:
        print(
            f"[RESULTADO] HGB mejora el MAE del Dummy en "
            f"{best_dummy_test['mae'] - primary_test['mae']:.4f} unidades THM en test."
        )
    else:
        print(
            f"[RESULTADO] HGB no supera al Dummy en test; diferencia MAE="
            f"{primary_test['mae'] - best_dummy_test['mae']:.4f}."
        )

    if best_overall_id != primary_hgb_id:
        print(
            f"[NOTA] El mejor candidato global en walk-forward es {best_overall_id}, "
            f"pero HGB se mantiene como familia principal del proyecto."
        )

    print("[NOTA] Con 26 muestras, los resultados siguen siendo exploratorios.")
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
