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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

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

EXPERIMENT_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = EXPERIMENT_DIR.parent.parent

CLEAN_TIMESERIES_CSV = (
    SCRIPT_DIR
    / "huelva_par_etl_outputs_v0_1"
    / "huelva_par_timeseries_clean_v0_1.csv"
)
LABEL_DATASET_CSV = (
    SCRIPT_DIR
    / "huelva_par_etl_outputs_v0_1"
    / "huelva_thm_par_feature_dataset_v0_1.csv"
)
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

VERSION = "v0.5-no-color"
EXPERIMENT_NOTE = "Replica exacta de la validacion formal v0.4 excluyendo solo color_hazen."
RANDOM_STATE = 42

FORMAL_TEST_FRACTION = 0.20
DEFAULT_TEST_ROWS = None
DEFAULT_WALK_FORWARD_MIN_TRAIN_ROWS = 10
MIN_DEVELOPMENT_ROWS = 14
MIN_TEST_ROWS = 5

WINDOW_HOURS: Tuple[int, ...] = (6, 12, 24, 48, 72)
MIN_WINDOW_COVERAGE_RATIO = 0.75

MAX_TRAIN_MISSING_RATIO = 0.35
MIN_TRAIN_VARIANCE = 1e-12

PRIMARY_FAMILY = "hgb"
WRITE_EXCEL = True

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

FEATURE_BASES: Tuple[str, ...] = (
    "turbidity_ntu",
    "conductivity_us_cm",
    "uv254_abs_m",
    "toc_mg_l",
    "doc_mg_l",
    "free_chlorine_mg_l",
    "ph",
    "temperature_c",
    "flow",
)

RIDGE_ALPHAS: Tuple[float, ...] = (
    1.0,
    10.0,
    100.0,
)

HGB_CONFIGS: Tuple[Dict[str, Any], ...] = (
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
        "min_samples_leaf": 3,
        "l2_regularization": 5.0,
        "max_bins": 63,
    },
    {
        "learning_rate": 0.03,
        "max_iter": 500,
        "max_leaf_nodes": 5,
        "min_samples_leaf": 4,
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
            "Validacion formal v0.5 sin color del soft-sensor THM de Huelva con "
            "features multiventana 6/12/24/48/72 h, seleccion walk-forward "
            "sobre desarrollo y test final completamente aislado."
        )
    )
    parser.add_argument("--clean-csv", default=None, help="Serie temporal limpia del ETL par.")
    parser.add_argument("--labels-csv", default=None, help="Dataset con muestras THM exactas.")
    parser.add_argument("--output-dir", default=None, help="Carpeta de salida.")
    parser.add_argument(
        "--test-rows",
        type=int,
        default=DEFAULT_TEST_ROWS,
        help=("Numero de ultimas muestras reservadas como test final. "
              "Si se omite, se usa el protocolo formal: ceil(20% del total), minimo 5."),
    )
    parser.add_argument(
        "--walk-forward-min-train-rows",
        type=int,
        default=DEFAULT_WALK_FORWARD_MIN_TRAIN_ROWS,
        help="Minimo de muestras antes de empezar walk-forward.",
    )
    parser.add_argument(
        "--min-window-coverage-ratio",
        type=float,
        default=MIN_WINDOW_COVERAGE_RATIO,
        help="Cobertura minima de una variable dentro de cada ventana temporal.",
    )
    parser.add_argument("--no-excel", action="store_true", help="No genera el reporte Excel.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


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


def load_clean_timeseries(clean_csv: Path) -> Tuple[pd.DataFrame, float]:
    if not clean_csv.exists():
        raise FileNotFoundError(f"No existe clean_csv: {clean_csv}")

    wanted = {"timestamp", *FEATURE_BASES}
    df = pd.read_csv(
        clean_csv,
        encoding="utf-8-sig",
        usecols=lambda column: column in wanted,
    )

    missing = wanted - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en clean_csv: {sorted(missing)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()].copy()

    for col in FEATURE_BASES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.groupby("timestamp", as_index=False)[list(FEATURE_BASES)]
        .mean(numeric_only=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    unique_ts = pd.DatetimeIndex(df["timestamp"].unique()).sort_values()
    if len(unique_ts) < 2:
        raise ValueError("No hay suficientes timestamps para calcular la cadencia.")

    gaps = pd.Series(unique_ts[1:] - unique_ts[:-1]).dt.total_seconds().div(60.0)
    positive = gaps[gaps > 0]
    if positive.empty:
        raise ValueError("No se pudo inferir la cadencia de la serie temporal.")

    median_gap_minutes = float(positive.median())
    return df, median_gap_minutes


def load_labels(labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists():
        raise FileNotFoundError(f"No existe labels_csv: {labels_csv}")

    df = pd.read_csv(labels_csv, encoding="utf-8-sig")
    required = {"sample_timestamp", "thm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en labels_csv: {sorted(missing)}")

    out = df.copy()
    out["sample_timestamp"] = pd.to_datetime(out["sample_timestamp"], errors="coerce")
    out["thm"] = pd.to_numeric(out["thm"], errors="coerce")

    if "sample_id" not in out.columns:
        out["sample_id"] = [f"Huelva_THM_{i + 1:03d}" for i in range(len(out))]
    if "sample_point" not in out.columns:
        out["sample_point"] = ""
    if "relative_file" not in out.columns:
        out["relative_file"] = ""
    if "thm_unit" not in out.columns:
        out["thm_unit"] = ""

    out["sample_point"] = out["sample_point"].map(normalize_text)
    out = out[out["sample_timestamp"].notna() & out["thm"].notna()].copy()
    out = out.sort_values("sample_timestamp").reset_index(drop=True)
    return out


def expected_points(window_hours: int, cadence_minutes: float) -> int:
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes debe ser > 0")
    return max(1, int(math.floor(window_hours * 60.0 / cadence_minutes)) + 1)


def build_multwindow_feature_dataset(
    clean_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    cadence_minutes: float,
    min_coverage_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < min_coverage_ratio <= 1:
        raise ValueError("--min-window-coverage-ratio debe estar en (0, 1].")

    indexed = clean_df.set_index("timestamp").sort_index()
    feature_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for _, sample in labels_df.iterrows():
        sample_ts = pd.Timestamp(sample["sample_timestamp"])
        feature_row: Dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "sample_timestamp": sample_ts,
            "sample_point": sample["sample_point"],
            "relative_file": sample["relative_file"],
            "thm": float(sample["thm"]),
            "thm_unit": sample.get("thm_unit", ""),
        }

        for window_hours in WINDOW_HOURS:
            start = sample_ts - pd.Timedelta(hours=window_hours)
            window = indexed.loc[(indexed.index >= start) & (indexed.index <= sample_ts)]
            expected = expected_points(window_hours, cadence_minutes)

            for base in FEATURE_BASES:
                values = pd.to_numeric(window[base], errors="coerce").dropna()
                count = int(len(values))
                coverage = min(1.0, safe_divide(count, expected))
                valid = bool(np.isfinite(coverage) and coverage >= min_coverage_ratio)

                mean_value = float(values.mean()) if valid and count > 0 else math.nan
                feature_row[f"{base}_mean_{window_hours}h"] = mean_value

                coverage_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "sample_timestamp": sample_ts,
                        "feature_base": base,
                        "window_hours": window_hours,
                        "expected_points": expected,
                        "valid_points": count,
                        "coverage_ratio": coverage,
                        "coverage_ok": valid,
                    }
                )

        for base in FEATURE_BASES:
            mean_6 = feature_row[f"{base}_mean_6h"]
            mean_12 = feature_row[f"{base}_mean_12h"]
            mean_24 = feature_row[f"{base}_mean_24h"]
            mean_48 = feature_row[f"{base}_mean_48h"]
            mean_72 = feature_row[f"{base}_mean_72h"]

            feature_row[f"{base}_contrast_6h_72h"] = (
                float(mean_6 - mean_72)
                if np.isfinite(mean_6) and np.isfinite(mean_72)
                else math.nan
            )
            feature_row[f"{base}_contrast_12h_48h"] = (
                float(mean_12 - mean_48)
                if np.isfinite(mean_12) and np.isfinite(mean_48)
                else math.nan
            )
            feature_row[f"{base}_contrast_6h_24h"] = (
                float(mean_6 - mean_24)
                if np.isfinite(mean_6) and np.isfinite(mean_24)
                else math.nan
            )

        feature_rows.append(feature_row)

    features_df = pd.DataFrame(feature_rows).sort_values("sample_timestamp").reset_index(drop=True)
    coverage_df = pd.DataFrame(coverage_rows)
    return features_df, coverage_df


def feature_groups() -> Dict[str, List[str]]:
    baseline = [f"{base}_mean_24h" for base in FEATURE_BASES]

    multiscale_means = [
        f"{base}_mean_{window_hours}h"
        for base in FEATURE_BASES
        for window_hours in WINDOW_HOURS
    ]

    compact_level_trend = []
    for base in FEATURE_BASES:
        compact_level_trend.extend(
            [
                f"{base}_mean_24h",
                f"{base}_contrast_6h_72h",
                f"{base}_contrast_12h_48h",
            ]
        )

    return {
        "baseline_24h": baseline,
        "multiscale_means": multiscale_means,
        "compact_level_trend": compact_level_trend,
    }


def compare_with_v0_2_24h(
    generated_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    merged = generated_df[["sample_id", "sample_timestamp"]].copy()
    for base in FEATURE_BASES:
        generated_col = f"{base}_mean_24h"
        previous_col = generated_col

        if previous_col not in labels_df.columns:
            rows.append(
                {
                    "feature": generated_col,
                    "available_in_v0_2_dataset": False,
                    "n_compared": 0,
                    "mean_abs_difference": math.nan,
                    "max_abs_difference": math.nan,
                }
            )
            continue

        left = generated_df[["sample_id", generated_col]].copy()
        right = labels_df[["sample_id", previous_col]].copy()
        right[previous_col] = pd.to_numeric(right[previous_col], errors="coerce")
        check = left.merge(right, on="sample_id", how="inner", suffixes=("_v0_3", "_v0_2"))
        valid = check[[f"{generated_col}_v0_3", f"{previous_col}_v0_2"]].dropna()

        if valid.empty:
            mean_abs_diff = math.nan
            max_abs_diff = math.nan
        else:
            diff = (
                valid[f"{generated_col}_v0_3"]
                - valid[f"{previous_col}_v0_2"]
            ).abs()
            mean_abs_diff = float(diff.mean())
            max_abs_diff = float(diff.max())

        rows.append(
            {
                "feature": generated_col,
                "available_in_v0_2_dataset": True,
                "n_compared": int(len(valid)),
                "mean_abs_difference": mean_abs_diff,
                "max_abs_difference": max_abs_diff,
            }
        )

    return pd.DataFrame(rows)


def resolve_temporal_split(
    n_rows: int,
    requested_test_rows: Optional[int],
) -> Tuple[int, int]:
    if requested_test_rows is None:
        test_rows = max(
            MIN_TEST_ROWS,
            int(math.ceil(FORMAL_TEST_FRACTION * n_rows)),
        )
    else:
        test_rows = max(MIN_TEST_ROWS, int(requested_test_rows))

    if test_rows >= n_rows:
        raise ValueError(
            f"Split temporal imposible: total={n_rows}, test={test_rows}."
        )

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
        raise ValueError("No queda ninguna feature valida en el fold de entrenamiento.")

    return kept, pd.DataFrame(rows)


def candidate_definitions(groups: Dict[str, List[str]]) -> List[Candidate]:
    candidates: List[Candidate] = [
        {
            "candidate_id": "dummy_mean",
            "family": "dummy",
            "feature_group": "labels_only",
            "params": {"strategy": "mean"},
        },
        {
            "candidate_id": "dummy_median",
            "family": "dummy",
            "feature_group": "labels_only",
            "params": {"strategy": "median"},
        },
    ]

    for group_name in groups:
        for alpha in RIDGE_ALPHAS:
            candidates.append(
                {
                    "candidate_id": f"ridge_{group_name}_alpha_{alpha:g}",
                    "family": "ridge",
                    "feature_group": group_name,
                    "params": {"alpha": alpha},
                }
            )

        for index, config in enumerate(HGB_CONFIGS, start=1):
            candidates.append(
                {
                    "candidate_id": f"hgb_{group_name}_cfg_{index}",
                    "family": "hgb",
                    "feature_group": group_name,
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


def make_feature_pipeline(candidate: Candidate, feature_cols: Sequence[str]) -> Pipeline:
    family = candidate["family"]
    params = candidate["params"]

    if family == "ridge":
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
        raise ValueError(f"Familia no soportada por make_feature_pipeline: {family}")

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(feature_cols, scale=scale)),
            ("model", estimator),
        ]
    )


def model_input(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    return df[list(feature_cols)].copy()


def fit_candidate(
    train_df: pd.DataFrame,
    candidate: Candidate,
    groups: Dict[str, List[str]],
) -> Tuple[Any, List[str], pd.DataFrame]:
    family = candidate["family"]

    if family == "dummy":
        estimator = DummyRegressor(strategy=str(candidate["params"]["strategy"]))
        x_dummy = np.zeros((len(train_df), 1), dtype=float)
        estimator.fit(x_dummy, train_df["thm"].to_numpy(dtype=float))
        return estimator, [], pd.DataFrame()

    requested = groups[candidate["feature_group"]]
    kept_features, audit = filter_features_on_training(train_df, requested)
    pipeline = make_feature_pipeline(candidate, kept_features)
    pipeline.fit(
        model_input(train_df, kept_features),
        train_df["thm"].to_numpy(dtype=float),
    )
    return pipeline, kept_features, audit


def predict_candidate(
    fitted_model: Any,
    candidate: Candidate,
    df: pd.DataFrame,
    features: Sequence[str],
) -> np.ndarray:
    if candidate["family"] == "dummy":
        x_dummy = np.zeros((len(df), 1), dtype=float)
        return np.asarray(fitted_model.predict(x_dummy), dtype=float)

    return np.asarray(
        fitted_model.predict(model_input(df, features)),
        dtype=float,
    )


def walk_forward_candidate(
    development_df: pd.DataFrame,
    candidate: Candidate,
    groups: Dict[str, List[str]],
    min_train_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: List[Dict[str, Any]] = []
    audit_rows: List[pd.DataFrame] = []

    for prediction_index in range(min_train_rows, len(development_df)):
        train_df = development_df.iloc[:prediction_index].copy()
        eval_df = development_df.iloc[[prediction_index]].copy()

        fitted, kept_features, audit = fit_candidate(
            train_df=train_df,
            candidate=candidate,
            groups=groups,
        )

        if not audit.empty:
            audit = audit.copy()
            audit["candidate_id"] = candidate["candidate_id"]
            audit["feature_group"] = candidate["feature_group"]
            audit["prediction_index"] = prediction_index
            audit["prediction_timestamp"] = eval_df["sample_timestamp"].iloc[0]
            audit_rows.append(audit)

        prediction = float(
            predict_candidate(
                fitted_model=fitted,
                candidate=candidate,
                df=eval_df,
                features=kept_features,
            )[0]
        )
        y_true = float(eval_df["thm"].iloc[0])

        prediction_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "family": candidate["family"],
                "feature_group": candidate["feature_group"],
                "params_json": json.dumps(candidate["params"], ensure_ascii=False, sort_keys=True),
                "sample_id": eval_df["sample_id"].iloc[0],
                "sample_timestamp": eval_df["sample_timestamp"].iloc[0],
                "sample_point": eval_df["sample_point"].iloc[0],
                "relative_file": eval_df["relative_file"].iloc[0],
                "train_rows": len(train_df),
                "train_start": train_df["sample_timestamp"].min(),
                "train_end": train_df["sample_timestamp"].max(),
                "train_target_mean": float(train_df["thm"].mean()),
                "n_features": len(kept_features),
                "features": "|".join(kept_features),
                "y_true": y_true,
                "y_pred": prediction,
                "error": prediction - y_true,
                "abs_error": abs(prediction - y_true),
            }
        )

    return (
        pd.DataFrame(prediction_rows),
        pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame(),
    )


def run_walk_forward_selection(
    development_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    min_train_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Candidate]]:
    all_predictions: List[pd.DataFrame] = []
    all_audits: List[pd.DataFrame] = []
    candidates_by_id: Dict[str, Candidate] = {}

    candidates = candidate_definitions(groups)
    total = len(candidates)

    for index, candidate in enumerate(candidates, start=1):
        candidates_by_id[candidate["candidate_id"]] = candidate
        print(
            f"[walk-forward {index}/{total}] {candidate['candidate_id']} "
            f"(group={candidate['feature_group']})"
        )

        predictions, audits = walk_forward_candidate(
            development_df=development_df,
            candidate=candidate,
            groups=groups,
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
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "feature_group": candidate["feature_group"],
                "params_json": json.dumps(candidate["params"], ensure_ascii=False, sort_keys=True),
                **regression_metrics(group["y_true"], group["y_pred"]),
            }
        )

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["rmse", "mae", "abs_bias", "candidate_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    metrics_df["global_rank"] = np.arange(1, len(metrics_df) + 1)
    metrics_df["family_rank"] = (
        metrics_df.groupby("family")["rmse"]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    best_dummy_mae = float(
        metrics_df[metrics_df["family"] == "dummy"]
        .sort_values("mae")
        .iloc[0]["mae"]
    )

    metrics_df["mae_improvement_vs_best_dummy"] = (
        best_dummy_mae - metrics_df["mae"]
    )
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


def feature_group_summary(walk_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for family in ("ridge", "hgb"):
        family_df = walk_metrics[walk_metrics["family"] == family]
        for group_name, group in family_df.groupby("feature_group"):
            best = group.sort_values(["rmse", "mae", "abs_bias"]).iloc[0]
            rows.append(
                {
                    "family": family,
                    "feature_group": group_name,
                    "best_candidate_id": best["candidate_id"],
                    "n": int(best["n"]),
                    "mae": float(best["mae"]),
                    "rmse": float(best["rmse"]),
                    "bias": float(best["bias"]),
                    "r2": float(best["r2"]),
                    "pearson_r": float(best["pearson_r"]),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["family", "rmse", "mae"]
    ).reset_index(drop=True)


def evaluate_selected_candidates_on_test(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_ids: Sequence[str],
    candidates_by_id: Dict[str, Candidate],
    groups: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]], pd.DataFrame]:
    metric_rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    fitted: Dict[str, Dict[str, Any]] = {}
    audit_frames: List[pd.DataFrame] = []

    for candidate_id in selected_ids:
        candidate = candidates_by_id[candidate_id]
        fitted_model, kept_features, audit = fit_candidate(
            train_df=development_df,
            candidate=candidate,
            groups=groups,
        )

        if not audit.empty:
            audit = audit.copy()
            audit["candidate_id"] = candidate_id
            audit["feature_group"] = candidate["feature_group"]
            audit_frames.append(audit)

        predictions = predict_candidate(
            fitted_model=fitted_model,
            candidate=candidate,
            df=test_df,
            features=kept_features,
        )

        metrics = regression_metrics(test_df["thm"], predictions)

        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "family": candidate["family"],
                "feature_group": candidate["feature_group"],
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
        pred_df["feature_group"] = candidate["feature_group"]
        prediction_frames.append(pred_df)

        fitted[candidate_id] = {
            "model": fitted_model,
            "candidate": candidate,
            "features": kept_features,
            "feature_audit": audit,
        }

    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["rmse", "mae", "abs_bias"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    audits_df = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

    best_dummy_mae = float(
        metrics_df[metrics_df["family"] == "dummy"]
        .sort_values("mae")
        .iloc[0]["mae"]
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
    candidate = fitted_info["candidate"]
    features = fitted_info["features"]

    if candidate["family"] == "dummy" or len(features) == 0 or len(test_df) < 3:
        return pd.DataFrame()

    result = permutation_importance(
        fitted_info["model"],
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
            "note": "Exploratory only: final test contains only five samples.",
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


def drift_development_vs_test(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for col in feature_cols:
        dev = pd.to_numeric(development_df[col], errors="coerce")
        tst = pd.to_numeric(test_df[col], errors="coerce")
        dev_valid = dev.dropna()
        tst_valid = tst.dropna()

        if dev_valid.empty:
            continue

        dev_min = float(dev_valid.min())
        dev_max = float(dev_valid.max())
        dev_mean = float(dev_valid.mean())
        dev_std = float(dev_valid.std()) if len(dev_valid) > 1 else math.nan
        dev_p25 = float(dev_valid.quantile(0.25))
        dev_p50 = float(dev_valid.quantile(0.50))
        dev_p75 = float(dev_valid.quantile(0.75))
        dev_iqr = dev_p75 - dev_p25

        if tst_valid.empty:
            test_mean = math.nan
            test_p50 = math.nan
            outside_ratio = math.nan
            std_shift = math.nan
            robust_shift = math.nan
        else:
            test_mean = float(tst_valid.mean())
            test_p50 = float(tst_valid.quantile(0.50))
            outside_ratio = float(((tst_valid < dev_min) | (tst_valid > dev_max)).mean())
            std_shift = safe_divide(abs(test_mean - dev_mean), dev_std)
            robust_shift = safe_divide(abs(test_p50 - dev_p50), dev_iqr)

        rows.append(
            {
                "feature": col,
                "development_count": int(dev_valid.size),
                "test_count": int(tst_valid.size),
                "development_missing_ratio": float(dev.isna().mean()),
                "test_missing_ratio": float(tst.isna().mean()),
                "development_mean": dev_mean,
                "test_mean": test_mean,
                "development_std": dev_std,
                "development_min": dev_min,
                "development_p25": dev_p25,
                "development_p50": dev_p50,
                "development_p75": dev_p75,
                "development_max": dev_max,
                "test_p50": test_p50,
                "test_outside_development_range_ratio": outside_ratio,
                "standardized_mean_shift": std_shift,
                "robust_median_shift_iqr": robust_shift,
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["test_outside_development_range_ratio", "robust_median_shift_iqr"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def create_bundle(
    candidate_id: str,
    candidate: Candidate,
    fitted_model: Any,
    features: Sequence[str],
    training_df: pd.DataFrame,
    clean_csv: Path,
    labels_csv: Path,
    training_scope: str,
    cadence_minutes: float,
    min_coverage_ratio: float,
) -> Dict[str, Any]:
    return {
        "version": VERSION,
        "created_at": datetime.now().isoformat(),
        "scope": "Huelva THM soft-sensor prediction only",
        "training_scope": training_scope,
        "clean_timeseries_csv": str(clean_csv),
        "labels_csv": str(labels_csv),
        "candidate_id": candidate_id,
        "family": candidate["family"],
        "feature_group": candidate["feature_group"],
        "params": candidate["params"],
        "features": list(features),
        "model": fitted_model,
        "window_hours": list(WINDOW_HOURS),
        "source_cadence_minutes": cadence_minutes,
        "min_window_coverage_ratio": min_coverage_ratio,
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

    ax.set_title("Huelva THM - test temporal final v0.5 sin color")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("THM")
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
    ax.set_title(f"Huelva THM - walk-forward desarrollo - {candidate_id}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("THM")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def markdown_table_from_metrics(df: pd.DataFrame, max_rows: Optional[int] = None) -> List[str]:
    selected = df.head(max_rows) if max_rows else df
    lines = [
        "| Candidato | Familia | Grupo | n | MAE | RMSE | Bias | R2 | Pearson r |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in selected.iterrows():
        lines.append(
            f"| `{row['candidate_id']}` | {row['family']} | {row['feature_group']} | "
            f"{int(row['n'])} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['bias']:.4f} | {row['r2']:.4f} | {row['pearson_r']:.4f} |"
        )

    return lines


def write_report(
    output_path: Path,
    clean_csv: Path,
    labels_csv: Path,
    full_df: pd.DataFrame,
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    group_summary: pd.DataFrame,
    walk_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    primary_hgb_id: str,
    best_overall_id: str,
    walk_min_train_rows: int,
    cadence_minutes: float,
    min_coverage_ratio: float,
    drift_df: pd.DataFrame,
) -> None:
    primary_candidate = walk_metrics[walk_metrics["candidate_id"] == primary_hgb_id].iloc[0]

    lines: List[str] = [
        "# Validacion formal soft-sensor THM Huelva v0.5 sin color",
        "",
        f"- Serie temporal: `{clean_csv}`",
        f"- Etiquetas THM: `{labels_csv}`",
        "- Alcance: prediccion THM mediante soft-sensor.",
        "- Fuera de alcance: propagacion y alertas.",
        "",
        "## Protocolo formal",
        "",
        f"- El test final se fija antes de evaluar: **{len(test_df)} muestras**, "
        "correspondientes a ceil(20% del total), con minimo de cinco.",
        "- La seleccion sigue siendo walk-forward expanding-window solo sobre desarrollo.",
        "- Se reconstruyen features desde la serie temporal limpia.",
        "- Ventanas temporales: 6, 12, 24, 48 y 72 horas.",
        "- Se comparan tres grupos de features para limitar la complejidad.",
        "- El mejor candidato global en walk-forward es el modelo seleccionado; "
        "Dummy, Ridge y HGB se comparan bajo el mismo protocolo temporal.",
        "- Se anade un diagnostico post-test de drift entre desarrollo y test.",
        "",
        "## Datos y split",
        "",
        f"- Muestras totales: **{len(full_df)}**",
        f"- Desarrollo: **{len(development_df)}**",
        f"- Test final: **{len(test_df)}**",
        f"- Inicio walk-forward: **{walk_min_train_rows}** muestras",
        f"- Cadencia mediana de la multiparametrica: **{cadence_minutes:.3f} min**",
        f"- Cobertura minima por ventana: **{min_coverage_ratio:.0%}**",
        f"- Rango: **{full_df['sample_timestamp'].min()} -> {full_df['sample_timestamp'].max()}**",
        "",
        "## Grupos de features",
        "",
    ]

    for group_name, cols in groups.items():
        lines.append(f"- `{group_name}`: {len(cols)} features candidatas.")

    lines.extend(
        [
            "",
            "## Mejor candidato por grupo/familia",
            "",
            "```text",
            group_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
            "```",
            "",
            "## Seleccion walk-forward",
            "",
            f"- Mejor HGB comparador: `{primary_hgb_id}`",
            f"- Grupo del mejor HGB: `{primary_candidate['feature_group']}`",
            f"- Mejor candidato global: `{best_overall_id}`",
            "",
        ]
    )
    lines.extend(markdown_table_from_metrics(walk_metrics, max_rows=12))

    lines.extend(
        [
            "",
            "## Evaluacion final en test",
            "",
        ]
    )
    lines.extend(markdown_table_from_metrics(test_metrics))

    lines.extend(
        [
            "",
            "## Drift desarrollo -> test",
            "",
            "El drift se calcula despues de cerrar la seleccion y es diagnostico; no interviene en la eleccion del modelo.",
            "",
            "```text",
            drift_df.head(20).to_string(index=False, float_format=lambda value: f"{value:.4f}"),
            "```",
            "",
            "## Interpretacion",
            "",
            f"Con {len(full_df)} muestras de laboratorio, esta ejecucion constituye la primera "
            "validacion formal reproducible con el conjunto actualmente disponible. La seleccion "
            "se realiza exclusivamente dentro de desarrollo mediante walk-forward; el test final "
            "permanece aislado hasta cerrar modelo, grupo de features e hiperparametros.",
            "",
            "El bundle de evaluacion se entrena solo con desarrollo y corresponde a las metricas "
            f"del test final. El bundle full-data se reentrena despues con las {len(full_df)} "
            "muestras disponibles, una vez cerrada la evaluacion, y no interviene en las metricas "
            "del test formal.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    clean_csv = Path(args.clean_csv).resolve() if args.clean_csv else CLEAN_TIMESERIES_CSV.resolve()
    labels_csv = Path(args.labels_csv).resolve() if args.labels_csv else LABEL_DATASET_CSV.resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else OUTPUT_DIR.resolve()
    write_excel = WRITE_EXCEL and not args.no_excel

    output_dir.mkdir(parents=True, exist_ok=True)

    clean_df, cadence_minutes = load_clean_timeseries(clean_csv)
    labels_df = load_labels(labels_csv)

    full_df, coverage_df = build_multwindow_feature_dataset(
        clean_df=clean_df,
        labels_df=labels_df,
        cadence_minutes=cadence_minutes,
        min_coverage_ratio=float(args.min_window_coverage_ratio),
    )

    groups = feature_groups()
    all_generated_features = list(
        dict.fromkeys(
            feature
            for columns in groups.values()
            for feature in columns
        )
    )

    missing_generated = [
        feature for feature in all_generated_features if feature not in full_df.columns
    ]
    if missing_generated:
        raise ValueError(f"Features generadas ausentes: {missing_generated}")

    baseline_qa = compare_with_v0_2_24h(full_df, labels_df)

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
    print(f"Version:               {VERSION}")
    print(f"Serie temporal:        {clean_csv}")
    print(f"Etiquetas:             {labels_csv}")
    print(f"Salida:                {output_dir}")
    print(f"Muestras:              {len(full_df)}")
    print(f"Desarrollo/Test:       {len(development_df)}/{len(test_df)}")
    print(f"Walk-forward min train:{walk_min_train_rows}")
    print(f"Cadencia mediana:      {cadence_minutes:.3f} min")
    print(f"Ventanas:              {WINDOW_HOURS}")
    print(f"Grupos de features:    {list(groups)}")
    print("")

    walk_metrics, walk_predictions, walk_audits, candidates_by_id = run_walk_forward_selection(
        development_df=development_df,
        groups=groups,
        min_train_rows=walk_min_train_rows,
    )

    group_summary = feature_group_summary(walk_metrics)

    best_dummy_id = best_candidate_id(walk_metrics, family="dummy")
    best_ridge_id = best_candidate_id(walk_metrics, family="ridge")
    primary_hgb_id = best_candidate_id(walk_metrics, family=PRIMARY_FAMILY)
    best_overall_id = best_candidate_id(walk_metrics)

    selected_ids = list(
        dict.fromkeys(
            [best_dummy_id, best_ridge_id, primary_hgb_id, best_overall_id]
        )
    )

    test_metrics, test_predictions, fitted_test, test_audits = evaluate_selected_candidates_on_test(
        development_df=development_df,
        test_df=test_df,
        selected_ids=selected_ids,
        candidates_by_id=candidates_by_id,
        groups=groups,
    )

    primary_eval_info = fitted_test[primary_hgb_id]
    primary_features = primary_eval_info["features"]

    primary_importance = permutation_importance_exploratory(
        fitted_info=primary_eval_info,
        test_df=test_df,
    )

    correlations = development_feature_correlations(
        development_df=development_df,
        feature_cols=all_generated_features,
    )

    drift_df = drift_development_vs_test(
        development_df=development_df,
        test_df=test_df,
        feature_cols=primary_features,
    )

    primary_eval_bundle = create_bundle(
        candidate_id=primary_hgb_id,
        candidate=primary_eval_info["candidate"],
        fitted_model=primary_eval_info["model"],
        features=primary_features,
        training_df=development_df,
        clean_csv=clean_csv,
        labels_csv=labels_csv,
        training_scope="development_only_for_final_test_evaluation",
        cadence_minutes=cadence_minutes,
        min_coverage_ratio=float(args.min_window_coverage_ratio),
    )

    overall_eval_info = fitted_test[best_overall_id]
    overall_eval_bundle = create_bundle(
        candidate_id=best_overall_id,
        candidate=overall_eval_info["candidate"],
        fitted_model=overall_eval_info["model"],
        features=overall_eval_info["features"],
        training_df=development_df,
        clean_csv=clean_csv,
        labels_csv=labels_csv,
        training_scope="development_only_for_final_test_evaluation",
        cadence_minutes=cadence_minutes,
        min_coverage_ratio=float(args.min_window_coverage_ratio),
    )

    primary_full_model, primary_full_features, primary_full_audit = fit_candidate(
        train_df=full_df,
        candidate=candidates_by_id[primary_hgb_id],
        groups=groups,
    )

    primary_full_bundle = create_bundle(
        candidate_id=primary_hgb_id,
        candidate=candidates_by_id[primary_hgb_id],
        fitted_model=primary_full_model,
        features=primary_full_features,
        training_df=full_df,
        clean_csv=clean_csv,
        labels_csv=labels_csv,
        training_scope="all_available_samples_after_evaluation",
        cadence_minutes=cadence_minutes,
        min_coverage_ratio=float(args.min_window_coverage_ratio),
    )

    overall_full_model, overall_full_features, overall_full_audit = fit_candidate(
        train_df=full_df,
        candidate=candidates_by_id[best_overall_id],
        groups=groups,
    )

    overall_full_bundle = create_bundle(
        candidate_id=best_overall_id,
        candidate=candidates_by_id[best_overall_id],
        fitted_model=overall_full_model,
        features=overall_full_features,
        training_df=full_df,
        clean_csv=clean_csv,
        labels_csv=labels_csv,
        training_scope="all_available_samples_after_evaluation",
        cadence_minutes=cadence_minutes,
        min_coverage_ratio=float(args.min_window_coverage_ratio),
    )

    full_df.to_csv(
        output_dir / "multwindow_feature_dataset_v0_5_no_color.csv",
        index=False,
        encoding="utf-8-sig",
    )
    coverage_df.to_csv(
        output_dir / "multwindow_feature_coverage_v0_5_no_color.csv",
        index=False,
        encoding="utf-8-sig",
    )
    baseline_qa.to_csv(
        output_dir / "baseline_24h_reproduction_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    walk_metrics.to_csv(
        output_dir / "walk_forward_candidate_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    walk_predictions.to_csv(
        output_dir / "walk_forward_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    walk_audits.to_csv(
        output_dir / "walk_forward_feature_filter_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_summary.to_csv(
        output_dir / "feature_group_walk_forward_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_metrics.to_csv(
        output_dir / "final_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_predictions.to_csv(
        output_dir / "final_test_predictions_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_audits.to_csv(
        output_dir / "final_test_feature_filter_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    correlations.to_csv(
        output_dir / "feature_target_correlations_development.csv",
        index=False,
        encoding="utf-8-sig",
    )
    drift_df.to_csv(
        output_dir / "feature_drift_development_vs_final_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary_importance.to_csv(
        output_dir / "primary_hgb_permutation_importance_test_exploratory.csv",
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

    joblib.dump(
        primary_eval_bundle,
        output_dir / "primary_hgb_evaluation_bundle.joblib",
    )
    joblib.dump(
        primary_full_bundle,
        output_dir / "primary_hgb_full_data_bundle.joblib",
    )
    joblib.dump(
        overall_eval_bundle,
        output_dir / "best_overall_evaluation_bundle.joblib",
    )
    joblib.dump(
        overall_full_bundle,
        output_dir / "best_overall_full_data_bundle.joblib",
    )

    metadata = {
        "version": VERSION,
        "experiment_note": EXPERIMENT_NOTE,
        "created_at": datetime.now().isoformat(),
        "scope": "prediction_only",
        "propagation_and_alerts_in_scope": False,
        "clean_timeseries_csv": str(clean_csv),
        "labels_csv": str(labels_csv),
        "output_dir": str(output_dir),
        "n_samples": len(full_df),
        "development_rows": len(development_df),
        "test_rows": len(test_df),
        "formal_test_fraction": FORMAL_TEST_FRACTION,
        "formal_test_rule": "ceil(20% of total samples), minimum 5",
        "walk_forward_min_train_rows": walk_min_train_rows,
        "walk_forward_predictions_per_candidate": len(development_df) - walk_min_train_rows,
        "window_hours": list(WINDOW_HOURS),
        "source_cadence_minutes": cadence_minutes,
        "min_window_coverage_ratio": float(args.min_window_coverage_ratio),
        "feature_groups": groups,
        "best_dummy_candidate_id": best_dummy_id,
        "best_ridge_candidate_id": best_ridge_id,
        "primary_hgb_candidate_id": primary_hgb_id,
        "best_overall_candidate_id": best_overall_id,
        "selection_criterion": "walk_forward_rmse_then_mae_then_abs_bias_on_development",
        "final_test_used_for_selection": False,
        "drift_analysis_used_for_selection": False,
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
        output_path=output_dir / "training_report_v0_5_no_color.md",
        clean_csv=clean_csv,
        labels_csv=labels_csv,
        full_df=full_df,
        development_df=development_df,
        test_df=test_df,
        groups=groups,
        group_summary=group_summary,
        walk_metrics=walk_metrics,
        test_metrics=test_metrics,
        primary_hgb_id=primary_hgb_id,
        best_overall_id=best_overall_id,
        walk_min_train_rows=walk_min_train_rows,
        cadence_minutes=cadence_minutes,
        min_coverage_ratio=float(args.min_window_coverage_ratio),
        drift_df=drift_df,
    )

    if write_excel:
        try:
            with pd.ExcelWriter(
                output_dir / "training_report_v0_5_no_color.xlsx",
                engine="openpyxl",
            ) as writer:
                group_summary.to_excel(writer, sheet_name="feature_groups", index=False)
                walk_metrics.to_excel(writer, sheet_name="walk_metrics", index=False)
                walk_predictions.to_excel(writer, sheet_name="walk_predictions", index=False)
                test_metrics.to_excel(writer, sheet_name="test_metrics", index=False)
                test_predictions.to_excel(writer, sheet_name="test_predictions", index=False)
                drift_df.to_excel(writer, sheet_name="feature_drift", index=False)
                baseline_qa.to_excel(writer, sheet_name="baseline_24h_qa", index=False)
                correlations.to_excel(writer, sheet_name="feature_correlations", index=False)
                coverage_df.to_excel(writer, sheet_name="window_coverage", index=False)
                primary_importance.to_excel(writer, sheet_name="importance_exploratory", index=False)
        except Exception as exc:
            print(f"[WARN] No se pudo generar Excel: {type(exc).__name__}: {exc}")

    print("")
    print("========================================")
    print("RESUMEN VALIDACION HUELVA v0.5 SIN COLOR")
    print("========================================")
    print(f"Mejor Dummy WF:       {best_dummy_id}")
    print(f"Mejor Ridge WF:       {best_ridge_id}")
    print(f"Mejor HGB WF:         {primary_hgb_id}")
    print(f"Mejor global WF:      {best_overall_id}")
    print("")
    print("Mejor por grupo/familia:")
    print(
        group_summary.round(4).to_string(index=False)
    )
    print("")
    print("Top 8 walk-forward:")
    print(
        walk_metrics[
            [
                "candidate_id",
                "family",
                "feature_group",
                "n",
                "mae",
                "rmse",
                "bias",
                "r2",
                "pearson_r",
            ]
        ]
        .head(8)
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
                "feature_group",
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

    primary_test = test_metrics[
        test_metrics["candidate_id"] == primary_hgb_id
    ].iloc[0]
    best_dummy_test = (
        test_metrics[test_metrics["family"] == "dummy"]
        .sort_values("mae")
        .iloc[0]
    )

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

    print(
        "[NOTA] El test final y el drift no se usan para seleccionar grupo, "
        "features ni hiperparametros."
    )
    print(
        f"[NOTA] Validacion formal con {len(full_df)} muestras: "
        f"{len(development_df)} desarrollo / {len(test_df)} test final aislado."
    )
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
