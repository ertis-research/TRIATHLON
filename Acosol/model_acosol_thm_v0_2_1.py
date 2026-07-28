from __future__ import annotations

"""Análisis HGB de Acosol con trazabilidad de situaciones especiales de planta."""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


VERSION = "v0.2.1"
TARGET_COL = "thm"
TIME_COL = "timestamp"

RAW_INPUT_FEATURES = [
    "turbidez",
    "cloro",
    "ph",
    "conductividad",
    "bromuro",
    "cot",
]

BASE_TIME_FEATURES = [
    "hour",
    "month",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]

UNITS_CONFIRMED_BY_PLANT = {
    "turbidez": "NTU",
    "cloro": "mg/L",
    "ph": "unidades de pH",
    "conductividad": "µS/cm a 20 ºC",
    "thm": "µg/L",
    "bromuro": "mg/L",
    "cot": "mg/L",
}

# Umbrales diagnósticos derivados de los valores observados y del contexto confirmado
# por planta. No son límites operativos ni umbrales de alarma definidos por planta.
DEFAULT_PH_SPECIAL_LOW_THRESHOLD = 5.0
DEFAULT_BROMURO_SPECIAL_HIGH_THRESHOLD = 10.0

# Rangos orientativos de auditoría. No corrigen ni eliminan datos automáticamente.
AUDIT_RANGES = {
    "turbidez": (0.0, 50.0),
    "cloro": (0.0, 5.0),
    "ph": (5.0, 10.0),
    "conductividad": (50.0, 2500.0),
    "bromuro": (0.0, 10.0),
    "cot": (0.0, 30.0),
    "thm": (0.0, 200.0),
}

NON_FEATURE_COLS = {
    "timestamp",
    "fecha",
    "hora_exacta",
    "fecha_raw",
    "hora_raw",
    "thm",
    "quality_flags",
    "source_file",
    "source_sheet",
    "source_priority",
    "original_row",
    "year",
    "special_flags",
    "ph_special_low",
    "bromuro_special_high",
    "special_plant_condition_candidate",
}

LAG_SPECS_HOURS = {
    "turbidez": [1, 3, 24],
    "cloro": [1, 3, 6, 24],
    "ph": [1, 3, 6, 24],
    "conductividad": [1, 3, 6, 24],
    "bromuro": [24],
    "cot": [6, 24],
}

ROLLING_SPECS_HOURS = {
    "turbidez": [6, 24],
    "cloro": [6, 24],
    "ph": [6, 24],
    "conductividad": [6, 24],
    "bromuro": [24],
    "cot": [24],
}

DIFF_SPECS_HOURS = {
    "cloro": [1, 3],
    "ph": [1, 3],
    "conductividad": [1, 3],
    "turbidez": [1, 3],
}


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    split_summary: pd.DataFrame


@dataclass
class ModelSpec:
    model_name: str
    feature_set: str
    model: object
    features: List[str]
    is_main_candidate: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento Acosol THM v0.2.1 con diagnóstico de situaciones especiales")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV generado por el ETL de Acosol, normalmente acosol_dataset_A_wide_v0_1.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_outputs_acosol_thm_v0_2_1_A_2025_2026",
        help="Carpeta de salida.",
    )
    parser.add_argument(
        "--experiment-name",
        default="A_2025_2026_v0_2_1",
        help="Nombre corto del experimento.",
    )
    parser.add_argument(
        "--year-filter",
        default="all",
        choices=["all", "2025", "2026"],
        help="Filtra el dataset por año antes del split temporal.",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=7,
        help="Días finales usados para test.",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=7,
        help="Días inmediatamente anteriores al test usados para validación.",
    )
    parser.add_argument(
        "--feature-mode",
        choices=["base", "history", "compare"],
        default="base",
        help=(
            "base: HGB estable actual con variables actuales + temporales; "
            "history: añade lags/rolling/diffs de entradas; "
            "compare: compara hgb_base e hgb_history."
        ),
    )
    parser.add_argument(
        "--special-mode",
        choices=["compare", "exclude", "keep"],
        default="compare",
        help=(
            "compare: entrena con filas especiales conservadas y excluidas; "
            "exclude: solo entrena excluyéndolas; "
            "keep: solo entrena conservándolas con flags diagnósticos."
        ),
    )
    parser.add_argument(
        "--ph-special-low-threshold",
        type=float,
        default=DEFAULT_PH_SPECIAL_LOW_THRESHOLD,
        help="Umbral para marcar pH bajo asociado a situación especial/calibración.",
    )
    parser.add_argument(
        "--bromuro-special-high-threshold",
        type=float,
        default=DEFAULT_BROMURO_SPECIAL_HIGH_THRESHOLD,
        help="Umbral para marcar bromuro alto asociado a situación especial/calibración.",
    )
    parser.add_argument(
        "--lag-tolerance-minutes",
        type=int,
        default=20,
        help="Tolerancia para emparejar lags horarios por timestamp.",
    )
    parser.add_argument(
        "--keep-hard-out-of-range",
        action="store_true",
        help=(
            "Por defecto se descartan filas marcadas con hard_out_of_range en quality_flags. "
            "Activa esta opción para conservarlas."
        ),
    )
    parser.add_argument("--random-state", type=int, default=42, help="Semilla reproducible.")
    parser.add_argument("--hgb-max-iter", type=int, default=700, help="Iteraciones máximas de HGB.")
    parser.add_argument("--hgb-learning-rate", type=float, default=0.04, help="Learning rate de HGB.")
    parser.add_argument("--hgb-l2", type=float, default=0.05, help="Regularización L2 de HGB.")
    parser.add_argument("--hgb-min-samples-leaf", type=int, default=20, help="min_samples_leaf de HGB.")
    parser.add_argument("--no-plots", action="store_true", help="No generar gráficas PNG.")
    args = parser.parse_args()
    if args.test_days <= 0 or args.val_days <= 0:
        parser.error("--test-days y --val-days deben ser mayores que cero.")
    if args.lag_tolerance_minutes < 0:
        parser.error("--lag-tolerance-minutes no puede ser negativo.")
    if args.hgb_max_iter <= 0 or args.hgb_learning_rate <= 0 or args.hgb_min_samples_leaf <= 0:
        parser.error("Los parámetros principales de HGB deben ser mayores que cero.")
    if args.hgb_l2 < 0:
        parser.error("--hgb-l2 no puede ser negativo.")
    if not 0.0 <= args.ph_special_low_threshold <= 14.0:
        parser.error("--ph-special-low-threshold debe estar entre 0 y 14.")
    if args.bromuro_special_high_threshold < 0:
        parser.error("--bromuro-special-high-threshold no puede ser negativo.")
    return args


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if not np.any(mask):
        return np.nan
    return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]) * 100.0)


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) <= 1:
        return np.nan
    return float(r2_score(y_true, y_pred))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    residuals = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "smape_pct": smape(y_true, y_pred),
        "bias_pred_minus_true": float(np.mean(residuals)),
        "median_abs_error": float(np.median(np.abs(residuals))),
        "p90_abs_error": float(np.percentile(np.abs(residuals), 90)),
        "p95_abs_error": float(np.percentile(np.abs(residuals), 95)),
        "max_abs_error": float(np.max(np.abs(residuals))),
        "r2": safe_r2(np.asarray(y_true), np.asarray(y_pred)),
    }


def load_dataset(input_csv: Path, year_filter: str, keep_hard_out_of_range: bool) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"No existe el fichero de entrada: {input_csv}")

    df = pd.read_csv(input_csv)
    if TIME_COL not in df.columns:
        raise ValueError(f"El CSV no contiene la columna obligatoria {TIME_COL!r}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"El CSV no contiene la columna objetivo {TARGET_COL!r}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL]).copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    if "year" not in df.columns:
        df["year"] = df[TIME_COL].dt.year
    if year_filter != "all":
        df = df[df["year"] == int(year_filter)].copy()

    if not keep_hard_out_of_range and "quality_flags" in df.columns:
        q = df["quality_flags"].fillna("").astype(str)
        df = df[~q.str.contains("hard_out_of_range", case=False, regex=False)].copy()

    for col in RAW_INPUT_FEATURES + [TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if df.empty:
        raise ValueError("El dataset queda vacío después de aplicar filtros.")
    return df


def add_validated_special_flags(
    df: pd.DataFrame,
    ph_special_low_threshold: float,
    bromuro_special_high_threshold: float,
) -> pd.DataFrame:
    df = df.copy()
    ph = pd.to_numeric(df["ph"], errors="coerce") if "ph" in df.columns else pd.Series(False, index=df.index)
    bromuro = pd.to_numeric(df["bromuro"], errors="coerce") if "bromuro" in df.columns else pd.Series(False, index=df.index)

    df["ph_special_low"] = ph.notna() & (ph < ph_special_low_threshold)
    df["bromuro_special_high"] = bromuro.notna() & (bromuro > bromuro_special_high_threshold)
    df["special_plant_condition_candidate"] = df["ph_special_low"] | df["bromuro_special_high"]

    flags = []
    for _, row in df.iterrows():
        current_flags = []
        if bool(row.get("ph_special_low", False)):
            current_flags.append("ph_special_low_validated_special_context")
        if bool(row.get("bromuro_special_high", False)):
            current_flags.append("bromuro_special_high_validated_special_context")
        flags.append(";".join(current_flags))
    df["special_flags"] = flags
    return df


def build_special_condition_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        TIME_COL,
        "year",
        "source_file",
        "quality_flags",
        "special_flags",
        "ph_special_low",
        "bromuro_special_high",
        "special_plant_condition_candidate",
        "turbidez",
        "cloro",
        "ph",
        "conductividad",
        "thm",
        "bromuro",
        "cot",
    ]
    keep_cols = [c for c in cols if c in df.columns]
    out = df[df["special_plant_condition_candidate"]].copy()
    return out[keep_cols].sort_values(TIME_COL).reset_index(drop=True)


def summarize_special_flags(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "special_plant_condition_candidate" not in df.columns:
        return pd.DataFrame()
    total = len(df)
    for flag_col in ["ph_special_low", "bromuro_special_high", "special_plant_condition_candidate"]:
        n = int(df[flag_col].sum()) if flag_col in df.columns else 0
        rows.append({"flag": flag_col, "rows": n, "pct_total": float(n / total * 100.0) if total else np.nan})
    if "special_plant_condition_candidate" in df.columns:
        tmp = df[df["special_plant_condition_candidate"]].copy()
        if not tmp.empty:
            tmp["date"] = tmp[TIME_COL].dt.date
            by_date = tmp.groupby("date").size().reset_index(name="rows")
            for _, r in by_date.iterrows():
                rows.append({"flag": f"special_by_date:{r['date']}", "rows": int(r["rows"]), "pct_total": float(r["rows"] / total * 100.0)})
    return pd.DataFrame(rows)


def build_audit_range_report(
    df: pd.DataFrame,
    ph_special_low_threshold: float,
    bromuro_special_high_threshold: float,
) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in AUDIT_RANGES.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        mask = s.notna() & ((s < lo) | (s > hi))
        for _, row in df.loc[mask].iterrows():
            value = float(row[col])
            diagnostic = "review"
            if value > hi and value / 10.0 <= hi:
                diagnostic = "possible_decimal_shift_x10"
            elif value > hi and value / 100.0 <= hi:
                diagnostic = "possible_decimal_shift_x100"
            elif value > hi and value / 1000.0 <= hi:
                diagnostic = "possible_decimal_shift_x1000"
            if col == "ph" and value < ph_special_low_threshold:
                diagnostic = "plant_confirmed_special_context_candidate"
            if col == "bromuro" and value > bromuro_special_high_threshold:
                diagnostic = "plant_confirmed_special_context_candidate"
            rows.append(
                {
                    "timestamp": row[TIME_COL],
                    "column": col,
                    "value": value,
                    "unit": UNITS_CONFIRMED_BY_PLANT.get(col, ""),
                    "audit_min": lo,
                    "audit_max": hi,
                    "diagnostic": diagnostic,
                    "special_flags": row.get("special_flags", ""),
                    "source_file": row.get("source_file", ""),
                    "quality_flags": row.get("quality_flags", ""),
                }
            )
    return pd.DataFrame(rows).sort_values(["timestamp", "column"]) if rows else pd.DataFrame(
        columns=["timestamp", "column", "value", "unit", "audit_min", "audit_max", "diagnostic", "special_flags", "source_file", "quality_flags"]
    )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df[TIME_COL].dt.hour.astype(float)
    df["month"] = df[TIME_COL].dt.month.astype(float)
    df["dayofyear"] = df[TIME_COL].dt.dayofyear.astype(float)
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["dayofyear_sin"] = np.sin(2.0 * np.pi * df["dayofyear"] / 365.25)
    df["dayofyear_cos"] = np.cos(2.0 * np.pi * df["dayofyear"] / 365.25)
    return df


def add_lag_features(df: pd.DataFrame, lag_specs: Dict[str, List[int]], tolerance_minutes: int) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy().sort_values(TIME_COL).reset_index(drop=True)
    created: List[str] = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for col, hours_list in lag_specs.items():
        if col not in df.columns:
            continue
        right = df[[TIME_COL, col]].copy()
        right[col] = pd.to_numeric(right[col], errors="coerce")
        right = right.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
        for h in hours_list:
            new_col = f"{col}_lag_{h}h"
            left = df[[TIME_COL]].copy()
            left["_row_id"] = np.arange(len(left))
            left["_lookup_ts"] = left[TIME_COL] - pd.Timedelta(hours=h)
            right_renamed = right.rename(columns={TIME_COL: "_lag_ts", col: new_col})
            merged = pd.merge_asof(
                left.sort_values("_lookup_ts"),
                right_renamed.sort_values("_lag_ts"),
                left_on="_lookup_ts",
                right_on="_lag_ts",
                direction="nearest",
                tolerance=tolerance,
            ).sort_values("_row_id")
            df[new_col] = merged[new_col].to_numpy()
            created.append(new_col)
    return df, created


def add_rolling_features(df: pd.DataFrame, rolling_specs: Dict[str, List[int]]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy().sort_values(TIME_COL).reset_index(drop=True)
    created: List[str] = []
    ts_indexed = df.set_index(TIME_COL, drop=False).sort_index()
    for col, hours_list in rolling_specs.items():
        if col not in ts_indexed.columns:
            continue
        series = pd.to_numeric(ts_indexed[col], errors="coerce")
        for h in hours_list:
            new_col = f"{col}_roll_mean_{h}h"
            ts_indexed[new_col] = series.rolling(f"{h}h", closed="left", min_periods=1).mean()
            created.append(new_col)
    return ts_indexed.reset_index(drop=True).sort_values(TIME_COL).reset_index(drop=True), created


def add_diff_features(df: pd.DataFrame, diff_specs: Dict[str, List[int]]) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    created: List[str] = []
    for col, hours_list in diff_specs.items():
        if col not in df.columns:
            continue
        for h in hours_list:
            lag_col = f"{col}_lag_{h}h"
            if lag_col not in df.columns:
                continue
            new_col = f"{col}_diff_{h}h"
            df[new_col] = pd.to_numeric(df[col], errors="coerce") - pd.to_numeric(df[lag_col], errors="coerce")
            created.append(new_col)
    return df, created


def add_history_features(df: pd.DataFrame, lag_tolerance_minutes: int) -> Tuple[pd.DataFrame, List[str]]:
    df, lag_cols = add_lag_features(df, LAG_SPECS_HOURS, lag_tolerance_minutes)
    df, rolling_cols = add_rolling_features(df, ROLLING_SPECS_HOURS)
    df, diff_cols = add_diff_features(df, DIFF_SPECS_HOURS)
    return df, lag_cols + rolling_cols + diff_cols


def build_feature_sets(df: pd.DataFrame, feature_mode: str, history_cols: List[str]) -> Dict[str, List[str]]:
    base_features = [c for c in RAW_INPUT_FEATURES + BASE_TIME_FEATURES if c in df.columns]
    history_features = base_features + [c for c in history_cols if c in df.columns]
    for feature_list in [base_features, history_features]:
        forbidden = [c for c in feature_list if c in NON_FEATURE_COLS]
        if forbidden:
            raise ValueError(f"Columnas prohibidas detectadas como features: {forbidden}")
        for c in feature_list:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[~np.isfinite(df[c]), c] = np.nan
    if feature_mode == "base":
        return {"base": base_features}
    if feature_mode == "history":
        return {"history": history_features}
    return {"base": base_features, "history": history_features}


def temporal_split(
    df: pd.DataFrame,
    test_days: int,
    val_days: int,
    reference_end: pd.Timestamp | None = None,
) -> SplitData:
    max_ts = pd.Timestamp(reference_end) if reference_end is not None else df[TIME_COL].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df[TIME_COL] <= val_start].copy()
    val = df[(df[TIME_COL] > val_start) & (df[TIME_COL] <= test_start)].copy()
    test = df[df[TIME_COL] > test_start].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(
            "Split temporal inválido: alguna partición queda vacía. "
            f"train={len(train)}, val={len(val)}, test={len(test)}."
        )
    rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rows.append(
            {
                "split": name,
                "rows": len(part),
                "min_timestamp": part[TIME_COL].min(),
                "max_timestamp": part[TIME_COL].max(),
                "special_rows": int(part.get("special_plant_condition_candidate", pd.Series(False, index=part.index)).sum()),
                "thm_mean": part[TARGET_COL].mean(),
                "thm_std": part[TARGET_COL].std(),
                "thm_min": part[TARGET_COL].min(),
                "thm_p25": part[TARGET_COL].quantile(0.25),
                "thm_median": part[TARGET_COL].median(),
                "thm_p75": part[TARGET_COL].quantile(0.75),
                "thm_max": part[TARGET_COL].max(),
            }
        )
    return SplitData(train=train, val=val, test=test, split_summary=pd.DataFrame(rows))


def make_hgb(random_state: int, max_iter: int, learning_rate: float, l2: float, min_samples_leaf: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2,
        min_samples_leaf=min_samples_leaf,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=random_state,
    )


def build_model_specs(
    feature_sets: Dict[str, List[str]],
    random_state: int,
    hgb_max_iter: int,
    hgb_learning_rate: float,
    hgb_l2: float,
    hgb_min_samples_leaf: int,
) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    base_key = "base" if "base" in feature_sets else next(iter(feature_sets.keys()))
    base_features = feature_sets[base_key]
    specs.append(ModelSpec("dummy_mean", base_key, DummyRegressor(strategy="mean"), base_features, False))
    specs.append(
        ModelSpec(
            "ridge_median_impute",
            base_key,
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                ]
            ),
            base_features,
            False,
        )
    )
    for feature_set, features in feature_sets.items():
        specs.append(
            ModelSpec(
                model_name=f"hgb_{feature_set}",
                feature_set=feature_set,
                model=make_hgb(random_state, hgb_max_iter, hgb_learning_rate, hgb_l2, hgb_min_samples_leaf),
                features=features,
                is_main_candidate=True,
            )
        )
    return specs


def fit_and_evaluate(spec: ModelSpec, split: SplitData, experiment_name: str, variant_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    spec.model.fit(split.train[spec.features], split.train[TARGET_COL])
    metrics_rows = []
    pred_frames = []
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        y = part[TARGET_COL].to_numpy(dtype=float)
        pred = np.asarray(spec.model.predict(part[spec.features]), dtype=float)
        m = compute_metrics(y, pred)
        m.update(
            {
                "experiment": experiment_name,
                "variant": variant_name,
                "model": spec.model_name,
                "feature_set": spec.feature_set,
                "split": split_name,
                "num_features": len(spec.features),
                "main_hgb_candidate": bool(spec.is_main_candidate),
            }
        )
        metrics_rows.append(m)
        pred_df = part[[TIME_COL, TARGET_COL]].copy()
        for meta_col in ["source_file", "quality_flags", "special_flags", "special_plant_condition_candidate", "year", "month"]:
            if meta_col in part.columns:
                pred_df[meta_col] = part[meta_col].values
        pred_df["experiment"] = experiment_name
        pred_df["variant"] = variant_name
        pred_df["model"] = spec.model_name
        pred_df["feature_set"] = spec.feature_set
        pred_df["split"] = split_name
        pred_df["y_true"] = y
        pred_df["y_pred"] = pred
        pred_df["residual_pred_minus_true"] = pred - y
        pred_df["abs_error"] = np.abs(pred - y)
        pred_frames.append(pred_df)
    return pd.DataFrame(metrics_rows), pd.concat(pred_frames, ignore_index=True), spec.model


def summarize_feature_missing_by_split(split: SplitData, feature_sets: Dict[str, List[str]], variant_name: str) -> pd.DataFrame:
    rows = []
    all_features = sorted({c for features in feature_sets.values() for c in features})
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        for c in all_features:
            if c not in part.columns:
                continue
            rows.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    "feature": c,
                    "missing_rows": int(part[c].isna().sum()),
                    "available_rows": int(part[c].notna().sum()),
                    "missing_pct": float(part[c].isna().mean() * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["variant", "split", "missing_pct", "feature"], ascending=[True, True, False, True])


def summarize_distributions_by_split(split: SplitData, columns: Sequence[str], variant_name: str) -> pd.DataFrame:
    rows = []
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        for c in columns:
            if c not in part.columns:
                continue
            s = pd.to_numeric(part[c], errors="coerce")
            rows.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    "column": c,
                    "unit": UNITS_CONFIRMED_BY_PLANT.get(c, ""),
                    "n": int(s.notna().sum()),
                    "missing": int(s.isna().sum()),
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                    "std": float(s.std()) if s.notna().sum() > 1 else np.nan,
                    "min": float(s.min()) if s.notna().any() else np.nan,
                    "p05": float(s.quantile(0.05)) if s.notna().any() else np.nan,
                    "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                    "median": float(s.median()) if s.notna().any() else np.nan,
                    "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
                    "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
                    "max": float(s.max()) if s.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_daily_test_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    test[TIME_COL] = pd.to_datetime(test[TIME_COL])
    test["date"] = test[TIME_COL].dt.date
    rows = []
    for (variant, model, feature_set, date), g in test.groupby(["variant", "model", "feature_set", "date"]):
        y = g["y_true"].to_numpy(dtype=float)
        pred = g["y_pred"].to_numpy(dtype=float)
        m = compute_metrics(y, pred)
        rows.append(
            {
                "variant": variant,
                "model": model,
                "feature_set": feature_set,
                "date": date,
                "n": m["n"],
                "special_rows": int(g.get("special_plant_condition_candidate", pd.Series(False, index=g.index)).sum()),
                "thm_true_mean": float(np.mean(y)),
                "thm_pred_mean": float(np.mean(pred)),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias_pred_minus_true": m["bias_pred_minus_true"],
                "p95_abs_error": m["p95_abs_error"],
            }
        )
    return pd.DataFrame(rows).sort_values(["variant", "model", "feature_set", "date"])


def summarize_errors_by_thm_range(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 2.0, 4.0, 6.0, 8.0, np.inf]
    labels = ["THM < 2", "2 <= THM < 4", "4 <= THM < 6", "6 <= THM < 8", "THM >= 8"]
    df = predictions.copy()
    df["thm_range"] = pd.cut(df["y_true"], bins=bins, labels=labels)
    rows = []
    for (variant, model, feature_set, split_name, thm_range), g in df.groupby(["variant", "model", "feature_set", "split", "thm_range"], observed=True):
        if g.empty:
            continue
        y = g["y_true"].to_numpy(dtype=float)
        pred = g["y_pred"].to_numpy(dtype=float)
        m = compute_metrics(y, pred)
        rows.append(
            {
                "variant": variant,
                "model": model,
                "feature_set": feature_set,
                "split": split_name,
                "thm_range": str(thm_range),
                "n": m["n"],
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias_pred_minus_true": m["bias_pred_minus_true"],
                "p95_abs_error": m["p95_abs_error"],
            }
        )
    return pd.DataFrame(rows)


def plot_test_timeseries(predictions: pd.DataFrame, output_path: Path, title: str) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return
    test[TIME_COL] = pd.to_datetime(test[TIME_COL])
    test = test.sort_values(TIME_COL)
    plt.figure(figsize=(12, 5))
    plt.plot(test[TIME_COL], test["y_true"], label="THM real")
    plt.plot(test[TIME_COL], test["y_pred"], label="THM predicho")
    if "special_plant_condition_candidate" in test.columns and test["special_plant_condition_candidate"].any():
        special = test[test["special_plant_condition_candidate"]]
        plt.scatter(special[TIME_COL], special["y_true"], label="situación especial", marker="x")
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel("THM (µg/L)")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_special_counts(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    base = summary[summary["flag"].isin(["ph_special_low", "bromuro_special_high", "special_plant_condition_candidate"])].copy()
    if base.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.bar(base["flag"], base["rows"])
    plt.title("Filas marcadas como situación especial")
    plt.xlabel("flag")
    plt.ylabel("filas")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def prepare_variant(
    base_df: pd.DataFrame,
    variant_name: str,
    feature_mode: str,
    lag_tolerance_minutes: int,
    test_days: int,
    val_days: int,
    reference_end: pd.Timestamp,
) -> Tuple[pd.DataFrame, Dict[str, List[str]], SplitData, List[str]]:
    df = base_df.copy()
    if variant_name == "exclude_special_conditions":
        df = df[~df["special_plant_condition_candidate"]].copy()
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    df = add_time_features(df)
    df, history_cols = add_history_features(df, lag_tolerance_minutes=lag_tolerance_minutes)
    feature_sets = build_feature_sets(df, feature_mode, history_cols)
    split = temporal_split(
        df,
        test_days=test_days,
        val_days=val_days,
        reference_end=reference_end,
    )
    return df, feature_sets, split, history_cols


def make_variant_list(special_mode: str) -> List[str]:
    if special_mode == "keep":
        return ["keep_special_conditions"]
    if special_mode == "exclude":
        return ["exclude_special_conditions"]
    return ["keep_special_conditions", "exclude_special_conditions"]


def save_variant_outputs(
    output_dir: Path,
    variant_name: str,
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    split: SplitData,
    distribution_report: pd.DataFrame,
    missing_report: pd.DataFrame,
    history_cols: List[str],
) -> pd.DataFrame:
    split_summary = split.split_summary.copy()
    split_summary.insert(0, "variant", variant_name)
    split_summary.to_csv(output_dir / f"split_summary_{variant_name}_v0_2_1.csv", index=False)
    missing_report.to_csv(output_dir / f"feature_missing_by_split_{variant_name}_v0_2_1.csv", index=False)
    distribution_report.to_csv(output_dir / f"distribution_by_split_{variant_name}_v0_2_1.csv", index=False)
    pd.DataFrame(
        [
            {"variant": variant_name, "feature_set": name, "num_features": len(features), "features": ";".join(features)}
            for name, features in feature_sets.items()
        ]
    ).to_csv(output_dir / f"feature_sets_{variant_name}_v0_2_1.csv", index=False)
    pd.DataFrame({"history_cols": history_cols}).to_csv(output_dir / f"history_cols_{variant_name}_v0_2_1.csv", index=False)
    df.to_csv(output_dir / f"dataset_model_input_{variant_name}_v0_2_1.csv", index=False)
    return split_summary


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    raw_df = load_dataset(input_csv, args.year_filter, args.keep_hard_out_of_range)
    raw_df = add_validated_special_flags(
        raw_df,
        ph_special_low_threshold=args.ph_special_low_threshold,
        bromuro_special_high_threshold=args.bromuro_special_high_threshold,
    )

    units_df = pd.DataFrame(
        [{"column": col, "unit": unit} for col, unit in UNITS_CONFIRMED_BY_PLANT.items()]
    )
    units_df.to_csv(output_dir / "confirmed_units_v0_2_1.csv", index=False)

    special_report = build_special_condition_report(raw_df)
    special_report.to_csv(output_dir / "special_condition_rows_v0_2_1.csv", index=False)
    special_summary = summarize_special_flags(raw_df)
    special_summary.to_csv(output_dir / "special_condition_summary_v0_2_1.csv", index=False)
    audit_range_report = build_audit_range_report(
        raw_df,
        ph_special_low_threshold=args.ph_special_low_threshold,
        bromuro_special_high_threshold=args.bromuro_special_high_threshold,
    )
    audit_range_report.to_csv(output_dir / "range_diagnostic_validated_v0_2_1.csv", index=False)
    raw_df.to_csv(output_dir / "dataset_with_special_flags_v0_2_1.csv", index=False)

    if not args.no_plots:
        plot_special_counts(special_summary, output_dir / "plots" / "special_condition_counts_v0_2_1.png")

    all_metrics = []
    all_predictions = []
    all_split_summaries = []
    all_missing_reports = []
    all_distribution_reports = []
    all_feature_sets_summary = []
    all_configs = []

    variant_names = make_variant_list(args.special_mode)
    reference_end = raw_df[TIME_COL].max()

    for variant_name in variant_names:
        df, feature_sets, split, history_cols = prepare_variant(
            raw_df,
            variant_name=variant_name,
            feature_mode=args.feature_mode,
            lag_tolerance_minutes=args.lag_tolerance_minutes,
            test_days=args.test_days,
            val_days=args.val_days,
            reference_end=reference_end,
        )

        distribution_cols = [TARGET_COL] + RAW_INPUT_FEATURES + BASE_TIME_FEATURES
        distribution_report = summarize_distributions_by_split(split, distribution_cols, variant_name)
        missing_report = summarize_feature_missing_by_split(split, feature_sets, variant_name)
        split_summary = save_variant_outputs(output_dir, variant_name, df, feature_sets, split, distribution_report, missing_report, history_cols)
        all_split_summaries.append(split_summary)
        all_missing_reports.append(missing_report)
        all_distribution_reports.append(distribution_report)

        feature_sets_summary = pd.DataFrame(
            [
                {"variant": variant_name, "feature_set": name, "num_features": len(features), "features": ";".join(features)}
                for name, features in feature_sets.items()
            ]
        )
        all_feature_sets_summary.append(feature_sets_summary)

        config = {
            "version": VERSION,
            "variant": variant_name,
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "experiment_name": args.experiment_name,
            "year_filter": args.year_filter,
            "special_mode": args.special_mode,
            "ph_special_low_threshold": args.ph_special_low_threshold,
            "bromuro_special_high_threshold": args.bromuro_special_high_threshold,
            "test_days": args.test_days,
            "val_days": args.val_days,
            "target_col": TARGET_COL,
            "target_unit": UNITS_CONFIRMED_BY_PLANT[TARGET_COL],
            "feature_mode": args.feature_mode,
            "feature_sets": {k: v for k, v in feature_sets.items()},
            "drop_hard_out_of_range": not args.keep_hard_out_of_range,
            "random_state": args.random_state,
            "hgb_max_iter": args.hgb_max_iter,
            "hgb_learning_rate": args.hgb_learning_rate,
            "hgb_l2": args.hgb_l2,
            "hgb_min_samples_leaf": args.hgb_min_samples_leaf,
            "lag_tolerance_minutes": args.lag_tolerance_minutes,
            "split_reference_end": str(reference_end),
            "rows_after_variant_filter": len(df),
            "special_rows_in_raw_dataset": int(raw_df["special_plant_condition_candidate"].sum()),
            "min_timestamp": str(df[TIME_COL].min()),
            "max_timestamp": str(df[TIME_COL].max()),
            "important_note": "THM no se usa como feature. Las situaciones especiales se usan para diagnóstico/exclusión, no como entrada del modelo.",
        }
        all_configs.append(config)

        specs = build_model_specs(
            feature_sets=feature_sets,
            random_state=args.random_state,
            hgb_max_iter=args.hgb_max_iter,
            hgb_learning_rate=args.hgb_learning_rate,
            hgb_l2=args.hgb_l2,
            hgb_min_samples_leaf=args.hgb_min_samples_leaf,
        )

        for spec in specs:
            metrics, predictions, fitted_model = fit_and_evaluate(spec, split, args.experiment_name, variant_name)
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            pred_path = output_dir / "predictions" / f"predictions_{args.experiment_name}_{variant_name}_{spec.model_name}_v0_2_1.csv"
            predictions.to_csv(pred_path, index=False)
            joblib.dump(fitted_model, output_dir / "models" / f"model_{args.experiment_name}_{variant_name}_{spec.model_name}_v0_2_1.joblib")
            if not args.no_plots and spec.model_name.startswith("hgb_"):
                plot_test_timeseries(
                    predictions,
                    output_dir / "plots" / f"test_timeseries_{args.experiment_name}_{variant_name}_{spec.model_name}_v0_2_1.png",
                    title=f"Acosol THM - {args.experiment_name} - {variant_name} - {spec.model_name} - test",
                )

    with open(output_dir / "run_config_v0_2_1.json", "w", encoding="utf-8") as f:
        json.dump(all_configs, f, ensure_ascii=False, indent=2)

    metrics_summary = pd.concat(all_metrics, ignore_index=True)
    metric_cols = [
        "experiment",
        "variant",
        "model",
        "feature_set",
        "split",
        "n",
        "num_features",
        "main_hgb_candidate",
        "mae",
        "rmse",
        "smape_pct",
        "bias_pred_minus_true",
        "median_abs_error",
        "p90_abs_error",
        "p95_abs_error",
        "max_abs_error",
        "r2",
    ]
    metrics_summary = metrics_summary[metric_cols]
    metrics_summary.to_csv(output_dir / "metrics_summary_v0_2_1.csv", index=False)

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    predictions_all.to_csv(output_dir / "predictions_all_models_v0_2_1.csv", index=False)

    daily_test_errors = summarize_daily_test_errors(predictions_all)
    daily_test_errors.to_csv(output_dir / "daily_test_errors_v0_2_1.csv", index=False)

    error_by_thm_range = summarize_errors_by_thm_range(predictions_all)
    error_by_thm_range.to_csv(output_dir / "errors_by_thm_range_v0_2_1.csv", index=False)

    split_summary_all = pd.concat(all_split_summaries, ignore_index=True)
    split_summary_all.to_csv(output_dir / "split_summary_v0_2_1.csv", index=False)
    missing_report_all = pd.concat(all_missing_reports, ignore_index=True)
    missing_report_all.to_csv(output_dir / "feature_missing_by_split_v0_2_1.csv", index=False)
    distribution_report_all = pd.concat(all_distribution_reports, ignore_index=True)
    distribution_report_all.to_csv(output_dir / "distribution_by_split_v0_2_1.csv", index=False)
    feature_sets_summary_all = pd.concat(all_feature_sets_summary, ignore_index=True)
    feature_sets_summary_all.to_csv(output_dir / "feature_sets_v0_2_1.csv", index=False)

    xlsx_path = output_dir / "acosol_model_report_v0_2_1.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        units_df.to_excel(writer, sheet_name="confirmed_units", index=False)
        pd.DataFrame(all_configs).to_excel(writer, sheet_name="config", index=False)
        special_summary.to_excel(writer, sheet_name="special_summary", index=False)
        special_report.to_excel(writer, sheet_name="special_rows", index=False)
        audit_range_report.to_excel(writer, sheet_name="range_diagnostic", index=False)
        feature_sets_summary_all.to_excel(writer, sheet_name="feature_sets", index=False)
        split_summary_all.to_excel(writer, sheet_name="split_summary", index=False)
        distribution_report_all.to_excel(writer, sheet_name="distribution_split", index=False)
        missing_report_all.to_excel(writer, sheet_name="feature_missing", index=False)
        metrics_summary.to_excel(writer, sheet_name="metrics", index=False)
        daily_test_errors.to_excel(writer, sheet_name="daily_test_errors", index=False)
        error_by_thm_range.to_excel(writer, sheet_name="errors_by_thm_range", index=False)
        predictions_all[predictions_all["split"] == "test"].to_excel(writer, sheet_name="test_predictions", index=False)

    hgb_val_test = metrics_summary[metrics_summary["model"].str.startswith("hgb_") & metrics_summary["split"].isin(["val", "test"])].copy()

    print("\nEntrenamiento Acosol THM v0.2.1 completado")
    print("==========================================")
    print(f"Entrada:        {input_csv}")
    print(f"Salida:         {output_dir}")
    print(f"Experimento:    {args.experiment_name}")
    print(f"Filtro años:    {args.year_filter}")
    print(f"Feature mode:   {args.feature_mode}")
    print(f"Special mode:   {args.special_mode}")
    print("Familia principal: HGB")
    print("Unidades confirmadas por planta:")
    for col, unit in UNITS_CONFIRMED_BY_PLANT.items():
        print(f"- {col}: {unit}")
    print(f"\nFilas base tras filtros iniciales: {len(raw_df)}")
    print(f"Filas marcadas como situación especial: {int(raw_df['special_plant_condition_candidate'].sum())}")
    print(f"- pH < {args.ph_special_low_threshold}: {int(raw_df['ph_special_low'].sum())}")
    print(f"- bromuro > {args.bromuro_special_high_threshold} mg/L: {int(raw_df['bromuro_special_high'].sum())}")

    print("\nSplit temporal por variante:")
    print(split_summary_all.to_string(index=False))

    print("\nComparativa HGB val/test:")
    if not hgb_val_test.empty:
        print(hgb_val_test[["variant", "model", "feature_set", "split", "n", "num_features", "mae", "rmse", "bias_pred_minus_true", "r2"]].to_string(index=False))

    print("\nMétricas completas:")
    print(metrics_summary.to_string(index=False))
    print(f"\nReporte Excel: {xlsx_path}")
    print("\nFicheros clave:")
    print(f"- {output_dir / 'special_condition_rows_v0_2_1.csv'}")
    print(f"- {output_dir / 'metrics_summary_v0_2_1.csv'}")
    print(f"- {output_dir / 'errors_by_thm_range_v0_2_1.csv'}")


if __name__ == "__main__":
    main()
