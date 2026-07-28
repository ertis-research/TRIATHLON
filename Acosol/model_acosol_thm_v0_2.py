from __future__ import annotations

"""Entrenamiento y comparación de modelos HGB para el soft-sensor THM de Acosol."""

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Evita bloqueos/rendimiento pobre por sobre-paralelización OpenMP/BLAS en algunos entornos.
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


VERSION = "v0.2"
TARGET_COL = "thm"
TIME_COL = "timestamp"
THM_UNIT = "µg/L"

UNITS_CONFIRMED_BY_PLANT = {
    "turbidez": "NTU",
    "cloro": "mg/L",
    "ph": "unidades de pH",
    "conductividad": "µS/cm a 20 ºC",
    "thm": THM_UNIT,
    "bromuro": "mg/L",
    "cot": "mg/L",
}

PH_SPECIAL_LOW_THRESHOLD = 5.0
BROMURO_SPECIAL_HIGH_THRESHOLD = 10.0

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

# Lags/rolling propuestos para Acosol v0.2. Solo usan variables de entrada, nunca THM.
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
}

# Rangos orientativos de auditoría. No filtran datos automáticamente. Los valores bajos
# de pH y altos de bromuro se marcan como posibles situaciones especiales confirmadas
# por planta, no como errores de decimal.
PLAUSIBLE_RANGES = {
    "turbidez": (0.0, 50.0),
    "cloro": (0.0, 5.0),
    "ph": (5.0, 10.0),
    "conductividad": (50.0, 2500.0),
    "bromuro": (0.0, 10.0),
    "cot": (0.0, 30.0),
    "thm": (0.0, 200.0),
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
    parser = argparse.ArgumentParser(description="Entrenamiento Acosol THM v0.2")
    parser.add_argument(
        "--input-csv",
        required=True,
        help="CSV generado por load_acosol_excels_v0_1.py, normalmente Dataset A.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_outputs_acosol_thm_v0_2",
        help="Carpeta de salida.",
    )
    parser.add_argument(
        "--experiment-name",
        default="A_2025_2026_v0_2",
        help="Nombre corto del experimento para nombrar salidas.",
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
        default="compare",
        help=(
            "base: solo variables actuales + temporales; "
            "history: añade lags/rolling/diffs; "
            "compare: entrena HGB base e HGB history para comparar."
        ),
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
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para modelos reproducibles.",
    )
    parser.add_argument(
        "--hgb-max-iter",
        type=int,
        default=700,
        help="Iteraciones máximas de HistGradientBoostingRegressor.",
    )
    parser.add_argument(
        "--hgb-learning-rate",
        type=float,
        default=0.04,
        help="Learning rate de HistGradientBoostingRegressor.",
    )
    parser.add_argument(
        "--hgb-l2",
        type=float,
        default=0.05,
        help="Regularización L2 de HistGradientBoostingRegressor.",
    )
    parser.add_argument(
        "--hgb-min-samples-leaf",
        type=int,
        default=20,
        help="min_samples_leaf de HistGradientBoostingRegressor.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="No generar gráficas PNG.",
    )
    args = parser.parse_args()
    if args.test_days <= 0 or args.val_days <= 0:
        parser.error("--test-days y --val-days deben ser mayores que cero.")
    if args.lag_tolerance_minutes < 0:
        parser.error("--lag-tolerance-minutes no puede ser negativo.")
    if args.hgb_max_iter <= 0 or args.hgb_learning_rate <= 0 or args.hgb_min_samples_leaf <= 0:
        parser.error("Los parámetros principales de HGB deben ser mayores que cero.")
    if args.hgb_l2 < 0:
        parser.error("--hgb-l2 no puede ser negativo.")
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


def add_lag_features(
    df: pd.DataFrame,
    lag_specs: Dict[str, List[int]],
    tolerance_minutes: int,
) -> Tuple[pd.DataFrame, List[str]]:
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


def add_rolling_features(
    df: pd.DataFrame,
    rolling_specs: Dict[str, List[int]],
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy().sort_values(TIME_COL).reset_index(drop=True)
    created: List[str] = []
    ts_indexed = df.set_index(TIME_COL, drop=False).sort_index()

    for col, hours_list in rolling_specs.items():
        if col not in ts_indexed.columns:
            continue
        series = pd.to_numeric(ts_indexed[col], errors="coerce")
        for h in hours_list:
            new_col = f"{col}_roll_mean_{h}h"
            rolled = series.rolling(f"{h}h", closed="left", min_periods=1).mean()
            ts_indexed[new_col] = rolled
            created.append(new_col)

    out = ts_indexed.reset_index(drop=True).sort_values(TIME_COL).reset_index(drop=True)
    return out, created


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


def temporal_split(df: pd.DataFrame, test_days: int, val_days: int) -> SplitData:
    max_ts = df[TIME_COL].max()
    test_start = max_ts - pd.Timedelta(days=test_days)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df[TIME_COL] <= val_start].copy()
    val = df[(df[TIME_COL] > val_start) & (df[TIME_COL] <= test_start)].copy()
    test = df[df[TIME_COL] > test_start].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            "Split temporal inválido: alguna partición queda vacía. "
            f"train={len(train)}, val={len(val)}, test={len(test)}. "
            "Prueba a reducir --test-days/--val-days."
        )

    summary_rows = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        summary_rows.append(
            {
                "split": name,
                "rows": len(part),
                "min_timestamp": part[TIME_COL].min(),
                "max_timestamp": part[TIME_COL].max(),
                "thm_mean": part[TARGET_COL].mean(),
                "thm_std": part[TARGET_COL].std(),
                "thm_min": part[TARGET_COL].min(),
                "thm_p25": part[TARGET_COL].quantile(0.25),
                "thm_median": part[TARGET_COL].median(),
                "thm_p75": part[TARGET_COL].quantile(0.75),
                "thm_max": part[TARGET_COL].max(),
            }
        )
    split_summary = pd.DataFrame(summary_rows)
    return SplitData(train=train, val=val, test=test, split_summary=split_summary)


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

    specs.append(
        ModelSpec(
            model_name="dummy_mean",
            feature_set=base_key,
            model=DummyRegressor(strategy="mean"),
            features=base_features,
            is_main_candidate=False,
        )
    )
    specs.append(
        ModelSpec(
            model_name="ridge_median_impute",
            feature_set=base_key,
            model=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                ]
            ),
            features=base_features,
            is_main_candidate=False,
        )
    )

    for feature_set, features in feature_sets.items():
        specs.append(
            ModelSpec(
                model_name=f"hgb_{feature_set}",
                feature_set=feature_set,
                model=make_hgb(
                    random_state=random_state,
                    max_iter=hgb_max_iter,
                    learning_rate=hgb_learning_rate,
                    l2=hgb_l2,
                    min_samples_leaf=hgb_min_samples_leaf,
                ),
                features=features,
                is_main_candidate=True,
            )
        )
    return specs


def fit_and_evaluate(spec: ModelSpec, split: SplitData, experiment_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    X_train = split.train[spec.features]
    y_train = split.train[TARGET_COL]

    spec.model.fit(X_train, y_train)

    metrics_rows = []
    pred_frames = []
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        X = part[spec.features]
        y = part[TARGET_COL].to_numpy(dtype=float)
        pred = np.asarray(spec.model.predict(X), dtype=float)

        m = compute_metrics(y, pred)
        m.update(
            {
                "experiment": experiment_name,
                "model": spec.model_name,
                "feature_set": spec.feature_set,
                "split": split_name,
                "num_features": len(spec.features),
                "main_hgb_candidate": bool(spec.is_main_candidate),
            }
        )
        metrics_rows.append(m)

        pred_df = part[[TIME_COL, TARGET_COL]].copy()
        for meta_col in ["source_file", "quality_flags", "year", "month"]:
            if meta_col in part.columns:
                pred_df[meta_col] = part[meta_col].values
        pred_df["experiment"] = experiment_name
        pred_df["model"] = spec.model_name
        pred_df["feature_set"] = spec.feature_set
        pred_df["split"] = split_name
        pred_df["y_true"] = y
        pred_df["y_pred"] = pred
        pred_df["residual_pred_minus_true"] = pred - y
        pred_df["abs_error"] = np.abs(pred - y)
        pred_frames.append(pred_df)

    return pd.DataFrame(metrics_rows), pd.concat(pred_frames, ignore_index=True), spec.model


def summarize_feature_missing_by_split(split: SplitData, feature_sets: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    all_features = sorted({c for features in feature_sets.values() for c in features})
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        for c in all_features:
            if c not in part.columns:
                continue
            rows.append(
                {
                    "split": split_name,
                    "feature": c,
                    "missing_rows": int(part[c].isna().sum()),
                    "available_rows": int(part[c].notna().sum()),
                    "missing_pct": float(part[c].isna().mean() * 100.0),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "missing_pct", "feature"], ascending=[True, False, True])


def summarize_distributions_by_split(split: SplitData, columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        for c in columns:
            if c not in part.columns:
                continue
            s = pd.to_numeric(part[c], errors="coerce")
            rows.append(
                {
                    "split": split_name,
                    "column": c,
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


def build_range_diagnostic_report(df: pd.DataFrame, split: SplitData) -> pd.DataFrame:
    split_lookup = []
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        split_lookup.append(part[[TIME_COL]].assign(split=split_name))
    split_df = pd.concat(split_lookup, ignore_index=True).drop_duplicates(TIME_COL)
    tmp = df.merge(split_df, on=TIME_COL, how="left")

    rows = []
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col not in tmp.columns:
            continue
        s = pd.to_numeric(tmp[col], errors="coerce")
        mask = s.notna() & ((s < lo) | (s > hi))
        for _, row in tmp.loc[mask].iterrows():
            value = float(row[col])
            diagnostic = "review"
            if col == "ph" and value < PH_SPECIAL_LOW_THRESHOLD:
                diagnostic = "plant_confirmed_special_context_candidate"
            elif col == "bromuro" and value > BROMURO_SPECIAL_HIGH_THRESHOLD:
                diagnostic = "plant_confirmed_special_context_candidate"
            elif value > hi and value / 10.0 <= hi:
                diagnostic = "possible_decimal_shift_x10"
            elif value > hi and value / 100.0 <= hi:
                diagnostic = "possible_decimal_shift_x100"
            elif value > hi and value / 1000.0 <= hi:
                diagnostic = "possible_decimal_shift_x1000"
            rows.append(
                {
                    "timestamp": row[TIME_COL],
                    "split": row.get("split", "unknown"),
                    "column": col,
                    "value": value,
                    "unit": UNITS_CONFIRMED_BY_PLANT.get(col, ""),
                    "plausible_min": lo,
                    "plausible_max": hi,
                    "diagnostic": diagnostic,
                    "source_file": row.get("source_file", ""),
                    "quality_flags": row.get("quality_flags", ""),
                }
            )
    return pd.DataFrame(rows).sort_values(["timestamp", "column"]) if rows else pd.DataFrame(
        columns=[
            "timestamp",
            "split",
            "column",
            "value",
            "unit",
            "plausible_min",
            "plausible_max",
            "diagnostic",
            "source_file",
            "quality_flags",
        ]
    )


def summarize_source_by_split(split: SplitData) -> pd.DataFrame:
    rows = []
    for split_name, part in [("train", split.train), ("val", split.val), ("test", split.test)]:
        if "source_file" not in part.columns:
            continue
        counts = part["source_file"].fillna("unknown").value_counts(dropna=False)
        for source, n in counts.items():
            rows.append({"split": split_name, "source_file": source, "rows": int(n)})
    return pd.DataFrame(rows)


def summarize_daily_test_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()
    test[TIME_COL] = pd.to_datetime(test[TIME_COL])
    test["date"] = test[TIME_COL].dt.date
    rows = []
    for (model, feature_set, date), g in test.groupby(["model", "feature_set", "date"]):
        y = g["y_true"].to_numpy(dtype=float)
        pred = g["y_pred"].to_numpy(dtype=float)
        m = compute_metrics(y, pred)
        rows.append(
            {
                "model": model,
                "feature_set": feature_set,
                "date": date,
                "n": m["n"],
                "thm_true_mean": float(np.mean(y)),
                "thm_pred_mean": float(np.mean(pred)),
                "mae": m["mae"],
                "rmse": m["rmse"],
                "bias_pred_minus_true": m["bias_pred_minus_true"],
                "p95_abs_error": m["p95_abs_error"],
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "feature_set", "date"])


def get_primary_hgb_name(feature_sets: Dict[str, List[str]]) -> str:
    # En modo compare mantenemos hgb_base como referencia principal estable y
    # evaluamos hgb_history como candidato enriquecido.
    if "base" in feature_sets:
        return "hgb_base"
    if "history" in feature_sets:
        return "hgb_history"
    return f"hgb_{next(iter(feature_sets.keys()))}"


def plot_test_timeseries(predictions: pd.DataFrame, output_path: Path, title: str) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return
    test[TIME_COL] = pd.to_datetime(test[TIME_COL])
    test = test.sort_values(TIME_COL)

    plt.figure(figsize=(12, 5))
    plt.plot(test[TIME_COL], test["y_true"], label="THM real")
    plt.plot(test[TIME_COL], test["y_pred"], label="THM predicho")
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel(f"THM ({THM_UNIT})")
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_test_scatter(predictions: pd.DataFrame, output_path: Path, title: str) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return
    y_true = test["y_true"].to_numpy(dtype=float)
    y_pred = test["y_pred"].to_numpy(dtype=float)
    lo = float(np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)]))
    hi = float(np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)]))

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([lo, hi], [lo, hi], linestyle="--", label="ideal")
    plt.title(title)
    plt.xlabel(f"THM real ({THM_UNIT})")
    plt.ylabel(f"THM predicho ({THM_UNIT})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_histogram(predictions: pd.DataFrame, output_path: Path, title: str) -> None:
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        return
    residuals = test["residual_pred_minus_true"].to_numpy(dtype=float)
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=25)
    plt.axvline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("predicho - real")
    plt.ylabel("frecuencia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_target_split_distribution(split_summary: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(split_summary["split"], split_summary["thm_mean"], marker="o", label="media")
    plt.plot(split_summary["split"], split_summary["thm_median"], marker="o", label="mediana")
    plt.title("Distribución de THM por split")
    plt.xlabel("split")
    plt.ylabel(f"THM ({THM_UNIT})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_plots(predictions_by_model: Dict[str, pd.DataFrame], output_dir: Path, experiment_name: str, main_model_name: str) -> None:
    for model_name, predictions in predictions_by_model.items():
        if model_name not in {"hgb_base", "hgb_history", main_model_name}:
            continue
        plot_test_timeseries(
            predictions,
            output_dir / "plots" / f"test_timeseries_{experiment_name}_{model_name}_v0_2.png",
            title=f"Acosol THM - {experiment_name} - {model_name} - test",
        )
        plot_test_scatter(
            predictions,
            output_dir / "plots" / f"test_scatter_{experiment_name}_{model_name}_v0_2.png",
            title=f"Acosol THM - {experiment_name} - {model_name} - test",
        )
        plot_residual_histogram(
            predictions,
            output_dir / "plots" / f"test_residuals_{experiment_name}_{model_name}_v0_2.png",
            title=f"Errores test - {experiment_name} - {model_name}",
        )


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "predictions").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)

    df = load_dataset(
        input_csv=input_csv,
        year_filter=args.year_filter,
        keep_hard_out_of_range=args.keep_hard_out_of_range,
    )
    df = add_time_features(df)
    df, history_cols = add_history_features(df, lag_tolerance_minutes=args.lag_tolerance_minutes)
    feature_sets = build_feature_sets(df, args.feature_mode, history_cols)
    split = temporal_split(df, test_days=args.test_days, val_days=args.val_days)

    split.split_summary.to_csv(output_dir / "split_summary_v0_2.csv", index=False)
    missing_report = summarize_feature_missing_by_split(split, feature_sets)
    missing_report.to_csv(output_dir / "feature_missing_by_split_v0_2.csv", index=False)

    distribution_cols = [TARGET_COL] + RAW_INPUT_FEATURES + BASE_TIME_FEATURES
    distribution_report = summarize_distributions_by_split(split, distribution_cols)
    distribution_report.to_csv(output_dir / "distribution_by_split_v0_2.csv", index=False)

    range_diagnostic_report = build_range_diagnostic_report(df, split)
    range_diagnostic_report.to_csv(output_dir / "range_decimal_diagnostic_v0_2.csv", index=False)

    source_by_split = summarize_source_by_split(split)
    source_by_split.to_csv(output_dir / "source_by_split_v0_2.csv", index=False)

    feature_sets_summary = pd.DataFrame(
        [
            {"feature_set": name, "num_features": len(features), "features": ";".join(features)}
            for name, features in feature_sets.items()
        ]
    )
    feature_sets_summary.to_csv(output_dir / "feature_sets_v0_2.csv", index=False)

    config = {
        "version": VERSION,
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "experiment_name": args.experiment_name,
        "year_filter": args.year_filter,
        "test_days": args.test_days,
        "val_days": args.val_days,
        "target_col": TARGET_COL,
        "target_unit": THM_UNIT,
        "units_confirmed_by_plant": UNITS_CONFIRMED_BY_PLANT,
        "time_col": TIME_COL,
        "feature_mode": args.feature_mode,
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "history_cols": history_cols,
        "drop_hard_out_of_range": not args.keep_hard_out_of_range,
        "random_state": args.random_state,
        "hgb_max_iter": args.hgb_max_iter,
        "hgb_learning_rate": args.hgb_learning_rate,
        "hgb_l2": args.hgb_l2,
        "hgb_min_samples_leaf": args.hgb_min_samples_leaf,
        "lag_tolerance_minutes": args.lag_tolerance_minutes,
        "rows_after_filters": len(df),
        "min_timestamp": str(df[TIME_COL].min()),
        "max_timestamp": str(df[TIME_COL].max()),
        "important_note": "THM no se usa como feature. Los lags/rolling son solo de variables de entrada.",
    }
    with open(output_dir / "run_config_v0_2.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    specs = build_model_specs(
        feature_sets=feature_sets,
        random_state=args.random_state,
        hgb_max_iter=args.hgb_max_iter,
        hgb_learning_rate=args.hgb_learning_rate,
        hgb_l2=args.hgb_l2,
        hgb_min_samples_leaf=args.hgb_min_samples_leaf,
    )

    all_metrics = []
    all_predictions = []
    predictions_by_model: Dict[str, pd.DataFrame] = {}
    primary_hgb_name = get_primary_hgb_name(feature_sets)

    for spec in specs:
        metrics, predictions, fitted_model = fit_and_evaluate(spec, split, args.experiment_name)
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        predictions_by_model[spec.model_name] = predictions

        pred_path = output_dir / "predictions" / f"predictions_{args.experiment_name}_{spec.model_name}_v0_2.csv"
        predictions.to_csv(pred_path, index=False)
        joblib.dump(fitted_model, output_dir / "models" / f"model_{args.experiment_name}_{spec.model_name}_v0_2.joblib")

    metrics_summary = pd.concat(all_metrics, ignore_index=True)
    metric_cols = [
        "experiment",
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
    metrics_summary.to_csv(output_dir / "metrics_summary_v0_2.csv", index=False)

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    predictions_all.to_csv(output_dir / "predictions_all_models_v0_2.csv", index=False)

    daily_test_errors = summarize_daily_test_errors(predictions_all)
    daily_test_errors.to_csv(output_dir / "daily_test_errors_v0_2.csv", index=False)

    if not args.no_plots:
        plot_target_split_distribution(split.split_summary, output_dir / "plots" / "target_distribution_by_split_v0_2.png")
        save_plots(predictions_by_model, output_dir, args.experiment_name, primary_hgb_name)

    xlsx_path = output_dir / "acosol_model_report_v0_2.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
        pd.DataFrame([config]).to_excel(writer, sheet_name="config", index=False)
        feature_sets_summary.to_excel(writer, sheet_name="feature_sets", index=False)
        split.split_summary.to_excel(writer, sheet_name="split_summary", index=False)
        distribution_report.to_excel(writer, sheet_name="distribution_split", index=False)
        missing_report.to_excel(writer, sheet_name="feature_missing", index=False)
        range_diagnostic_report.to_excel(writer, sheet_name="range_diagnostic", index=False)
        source_by_split.to_excel(writer, sheet_name="source_split", index=False)
        metrics_summary.to_excel(writer, sheet_name="metrics", index=False)
        daily_test_errors.to_excel(writer, sheet_name="daily_test_errors", index=False)
        predictions_all[predictions_all["split"] == "test"].to_excel(writer, sheet_name="test_predictions", index=False)

    primary_test = metrics_summary[(metrics_summary["model"] == primary_hgb_name) & (metrics_summary["split"] == "test")]
    primary_val = metrics_summary[(metrics_summary["model"] == primary_hgb_name) & (metrics_summary["split"] == "val")]
    hgb_val_test = metrics_summary[metrics_summary["model"].str.startswith("hgb_") & metrics_summary["split"].isin(["val", "test"])].copy()

    print("\nEntrenamiento Acosol THM v0.2 completado")
    print("========================================")
    print(f"Entrada:        {input_csv}")
    print(f"Salida:         {output_dir}")
    print(f"Experimento:    {args.experiment_name}")
    print(f"Filtro años:    {args.year_filter}")
    print(f"Feature mode:   {args.feature_mode}")
    print("Familia principal: HGB")
    print(f"HGB de referencia estable: {primary_hgb_name}")
    print(f"Filas usadas:   {len(df)}")
    print(f"Rango temporal: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")

    print("\nFeature sets:")
    for name, features in feature_sets.items():
        print(f"- {name}: {len(features)} features")

    print("\nSplit temporal:")
    print(split.split_summary.to_string(index=False))

    if not primary_val.empty:
        r = primary_val.iloc[0]
        print("\nMétrica HGB de referencia en validación:")
        print(f"- MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} bias={r['bias_pred_minus_true']:.4f} R2={r['r2']:.4f}")
    if not primary_test.empty:
        r = primary_test.iloc[0]
        print("\nMétrica HGB de referencia en test:")
        print(f"- MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} bias={r['bias_pred_minus_true']:.4f} R2={r['r2']:.4f}")

    if not hgb_val_test.empty:
        print("\nComparativa HGB val/test:")
        print(hgb_val_test[["model", "feature_set", "split", "n", "num_features", "mae", "rmse", "bias_pred_minus_true", "r2"]].to_string(index=False))

    print("\nMétricas completas:")
    print(metrics_summary.to_string(index=False))
    print(f"\nReporte Excel: {xlsx_path}")

    if len(range_diagnostic_report) > 0:
        print(f"\nAviso: range_decimal_diagnostic_v0_2.csv contiene {len(range_diagnostic_report)} valores a revisar.")
        print("No se han corregido automáticamente; incluye auditoría de rango y situaciones especiales.")


if __name__ == "__main__":
    main()
