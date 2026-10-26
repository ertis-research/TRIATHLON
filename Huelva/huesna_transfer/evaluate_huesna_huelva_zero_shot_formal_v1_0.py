from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
HUELVA_ROOT = SCRIPT_DIR.parent
SFTPS_ROOT = HUELVA_ROOT.parent
HUESNA_ROOT = (SFTPS_ROOT / "Huesna").resolve()

HUELVA_CLEAN = (
    HUELVA_ROOT
    / "huelva_par_etl_outputs_v0_1"
    / "huelva_par_timeseries_clean_v0_1.csv"
)
HUELVA_THDATA = (
    HUELVA_ROOT
    / "huelva_par_etl_outputs_v0_1"
    / "huelva_thm_par_feature_dataset_v0_1.csv"
)

HUESNA_SCRIPT = HUESNA_ROOT / "model_v5_8_12_incident_intervals.py"
HUESNA_OUTPUT = (
    HUESNA_ROOT
    / "model_outputs_thm_softsensor_v5_8_12_incidents_20260921_135541"
)
HUESNA_BUNDLE = HUESNA_OUTPUT / "instant_softsensor_bundle.pkl"

OUTPUT_DIR = SCRIPT_DIR / "outputs" / "formal_v1_0"

RUN_ID = "v5_8_12_20260921"
LABEL = "Huesna v5.8.12 formal 2026-09-21"

HUESNA_TO_HUELVA = {
    "PH": "ph",
    "UVAS": "uv254_abs_m",
    "conductividad": "conductivity_us_cm",
    "turbidez": "turbidity_ntu",
    "Cloro": "free_chlorine_mg_l",
    "Temperatura": "temperature_c",
}

REGULARIZATION_FREQ = "1min"
MAX_GAP_MINUTES = 5
REGULARIZATION_FFILL_LIMIT = 5
FALLBACK_ALIGNMENT_MINUTES = 5


def load_module(path: Path, module_name: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"No existe el script Huesna: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el modulo: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-7)
    return float(
        100.0
        * np.mean(2.0 * np.abs(y_pred - y_true) / denom)
    )


def pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan,
        "pearson_r": pearson_safe(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def load_huelva_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean = pd.read_csv(HUELVA_CLEAN, encoding="utf-8-sig")
    targets = pd.read_csv(HUELVA_THDATA, encoding="utf-8-sig")

    required_clean = {"timestamp", *HUESNA_TO_HUELVA.values()}
    missing_clean = required_clean - set(clean.columns)
    if missing_clean:
        raise ValueError(
            f"Faltan columnas en Huelva clean: {sorted(missing_clean)}"
        )

    required_targets = {"sample_timestamp", "thm"}
    missing_targets = required_targets - set(targets.columns)
    if missing_targets:
        raise ValueError(
            f"Faltan columnas THM Huelva: {sorted(missing_targets)}"
        )

    clean["timestamp"] = pd.to_datetime(
        clean["timestamp"],
        errors="coerce",
    )
    clean = clean[clean["timestamp"].notna()].copy()
    clean = clean.sort_values("timestamp")

    targets["sample_timestamp"] = pd.to_datetime(
        targets["sample_timestamp"],
        errors="coerce",
    )
    targets["thm"] = pd.to_numeric(
        targets["thm"],
        errors="coerce",
    )
    targets = targets[
        targets["sample_timestamp"].notna()
        & targets["thm"].notna()
    ].copy()
    targets = targets.sort_values("sample_timestamp").reset_index(drop=True)

    if "sample_id" not in targets.columns:
        targets["sample_id"] = [
            f"Huelva_THM_{i + 1:03d}"
            for i in range(len(targets))
        ]
    if "sample_point" not in targets.columns:
        targets["sample_point"] = ""
    if "thm_unit" not in targets.columns:
        targets["thm_unit"] = "ug/L"

    return clean, targets


def build_huelva_wide(clean: pd.DataFrame) -> pd.DataFrame:
    selected = pd.DataFrame({"timestamp": clean["timestamp"]})

    for huesna_name, huelva_name in HUESNA_TO_HUELVA.items():
        selected[huesna_name] = pd.to_numeric(
            clean[huelva_name],
            errors="coerce",
        )

    wide = (
        selected.groupby("timestamp", as_index=True)
        .mean(numeric_only=True)
        .sort_index()
    )
    wide.index = pd.DatetimeIndex(wide.index)
    wide.index.name = "timestamp"
    return wide


def call_build_feature_table(
    module: Any,
    wide_for_features: pd.DataFrame,
    current_cols: List[str],
    bundle: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    kwargs: Dict[str, Any] = {
        "wide_reg": wide_for_features,
        "feature_base_cols": current_cols,
        "use_history_multi": bool(
            bundle.get("uses_history_multi", True)
        ),
        "max_history_minutes": int(
            bundle.get("buffer_multi_required_minutes", 240)
        ),
        "change_lookback": 30,
        "calendar_mode": str(
            bundle.get("calendar_mode", "all")
        ),
        "clean_feature_outliers": bool(
            bundle.get("feature_cleaning_enabled", True)
        ),
        "feature_clean_ffill_limit": int(
            dict(
                bundle.get("feature_cleaning_meta", {}) or {}
            ).get("ffill_limit", 10)
        ),
        "min_valid_target": 0.1,
        "max_valid_target": None,
    }

    signature = inspect.signature(module.build_feature_table)

    if "clean_extreme_target_glitches" in signature.parameters:
        kwargs["clean_extreme_target_glitches"] = False

    if "confirmed_feature_intervals" in signature.parameters:
        # Las incidencias confirmadas de Huesna no se trasladan
        # temporalmente a Huelva.
        kwargs["confirmed_feature_intervals"] = []

    return module.build_feature_table(**kwargs)


def build_features(
    module: Any,
    wide: pd.DataFrame,
    bundle: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    current_cols = list(bundle["current_cols"])
    expected_current = list(HUESNA_TO_HUELVA.keys())

    if current_cols != expected_current:
        raise ValueError(
            "Las variables base del bundle no coinciden con el mapping Huelva.\n"
            f"Bundle: {current_cols}\n"
            f"Esperado: {expected_current}"
        )

    wide_reg, n_segments = module.regularize_by_segments(
        wide=wide,
        freq=REGULARIZATION_FREQ,
        max_gap_minutes=MAX_GAP_MINUTES,
        ffill_limit=REGULARIZATION_FFILL_LIMIT,
    )

    wide_for_features = wide_reg.copy()
    wide_for_features[module.TARGET_COL] = 1.0

    built, built_current_cols, built_feature_cols = call_build_feature_table(
        module=module,
        wide_for_features=wide_for_features,
        current_cols=current_cols,
        bundle=bundle,
    )

    expected_features = list(bundle["selected_feature_cols"])

    if built_current_cols != current_cols:
        raise ValueError(
            f"Current cols no coinciden: "
            f"{built_current_cols} != {current_cols}"
        )

    if built_feature_cols != expected_features:
        missing = [
            c
            for c in expected_features
            if c not in built_feature_cols
        ]
        extra = [
            c
            for c in built_feature_cols
            if c not in expected_features
        ]
        raise ValueError(
            "El feature engineering no reproduce el contrato del bundle.\n"
            f"Esperadas: {len(expected_features)}\n"
            f"Generadas: {len(built_feature_cols)}\n"
            f"Missing: {missing[:20]}\n"
            f"Extra: {extra[:20]}"
        )

    x = built[expected_features].copy()
    current_x = built[current_cols].copy()

    meta = {
        "n_segments": int(n_segments),
        "regularization_freq": REGULARIZATION_FREQ,
        "max_gap_minutes": MAX_GAP_MINUTES,
        "regularization_ffill_limit": REGULARIZATION_FFILL_LIMIT,
        "calendar_mode": str(bundle.get("calendar_mode", "all")),
        "feature_count": len(expected_features),
        "feature_cleaning_enabled": bool(
            bundle.get("feature_cleaning_enabled", True)
        ),
        "history_minutes": int(
            bundle.get("buffer_multi_required_minutes", 240)
        ),
    }

    return x, current_x, meta


def choose_feature_timestamp(
    sample_ts: pd.Timestamp,
    feature_index: pd.DatetimeIndex,
) -> Tuple[pd.Timestamp | None, float | None, str]:
    if sample_ts in feature_index:
        return sample_ts, 0.0, "exact"

    previous = feature_index[feature_index <= sample_ts]
    if len(previous) == 0:
        return None, None, "missing"

    chosen = previous[-1]
    delay = float(
        (sample_ts - chosen).total_seconds() / 60.0
    )

    if delay <= FALLBACK_ALIGNMENT_MINUTES:
        return chosen, delay, "previous_causal"

    return None, delay, "missing"


def predict_hgb(
    x: pd.DataFrame,
    current_x: pd.DataFrame,
    targets: pd.DataFrame,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    feature_index = pd.DatetimeIndex(x.index)
    model = bundle.get("selected_model")

    if model is None:
        raise ValueError(
            "El bundle estable de Huesna no contiene selected_model."
        )

    rows: List[Dict[str, Any]] = []

    for _, target_row in targets.iterrows():
        sample_ts = pd.Timestamp(
            target_row["sample_timestamp"]
        )
        y_true = float(target_row["thm"])

        feature_ts, delay, alignment = choose_feature_timestamp(
            sample_ts=sample_ts,
            feature_index=feature_index,
        )

        row: Dict[str, Any] = {
            "sample_id": target_row.get("sample_id", ""),
            "sample_timestamp": sample_ts,
            "sample_point": target_row.get("sample_point", ""),
            "thm_unit": target_row.get("thm_unit", ""),
            "thm_real": y_true,
            "feature_timestamp": feature_ts,
            "alignment_mode": alignment,
            "alignment_delay_minutes": delay,
            "feature_nan_count": np.nan,
            "huesna_v5_8_12_hgb_pred": np.nan,
            "huesna_v5_8_12_hgb_error": np.nan,
            "huesna_v5_8_12_hgb_abs_error": np.nan,
        }

        if feature_ts is None:
            rows.append(row)
            continue

        x_row = x.loc[[feature_ts]].copy()
        row["feature_nan_count"] = int(
            x_row.isna().sum(axis=1).iloc[0]
        )

        if int(row["feature_nan_count"]) > 0:
            rows.append(row)
            continue

        pred = float(
            np.asarray(model.predict(x_row)).reshape(-1)[0]
        )

        row["huesna_v5_8_12_hgb_pred"] = pred
        row["huesna_v5_8_12_hgb_error"] = pred - y_true
        row["huesna_v5_8_12_hgb_abs_error"] = abs(
            pred - y_true
        )
        rows.append(row)

    return pd.DataFrame(rows)


def base_drift(
    current_x: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    split_path = (
        HUESNA_OUTPUT
        / "split_feature_distribution_debug.csv"
    )

    if not split_path.exists():
        return pd.DataFrame()

    stats = pd.read_csv(
        split_path,
        encoding="utf-8-sig",
    )
    train_stats = stats[
        (stats["split"].astype(str) == "train")
        & stats["feature"].isin(HUESNA_TO_HUELVA.keys())
    ].copy()

    if train_stats.empty:
        return pd.DataFrame()

    support_mask = pd.Series(
        False,
        index=current_x.index,
    )
    history = pd.Timedelta(minutes=240)

    for sample_ts in pd.to_datetime(
        targets["sample_timestamp"]
    ):
        support_mask = support_mask | (
            (current_x.index >= sample_ts - history)
            & (current_x.index <= sample_ts)
        )

    support = current_x.loc[support_mask].copy()
    rows: List[Dict[str, Any]] = []

    for feature in HUESNA_TO_HUELVA:
        h_row = train_stats[
            train_stats["feature"] == feature
        ]

        if h_row.empty or feature not in support.columns:
            continue

        h = h_row.iloc[0]
        s = pd.to_numeric(
            support[feature],
            errors="coerce",
        ).dropna()

        if s.empty:
            continue

        h_min = float(h["min"])
        h_max = float(h["max"])
        h_p25 = float(h["p25"])
        h_p50 = float(h["p50"])
        h_p75 = float(h["p75"])
        h_iqr = h_p75 - h_p25
        hu_p50 = float(s.quantile(0.50))

        rows.append(
            {
                "feature": feature,
                "huesna_train_min": h_min,
                "huesna_train_p25": h_p25,
                "huesna_train_p50": h_p50,
                "huesna_train_p75": h_p75,
                "huesna_train_max": h_max,
                "huelva_support_min": float(s.min()),
                "huelva_support_p25": float(
                    s.quantile(0.25)
                ),
                "huelva_support_p50": hu_p50,
                "huelva_support_p75": float(
                    s.quantile(0.75)
                ),
                "huelva_support_max": float(s.max()),
                "huelva_outside_huesna_train_range_ratio": float(
                    ((s < h_min) | (s > h_max)).mean()
                ),
                "robust_median_shift_huesna_iqr": (
                    abs(hu_p50 - h_p50) / h_iqr
                    if h_iqr > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    required = [
        HUELVA_CLEAN,
        HUELVA_THDATA,
        HUESNA_SCRIPT,
        HUESNA_BUNDLE,
    ]
    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Faltan ficheros requeridos:\n"
            + "\n".join(missing)
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean, targets = load_huelva_inputs()
    wide = build_huelva_wide(clean)

    module = load_module(
        HUESNA_SCRIPT,
        "huesna_v5_8_12_formal_transfer",
    )
    bundle = joblib.load(HUESNA_BUNDLE)

    x, current_x, feature_meta = build_features(
        module=module,
        wide=wide,
        bundle=bundle,
    )

    predictions = predict_hgb(
        x=x,
        current_x=current_x,
        targets=targets,
        bundle=bundle,
    )

    valid = predictions[
        [
            "thm_real",
            "huesna_v5_8_12_hgb_pred",
        ]
    ].dropna()

    coverage_n = int(len(valid))
    coverage_total = int(len(targets))
    coverage_ratio = (
        coverage_n / coverage_total
        if coverage_total
        else np.nan
    )

    missing_predictions = predictions[
        predictions["huesna_v5_8_12_hgb_pred"].isna()
    ].copy()

    metrics = compute_metrics(
        valid["thm_real"].to_numpy(dtype=float),
        valid["huesna_v5_8_12_hgb_pred"].to_numpy(
            dtype=float
        ),
    )

    split_meta = dict(
        bundle.get("split_meta", {}) or {}
    )

    metrics_df = pd.DataFrame(
        [
            {
                "run_id": RUN_ID,
                "label": LABEL,
                "calendar_mode": bundle.get(
                    "calendar_mode"
                ),
                "feature_count": len(
                    bundle.get(
                        "selected_feature_cols",
                        [],
                    )
                ),
                "hgb_accepted_in_huesna": bundle.get(
                    "hgb_accepted"
                ),
                "train_end": split_meta.get(
                    "train_end"
                ),
                "dataset_end": split_meta.get(
                    "dataset_end"
                ),
                "coverage_n": coverage_n,
                "coverage_total": coverage_total,
                "coverage_ratio": coverage_ratio,
                **metrics,
            }
        ]
    )

    drift = base_drift(
        current_x=current_x,
        targets=targets,
    )

    run_config = {
        "run_id": RUN_ID,
        "label": LABEL,
        "huesna_script": str(HUESNA_SCRIPT),
        "huesna_output": str(HUESNA_OUTPUT),
        "huesna_bundle": str(HUESNA_BUNDLE),
        "huelva_clean": str(HUELVA_CLEAN),
        "huelva_thdata": str(HUELVA_THDATA),
        "n_huelva_labels": int(len(targets)),
        "prediction_coverage_n": coverage_n,
        "prediction_coverage_total": coverage_total,
        "prediction_coverage_ratio": coverage_ratio,
        "feature_meta": feature_meta,
        "bundle_calendar_mode": bundle.get(
            "calendar_mode"
        ),
        "bundle_feature_count": len(
            bundle.get(
                "selected_feature_cols",
                [],
            )
        ),
        "bundle_hgb_accepted": bundle.get(
            "hgb_accepted"
        ),
    }

    predictions.to_csv(
        OUTPUT_DIR
        / "zero_shot_predictions_huesna_v5_8_12.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_df.to_csv(
        OUTPUT_DIR
        / "zero_shot_metrics_huesna_v5_8_12.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_predictions.to_csv(
        OUTPUT_DIR
        / "zero_shot_missing_predictions_huesna_v5_8_12.csv",
        index=False,
        encoding="utf-8-sig",
    )
    drift.to_csv(
        OUTPUT_DIR
        / "zero_shot_base_feature_drift_huesna_v5_8_12.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (
        OUTPUT_DIR
        / "zero_shot_run_config_v0_1.json"
    ).write_text(
        json.dumps(
            run_config,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        predictions["sample_timestamp"],
        predictions["thm_real"],
        marker="o",
        label="THM real Huelva",
    )
    ax.plot(
        predictions["sample_timestamp"],
        predictions["huesna_v5_8_12_hgb_pred"],
        marker="o",
        label="Huesna v5.8.12 zero-shot",
    )
    ax.set_xlabel("Fecha")
    ax.set_ylabel("THM (ug/L)")
    ax.set_title(
        "Transferencia formal zero-shot: "
        "Huesna v5.8.12 -> Huelva"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / "zero_shot_huesna_v5_8_12_vs_real.png",
        dpi=170,
    )
    plt.close(fig)

    report_lines = [
        "# Transferencia formal zero-shot Huesna -> Huelva",
        "",
        f"- Fuente Huesna: `{LABEL}`",
        f"- Muestras Huelva exactas: **{len(targets)}**",
        "- Sin reentrenamiento ni calibracion con Huelva.",
        (
            f"- Cobertura predictiva: **{coverage_n}/{coverage_total} "
            f"({coverage_ratio:.1%})**."
        ),
        "- Se usa el contrato exacto del bundle estable de Huesna.",
        "",
        "## Metricas globales",
        "",
        "```text",
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "```",
        "",
        "## Nota metodologica",
        "",
        (
            "Las incidencias temporales confirmadas de Huesna "
            "no se trasladan a Huelva. El feature engineering "
            "del bundle se reproduce, pero sin invalidar "
            "intervalos de Huelva por incidencias especificas "
            "del sistema fuente."
        ),
    ]

    (
        OUTPUT_DIR
        / "zero_shot_report_v0_1.md"
    ).write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    row = metrics_df.iloc[0]

    print("============================================")
    print("ZERO-SHOT FORMAL: HUESNA v5.8.12 -> HUELVA")
    print("============================================")
    print(f"Muestras THM Huelva:     {len(targets)}")
    print(f"Filas Huelva wide:       {len(wide)}")
    print(f"Segmentos detectados:    {feature_meta['n_segments']}")
    print(
        f"Features Huesna:         "
        f"{feature_meta['feature_count']}"
    )
    print(
        f"Predicciones:            "
        f"{coverage_n}/{coverage_total} "
        f"({coverage_ratio:.1%})"
    )
    if not missing_predictions.empty:
        print("")
        print("Muestras sin prediccion:")
        cols = [
            "sample_id",
            "sample_timestamp",
            "thm_real",
            "feature_timestamp",
            "alignment_mode",
            "alignment_delay_minutes",
            "feature_nan_count",
        ]
        print(
            missing_predictions[cols].to_string(
                index=False
            )
        )
    print("")
    print(
        f"MAE={row['mae']:.6f}  "
        f"RMSE={row['rmse']:.6f}  "
        f"bias={row['bias']:.6f}  "
        f"R2={row['r2']:.6f}  "
        f"Pearson={row['pearson_r']:.6f}  "
        f"sMAPE={row['smape']:.6f}"
    )
    print("")
    print(f"Salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
