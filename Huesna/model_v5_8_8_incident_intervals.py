from __future__ import annotations

"""
THM v5.8.8 - Soft sensor HGB dinámico con auditoría de calidad trazable y textos consolidados.

Objetivo:
    Estimar THM(t) a partir de las variables de la multiparamétrica,
    sin usar THM como feature de entrada.

Enfoque actual:
    1) Lectura robusta de CSV y alineación temporal X-THM.
    2) Construcción de features permitidas:
       - variables actuales de la multiparamétrica
       - histórico reciente de la multiparamétrica: lags, diferencias,
         medias móviles, mínimos, máximos, rangos y desviaciones
       - variantes de calendario comparables: all / no_doy / no_calendar.
    3) Modelo principal operativo:
       - HistGradientBoostingRegressor global.
       - Selección de hiperparámetros en validación.
       - Selección de variante de calendario según validación HGB.
       - Evaluación final en test.
    4) Baselines de control de calidad:
       - Dummy median.
       - Ridge sobre variables actuales únicamente.
       - No sustituyen al HGB como modelo operativo; solo generan avisos.
    5) Validación robusta:
       - compara HGB contra Dummy/Ridge para detectar ejecuciones dudosas;
       - marca estado OK/WARNING;
       - mantiene HGB como modelo guardado para despliegue/seguimiento.
    6) Ventanas temporales dinámicas:
       - por defecto usa test = últimos N días disponibles;
       - validación = N días inmediatamente anteriores;
       - entrenamiento = histórico anterior, opcionalmente limitado a una ventana reciente;
       - al añadir nuevos CSV, las ventanas se desplazan automáticamente.
    7) Auditoría de calidad de dato:
       - separa ausencia de THM alineado de THM anómalo real;
       - guarda filas THM descartadas por reglas de valor;
       - guarda filas X centinela detectadas antes de reparar lags/rolling.
    8) Exportación de depuración:
       - guarda las features de test usadas por el HGB para diagnóstico de régimen/sesgo.

IMPORTANTE:
    - THM nunca entra como feature X.
    - El histórico de THM no se usa como entrada.
    - El modelo está pensado para despliegue real usando solo la multiparamétrica.
    - El escenario operativo por defecto usa todos los datos válidos.
    - Las comparativas de incidente solo se activan si se proporciona explícitamente
      una ventana de exclusión por argumentos.
"""


import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


TARGET_SENSOR = "DEPOSITO ETAP MULTIPARAMETRICA"
TARGET_COL = "Trihalometanos"
BASE_FEATURES_PREFERRED = ["PH", "UVAS", "conductividad", "turbidez", "Cloro", "Temperatura"]
TARGET_METRIC_CANDIDATES = [
    "Trihalometanos", "Trihalometano", "THM", "THMs", "TTHM", "TTHMs",
    "Trihalomethanes", "Total Trihalomethanes"
]
DEFAULT_OUTPUT_DIR = "model_outputs_thm_softsensor_v5_8_5_hgb_dynamic_windows_quality_audit"
DEFAULT_MIN_VALID_TARGET = 0.1

# Rangos físicos amplios para limpiar glitches evidentes antes de construir lags/rolling.
# No son umbrales normativos de calidad de agua. Son barreras técnicas para evitar
# que lecturas imposibles o de inicialización contaminen lags/rolling durante horas.
FEATURE_PHYSICAL_BOUNDS = {
    "PH": (4.0, 10.5),
    "UVAS": (0.0, 20.0),
    "conductividad": (0.0, 1000.0),
    "turbidez": (0.0, 100.0),
    "Cloro": (0.0, 20.0),
    "Temperatura": (0.0, 45.0),
}

# Valores centinela/anómalos observados en CSV bruto de ETAP el 2026-05-06 08:06:
# Cloro=0, turbidez=0, conductividad=50, UVAS=1, PH=5, THM=0, Temperatura=0.
# Un valor aislado puede ser dudoso; varios a la vez en el mismo minuto indican
# fila inválida de equipo/comunicación/inicialización. El filtro de fila solo se
# activa si coinciden varias señales sospechosas a la vez.
FEATURE_SENTINEL_RULES = {
    "PH": ("<=", 5.1),
    "UVAS": ("<=", 1.1),
    "conductividad": ("<=", 80.0),
    "turbidez": ("<=", 0.001),
    "Cloro": ("<=", 0.05),
    "Temperatura": ("<=", 1.0),
}
ROW_SENTINEL_MIN_HITS = 3


# -----------------------------------------------------------------------------
# Métricas / utilidades
# -----------------------------------------------------------------------------


def smape_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-7)
    return float(100.0 * np.mean(2.0 * np.abs(y_pred - y_true) / denom))



def compute_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes incompatibles en métricas: y_true={y_true.shape}, y_pred={y_pred.shape}")
    error = y_pred - y_true
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "smape": smape_np(y_true, y_pred),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y_true, y_pred)),
    }



def print_metrics_block(title: str, metrics: Dict[str, float], indent: int = 2) -> None:
    pad = " " * indent
    print(f"{title}:")
    for k in ["mae", "rmse", "smape", "bias", "r2"]:
        if k in metrics:
            print(f"{pad}{k}: {metrics[k]:.6f}")



def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return float(value)
    except Exception:
        return default



def safe_int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
        return int(value)
    except Exception:
        return None



def subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "mae": np.nan, "rmse": np.nan, "smape": np.nan, "r2": np.nan}
    m = compute_metrics(y_true[mask], y_pred[mask])
    m["count"] = count
    return m



def metric_or_fallback(metrics: Dict[str, float], key: str, fallback: float) -> float:
    val = metrics.get(key, np.nan)
    if pd.isna(val):
        return fallback
    return float(val)



def composite_rmse_score(global_metrics: Dict[str, float], high_metrics: Dict[str, float], change_metrics: Dict[str, float],
                         w_global: float = 0.4, w_high: float = 0.3, w_change: float = 0.3) -> float:
    g = metric_or_fallback(global_metrics, "rmse", np.inf)
    h = metric_or_fallback(high_metrics, "rmse", g)
    c = metric_or_fallback(change_metrics, "rmse", g)
    return float(w_global * g + w_high * h + w_change * c)


# -----------------------------------------------------------------------------
# Lectura / transformación CSV
# -----------------------------------------------------------------------------





def normalize_text(s: object) -> str:
    return str(s).strip().lower()


def is_target_metric_name(metric: object) -> bool:
    m = normalize_text(metric)
    if not m or m == "nan":
        return False
    if "trihalo" in m:
        return True
    if m in {"thm", "thms", "tthm", "tthms"}:
        return True
    if "total trihalometh" in m:
        return True
    return False


def choose_feature_metrics(wide: pd.DataFrame) -> List[str]:
    available = [c for c in wide.columns if c != "segment_id" and not is_target_metric_name(c)]
    preferred = [c for c in BASE_FEATURES_PREFERRED if c in available]
    extras = [c for c in available if c not in preferred]
    chosen = preferred + extras
    if not chosen:
        raise ValueError("No se han encontrado variables independientes válidas en el sensor de multiparamétrica")
    return chosen


def discover_target_series(df_long: pd.DataFrame,
                           feature_index: pd.DatetimeIndex,
                           target_sensor: Optional[str] = None,
                           target_metric: Optional[str] = None,
                           tolerance_minutes: int = 5) -> Tuple[pd.Series, Dict[str, object]]:
    df = df_long.copy()
    if target_sensor:
        df = df.loc[df["sensor"] == target_sensor].copy()
    if target_metric:
        df = df.loc[df["metric"].astype(str).str.strip() == target_metric].copy()
    else:
        df = df.loc[df["metric"].map(is_target_metric_name)].copy()

    if df.empty:
        sensors = (
            df_long.loc[df_long["metric"].map(is_target_metric_name), ["sensor", "metric"]]
            .drop_duplicates()
            .sort_values(["sensor", "metric"])
        )
        hint = sensors.to_dict("records")[:20]
        raise ValueError(
            "No se ha encontrado ninguna métrica objetivo tipo THM/Trihalometanos en los CSV. "
            f"Primeras combinaciones candidatas detectadas: {hint}"
        )

    grouped = (
        df.groupby(["sensor", "metric", "timestamp"], as_index=False)["value"].mean()
        .sort_values("timestamp")
    )

    candidates = []
    feature_ts = pd.DataFrame({"timestamp": pd.DatetimeIndex(feature_index).sort_values()})
    tol = pd.Timedelta(minutes=tolerance_minutes)

    for (sensor, metric), sub in grouped.groupby(["sensor", "metric"]):
        sub = sub[["timestamp", "value"]].dropna().sort_values("timestamp")
        exact_overlap = int(sub["timestamp"].isin(feature_index).sum())
        merged = pd.merge_asof(
            feature_ts,
            sub.rename(columns={"value": "target_value"}),
            on="timestamp",
            direction="nearest",
            tolerance=tol,
        )
        aligned_count = int(merged["target_value"].notna().sum())
        candidates.append({
            "sensor": sensor,
            "metric": metric,
            "exact_overlap": exact_overlap,
            "aligned_count": aligned_count,
            "n_rows": int(len(sub)),
        })

    cand_df = pd.DataFrame(candidates).sort_values(
        ["aligned_count", "exact_overlap", "n_rows"], ascending=[False, False, False]
    )
    best = cand_df.iloc[0].to_dict()
    best_sensor = best["sensor"]
    best_metric = best["metric"]

    best_sub = (
        grouped.loc[(grouped["sensor"] == best_sensor) & (grouped["metric"] == best_metric), ["timestamp", "value"]]
        .dropna()
        .sort_values("timestamp")
        .rename(columns={"value": TARGET_COL})
    )

    aligned = pd.merge_asof(
        feature_ts,
        best_sub,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
    ).set_index("timestamp")[TARGET_COL]

    meta = {
        "selected_target_sensor": best_sensor,
        "selected_target_metric": best_metric,
        "selected_target_exact_overlap": int(best["exact_overlap"]),
        "selected_target_aligned_count": int(best["aligned_count"]),
        "selected_target_rows": int(best["n_rows"]),
        "target_candidate_table": cand_df.to_dict("records"),
    }
    return aligned, meta


def audit_feature_target_overlap(df_long: pd.DataFrame,
                                 feature_sensor: str,
                                 feature_index_raw: pd.DatetimeIndex,
                                 selected_target_sensor: str,
                                 selected_target_metric: str,
                                 tolerance_minutes: int = 5) -> Dict[str, object]:
    df = df_long.copy()
    if "source_file" not in df.columns:
        df["source_file"] = "unknown"

    feature_rows = df.loc[df["sensor"] == feature_sensor].copy()
    target_rows = df.loc[
        (df["sensor"] == selected_target_sensor) &
        (df["metric"].astype(str).str.strip() == str(selected_target_metric).strip())
    ].copy()

    feature_files = set(feature_rows["source_file"].dropna().astype(str).unique())
    target_files = set(target_rows["source_file"].dropna().astype(str).unique())

    feature_ts = pd.DatetimeIndex(pd.Series(feature_index_raw).dropna().sort_values().unique())
    target_exact_ts = pd.DatetimeIndex(target_rows["timestamp"].dropna().sort_values().unique())

    exact_overlap_ts = feature_ts.intersection(target_exact_ts)
    feature_without_exact = feature_ts.difference(target_exact_ts)

    tol = pd.Timedelta(minutes=tolerance_minutes)
    feature_df = pd.DataFrame({"timestamp": feature_ts})
    target_df = (
        target_rows.groupby("timestamp", as_index=False)["value"].mean()
        .sort_values("timestamp")
        .rename(columns={"value": "target_value"})
    )

    aligned = pd.merge_asof(
        feature_df,
        target_df,
        on="timestamp",
        direction="nearest",
        tolerance=tol,
    )

    aligned_count = int(aligned["target_value"].notna().sum())
    aligned_missing_count = int(aligned["target_value"].isna().sum())

    audit = {
        "csv_total": int(df["source_file"].nunique()),
        "csv_with_feature_sensor": int(len(feature_files)),
        "csv_with_target_metric": int(len(target_files)),
        "csv_with_both_feature_and_target": int(len(feature_files & target_files)),
        "raw_feature_rows": int(len(feature_rows)),
        "raw_target_rows": int(len(target_rows)),
        "feature_timestamps_raw": int(len(feature_ts)),
        "target_timestamps_raw": int(len(target_exact_ts)),
        "exact_overlap_timestamps": int(len(exact_overlap_ts)),
        "feature_timestamps_without_exact_target": int(len(feature_without_exact)),
        "aligned_timestamps_with_target": aligned_count,
        "aligned_timestamps_without_target": aligned_missing_count,
        "feature_time_min": str(feature_ts.min()) if len(feature_ts) else None,
        "feature_time_max": str(feature_ts.max()) if len(feature_ts) else None,
        "target_time_min": str(target_exact_ts.min()) if len(target_exact_ts) else None,
        "target_time_max": str(target_exact_ts.max()) if len(target_exact_ts) else None,
        "alignment_tolerance_minutes": int(tolerance_minutes),
    }
    return audit




# Compat aliases kept to avoid runtime NameError in main().





def read_single_csv(path: Path) -> pd.DataFrame:
    """
    Lector robusto para CSV semicolonados sin asumir cabecera.
    Muchos ficheros del usuario son *headerless* y algunos incluyen caracteres raros
    en la unidad (µg/l, ºC, MicroS/cm). Por eso:
      - forzamos 5 columnas fijas
      - no usamos la primera fila como cabecera
      - toleramos errores de codificación
      - si existiera una cabecera real, se eliminará al no poder parsear timestamp/value
    """
    columns = ["timestamp", "sensor", "metric", "value", "unit"]
    read_kwargs = dict(
        sep=";",
        header=None,
        names=columns,
        usecols=[0, 1, 2, 3, 4],
        engine="python",
        dtype=str,
        on_bad_lines="skip",
    )

    df = None
    for enc in ("utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, encoding_errors="replace", **read_kwargs)
            break
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["sensor"] = out["sensor"].astype(str).str.strip()
    out["metric"] = out["metric"].astype(str).str.strip()
    out["value"] = (
        out["value"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    )
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    # Eliminar posibles filas de cabecera reales o basura no parseable.
    out = out.dropna(subset=["timestamp", "sensor", "metric", "value"])
    out = out.loc[(out["sensor"] != "") & (out["metric"] != "")].copy()
    return out



def load_all_csvs(csv_dir: Path) -> Tuple[pd.DataFrame, int]:
    files = sorted(csv_dir.glob("*.csv"))
    parts = []
    for f in files:
        part = read_single_csv(f)
        if not part.empty:
            part = part.copy()
            part["source_file"] = f.name
            parts.append(part)
    if not parts:
        raise FileNotFoundError(f"No se han encontrado CSV válidos en {csv_dir}")
    return pd.concat(parts, ignore_index=True), len(files)



def pivot_sensor_wide(df_long: pd.DataFrame, sensor: str) -> pd.DataFrame:
    df = df_long.loc[df_long["sensor"] == sensor].copy()
    if df.empty:
        raise ValueError(f"No hay datos para el sensor '{sensor}'")

    # Agrupación previa por timestamp+metric por si hubiera duplicados.
    df = df.groupby(["timestamp", "metric"], as_index=False)["value"].mean()
    wide = df.pivot(index="timestamp", columns="metric", values="value").sort_index()
    wide.columns.name = None
    return wide



def regularize_by_segments(wide: pd.DataFrame, freq: str = "1min", max_gap_minutes: int = 5,
                           ffill_limit: int = 5) -> Tuple[pd.DataFrame, int]:
    if wide.empty:
        raise ValueError("No hay datos para regularizar")

    wide = wide.sort_index().copy()
    diffs = wide.index.to_series().diff().dt.total_seconds().div(60)
    segment_ids = (diffs.isna() | (diffs > max_gap_minutes)).cumsum().astype(int)
    wide["segment_id"] = segment_ids.values

    regs = []
    for seg_id, seg in wide.groupby("segment_id"):
        seg = seg.drop(columns=["segment_id"]).sort_index()
        full_index = pd.date_range(seg.index.min(), seg.index.max(), freq=freq)
        seg_reg = seg.reindex(full_index)
        seg_reg = seg_reg.ffill(limit=ffill_limit)
        seg_reg["segment_id"] = seg_id
        regs.append(seg_reg)

    reg = pd.concat(regs).sort_index()
    reg.index.name = "timestamp"
    return reg, int(segment_ids.max())


# -----------------------------------------------------------------------------
# Features / etiquetas
# -----------------------------------------------------------------------------




def _sentinel_mask_for_feature(col: str, s: pd.Series) -> pd.Series:
    rule = FEATURE_SENTINEL_RULES.get(str(col))
    if rule is None:
        return pd.Series(False, index=s.index)
    op, threshold = rule
    if op == "<=":
        return s <= float(threshold)
    if op == "<":
        return s < float(threshold)
    if op == ">=":
        return s >= float(threshold)
    if op == ">":
        return s > float(threshold)
    return pd.Series(False, index=s.index)


def sanitize_current_features(
    current_df: pd.DataFrame,
    segment_ids: pd.Series,
    enabled: bool = True,
    ffill_limit: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Limpia glitches evidentes en variables de entrada antes de crear histórico.

    Criterio v5.8.5:
        1) Detecta filas centinela cuando varias variables caen simultáneamente
           en valores imposibles/sospechosos. Ejemplo confirmado en CSV bruto:
           2026-05-06 08:06 con Cloro=0, turbidez=0, conductividad=50,
           UVAS=1, PH=5 y Temperatura=0.
        2) En esas filas, invalida todas las X para que no entren en lags/rolling.
        3) Aplica además rangos físicos muy amplios por variable.
        4) Repara solo con ffill/bfill corto dentro del mismo segmento temporal.

    Esto no usa THM como entrada ni como criterio de limpieza de X.
    """
    out = current_df.copy()
    meta: Dict[str, object] = {
        "enabled": bool(enabled),
        "ffill_limit": int(ffill_limit),
        "total_values_cleaned": 0,
        "row_sentinel_min_hits": int(ROW_SENTINEL_MIN_HITS),
        "row_sentinel_rows_cleaned": 0,
        "row_sentinel_timestamps": [],
        "by_feature": {},
    }
    if not enabled:
        return out, meta

    numeric = out.apply(pd.to_numeric, errors="coerce")

    sentinel_hits = pd.DataFrame(index=out.index)
    for col in numeric.columns:
        sentinel_hits[col] = _sentinel_mask_for_feature(col, numeric[col]).fillna(False)
    hit_count = sentinel_hits.sum(axis=1)
    row_invalid = hit_count >= int(ROW_SENTINEL_MIN_HITS)

    if row_invalid.any():
        timestamps = [str(ts) for ts in out.index[row_invalid]]
        meta["row_sentinel_rows_cleaned"] = int(row_invalid.sum())
        meta["row_sentinel_timestamps"] = timestamps
        sentinel_records = []
        for ts in out.index[row_invalid]:
            record = {
                "timestamp": str(ts),
                "sentinel_hit_count": int(hit_count.loc[ts]),
            }
            for col in out.columns:
                value = pd.to_numeric(pd.Series([out.loc[ts, col]]), errors="coerce").iloc[0]
                record[str(col)] = None if pd.isna(value) else float(value)
                record[f"{col}_sentinel_hit"] = bool(sentinel_hits.loc[ts, col])
            sentinel_records.append(record)
        meta["row_sentinel_records"] = sentinel_records
        # Invalidamos todas las variables X en filas centinela. Luego se repara
        # por segmento para que una fila corrupta no contamine ventanas de hasta 240 min.
        numeric.loc[row_invalid, :] = np.nan

    seg = pd.Series(segment_ids, index=out.index)
    for col in numeric.columns:
        s = pd.to_numeric(numeric[col], errors="coerce")
        invalid = pd.Series(False, index=out.index)

        # Cuenta como limpiado por fila centinela si esa fila tenía valor no nulo originalmente.
        row_sentinel_cleaned_count = int((row_invalid & out[col].notna()).sum())

        bounds = FEATURE_PHYSICAL_BOUNDS.get(str(col))
        if bounds is not None:
            low, high = bounds
            invalid = invalid | (s < low) | (s > high)

        physical_cleaned_count = int(invalid.fillna(False).sum())
        if physical_cleaned_count:
            s = s.mask(invalid)

        repaired_parts = []
        for _, idx in seg.groupby(seg).groups.items():
            ss = s.loc[idx].copy()
            ss = ss.ffill(limit=int(ffill_limit)).bfill(limit=1)
            repaired_parts.append(ss)
        if repaired_parts:
            s = pd.concat(repaired_parts).sort_index()

        out[col] = s.reindex(out.index)
        total_col_cleaned = row_sentinel_cleaned_count + physical_cleaned_count
        meta["by_feature"][col] = {
            "values_cleaned": int(total_col_cleaned),
            "row_sentinel_values_cleaned": int(row_sentinel_cleaned_count),
            "physical_values_cleaned": int(physical_cleaned_count),
            "physical_bounds": bounds,
            "sentinel_rule": FEATURE_SENTINEL_RULES.get(str(col)),
        }
        meta["total_values_cleaned"] += int(total_col_cleaned)

    return out, meta

def clean_target_series(
    y: pd.Series,
    min_valid_target: float = DEFAULT_MIN_VALID_TARGET,
    max_valid_target: Optional[float] = None,
) -> Tuple[pd.Series, Dict[str, object]]:
    """Elimina valores objetivo no válidos antes de entrenar/evaluar.

    En los datos recientes aparecen lecturas aisladas de THM = 0 mientras el
    entorno real está ~10-12. Esas lecturas no son representativas y pueden
    distorsionar validación, gráficas y aprendizaje. Por defecto se descarta
    THM <= 0.1. Se puede desactivar con --min-valid-target 0.
    """
    out = pd.to_numeric(y, errors="coerce").copy()
    invalid_value = pd.Series(False, index=out.index)
    if min_valid_target is not None:
        invalid_value = invalid_value | (out <= float(min_valid_target))
    if max_valid_target is not None:
        invalid_value = invalid_value | (out > float(max_valid_target))
    invalid_missing = out.isna()
    invalid = invalid_value | invalid_missing

    anomaly_records = []
    for ts, value in out.loc[invalid_value & out.notna()].items():
        rules = []
        if min_valid_target is not None and value <= float(min_valid_target):
            rules.append(f"value <= {float(min_valid_target)}")
        if max_valid_target is not None and value > float(max_valid_target):
            rules.append(f"value > {float(max_valid_target)}")
        anomaly_records.append({
            "timestamp": str(ts),
            "target": TARGET_COL,
            "value": float(value),
            "rule": " | ".join(rules),
        })

    meta = {
        "min_valid_target": None if min_valid_target is None else float(min_valid_target),
        "max_valid_target": None if max_valid_target is None else float(max_valid_target),
        "rows_invalid_target": int(invalid.sum()),
        "rows_invalid_target_missing": int(invalid_missing.sum()),
        "rows_invalid_target_value_rules": int((invalid_value & out.notna()).sum()),
        "target_anomaly_records": anomaly_records,
    }
    out = out.mask(invalid)
    return out, meta


def build_data_quality_report(primary_res: Dict[str, object]) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construye un pequeño reporte de calidad de dato para justificar limpiezas."""
    feature_cleaning_meta = dict(primary_res.get("feature_cleaning_meta", {}) or {})
    target_cleaning_meta = dict(primary_res.get("target_cleaning_meta", {}) or {})

    rows_with_target_after_align = int(primary_res.get("rows_with_target_after_align", 0) or 0)
    rows_without_target_alignment = int(primary_res.get("rows_lost_before_model", 0) or 0)
    rows_invalid_target_total = int(target_cleaning_meta.get("rows_invalid_target", 0) or 0)
    rows_invalid_target_value_rules = int(target_cleaning_meta.get("rows_invalid_target_value_rules", max(0, rows_invalid_target_total - rows_without_target_alignment)) or 0)
    rows_lost_feature_eng = int(primary_res.get("rows_lost_feature_eng", 0) or 0)

    row_sentinel_timestamps = list(feature_cleaning_meta.get("row_sentinel_timestamps", []) or [])
    if len(row_sentinel_timestamps) > 20:
        row_sentinel_timestamps_preview = row_sentinel_timestamps[:20]
    else:
        row_sentinel_timestamps_preview = row_sentinel_timestamps
    row_sentinel_timestamps_preview_count = len(row_sentinel_timestamps_preview)

    report = {
        "rows_with_target_after_align": rows_with_target_after_align,
        "rows_without_target_alignment": rows_without_target_alignment,
        "target_cleaning": {
            "min_valid_target": target_cleaning_meta.get("min_valid_target"),
            "max_valid_target": target_cleaning_meta.get("max_valid_target"),
            "rows_invalid_total_before_dropna": rows_invalid_target_total,
            "rows_invalid_due_to_missing_alignment": rows_without_target_alignment,
            "rows_invalid_due_to_value_rules": rows_invalid_target_value_rules,
        },
        "feature_cleaning": {
            "total_values_cleaned": int(feature_cleaning_meta.get("total_values_cleaned", 0) or 0),
            "row_sentinel_rows_cleaned": int(feature_cleaning_meta.get("row_sentinel_rows_cleaned", 0) or 0),
            "row_sentinel_timestamps_preview": row_sentinel_timestamps_preview,
            "row_sentinel_timestamps_preview_count": row_sentinel_timestamps_preview_count,
            "row_sentinel_timestamps_total": len(row_sentinel_timestamps),
        },
        "rows_lost_after_feature_engineering_dropna": rows_lost_feature_eng,
    }

    summary_rows = [
        {"category": "target", "metric": "rows_with_target_after_align", "value": rows_with_target_after_align, "detail": "Filas con THM disponible tras alineación temporal X-THM"},
        {"category": "target", "metric": "rows_without_target_alignment", "value": rows_without_target_alignment, "detail": "Filas sin THM alineado; no son anomalías, solo ausencia de objetivo"},
        {"category": "target", "metric": "rows_invalid_target_total_before_dropna", "value": rows_invalid_target_total, "detail": "Total marcado como inválido por limpieza del objetivo antes de dropna"},
        {"category": "target", "metric": "rows_invalid_due_to_value_rules", "value": rows_invalid_target_value_rules, "detail": "THM descartados por valor inválido/anómalo (sin contar ausencia de alineación)"},
        {"category": "features", "metric": "total_values_cleaned", "value": int(feature_cleaning_meta.get("total_values_cleaned", 0) or 0), "detail": "Valores X limpiados por reglas físicas o centinelas"},
        {"category": "features", "metric": "row_sentinel_rows_cleaned", "value": int(feature_cleaning_meta.get("row_sentinel_rows_cleaned", 0) or 0), "detail": "Filas donde se limpió una fila completa por patrón centinela"},
        {"category": "features", "metric": "rows_lost_after_feature_engineering_dropna", "value": rows_lost_feature_eng, "detail": "Filas descartadas después de construir features por NaN residual"},
    ]
    summary_df = pd.DataFrame(summary_rows)

    feature_rows = []
    by_feature = dict(feature_cleaning_meta.get("by_feature", {}) or {})
    for feat, meta in by_feature.items():
        feature_rows.append({
            "feature": feat,
            "values_cleaned": int(meta.get("values_cleaned", 0) or 0),
            "row_sentinel_values_cleaned": int(meta.get("row_sentinel_values_cleaned", 0) or 0),
            "physical_values_cleaned": int(meta.get("physical_values_cleaned", 0) or 0),
            "physical_bounds": json.dumps(meta.get("physical_bounds"), ensure_ascii=False),
            "sentinel_rule": json.dumps(meta.get("sentinel_rule"), ensure_ascii=False),
        })
    feature_df = pd.DataFrame(feature_rows)
    if not feature_df.empty:
        feature_df = feature_df.sort_values(["values_cleaned", "feature"], ascending=[False, True]).reset_index(drop=True)

    target_anomalies_df = pd.DataFrame(list(target_cleaning_meta.get("target_anomaly_records", []) or []))
    if target_anomalies_df.empty:
        target_anomalies_df = pd.DataFrame(columns=["timestamp", "target", "value", "rule"])

    x_sentinel_df = pd.DataFrame(list(feature_cleaning_meta.get("row_sentinel_records", []) or []))
    if x_sentinel_df.empty:
        x_sentinel_df = pd.DataFrame(columns=["timestamp", "sentinel_hit_count"])

    return report, summary_df, feature_df, target_anomalies_df, x_sentinel_df


def add_calendar_features(df: pd.DataFrame, mode: str = "all") -> pd.DataFrame:
    out = df.copy()
    if mode == "no_calendar":
        return out

    ts = pd.DatetimeIndex(out.index)
    minute_of_day = ts.hour * 60 + ts.minute
    dow = ts.dayofweek

    out["minute_day_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    out["minute_day_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    if mode == "all":
        doy = ts.dayofyear
        out["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
        out["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    elif mode != "no_doy":
        raise ValueError(f"Modo de calendario no soportado: {mode}")
    return out


def build_feature_table(wide_reg: pd.DataFrame,
                        feature_base_cols: Sequence[str],
                        use_history_multi: bool,
                        max_history_minutes: int,
                        change_lookback: int,
                        calendar_mode: str = "all",
                        clean_feature_outliers: bool = True,
                        feature_clean_ffill_limit: int = 10,
                        min_valid_target: float = DEFAULT_MIN_VALID_TARGET,
                        max_valid_target: Optional[float] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if TARGET_COL not in wide_reg.columns:
        raise ValueError(f"No existe la columna objetivo '{TARGET_COL}' en los datos ya alineados")

    current_df = pd.DataFrame(
        {col: pd.to_numeric(wide_reg[col], errors="coerce") for col in feature_base_cols},
        index=wide_reg.index,
    )
    current_cols = list(current_df.columns)

    current_df, feature_cleaning_meta = sanitize_current_features(
        current_df=current_df,
        segment_ids=wide_reg["segment_id"],
        enabled=bool(clean_feature_outliers),
        ffill_limit=int(feature_clean_ffill_limit),
    )

    calendar_df = add_calendar_features(pd.DataFrame(index=wide_reg.index), mode=calendar_mode)
    feature_frames = [current_df]
    if not calendar_df.empty:
        feature_frames.append(calendar_df)

    if use_history_multi:
        lag_candidates = [1, 5, 15, 30, 60, 120, 180, 240]
        lags = [l for l in lag_candidates if l <= max_history_minutes]
        roll_candidates = [5, 15, 30, 60, 120, 180, 240]
        rolls = [r for r in roll_candidates if r <= max_history_minutes]

        extra_cols = {}
        for col in current_cols:
            s = current_df[col]
            for lag in lags:
                extra_cols[f"{col}_lag_{lag}"] = s.shift(lag)
            for diff_lag in [1, 5, 15, 30, 60]:
                if diff_lag <= max_history_minutes:
                    extra_cols[f"{col}_diff_{diff_lag}"] = s - s.shift(diff_lag)
            for win in rolls:
                roll = s.rolling(win, min_periods=max(2, min(win, 5)))
                extra_cols[f"{col}_rollmean_{win}"] = roll.mean()
                if win >= 5:
                    extra_cols[f"{col}_rollstd_{win}"] = roll.std()
                    roll_min = roll.min()
                    roll_max = roll.max()
                    extra_cols[f"{col}_rollmin_{win}"] = roll_min
                    extra_cols[f"{col}_rollmax_{win}"] = roll_max
                    extra_cols[f"{col}_rollrange_{win}"] = roll_max - roll_min
        feature_frames.append(pd.DataFrame(extra_cols, index=wide_reg.index))

    feat_df = pd.concat(feature_frames, axis=1)
    feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()].copy()

    labels = pd.DataFrame(index=wide_reg.index)
    labels[TARGET_COL], target_cleaning_meta = clean_target_series(
        wide_reg[TARGET_COL],
        min_valid_target=min_valid_target,
        max_valid_target=max_valid_target,
    )

    seg = wide_reg["segment_id"]
    labels["thm_change_abs"] = np.nan
    for _, idx in seg.groupby(seg).groups.items():
        y_seg = labels.loc[idx, TARGET_COL]
        labels.loc[idx, "thm_change_abs"] = (y_seg - y_seg.shift(change_lookback)).abs().fillna(0.0)

    out = pd.concat([feat_df, labels, wide_reg[["segment_id"]]], axis=1)
    out = out.loc[:, ~out.columns.duplicated()].copy()

    feature_cols = [c for c in feat_df.columns if TARGET_COL.lower() not in c.lower()]
    bad_cols = [c for c in feature_cols if TARGET_COL.lower() in c.lower()]
    if bad_cols:
        raise ValueError(f"Fuga de THM detectada en features: {bad_cols}")

    out = out.dropna(subset=[TARGET_COL])
    out = out.dropna(subset=feature_cols)
    out.attrs["feature_cleaning_meta"] = feature_cleaning_meta
    out.attrs["target_cleaning_meta"] = target_cleaning_meta
    return out, current_cols, feature_cols


# -----------------------------------------------------------------------------
# División temporal
# -----------------------------------------------------------------------------


def temporal_split(df: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split temporal inválido; revisa ratios o cantidad de datos")
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    return train_df, val_df, test_df


def temporal_split_dynamic_windows(
    df: pd.DataFrame,
    split_mode: str,
    train_ratio: float,
    val_ratio: float,
    val_window_days: float,
    test_window_days: float,
    train_history_days: float,
    min_train_rows: int,
    min_val_rows: int,
    min_test_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Divide train/val/test en ventanas temporales adaptadas a datos entrantes.

    Modo dynamic (por defecto):
        - test = últimos --test-window-days días disponibles;
        - val = --val-window-days días inmediatamente anteriores;
        - train = histórico anterior a val, o solo los últimos --train-history-days
          días si se indica un valor > 0.

    Esto evita que el tamaño relativo del dataset sea el criterio principal y hace
    que, al entrar nuevos CSV, el periodo de evaluación avance de forma explícita.
    """
    if df.empty:
        raise ValueError("No hay filas para dividir en train/val/test")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("El DataFrame debe tener DatetimeIndex para usar ventanas temporales")

    df = df.sort_index().copy()
    split_mode = str(split_mode).lower().strip()

    if split_mode == "ratio":
        train_df, val_df, test_df = temporal_split(df, train_ratio=train_ratio, val_ratio=val_ratio)
        meta = {
            "split_mode": "ratio",
            "fallback_used": False,
            "train_start": str(train_df.index.min()),
            "train_end": str(train_df.index.max()),
            "val_start": str(val_df.index.min()),
            "val_end": str(val_df.index.max()),
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
        }
        return train_df, val_df, test_df, meta

    if split_mode != "dynamic":
        raise ValueError(f"split_mode no soportado: {split_mode}")

    if val_window_days <= 0 or test_window_days <= 0:
        raise ValueError("--val-window-days y --test-window-days deben ser > 0")
    if train_history_days < 0:
        raise ValueError("--train-history-days debe ser >= 0; usa 0 para histórico completo")

    max_ts = df.index.max()
    min_ts = df.index.min()
    test_start = max_ts - pd.Timedelta(days=float(test_window_days))
    val_end = test_start
    val_start = val_end - pd.Timedelta(days=float(val_window_days))

    if train_history_days and train_history_days > 0:
        train_start = val_start - pd.Timedelta(days=float(train_history_days))
        train_mask = (df.index >= train_start) & (df.index < val_start)
        train_start_effective = train_start
    else:
        train_mask = df.index < val_start
        train_start_effective = min_ts

    val_mask = (df.index >= val_start) & (df.index < val_end)
    test_mask = df.index >= test_start

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    fallback_reason = None
    if len(train_df) < int(min_train_rows):
        fallback_reason = f"train_rows<{min_train_rows}"
    elif len(val_df) < int(min_val_rows):
        fallback_reason = f"val_rows<{min_val_rows}"
    elif len(test_df) < int(min_test_rows):
        fallback_reason = f"test_rows<{min_test_rows}"

    if fallback_reason is not None:
        # Fallback explícito para no romper ejecuciones con pocos datos iniciales.
        train_df, val_df, test_df = temporal_split(df, train_ratio=train_ratio, val_ratio=val_ratio)
        meta = {
            "split_mode": "dynamic",
            "effective_split_mode": "ratio_fallback",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "requested_val_window_days": float(val_window_days),
            "requested_test_window_days": float(test_window_days),
            "requested_train_history_days": float(train_history_days),
            "train_start": str(train_df.index.min()),
            "train_end": str(train_df.index.max()),
            "val_start": str(val_df.index.min()),
            "val_end": str(val_df.index.max()),
            "test_start": str(test_df.index.min()),
            "test_end": str(test_df.index.max()),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_ratio": float(train_ratio),
            "val_ratio": float(val_ratio),
        }
        return train_df, val_df, test_df, meta

    meta = {
        "split_mode": "dynamic",
        "effective_split_mode": "dynamic",
        "fallback_used": False,
        "val_window_days": float(val_window_days),
        "test_window_days": float(test_window_days),
        "train_history_days": float(train_history_days),
        "train_history_policy": "all_available_before_validation" if not train_history_days else "limited_recent_history",
        "train_start_requested": str(train_start_effective),
        "val_start_requested": str(val_start),
        "val_end_requested": str(val_end),
        "test_start_requested": str(test_start),
        "dataset_start": str(min_ts),
        "dataset_end": str(max_ts),
        "train_start": str(train_df.index.min()),
        "train_end": str(train_df.index.max()),
        "val_start": str(val_df.index.min()),
        "val_end": str(val_df.index.max()),
        "test_start": str(test_df.index.min()),
        "test_end": str(test_df.index.max()),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "min_train_rows": int(min_train_rows),
        "min_val_rows": int(min_val_rows),
        "min_test_rows": int(min_test_rows),
    }
    return train_df, val_df, test_df, meta


# -----------------------------------------------------------------------------
# Modelos
# -----------------------------------------------------------------------------


def build_hgb_regressor(params: Dict[str, object], random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=safe_float(params.get("learning_rate"), 0.05),
        max_iter=int(safe_float(params.get("max_iter"), 300)),
        max_leaf_nodes=int(safe_float(params.get("max_leaf_nodes"), 31)),
        min_samples_leaf=int(safe_float(params.get("min_samples_leaf"), 40)),
        l2_regularization=safe_float(params.get("l2_regularization"), 0.0),
        max_depth=safe_int_or_none(params.get("max_depth")),
        random_state=random_state,
    )






def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, random_state=42)),
    ])






# -----------------------------------------------------------------------------
# Selección de modelos
# -----------------------------------------------------------------------------


def evaluate_prediction_set(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            high_mask: np.ndarray,
                            change_mask: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float]:
    g = compute_metrics(y_true, y_pred)
    h = subset_metrics(y_true, y_pred, high_mask)
    c = subset_metrics(y_true, y_pred, change_mask)
    score = composite_rmse_score(g, h, c)
    return g, h, c, score




def select_best_global_hgb(X_train: pd.DataFrame,
                           y_train: np.ndarray,
                           abs_change_train: np.ndarray,
                           X_val: pd.DataFrame,
                           y_val: np.ndarray,
                           high_mask_val: np.ndarray,
                           change_mask_val: np.ndarray,
                           high_threshold: float,
                           change_threshold: float,
                           random_state: int,
                           extra_sample_weight: Optional[np.ndarray] = None) -> Tuple[Dict[str, object], pd.DataFrame, np.ndarray, HistGradientBoostingRegressor]:
    grid = [
        {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 15, "min_samples_leaf": 80, "l2_regularization": 0.0, "max_depth": None},
        {"learning_rate": 0.03, "max_iter": 400, "max_leaf_nodes": 31, "min_samples_leaf": 80, "l2_regularization": 0.001, "max_depth": None},
        {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 60, "l2_regularization": 0.001, "max_depth": 6},
        {"learning_rate": 0.05, "max_iter": 500, "max_leaf_nodes": 63, "min_samples_leaf": 60, "l2_regularization": 0.001, "max_depth": 6},
    ]

    rows = []
    best = None
    best_score = np.inf
    best_model = None
    best_pred_val = None

    sw = 1.0 + 0.6 * (y_train >= high_threshold).astype(float) + 0.6 * (abs_change_train >= change_threshold).astype(float)
    if extra_sample_weight is not None:
        extra_sample_weight = np.asarray(extra_sample_weight, dtype=float).reshape(-1)
        if len(extra_sample_weight) != len(sw):
            raise ValueError("extra_sample_weight tiene longitud distinta a y_train")
        sw = sw * extra_sample_weight

    for params in grid:
        model = build_hgb_regressor(params, random_state=random_state)
        model.fit(X_train, y_train, sample_weight=sw)
        pred_val = np.asarray(model.predict(X_val), dtype=float).reshape(-1)
        g, h, c, score = evaluate_prediction_set(y_val, pred_val, high_mask_val, change_mask_val)
        rows.append({
            **params,
            "score": score,
            "val_rmse": g["rmse"],
            "val_mae": g["mae"],
            "val_r2": g["r2"],
            "val_high_rmse": h["rmse"],
            "val_change_rmse": c["rmse"],
        })
        if score < best_score:
            best_score = score
            best = params
            best_model = model
            best_pred_val = pred_val

    return best, pd.DataFrame(rows), best_pred_val, best_model









# -----------------------------------------------------------------------------
# Gráficas / persistencia
# -----------------------------------------------------------------------------


def save_simple_plot(
    df: Optional[pd.DataFrame] = None,
    x: Optional[Sequence] = None,
    y: Optional[Sequence] = None,
    title: str = "",
    out_path: Optional[Path] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xtick_labels: Optional[Sequence[str]] = None,
    rotate_xticks: int = 0,
    highlight_index: Optional[int] = None,
    highlight_label: Optional[str] = None,
) -> None:
    if out_path is None:
        raise ValueError("out_path es obligatorio")

    if df is not None:
        if not isinstance(x, str) or not isinstance(y, str):
            raise ValueError("Si se pasa df, x e y deben ser nombres de columna")
        x_values = df[x].values
        y_values = df[y].values
        xlabel = xlabel or x
        ylabel = ylabel or y
    else:
        if x is None or y is None:
            raise ValueError("Si no se pasa df, x e y deben contener arrays/secuencias")
        x_values = np.asarray(x)
        y_values = np.asarray(y)

    plt.figure(figsize=(8, 4.5))
    plt.plot(x_values, y_values, marker="o" if xtick_labels is not None else None)
    if highlight_index is not None and 0 <= int(highlight_index) < len(x_values):
        hi = int(highlight_index)
        plt.scatter([x_values[hi]], [y_values[hi]], s=80, zorder=5)
        if highlight_label:
            plt.annotate(
                highlight_label,
                xy=(x_values[hi], y_values[hi]),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=9,
            )
    plt.title(title)
    plt.xlabel(xlabel or "x")
    plt.ylabel(ylabel or "y")
    if xtick_labels is not None:
        plt.xticks(ticks=np.arange(len(xtick_labels)), labels=list(xtick_labels), rotation=rotate_xticks)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_predictions_plot(
    ts: Optional[pd.Index] = None,
    y_true: Optional[np.ndarray] = None,
    preds: Optional[Dict[str, np.ndarray]] = None,
    out_path: Optional[Path] = None,
    max_points: int = 1200,
    timestamps: Optional[pd.Index] = None,
    y_pred: Optional[np.ndarray] = None,
    title: str = "THM real vs predicciones en test",
) -> None:
    if ts is None:
        ts = timestamps
    if out_path is None or ts is None or y_true is None:
        raise ValueError("out_path, timestamps/ts e y_true son obligatorios")
    if preds is None:
        if y_pred is None:
            raise ValueError("Debe pasarse preds o y_pred")
        preds = {"Predicción": np.asarray(y_pred)}

    if len(ts) > max_points:
        step = max(1, len(ts) // max_points)
        ts = ts[::step]
        y_true = y_true[::step]
        preds = {k: v[::step] for k, v in preds.items()}

    plt.figure(figsize=(12, 5.5))
    plt.plot(ts, y_true, label="Real", linewidth=1.5)
    for name, pred in preds.items():
        plt.plot(ts, pred, label=name, alpha=0.9)
    plt.title(title)
    plt.xlabel("timestamp")
    plt.ylabel("THM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()




def _dedupe_keep_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _numeric_summary(series: pd.Series) -> Dict[str, object]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "max": np.nan,
        }
    return {
        "count": int(s.shape[0]),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def export_feature_debug_tables(output_dir: Path,
                                primary_res: Dict[str, object],
                                feat_imp_df: pd.DataFrame,
                                top_n_features: int = 60) -> Dict[str, object]:
    """Exporta tablas de depuración con las X usadas por el HGB.

    Estas salidas no se usan para inferencia productiva. Sirven para diagnosticar
    cambios de régimen cuando `test_predictions.csv` solo contiene y/pred.
    """
    output_dir = Path(output_dir)
    test_df = primary_res["test_df"].copy()
    current_cols = list(primary_res.get("current_cols", []) or [])
    model_feature_cols = list(primary_res.get("model_feature_cols", []) or [])
    selected_feature_cols = list(primary_res.get("selected_feature_cols", model_feature_cols) or [])

    top_features: List[str] = []
    if feat_imp_df is not None and not feat_imp_df.empty and "feature" in feat_imp_df.columns:
        top_features = [
            str(f)
            for f in feat_imp_df.head(int(top_n_features))["feature"].tolist()
            if str(f) in test_df.columns
        ]

    debug_feature_cols = _dedupe_keep_order(current_cols + top_features)
    all_feature_cols = [c for c in selected_feature_cols if c in test_df.columns]
    if not all_feature_cols:
        all_feature_cols = [c for c in model_feature_cols if c in test_df.columns]

    base = pd.DataFrame(index=test_df.index)
    base["y_true"] = np.asarray(primary_res["y_test"], dtype=float)
    base["pred_dummy"] = np.asarray(primary_res["test_pred_dummy"], dtype=float)
    base["pred_ridge_current"] = np.asarray(primary_res["test_pred_ridge"], dtype=float)
    base["pred_hgb_candidate"] = np.asarray(primary_res["test_pred_hgb"], dtype=float)
    base["pred_selected"] = np.asarray(primary_res["test_pred_selected"], dtype=float)
    base["selected_model_type"] = primary_res["selected_model_type"]
    base["error_pred_minus_true"] = base["pred_selected"] - base["y_true"]
    base["abs_error"] = base["error_pred_minus_true"].abs()
    base["is_high"] = np.asarray(primary_res["high_mask_test"], dtype=int)
    base["is_change"] = np.asarray(primary_res["change_mask_test"], dtype=int)
    base["is_incident"] = np.asarray(primary_res["incident_mask_test"], dtype=int)
    base["abs_change_true"] = test_df["thm_change_abs"].to_numpy(dtype=float)
    base["date"] = pd.DatetimeIndex(base.index).date.astype(str)
    base["hour"] = pd.DatetimeIndex(base.index).hour

    # Tabla ligera: variables actuales + top features por importancia.
    top_debug = pd.concat([base, test_df[debug_feature_cols]], axis=1)
    top_debug = top_debug.reset_index().rename(columns={"index": "timestamp"})
    top_debug.to_csv(output_dir / "test_features_top_debug.csv", index=False)

    # Tabla completa: todas las features seleccionadas usadas por el HGB.
    # Puede ser más grande, pero es la más útil si luego queremos auditar una feature concreta.
    full_debug_cols = _dedupe_keep_order(list(base.columns) + all_feature_cols)
    full_debug = pd.concat([base, test_df[all_feature_cols]], axis=1)
    full_debug = full_debug.loc[:, ~full_debug.columns.duplicated()].copy()
    full_debug = full_debug.reset_index().rename(columns={"index": "timestamp"})
    full_debug.to_csv(output_dir / "test_features_debug.csv", index=False)

    # Resumen diario de y/pred/error y features principales.
    daily_features = _dedupe_keep_order(current_cols + top_features[: min(len(top_features), 30)])
    daily_source = pd.concat([base[["y_true", "pred_selected", "error_pred_minus_true", "abs_error", "date"]], test_df[daily_features]], axis=1)
    daily_rows = []
    for date, g in daily_source.groupby("date", sort=True):
        row = {"date": date, "n_rows": int(len(g))}
        for col in ["y_true", "pred_selected", "error_pred_minus_true", "abs_error"] + daily_features:
            if col in g.columns:
                row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
                row[f"{col}_min"] = float(pd.to_numeric(g[col], errors="coerce").min())
                row[f"{col}_max"] = float(pd.to_numeric(g[col], errors="coerce").max())
        daily_rows.append(row)
    daily_df = pd.DataFrame(daily_rows)
    daily_df.to_csv(output_dir / "test_feature_daily_summary.csv", index=False)

    # Resumen por split para comparar train/val/test en las features importantes.
    split_rows = []
    split_sources = [
        ("train", primary_res["train_df"]),
        ("val", primary_res["val_df"]),
        ("test", primary_res["test_df"]),
    ]
    split_feature_cols = _dedupe_keep_order(current_cols + top_features[: min(len(top_features), 80)])
    for split_name, split_df in split_sources:
        for col in split_feature_cols:
            if col not in split_df.columns:
                continue
            row = {
                "split": split_name,
                "feature": col,
                **_numeric_summary(split_df[col]),
            }
            split_rows.append(row)
    split_summary_df = pd.DataFrame(split_rows)
    split_summary_df.to_csv(output_dir / "split_feature_distribution_debug.csv", index=False)

    manifest = {
        "created_files": [
            "test_features_debug.csv",
            "test_features_top_debug.csv",
            "test_feature_daily_summary.csv",
            "split_feature_distribution_debug.csv",
            "feature_debug_manifest.json",
        ],
        "n_test_rows": int(len(test_df)),
        "n_current_features": int(len(current_cols)),
        "n_model_features": int(len(model_feature_cols)),
        "n_selected_features": int(len(selected_feature_cols)),
        "n_exported_full_features": int(len(all_feature_cols)),
        "n_exported_top_debug_features": int(len(debug_feature_cols)),
        "top_n_features_requested": int(top_n_features),
        "current_features": current_cols,
        "top_debug_features": debug_feature_cols,
        "full_feature_columns": all_feature_cols,
        "note": (
            "Estas tablas son para diagnóstico offline de régimen/sesgo. "
            "THM no se usa como feature de entrada; y_true/pred se exportan solo como columnas de auditoría."
        ),
    }
    (output_dir / "feature_debug_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------





def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="THM v5.8.8 - Soft sensor HGB dinámico con auditoría de calidad trazable")
    p.add_argument("--csv-dir", default="csv_descargados", help="Carpeta con CSVs")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Carpeta de salida")
    p.add_argument("--sensor", default=TARGET_SENSOR, help="Sensor de la multiparamétrica (features)")
    p.add_argument("--target-sensor", default=None, help="Sensor del THM. Por defecto se autodetecta")
    p.add_argument("--target-metric", default=None, help="Nombre exacto de la métrica objetivo THM. Por defecto se autodetecta")
    p.add_argument("--target-tolerance-minutes", type=int, default=5, help="Tolerancia para alinear THM con multiparamétrica")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70, help="Solo se usa directamente con --split-mode ratio o como fallback")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Solo se usa directamente con --split-mode ratio o como fallback")
    p.add_argument(
        "--split-mode",
        choices=["dynamic", "ratio"],
        default="dynamic",
        help="dynamic usa ventanas por días que avanzan al añadir CSV; ratio usa porcentajes históricos",
    )
    p.add_argument("--val-window-days", type=float, default=7.0, help="Días inmediatamente anteriores al test usados como validación")
    p.add_argument("--test-window-days", type=float, default=7.0, help="Últimos N días disponibles usados como test")
    p.add_argument(
        "--train-history-days",
        type=float,
        default=0.0,
        help="Días de histórico de entrenamiento antes de validación; 0 = todo el histórico disponible",
    )
    p.add_argument("--min-train-rows", type=int, default=1000, help="Mínimo de filas en train para aceptar split dinámico")
    p.add_argument("--min-val-rows", type=int, default=300, help="Mínimo de filas en validación para aceptar split dinámico")
    p.add_argument("--min-test-rows", type=int, default=300, help="Mínimo de filas en test para aceptar split dinámico")
    p.add_argument("--freq", default="1min")
    p.add_argument("--max-gap-minutes", type=int, default=5)
    p.add_argument("--ffill-limit", type=int, default=5)
    p.add_argument("--disable-feature-cleaning", action="store_true", help="Desactiva limpieza conservadora de glitches en variables X")
    p.add_argument("--feature-clean-ffill-limit", type=int, default=10, help="Relleno máximo tras limpiar glitches de variables X")
    p.add_argument("--min-valid-target", type=float, default=DEFAULT_MIN_VALID_TARGET, help="Descarta THM <= este valor antes de entrenar/evaluar; usa 0 para desactivar")
    p.add_argument("--max-valid-target", type=float, default=None, help="Descarta THM > este valor si se define")
    p.add_argument("--current-only", action="store_true", help="Usar solo lectura instantánea de la multiparamétrica")
    p.add_argument("--max-history-minutes", type=int, default=240, help="Buffer histórico de multiparamétrica si no se usa --current-only")
    p.add_argument(
        "--calendar-mode",
        choices=["compare", "all", "no_doy", "no_calendar"],
        default="compare",
        help="compare evalúa all/no_doy/no_calendar; el resto fuerza una variante fija",
    )
    p.add_argument(
        "--acceptance-min-rmse-improvement",
        type=float,
        default=0.02,
        help="Mejora mínima relativa de RMSE en validación para considerar HGB sin aviso frente al mejor baseline",
    )
    p.add_argument(
        "--acceptance-max-bias-rmse-ratio",
        type=float,
        default=0.85,
        help="Máximo |bias|/RMSE permitido en validación para considerar HGB sin aviso",
    )

    p.add_argument(
        "--incident-intervals-csv",
        default=None,
        help=(
            "CSV con intervalos confirmados por planta. Columnas obligatorias: start,end. "
            "Columnas opcionales: label,buffer_before_minutes,buffer_after_minutes,notes."
        ),
    )
    p.add_argument(
        "--incident-policy",
        choices=["exclude", "compare", "include"],
        default="exclude",
        help=(
            "exclude usa como escenario operativo la exclusión de intervalos; compare evalúa "
            "include y exclude pero selecciona entre los escenarios excluidos; include ignora los intervalos."
        ),
    )
    p.add_argument(
        "--incident-buffer-before-minutes",
        type=int,
        default=0,
        help="Minutos adicionales excluidos antes de cada incidencia.",
    )
    p.add_argument(
        "--incident-buffer-after-minutes",
        type=int,
        default=None,
        help=(
            "Minutos adicionales excluidos después de cada incidencia. Por defecto usa "
            "--max-history-minutes para evitar contaminación de lags y rollings."
        ),
    )
    p.add_argument("--exclude-window-start", default=None, help="Compatibilidad: inicio de un único incidente")
    p.add_argument("--exclude-window-end", default=None, help="Compatibilidad: fin de un único incidente")

    p.add_argument("--change-lookback", type=int, default=30, help="Ventana para definir episodio de cambio fuerte")
    p.add_argument("--change-quantile", type=float, default=0.75, help="Cuantil de train para definir cambio fuerte")
    p.add_argument("--high-threshold-mode", choices=["train_quantile", "fixed"], default="train_quantile")
    p.add_argument("--high-threshold-quantile", type=float, default=0.75)
    p.add_argument("--high-threshold-fixed", type=float, default=None)

    p.add_argument("--importance-sample-size", type=int, default=2500)
    p.add_argument("--importance-n-repeats", type=int, default=5)
    p.add_argument(
        "--log-level",
        choices=["compact", "verbose"],
        default="compact",
        help="compact muestra un resumen legible y guarda el detalle en training_verbose.log; verbose muestra el log completo",
    )
    return p.parse_args()


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    s = pd.Series(np.asarray(values, dtype=float)).dropna()
    if s.empty:
        return {k: np.nan for k in ["count", "min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean", "median", "std"]}
    return {
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "p01": float(s.quantile(0.01)),
        "p05": float(s.quantile(0.05)),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if s.shape[0] > 1 else 0.0,
    }


def print_distribution_block(title: str, dist: Dict[str, float]) -> None:
    print(title)
    print(f"  count:   {int(dist['count'])}")
    print(f"  min/max: {dist['min']:.6f} / {dist['max']:.6f}")
    print(f"  mean:    {dist['mean']:.6f}")
    print(f"  median:  {dist['median']:.6f}")
    print(f"  std:     {dist['std']:.6f}")
    print(f"  p01/p05/p10: {dist['p01']:.6f} / {dist['p05']:.6f} / {dist['p10']:.6f}")
    print(f"  p25/p50/p75: {dist['p25']:.6f} / {dist['p50']:.6f} / {dist['p75']:.6f}")
    print(f"  p90/p95/p99: {dist['p90']:.6f} / {dist['p95']:.6f} / {dist['p99']:.6f}")


def parse_feature_family(name: str, current_cols: Sequence[str]) -> Tuple[str, str]:
    if name in {"minute_day_sin", "minute_day_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"}:
        return "calendar", name
    for base in current_cols:
        if name == base:
            return base, "current"
        if name.startswith(base + "_lag_"):
            return base, "lag"
        if name.startswith(base + "_diff_"):
            return base, "diff"
        if name.startswith(base + "_rollmean_"):
            return base, "rollmean"
        if name.startswith(base + "_rollstd_"):
            return base, "rollstd"
        if name.startswith(base + "_rollmin_"):
            return base, "rollmin"
        if name.startswith(base + "_rollmax_"):
            return base, "rollmax"
        if name.startswith(base + "_rollrange_"):
            return base, "rollrange"
    return "other", "other"


def compute_permutation_importances_df(model,
                                       X_val: pd.DataFrame,
                                       y_val: np.ndarray,
                                       current_cols: Sequence[str],
                                       random_state: int,
                                       sample_size: int,
                                       n_repeats: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(X_val)
    if n == 0:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "base_var", "family"]), pd.DataFrame(columns=["base_var", "family", "importance_mean"])
    if n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(np.arange(n), size=sample_size, replace=False))
        X_imp = X_val.iloc[idx].copy()
        y_imp = np.asarray(y_val)[idx]
    else:
        X_imp = X_val.copy()
        y_imp = np.asarray(y_val).copy()

    result = permutation_importance(
        model,
        X_imp,
        y_imp,
        scoring="neg_root_mean_squared_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    feat_df = pd.DataFrame({
        "feature": X_imp.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    parsed = feat_df["feature"].map(lambda s: parse_feature_family(str(s), current_cols))
    feat_df["base_var"] = [p[0] for p in parsed]
    feat_df["family"] = [p[1] for p in parsed]
    feat_df = feat_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    family_df = (
        feat_df.groupby(["base_var", "family"], as_index=False)["importance_mean"]
        .sum()
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    return feat_df, family_df








def select_primary_calendar_variant_robust(include_results: List[Dict[str, object]],
                                           prefer_all_calendar_tolerance: float = 0.05) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Selecciona la variante operativa con una política más robusta que v5.8.5/v5.8.6.

    Problema detectado en junio 2026:
        - `no_calendar` ganaba por poco en RMSE de validación.
        - `all_calendar` era ligeramente peor en validación, pero mucho más estable
          en test/régimen estacional alto.

    Política v5.8.7:
        1) Calcula la mejor variante por RMSE HGB de validación, como antes.
        2) Si existe `all_calendar` y su RMSE de validación está dentro de una
           tolerancia relativa pequeña respecto a la mejor variante, se prefiere
           `all_calendar` por robustez estacional.
        3) Esto no usa test para seleccionar. Solo usa validación y una regla
           conservadora definida a priori.
    """
    if not include_results:
        raise ValueError("include_results vacío; no hay variantes para seleccionar")

    best_by_val = min(include_results, key=lambda r: float(r["hgb_val_metrics"]["rmse"]))
    all_calendar = None
    for res in include_results:
        if str(res.get("variant_label")) == "all_calendar" or str(res.get("variant_mode")) == "all":
            all_calendar = res
            break

    best_rmse = float(best_by_val["hgb_val_metrics"]["rmse"])
    selected = best_by_val
    reason = "best_validation_rmse"
    relative_gap_all_vs_best = None

    if all_calendar is not None:
        all_rmse = float(all_calendar["hgb_val_metrics"]["rmse"])
        if best_rmse > 0 and np.isfinite(best_rmse) and np.isfinite(all_rmse):
            relative_gap_all_vs_best = float((all_rmse - best_rmse) / best_rmse)
        else:
            relative_gap_all_vs_best = np.nan

        if all_calendar is best_by_val:
            selected = all_calendar
            reason = "all_calendar_is_best_validation_rmse"
        elif (
            relative_gap_all_vs_best is not None
            and np.isfinite(relative_gap_all_vs_best)
            and relative_gap_all_vs_best <= float(prefer_all_calendar_tolerance)
        ):
            selected = all_calendar
            reason = (
                "prefer_all_calendar_within_validation_tolerance"
            )

    meta = {
        "policy": "v5.8.8_robust_calendar_selection",
        "selection_uses_test": False,
        "best_by_validation_variant_label": best_by_val.get("variant_label"),
        "best_by_validation_variant_mode": best_by_val.get("variant_mode"),
        "best_by_validation_hgb_val_rmse": best_rmse,
        "all_calendar_available": all_calendar is not None,
        "all_calendar_hgb_val_rmse": None if all_calendar is None else float(all_calendar["hgb_val_metrics"]["rmse"]),
        "relative_gap_all_calendar_vs_best": relative_gap_all_vs_best,
        "prefer_all_calendar_tolerance": float(prefer_all_calendar_tolerance),
        "selected_variant_label": selected.get("variant_label"),
        "selected_variant_mode": selected.get("variant_mode"),
        "selection_reason": reason,
        "note": (
            "Si all_calendar está dentro de la tolerancia de validación, se prefiere "
            "por robustez estacional. Esta regla evita seleccionar variantes sin calendario "
            "que ganan marginalmente en validación pero pueden colapsar en cambios de régimen."
        ),
    }
    return selected, meta



def load_incident_intervals(args: argparse.Namespace) -> List[Dict[str, object]]:
    """Carga intervalos confirmados y calcula la ventana efectiva de exclusión.

    La ventana posterior se amplía por defecto con max_history_minutes para que
    ninguna fila de entrenamiento conserve lags o rollings contaminados por la
    incidencia. Los límites son inclusivos.
    """
    default_before = max(0, int(args.incident_buffer_before_minutes))
    default_after = (
        max(0, int(args.incident_buffer_after_minutes))
        if args.incident_buffer_after_minutes is not None
        else max(0, int(args.max_history_minutes))
    )

    rows: List[Dict[str, object]] = []
    if args.incident_intervals_csv:
        path = Path(args.incident_intervals_csv)
        if not path.exists():
            raise FileNotFoundError(f"No existe el CSV de incidencias: {path}")
        try:
            incident_df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
        except UnicodeDecodeError:
            incident_df = pd.read_csv(path, sep=None, engine="python", encoding="latin1")
        incident_df.columns = [str(c).strip() for c in incident_df.columns]
        required = {"start", "end"}
        if not required.issubset(incident_df.columns):
            raise ValueError(
                f"El CSV de incidencias debe contener start,end. Columnas: {list(incident_df.columns)}"
            )
        for i, row in incident_df.iterrows():
            rows.append({
                "start": row.get("start"),
                "end": row.get("end"),
                "label": row.get("label", f"incident_{i + 1:03d}"),
                "buffer_before_minutes": row.get("buffer_before_minutes", default_before),
                "buffer_after_minutes": row.get("buffer_after_minutes", default_after),
                "notes": row.get("notes", ""),
                "source": str(path),
            })

    if args.exclude_window_start or args.exclude_window_end:
        if not (args.exclude_window_start and args.exclude_window_end):
            raise ValueError("Deben definirse juntos --exclude-window-start y --exclude-window-end")
        rows.append({
            "start": args.exclude_window_start,
            "end": args.exclude_window_end,
            "label": "cli_single_incident",
            "buffer_before_minutes": default_before,
            "buffer_after_minutes": default_after,
            "notes": "Intervalo definido por línea de comandos",
            "source": "cli",
        })

    intervals: List[Dict[str, object]] = []
    for i, row in enumerate(rows):
        start_ts = pd.to_datetime(row.get("start"), errors="coerce", dayfirst=False)
        end_ts = pd.to_datetime(row.get("end"), errors="coerce", dayfirst=False)
        if pd.isna(start_ts) or pd.isna(end_ts):
            raise ValueError(f"Intervalo de incidencia {i + 1} no interpretable: {row}")
        if end_ts < start_ts:
            raise ValueError(f"Fin anterior al inicio en intervalo {i + 1}: {row}")

        before_raw = row.get("buffer_before_minutes", default_before)
        after_raw = row.get("buffer_after_minutes", default_after)
        before = default_before if pd.isna(before_raw) else max(0, int(float(before_raw)))
        after = default_after if pd.isna(after_raw) else max(0, int(float(after_raw)))

        intervals.append({
            "label": str(row.get("label") or f"incident_{i + 1:03d}"),
            "start": str(pd.Timestamp(start_ts)),
            "end": str(pd.Timestamp(end_ts)),
            "effective_start": str(pd.Timestamp(start_ts) - pd.Timedelta(minutes=before)),
            "effective_end": str(pd.Timestamp(end_ts) + pd.Timedelta(minutes=after)),
            "buffer_before_minutes": before,
            "buffer_after_minutes": after,
            "notes": "" if pd.isna(row.get("notes")) else str(row.get("notes", "")),
            "source": str(row.get("source", "unknown")),
        })
    return intervals


def incident_mask_from_intervals(
    index: pd.Index,
    intervals: Sequence[Dict[str, object]],
) -> Tuple[np.ndarray, Dict[str, object]]:
    idx = pd.DatetimeIndex(index)
    mask = np.zeros(len(idx), dtype=bool)
    details: List[Dict[str, object]] = []
    for interval in intervals:
        start_ts = pd.Timestamp(interval["effective_start"])
        end_ts = pd.Timestamp(interval["effective_end"])
        interval_mask = np.asarray((idx >= start_ts) & (idx <= end_ts), dtype=bool)
        mask |= interval_mask
        details.append({**interval, "rows_marked": int(interval_mask.sum())})
    return mask, {
        "enabled": bool(intervals),
        "interval_count": len(intervals),
        "rows_marked": int(mask.sum()),
        "intervals": details,
    }


def apply_incident_strategy(
    model_df: pd.DataFrame,
    intervals: Sequence[Dict[str, object]],
    strategy: str,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if strategy not in {"include", "exclude_strict"}:
        raise ValueError(f"Estrategia de incidente no soportada: {strategy}")

    mask, base_meta = incident_mask_from_intervals(model_df.index, intervals)
    out = model_df.copy()
    out["is_incident"] = mask.astype(int)

    if strategy == "include":
        return out, {**base_meta, "strategy": strategy, "rows_removed": 0}

    filtered = out.loc[~mask].copy()
    filtered["is_incident"] = 0
    return filtered, {
        **base_meta,
        "strategy": strategy,
        "rows_removed": int(mask.sum()),
    }



def run_hgb_variant(variant_mode: str,
                    wide_reg: pd.DataFrame,
                    feature_base_cols: Sequence[str],
                    use_history_multi: bool,
                    args: argparse.Namespace,
                    incident_strategy: str = "include") -> Dict[str, object]:
    model_df, current_cols, model_feature_cols = build_feature_table(
        wide_reg=wide_reg,
        feature_base_cols=feature_base_cols,
        use_history_multi=use_history_multi,
        max_history_minutes=args.max_history_minutes,
        change_lookback=args.change_lookback,
        calendar_mode=variant_mode,
        clean_feature_outliers=not args.disable_feature_cleaning,
        feature_clean_ffill_limit=args.feature_clean_ffill_limit,
        min_valid_target=args.min_valid_target,
        max_valid_target=args.max_valid_target,
    )

    feature_cleaning_meta = dict(model_df.attrs.get("feature_cleaning_meta", {}))
    target_cleaning_meta = dict(model_df.attrs.get("target_cleaning_meta", {}))

    rows_with_target_after_align = int(wide_reg[TARGET_COL].notna().sum())
    rows_lost_before_model = int(len(wide_reg) - rows_with_target_after_align)

    model_df, incident_meta = apply_incident_strategy(
        model_df=model_df,
        intervals=getattr(args, "incident_intervals", []),
        strategy=incident_strategy,
    )
    rows_lost_feature_eng = int(rows_with_target_after_align - len(model_df))

    train_df, val_df, test_df, split_meta = temporal_split_dynamic_windows(
        model_df,
        split_mode=args.split_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        val_window_days=args.val_window_days,
        test_window_days=args.test_window_days,
        train_history_days=args.train_history_days,
        min_train_rows=args.min_train_rows,
        min_val_rows=args.min_val_rows,
        min_test_rows=args.min_test_rows,
    )

    X_train_current = train_df[current_cols].copy()
    X_val_current = val_df[current_cols].copy()
    X_test_current = test_df[current_cols].copy()

    X_train_model = train_df[model_feature_cols].copy()
    X_val_model = val_df[model_feature_cols].copy()
    X_test_model = test_df[model_feature_cols].copy()

    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = val_df[TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    abs_change_train = train_df["thm_change_abs"].to_numpy(dtype=float)
    abs_change_val = val_df["thm_change_abs"].to_numpy(dtype=float)
    abs_change_test = test_df["thm_change_abs"].to_numpy(dtype=float)

    incident_mask_val = val_df.get("is_incident", pd.Series(0, index=val_df.index)).to_numpy(dtype=bool)
    incident_mask_test = test_df.get("is_incident", pd.Series(0, index=test_df.index)).to_numpy(dtype=bool)

    sample_weight_train = train_df.get("sample_weight_incident", pd.Series(1.0, index=train_df.index)).to_numpy(dtype=float)

    if args.high_threshold_mode == "fixed":
        if args.high_threshold_fixed is None:
            raise ValueError("Debes indicar --high-threshold-fixed cuando --high-threshold-mode=fixed")
        high_threshold = float(args.high_threshold_fixed)
    else:
        high_threshold = float(pd.Series(y_train).quantile(args.high_threshold_quantile))
    change_threshold = float(pd.Series(abs_change_train).quantile(args.change_quantile))

    high_mask_val = y_val >= high_threshold
    change_mask_val = abs_change_val >= change_threshold
    high_mask_test = y_test >= high_threshold
    change_mask_test = abs_change_test >= change_threshold

    dist_train = summarize_distribution(y_train)
    dist_val = summarize_distribution(y_val)
    dist_test = summarize_distribution(y_test)

    # ------------------------------------------------------------------
    # Baselines en validación: se entrenan solo con train.
    # Sirven como referencia objetiva para generar avisos sobre HGB.
    # ------------------------------------------------------------------
    dummy = DummyRegressor(strategy="median")
    dummy.fit(X_train_current, y_train)
    pred_dummy_val = np.asarray(dummy.predict(X_val_current), dtype=float).reshape(-1)

    ridge = make_ridge_pipeline(alpha=1.0)
    ridge.fit(X_train_current, y_train)
    pred_ridge_val = np.asarray(ridge.predict(X_val_current), dtype=float).reshape(-1)

    dummy_val_metrics, dummy_high_val, dummy_change_val, dummy_val_score = evaluate_prediction_set(
        y_val, pred_dummy_val, high_mask_val, change_mask_val
    )
    ridge_val_metrics, ridge_high_val, ridge_change_val, ridge_val_score = evaluate_prediction_set(
        y_val, pred_ridge_val, high_mask_val, change_mask_val
    )

    # ------------------------------------------------------------------
    # HGB operativo: selección de hiperparámetros solo entre configuraciones HGB.
    # Los baselines se evalúan después como control de calidad.
    # ------------------------------------------------------------------
    best_params, selection_df, pred_hgb_val, hgb_train_model = select_best_global_hgb(
        X_train=X_train_model,
        y_train=y_train,
        abs_change_train=abs_change_train,
        X_val=X_val_model,
        y_val=y_val,
        high_mask_val=high_mask_val,
        change_mask_val=change_mask_val,
        high_threshold=high_threshold,
        change_threshold=change_threshold,
        random_state=args.random_state,
        extra_sample_weight=sample_weight_train,
    )
    hgb_val_metrics, hgb_high_val, hgb_change_val, hgb_val_score = evaluate_prediction_set(
        y_val, pred_hgb_val, high_mask_val, change_mask_val
    )

    baseline_candidates = [
        {"type": "dummy_median", "val_metrics": dummy_val_metrics, "val_score": dummy_val_score},
        {"type": "ridge_current_only", "val_metrics": ridge_val_metrics, "val_score": ridge_val_score},
    ]
    best_baseline = min(baseline_candidates, key=lambda x: x["val_metrics"]["rmse"])
    best_baseline_rmse = float(best_baseline["val_metrics"]["rmse"])

    hgb_val_rmse = float(hgb_val_metrics["rmse"])
    hgb_val_bias_ratio = abs(float(hgb_val_metrics.get("bias", 0.0))) / max(hgb_val_rmse, 1e-12)
    required_hgb_rmse = best_baseline_rmse * (1.0 - float(args.acceptance_min_rmse_improvement))
    hgb_beats_baseline = hgb_val_rmse <= required_hgb_rmse
    hgb_bias_ok = hgb_val_bias_ratio <= float(args.acceptance_max_bias_rmse_ratio)
    hgb_accepted = bool(hgb_beats_baseline and hgb_bias_ok)

    # ------------------------------------------------------------------
    # Modelos finales reentrenados con train+val para evaluar test.
    # Se mantiene HGB como modelo operativo; los baselines se entrenan
    # para control de calidad y comparación.
    # ------------------------------------------------------------------
    trainval_df = pd.concat([train_df, val_df], axis=0)
    X_trainval_current = trainval_df[current_cols].copy()
    X_trainval_model = trainval_df[model_feature_cols].copy()
    y_trainval = trainval_df[TARGET_COL].to_numpy(dtype=float)
    abs_change_trainval = trainval_df["thm_change_abs"].to_numpy(dtype=float)
    sample_weight_trainval = trainval_df.get("sample_weight_incident", pd.Series(1.0, index=trainval_df.index)).to_numpy(dtype=float)

    dummy_final = DummyRegressor(strategy="median")
    dummy_final.fit(X_trainval_current, y_trainval)
    pred_dummy_test = np.asarray(dummy_final.predict(X_test_current), dtype=float).reshape(-1)

    ridge_final = make_ridge_pipeline(alpha=1.0)
    ridge_final.fit(X_trainval_current, y_trainval)
    pred_ridge_test = np.asarray(ridge_final.predict(X_test_current), dtype=float).reshape(-1)

    hgb_final = build_hgb_regressor(best_params, random_state=args.random_state)
    sw_trainval = 1.0 + 0.6 * (y_trainval >= high_threshold).astype(float) + 0.6 * (abs_change_trainval >= change_threshold).astype(float)
    sw_trainval = sw_trainval * sample_weight_trainval
    hgb_final.fit(X_trainval_model, y_trainval, sample_weight=sw_trainval)

    pred_hgb_test = np.asarray(hgb_final.predict(X_test_model), dtype=float).reshape(-1)
    y_min = float(np.nanmin(y_trainval))
    y_max = float(np.nanmax(y_trainval))
    pred_hgb_test = np.clip(pred_hgb_test, y_min, y_max)

    m_dummy = compute_metrics(y_test, pred_dummy_test)
    m_ridge = compute_metrics(y_test, pred_ridge_test)
    m_hgb = compute_metrics(y_test, pred_hgb_test)

    high_sub_hgb = subset_metrics(y_test, pred_hgb_test, high_mask_test)
    change_sub_hgb = subset_metrics(y_test, pred_hgb_test, change_mask_test)
    incident_sub_hgb = subset_metrics(y_test, pred_hgb_test, incident_mask_test)
    normal_sub_hgb = subset_metrics(y_test, pred_hgb_test, ~incident_mask_test)

    # Modelo operativo: mantenemos HGB como línea principal.
    # Dummy/Ridge NO sustituyen al HGB; funcionan como controles de calidad.
    validation_warnings: List[str] = []
    if not hgb_beats_baseline:
        validation_warnings.append(
            "HGB no mejora al mejor baseline simple en RMSE de validación; revisar ventana temporal/estabilidad"
        )
    if not hgb_bias_ok:
        validation_warnings.append(
            "HGB presenta sesgo relativo alto en validación; revisar calibración"
        )

    hgb_validation_status = "OK" if not validation_warnings else "WARNING"
    selected_model_type = "hgb_global"
    selected_model = hgb_final
    selected_feature_cols = model_feature_cols
    selected_current_only = False
    pred_selected_test = pred_hgb_test
    selected_val_metrics = hgb_val_metrics
    selected_val_score = hgb_val_score
    if hgb_validation_status == "OK":
        selected_reason = "HGB operativo: supera controles frente a baselines en validación"
    else:
        selected_reason = "HGB operativo con avisos de validación; baselines usados solo como control"

    m_selected = compute_metrics(y_test, pred_selected_test)
    high_sub_selected = subset_metrics(y_test, pred_selected_test, high_mask_test)
    change_sub_selected = subset_metrics(y_test, pred_selected_test, change_mask_test)
    incident_sub_selected = subset_metrics(y_test, pred_selected_test, incident_mask_test)
    normal_sub_selected = subset_metrics(y_test, pred_selected_test, ~incident_mask_test)

    selection_df = selection_df.copy()
    selection_df["variant_mode"] = variant_mode
    selection_df["incident_strategy"] = incident_strategy

    val_selection_row = selection_df.sort_values("score", ascending=True).iloc[0].to_dict()
    val_score = float(val_selection_row["score"])
    test_score = composite_rmse_score(m_hgb, high_sub_hgb, change_sub_hgb)

    acceptance = {
        # Compatibilidad: hgb_accepted indica si pasa los controles estrictos.
        # Aunque sea False, el modelo operativo guardado sigue siendo HGB.
        "hgb_accepted": hgb_accepted,
        "hgb_validation_status": hgb_validation_status,
        "validation_warnings": validation_warnings,
        "selected_model_type": selected_model_type,
        "operational_model_type": selected_model_type,
        "baseline_policy": "control_only_not_operational_replacement",
        "selected_reason": selected_reason,
        "best_baseline_type": best_baseline["type"],
        "best_baseline_val_rmse": best_baseline_rmse,
        "hgb_val_rmse": hgb_val_rmse,
        "required_hgb_val_rmse": required_hgb_rmse,
        "hgb_val_bias_ratio": hgb_val_bias_ratio,
        "hgb_beats_baseline": hgb_beats_baseline,
        "hgb_bias_ok": hgb_bias_ok,
        "acceptance_min_rmse_improvement": float(args.acceptance_min_rmse_improvement),
        "acceptance_max_bias_rmse_ratio": float(args.acceptance_max_bias_rmse_ratio),
    }

    return {
        "variant_mode": variant_mode,
        "variant_label": {"all": "all_calendar", "no_doy": "no_doy", "no_calendar": "no_calendar"}.get(variant_mode, variant_mode),
        "incident_strategy": incident_strategy,
        "model_df": model_df,
        "current_cols": current_cols,
        "model_feature_cols": model_feature_cols,
        "selected_feature_cols": selected_feature_cols,
        "selected_current_only": selected_current_only,
        "rows_with_target_after_align": rows_with_target_after_align,
        "rows_lost_before_model": rows_lost_before_model,
        "rows_lost_feature_eng": rows_lost_feature_eng,
        "feature_cleaning_meta": feature_cleaning_meta,
        "target_cleaning_meta": target_cleaning_meta,
        "incident_meta": incident_meta,
        "split_meta": split_meta,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_val_model": X_val_model,
        "y_val": y_val,
        "y_test": y_test,
        "high_mask_test": high_mask_test,
        "change_mask_test": change_mask_test,
        "incident_mask_test": incident_mask_test,
        "dist_train": dist_train,
        "dist_val": dist_val,
        "dist_test": dist_test,
        "high_threshold": high_threshold,
        "change_threshold": change_threshold,
        "dummy_model": dummy_final,
        "ridge_model": ridge_final,
        "global_model": hgb_final,
        "hgb_global_model": hgb_final,
        "selected_model": selected_model,
        "selected_model_type": selected_model_type,
        "best_params": best_params,
        "selection_df": selection_df,
        "dummy_val_metrics": dummy_val_metrics,
        "ridge_val_metrics": ridge_val_metrics,
        "hgb_val_metrics": hgb_val_metrics,
        "selected_val_metrics": selected_val_metrics,
        "dummy_val_score": dummy_val_score,
        "ridge_val_score": ridge_val_score,
        "hgb_val_score": hgb_val_score,
        "selected_val_score": selected_val_score,
        "test_pred_dummy": pred_dummy_test,
        "test_pred_ridge": pred_ridge_test,
        "test_pred_global": pred_hgb_test,
        "test_pred_hgb": pred_hgb_test,
        "test_pred_selected": pred_selected_test,
        "m_dummy": m_dummy,
        "m_ridge": m_ridge,
        "m_global": m_hgb,
        "m_hgb": m_hgb,
        "m_selected": m_selected,
        "high_sub_global": high_sub_hgb,
        "change_sub_global": change_sub_hgb,
        "incident_sub_global": incident_sub_hgb,
        "normal_sub_global": normal_sub_hgb,
        "high_sub_selected": high_sub_selected,
        "change_sub_selected": change_sub_selected,
        "incident_sub_selected": incident_sub_selected,
        "normal_sub_selected": normal_sub_selected,
        "acceptance": acceptance,
        "val_score": val_score,
        "test_score": test_score,
    }


def run_training(args: argparse.Namespace) -> None:
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_history_multi = not args.current_only
    buffer_required = 0 if args.current_only else int(args.max_history_minutes)
    if args.calendar_mode == "compare":
        calendar_modes = ["all", "no_doy", "no_calendar"]
    else:
        calendar_modes = [args.calendar_mode]

    incident_intervals = load_incident_intervals(args)
    args.incident_intervals = incident_intervals
    incident_defined = bool(incident_intervals)
    incident_columns = [
        "label", "start", "end", "effective_start", "effective_end",
        "buffer_before_minutes", "buffer_after_minutes", "notes", "source",
    ]
    pd.DataFrame(incident_intervals, columns=incident_columns).to_csv(
        output_dir / "incident_intervals_applied.csv", index=False, encoding="utf-8-sig"
    )
    if not incident_defined or args.incident_policy == "include":
        incident_scenarios = [{"name": "include", "label": "baseline"}]
    elif args.incident_policy == "compare":
        incident_scenarios = [
            {"name": "include", "label": "baseline"},
            {"name": "exclude_strict", "label": "exclude_confirmed_incidents"},
        ]
    else:
        incident_scenarios = [
            {"name": "exclude_strict", "label": "exclude_confirmed_incidents"},
        ]

    print("\n==============================")
    print("ENTRENAMIENTO THM v5.8.8 SOFT-SENSOR HGB DINÁMICO")
    print("==============================")
    print(f"Carpeta CSV:            {csv_dir.resolve()}")
    print(f"Carpeta salida:         {output_dir.resolve()}")
    print(f"Sensor multiparamétrica:{args.sensor}")
    print("Objetivo:               estimación instantánea de THM(t)")
    print("Usa THM como entrada:   False")
    print("Usa histórico THM X:    False")
    print(f"Usa histórico multi X:  {use_history_multi}")
    print(f"Buffer multi requerido: {buffer_required} min")
    print(f"Random state:           {args.random_state}")
    print(f"Split temporal:         {args.split_mode}")
    if args.split_mode == "dynamic":
        train_policy = "todo el histórico previo" if args.train_history_days <= 0 else f"últimos {args.train_history_days:g} días antes de validación"
        print(f"Ventana validación:     {args.val_window_days:g} días previos al test")
        print(f"Ventana test:           últimos {args.test_window_days:g} días disponibles")
        print(f"Ventana entrenamiento:  {train_policy}")
    else:
        print(f"Ratios train/val/test:  {args.train_ratio:.2f} / {args.val_ratio:.2f} / {1.0 - args.train_ratio - args.val_ratio:.2f}")
    print("Modelo principal:       HGB global")
    print("Modelo operativo:       HGB global")
    print(f"Calendario:             {', '.join(calendar_modes)}")
    print(f"Control baseline RMSE:  aviso si HGB no mejora >= {100.0 * args.acceptance_min_rmse_improvement:.2f}% en validación")
    print(f"Control bias/RMSE:      aviso si HGB > {args.acceptance_max_bias_rmse_ratio:.3f}")
    if incident_defined:
        print(f"Incidencias confirmadas:{len(incident_intervals)}")
        print(f"Política incidencias:   {args.incident_policy}")
        for interval in incident_intervals:
            print(
                "  - "
                f"{interval['label']}: {interval['start']} -> {interval['end']} | "
                f"exclusión efectiva {interval['effective_start']} -> {interval['effective_end']}"
            )
    else:
        print("Incidencias confirmadas:0")
    print(f"Change lookback:        {args.change_lookback} min")

    df_long, n_csv = load_all_csvs(csv_dir)
    print(f"CSV leídos:             {n_csv}")
    print(f"Filas totales:          {len(df_long)}")
    print(f"Rango temporal:         {df_long['timestamp'].min()} -> {df_long['timestamp'].max()}")

    wide = pivot_sensor_wide(df_long, args.sensor)
    feature_base_cols = choose_feature_metrics(wide)
    print(f"Filas sensor wide:      {len(wide)}")
    print(f"Columnas wide:          {list(wide.columns)}")
    print(f"Variables X usadas:     {feature_base_cols}")

    wide_reg, n_segments = regularize_by_segments(
        wide=wide,
        freq=args.freq,
        max_gap_minutes=args.max_gap_minutes,
        ffill_limit=args.ffill_limit,
    )
    print(f"Segmentos detectados:   {n_segments}")
    print(f"Filas regularizadas:    {len(wide_reg)}")

    aligned_target_series, target_meta = discover_target_series(
        df_long=df_long,
        feature_index=wide_reg.index,
        tolerance_minutes=args.target_tolerance_minutes,
        target_sensor=args.target_sensor,
        target_metric=args.target_metric,
    )
    wide_reg[TARGET_COL] = pd.Series(aligned_target_series).reindex(wide_reg.index)
    print(f"Sensor THM detectado:   {target_meta['selected_target_sensor']}")
    print(f"Métrica THM detectada:  {target_meta['selected_target_metric']}")
    print(f"Alineaciones THM válidas:{target_meta['selected_target_aligned_count']}")

    overlap_audit = audit_feature_target_overlap(
        df_long=df_long,
        feature_sensor=args.sensor,
        feature_index_raw=wide.index,
        selected_target_sensor=target_meta["selected_target_sensor"],
        selected_target_metric=target_meta["selected_target_metric"],
        tolerance_minutes=args.target_tolerance_minutes,
    )
    print("\nAuditoría X-THM:")
    print(f"  CSV con sensor X:                 {overlap_audit['csv_with_feature_sensor']} / {overlap_audit['csv_total']}")
    print(f"  CSV con THM:                      {overlap_audit['csv_with_target_metric']} / {overlap_audit['csv_total']}")
    print(f"  CSV con X + THM:                  {overlap_audit['csv_with_both_feature_and_target']} / {overlap_audit['csv_total']}")
    print(f"  Timestamps X crudos:              {overlap_audit['feature_timestamps_raw']}")
    print(f"  Timestamps THM crudos:            {overlap_audit['target_timestamps_raw']}")
    print(f"  Solape exacto X∩THM:              {overlap_audit['exact_overlap_timestamps']}")
    print(f"  X sin THM exacto:                 {overlap_audit['feature_timestamps_without_exact_target']}")
    print(f"  X alineados con THM (±tol):       {overlap_audit['aligned_timestamps_with_target']}")
    print(f"  X sin THM alineable (±tol):       {overlap_audit['aligned_timestamps_without_target']}")
    print(f"  Rango temporal X útil:            {overlap_audit['feature_time_min']} -> {overlap_audit['feature_time_max']}")
    print(f"  Rango temporal THM útil:          {overlap_audit['target_time_min']} -> {overlap_audit['target_time_max']}")

    scenario_results = []
    for calendar_mode in calendar_modes:
        for sc in incident_scenarios:
            res = run_hgb_variant(
                variant_mode=calendar_mode,
                wide_reg=wide_reg,
                feature_base_cols=feature_base_cols,
                use_history_multi=use_history_multi,
                args=args,
                incident_strategy=sc["name"],
            )
            scenario_results.append(res)

    if incident_defined and args.incident_policy in {"exclude", "compare"}:
        selectable_results = [
            r for r in scenario_results if r["incident_strategy"] == "exclude_strict"
        ]
    else:
        selectable_results = [r for r in scenario_results if r["incident_strategy"] == "include"]
    if not selectable_results:
        raise RuntimeError("No hay escenarios seleccionables tras aplicar la política de incidencias")

    # Selección principal v5.8.8: primero fija el escenario de datos operativo
    # (exclusión de incidencias confirmadas) y después selecciona la variante de calendario.
    primary_res, robust_calendar_selection_meta = select_primary_calendar_variant_robust(
        selectable_results,
        prefer_all_calendar_tolerance=0.05,
    )
    primary_res["robust_calendar_selection_meta"] = robust_calendar_selection_meta

    print(f"Filas con THM alineado: {primary_res['rows_with_target_after_align']}")
    print(f"Filas sin THM alineado: {primary_res['rows_lost_before_model']}")
    print("\nComparativa variantes/calendario:")
    scenario_rows = []
    for res in scenario_results:
        meta = res["incident_meta"]
        acc = res["acceptance"]
        data_scenario_label = "baseline" if res["incident_strategy"] == "include" else res["incident_strategy"]
        row = {
            "variant_mode": res["variant_mode"],
            "variant_label": res["variant_label"],
            "data_scenario": res["incident_strategy"],
            "data_scenario_label": data_scenario_label,
            "selected_model_type": res["selected_model_type"],
            "hgb_accepted": acc["hgb_accepted"],
            "hgb_validation_status": acc["hgb_validation_status"],
            "hgb_validation_warnings": " | ".join(acc.get("validation_warnings", [])),
            "hgb_validation_reason": acc["selected_reason"],
            "best_baseline_type": acc["best_baseline_type"],
            "hgb_val_rmse": res["hgb_val_metrics"]["rmse"],
            "hgb_val_mae": res["hgb_val_metrics"]["mae"],
            "hgb_val_bias": res["hgb_val_metrics"].get("bias", np.nan),
            "hgb_val_bias_ratio": acc["hgb_val_bias_ratio"],
            "dummy_val_rmse": res["dummy_val_metrics"]["rmse"],
            "ridge_val_rmse": res["ridge_val_metrics"]["rmse"],
            "selected_val_rmse": res["selected_val_metrics"]["rmse"],
            "selected_val_mae": res["selected_val_metrics"]["mae"],
            "selected_val_bias": res["selected_val_metrics"].get("bias", np.nan),
            "hgb_test_rmse": res["m_hgb"]["rmse"],
            "hgb_test_mae": res["m_hgb"]["mae"],
            "hgb_test_smape": res["m_hgb"]["smape"],
            "dummy_test_rmse": res["m_dummy"]["rmse"],
            "ridge_test_rmse": res["m_ridge"]["rmse"],
            "selected_test_rmse": res["m_selected"]["rmse"],
            "selected_test_mae": res["m_selected"]["mae"],
            "selected_test_smape": res["m_selected"]["smape"],
            "rows_model": len(res["model_df"]),
            "rows_removed": meta.get("rows_removed", 0),
            "rows_marked": meta.get("rows_marked", 0),
            "incident_weight": 1.0,
            "rows_train": len(res["train_df"]),
            "rows_val": len(res["val_df"]),
            "rows_test": len(res["test_df"]),
            "train_start": res.get("split_meta", {}).get("train_start"),
            "train_end": res.get("split_meta", {}).get("train_end"),
            "val_start": res.get("split_meta", {}).get("val_start"),
            "val_end": res.get("split_meta", {}).get("val_end"),
            "test_start": res.get("split_meta", {}).get("test_start"),
            "test_end": res.get("split_meta", {}).get("test_end"),
            "split_effective_mode": res.get("split_meta", {}).get("effective_split_mode", res.get("split_meta", {}).get("split_mode")),
            "data_window_test_count": res["incident_sub_global"]["count"],
        }
        scenario_rows.append(row)
        status = acc["hgb_validation_status"]
        print(
            f"  {res['variant_label']} / {data_scenario_label}: "
            f"HGB_val_rmse={res['hgb_val_metrics']['rmse']:.6f} | "
            f"best_baseline={acc['best_baseline_type']}({acc['best_baseline_val_rmse']:.6f}) | "
            f"operativo=HGB | "
            f"hgb_test_rmse={res['m_hgb']['rmse']:.6f} | {status}"
        )

    print("\nEscenario principal para artefactos:")
    print(f"  Variante calendario:  {primary_res['variant_label']}")
    if primary_res.get("robust_calendar_selection_meta"):
        print(f"  Selección calendario: {primary_res['robust_calendar_selection_meta'].get('selection_reason')}")
        print(f"  Mejor validación:     {primary_res['robust_calendar_selection_meta'].get('best_by_validation_variant_label')}")
    print(f"  Escenario datos:      baseline operativo")
    print(f"  Modelo operativo:     {primary_res['selected_model_type']}")
    print(f"  Estado validación:    {primary_res['acceptance']['hgb_validation_status']}")
    print(f"  Motivo:              {primary_res['acceptance']['selected_reason']}")
    if primary_res['acceptance'].get('validation_warnings'):
        for warn in primary_res['acceptance']['validation_warnings']:
            print(f"  Aviso:               {warn}")
    print(f"Filas finales modelo:   {len(primary_res['model_df'])}")
    print(f"Filas descartadas por FE/dropna: {primary_res['rows_lost_feature_eng']}")
    target_invalid_total = int(primary_res.get('target_cleaning_meta', {}).get('rows_invalid_target', 0) or 0)
    target_invalid_value_only = max(0, target_invalid_total - int(primary_res.get('rows_lost_before_model', 0) or 0))
    print(f"THM anómalos descartados: {target_invalid_value_only}")
    print(f"Valores X limpiados:    {primary_res.get('feature_cleaning_meta', {}).get('total_values_cleaned', 0)}")
    print(f"Filas X centinela limpiadas: {primary_res.get('feature_cleaning_meta', {}).get('row_sentinel_rows_cleaned', 0)}")
    sentinel_ts = primary_res.get('feature_cleaning_meta', {}).get('row_sentinel_timestamps', [])
    if sentinel_ts:
        print(f"Primeras filas X centinela: {', '.join(sentinel_ts[:5])}")
    print(f"Nº features actuales:   {len(primary_res['current_cols'])}")
    print(f"Nº features HGB:        {len(primary_res['model_feature_cols'])}")
    print(f"Nº features operativo:  {len(primary_res['selected_feature_cols'])}")
    split_meta = primary_res.get("split_meta", {})
    print(f"Split efectivo:         {split_meta.get('effective_split_mode', split_meta.get('split_mode', 'unknown'))}")
    if split_meta.get("fallback_used"):
        print(f"Fallback split:         {split_meta.get('fallback_reason', 'unknown')}")
    print(f"Train periodo:          {split_meta.get('train_start')} -> {split_meta.get('train_end')}")
    print(f"Val periodo:            {split_meta.get('val_start')} -> {split_meta.get('val_end')}")
    print(f"Test periodo:           {split_meta.get('test_start')} -> {split_meta.get('test_end')}")
    print(f"Rows train:             {len(primary_res['train_df'])}")
    print(f"Rows val:               {len(primary_res['val_df'])}")
    print(f"Rows test:              {len(primary_res['test_df'])}")
    print(f"X_train_current:        {(len(primary_res['train_df']), len(primary_res['current_cols']))}")
    print(f"X_train_hgb:            {(len(primary_res['train_df']), len(primary_res['model_feature_cols']))}")
    print(f"Umbral THM alto:        {primary_res['high_threshold']:.6f} (modo={args.high_threshold_mode})")
    print(f"Umbral cambio fuerte:   {primary_res['change_threshold']:.6f}")

    print("\nDistribución THM por split (escenario principal):")
    print_distribution_block("  Train", primary_res["dist_train"])
    print_distribution_block("  Val", primary_res["dist_val"])
    print_distribution_block("  Test", primary_res["dist_test"])

    print("\nMétricas en validación (escenario principal):")
    print_metrics_block("  Dummy median", primary_res["dummy_val_metrics"], indent=4)
    print_metrics_block("  Ridge current-only", primary_res["ridge_val_metrics"], indent=4)
    print_metrics_block("  HGB operativo", primary_res["hgb_val_metrics"], indent=4)
    print_metrics_block("  Modelo operativo HGB", primary_res["selected_val_metrics"], indent=4)
    print("\nControl de calidad HGB:")
    for k, v in primary_res["acceptance"].items():
        print(f"  {k}: {v}")

    print("\nMejor configuración HGB operativo en validación:")
    for k, v in primary_res["best_params"].items():
        print(f"  {k}: {v}")

    print("\n==============================")
    print("MÉTRICAS EN TEST (escenario principal)")
    print("==============================")
    print_metrics_block("Dummy median", primary_res["m_dummy"])
    print_metrics_block("Ridge current-only", primary_res["m_ridge"])
    print_metrics_block("HGB operativo", primary_res["m_hgb"])
    print_metrics_block("Modelo operativo HGB", primary_res["m_selected"])

    print("\nSubconjuntos en test - HGB operativo:")
    print_metrics_block("  THM alto", primary_res["high_sub_global"], indent=4)
    print(f"    count: {primary_res['high_sub_global']['count']}")
    print_metrics_block("  Cambio fuerte", primary_res["change_sub_global"], indent=4)
    print(f"    count: {primary_res['change_sub_global']['count']}")
    print_metrics_block("  Incidente", primary_res["incident_sub_global"], indent=4)
    print(f"    count: {primary_res['incident_sub_global']['count']}")
    print_metrics_block("  No incidente", primary_res["normal_sub_global"], indent=4)
    print(f"    count: {primary_res['normal_sub_global']['count']}")

    print("\nSubconjuntos en test - modelo operativo HGB:")
    print_metrics_block("  THM alto", primary_res["high_sub_selected"], indent=4)
    print(f"    count: {primary_res['high_sub_selected']['count']}")
    print_metrics_block("  Cambio fuerte", primary_res["change_sub_selected"], indent=4)
    print(f"    count: {primary_res['change_sub_selected']['count']}")
    print_metrics_block("  Incidente", primary_res["incident_sub_selected"], indent=4)
    print(f"    count: {primary_res['incident_sub_selected']['count']}")
    print_metrics_block("  No incidente", primary_res["normal_sub_selected"], indent=4)
    print(f"    count: {primary_res['normal_sub_selected']['count']}")

    feat_imp_df, family_imp_df = compute_permutation_importances_df(
        model=primary_res["hgb_global_model"],
        X_val=primary_res["X_val_model"],
        y_val=primary_res["y_val"],
        current_cols=primary_res["current_cols"],
        random_state=args.random_state,
        sample_size=args.importance_sample_size,
        n_repeats=args.importance_n_repeats,
    )
    print("\nTop 20 features por permutation importance (HGB operativo, validación):")
    for _, row in feat_imp_df.head(20).iterrows():
        print(f"  {row['feature']}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")
    print("\nTop 20 familias/base_var por permutation importance (HGB operativo, validación):")
    for _, row in family_imp_df.head(20).iterrows():
        print(f"  {row['base_var']} / {row['family']}: {row['importance_mean']:.6f}")

    union_index = pd.Index(sorted(set().union(*[set(pd.Index(res["test_df"].index)) for res in scenario_results])))
    test_pred_df = pd.DataFrame(index=union_index)

    primary_temp = pd.DataFrame({
        "timestamp": primary_res["test_df"].index,
        "y_true": primary_res["y_test"],
        "pred_dummy": primary_res["test_pred_dummy"],
        "pred_ridge_current": primary_res["test_pred_ridge"],
        "pred_hgb_candidate": primary_res["test_pred_hgb"],
        "pred_selected": primary_res["test_pred_selected"],
        "selected_model_type": primary_res["selected_model_type"],
        "is_high": primary_res["high_mask_test"].astype(int),
        "is_change": primary_res["change_mask_test"].astype(int),
        "is_incident": primary_res["incident_mask_test"].astype(int),
        "abs_change_true": primary_res["test_df"]["thm_change_abs"].to_numpy(dtype=float),
    }).set_index("timestamp")
    test_pred_df = test_pred_df.join(primary_temp, how="left")

    for res in scenario_results:
        suffix = f"{res['variant_label']}_{res['incident_strategy']}"
        scenario_temp = pd.DataFrame({
            "timestamp": res["test_df"].index,
            f"pred_hgb_{suffix}": res["test_pred_hgb"],
            f"pred_selected_{suffix}": res["test_pred_selected"],
            f"in_test_{suffix}": 1,
        }).set_index("timestamp")
        test_pred_df = test_pred_df.join(scenario_temp, how="left")

    for res in scenario_results:
        suffix = f"{res['variant_label']}_{res['incident_strategy']}"
        col = f"in_test_{suffix}"
        if col in test_pred_df.columns:
            test_pred_df[col] = test_pred_df[col].fillna(0).astype(int)

    # Alias estables del escenario operativo para facilitar diagnósticos posteriores.
    # No se usa el sufijo include porque las incidencias confirmadas han sido excluidas.
    for res in selectable_results:
        operational_col = f"pred_hgb_{res['variant_label']}_operational"
        temp = pd.Series(res["test_pred_hgb"], index=res["test_df"].index, name=operational_col)
        test_pred_df = test_pred_df.join(temp, how="left")

    test_pred_df = test_pred_df.reset_index().rename(columns={"index": "timestamp"})
    test_pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    feature_debug_manifest = export_feature_debug_tables(
        output_dir=output_dir,
        primary_res=primary_res,
        feat_imp_df=feat_imp_df,
        top_n_features=80,
    )
    print("\nDebug features exportadas:")
    for fname in feature_debug_manifest.get("created_files", []):
        print(f"  - {(output_dir / fname).as_posix()}")


    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(output_dir / "scenario_model_acceptance_comparison.csv", index=False)
    # Compatibilidad con versiones previas.
    scenario_df.to_csv(output_dir / "scenario_comparison.csv", index=False)
    all_selection_df = pd.concat([r["selection_df"] for r in scenario_results], ignore_index=True)
    all_selection_df.to_csv(output_dir / "model_selection_results.csv", index=False)
    feat_imp_df.to_csv(output_dir / "feature_importance_permutation.csv", index=False)
    family_imp_df.to_csv(output_dir / "feature_importance_aggregated.csv", index=False)

    split_stats_df = pd.DataFrame([
        {"split": "train", **primary_res["dist_train"]},
        {"split": "val", **primary_res["dist_val"]},
        {"split": "test", **primary_res["dist_test"]},
    ])
    split_stats_df.to_csv(output_dir / "split_target_distribution.csv", index=False)

    inference_contract = {
        "objective": "instantaneous_thm_estimation",
        "uses_thm_as_input": False,
        "uses_thm_history_as_input": False,
        "uses_multiparametric_history_as_input": use_history_multi,
        "buffer_multi_required_minutes": buffer_required if not primary_res["selected_current_only"] else 0,
        "current_feature_columns": primary_res["current_cols"],
        "model_feature_columns": primary_res["selected_feature_cols"],
        "selected_feature_columns": primary_res["selected_feature_cols"],
        "hgb_model_feature_columns": primary_res["model_feature_cols"],
        "selected_model_type": primary_res["selected_model_type"],
        "hgb_accepted": primary_res["acceptance"]["hgb_accepted"],
        "target_column": TARGET_COL,
        "target_sensor_selected": target_meta["selected_target_sensor"],
        "target_metric_selected": target_meta["selected_target_metric"],
        "feature_sensor": args.sensor,
        "feature_cleaning_enabled": not args.disable_feature_cleaning,
        "feature_cleaning_meta": primary_res.get("feature_cleaning_meta", {}),
        "target_cleaning_meta": primary_res.get("target_cleaning_meta", {}),
        "calendar_mode_selected": primary_res["variant_mode"],
        "robust_calendar_selection_meta": primary_res.get("robust_calendar_selection_meta", {}),
        "calendar_mode_requested": args.calendar_mode,
        "calendar_modes_compared": calendar_modes,
        "split_mode_requested": args.split_mode,
        "split_meta": primary_res.get("split_meta", {}),
        "val_window_days": float(args.val_window_days),
        "test_window_days": float(args.test_window_days),
        "train_history_days": float(args.train_history_days),
        "incident_policy": args.incident_policy,
        "incident_intervals_csv": args.incident_intervals_csv,
        "incident_intervals": incident_intervals,
        "data_scenarios_compared": [s["label"] for s in incident_scenarios],
        "primary_data_scenario": primary_res["incident_strategy"],
        "incident_intervals": incident_intervals,
        "change_lookback_minutes": int(args.change_lookback),
        "high_threshold_mode": args.high_threshold_mode,
        "high_threshold_train_quantile": float(pd.Series(primary_res["train_df"][TARGET_COL]).quantile(args.high_threshold_quantile)),
        "high_threshold_fixed": args.high_threshold_fixed,
        "high_threshold_effective": primary_res["high_threshold"],
        "change_quantile": args.change_quantile,
        "change_threshold_effective": primary_res["change_threshold"],
        "acceptance": primary_res["acceptance"],
    }
    (output_dir / "current_feature_columns.csv").write_text("\n".join(primary_res["current_cols"]), encoding="utf-8")
    (output_dir / "model_feature_columns.csv").write_text("\n".join(primary_res["selected_feature_cols"]), encoding="utf-8")
    (output_dir / "hgb_model_feature_columns.csv").write_text("\n".join(primary_res["model_feature_cols"]), encoding="utf-8")
    (output_dir / "inference_contract.json").write_text(json.dumps(inference_contract, indent=2, ensure_ascii=False), encoding="utf-8")

    bundle = {
        "model_type": "hgb_global",
        "selected_model_type": primary_res["selected_model_type"],
        "feature_sensor": args.sensor,
        "target_sensor_selected": target_meta["selected_target_sensor"],
        "target_metric_selected": target_meta["selected_target_metric"],
        "uses_history_multi": use_history_multi and not primary_res["selected_current_only"],
        "buffer_multi_required_minutes": buffer_required if not primary_res["selected_current_only"] else 0,
        "current_cols": primary_res["current_cols"],
        "model_feature_cols": primary_res["selected_feature_cols"],
        "selected_feature_cols": primary_res["selected_feature_cols"],
        "hgb_model_feature_cols": primary_res["model_feature_cols"],
        "calendar_mode": primary_res["variant_mode"],
        "robust_calendar_selection_meta": primary_res.get("robust_calendar_selection_meta", {}),
        "calendar_modes_compared": calendar_modes,
        "split_mode_requested": args.split_mode,
        "split_meta": primary_res.get("split_meta", {}),
        "best_params": primary_res["best_params"],
        "high_threshold": primary_res["high_threshold"],
        "change_threshold": primary_res["change_threshold"],
        "feature_cleaning_enabled": not args.disable_feature_cleaning,
        "feature_cleaning_meta": primary_res.get("feature_cleaning_meta", {}),
        "target_cleaning_meta": primary_res.get("target_cleaning_meta", {}),
        # Compatibilidad: global_model apunta al HGB operativo.
        "global_model": primary_res["selected_model"],
        "selected_model": primary_res["selected_model"],
        "hgb_global_model": primary_res["hgb_global_model"],
        "dummy_model": primary_res["dummy_model"],
        "ridge_model": primary_res["ridge_model"],
        "hgb_accepted": primary_res["acceptance"]["hgb_accepted"],
        "acceptance": primary_res["acceptance"],
        "data_scenarios_compared": [s["label"] for s in incident_scenarios],
    }
    joblib.dump(bundle, output_dir / "instant_softsensor_bundle.pkl")
    joblib.dump(primary_res["selected_model"], output_dir / "final_selected_model.pkl")
    joblib.dump(primary_res["hgb_global_model"], output_dir / "final_hgb_global_model.pkl")

    save_simple_plot(
        x=np.arange(len(scenario_df)),
        y=scenario_df["selected_test_rmse"].to_numpy(dtype=float),
        xlabel="Variante / escenario",
        ylabel="RMSE test HGB operativo",
        title="Comparación HGB operativo",
        out_path=output_dir / "selected_model_scenario_comparison_rmse.png",
        xtick_labels=(scenario_df["variant_label"] + " / " + scenario_df["data_scenario_label"]).tolist(),
        rotate_xticks=35,
    )
    save_simple_plot(
        x=np.arange(len(scenario_df)),
        y=scenario_df["hgb_test_rmse"].to_numpy(dtype=float),
        xlabel="Variante / escenario",
        ylabel="RMSE test HGB operativo",
        title="Comparación HGB operativo",
        out_path=output_dir / "hgb_scenario_comparison_rmse.png",
        xtick_labels=(scenario_df["variant_label"] + " / " + scenario_df["data_scenario_label"]).tolist(),
        rotate_xticks=35,
    )
    selection_scores = all_selection_df["score"].to_numpy(dtype=float)
    best_selection_idx = int(np.nanargmin(selection_scores)) if len(selection_scores) else None
    save_simple_plot(
        x=np.arange(len(all_selection_df)),
        y=selection_scores,
        xlabel="Configuración HGB evaluada",
        ylabel="RMSE compuesto de validación",
        title="Selección HGB global",
        out_path=output_dir / "global_model_selection_score.png",
        highlight_index=best_selection_idx,
        highlight_label="Best config",
    )
    save_predictions_plot(
        timestamps=primary_res["test_df"].index,
        y_true=primary_res["y_test"],
        y_pred=primary_res["test_pred_selected"],
        out_path=output_dir / "predictions_vs_real_test.png",
        title="Predicción vs real (HGB operativo)",
    )
    save_predictions_plot(
        timestamps=primary_res["test_df"].index,
        y_true=primary_res["y_test"],
        y_pred=primary_res["test_pred_hgb"],
        out_path=output_dir / "predictions_vs_real_hgb_candidate_test.png",
        title="Predicción vs real (HGB operativo - copia de control)",
    )

    (
        data_quality_report,
        data_quality_summary_df,
        data_quality_feature_df,
        data_quality_target_anomalies_df,
        data_quality_x_sentinel_df,
    ) = build_data_quality_report(primary_res)
    (output_dir / "data_quality_report.json").write_text(
        json.dumps(data_quality_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    data_quality_summary_df.to_csv(output_dir / "data_quality_summary.csv", index=False)
    data_quality_feature_df.to_csv(output_dir / "data_quality_feature_cleaning.csv", index=False)
    data_quality_target_anomalies_df.to_csv(output_dir / "data_quality_target_anomalies.csv", index=False)
    data_quality_x_sentinel_df.to_csv(output_dir / "data_quality_x_sentinel_rows.csv", index=False)

    metadata = {
        "n_csv": n_csv,
        "n_rows_long": int(len(df_long)),
        "feature_sensor": args.sensor,
        "feature_columns_used": feature_base_cols,
        "target_sensor_selected": target_meta["selected_target_sensor"],
        "target_metric_selected": target_meta["selected_target_metric"],
        "target_meta": target_meta,
        "overlap_audit": overlap_audit,
        "buffer_multi_required_minutes": buffer_required,
        "calendar_mode_requested": args.calendar_mode,
        "calendar_modes_compared": calendar_modes,
        "calendar_mode_selected": primary_res["variant_mode"],
        "split_mode_requested": args.split_mode,
        "split_meta": primary_res.get("split_meta", {}),
        "val_window_days": float(args.val_window_days),
        "test_window_days": float(args.test_window_days),
        "train_history_days": float(args.train_history_days),
        "incident_policy": args.incident_policy,
        "incident_intervals_csv": args.incident_intervals_csv,
        "incident_intervals": incident_intervals,
        "scenario_comparison": scenario_rows,
        "primary_data_scenario": primary_res["incident_strategy"],
        "selected_model_type": primary_res["selected_model_type"],
        "acceptance": primary_res["acceptance"],
        "primary_metrics_test_selected": primary_res["m_selected"],
        "primary_metrics_test_hgb_candidate": primary_res["m_hgb"],
        "primary_metrics_test_dummy": primary_res["m_dummy"],
        "primary_metrics_test_ridge": primary_res["m_ridge"],
        "primary_high_subset_test_hgb": primary_res["high_sub_global"],
        "primary_change_subset_test_hgb": primary_res["change_sub_global"],
        "primary_incident_subset_test_hgb": primary_res["incident_sub_global"],
        "primary_high_subset_test_selected": primary_res["high_sub_selected"],
        "primary_change_subset_test_selected": primary_res["change_sub_selected"],
        "primary_incident_subset_test_selected": primary_res["incident_sub_selected"],
        "split_distributions": {
            "train": primary_res["dist_train"],
            "val": primary_res["dist_val"],
            "test": primary_res["dist_test"],
        },
        "best_params_hgb_candidate": primary_res["best_params"],
        "selection_rows_hgb_global": all_selection_df.to_dict(orient="records"),
        "importance_top20_hgb_candidate": feat_imp_df.head(20).to_dict(orient="records"),
        "importance_aggregated_top20_hgb_candidate": family_imp_df.head(20).to_dict(orient="records"),
        "data_quality_report": data_quality_report,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nArtefactos guardados:")
    for fname in [
        "instant_softsensor_bundle.pkl",
        "final_selected_model.pkl",
        "final_hgb_global_model.pkl",
        "inference_contract.json",
        "current_feature_columns.csv",
        "model_feature_columns.csv",
        "hgb_model_feature_columns.csv",
        "scenario_model_acceptance_comparison.csv",
        "scenario_comparison.csv",
        "model_selection_results.csv",
        "feature_importance_permutation.csv",
        "feature_importance_aggregated.csv",
        "split_target_distribution.csv",
        "test_predictions.csv",
        "test_features_debug.csv",
        "test_features_top_debug.csv",
        "test_feature_daily_summary.csv",
        "split_feature_distribution_debug.csv",
        "feature_debug_manifest.json",
        "selected_model_scenario_comparison_rmse.png",
        "hgb_scenario_comparison_rmse.png",
        "global_model_selection_score.png",
        "predictions_vs_real_test.png",
        "predictions_vs_real_hgb_candidate_test.png",
        "data_quality_report.json",
        "data_quality_summary.csv",
        "data_quality_feature_cleaning.csv",
        "data_quality_target_anomalies.csv",
        "data_quality_x_sentinel_rows.csv",
        "metadata.json",
    ]:
        print(f"- {(output_dir / fname).as_posix()}")



def _fmt_metric(value: object) -> str:
    try:
        value_float = float(value)
        if np.isnan(value_float):
            return "nan"
        return f"{value_float:.3f}"
    except Exception:
        return "n/a"


def print_compact_training_summary(args: argparse.Namespace, verbose_log_path: Path) -> None:
    output_dir = Path(args.output_dir)
    metadata_path = output_dir / "metadata.json"
    scenario_path = output_dir / "scenario_comparison.csv"
    quality_path = output_dir / "data_quality_report.json"

    metadata: Dict[str, object] = {}
    quality: Dict[str, object] = {}
    scenario_df = pd.DataFrame()

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if scenario_path.exists():
        scenario_df = pd.read_csv(scenario_path)

    split_meta = metadata.get("split_meta", {}) if isinstance(metadata, dict) else {}
    acceptance = metadata.get("acceptance", {}) if isinstance(metadata, dict) else {}
    overlap = metadata.get("overlap_audit", {}) if isinstance(metadata, dict) else {}
    target_meta = metadata.get("target_meta", {}) if isinstance(metadata, dict) else {}

    selected_variant_raw = metadata.get("calendar_mode_selected", "n/a")
    selected_variant = {"all": "all_calendar", "no_doy": "no_doy", "no_calendar": "no_calendar"}.get(
        selected_variant_raw, selected_variant_raw
    )
    selected_model = metadata.get("selected_model_type", "n/a")
    test_metrics = metadata.get("primary_metrics_test_selected", {})
    ridge_metrics = metadata.get("primary_metrics_test_ridge", {})
    dummy_metrics = metadata.get("primary_metrics_test_dummy", {})

    print("\n========================================")
    print("THM v5.8.8 | RESUMEN DE ENTRENAMIENTO")
    print("========================================")
    print(f"Estado:              {acceptance.get('hgb_validation_status', 'n/a')}")
    print(f"Modelo seleccionado: {selected_model} / {selected_variant}")
    print(f"Salida:              {output_dir.resolve()}")
    print(f"Filtro THM:          ({args.min_valid_target}, {args.max_valid_target if args.max_valid_target is not None else 'sin máximo'}]")
    intervals = metadata.get("incident_intervals", []) if isinstance(metadata, dict) else []
    primary_scenario = metadata.get("primary_data_scenario", "include") if isinstance(metadata, dict) else "include"
    print(f"Escenario datos:     {primary_scenario}")
    print(f"Incidencias excl.:   {len(intervals) if isinstance(intervals, list) else 0}")
    if isinstance(intervals, list):
        for interval in intervals:
            print(
                "  - "
                f"{interval.get('label', 'incidencia')}: "
                f"{interval.get('effective_start', 'n/a')} -> {interval.get('effective_end', 'n/a')}"
            )

    warnings = acceptance.get("validation_warnings", []) if isinstance(acceptance, dict) else []
    for warning in warnings:
        print(f"AVISO:               {warning}")

    try:
        selected_test_rmse = float(test_metrics.get("rmse", np.nan))
        ridge_test_rmse = float(ridge_metrics.get("rmse", np.nan))
        if np.isfinite(selected_test_rmse) and np.isfinite(ridge_test_rmse) and selected_test_rmse > ridge_test_rmse:
            print(
                "AVISO TEST:          "
                f"Ridge supera al HGB en test (RMSE {ridge_test_rmse:.3f} < {selected_test_rmse:.3f})"
            )
    except Exception:
        pass

    print("\nDatos")
    print(f"  CSV leídos:         {metadata.get('n_csv', 'n/a')}")
    print(f"  Filas brutas:       {metadata.get('n_rows_long', 'n/a')}")
    print(f"  X útil:             {overlap.get('feature_time_min', 'n/a')} -> {overlap.get('feature_time_max', 'n/a')}")
    print(f"  THM útil:           {overlap.get('target_time_min', 'n/a')} -> {overlap.get('target_time_max', 'n/a')}")
    print(f"  THM alineados:      {target_meta.get('selected_target_aligned_count', 'n/a')}")

    print("\nSplit temporal")
    print(f"  Train: {split_meta.get('train_start', 'n/a')} -> {split_meta.get('train_end', 'n/a')} ({split_meta.get('train_rows', 'n/a')} filas)")
    print(f"  Val:   {split_meta.get('val_start', 'n/a')} -> {split_meta.get('val_end', 'n/a')} ({split_meta.get('val_rows', 'n/a')} filas)")
    print(f"  Test:  {split_meta.get('test_start', 'n/a')} -> {split_meta.get('test_end', 'n/a')} ({split_meta.get('test_rows', 'n/a')} filas)")

    if not scenario_df.empty:
        preferred_scenario = metadata.get("primary_data_scenario", "include") if isinstance(metadata, dict) else "include"
        baseline = scenario_df[scenario_df["data_scenario"] == preferred_scenario].copy()
        if baseline.empty:
            baseline = scenario_df.copy()
        print("\nComparativa HGB")
        print("  variante             RMSE val   RMSE test   bias val   estado")
        for _, row in baseline.iterrows():
            print(
                f"  {str(row.get('variant_label', 'n/a')):<20} "
                f"{_fmt_metric(row.get('hgb_val_rmse')):>8}   "
                f"{_fmt_metric(row.get('hgb_test_rmse')):>9}   "
                f"{_fmt_metric(row.get('hgb_val_bias')):>8}   "
                f"{str(row.get('hgb_validation_status', 'n/a'))}"
            )

    print("\nMétricas test seleccionadas")
    print(
        "  HGB:   "
        f"MAE={_fmt_metric(test_metrics.get('mae'))} | "
        f"RMSE={_fmt_metric(test_metrics.get('rmse'))} | "
        f"bias={_fmt_metric(test_metrics.get('bias'))} | "
        f"R²={_fmt_metric(test_metrics.get('r2'))}"
    )
    print(
        "  Ridge: "
        f"MAE={_fmt_metric(ridge_metrics.get('mae'))} | "
        f"RMSE={_fmt_metric(ridge_metrics.get('rmse'))} | "
        f"bias={_fmt_metric(ridge_metrics.get('bias'))}"
    )
    print(
        "  Dummy: "
        f"MAE={_fmt_metric(dummy_metrics.get('mae'))} | "
        f"RMSE={_fmt_metric(dummy_metrics.get('rmse'))} | "
        f"bias={_fmt_metric(dummy_metrics.get('bias'))}"
    )

    target_quality = quality.get("target_cleaning", {}) if isinstance(quality, dict) else {}
    feature_quality = quality.get("feature_cleaning", {}) if isinstance(quality, dict) else {}
    print("\nCalidad de datos")
    print(f"  THM anómalos:        {target_quality.get('rows_invalid_due_to_value_rules', 'consultar CSV')}")
    print(f"  Valores X limpiados: {feature_quality.get('total_values_cleaned', 'consultar CSV')}")
    print(f"  Filas centinela X:   {feature_quality.get('row_sentinel_rows_cleaned', 'consultar CSV')}")

    importance_path = output_dir / "feature_importance_aggregated.csv"
    if importance_path.exists():
        imp = pd.read_csv(importance_path).head(5)
        if not imp.empty:
            print("\nTop 5 familias de features")
            for _, row in imp.iterrows():
                print(f"  {row.get('base_var', 'n/a')} / {row.get('family', 'n/a')}: {_fmt_metric(row.get('importance_mean'))}")

    print("\nArchivos clave")
    for name in [
        "final_selected_model.pkl",
        "instant_softsensor_bundle.pkl",
        "test_predictions.csv",
        "scenario_comparison.csv",
        "data_quality_target_anomalies.csv",
        "incident_intervals_applied.csv",
        "metadata.json",
    ]:
        path = output_dir / name
        if path.exists():
            print(f"  - {path.as_posix()}")
    print(f"  - Log detallado: {verbose_log_path.as_posix()}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_level == "verbose":
        run_training(args)
        return

    verbose_log_path = output_dir / "training_verbose.log"
    buffer = io.StringIO()

    print("Iniciando entrenamiento THM v5.8.8...")
    print(f"Entrada: {Path(args.csv_dir).resolve()}")
    print(f"Salida:  {output_dir.resolve()}")
    print("El detalle completo se guardará en training_verbose.log.")

    try:
        with contextlib.redirect_stdout(buffer):
            run_training(args)
    except Exception:
        verbose_log_path.write_text(buffer.getvalue(), encoding="utf-8")
        print(f"\nERROR: el entrenamiento no finalizó. Log parcial: {verbose_log_path}")
        raise

    verbose_log_path.write_text(buffer.getvalue(), encoding="utf-8")
    print_compact_training_summary(args, verbose_log_path)


if __name__ == "__main__":
    main()
