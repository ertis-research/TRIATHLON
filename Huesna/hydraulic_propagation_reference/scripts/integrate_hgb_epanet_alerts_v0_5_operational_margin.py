from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "0.5.0"
LEVEL_TO_RANK = {"not_classified": -1, "normal": 0, "prealerta": 1, "alerta": 2}
RANK_TO_LEVEL = {-1: "not_classified", 0: "normal", 1: "prealerta", 2: "alerta"}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el fichero: {path}")
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\xa0", " ").strip()


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {clean_text(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lookup.get(clean_text(candidate).lower())
        if found is not None:
            return found
    return None


def classify_values(
    values: pd.Series,
    prealert: pd.Series | float,
    alert: pd.Series | float,
) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    pre = pd.to_numeric(prealert, errors="coerce") if isinstance(prealert, pd.Series) else float(prealert)
    alt = pd.to_numeric(alert, errors="coerce") if isinstance(alert, pd.Series) else float(alert)

    out = pd.Series("not_classified", index=values.index, dtype="object")
    valid = x.notna()
    if isinstance(pre, pd.Series):
        valid &= pre.notna()
    if isinstance(alt, pd.Series):
        valid &= alt.notna()

    out.loc[valid] = "normal"
    out.loc[valid & (x >= pre)] = "prealerta"
    out.loc[valid & (x >= alt)] = "alerta"
    return out


def level_to_rank(level: pd.Series) -> pd.Series:
    return level.map(LEVEL_TO_RANK).fillna(-1).astype(int)


def rank_to_level(rank: pd.Series) -> pd.Series:
    return rank.map(RANK_TO_LEVEL).fillna("not_classified")


def normalize_targets(raw: pd.DataFrame, reference_delay_col: str) -> pd.DataFrame:
    order_col = first_existing_col(raw, ["orden", "order"])
    ref_col = first_existing_col(raw, ["reference_name", "red", "node_name"])
    node_col = first_existing_col(raw, ["node_id", "mapped_node_id"])
    node_name_col = first_existing_col(raw, ["node_name", "reference_name", "red"])
    pre_col = first_existing_col(raw, ["prealert_threshold", "prealerta_threshold", "prealert"])
    alert_col = first_existing_col(raw, ["alert_threshold", "alerta_threshold", "alert"])
    zone_col = first_existing_col(raw, ["alert_zone", "zone"])
    source_zone_col = first_existing_col(raw, ["threshold_source_zone", "source_zone"])
    label_col = first_existing_col(raw, ["alert_zone_label", "zone_label"])

    delay_col = reference_delay_col
    if delay_col not in raw.columns:
        delay_col = first_existing_col(
            raw,
            [reference_delay_col, "delay_h", "tiempo_total_h", "tiempo_recorrido_h"],
        )

    missing = []
    if ref_col is None:
        missing.append("reference_name/red")
    if node_col is None:
        missing.append("node_id/mapped_node_id")
    if pre_col is None:
        missing.append("prealert_threshold")
    if alert_col is None:
        missing.append("alert_threshold")
    if delay_col is None:
        missing.append(reference_delay_col)
    if missing:
        raise ValueError(f"Faltan columnas en targets: {missing}. Columnas: {list(raw.columns)}")

    out = pd.DataFrame(index=raw.index)
    out["orden"] = (
        pd.to_numeric(raw[order_col], errors="coerce")
        if order_col is not None
        else np.arange(1, len(raw) + 1, dtype=float)
    )
    out["reference_name"] = raw[ref_col].map(clean_text)
    out["node_id"] = raw[node_col].map(clean_text)
    out["node_name"] = (
        raw[node_name_col].map(clean_text)
        if node_name_col is not None
        else out["reference_name"]
    )
    out["delay_h"] = pd.to_numeric(raw[delay_col], errors="coerce")
    out["delay_source_col"] = str(delay_col)
    out["alert_zone"] = raw[zone_col].map(clean_text) if zone_col is not None else ""
    out["threshold_source_zone"] = (
        raw[source_zone_col].map(clean_text)
        if source_zone_col is not None
        else out["alert_zone"]
    )
    out["alert_zone_label"] = (
        raw[label_col].map(clean_text)
        if label_col is not None
        else out["alert_zone"]
    )
    out["prealert_threshold"] = pd.to_numeric(raw[pre_col], errors="coerce")
    out["alert_threshold"] = pd.to_numeric(raw[alert_col], errors="coerce")

    negative_delay = out["delay_h"].notna() & (out["delay_h"] < 0)
    invalid_threshold_order = (
        out["prealert_threshold"].notna()
        & out["alert_threshold"].notna()
        & (out["prealert_threshold"] >= out["alert_threshold"])
    )
    if negative_delay.any():
        bad = out.loc[negative_delay, ["reference_name", "node_id", "delay_h"]]
        raise ValueError(f"Hay retardos negativos en targets:\n{bad.to_string(index=False)}")
    if invalid_threshold_order.any():
        bad = out.loc[
            invalid_threshold_order,
            ["reference_name", "node_id", "prealert_threshold", "alert_threshold"],
        ]
        raise ValueError(f"Hay umbrales inválidos (prealerta >= alerta):\n{bad.to_string(index=False)}")

    raw_count = len(out)
    valid = (
        out["node_id"].ne("")
        & out["delay_h"].notna()
        & np.isfinite(out["delay_h"])
        & out["prealert_threshold"].notna()
        & np.isfinite(out["prealert_threshold"])
        & out["alert_threshold"].notna()
        & np.isfinite(out["alert_threshold"])
    )
    out = out.loc[valid].copy()

    duplicate_key = [
        "orden",
        "reference_name",
        "node_id",
        "delay_h",
        "alert_zone",
        "prealert_threshold",
        "alert_threshold",
    ]
    before_dedup = len(out)
    out = out.drop_duplicates(duplicate_key, keep="first").copy()

    out["orden_int"] = out["orden"].fillna(-1).astype(int)
    out["target_id"] = (
        "ref_"
        + out["orden_int"].map(lambda value: f"{value:03d}" if value >= 0 else "NA")
        + "__"
        + out["node_id"]
        + "__"
        + out["reference_name"]
    )
    out = out.drop(columns=["orden_int"])
    out = out.sort_values(["orden", "node_id", "reference_name"]).reset_index(drop=True)
    out.attrs["n_raw_rows"] = int(raw_count)
    out.attrs["n_dropped_invalid_rows"] = int(raw_count - before_dedup)
    out.attrs["n_dropped_duplicate_rows"] = int(before_dedup - len(out))
    return out


def load_targets(
    epanet_dir: Optional[Path],
    targets_csv: Optional[Path],
    reference_delay_col: str,
) -> pd.DataFrame:
    if targets_csv is not None:
        source = targets_csv.resolve()
        raw = read_csv_flexible(source)
    else:
        if epanet_dir is None:
            raise ValueError("Debe indicarse --epanet-dir o --targets-csv")
        directory = epanet_dir.resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"No existe --epanet-dir: {directory}")
        candidates = [
            directory / "mapped_reference_rows_with_alert_zone.csv",
            directory / "comparison_travel_times_with_alert_zones.csv",
            directory / "targets_mapped_references.csv",
        ]
        existing = [candidate for candidate in candidates if candidate.is_file()]
        if not existing:
            raise FileNotFoundError(
                "No se encontró un CSV de referencias mapeadas con zona de alerta en --epanet-dir"
            )
        source = existing[0]
        raw = read_csv_flexible(source)

    targets = normalize_targets(raw, reference_delay_col)
    if targets.empty:
        raise ValueError("No hay targets mapeados y clasificables después de validar la entrada")
    targets.attrs["source_path"] = str(source)
    return targets


def prepare_predictions(
    pred_df: pd.DataFrame,
    time_col: str,
    pred_col: str,
    control_col: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = [time_col, pred_col]
    if control_col:
        required.append(control_col)
    missing = [column for column in required if column not in pred_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en predictions CSV: {missing}. Disponibles: {list(pred_df.columns)}")

    out = pred_df[required].copy()
    n_input = len(out)
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    if control_col:
        out[control_col] = pd.to_numeric(out[control_col], errors="coerce")

    out = out[out[time_col].notna()].sort_values(time_col).reset_index(drop=True)
    n_invalid_time = n_input - len(out)
    n_duplicate_time = int(out.duplicated(time_col, keep="last").sum())
    if n_duplicate_time:
        out = out.drop_duplicates(time_col, keep="last").reset_index(drop=True)

    if out.empty:
        raise ValueError("No quedan timestamps válidos en predictions CSV")
    if out[pred_col].notna().sum() == 0:
        raise ValueError(f"La columna de predicción '{pred_col}' no contiene valores numéricos válidos")

    return out, {
        "n_input_rows": int(n_input),
        "n_invalid_timestamp_rows": int(n_invalid_time),
        "n_duplicate_timestamp_rows_kept_last": int(n_duplicate_time),
    }


def numeric_datetime_ns(series: pd.Series) -> np.ndarray:
    return series.astype("datetime64[ns]").astype("int64").to_numpy()


def interpolate_numeric_series(
    source_times: pd.Series,
    source_values: pd.Series,
    query_times: pd.Series,
    method: str,
    max_gap_h: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(source_times, errors="coerce"),
            "value": pd.to_numeric(source_values, errors="coerce"),
        }
    ).dropna().sort_values("timestamp")
    source = source.drop_duplicates("timestamp", keep="last")

    n_query = len(query_times)
    values = np.full(n_query, np.nan, dtype=float)
    in_range = np.zeros(n_query, dtype=bool)
    nearest_gap_h = np.full(n_query, np.nan, dtype=float)
    interpolation_ok = np.zeros(n_query, dtype=bool)
    if source.empty:
        return values, in_range, nearest_gap_h, interpolation_ok

    src_x = numeric_datetime_ns(source["timestamp"])
    src_y = source["value"].to_numpy(dtype=float)
    query = pd.to_datetime(query_times, errors="coerce")
    valid_query = query.notna().to_numpy()
    query_x = query.astype("datetime64[ns]").astype("int64").to_numpy()

    in_range = valid_query & (query_x >= src_x.min()) & (query_x <= src_x.max())
    right_idx = np.searchsorted(src_x, query_x, side="left")
    left_idx = np.clip(right_idx - 1, 0, len(src_x) - 1)
    right_idx_clip = np.clip(right_idx, 0, len(src_x) - 1)
    left_gap_ns = np.abs(query_x - src_x[left_idx])
    right_gap_ns = np.abs(src_x[right_idx_clip] - query_x)
    nearest_gap_h = np.minimum(left_gap_ns, right_gap_ns) / 1e9 / 3600.0
    nearest_gap_h[~valid_query] = np.nan

    if method == "linear":
        interpolated = np.interp(query_x.astype(float), src_x.astype(float), src_y)
        values[in_range] = interpolated[in_range]
    elif method == "previous":
        previous_idx = np.searchsorted(src_x, query_x, side="right") - 1
        ok = in_range & (previous_idx >= 0)
        values[ok] = src_y[previous_idx[ok]]
    elif method == "nearest":
        choose_right = right_gap_ns < left_gap_ns
        nearest_idx = np.where(choose_right, right_idx_clip, left_idx)
        values[in_range] = src_y[nearest_idx[in_range]]
    else:
        raise ValueError(f"Método de interpolación no válido: {method}")

    interpolation_ok = in_range & np.isfinite(values)
    if max_gap_h is not None:
        interpolation_ok &= nearest_gap_h <= max_gap_h
        values[~interpolation_ok] = np.nan
    return values, in_range, nearest_gap_h, interpolation_ok


def build_forecast_long(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    time_col: str,
    pred_col: str,
    control_col: Optional[str],
    chunk_targets: int,
) -> pd.DataFrame:
    base_columns = [time_col, pred_col] + ([control_col] if control_col else [])
    base = predictions[base_columns].rename(columns={time_col: "source_timestamp"})
    frames: List[pd.DataFrame] = []
    n_source = len(base)

    for start in range(0, len(targets), max(1, chunk_targets)):
        chunk = targets.iloc[start : start + max(1, chunk_targets)].reset_index(drop=True)
        repeated_targets = chunk.loc[chunk.index.repeat(n_source)].reset_index(drop=True)
        repeated_source = pd.concat([base] * len(chunk), ignore_index=True)
        block = pd.concat([repeated_targets, repeated_source], axis=1)
        block["timestamp"] = block["source_timestamp"] + pd.to_timedelta(block["delay_h"], unit="h")
        block["source_in_temporal_range"] = True
        block["nearest_source_gap_h"] = 0.0
        block["interpolation_ok"] = block[pred_col].notna()
        block["propagation_method"] = "forecast_destination_time_equals_source_time_plus_delay"
        frames.append(block)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_nowcast_long(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    time_col: str,
    pred_col: str,
    control_col: Optional[str],
    interpolation: str,
    max_gap_h: Optional[float],
    chunk_targets: int,
) -> pd.DataFrame:
    destination_times = predictions[time_col].reset_index(drop=True)
    frames: List[pd.DataFrame] = []
    n_times = len(destination_times)

    for start in range(0, len(targets), max(1, chunk_targets)):
        chunk = targets.iloc[start : start + max(1, chunk_targets)].reset_index(drop=True)
        repeated_targets = chunk.loc[chunk.index.repeat(n_times)].reset_index(drop=True)
        block = repeated_targets.copy()
        block["timestamp"] = np.tile(destination_times.to_numpy(), len(chunk))
        block["source_timestamp"] = block["timestamp"] - pd.to_timedelta(block["delay_h"], unit="h")

        pred_values, in_range, nearest_gap_h, interpolation_ok = interpolate_numeric_series(
            predictions[time_col],
            predictions[pred_col],
            block["source_timestamp"],
            interpolation,
            max_gap_h,
        )
        block[pred_col] = pred_values
        block["source_in_temporal_range"] = in_range
        block["nearest_source_gap_h"] = nearest_gap_h
        block["interpolation_ok"] = interpolation_ok
        block["propagation_method"] = f"nowcast_source_time_interpolation_{interpolation}"

        if control_col:
            control_values, _, _, _ = interpolate_numeric_series(
                predictions[time_col],
                predictions[control_col],
                block["source_timestamp"],
                interpolation,
                max_gap_h,
            )
            block[control_col] = control_values
        frames.append(block)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_operational_long(
    pred_df: pd.DataFrame,
    targets: pd.DataFrame,
    time_col: str,
    pred_col: str,
    prediction_label: str,
    safety_margin: float,
    time_shift_mode: str,
    control_col: Optional[str],
    interpolation: str,
    max_gap_h: Optional[float],
    chunk_targets: int,
) -> pd.DataFrame:
    use_control = bool(control_col and control_col in pred_df.columns)
    effective_control = control_col if use_control else None

    if time_shift_mode == "forecast":
        long = build_forecast_long(
            pred_df,
            targets,
            time_col,
            pred_col,
            effective_control,
            chunk_targets,
        )
    elif time_shift_mode == "nowcast":
        long = build_nowcast_long(
            pred_df,
            targets,
            time_col,
            pred_col,
            effective_control,
            interpolation,
            max_gap_h,
            chunk_targets,
        )
    else:
        raise ValueError("--time-shift-mode debe ser forecast o nowcast")

    long["prediction_label"] = prediction_label
    long["safety_margin"] = float(safety_margin)
    long["thm_predicted"] = pd.to_numeric(long[pred_col], errors="coerce")
    long["thm_predicted_upper"] = long["thm_predicted"] + float(safety_margin)
    long["alert_level_model"] = classify_values(
        long["thm_predicted"], long["prealert_threshold"], long["alert_threshold"]
    )
    long["alert_level_conservative"] = classify_values(
        long["thm_predicted_upper"], long["prealert_threshold"], long["alert_threshold"]
    )
    long["margin_to_prealert_model"] = long["thm_predicted"] - long["prealert_threshold"]
    long["margin_to_alert_model"] = long["thm_predicted"] - long["alert_threshold"]
    long["margin_to_prealert_conservative"] = long["thm_predicted_upper"] - long["prealert_threshold"]
    long["margin_to_alert_conservative"] = long["thm_predicted_upper"] - long["alert_threshold"]
    long["classification_status"] = np.where(
        long["alert_level_conservative"].eq("not_classified"),
        "not_classified",
        "classified",
    )

    if effective_control:
        long["thm_control"] = pd.to_numeric(long[effective_control], errors="coerce")
        long["alert_level_control"] = classify_values(
            long["thm_control"], long["prealert_threshold"], long["alert_threshold"]
        )
        rank_control = level_to_rank(long["alert_level_control"])
        rank_prediction = level_to_rank(long["alert_level_conservative"])
        valid = (rank_control >= 0) & (rank_prediction >= 0)
        long["alert_match_conservative_vs_control"] = valid & (rank_prediction == rank_control)
        long["missed_prealert_or_alert"] = valid & (rank_control >= 1) & (rank_prediction == 0)
        long["false_prealert_or_alert"] = valid & (rank_control == 0) & (rank_prediction >= 1)
        long["underclassified"] = valid & (rank_prediction < rank_control)
        long["overclassified"] = valid & (rank_prediction > rank_control)
        long["error_upper_minus_control"] = long["thm_predicted_upper"] - long["thm_control"]

    preferred = [
        "timestamp",
        "source_timestamp",
        "target_id",
        "orden",
        "reference_name",
        "node_id",
        "node_name",
        "delay_h",
        "delay_source_col",
        "alert_zone",
        "threshold_source_zone",
        "alert_zone_label",
        "prealert_threshold",
        "alert_threshold",
        "prediction_label",
        "thm_predicted",
        "safety_margin",
        "thm_predicted_upper",
        "alert_level_model",
        "alert_level_conservative",
        "margin_to_prealert_model",
        "margin_to_alert_model",
        "margin_to_prealert_conservative",
        "margin_to_alert_conservative",
        "classification_status",
        "propagation_method",
        "source_in_temporal_range",
        "nearest_source_gap_h",
        "interpolation_ok",
        "thm_control",
        "alert_level_control",
        "alert_match_conservative_vs_control",
        "missed_prealert_or_alert",
        "false_prealert_or_alert",
        "underclassified",
        "overclassified",
        "error_upper_minus_control",
    ]
    columns = [column for column in preferred if column in long.columns]
    rest = [column for column in long.columns if column not in columns]
    return long[columns + rest].sort_values(["target_id", "timestamp"]).reset_index(drop=True)


def safe_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else math.nan


def summarize_alerts(df: pd.DataFrame, group_cols: Sequence[str], level_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["_is_classified"] = tmp[level_col].ne("not_classified")
    tmp["_is_normal"] = tmp[level_col].eq("normal")
    tmp["_is_prealerta"] = tmp[level_col].eq("prealerta")
    tmp["_is_alerta"] = tmp[level_col].eq("alerta")

    aggregations: Dict[str, Tuple[str, str]] = {
        "n_rows": (level_col, "size"),
        "n_classified": ("_is_classified", "sum"),
        "n_normal": ("_is_normal", "sum"),
        "n_prealerta": ("_is_prealerta", "sum"),
        "n_alerta": ("_is_alerta", "sum"),
        "max_thm_predicted": ("thm_predicted", "max"),
        "max_thm_predicted_upper": ("thm_predicted_upper", "max"),
    }

    if "alert_level_control" in tmp.columns:
        tmp["_control_normal"] = tmp["alert_level_control"].eq("normal")
        tmp["_control_prealerta"] = tmp["alert_level_control"].eq("prealerta")
        tmp["_control_alerta"] = tmp["alert_level_control"].eq("alerta")
        aggregations.update(
            {
                "n_control_normal": ("_control_normal", "sum"),
                "n_control_prealerta": ("_control_prealerta", "sum"),
                "n_control_alerta": ("_control_alerta", "sum"),
                "n_missed_prealert_or_alert": ("missed_prealert_or_alert", "sum"),
                "n_false_prealert_or_alert": ("false_prealert_or_alert", "sum"),
            }
        )

    out = (
        tmp.groupby(list(group_cols), dropna=False, sort=True)
        .agg(**aggregations)
        .reset_index()
    )
    count_columns = [column for column in out.columns if column.startswith("n_")]
    for column in count_columns:
        out[column] = out[column].astype(int)
    return out



def summarize_by_node_conservative(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tmp = df.copy()
    tmp["rank_conservative"] = level_to_rank(tmp["alert_level_conservative"])
    aggregations: Dict[str, Tuple[str, str | Any]] = {
        "rank_conservative": ("rank_conservative", "max"),
        "thm_predicted": ("thm_predicted", "max"),
        "thm_predicted_upper": ("thm_predicted_upper", "max"),
        "safety_margin": ("safety_margin", "max"),
        "prealert_threshold": ("prealert_threshold", "min"),
        "alert_threshold": ("alert_threshold", "min"),
        "alert_zone": ("alert_zone", lambda values: "|".join(sorted(set(map(str, values))))),
        "threshold_source_zone": (
            "threshold_source_zone",
            lambda values: "|".join(sorted(set(map(str, values)))),
        ),
    }
    node_ts = (
        tmp.groupby(["node_id", "node_name", "timestamp"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )
    node_ts["alert_level_conservative"] = rank_to_level(node_ts["rank_conservative"])
    node_summary = summarize_alerts(
        node_ts,
        [
            "node_id",
            "node_name",
            "alert_zone",
            "threshold_source_zone",
            "prealert_threshold",
            "alert_threshold",
        ],
        "alert_level_conservative",
    )
    return node_summary, node_ts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Integración HGB + retardos EPANET + alertas v{SCRIPT_VERSION}"
    )
    parser.add_argument("--predictions-csv", required=True, type=Path)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--epanet-dir", type=Path)
    source_group.add_argument("--targets-csv", type=Path)
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--pred-col", default="pred_selected")
    parser.add_argument("--prediction-label", default="pred")
    parser.add_argument(
        "--control-col",
        default=None,
        help="Opcional y solo para evaluación offline; representa el control en el origen.",
    )
    parser.add_argument("--reference-delay-col", default="tiempo_total_h")
    parser.add_argument("--time-shift-mode", choices=["forecast", "nowcast"], default="forecast")
    parser.add_argument(
        "--interpolation",
        choices=["linear", "previous", "nearest"],
        default="linear",
        help="Método usado únicamente en modo nowcast.",
    )
    parser.add_argument(
        "--max-gap-h",
        type=float,
        default=1.0,
        help="Máxima distancia a una muestra de origen en nowcast. Usa un valor negativo para desactivarla.",
    )
    parser.add_argument("--safety-margin", type=float, default=4.0)
    parser.add_argument("--chunk-targets", type=int, default=25)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not math.isfinite(args.safety_margin) or args.safety_margin < 0:
        raise ValueError("--safety-margin debe ser un número finito y no negativo")
    if args.max_gap_h is not None and not math.isfinite(args.max_gap_h):
        raise ValueError("--max-gap-h debe ser finito")
    max_gap_h = None if args.max_gap_h is not None and args.max_gap_h < 0 else args.max_gap_h
    if args.chunk_targets < 1:
        raise ValueError("--chunk-targets debe ser >= 1")

    pred_path = args.predictions_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_raw = read_csv_flexible(pred_path)
    pred_df, prediction_cleaning = prepare_predictions(
        pred_raw,
        args.time_col,
        args.pred_col,
        args.control_col,
    )
    targets = load_targets(args.epanet_dir, args.targets_csv, args.reference_delay_col)

    long = build_operational_long(
        pred_df=pred_df,
        targets=targets,
        time_col=args.time_col,
        pred_col=args.pred_col,
        prediction_label=args.prediction_label,
        safety_margin=args.safety_margin,
        time_shift_mode=args.time_shift_mode,
        control_col=args.control_col,
        interpolation=args.interpolation,
        max_gap_h=max_gap_h,
        chunk_targets=args.chunk_targets,
    )
    if long.empty:
        raise ValueError("La integración no ha generado filas")

    target_group_cols = [
        "target_id",
        "orden",
        "reference_name",
        "node_id",
        "node_name",
        "delay_h",
        "alert_zone",
        "threshold_source_zone",
        "prealert_threshold",
        "alert_threshold",
    ]
    by_target_model = summarize_alerts(long, target_group_cols, "alert_level_model")
    by_target_conservative = summarize_alerts(long, target_group_cols, "alert_level_conservative")
    by_timestamp_conservative = summarize_alerts(long, ["timestamp"], "alert_level_conservative")
    by_node_conservative, node_ts = summarize_by_node_conservative(long)

    safe_to_csv(targets, output_dir / "targets_mapped_references.csv")
    safe_to_csv(long, output_dir / "thm_alerts_operational_mapped_references.csv")
    safe_to_csv(by_target_model, output_dir / "alert_summary_by_target_model_mapped_references.csv")
    safe_to_csv(
        by_target_conservative,
        output_dir / "alert_summary_by_target_conservative_mapped_references.csv",
    )
    safe_to_csv(
        by_node_conservative,
        output_dir / "alert_summary_by_node_conservative_mapped_references.csv",
    )
    safe_to_csv(
        by_timestamp_conservative,
        output_dir / "alert_summary_by_timestamp_conservative_mapped_references.csv",
    )
    safe_to_csv(node_ts, output_dir / "alert_node_timestamp_conservative_mapped_references.csv")

    counts_model = {str(key): int(value) for key, value in long["alert_level_model"].value_counts(dropna=False).to_dict().items()}
    counts_conservative = {
        str(key): int(value)
        for key, value in long["alert_level_conservative"].value_counts(dropna=False).to_dict().items()
    }
    report = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictions_csv": str(pred_path),
        "targets_source": targets.attrs.get("source_path"),
        "output_dir": str(output_dir),
        "pred_col": args.pred_col,
        "prediction_label": args.prediction_label,
        "control_col": args.control_col,
        "reference_delay_col": args.reference_delay_col,
        "time_shift_mode": args.time_shift_mode,
        "interpolation": args.interpolation if args.time_shift_mode == "nowcast" else None,
        "max_gap_h": max_gap_h if args.time_shift_mode == "nowcast" else None,
        "safety_margin": float(args.safety_margin),
        "prediction_cleaning": prediction_cleaning,
        "target_cleaning": {
            "n_raw_rows": targets.attrs.get("n_raw_rows"),
            "n_dropped_invalid_rows": targets.attrs.get("n_dropped_invalid_rows"),
            "n_dropped_duplicate_rows": targets.attrs.get("n_dropped_duplicate_rows"),
        },
        "n_prediction_rows": int(len(pred_df)),
        "n_targets": int(len(targets)),
        "n_unique_nodes": int(targets["node_id"].nunique()),
        "n_output_rows": int(len(long)),
        "n_classified_rows": int((long["classification_status"] == "classified").sum()),
        "prediction_time_min": str(pred_df[args.time_col].min()),
        "prediction_time_max": str(pred_df[args.time_col].max()),
        "destination_time_min": str(long["timestamp"].min()),
        "destination_time_max": str(long["timestamp"].max()),
        "alert_counts_model": counts_model,
        "alert_counts_conservative": counts_conservative,
        "method_note": (
            "La propagación es un desplazamiento temporal según delay_h. No modela formación, "
            "decaimiento ni reacción química de THM en EPANET. Forecast proyecta cada valor de "
            "origen a t+delay. Nowcast reconstruye el valor de origen en t-delay mediante la "
            "interpolación indicada."
        ),
        "control_note": (
            "Cuando se usa --control-col, el control pertenece a la serie de origen y se desplaza "
            "o interpola igual que la predicción. No constituye una validación con THM medido en destino."
        ),
    }
    write_json(output_dir / "operational_alert_report.json", report)

    print("====================================================")
    print(f"HGB + EPANET OPERATIONAL ALERTS v{SCRIPT_VERSION}")
    print("====================================================")
    print(f"Predicciones: {pred_path}")
    print(f"Targets:      {targets.attrs.get('source_path')}")
    print(f"Salida:       {output_dir}")
    print(f"Modo:         {args.time_shift_mode}")
    print(f"Margen:       +{args.safety_margin:g}")
    print(f"Filas origen: {len(pred_df)}")
    print(f"Targets:      {len(targets)} ({targets['node_id'].nunique()} nodos únicos)")
    print(f"Filas salida: {len(long)}")
    print("Alert counts model:", counts_model)
    print("Alert counts conservative:", counts_conservative)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
