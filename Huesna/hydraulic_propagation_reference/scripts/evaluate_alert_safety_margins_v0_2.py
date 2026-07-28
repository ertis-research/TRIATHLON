from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_VERSION = "0.2.0"
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


def parse_margins(text: str) -> List[float]:
    margins: List[float] = []
    for part in re.split(r"[;, ]+", clean_text(text)):
        if not part:
            continue
        value = float(part)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Todos los márgenes deben ser finitos y no negativos")
        margins.append(value)
    if not margins:
        raise ValueError("No se ha indicado ningún margen válido")
    return sorted(set(margins))


def norm_col_lookup(columns: Iterable[str]) -> Dict[str, str]:
    return {clean_text(column).lower(): column for column in columns}


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = norm_col_lookup(df.columns)
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
    label_col = first_existing_col(raw, ["alert_zone_label", "zone_label"])

    delay_col = reference_delay_col
    if delay_col not in raw.columns:
        delay_col = first_existing_col(
            raw,
            [reference_delay_col, "delay_h", "tiempo_total_h", "tiempo_recorrido_h"],
        )

    required = {
        "reference": ref_col,
        "node": node_col,
        "prealert": pre_col,
        "alert": alert_col,
        "delay": delay_col,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"Faltan columnas necesarias en targets: {missing}. Columnas: {list(raw.columns)}")

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

    n_raw = len(out)
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
    out.attrs["n_raw_rows"] = int(n_raw)
    out.attrs["n_dropped_invalid_rows"] = int(n_raw - before_dedup)
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
        candidate_files = [
            directory / "mapped_reference_rows_with_alert_zone.csv",
            directory / "comparison_travel_times_with_alert_zones.csv",
            directory / "targets_mapped_references.csv",
        ]
        existing = [path for path in candidate_files if path.is_file()]
        if not existing:
            raise FileNotFoundError(
                "No se ha encontrado un CSV de targets en --epanet-dir. Usa --targets-csv o genera primero el baseline."
            )
        source = existing[0]
        raw = read_csv_flexible(source)

    targets = normalize_targets(raw, reference_delay_col)
    if targets.empty:
        raise ValueError("No hay targets mapeados y clasificables después de validar la entrada")
    targets.attrs["source_path"] = str(source)
    return targets


def prepare_predictions(
    raw: pd.DataFrame,
    time_col: str,
    pred_col: str,
    control_col: str,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    required = [time_col, pred_col, control_col]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas en predictions CSV: {missing}. Disponibles: {list(raw.columns)}")

    out = raw[required].copy()
    n_input = len(out)
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out[pred_col] = pd.to_numeric(out[pred_col], errors="coerce")
    out[control_col] = pd.to_numeric(out[control_col], errors="coerce")
    out = out[out[time_col].notna()].sort_values(time_col).reset_index(drop=True)
    n_invalid_time = n_input - len(out)
    n_duplicate_time = int(out.duplicated(time_col, keep="last").sum())
    if n_duplicate_time:
        out = out.drop_duplicates(time_col, keep="last").reset_index(drop=True)

    if out.empty:
        raise ValueError("No quedan timestamps válidos en predictions CSV")
    if out[pred_col].notna().sum() == 0:
        raise ValueError(f"La columna '{pred_col}' no contiene predicciones numéricas válidas")
    if out[control_col].notna().sum() == 0:
        raise ValueError(f"La columna '{control_col}' no contiene valores de control numéricos válidos")

    return out, {
        "n_input_rows": int(n_input),
        "n_invalid_timestamp_rows": int(n_invalid_time),
        "n_duplicate_timestamp_rows_kept_last": int(n_duplicate_time),
    }


def build_long_predictions(
    pred_df: pd.DataFrame,
    targets: pd.DataFrame,
    time_col: str,
    pred_col: str,
    control_col: str,
    margin: float,
) -> pd.DataFrame:
    base = pred_df[[time_col, pred_col, control_col]].copy()
    base = base.rename(columns={time_col: "source_timestamp"})
    base["_join_key"] = 1
    target_rows = targets.copy()
    target_rows["_join_key"] = 1

    long = target_rows.merge(base, on="_join_key", how="inner").drop(columns=["_join_key"])
    long["margin"] = float(margin)
    long["destination_timestamp"] = long["source_timestamp"] + pd.to_timedelta(long["delay_h"], unit="h")
    long["timestamp"] = long["destination_timestamp"]
    long["pred_base"] = pd.to_numeric(long[pred_col], errors="coerce")
    long["pred_upper"] = long["pred_base"] + float(margin)
    long["source_control_value"] = pd.to_numeric(long[control_col], errors="coerce")
    long["control_value"] = long["source_control_value"]

    long["alert_level_pred_base"] = classify_values(
        long["pred_base"], long["prealert_threshold"], long["alert_threshold"]
    )
    long["alert_level_pred_upper"] = classify_values(
        long["pred_upper"], long["prealert_threshold"], long["alert_threshold"]
    )
    long["alert_level_control"] = classify_values(
        long["source_control_value"], long["prealert_threshold"], long["alert_threshold"]
    )

    long["rank_pred_base"] = level_to_rank(long["alert_level_pred_base"])
    long["rank_pred_upper"] = level_to_rank(long["alert_level_pred_upper"])
    long["rank_control"] = level_to_rank(long["alert_level_control"])

    valid = (long["rank_pred_upper"] >= 0) & (long["rank_control"] >= 0)
    long["alert_match"] = valid & (long["rank_pred_upper"] == long["rank_control"])
    long["control_event"] = valid & (long["rank_control"] >= 1)
    long["pred_event"] = valid & (long["rank_pred_upper"] >= 1)
    long["missed_prealert_or_alert"] = valid & (long["rank_control"] >= 1) & (long["rank_pred_upper"] == 0)
    long["missed_alert"] = valid & (long["rank_control"] == 2) & (long["rank_pred_upper"] < 2)
    long["false_prealert_or_alert"] = valid & (long["rank_control"] == 0) & (long["rank_pred_upper"] >= 1)
    long["false_alert"] = valid & (long["rank_control"] < 2) & (long["rank_pred_upper"] == 2)
    long["underclassified"] = valid & (long["rank_pred_upper"] < long["rank_control"])
    long["overclassified"] = valid & (long["rank_pred_upper"] > long["rank_control"])
    long["margin_to_prealert_upper"] = long["pred_upper"] - long["prealert_threshold"]
    long["margin_to_alert_upper"] = long["pred_upper"] - long["alert_threshold"]
    long["error_pred_upper_minus_control"] = long["pred_upper"] - long["source_control_value"]
    return long


def safe_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else math.nan


def summarize_bool_counts(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows = int(len(df))
    valid = (df["rank_pred_upper"] >= 0) & (df["rank_control"] >= 0)
    n_valid = int(valid.sum())
    control_event = valid & (df["rank_control"] >= 1)
    pred_event = valid & (df["rank_pred_upper"] >= 1)

    true_positive = int((control_event & pred_event).sum())
    false_positive = int((~control_event & pred_event & valid).sum())
    false_negative = int((control_event & ~pred_event).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else math.nan
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0
        else math.nan
    )

    return {
        "n_rows": n_rows,
        "n_classified": n_valid,
        "n_control_normal": int((df["alert_level_control"] == "normal").sum()),
        "n_control_prealerta": int((df["alert_level_control"] == "prealerta").sum()),
        "n_control_alerta": int((df["alert_level_control"] == "alerta").sum()),
        "n_pred_normal": int((df["alert_level_pred_upper"] == "normal").sum()),
        "n_pred_prealerta": int((df["alert_level_pred_upper"] == "prealerta").sum()),
        "n_pred_alerta": int((df["alert_level_pred_upper"] == "alerta").sum()),
        "n_match": int((valid & (df["rank_pred_upper"] == df["rank_control"])).sum()),
        "match_rate": float((valid & (df["rank_pred_upper"] == df["rank_control"])).sum() / n_valid) if n_valid else math.nan,
        "n_control_event_ge_prealerta": int(control_event.sum()),
        "n_pred_event_ge_prealerta": int(pred_event.sum()),
        "n_true_positive_event": true_positive,
        "n_false_prealert_or_alert": false_positive,
        "n_missed_prealert_or_alert": false_negative,
        "n_missed_alert": int(df["missed_alert"].sum()),
        "n_false_alert": int(df["false_alert"].sum()),
        "n_underclassified": int(df["underclassified"].sum()),
        "n_overclassified": int(df["overclassified"].sum()),
        "precision_event_ge_prealerta": float(precision),
        "recall_event_ge_prealerta": float(recall),
        "f1_event_ge_prealerta": float(f1),
        "mean_error_pred_upper_minus_control": safe_float(df["error_pred_upper_minus_control"].mean()),
        "mae_pred_upper_vs_control": safe_float(df["error_pred_upper_minus_control"].abs().mean()),
        "max_pred_upper": safe_float(df["pred_upper"].max()),
        "max_control": safe_float(df["source_control_value"].max()),
    }


def summarize_by_target(long: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "margin",
        "target_id",
        "orden",
        "reference_name",
        "node_id",
        "node_name",
        "delay_h",
        "alert_zone",
        "prealert_threshold",
        "alert_threshold",
    ]
    for keys, group in long.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row.update(summarize_bool_counts(group))
        rows.append(row)
    return pd.DataFrame(rows)


def deduplicate_by_node_timestamp(long: pd.DataFrame) -> pd.DataFrame:
    node_ts = (
        long.groupby(["margin", "node_id", "node_name", "timestamp"], dropna=False)
        .agg(
            rank_pred_upper=("rank_pred_upper", "max"),
            rank_control=("rank_control", "max"),
            pred_upper=("pred_upper", "max"),
            source_control_value=("source_control_value", "max"),
            prealert_threshold=("prealert_threshold", "min"),
            alert_threshold=("alert_threshold", "min"),
            alert_zone=("alert_zone", lambda values: "|".join(sorted(set(map(str, values))))),
            n_reference_rows=("target_id", "nunique"),
        )
        .reset_index()
    )
    node_ts["control_value"] = node_ts["source_control_value"]
    node_ts["alert_level_pred_upper"] = rank_to_level(node_ts["rank_pred_upper"])
    node_ts["alert_level_control"] = rank_to_level(node_ts["rank_control"])

    valid = (node_ts["rank_pred_upper"] >= 0) & (node_ts["rank_control"] >= 0)
    node_ts["alert_match"] = valid & (node_ts["rank_pred_upper"] == node_ts["rank_control"])
    node_ts["control_event"] = valid & (node_ts["rank_control"] >= 1)
    node_ts["pred_event"] = valid & (node_ts["rank_pred_upper"] >= 1)
    node_ts["missed_prealert_or_alert"] = valid & (node_ts["rank_control"] >= 1) & (node_ts["rank_pred_upper"] == 0)
    node_ts["missed_alert"] = valid & (node_ts["rank_control"] == 2) & (node_ts["rank_pred_upper"] < 2)
    node_ts["false_prealert_or_alert"] = valid & (node_ts["rank_control"] == 0) & (node_ts["rank_pred_upper"] >= 1)
    node_ts["false_alert"] = valid & (node_ts["rank_control"] < 2) & (node_ts["rank_pred_upper"] == 2)
    node_ts["underclassified"] = valid & (node_ts["rank_pred_upper"] < node_ts["rank_control"])
    node_ts["overclassified"] = valid & (node_ts["rank_pred_upper"] > node_ts["rank_control"])
    node_ts["error_pred_upper_minus_control"] = node_ts["pred_upper"] - node_ts["source_control_value"]
    return node_ts


def summarize_by_node(node_ts: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "margin",
        "node_id",
        "node_name",
        "alert_zone",
        "prealert_threshold",
        "alert_threshold",
    ]
    for keys, group in node_ts.groupby(group_cols, dropna=False, sort=True):
        row = dict(zip(group_cols, keys))
        row.update(summarize_bool_counts(group))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_candidate_margins(
    global_df: pd.DataFrame,
    false_event_penalty_per_1000: float,
) -> pd.DataFrame:
    if global_df.empty:
        return global_df.copy()

    out = global_df.copy()
    out["false_event_rate_per_1000_rows"] = (
        out["n_false_prealert_or_alert"] / out["n_classified"].replace(0, np.nan) * 1000.0
    )
    out["missed_event_rate"] = (
        out["n_missed_prealert_or_alert"]
        / out["n_control_event_ge_prealerta"].replace(0, np.nan)
    )
    out["false_event_rate"] = (
        out["n_false_prealert_or_alert"]
        / out["n_pred_event_ge_prealerta"].replace(0, np.nan)
    )
    out["heuristic_false_event_penalty_per_1000"] = float(false_event_penalty_per_1000)
    out["heuristic_balanced_score"] = (
        out["f1_event_ge_prealerta"].fillna(0.0)
        - float(false_event_penalty_per_1000)
        * out["false_event_rate_per_1000_rows"].fillna(0.0)
    )

    by_f1 = out.sort_values(
        ["f1_event_ge_prealerta", "recall_event_ge_prealerta", "precision_event_ge_prealerta", "margin"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    by_f1["candidate_rank_by_f1"] = np.arange(1, len(by_f1) + 1)

    by_heuristic = out.sort_values(
        ["heuristic_balanced_score", "f1_event_ge_prealerta", "recall_event_ge_prealerta", "margin"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    by_heuristic["candidate_rank_by_heuristic"] = np.arange(1, len(by_heuristic) + 1)

    return by_f1.merge(
        by_heuristic[["margin", "candidate_rank_by_heuristic"]],
        on="margin",
        how="left",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Evaluación offline de márgenes v{SCRIPT_VERSION}")
    parser.add_argument("--predictions-csv", required=True, type=Path)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--epanet-dir", type=Path)
    source_group.add_argument("--targets-csv", type=Path)
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--pred-col", default="pred_selected")
    parser.add_argument("--control-col", default="y_true")
    parser.add_argument("--reference-delay-col", default="tiempo_total_h")
    parser.add_argument("--margins", default="0,1,2,3,4,5,6")
    parser.add_argument(
        "--false-event-penalty-per-1000",
        type=float,
        default=0.05,
        help="Peso explícito de la puntuación heurística; no representa un óptimo productivo.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--export-long", action="store_true")
    parser.add_argument("--export-node-timestamp", action="store_true")
    args = parser.parse_args()

    if (
        not math.isfinite(args.false_event_penalty_per_1000)
        or args.false_event_penalty_per_1000 < 0
    ):
        raise ValueError("--false-event-penalty-per-1000 debe ser finito y no negativo")

    margins = parse_margins(args.margins)
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

    global_node_rows: List[Dict[str, Any]] = []
    global_reference_rows: List[Dict[str, Any]] = []
    by_target_frames: List[pd.DataFrame] = []
    by_node_frames: List[pd.DataFrame] = []
    node_ts_frames: List[pd.DataFrame] = []
    long_frames: List[pd.DataFrame] = []

    for margin in margins:
        long = build_long_predictions(
            pred_df,
            targets,
            args.time_col,
            args.pred_col,
            args.control_col,
            margin,
        )
        node_ts = deduplicate_by_node_timestamp(long)

        node_global = {"margin": float(margin), "aggregation_scope": "node_timestamp_deduplicated"}
        node_global.update(summarize_bool_counts(node_ts))
        global_node_rows.append(node_global)

        reference_global = {"margin": float(margin), "aggregation_scope": "reference_weighted_legacy"}
        reference_global.update(summarize_bool_counts(long))
        global_reference_rows.append(reference_global)

        by_target_frames.append(summarize_by_target(long))
        by_node_frames.append(summarize_by_node(node_ts))
        if args.export_node_timestamp:
            node_ts_frames.append(node_ts)

        if args.export_long:
            keep_cols = [
                "margin",
                "timestamp",
                "destination_timestamp",
                "source_timestamp",
                "target_id",
                "orden",
                "reference_name",
                "node_id",
                "node_name",
                "delay_h",
                "alert_zone",
                "prealert_threshold",
                "alert_threshold",
                "pred_base",
                "pred_upper",
                "source_control_value",
                "alert_level_pred_base",
                "alert_level_pred_upper",
                "alert_level_control",
                "alert_match",
                "missed_prealert_or_alert",
                "missed_alert",
                "false_prealert_or_alert",
                "false_alert",
                "underclassified",
                "overclassified",
                "margin_to_prealert_upper",
                "margin_to_alert_upper",
                "error_pred_upper_minus_control",
            ]
            long_frames.append(long[keep_cols].copy())

    global_df = pd.DataFrame(global_node_rows).sort_values("margin").reset_index(drop=True)
    global_reference_df = pd.DataFrame(global_reference_rows).sort_values("margin").reset_index(drop=True)
    by_target_df = pd.concat(by_target_frames, ignore_index=True)
    by_node_df = pd.concat(by_node_frames, ignore_index=True)
    node_ts_df = pd.concat(node_ts_frames, ignore_index=True) if node_ts_frames else pd.DataFrame()
    candidates_df = choose_candidate_margins(
        global_df,
        args.false_event_penalty_per_1000,
    )

    safe_to_csv(targets, output_dir / "targets_used.csv")
    safe_to_csv(global_df, output_dir / "margin_evaluation_global.csv")
    safe_to_csv(
        global_reference_df,
        output_dir / "margin_evaluation_global_reference_weighted.csv",
    )
    safe_to_csv(by_target_df, output_dir / "margin_evaluation_by_target.csv")
    safe_to_csv(by_node_df, output_dir / "margin_evaluation_by_node.csv")
    safe_to_csv(candidates_df, output_dir / "margin_recommendation_candidates.csv")
    if args.export_node_timestamp:
        safe_to_csv(node_ts_df, output_dir / "margin_evaluation_node_timestamp.csv")
    if args.export_long:
        safe_to_csv(pd.concat(long_frames, ignore_index=True), output_dir / "margin_evaluation_long.csv")

    top_f1 = (
        candidates_df.sort_values("candidate_rank_by_f1").iloc[0].to_dict()
        if not candidates_df.empty
        else {}
    )
    top_heuristic = (
        candidates_df.sort_values("candidate_rank_by_heuristic").iloc[0].to_dict()
        if not candidates_df.empty
        else {}
    )
    outputs = [
        "targets_used.csv",
        "margin_evaluation_global.csv",
        "margin_evaluation_global_reference_weighted.csv",
        "margin_evaluation_by_target.csv",
        "margin_evaluation_by_node.csv",
        "margin_recommendation_candidates.csv",
    ]
    if args.export_node_timestamp:
        outputs.append("margin_evaluation_node_timestamp.csv")
    if args.export_long:
        outputs.append("margin_evaluation_long.csv")

    report = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictions_csv": str(pred_path),
        "targets_source": targets.attrs.get("source_path"),
        "output_dir": str(output_dir),
        "time_col": args.time_col,
        "pred_col": args.pred_col,
        "control_col": args.control_col,
        "reference_delay_col": args.reference_delay_col,
        "margins": margins,
        "false_event_penalty_per_1000": float(args.false_event_penalty_per_1000),
        "prediction_cleaning": prediction_cleaning,
        "target_cleaning": {
            "n_raw_rows": targets.attrs.get("n_raw_rows"),
            "n_dropped_invalid_rows": targets.attrs.get("n_dropped_invalid_rows"),
            "n_dropped_duplicate_rows": targets.attrs.get("n_dropped_duplicate_rows"),
        },
        "n_prediction_rows": int(len(pred_df)),
        "n_targets": int(len(targets)),
        "n_unique_nodes": int(targets["node_id"].nunique()),
        "time_min": str(pred_df[args.time_col].min()),
        "time_max": str(pred_df[args.time_col].max()),
        "primary_global_aggregation": "node_timestamp_deduplicated",
        "outputs": outputs,
        "top_candidate_by_event_f1": top_f1,
        "top_candidate_by_heuristic_score": top_heuristic,
        "evaluation_scope_note": (
            "El control es y_true/control_col medido en el origen. Predicción y control se "
            "clasifican con los umbrales de cada destino y se desplazan juntos por delay_h. "
            "Por tanto, esta evaluación mide el comportamiento del soft-sensor de origen bajo "
            "los umbrales de destino; no valida la propagación hidráulica con THM medido en destino."
        ),
        "weighting_note": (
            "margin_evaluation_global.csv usa una fila por nodo y timestamp para evitar que "
            "referencias duplicadas ponderen varias veces. La agregación histórica por referencia "
            "se conserva por transparencia en margin_evaluation_global_reference_weighted.csv."
        ),
        "heuristic_note": (
            "heuristic_balanced_score es una regla auxiliar parametrizada, no un óptimo matemático "
            "ni una configuración productiva final."
        ),
    }
    write_json(output_dir / "margin_evaluation_report.json", report)

    show_cols = [
        "margin",
        "n_control_event_ge_prealerta",
        "n_pred_event_ge_prealerta",
        "n_true_positive_event",
        "n_missed_prealert_or_alert",
        "n_false_prealert_or_alert",
        "precision_event_ge_prealerta",
        "recall_event_ge_prealerta",
        "f1_event_ge_prealerta",
    ]
    print("========================================")
    print(f"ALERT SAFETY MARGIN EVALUATION v{SCRIPT_VERSION}")
    print("========================================")
    print(f"Predicciones: {pred_path}")
    print(f"Targets:      {targets.attrs.get('source_path')}")
    print(f"Salida:       {output_dir}")
    print(f"Targets:      {len(targets)} ({targets['node_id'].nunique()} nodos únicos)")
    print("\nResumen global deduplicado por nodo y timestamp:")
    print(global_df[show_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
