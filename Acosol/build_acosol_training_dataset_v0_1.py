from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "v0.1"

TAG_MAP = {
    "SC_81_16_CLT02_PV": "cloro",
    "SC_81_16_TRB01_PV": "turbidez",
    "SC_81_16_COND01_PV": "conductividad",
    "SC_81_16_PH01_PV": "ph",
    "SC_81_16_TEMP01_PV": "temperatura",
    "SC_81_16_THM01_PV": "thm",
    "SC_81_16_CARB01_TOC_PV": "cot",
    "SC_81_16_CARB01_TIC_PV": "tic",
    "SC_81_16_BROM01_PV": "bromuro",
}

EXPECTED_TAGS = set(TAG_MAP)
FEATURE_COLUMNS = [
    "cloro",
    "turbidez",
    "conductividad",
    "ph",
    "temperatura",
    "cot",
    "tic",
    "bromuro",
]
TARGET_COLUMN = "thm"

MODEL_FEATURES_RECOMMENDED_CURRENTLY = [
    "cloro",
    "turbidez",
    "conductividad",
    "ph",
    "cot",
    "tic",
    "bromuro",
]

PRODUCTION_FILENAME_RE = re.compile(r"^ABASTCLI01_(\d{8})_(\d{6})\.csv$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye la serie temporal y el dataset supervisado de Acosol a partir de los snapshots SFTP."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("sftp_raw"), help="Carpeta con los CSV originales.")
    parser.add_argument("--output-dir", type=Path, default=Path("acosol_sftp_dataset_v0_1"), help="Carpeta de salida.")
    parser.add_argument("--gap-threshold-minutes", type=float, default=7.5, help="Umbral para marcar huecos temporales.")
    return parser.parse_args()


def parse_filename_timestamp(filename: str) -> datetime | None:
    match = PRODUCTION_FILENAME_RE.fullmatch(filename)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        return datetime.strptime(date_part + time_part, "%d%m%Y%H%M%S")
    except ValueError:
        return None


def decode_bytes(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1-replace"


def parse_decimal(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "sin datos", "s/d"}:
        return None
    try:
        number = float(text.replace(",", "."))
    except ValueError:
        return None
    return number if np.isfinite(number) else None


def parse_internal_timestamp(value: Any) -> datetime | None:
    text = str(value).strip()
    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def inspect_snapshot(path: Path, filename_timestamp: datetime) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    report: dict[str, Any] = {
        "source_file": path.name,
        "filename_timestamp": filename_timestamp,
        "parse_ok": False,
        "encoding": "",
        "row_count": 0,
        "unique_tag_count": 0,
        "internal_timestamp": pd.NaT,
        "all_rows_same_timestamp": False,
        "filename_timestamp_matches_internal": False,
        "missing_tags": "",
        "unknown_tags": "",
        "duplicate_tags": "",
        "invalid_value_tags": "",
        "has_thm_tag": False,
        "thm_value_valid": False,
        "complete_expected_tags": False,
        "quality_flags": "",
        "error": "",
    }

    try:
        text, encoding = decode_bytes(path.read_bytes())
        report["encoding"] = encoding
        reader = csv.DictReader(text.splitlines(), delimiter=";")

        if reader.fieldnames is None:
            report["error"] = "missing_header"
            report["quality_flags"] = "parse_error"
            return None, report

        fields = [str(field).strip() for field in reader.fieldnames]
        if fields != ["tag", "timestamp", "value"]:
            report["error"] = "unexpected_columns:" + "|".join(fields)
            report["quality_flags"] = "parse_error"
            return None, report

        rows = list(reader)
        report["row_count"] = len(rows)

        tags: list[str] = []
        timestamps: list[datetime] = []
        values_by_tag: dict[str, float | None] = {}
        invalid_value_tags: list[str] = []

        for row in rows:
            tag = str(row.get("tag", "")).strip()
            timestamp = parse_internal_timestamp(row.get("timestamp", ""))
            value = parse_decimal(row.get("value", ""))

            if tag:
                tags.append(tag)
                values_by_tag[tag] = value
                if tag in EXPECTED_TAGS and value is None:
                    invalid_value_tags.append(tag)

            if timestamp is not None:
                timestamps.append(timestamp)

        tag_counter = Counter(tags)
        tag_set = set(tags)
        missing_tags = sorted(EXPECTED_TAGS - tag_set)
        unknown_tags = sorted(tag_set - EXPECTED_TAGS)
        duplicate_tags = sorted(tag for tag, count in tag_counter.items() if count > 1)
        invalid_value_tags = sorted(set(invalid_value_tags))

        report["unique_tag_count"] = len(tag_set)
        report["missing_tags"] = ";".join(missing_tags)
        report["unknown_tags"] = ";".join(unknown_tags)
        report["duplicate_tags"] = ";".join(duplicate_tags)
        report["invalid_value_tags"] = ";".join(invalid_value_tags)
        report["has_thm_tag"] = "SC_81_16_THM01_PV" in tag_set
        report["thm_value_valid"] = values_by_tag.get("SC_81_16_THM01_PV") is not None
        report["complete_expected_tags"] = tag_set == EXPECTED_TAGS and not duplicate_tags and len(rows) == len(EXPECTED_TAGS)

        unique_timestamps = sorted(set(timestamps))
        if len(unique_timestamps) == 1 and len(timestamps) == len(rows):
            internal_timestamp = unique_timestamps[0]
            report["internal_timestamp"] = internal_timestamp
            report["all_rows_same_timestamp"] = True
            report["filename_timestamp_matches_internal"] = internal_timestamp == filename_timestamp
        elif unique_timestamps:
            report["internal_timestamp"] = unique_timestamps[0]

        flags: list[str] = []
        if len(timestamps) != len(rows):
            flags.append("invalid_internal_timestamp")
        if not report["all_rows_same_timestamp"]:
            flags.append("inconsistent_internal_timestamps")
        if report["all_rows_same_timestamp"] and not report["filename_timestamp_matches_internal"]:
            flags.append("filename_timestamp_mismatch")
        if unknown_tags:
            flags.append("unknown_tags")
        if duplicate_tags:
            flags.append("duplicate_tags")
        if invalid_value_tags:
            flags.append("invalid_numeric_value")
        if missing_tags:
            non_thm_missing = [tag for tag in missing_tags if tag != "SC_81_16_THM01_PV"]
            if non_thm_missing:
                flags.append("missing_input_tags")
            if "SC_81_16_THM01_PV" in missing_tags:
                flags.append("missing_thm")

        report["quality_flags"] = ";".join(flags) if flags else "ok"
        report["parse_ok"] = True

        if not report["all_rows_same_timestamp"]:
            return None, report

        observation: dict[str, Any] = {
            "timestamp": report["internal_timestamp"],
            "filename_timestamp": filename_timestamp,
            "source_file": path.name,
            "quality_flags": report["quality_flags"],
            "filename_timestamp_matches_internal": report["filename_timestamp_matches_internal"],
        }

        for tag, canonical_name in TAG_MAP.items():
            value = values_by_tag.get(tag)
            observation[canonical_name] = np.nan if value is None else value

        return observation, report

    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["quality_flags"] = "parse_error"
        return None, report


def append_flag(existing: str, new_flag: str) -> str:
    existing = str(existing).strip()
    if not existing or existing == "ok":
        return new_flag
    flags = existing.split(";")
    if new_flag not in flags:
        flags.append(new_flag)
    return ";".join(flags)


def add_temporal_quality_flags(df: pd.DataFrame, gap_threshold_minutes: float) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    df["previous_timestamp"] = df["timestamp"].shift(1)
    df["gap_minutes"] = (df["timestamp"] - df["previous_timestamp"]).dt.total_seconds() / 60.0

    duplicate_mask = df["timestamp"].duplicated(keep=False)
    gap_mask = df["gap_minutes"] > gap_threshold_minutes

    for idx in df.index[duplicate_mask]:
        df.at[idx, "quality_flags"] = append_flag(df.at[idx, "quality_flags"], "duplicate_timestamp")
    for idx in df.index[gap_mask]:
        df.at[idx, "quality_flags"] = append_flag(df.at[idx, "quality_flags"], "temporal_gap_before")

    return df


def main() -> int:
    args = parse_args()

    if args.gap_threshold_minutes <= 0:
        raise SystemExit("--gap-threshold-minutes debe ser > 0.")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de entrada: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    production_files: list[tuple[datetime, Path]] = []
    ignored_csv = 0

    for path in args.input_dir.glob("*.csv"):
        filename_timestamp = parse_filename_timestamp(path.name)
        if filename_timestamp is None:
            ignored_csv += 1
            continue
        production_files.append((filename_timestamp, path))

    production_files.sort(key=lambda item: item[0])

    print("Construcción dataset SFTP Acosol v0.1")
    print("=====================================")
    print(f"Carpeta de entrada: {args.input_dir}")
    print(f"Carpeta de salida:  {args.output_dir}")
    print(f"CSV productivos:    {len(production_files)}")
    print(f"CSV ignorados:      {ignored_csv}")

    if not production_files:
        raise SystemExit("No se han encontrado CSV productivos.")

    observations: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []

    for index, (filename_timestamp, path) in enumerate(production_files, start=1):
        observation, report = inspect_snapshot(path, filename_timestamp)
        reports.append(report)
        if observation is not None:
            observations.append(observation)
        if index % 1000 == 0 or index == len(production_files):
            print(f"Procesados {index}/{len(production_files)} ficheros...")

    report_df = pd.DataFrame(reports)
    timeseries = pd.DataFrame(observations)

    if timeseries.empty:
        raise SystemExit("No se ha podido construir ninguna observación temporal válida.")

    timeseries["timestamp"] = pd.to_datetime(timeseries["timestamp"])
    timeseries["filename_timestamp"] = pd.to_datetime(timeseries["filename_timestamp"])

    for column in [*FEATURE_COLUMNS, TARGET_COLUMN]:
        timeseries[column] = pd.to_numeric(timeseries[column], errors="coerce")

    timeseries = add_temporal_quality_flags(timeseries, args.gap_threshold_minutes)

    full_columns = [
        "timestamp",
        "cloro",
        "turbidez",
        "conductividad",
        "ph",
        "temperatura",
        "cot",
        "tic",
        "bromuro",
        "thm",
        "source_file",
        "filename_timestamp",
        "filename_timestamp_matches_internal",
        "gap_minutes",
        "quality_flags",
    ]
    timeseries = timeseries[full_columns].copy()

    supervised = timeseries[timeseries["thm"].notna()].copy()
    supervised = supervised.sort_values("timestamp").reset_index(drop=True)

    first_supervised_timestamp = supervised["timestamp"].min() if not supervised.empty else pd.NaT
    if pd.notna(first_supervised_timestamp):
        missing_thm_after_start = timeseries["timestamp"].ge(first_supervised_timestamp) & timeseries["thm"].isna()
    else:
        missing_thm_after_start = pd.Series(False, index=timeseries.index)

    report_df["filename_timestamp"] = pd.to_datetime(report_df["filename_timestamp"], errors="coerce")
    report_df["internal_timestamp"] = pd.to_datetime(report_df["internal_timestamp"], errors="coerce")

    issues = report_df[
        (~report_df["parse_ok"])
        | report_df["unknown_tags"].fillna("").astype(str).ne("")
        | report_df["duplicate_tags"].fillna("").astype(str).ne("")
        | report_df["invalid_value_tags"].fillna("").astype(str).ne("")
        | (report_df["has_thm_tag"].astype(bool) & ~report_df["thm_value_valid"].astype(bool))
        | (report_df["all_rows_same_timestamp"].astype(bool) & ~report_df["filename_timestamp_matches_internal"].astype(bool))
    ].copy()

    if missing_thm_after_start.any():
        source_files = set(timeseries.loc[missing_thm_after_start, "source_file"])
        issues = pd.concat(
            [issues, report_df[report_df["source_file"].isin(source_files)]],
            ignore_index=True,
        ).drop_duplicates(subset=["source_file"])

    timeseries_path = args.output_dir / "acosol_sftp_timeseries_v0_1.csv"
    supervised_path = args.output_dir / "acosol_sftp_supervised_v0_1.csv"
    audit_path = args.output_dir / "acosol_sftp_snapshot_audit_v0_1.csv"
    issues_path = args.output_dir / "acosol_sftp_issues_v0_1.csv"
    summary_path = args.output_dir / "acosol_sftp_dataset_summary_v0_1.csv"
    metadata_path = args.output_dir / "acosol_sftp_dataset_metadata_v0_1.json"

    timeseries.to_csv(timeseries_path, index=False, encoding="utf-8-sig")
    supervised.to_csv(supervised_path, index=False, encoding="utf-8-sig")
    report_df.to_csv(audit_path, index=False, encoding="utf-8-sig")
    issues.to_csv(issues_path, index=False, encoding="utf-8-sig")

    temperature = pd.to_numeric(supervised["temperatura"], errors="coerce") if not supervised.empty else pd.Series(dtype=float)

    summary_rows = [
        {"metric": "version", "value": VERSION},
        {"metric": "production_files_found", "value": len(production_files)},
        {"metric": "ignored_non_productive_csv", "value": ignored_csv},
        {"metric": "parse_ok_files", "value": int(report_df["parse_ok"].sum())},
        {"metric": "parse_error_files", "value": int((~report_df["parse_ok"]).sum())},
        {"metric": "timeseries_rows", "value": len(timeseries)},
        {"metric": "timeseries_min_timestamp", "value": timeseries["timestamp"].min()},
        {"metric": "timeseries_max_timestamp", "value": timeseries["timestamp"].max()},
        {"metric": "supervised_rows", "value": len(supervised)},
        {"metric": "supervised_min_timestamp", "value": supervised["timestamp"].min() if len(supervised) else pd.NaT},
        {"metric": "supervised_max_timestamp", "value": supervised["timestamp"].max() if len(supervised) else pd.NaT},
        {"metric": "missing_thm_after_first_valid_thm", "value": int(missing_thm_after_start.sum())},
        {"metric": "timeseries_duplicate_timestamps", "value": int(timeseries["timestamp"].duplicated(keep=False).sum())},
        {"metric": "timeseries_temporal_gaps", "value": int((timeseries["gap_minutes"] > args.gap_threshold_minutes).sum())},
        {"metric": "issues_rows", "value": len(issues)},
        {"metric": "temperature_available_rows", "value": int(temperature.notna().sum())},
        {"metric": "temperature_unique_values", "value": int(temperature.nunique(dropna=True))},
        {"metric": "temperature_min", "value": float(temperature.min()) if temperature.notna().any() else np.nan},
        {"metric": "temperature_max", "value": float(temperature.max()) if temperature.notna().any() else np.nan},
        {"metric": "temperature_std", "value": float(temperature.std()) if temperature.notna().sum() > 1 else np.nan},
        {"metric": "current_recommended_model_features", "value": ";".join(MODEL_FEATURES_RECOMMENDED_CURRENTLY)},
        {"metric": "temperature_model_status", "value": "retained_but_excluded_until_hach_tcpip_measurement_is_dynamic"},
    ]
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    metadata = {
        "version": VERSION,
        "tag_map": TAG_MAP,
        "feature_columns_available": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "current_recommended_model_features": MODEL_FEATURES_RECOMMENDED_CURRENTLY,
        "temperature_model_status": (
            "Retained in the dataset but excluded from current model features because "
            "plant reported that the Hach TCP/IP license is expired and the temperature "
            "measurement is static. Reevaluate after the signal becomes dynamic and "
            "sufficient representative history is available."
        ),
        "canonical_timestamp": "internal CSV timestamp",
        "filename_timestamp_role": "traceability and consistency check only",
        "supervised_dataset_rule": "rows with numeric THM value",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("RESUMEN")
    print("=" * 100)
    print(f"Serie temporal completa:             {len(timeseries)} filas")
    print(f"Dataset supervisado con THM:         {len(supervised)} filas")
    print(f"Primer timestamp supervisado:        {first_supervised_timestamp}")
    print(f"Último timestamp supervisado:        {supervised['timestamp'].max() if not supervised.empty else pd.NaT}")
    print(f"THM ausente después de su inicio:    {int(missing_thm_after_start.sum())}")
    print(f"Incidencias de snapshot:             {len(issues)}")
    print(f"Duplicados temporales (serie):       {int(timeseries['timestamp'].duplicated(keep=False).sum())}")
    print(f"Huecos > {args.gap_threshold_minutes:g} min (serie):         {int((timeseries['gap_minutes'] > args.gap_threshold_minutes).sum())}")

    if not supervised.empty:
        print("\nTemperatura en periodo supervisado:")
        print(f"  Valores disponibles:               {int(temperature.notna().sum())}")
        print(f"  Valores únicos:                    {int(temperature.nunique(dropna=True))}")
        if temperature.notna().any():
            print(f"  Mínimo:                            {temperature.min():.6g}")
            print(f"  Máximo:                            {temperature.max():.6g}")
        print("  Uso actual en modelo:              EXCLUIDA temporalmente")

    print("\nSalidas:")
    print(f"- {timeseries_path}")
    print(f"- {supervised_path}")
    print(f"- {audit_path}")
    print(f"- {issues_path}")
    print(f"- {summary_path}")
    print(f"- {metadata_path}")
    print("\n[OK] Dataset construido correctamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
