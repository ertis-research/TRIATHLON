from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

MANIFEST_CSV = SCRIPT_DIR / "huelva_multiparam_sftp" / "logs" / "huelva_sftp_selective_download_manifest_v0_15.csv"
WINDOWS_CSV = SCRIPT_DIR / "huelva_outputs_v0_2_thm_pdf" / "huelva_required_multiparametric_windows_exact_v0_2.csv"

OUTPUT_DIR = SCRIPT_DIR / "huelva_par_etl_outputs_v0_1"

# Decisiones v0.1:
# - Solo ficheros par.
# - Ficheros fp fuera.
# - Valor corregido válido como principal.
# - Valor medido válido como fallback.
# - Valores marcados explícitamente como no válidos se excluyen de la señal final.
# - Nombres naturales en salidas.
# - Variables espectro-resumen dentro de par desactivadas por defecto para el dataset de modelo.
INCLUDE_PAR_SPECTRO_SUMMARY_IN_FEATURES = False
INCLUDE_PATTERN_ALARMS_IN_FEATURES = False

EXPECTED_SAMPLE_INTERVAL_MINUTES = 2.0
MIN_WINDOW_COVERAGE_RATIO_WARN = 0.10

WRITE_RAW_WIDE = True
WRITE_EXCEL = True

VERSION = "v0.1"

warnings.filterwarnings(
    "ignore",
    message="Could not infer format.*",
    category=UserWarning,
)


CORE_VARIABLES: Dict[str, Dict[str, Any]] = {
    "turbidity_ntu": {
        "label": "Turb EPA",
        "unit": "NTU",
        "group": "core",
        "measured_col": "i 25190501",
        "measured_status_col": "i 25190501.1",
        "corrected_col": "i 25190501.2",
        "corrected_status_col": "i 25190501.3",
    },
    "conductivity_us_cm": {
        "label": "Conductividad",
        "unit": "uS/cm",
        "group": "core",
        "measured_col": "cond 25191200",
        "measured_status_col": "cond 25191200.1",
        "corrected_col": "cond 25191200.2",
        "corrected_status_col": "cond 25191200.3",
    },
    "uv254_abs_m": {
        "label": "UV254",
        "unit": "Abs/m",
        "group": "core",
        "measured_col": "i 25190501.4",
        "measured_status_col": "i 25190501.5",
        "corrected_col": "i 25190501.6",
        "corrected_status_col": "i 25190501.7",
    },
    "toc_mg_l": {
        "label": "TOCeq",
        "unit": "mg/l",
        "group": "core",
        "measured_col": "i 25190501.8",
        "measured_status_col": "i 25190501.9",
        "corrected_col": "i 25190501.10",
        "corrected_status_col": "i 25190501.11",
    },
    "doc_mg_l": {
        "label": "DOCeq",
        "unit": "mg/l",
        "group": "core",
        "measured_col": "i 25190501.12",
        "measured_status_col": "i 25190501.13",
        "corrected_col": "i 25190501.14",
        "corrected_status_col": "i 25190501.15",
    },
    "free_chlorine_mg_l": {
        "label": "Cloro libre",
        "unit": "mg/l",
        "group": "core",
        "measured_col": "chlo 23441652",
        "measured_status_col": "chlo 23441652.1",
        "corrected_col": "chlo 23441652.2",
        "corrected_status_col": "chlo 23441652.3",
    },
    "ph": {
        "label": "pH",
        "unit": "",
        "group": "core",
        "measured_col": "ise 25191419",
        "measured_status_col": "ise 25191419.1",
        "corrected_col": "ise 25191419.2",
        "corrected_status_col": "ise 25191419.3",
    },
    "temperature_c": {
        "label": "Temperatura",
        "unit": "C",
        "group": "core",
        "measured_col": "ise 25191419.4",
        "measured_status_col": "ise 25191419.5",
        "corrected_col": "ise 25191419.6",
        "corrected_status_col": "ise 25191419.7",
    },
    "color_hazen": {
        "label": "COLORtru",
        "unit": "Hazen",
        "group": "core",
        "measured_col": "i 25190501.16",
        "measured_status_col": "i 25190501.17",
        "corrected_col": "i 25190501.18",
        "corrected_status_col": "i 25190501.19",
    },
    "flow": {
        "label": "Flow",
        "unit": "0/1",
        "group": "core",
        "measured_col": "Entrada digital",
        "measured_status_col": "Entrada digital.1",
        "corrected_col": None,
        "corrected_status_col": None,
    },
}


PAR_SPECTRO_SUMMARY_VARIABLES: Dict[str, Dict[str, Any]] = {
    "spec_toc_mg_l": {
        "label": "TOCeq spectro",
        "unit": "mg/l",
        "group": "par_spectro_summary",
        "measured_col": "spec 23010204",
        "measured_status_col": "spec 23010204.1",
        "corrected_col": "spec 23010204.2",
        "corrected_status_col": "spec 23010204.3",
    },
    "spec_doc_mg_l": {
        "label": "DOCeq spectro",
        "unit": "mg/l",
        "group": "par_spectro_summary",
        "measured_col": "spec 23010204.4",
        "measured_status_col": "spec 23010204.5",
        "corrected_col": "spec 23010204.6",
        "corrected_status_col": "spec 23010204.7",
    },
    "spec_uv254t_abs_m": {
        "label": "UV254t spectro",
        "unit": "Abs/m",
        "group": "par_spectro_summary",
        "measured_col": "spec 23010204.8",
        "measured_status_col": "spec 23010204.9",
        "corrected_col": "spec 23010204.10",
        "corrected_status_col": "spec 23010204.11",
    },
    "spec_uv254f_abs_m": {
        "label": "UV254f spectro",
        "unit": "Abs/m",
        "group": "par_spectro_summary",
        "measured_col": "spec 23010204.12",
        "measured_status_col": "spec 23010204.13",
        "corrected_col": "spec 23010204.14",
        "corrected_status_col": "spec 23010204.15",
    },
}


PATTERN_ALARM_VARIABLES: Dict[str, Dict[str, Any]] = {
    "pattern_alarm": {
        "label": "Pattern Alarm",
        "unit": "alarm value",
        "group": "alarm",
        "measured_col": "ana::tool",
        "measured_status_col": "ana::tool.1",
        "corrected_col": None,
        "corrected_status_col": None,
    },
    "pattern_alarm_2": {
        "label": "Pattern alarm 2",
        "unit": "alarm value",
        "group": "alarm",
        "measured_col": "ana::tool.2",
        "measured_status_col": "ana::tool.3",
        "corrected_col": None,
        "corrected_status_col": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye dataset THM Huelva desde CSV par seleccionados.")
    parser.add_argument("--manifest-csv", default=None, help="Manifest de descarga selectiva.")
    parser.add_argument("--windows-csv", default=None, help="CSV de ventanas THM exactas.")
    parser.add_argument("--output-dir", default=None, help="Carpeta de salida.")
    parser.add_argument("--include-par-spectro-summary", action="store_true", help="Incluye spec_* de par en features.")
    parser.add_argument(
        "--include-pattern-alarms",
        action="store_true",
        help=(
            "Incluye indicadores diagnósticos ana::tool como features. "
            "No corresponde a la lógica operativa de alertas, fuera del alcance UMA."
        ),
    )
    parser.add_argument("--no-raw-wide", action="store_true", help="No guarda timeseries raw wide.")
    parser.add_argument("--no-excel", action="store_true", help="No genera Excel.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    return text


def classify_file_kind(path: Any) -> str:
    name = Path(str(path)).name.lower()

    if re.search(r"(^|_)par(\.|_|$)", name):
        return "par"

    if re.search(r"(^|_)fp(\.|_|$)", name):
        return "fp"

    if re.search(r"(^|_)log(\.|_|$)", name):
        return "log"

    if "filetest" in name:
        return "filetest"

    return "unknown"


def to_number_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()

    text = text.replace({
        "": pd.NA,
        "nan": pd.NA,
        "None": pd.NA,
        "NaN": pd.NA,
        "NULL": pd.NA,
        "null": pd.NA,
    })

    # Soporta coma decimal sin romper miles con punto en casos simples.
    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(" ", "", regex=False)
    comma_decimal_mask = text.str.contains(",", na=False) & ~text.str.contains(r"\.\d+$", regex=True, na=False)
    text.loc[comma_decimal_mask] = text.loc[comma_decimal_mask].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    return pd.to_numeric(text, errors="coerce")


def parse_datetime_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA})

    parsed = pd.to_datetime(text, errors="coerce", dayfirst=False)

    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)

    return parsed


def is_ok_status(value: Any) -> Optional[bool]:
    text = normalize_text(value).lower()

    if not text or text in ["nan", "none", "nat"]:
        return None

    if "ok" in text and "not ok" not in text:
        return True

    if any(token in text for token in ["error", "fail", "alarm", "warning", "invalid", "bad"]):
        return False

    return None


def clean_status_series(series: pd.Series) -> pd.Series:
    return series.map(is_ok_status)


def existing_col(df: pd.DataFrame, col: Optional[str]) -> Optional[str]:
    if col is None:
        return None

    if col in df.columns:
        return col

    return None


def choose_value_and_status(
    df: pd.DataFrame,
    measured_col: Optional[str],
    measured_status_col: Optional[str],
    corrected_col: Optional[str],
    corrected_status_col: Optional[str],
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    measured_col = existing_col(df, measured_col)
    corrected_col = existing_col(df, corrected_col)
    measured_status_col = existing_col(df, measured_status_col)
    corrected_status_col = existing_col(df, corrected_status_col)

    measured = to_number_series(df[measured_col]) if measured_col else pd.Series(pd.NA, index=df.index, dtype="Float64")
    corrected = to_number_series(df[corrected_col]) if corrected_col else pd.Series(pd.NA, index=df.index, dtype="Float64")

    measured_status_raw = df[measured_status_col] if measured_status_col else pd.Series(pd.NA, index=df.index)
    corrected_status_raw = df[corrected_status_col] if corrected_status_col else pd.Series(pd.NA, index=df.index)

    measured_status_ok = clean_status_series(measured_status_raw)
    corrected_status_ok = clean_status_series(corrected_status_raw)

    # Un status desconocido no invalida el dato. Solo se rechaza cuando el
    # sistema lo marca explícitamente como no válido.
    corrected_usable = corrected.notna() & corrected_status_ok.ne(False)
    measured_usable = measured.notna() & measured_status_ok.ne(False)

    use_corrected = corrected_usable
    use_measured = ~use_corrected & measured_usable

    value = pd.Series(pd.NA, index=df.index, dtype="Float64")
    value.loc[use_corrected] = corrected.loc[use_corrected]
    value.loc[use_measured] = measured.loc[use_measured]

    status_ok = pd.Series(pd.NA, index=df.index, dtype="boolean")
    status_ok.loc[use_corrected] = corrected_status_ok.loc[use_corrected]
    status_ok.loc[use_measured] = measured_status_ok.loc[use_measured]

    source = pd.Series("missing", index=df.index, dtype="object")
    source.loc[use_corrected] = "corrected"
    source.loc[use_measured] = "measured"

    rejected_non_ok = (
        value.isna()
        & (corrected.notna() | measured.notna())
        & (
            corrected_status_ok.eq(False)
            | measured_status_ok.eq(False)
        )
    )
    source.loc[rejected_non_ok] = "rejected_non_ok"
    status_ok.loc[rejected_non_ok] = False

    return measured, corrected, value, status_ok, source


def variable_config(include_par_spectro_summary: bool, include_pattern_alarms: bool) -> Dict[str, Dict[str, Any]]:
    config: Dict[str, Dict[str, Any]] = {}
    config.update(CORE_VARIABLES)

    if include_par_spectro_summary:
        config.update(PAR_SPECTRO_SUMMARY_VARIABLES)

    if include_pattern_alarms:
        config.update(PATTERN_ALARM_VARIABLES)

    return config


def audit_variable_config() -> pd.DataFrame:
    rows = []

    for name, info in {
        **CORE_VARIABLES,
        **PAR_SPECTRO_SUMMARY_VARIABLES,
        **PATTERN_ALARM_VARIABLES,
    }.items():
        rows.append({
            "natural_name": name,
            "label": info["label"],
            "unit": info["unit"],
            "group": info["group"],
            "measured_col": info["measured_col"],
            "measured_status_col": info["measured_status_col"],
            "corrected_col": info["corrected_col"],
            "corrected_status_col": info["corrected_status_col"],
            "included_by_default_features": info["group"] == "core",
        })

    return pd.DataFrame(rows)


def read_manifest(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv, dtype=str, encoding="utf-8-sig")

    if "local_path" not in df.columns:
        raise ValueError(f"Manifest sin columna local_path: {manifest_csv}")

    df = df.copy()
    df["local_path"] = df["local_path"].map(lambda value: str(Path(normalize_text(value)).resolve()))
    df["file_kind"] = df["local_path"].map(classify_file_kind)
    df["local_exists"] = df["local_path"].map(lambda value: Path(value).exists())

    return df


def read_one_par_csv(path: Path, variables: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )

    df.columns = [normalize_text(col) for col in df.columns]

    if "Timestamp" not in df.columns:
        raise ValueError(f"CSV par sin columna Timestamp: {path}")

    out = pd.DataFrame(index=df.index)
    out["timestamp"] = parse_datetime_series(df["Timestamp"])
    out["source_file"] = path.name
    out["source_path"] = str(path)
    out["source_kind"] = "par"

    # La primera fila suele ser descriptiva: Measurement interval...
    out = out[out["timestamp"].notna()].copy()
    df = df.loc[out.index].copy()

    for var_name, info in variables.items():
        measured, corrected, value, status_ok, source = choose_value_and_status(
            df=df,
            measured_col=info.get("measured_col"),
            measured_status_col=info.get("measured_status_col"),
            corrected_col=info.get("corrected_col"),
            corrected_status_col=info.get("corrected_status_col"),
        )

        out[f"{var_name}_measured"] = measured
        out[f"{var_name}_corrected"] = corrected
        out[var_name] = value
        out[f"{var_name}_status_ok"] = status_ok
        out[f"{var_name}_source"] = source

    return out


def read_all_par_timeseries(manifest_df: pd.DataFrame, variables: Dict[str, Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    par_df = manifest_df[(manifest_df["file_kind"] == "par") & manifest_df["local_exists"]].copy()

    rows = []
    issues = []

    for index, row in par_df.iterrows():
        path = Path(row["local_path"])

        print(f"[par {len(rows) + 1}/{len(par_df)}] {path.name}")

        try:
            file_df = read_one_par_csv(path, variables)
            rows.append(file_df)

            issues.append({
                "source_file": path.name,
                "source_path": str(path),
                "read_ok": True,
                "rows": len(file_df),
                "min_timestamp": file_df["timestamp"].min(),
                "max_timestamp": file_df["timestamp"].max(),
                "error": "",
            })

        except Exception as exc:
            issues.append({
                "source_file": path.name,
                "source_path": str(path),
                "read_ok": False,
                "rows": 0,
                "min_timestamp": "",
                "max_timestamp": "",
                "error": f"{type(exc).__name__}: {exc}",
            })

    if not rows:
        raise ValueError("No se pudo leer ningún fichero par.")

    raw = pd.concat(rows, ignore_index=True)
    raw = raw.sort_values(["timestamp", "source_file"]).reset_index(drop=True)

    issues_df = pd.DataFrame(issues)

    return raw, issues_df


def aggregate_timeseries_by_timestamp(raw_df: pd.DataFrame, variables: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    numeric_cols = []
    status_cols = []

    for var in variables:
        numeric_cols.extend([
            f"{var}_measured",
            f"{var}_corrected",
            var,
        ])
        status_cols.append(f"{var}_status_ok")

    numeric_cols = [col for col in numeric_cols if col in raw_df.columns]
    status_cols = [col for col in status_cols if col in raw_df.columns]

    agg_map: Dict[str, Any] = {}

    for col in numeric_cols:
        agg_map[col] = "mean"

    for col in status_cols:
        agg_map[col] = "mean"

    agg_map["source_file"] = lambda values: "|".join(sorted(set(map(str, values))))
    agg_map["source_path"] = lambda values: "|".join(sorted(set(map(str, values))))

    clean = raw_df.groupby("timestamp", as_index=False).agg(agg_map)
    clean["n_source_rows"] = raw_df.groupby("timestamp").size().values
    clean["n_source_files"] = raw_df.groupby("timestamp")["source_file"].nunique().values
    clean = clean.sort_values("timestamp").reset_index(drop=True)

    return clean


def parse_windows(windows_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(windows_csv, dtype=str, encoding="utf-8-sig")

    required = {"sample_timestamp", "sample_point", "thm", "download_start", "download_end"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"WINDOWS_CSV incompleto. Faltan columnas: {sorted(missing)}")

    out = df.copy()
    out["sample_timestamp_dt"] = parse_datetime_series(out["sample_timestamp"])
    out["download_start_dt"] = parse_datetime_series(out["download_start"])
    out["download_end_dt"] = parse_datetime_series(out["download_end"])
    out["thm_numeric"] = to_number_series(out["thm"])

    out = out[
        out["sample_timestamp_dt"].notna()
        & out["download_start_dt"].notna()
        & out["download_end_dt"].notna()
        & out["thm_numeric"].notna()
    ].copy()

    out = out.sort_values("sample_timestamp_dt").reset_index(drop=True)
    out["sample_id"] = [f"Huelva_THM_{i + 1:03d}" for i in range(len(out))]

    return out


def expected_points_for_window(start: pd.Timestamp, end: pd.Timestamp) -> float:
    minutes = max((end - start).total_seconds() / 60.0, 0.0)

    if EXPECTED_SAMPLE_INTERVAL_MINUTES <= 0:
        return math.nan

    return minutes / EXPECTED_SAMPLE_INTERVAL_MINUTES + 1.0


def last_non_null_by_time(window_df: pd.DataFrame, column: str) -> float:
    series = window_df[["timestamp", column]].dropna(subset=[column]).sort_values("timestamp")

    if series.empty:
        return math.nan

    return float(series[column].iloc[-1])


def first_non_null_by_time(window_df: pd.DataFrame, column: str) -> float:
    series = window_df[["timestamp", column]].dropna(subset=[column]).sort_values("timestamp")

    if series.empty:
        return math.nan

    return float(series[column].iloc[0])


def build_feature_dataset(
    windows_df: pd.DataFrame,
    clean_ts: pd.DataFrame,
    variables: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []

    for _, sample in windows_df.iterrows():
        start = pd.Timestamp(sample["download_start_dt"])
        end = pd.Timestamp(sample["download_end_dt"])

        mask = (clean_ts["timestamp"] >= start) & (clean_ts["timestamp"] <= end)
        window = clean_ts[mask].copy()

        expected_points = expected_points_for_window(start, end)
        observed_timestamps = int(window["timestamp"].nunique()) if not window.empty else 0
        window_coverage = observed_timestamps / expected_points if expected_points else math.nan

        source_files = []
        if not window.empty and "source_file" in window.columns:
            for item in window["source_file"].dropna().astype(str).tolist():
                source_files.extend([part for part in item.split("|") if part])

        source_files = sorted(set(source_files))

        row: Dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "sample_timestamp": sample["sample_timestamp_dt"],
            "sample_point": sample.get("sample_point", ""),
            "thm": sample["thm_numeric"],
            "thm_unit": sample.get("thm_unit", ""),
            "relative_file": sample.get("relative_file", ""),
            "download_start": start,
            "download_end": end,
            "history_hours": sample.get("history_hours", ""),
            "n_observed_timestamps_24h": observed_timestamps,
            "expected_timestamps_24h": expected_points,
            "window_coverage_ratio_24h": window_coverage,
            "n_source_files_24h": len(source_files),
            "source_files_24h": "|".join(source_files),
        }

        for var_name, info in variables.items():
            if var_name not in clean_ts.columns:
                continue

            values = window[var_name].dropna() if not window.empty else pd.Series(dtype=float)
            status_col = f"{var_name}_status_ok"
            status_values = window[status_col].dropna() if status_col in window.columns and not window.empty else pd.Series(dtype=float)

            count = int(values.count())
            coverage = count / expected_points if expected_points else math.nan

            row[f"{var_name}_mean_24h"] = float(values.mean()) if count else math.nan
            row[f"{var_name}_median_24h"] = float(values.median()) if count else math.nan
            row[f"{var_name}_min_24h"] = float(values.min()) if count else math.nan
            row[f"{var_name}_max_24h"] = float(values.max()) if count else math.nan
            row[f"{var_name}_std_24h"] = float(values.std()) if count > 1 else math.nan
            row[f"{var_name}_first_24h"] = first_non_null_by_time(window, var_name) if count else math.nan
            row[f"{var_name}_last_24h"] = last_non_null_by_time(window, var_name) if count else math.nan
            row[f"{var_name}_delta_24h"] = (
                row[f"{var_name}_last_24h"] - row[f"{var_name}_first_24h"]
                if count and pd.notna(row[f"{var_name}_last_24h"]) and pd.notna(row[f"{var_name}_first_24h"])
                else math.nan
            )
            row[f"{var_name}_count_24h"] = count
            row[f"{var_name}_coverage_ratio_24h"] = coverage
            row[f"{var_name}_ok_ratio_24h"] = float(status_values.mean()) if len(status_values) else math.nan

            coverage_rows.append({
                "sample_id": sample["sample_id"],
                "sample_timestamp": sample["sample_timestamp_dt"],
                "sample_point": sample.get("sample_point", ""),
                "variable": var_name,
                "variable_group": info.get("group", ""),
                "count_24h": count,
                "expected_timestamps_24h": expected_points,
                "coverage_ratio_24h": coverage,
                "ok_ratio_24h": row[f"{var_name}_ok_ratio_24h"],
                "warn_low_coverage": bool(pd.notna(coverage) and coverage < MIN_WINDOW_COVERAGE_RATIO_WARN),
            })

        feature_rows.append(row)

    features_df = pd.DataFrame(feature_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    return features_df, coverage_df


def write_report(
    output_path: Path,
    manifest_csv: Path,
    windows_csv: Path,
    manifest_df: pd.DataFrame,
    issues_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    features_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    variables: Dict[str, Dict[str, Any]],
) -> None:
    lines = []
    lines.append("# ETL Huelva par → dataset THM")
    lines.append("")
    lines.append(f"- Versión: `{VERSION}`")
    lines.append(f"- Manifest: `{manifest_csv}`")
    lines.append(f"- Windows CSV: `{windows_csv}`")
    lines.append("")
    lines.append("## Decisiones")
    lines.append("")
    lines.append("- Solo se leen ficheros `par`.")
    lines.append("- Los ficheros `fp` quedan fuera.")
    lines.append(
        "- Valor corregido válido como principal; valor medido válido como fallback. "
        "Los valores explícitamente no válidos se excluyen de la señal final."
    )
    lines.append("- Nombres naturales en las salidas.")
    lines.append("- Se conservan medido/corregido/status en la serie intermedia.")
    lines.append("")
    lines.append("## Resumen")
    lines.append("")
    lines.append(f"- Ficheros en manifest: **{len(manifest_df)}**")
    lines.append(f"- Ficheros `par` existentes: **{int(((manifest_df['file_kind'] == 'par') & manifest_df['local_exists']).sum())}**")
    lines.append(f"- Ficheros leídos OK: **{int(issues_df['read_ok'].sum())}** / {len(issues_df)}")
    lines.append(f"- Filas raw `par`: **{len(raw_df)}**")
    lines.append(f"- Timestamps únicos limpios: **{len(clean_df)}**")
    lines.append(f"- Muestras THM exactas: **{len(windows_df)}**")
    lines.append(f"- Dataset final filas: **{len(features_df)}**")
    lines.append(f"- Variables usadas en features: **{len(variables)}**")
    lines.append("")
    lines.append("## Variables usadas")
    lines.append("")
    for name, info in variables.items():
        lines.append(f"- `{name}` — {info.get('label', '')} [{info.get('unit', '')}] ({info.get('group', '')})")
    lines.append("")
    lines.append("## Cobertura")
    lines.append("")
    if not coverage_df.empty:
        low = coverage_df[coverage_df["warn_low_coverage"] == True]
        lines.append(f"- Pares muestra-variable con cobertura baja (<{MIN_WINDOW_COVERAGE_RATIO_WARN:.0%}): **{len(low)}**")
        avg_cov = coverage_df.groupby("variable")["coverage_ratio_24h"].mean().sort_values(ascending=False)
        lines.append("")
        lines.append("Cobertura media por variable:")
        lines.append("")
        for var, value in avg_cov.items():
            lines.append(f"- `{var}`: {value:.3f}")
    lines.append("")
    lines.append("## Salidas")
    lines.append("")
    lines.append("- `huelva_par_timeseries_raw_wide_v0_1.csv`")
    lines.append("- `huelva_par_timeseries_clean_v0_1.csv`")
    lines.append("- `huelva_thm_par_feature_dataset_v0_1.csv`")
    lines.append("- `huelva_thm_par_window_coverage_v0_1.csv`")
    lines.append("- `huelva_par_column_mapping_v0_1.csv`")
    lines.append("- `huelva_par_etl_report_v0_1.xlsx`")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else Path(MANIFEST_CSV).resolve()
    windows_csv = Path(args.windows_csv).resolve() if args.windows_csv else Path(WINDOWS_CSV).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(OUTPUT_DIR).resolve()

    include_par_spectro_summary = bool(args.include_par_spectro_summary or INCLUDE_PAR_SPECTRO_SUMMARY_IN_FEATURES)
    include_pattern_alarms = bool(args.include_pattern_alarms or INCLUDE_PATTERN_ALARMS_IN_FEATURES)
    write_raw_wide = WRITE_RAW_WIDE and not args.no_raw_wide
    write_excel = WRITE_EXCEL and not args.no_excel

    if not manifest_csv.exists():
        raise FileNotFoundError(f"No existe manifest_csv: {manifest_csv}")

    if not windows_csv.exists():
        raise FileNotFoundError(f"No existe windows_csv: {windows_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    variables = variable_config(
        include_par_spectro_summary=include_par_spectro_summary,
        include_pattern_alarms=include_pattern_alarms,
    )

    print("========================================")
    print("ETL Huelva par -> dataset THM")
    print("========================================")
    print(f"Versión:       {VERSION}")
    print(f"Manifest:      {manifest_csv}")
    print(f"Windows CSV:   {windows_csv}")
    print(f"Output dir:    {output_dir}")
    print(f"Variables:     {len(variables)}")
    print(f"Par spectro:   {include_par_spectro_summary}")
    print(f"Diagnóstico ana::tool: {include_pattern_alarms}")
    print("")

    manifest_df = read_manifest(manifest_csv)

    print("Manifest:")
    print(manifest_df["file_kind"].value_counts(dropna=False).to_string())
    print("")

    raw_df, issues_df = read_all_par_timeseries(manifest_df, variables)
    clean_df = aggregate_timeseries_by_timestamp(raw_df, variables)
    windows_df = parse_windows(windows_csv)
    features_df, coverage_df = build_feature_dataset(windows_df, clean_df, variables)
    mapping_df = audit_variable_config()

    raw_out = output_dir / "huelva_par_timeseries_raw_wide_v0_1.csv"
    clean_out = output_dir / "huelva_par_timeseries_clean_v0_1.csv"
    features_out = output_dir / "huelva_thm_par_feature_dataset_v0_1.csv"
    coverage_out = output_dir / "huelva_thm_par_window_coverage_v0_1.csv"
    mapping_out = output_dir / "huelva_par_column_mapping_v0_1.csv"
    issues_out = output_dir / "huelva_par_read_issues_v0_1.csv"
    report_md = output_dir / "huelva_par_etl_report_v0_1.md"
    report_xlsx = output_dir / "huelva_par_etl_report_v0_1.xlsx"

    if write_raw_wide:
        raw_df.to_csv(raw_out, index=False, encoding="utf-8-sig")

    clean_df.to_csv(clean_out, index=False, encoding="utf-8-sig")
    features_df.to_csv(features_out, index=False, encoding="utf-8-sig")
    coverage_df.to_csv(coverage_out, index=False, encoding="utf-8-sig")
    mapping_df.to_csv(mapping_out, index=False, encoding="utf-8-sig")
    issues_df.to_csv(issues_out, index=False, encoding="utf-8-sig")

    write_report(
        output_path=report_md,
        manifest_csv=manifest_csv,
        windows_csv=windows_csv,
        manifest_df=manifest_df,
        issues_df=issues_df,
        raw_df=raw_df,
        clean_df=clean_df,
        windows_df=windows_df,
        features_df=features_df,
        coverage_df=coverage_df,
        variables=variables,
    )

    if write_excel:
        try:
            with pd.ExcelWriter(report_xlsx, engine="openpyxl") as writer:
                features_df.to_excel(writer, sheet_name="feature_dataset", index=False)
                coverage_df.to_excel(writer, sheet_name="window_coverage", index=False)
                mapping_df.to_excel(writer, sheet_name="column_mapping", index=False)
                issues_df.to_excel(writer, sheet_name="read_issues", index=False)
                clean_df.head(5000).to_excel(writer, sheet_name="timeseries_clean_head", index=False)
        except Exception as exc:
            print(f"[WARN] No se pudo escribir Excel: {type(exc).__name__}: {exc}")
            report_xlsx = None

    print("")
    print("========================================")
    print("ETL completado")
    print("========================================")
    print(f"Ficheros par leídos OK:       {int(issues_df['read_ok'].sum())}/{len(issues_df)}")
    print(f"Filas raw par:                {len(raw_df)}")
    print(f"Timestamps únicos limpios:    {len(clean_df)}")
    print(f"Muestras THM exactas:         {len(windows_df)}")
    print(f"Dataset final:                {features_out}")
    print(f"Cobertura ventanas:           {coverage_out}")
    print(f"Serie limpia:                 {clean_out}")

    if write_raw_wide:
        print(f"Serie raw audit:              {raw_out}")

    print(f"Mapeo columnas:               {mapping_out}")
    print(f"Reporte Markdown:             {report_md}")

    if report_xlsx:
        print(f"Reporte Excel:                {report_xlsx}")

    if not coverage_df.empty:
        print("")
        print("Cobertura media por variable:")
        print(
            coverage_df
            .groupby("variable")["coverage_ratio_24h"]
            .mean()
            .sort_values(ascending=False)
            .round(3)
            .to_string()
        )

        low = coverage_df[coverage_df["warn_low_coverage"] == True]
        if not low.empty:
            print("")
            print(f"[WARN] Hay {len(low)} pares muestra-variable con cobertura baja.")


if __name__ == "__main__":
    main()
