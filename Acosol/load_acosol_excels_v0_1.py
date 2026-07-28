from __future__ import annotations

"""ETL reproducible para normalizar y auditar los Excel de Acosol THM."""

import argparse
import math
import re
import warnings
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BASE_FEATURES = ["turbidez", "cloro", "ph", "conductividad", "bromuro", "cot"]
TARGET_COL = "thm"
# Rangos técnicos conservadores para detectar datos imposibles o errores claros de formato.
# No representan límites operativos ni umbrales de alarma definidos por planta.
HARD_RANGES = {
    "turbidez": (0.0, 100.0),
    "cloro": (0.0, 10.0),
    "ph": (0.0, 14.0),
    "conductividad": (10.0, 5000.0),
    "thm": (0.0, 250.0),
    "bromuro": (0.0, 50.0),
    "cot": (0.0, 50.0),
}

# Rangos orientativos de auditoría. Solo generan flags y no eliminan registros.
SOFT_RANGES = {
    "cloro": (0.1, 5.0),
    "ph": (6.0, 9.5),
    "conductividad": (100.0, 2000.0),
    "bromuro": (0.0, 25.0),
    "cot": (0.0, 20.0),
    "thm": (0.0, 150.0),
}

CANONICAL_RENAMES = {
    "fecha": "fecha",
    "hora exacta": "hora_exacta",
    "hora": "hora_exacta",
    "turbidez": "turbidez",
    "cloro": "cloro",
    "ph": "ph",
    "conductividad": "conductividad",
    "thm": "thm",
    "bromuro": "bromuro",
    "cot": "cot",
}


@dataclass(frozen=True)
class SourceConfig:
    file_path: Path
    sheet_name: str
    priority: int


def normalize_header(col: Any) -> str:
    s = str(col).strip().lower()
    s = s.replace("µ", "u")
    s = re.sub(r"\([^)]*\)", "", s)  # quita unidades entre paréntesis
    s = s.replace(".", " ").replace("_", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return CANONICAL_RENAMES.get(s, s.replace(" ", "_"))


def choose_priority(path: Path) -> int:
    name = path.name.lower()
    if "actualización" in name or "actualizacion" in name or "04062026" in name:
        return 30
    if "mensuales 2026" in name:
        return 20
    if "abril" in name and "mayo" in name:
        return 10
    return 0


def discover_sources(input_dir: Path) -> list[SourceConfig]:
    sources: list[SourceConfig] = []
    for path in sorted(input_dir.glob("*.xlsx")):
        if not path.name.lower().startswith("datos"):
            continue
        xl = pd.ExcelFile(path)
        for sheet_name in xl.sheet_names:
            try:
                preview = pd.read_excel(path, sheet_name=sheet_name, nrows=3)
            except Exception as exc:
                warnings.warn(
                    f"No se pudo revisar {path.name!r}, hoja {sheet_name!r}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if preview.empty:
                continue
            norm_cols = {normalize_header(c) for c in preview.columns}
            if {"fecha", "hora_exacta", "thm"}.issubset(norm_cols):
                sources.append(SourceConfig(path, sheet_name, choose_priority(path)))
    return sources


def parse_time_fraction(value: Any) -> float | None:
    """Devuelve fracción de día [0,1) a partir de hora Excel/string/time/Timestamp."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None

    if isinstance(value, pd.Timestamp):
        return (value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6) / 86400.0
    if isinstance(value, datetime):
        return (value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6) / 86400.0
    if isinstance(value, time):
        return (value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6) / 86400.0
    if isinstance(value, timedelta):
        return (value.total_seconds() % 86400) / 86400.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        # Excel puede representar horas como fracción de día o como día+fracción.
        return float(value) % 1.0

    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "sin datos", "s/d"}:
        return None

    text_norm = text.replace(";", ":")
    parsed = pd.to_datetime(text_norm, errors="coerce")
    if pd.notna(parsed):
        return (parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1e6) / 86400.0

    return None


def build_timestamp(fecha: Any, hora: Any) -> pd.Timestamp | pd.NaT:
    d = pd.to_datetime(fecha, errors="coerce")
    frac = parse_time_fraction(hora)
    if pd.isna(d) or frac is None:
        return pd.NaT
    return (pd.Timestamp(d).normalize() + pd.to_timedelta(frac, unit="D")).round("s")


def to_numeric_series(s: pd.Series) -> pd.Series:
    # Convierte textos con coma decimal y marca 'SIN DATOS' como NaN.
    return pd.to_numeric(
        s.astype(str)
         .str.strip()
         .str.replace(",", ".", regex=False)
         .replace({"": np.nan, "nan": np.nan, "NaT": np.nan, "SIN DATOS": np.nan, "sin datos": np.nan}),
        errors="coerce",
    )


def load_one_source(src: SourceConfig) -> pd.DataFrame:
    raw = pd.read_excel(src.file_path, sheet_name=src.sheet_name)
    raw = raw.rename(columns={c: normalize_header(c) for c in raw.columns})
    raw = raw.loc[:, ~raw.columns.str.startswith("unnamed")].copy()

    # Conserva datos originales de fecha/hora antes del forward-fill.
    raw["fecha_raw"] = raw.get("fecha")
    raw["hora_raw"] = raw.get("hora_exacta")
    raw["fecha"] = pd.to_datetime(raw.get("fecha"), errors="coerce").ffill()

    expected = ["fecha", "hora_exacta", *BASE_FEATURES, TARGET_COL]
    for col in expected:
        if col not in raw.columns:
            raw[col] = np.nan

    out = raw[expected + ["fecha_raw", "hora_raw"]].copy()
    out["source_file"] = src.file_path.name
    out["source_sheet"] = src.sheet_name
    out["source_priority"] = src.priority
    out["original_row"] = np.arange(2, len(out) + 2)  # cabecera en fila 1 de Excel

    # Elimina filas completamente vacías en variables y hora.
    value_cols = ["hora_exacta", *BASE_FEATURES, TARGET_COL]
    nonempty = out[value_cols].notna().any(axis=1)
    out = out.loc[nonempty].copy()

    for col in [*BASE_FEATURES, TARGET_COL]:
        out[col] = to_numeric_series(out[col])

    out["timestamp"] = [build_timestamp(f, h) for f, h in zip(out["fecha"], out["hora_exacta"])]
    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["dayofyear"] = ts.dt.dayofyear
    df["dayofweek"] = ts.dt.dayofweek
    df["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    return df


def clean_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica correcciones muy evidentes y genera quality_flags."""
    df = df.copy()
    flags: list[list[str]] = [[] for _ in range(len(df))]

    def add_flag(mask: pd.Series, flag: str) -> None:
        idx = np.flatnonzero(mask.fillna(False).to_numpy())
        for i in idx:
            flags[i].append(flag)

    # Heurísticas de escala identificadas durante la auditoría de estos Excel.
    # Cada transformación queda registrada en quality_flags para mantener trazabilidad.
    mask = df["cloro"].gt(10) & df["cloro"].div(100).between(0, 10)
    add_flag(mask, "auto_scale_cloro_div100")
    df.loc[mask, "cloro"] = df.loc[mask, "cloro"] / 100.0

    mask = df["ph"].gt(14) & df["ph"].div(100).between(0, 14)
    add_flag(mask, "auto_scale_ph_div100")
    df.loc[mask, "ph"] = df.loc[mask, "ph"] / 100.0

    mask = df["ph"].gt(14) & df["ph"].div(10).between(0, 14)
    add_flag(mask, "auto_scale_ph_div10")
    df.loc[mask, "ph"] = df.loc[mask, "ph"] / 10.0

    mask = df["conductividad"].gt(0) & df["conductividad"].lt(10) & df["conductividad"].mul(100).between(100, 2000)
    add_flag(mask, "auto_scale_conductividad_mul100")
    df.loc[mask, "conductividad"] = df.loc[mask, "conductividad"] * 100.0

    mask = df["bromuro"].gt(50) & df["bromuro"].div(100).between(0, 50)
    add_flag(mask, "auto_scale_bromuro_div100")
    df.loc[mask, "bromuro"] = df.loc[mask, "bromuro"] / 100.0

    mask = df["cot"].gt(100) & df["cot"].div(1000).between(0, 50)
    add_flag(mask, "auto_scale_cot_div1000")
    df.loc[mask, "cot"] = df.loc[mask, "cot"] / 1000.0

    # Flags por faltantes.
    add_flag(df["timestamp"].isna(), "missing_timestamp")
    for col in [*BASE_FEATURES, TARGET_COL]:
        add_flag(df[col].isna(), f"missing_{col}")

    # Rangos duros. Estos sí se convierten a NaN para features/target.
    for col, (lo, hi) in HARD_RANGES.items():
        invalid = df[col].notna() & ~df[col].between(lo, hi)
        add_flag(invalid, f"hard_out_of_range_{col}")
        df.loc[invalid, col] = np.nan

    # Rangos blandos solo informativos.
    for col, (lo, hi) in SOFT_RANGES.items():
        soft = df[col].notna() & ~df[col].between(lo, hi)
        add_flag(soft, f"soft_out_of_range_{col}")

    df["quality_flags"] = [";".join(x) if x else "ok" for x in flags]
    return df


def resolve_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resuelve solapes por timestamp y conserva todas las filas sin timestamp válido."""
    df = df.copy()
    valid = df[df["timestamp"].notna()].copy()
    invalid = df[df["timestamp"].isna()].copy()

    valid = valid.sort_values(
        ["timestamp", "source_priority", "source_file", "original_row"],
        ascending=[True, False, True, True],
    )

    dup_mask = valid.duplicated("timestamp", keep=False)
    dup = valid.loc[dup_mask].copy()
    reports = []
    if not dup.empty:
        compare_cols = [*BASE_FEATURES, TARGET_COL]
        for ts, group in dup.groupby("timestamp", sort=True):
            # Hay conflicto si una variable presenta más de un valor no nulo distinto.
            conflict_cols = []
            for col in compare_cols:
                vals = group[col].dropna().round(10).unique()
                if len(vals) > 1:
                    conflict_cols.append(col)
            reports.append(
                {
                    "timestamp": ts,
                    "n_rows": len(group),
                    "kept_source_file": group.iloc[0]["source_file"],
                    "kept_source_sheet": group.iloc[0]["source_sheet"],
                    "conflict_cols": ";".join(conflict_cols) if conflict_cols else "none",
                    "all_sources": " | ".join(group["source_file"].astype(str).tolist()),
                }
            )
    duplicate_report = pd.DataFrame(reports)

    # Para cada timestamp válido se conserva la fila de mayor prioridad. Las filas sin
    # timestamp no se deduplican entre sí para no perder información de auditoría.
    deduped_valid = valid.drop_duplicates("timestamp", keep="first")
    deduped = pd.concat([deduped_valid, invalid], ignore_index=True, sort=False)
    deduped = deduped.sort_values(
        ["timestamp", "source_priority", "source_file", "original_row"],
        ascending=[True, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return deduped, duplicate_report


def summarize(df_all: pd.DataFrame, df_dedup: pd.DataFrame, dataset_a: pd.DataFrame, dataset_b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def add(metric: str, value: Any) -> None:
        rows.append({"metric": metric, "value": value})

    add("raw_rows_all_sources", len(df_all))
    add("rows_with_valid_timestamp_all_sources", int(df_all["timestamp"].notna().sum()))
    add("deduped_rows", len(df_dedup))
    add("dataset_A_rows", len(dataset_a))
    add("dataset_B_rows", len(dataset_b))
    add("dataset_A_min_timestamp", dataset_a["timestamp"].min())
    add("dataset_A_max_timestamp", dataset_a["timestamp"].max())
    add("dataset_B_min_timestamp", dataset_b["timestamp"].min() if len(dataset_b) else pd.NaT)
    add("dataset_B_max_timestamp", dataset_b["timestamp"].max() if len(dataset_b) else pd.NaT)
    add("dataset_A_2025_rows", int((dataset_a["year"] == 2025).sum()))
    add("dataset_A_2026_rows", int((dataset_a["year"] == 2026).sum()))
    add("dataset_B_2025_rows", int((dataset_b["year"] == 2025).sum()))
    add("dataset_B_2026_rows", int((dataset_b["year"] == 2026).sum()))
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = ["timestamp", TARGET_COL, *BASE_FEATURES]
    rows = []
    for col in cols:
        rows.append({
            "dataset": name,
            "column": col,
            "missing_rows": int(df[col].isna().sum()),
            "available_rows": int(df[col].notna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100) if len(df) else 0.0,
        })
    return pd.DataFrame(rows)


def flag_report(df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    for text in df["quality_flags"].fillna("ok"):
        for flag in str(text).split(";"):
            flag = flag.strip() or "ok"
            counts[flag] = counts.get(flag, 0) + 1
    return pd.DataFrame([{"quality_flag": k, "rows": v} for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))])


def describe_numeric(df: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = [TARGET_COL, *BASE_FEATURES]
    desc = df[cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T.reset_index(names="column")
    desc.insert(0, "dataset", name)
    return desc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normaliza los Excel de Acosol y genera los datasets A y B."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Carpeta que contiene los Excel originales de Acosol.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("acosol_outputs_v0_1"),
        help="Carpeta donde se guardarán datasets y reportes de auditoría.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(args.input_dir)
    if not sources:
        raise SystemExit(f"No se han encontrado hojas válidas en {args.input_dir}")

    loaded = [load_one_source(src) for src in sources]
    all_sources = pd.concat(loaded, ignore_index=True)
    cleaned = clean_numeric_values(all_sources)
    deduped, duplicate_report = resolve_duplicates(cleaned)
    deduped = add_calendar_features(deduped)

    # Dataset A: timestamp y target válidos. Features pueden quedar parcialmente incompletas.
    dataset_a = deduped.loc[deduped["timestamp"].notna() & deduped[TARGET_COL].notna()].copy()
    dataset_a = dataset_a.sort_values("timestamp")

    # Dataset B: timestamp, target y todas las features base completas.
    dataset_b = dataset_a.dropna(subset=BASE_FEATURES + [TARGET_COL]).copy()
    dataset_b = dataset_b.sort_values("timestamp")

    # Orden de columnas legible.
    front_cols = [
        "timestamp", "year", "month", "dayofyear", "dayofweek", "hour",
        TARGET_COL, *BASE_FEATURES,
        "quality_flags", "source_file", "source_sheet", "original_row", "source_priority",
    ]
    remaining = [c for c in deduped.columns if c not in front_cols]
    front_cols_existing = [c for c in front_cols if c in deduped.columns]

    all_sources.to_csv(args.output_dir / "acosol_raw_all_sources_v0_1.csv", index=False)
    deduped[front_cols_existing + remaining].to_csv(args.output_dir / "acosol_standardized_deduped_v0_1.csv", index=False)
    dataset_a[front_cols_existing + remaining].to_csv(args.output_dir / "acosol_dataset_A_wide_v0_1.csv", index=False)
    dataset_b[front_cols_existing + remaining].to_csv(args.output_dir / "acosol_dataset_B_complete_v0_1.csv", index=False)

    summary = summarize(all_sources, deduped, dataset_a, dataset_b)
    source_report = (
        all_sources.groupby(["source_file", "source_sheet"], dropna=False)
        .agg(raw_rows=("timestamp", "size"), valid_timestamp_rows=("timestamp", lambda x: int(x.notna().sum())))
        .reset_index()
    )
    missing = pd.concat([
        missing_report(dataset_a, "Dataset A"),
        missing_report(dataset_b, "Dataset B"),
    ], ignore_index=True)
    flags = flag_report(deduped)
    numeric_desc = pd.concat([
        describe_numeric(dataset_a, "Dataset A"),
        describe_numeric(dataset_b, "Dataset B"),
    ], ignore_index=True)

    summary.to_csv(args.output_dir / "audit_summary_v0_1.csv", index=False)
    source_report.to_csv(args.output_dir / "source_report_v0_1.csv", index=False)
    duplicate_report.to_csv(args.output_dir / "duplicate_report_v0_1.csv", index=False)
    missing.to_csv(args.output_dir / "missing_report_v0_1.csv", index=False)
    flags.to_csv(args.output_dir / "quality_flags_report_v0_1.csv", index=False)
    numeric_desc.to_csv(args.output_dir / "numeric_describe_v0_1.csv", index=False)

    print("\nETL Acosol THM v0.1 completado")
    print("===============================")
    print(f"Fuentes válidas encontradas: {len(sources)}")
    for src in sources:
        print(f"- {src.file_path.name} / {src.sheet_name} / priority={src.priority}")
    print("\nResumen:")
    print(summary.to_string(index=False))
    print(f"\nSalida: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
