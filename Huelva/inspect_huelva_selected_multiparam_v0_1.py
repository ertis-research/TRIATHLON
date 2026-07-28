from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

MANIFEST_CSV = SCRIPT_DIR / "huelva_multiparam_sftp" / "logs" / "huelva_sftp_selective_download_manifest_v0_15.csv"
OUTPUT_DIR = SCRIPT_DIR / "huelva_multiparam_sftp" / "selected_inspection_v0_1_clean"

SAMPLE_ROWS_PER_FILE = 300
PREVIEW_ROWS_PER_FILE = 8

ENCODINGS_TO_TRY = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
DELIMITERS_TO_TRY = [";", ",", "\t", "|"]

INSPECTION_MODE = "quick"  # quick | full

WRITE_EXCEL = True

VERSION = "v0.1"

warnings.filterwarnings(
    "ignore",
    message="Could not infer format.*",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspecciona los CSV par seleccionados por el downloader selectivo v0.15."
    )
    parser.add_argument("--manifest-csv", default=None, help="Manifest de descarga selectiva.")
    parser.add_argument("--output-dir", default=None, help="Carpeta de salida.")
    parser.add_argument("--mode", choices=["quick", "full"], default=None, help="quick o full.")
    parser.add_argument("--no-excel", action="store_true", help="No genera Excel.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value)
    text = text.strip()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_column_name(value: Any) -> str:
    text = normalize_text(value).lower()
    text = text.replace("\ufeff", "")
    text = re.sub(r"[^a-z0-9áéíóúüñ]+", "_", text)
    text = text.strip("_")

    return text


def classify_file_kind(path: Path) -> str:
    name = path.name.lower()

    if "filetest" in name:
        return "filetest"

    if re.search(r"(^|_)par(\.|_|$)", name):
        return "par"

    if re.search(r"(^|_)fp(\.|_|$)", name):
        return "fp"

    if re.search(r"(^|_)log(\.|_|$)", name):
        return "log"

    return "unknown"


def sniff_delimiter(path: Path, encoding: str) -> Optional[str]:
    try:
        with path.open("r", encoding=encoding, errors="replace", newline="") as handle:
            sample = handle.read(16384)

        if not sample.strip():
            return None

        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
            return dialect.delimiter
        except csv.Error:
            pass

        lines = [line for line in sample.splitlines()[:20] if line.strip()]
        if not lines:
            return None

        scores = {}
        for delimiter in DELIMITERS_TO_TRY:
            counts = [line.count(delimiter) for line in lines]
            scores[delimiter] = sum(counts)

        best = max(scores, key=scores.get)

        if scores[best] > 0:
            return best

        return ","

    except Exception:
        return None


def read_csv_sample(path: Path, nrows: int) -> Tuple[Optional[pd.DataFrame], str, str, str]:
    last_error = ""

    for encoding in ENCODINGS_TO_TRY:
        sniffed = sniff_delimiter(path, encoding)
        delimiters = [sniffed] if sniffed else []
        delimiters.extend([item for item in DELIMITERS_TO_TRY if item not in delimiters])

        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    path,
                    sep=delimiter,
                    encoding=encoding,
                    dtype=str,
                    nrows=nrows,
                    engine="python",
                    on_bad_lines="skip",
                    index_col=False,
                )

                df.columns = [normalize_text(col) for col in df.columns]

                # Si se lee como una sola columna pero hay delimitadores claros, probar otro sep.
                if df.shape[1] <= 1 and delimiter != ",":
                    continue

                return df, encoding, delimiter, ""

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

    return None, "", "", last_error or "No se pudo leer el CSV."


def count_rows_fast(path: Path, encoding: str) -> Optional[int]:
    try:
        with path.open("r", encoding=encoding or "utf-8", errors="replace", newline="") as handle:
            line_count = sum(1 for _ in handle)

        if line_count <= 0:
            return 0

        return line_count - 1

    except Exception:
        return None


def is_numeric_like(series: pd.Series) -> float:
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(100)

    if sample.empty:
        return 0.0

    normalized = (
        sample
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    parsed = pd.to_numeric(normalized, errors="coerce")

    return float(parsed.notna().mean())


def values_look_datetime(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(40)

    if sample.empty:
        return False

    patterns = [
        re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"),
        re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"),
        re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}[ T]\d{1,2}:\d{2}"),
        re.compile(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{2}"),
    ]

    hits = 0
    for value in sample:
        if any(pattern.search(value) for pattern in patterns):
            hits += 1

    return hits >= max(2, int(len(sample) * 0.5))


def parse_datetime_candidate(series: pd.Series, column_name: str) -> Tuple[float, str, str, bool]:
    normalized_name = normalize_column_name(column_name)
    name_candidate = any(
        token in normalized_name
        for token in ["fecha", "hora", "date", "time", "timestamp", "datetime", "sample", "created", "updated"]
    )
    value_candidate = values_look_datetime(series)

    if not name_candidate and not value_candidate:
        return 0.0, "", "", False

    sample = series.dropna().astype(str).str.strip()
    sample = sample[sample != ""].head(100)

    if sample.empty:
        return 0.0, "", "", True

    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
    ratio = float(parsed.notna().mean())

    if parsed.notna().any():
        return ratio, str(parsed.min()), str(parsed.max()), True

    return ratio, "", "", True


def load_manifest(manifest_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv, dtype=str, encoding="utf-8-sig")

    required_cols = {"remote_path", "local_path", "download_status"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Manifest incompleto. Faltan columnas: {sorted(missing)}")

    df = df.copy()
    df["local_path_resolved"] = df["local_path"].map(lambda value: str(Path(normalize_text(value)).resolve()))
    df["local_exists"] = df["local_path_resolved"].map(lambda value: Path(value).exists())
    df["file_kind_from_name"] = df["local_path_resolved"].map(lambda value: classify_file_kind(Path(value)))

    return df


def inspect_one_file(row: pd.Series, mode: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    local_path = Path(row["local_path_resolved"])
    remote_path = normalize_text(row.get("remote_path", ""))
    file_kind = classify_file_kind(local_path)

    file_info: Dict[str, Any] = {
        "remote_path": remote_path,
        "local_path": str(local_path),
        "filename": local_path.name,
        "file_kind": file_kind,
        "download_status": normalize_text(row.get("download_status", "")),
        "selection_reason": normalize_text(row.get("selection_reason", "")),
        "overlapping_windows": normalize_text(row.get("overlapping_windows", "")),
        "sample_min_datetime_remote_index": normalize_text(row.get("sample_min_datetime", "")),
        "sample_max_datetime_remote_index": normalize_text(row.get("sample_max_datetime", "")),
        "exists": local_path.exists(),
        "size_bytes": local_path.stat().st_size if local_path.exists() else None,
        "read_ok": False,
        "encoding": "",
        "delimiter": "",
        "read_error": "",
        "sample_rows": None,
        "sample_columns": None,
        "columns_joined": "",
        "best_datetime_column": "",
        "sample_min_datetime_detected": "",
        "sample_max_datetime_detected": "",
        "full_rows_estimate": None,
    }

    column_rows: List[Dict[str, Any]] = []
    preview_rows: List[Dict[str, Any]] = []

    if not local_path.exists():
        file_info["read_error"] = "No existe local_path."
        return file_info, column_rows, preview_rows

    df, encoding, delimiter, error = read_csv_sample(local_path, SAMPLE_ROWS_PER_FILE)

    file_info["encoding"] = encoding
    file_info["delimiter"] = delimiter
    file_info["read_error"] = error

    if df is None:
        return file_info, column_rows, preview_rows

    file_info["read_ok"] = True
    file_info["sample_rows"] = int(len(df))
    file_info["sample_columns"] = int(len(df.columns))
    file_info["columns_joined"] = " | ".join([str(col) for col in df.columns])

    if mode == "full":
        file_info["full_rows_estimate"] = count_rows_fast(local_path, encoding)

    best_datetime = {
        "column": "",
        "ratio": 0.0,
        "min": "",
        "max": "",
    }

    for position, column in enumerate(df.columns):
        series = df[column]
        numeric_ratio = is_numeric_like(series)
        dt_ratio, dt_min, dt_max, dt_considered = parse_datetime_candidate(series, str(column))

        if dt_ratio > best_datetime["ratio"]:
            best_datetime = {
                "column": str(column),
                "ratio": dt_ratio,
                "min": dt_min,
                "max": dt_max,
            }

        examples = [
            normalize_text(value)
            for value in series.dropna().astype(str).head(8).tolist()
        ]

        column_rows.append({
            "remote_path": remote_path,
            "filename": local_path.name,
            "file_kind": file_kind,
            "column_position": position,
            "column": str(column),
            "normalized_column": normalize_column_name(column),
            "sample_non_empty": int(series.dropna().astype(str).str.strip().ne("").sum()),
            "sample_numeric_ratio": numeric_ratio,
            "sample_datetime_considered": dt_considered,
            "sample_datetime_ratio": dt_ratio,
            "sample_min_datetime_detected": dt_min,
            "sample_max_datetime_detected": dt_max,
            "example_values": " | ".join(examples),
        })

    if best_datetime["ratio"] > 0:
        file_info["best_datetime_column"] = best_datetime["column"]
        file_info["sample_min_datetime_detected"] = best_datetime["min"]
        file_info["sample_max_datetime_detected"] = best_datetime["max"]

    preview = df.head(PREVIEW_ROWS_PER_FILE)

    for row_number, (_, preview_row) in enumerate(preview.iterrows(), start=1):
        preview_rows.append({
            "remote_path": remote_path,
            "filename": local_path.name,
            "file_kind": file_kind,
            "preview_row_number": row_number,
            "row_json": json.dumps(preview_row.to_dict(), ensure_ascii=False, default=str),
        })

    return file_info, column_rows, preview_rows


def summarize_schemas(files_df: pd.DataFrame) -> pd.DataFrame:
    if files_df.empty:
        return pd.DataFrame()

    return (
        files_df
        .fillna("")
        .groupby(["file_kind", "delimiter", "sample_columns", "columns_joined"], dropna=False)
        .size()
        .reset_index(name="file_count")
        .sort_values(["file_count", "file_kind"], ascending=[False, True])
    )


def summarize_columns(columns_df: pd.DataFrame) -> pd.DataFrame:
    if columns_df.empty:
        return pd.DataFrame()

    return (
        columns_df
        .groupby(["file_kind", "normalized_column", "column"], dropna=False)
        .agg(
            file_count=("remote_path", "nunique"),
            avg_numeric_ratio=("sample_numeric_ratio", "mean"),
            avg_datetime_ratio=("sample_datetime_ratio", "mean"),
            examples=("example_values", lambda values: " || ".join([str(value) for value in list(values)[:4]])),
        )
        .reset_index()
        .sort_values(["file_kind", "file_count", "normalized_column"], ascending=[True, False, True])
    )


def write_markdown_report(
    output_path: Path,
    manifest_df: pd.DataFrame,
    files_df: pd.DataFrame,
    schemas_df: pd.DataFrame,
    columns_summary_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Inspección CSV seleccionados Huelva")
    lines.append("")
    lines.append(f"- Versión: `{VERSION}`")
    lines.append(f"- Manifest: `{MANIFEST_CSV}`")
    lines.append("")

    lines.append("## Resumen")
    lines.append("")
    lines.append(f"- Filas manifest: **{len(manifest_df)}**")
    lines.append(f"- Ficheros locales existentes: **{int(manifest_df['local_exists'].sum())}**")

    if not files_df.empty:
        lines.append(f"- Ficheros leídos correctamente: **{int(files_df['read_ok'].sum())}**")
        lines.append(f"- Tamaño total local: **{files_df['size_bytes'].fillna(0).sum() / (1024 * 1024):.2f} MB**")
        lines.append("")

        lines.append("## Tipos de fichero")
        lines.append("")
        for kind, count in files_df["file_kind"].value_counts(dropna=False).items():
            lines.append(f"- `{kind}`: {count}")

        lines.append("")
        lines.append("## Esquemas detectados")
        lines.append("")
        for _, row in schemas_df.head(15).iterrows():
            columns_text = str(row["columns_joined"])
            if len(columns_text) > 260:
                columns_text = columns_text[:260] + "..."

            lines.append(
                f"- `{row['file_kind']}` | sep=`{row['delimiter']}` | cols={row['sample_columns']} "
                f"| ficheros={row['file_count']} | `{columns_text}`"
            )

        lines.append("")
        lines.append("## Columnas más frecuentes")
        lines.append("")
        for _, row in columns_summary_df.head(50).iterrows():
            lines.append(
                f"- `{row['file_kind']}` | `{row['column']}` "
                f"| ficheros={row['file_count']} "
                f"| numeric≈{row['avg_numeric_ratio']:.2f} "
                f"| datetime≈{row['avg_datetime_ratio']:.2f}"
            )

        lines.append("")
        lines.append("## Próximo paso")
        lines.append("")
        lines.append(
            "Con estos esquemas se decide qué columnas son timestamp, parámetro/sensor y valor. "
            "Después se construye el ETL limpio para unir las medidas con las ventanas THM."
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    manifest_csv = Path(args.manifest_csv).resolve() if args.manifest_csv else Path(MANIFEST_CSV).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(OUTPUT_DIR).resolve()
    mode = args.mode or INSPECTION_MODE
    write_excel = WRITE_EXCEL and not args.no_excel

    if not manifest_csv.exists():
        raise FileNotFoundError(f"No existe manifest: {manifest_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("========================================")
    print("Inspección CSV seleccionados Huelva")
    print("========================================")
    print(f"Versión:       {VERSION}")
    print(f"Manifest:      {manifest_csv}")
    print(f"Output dir:    {output_dir}")
    print(f"Modo:          {mode}")
    print("")

    manifest_df = load_manifest(manifest_csv)

    print(f"Ficheros en manifest:     {len(manifest_df)}")
    print(f"Ficheros locales existen: {int(manifest_df['local_exists'].sum())}")
    print("")

    file_rows: List[Dict[str, Any]] = []
    column_rows: List[Dict[str, Any]] = []
    preview_rows: List[Dict[str, Any]] = []

    for index, row in manifest_df.iterrows():
        counter = len(file_rows) + 1
        local_path = Path(row["local_path_resolved"])
        print(f"[{counter}/{len(manifest_df)}] {local_path.name}")

        file_info, file_column_rows, file_preview_rows = inspect_one_file(row, mode)
        file_rows.append(file_info)
        column_rows.extend(file_column_rows)
        preview_rows.extend(file_preview_rows)

    files_df = pd.DataFrame(file_rows)
    columns_df = pd.DataFrame(column_rows)
    preview_df = pd.DataFrame(preview_rows)
    schemas_df = summarize_schemas(files_df)
    columns_summary_df = summarize_columns(columns_df)

    manifest_out = output_dir / "huelva_selected_manifest_resolved_v0_1.csv"
    files_out = output_dir / "huelva_selected_file_inventory_v0_1.csv"
    schemas_out = output_dir / "huelva_selected_schema_summary_v0_1.csv"
    columns_out = output_dir / "huelva_selected_columns_by_file_v0_1.csv"
    columns_summary_out = output_dir / "huelva_selected_column_summary_v0_1.csv"
    preview_out = output_dir / "huelva_selected_preview_rows_v0_1.csv"
    report_md = output_dir / "huelva_selected_inspection_report_v0_1.md"
    excel_out = output_dir / "huelva_selected_inspection_report_v0_1.xlsx"

    manifest_df.to_csv(manifest_out, index=False, encoding="utf-8-sig")
    files_df.to_csv(files_out, index=False, encoding="utf-8-sig")
    schemas_df.to_csv(schemas_out, index=False, encoding="utf-8-sig")
    columns_df.to_csv(columns_out, index=False, encoding="utf-8-sig")
    columns_summary_df.to_csv(columns_summary_out, index=False, encoding="utf-8-sig")
    preview_df.to_csv(preview_out, index=False, encoding="utf-8-sig")

    write_markdown_report(report_md, manifest_df, files_df, schemas_df, columns_summary_df)

    if write_excel:
        try:
            with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
                manifest_df.to_excel(writer, sheet_name="manifest", index=False)
                files_df.to_excel(writer, sheet_name="files", index=False)
                schemas_df.to_excel(writer, sheet_name="schemas", index=False)
                columns_summary_df.to_excel(writer, sheet_name="columns_summary", index=False)
                columns_df.to_excel(writer, sheet_name="columns_by_file", index=False)
                preview_df.to_excel(writer, sheet_name="preview_rows", index=False)
        except Exception as exc:
            print(f"[WARN] No se pudo escribir Excel: {type(exc).__name__}: {exc}")
            excel_out = None

    print("")
    print("========================================")
    print("Inspección completada")
    print("========================================")
    print(f"Manifest resuelto:      {manifest_out}")
    print(f"Inventario ficheros:    {files_out}")
    print(f"Resumen esquemas:       {schemas_out}")
    print(f"Resumen columnas:       {columns_summary_out}")
    print(f"Columnas por fichero:   {columns_out}")
    print(f"Preview filas:          {preview_out}")
    print(f"Reporte Markdown:       {report_md}")

    if excel_out:
        print(f"Reporte Excel:          {excel_out}")

    if not files_df.empty:
        print("")
        print("Resumen por tipo:")
        print(files_df["file_kind"].value_counts(dropna=False).to_string())

        print("")
        print("Resumen lectura:")
        print(files_df["read_ok"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
