from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber


VERSION = "v0.2"


SAMPLE_DATETIME_PATTERNS = [
    # FECHA DE TOMA: 3/06/2026 HORA 08:15
    r"FECHA\s+(?:DE\s+)?TOMA\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:\n|\r|\s)+HORA\s*[:\-]?\s*(\d{1,2}[:.]\d{2}(?::\d{2})?)",

    # FECHA TOMA 3/06/2026 HORA: 08:15
    r"FECHA\s+TOMA\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:\n|\r|\s)+HORA\s*[:\-]?\s*(\d{1,2}[:.]\d{2}(?::\d{2})?)",
]


DATE_PATTERNS = [
    # FECHA DE TOMA: 3/06/2026
    r"FECHA\s+(?:DE\s+)?TOMA\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",

    # Fecha muestra: 3/06/2026
    r"(?:fecha\s*(?:de)?\s*(?:toma|muestreo|muestra)|fecha\s*muestra)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
]


TIME_PATTERNS = [
    # HORA 08:15 / HORA: 08:15
    r"\bHORA\s*[:\-]?\s*(\d{1,2}[:.]\d{2}(?::\d{2})?)",

    # Hora de toma: 08:15
    r"(?:hora\s*(?:de)?\s*(?:toma|muestreo|muestra)|hora\s*muestra)\s*[:\-]?\s*(\d{1,2}[:.]\d{2}(?::\d{2})?)",
]


THM_TOTAL_PATTERNS = [
    # Formato real:
    # Suma de Trihalometanos PE-COR-025 100 30.6 µg/L
    r"Suma\s+de\s+Trihalometanos\s+.*?\s(\d+(?:[,.]\d+)?)\s*(?:µg/L|ug/L|microg/L)",

    # Variante posible:
    # Trihalometanos totales ... 30.6 µg/L
    r"Trihalometanos\s+totales\s+.*?\s(\d+(?:[,.]\d+)?)\s*(?:µg/L|ug/L|microg/L)",

    # Variante defensiva:
    # THM totales ... 30.6 µg/L
    r"\bTHM(?:s)?\s+totales\s+.*?\s(\d+(?:[,.]\d+)?)\s*(?:µg/L|ug/L|microg/L)",
]


SAMPLE_POINT_PATTERNS = [
    r"#\s*DENOMINACI[ÓO]N\s+MUESTRA\s*[:\-]?\s*(.+)",
    r"DENOMINACI[ÓO]N\s+MUESTRA\s*[:\-]?\s*(.+)",
]


REPORT_NUMBER_PATTERNS = [
    r"INFORME\s+N[ºO]\s*[:\-]?\s*(\d+)",
]


ANALYSIS_NUMBER_PATTERNS = [
    r"AN[ÁA]LISIS\s+N[ºO]\s*[:\-]?\s*(\d+)",
]


def normalize_spaces(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_path: Path) -> str:
    pages = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    return normalize_spaces("\n".join(pages))


def parse_decimal(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None

    value = value.strip()

    if "," in value:
        value = value.replace(".", "").replace(",", ".")

    try:
        return float(value)
    except ValueError:
        return None


def find_first(patterns: List[str], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if match:
            return match.group(1).strip()

    return None


def find_sample_datetime(text: str) -> Tuple[Optional[str], Optional[str]]:
    for pattern in SAMPLE_DATETIME_PATTERNS:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if match:
            return match.group(1).strip(), match.group(2).strip()

    return None, None


def parse_date_from_filename(pdf_path: Path) -> Optional[str]:
    # Ejemplo:
    # JUNIO THM 03-06-2026_Salida Depósito Nº2_4703931.pdf
    match = re.search(r"(\d{1,2}-\d{1,2}-\d{4})", pdf_path.name)

    if match:
        return match.group(1)

    return None


def clean_sample_point(value: Optional[str]) -> str:
    if not value:
        return "Salida Depósito Nº2"

    value = value.strip()
    value = re.sub(r"\s+", " ", value)

    # Evita arrastrar texto si el extractor de PDF junta líneas.
    for stop_word in [
        "DESCRIPCIÓN MUESTRA",
        "DESCRIPCION MUESTRA",
        "FECHA RECEPCIÓN",
        "FECHA RECEPCION",
    ]:
        if stop_word in value.upper():
            value = value[: value.upper().find(stop_word)].strip()

    return value or "Salida Depósito Nº2"


def parse_timestamp(
    date_raw: Optional[str],
    time_raw: Optional[str],
) -> Tuple[Optional[pd.Timestamp], Optional[str]]:
    if not date_raw or not time_raw:
        return None, None

    time_raw = time_raw.replace(".", ":")

    dt_text = f"{date_raw} {time_raw}"
    timestamp = pd.to_datetime(dt_text, dayfirst=True, errors="coerce")

    if pd.isna(timestamp):
        return None, "invalid_sample_timestamp"

    return timestamp, None


def classify_row(
    date_raw: Optional[str],
    time_raw: Optional[str],
    sample_timestamp: Optional[pd.Timestamp],
    thm_value: Optional[float],
    timestamp_error: Optional[str],
) -> Tuple[str, str]:
    flags = []

    if thm_value is None:
        flags.append("missing_thm")

    if not date_raw:
        flags.append("missing_sample_date")

    if date_raw and not time_raw:
        flags.append("missing_sample_time")

    if timestamp_error:
        flags.append(timestamp_error)

    if thm_value is not None and date_raw and time_raw and sample_timestamp is not None:
        return "exact_datetime_ok", ""

    if thm_value is not None and date_raw and not time_raw:
        return "date_only_no_time", ";".join(flags)

    return "review", ";".join(flags)


def parse_pdf(pdf_path: Path, pdf_root: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "source_file": str(pdf_path),
        "relative_file": str(pdf_path.relative_to(pdf_root)),
        "filename": pdf_path.name,
        "year_folder": pdf_path.parent.name,
        "version": VERSION,
    }

    try:
        text = extract_text_from_pdf(pdf_path)

    except Exception as exc:
        row.update({
            "report_number": None,
            "analysis_number": None,
            "sample_point": "Salida Depósito Nº2",
            "sample_date_raw": None,
            "sample_date_source": None,
            "sample_time_raw": None,
            "sample_timestamp": None,
            "sample_datetime_quality": "pdf_read_error",
            "thm": None,
            "thm_unit": "µg/L",
            "status": "pdf_read_error",
            "quality_flags": f"pdf_read_error:{type(exc).__name__}",
            "text_preview": "",
        })
        return row

    report_number = find_first(REPORT_NUMBER_PATTERNS, text)
    analysis_number = find_first(ANALYSIS_NUMBER_PATTERNS, text)

    sample_point_raw = find_first(SAMPLE_POINT_PATTERNS, text)
    sample_point = clean_sample_point(sample_point_raw)

    date_raw, time_raw = find_sample_datetime(text)
    sample_date_source = "fecha_de_toma"

    if not date_raw:
        date_raw = find_first(DATE_PATTERNS, text)
        sample_date_source = "fecha_de_toma"

    if not date_raw:
        date_raw = parse_date_from_filename(pdf_path)
        sample_date_source = "filename" if date_raw else None

    if not time_raw:
        time_raw = find_first(TIME_PATTERNS, text)

    thm_raw = find_first(THM_TOTAL_PATTERNS, text)
    thm_value = parse_decimal(thm_raw)

    sample_timestamp, timestamp_error = parse_timestamp(date_raw, time_raw)

    status, quality_flags = classify_row(
        date_raw=date_raw,
        time_raw=time_raw,
        sample_timestamp=sample_timestamp,
        thm_value=thm_value,
        timestamp_error=timestamp_error,
    )

    if status == "exact_datetime_ok":
        sample_datetime_quality = "exact_datetime"
    elif status == "date_only_no_time":
        sample_datetime_quality = "date_only"
    else:
        sample_datetime_quality = "review"

    row.update({
        "report_number": report_number,
        "analysis_number": analysis_number,
        "sample_point": sample_point,
        "sample_date_raw": date_raw,
        "sample_date_source": sample_date_source,
        "sample_time_raw": time_raw,
        "sample_timestamp": sample_timestamp,
        "sample_datetime_quality": sample_datetime_quality,
        "thm": thm_value,
        "thm_unit": "µg/L",
        "status": status,
        "quality_flags": quality_flags,
        "text_preview": text[:2000],
    })

    return row


def build_required_download_windows(
    df_exact: pd.DataFrame,
    history_hours: int,
) -> pd.DataFrame:
    if df_exact.empty:
        return pd.DataFrame(columns=[
            "sample_timestamp",
            "sample_point",
            "thm",
            "thm_unit",
            "relative_file",
            "download_start",
            "download_end",
            "history_hours",
            "reason",
        ])

    out = df_exact[
        [
            "sample_timestamp",
            "sample_point",
            "thm",
            "thm_unit",
            "relative_file",
        ]
    ].copy()

    out["download_start"] = out["sample_timestamp"] - pd.to_timedelta(history_hours, unit="h")
    out["download_end"] = out["sample_timestamp"]
    out["history_hours"] = history_hours
    out["reason"] = f"THM sample + previous {history_hours}h for lag/rolling features"

    return out


def build_assumed_time_windows(
    df_date_only: pd.DataFrame,
    history_hours: int,
    assumed_time: Optional[str],
) -> pd.DataFrame:
    if df_date_only.empty or not assumed_time:
        return pd.DataFrame()

    out = df_date_only[
        [
            "sample_date_raw",
            "sample_point",
            "thm",
            "thm_unit",
            "relative_file",
        ]
    ].copy()

    assumed_time = assumed_time.replace(".", ":")

    out["assumed_time"] = assumed_time
    out["sample_timestamp_assumed"] = pd.to_datetime(
        out["sample_date_raw"].astype(str) + " " + assumed_time,
        dayfirst=True,
        errors="coerce",
    )

    out["download_start"] = out["sample_timestamp_assumed"] - pd.to_timedelta(history_hours, unit="h")
    out["download_end"] = out["sample_timestamp_assumed"]
    out["history_hours"] = history_hours
    out["reason"] = (
        f"DATE-ONLY THM sample with assumed time {assumed_time}; "
        f"not for main training unless validated"
    )

    return out


def make_summary(
    df: pd.DataFrame,
    exact: pd.DataFrame,
    date_only: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "pdf_files", "value": len(df)},
        {"metric": "exact_datetime_ok", "value": len(exact)},
        {"metric": "date_only_no_time", "value": len(date_only)},
        {"metric": "review_or_errors", "value": len(review)},
        {
            "metric": "missing_sample_time",
            "value": int(df["quality_flags"].fillna("").str.contains("missing_sample_time").sum()),
        },
        {
            "metric": "missing_thm",
            "value": int(df["quality_flags"].fillna("").str.contains("missing_thm").sum()),
        },
        {
            "metric": "exact_min_timestamp",
            "value": exact["sample_timestamp"].min() if not exact.empty else None,
        },
        {
            "metric": "exact_max_timestamp",
            "value": exact["sample_timestamp"].max() if not exact.empty else None,
        },
        {
            "metric": "exact_thm_mean",
            "value": exact["thm"].mean() if not exact.empty else None,
        },
        {
            "metric": "exact_thm_min",
            "value": exact["thm"].min() if not exact.empty else None,
        },
        {
            "metric": "exact_thm_max",
            "value": exact["thm"].max() if not exact.empty else None,
        },
        {
            "metric": "date_only_thm_mean",
            "value": date_only["thm"].mean() if not date_only.empty else None,
        },
        {
            "metric": "date_only_thm_min",
            "value": date_only["thm"].min() if not date_only.empty else None,
        },
        {
            "metric": "date_only_thm_max",
            "value": date_only["thm"].max() if not date_only.empty else None,
        },
    ]

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae fecha/hora de muestra y THM desde PDFs de Aguas de Huelva."
    )

    parser.add_argument(
        "--pdf-root",
        required=True,
        help=r"Carpeta raíz con PDFs, por ejemplo .\THM_Depositos",
    )

    parser.add_argument(
        "--output-dir",
        default=r".\huelva_outputs_v0_2_thm_pdf",
        help="Carpeta de salida",
    )

    parser.add_argument(
        "--history-hours",
        type=int,
        default=24,
        help="Horas previas necesarias para descarga multiparamétrica",
    )

    parser.add_argument(
        "--assume-time-for-date-only",
        default=None,
        help=(
            "Hora opcional para generar ventanas diagnósticas de PDFs con solo fecha, "
            "por ejemplo 08:00. No se usa para el dataset principal."
        ),
    )

    args = parser.parse_args()

    pdf_root = Path(args.pdf_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_root.rglob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"No se han encontrado PDFs en {pdf_root}")

    rows = [parse_pdf(pdf_path, pdf_root) for pdf_path in pdfs]
    df = pd.DataFrame(rows)

    df["sample_timestamp"] = pd.to_datetime(df["sample_timestamp"], errors="coerce")
    df["thm"] = pd.to_numeric(df["thm"], errors="coerce")

    exact = df[
        (df["status"] == "exact_datetime_ok")
        & df["sample_timestamp"].notna()
        & df["thm"].notna()
    ].copy()

    date_only = df[
        (df["status"] == "date_only_no_time")
        & df["sample_date_raw"].notna()
        & df["thm"].notna()
    ].copy()

    review = df[
        ~df["status"].isin(["exact_datetime_ok", "date_only_no_time"])
    ].copy()

    exact_windows = build_required_download_windows(
        exact,
        history_hours=args.history_hours,
    )

    assumed_windows = build_assumed_time_windows(
        date_only,
        history_hours=args.history_hours,
        assumed_time=args.assume_time_for_date_only,
    )

    summary = make_summary(
        df=df,
        exact=exact,
        date_only=date_only,
        review=review,
    )

    out_all = output_dir / "huelva_thm_pdf_extraction_all_v0_2.csv"
    out_exact = output_dir / "huelva_thm_exact_datetime_samples_v0_2.csv"
    out_date_only = output_dir / "huelva_thm_date_only_candidates_v0_2.csv"
    out_review = output_dir / "huelva_thm_review_pdfs_v0_2.csv"
    out_exact_windows = output_dir / "huelva_required_multiparametric_windows_exact_v0_2.csv"
    out_assumed_windows = output_dir / "huelva_required_multiparametric_windows_date_only_assumed_v0_2.csv"
    out_xlsx = output_dir / "huelva_thm_pdf_extraction_report_v0_2.xlsx"

    df.to_csv(out_all, index=False, encoding="utf-8-sig")
    exact.to_csv(out_exact, index=False, encoding="utf-8-sig")
    date_only.to_csv(out_date_only, index=False, encoding="utf-8-sig")
    review.to_csv(out_review, index=False, encoding="utf-8-sig")
    exact_windows.to_csv(out_exact_windows, index=False, encoding="utf-8-sig")

    if not assumed_windows.empty:
        assumed_windows.to_csv(out_assumed_windows, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Resumen", index=False)
        exact.to_excel(writer, sheet_name="Exactas_entrenamiento", index=False)
        date_only.to_excel(writer, sheet_name="Solo_fecha_candidatas", index=False)
        review.to_excel(writer, sheet_name="Revisar_descartar", index=False)
        exact_windows.to_excel(writer, sheet_name="Ventanas_exactas", index=False)

        if not assumed_windows.empty:
            assumed_windows.to_excel(writer, sheet_name="Ventanas_hora_asumida", index=False)

        df.to_excel(writer, sheet_name="Todos_los_PDFs", index=False)

    print("========================================")
    print("Extracción THM PDFs Huelva completada")
    print("========================================")
    print(f"Versión:                         {VERSION}")
    print(f"PDFs encontrados:                {len(df)}")
    print(f"Muestras exactas para entrenar:  {len(exact)}")
    print(f"Candidatas solo con fecha:       {len(date_only)}")
    print(f"Revisar/errores:                 {len(review)}")
    print("")
    print(f"Salida completa:                 {out_all}")
    print(f"Exactas entrenamiento:           {out_exact}")
    print(f"Solo fecha candidatas:           {out_date_only}")
    print(f"PDFs a revisar:                  {out_review}")
    print(f"Ventanas multiparamétrica exactas:{out_exact_windows}")

    if not assumed_windows.empty:
        print(f"Ventanas con hora asumida:       {out_assumed_windows}")

    print(f"Reporte Excel:                   {out_xlsx}")


if __name__ == "__main__":
    main()