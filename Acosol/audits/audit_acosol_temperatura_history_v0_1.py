from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


VERSION = "v0.1"

DEFAULT_HISTORICAL_DIR = Path("archive") / "datos_historicos"
DEFAULT_SFTP_TIMESERIES = (
    Path("acosol_sftp_dataset_v0_1")
    / "acosol_sftp_timeseries_v0_1.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path("acosol_sftp_dataset_v0_1")
    / "temperature_audit"
)

TEMPERATURE_TOKENS = {
    "temperatura",
    "temperature",
    "temp",
}

THM_TOKENS = {
    "thm",
    "trihalometanos",
    "trihalomethanes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Auditoría histórica de temperatura en Acosol para determinar "
            "si existe un periodo supervisado utilizable para comparar un "
            "soft-sensor con y sin temperatura."
        )
    )
    parser.add_argument(
        "--historical-dir",
        type=Path,
        default=DEFAULT_HISTORICAL_DIR,
    )
    parser.add_argument(
        "--sftp-timeseries",
        type=Path,
        default=DEFAULT_SFTP_TIMESERIES,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=3,
        help=(
            "Decimales usados para ignorar ruido numérico al contar "
            "valores únicos de temperatura."
        ),
    )
    parser.add_argument(
        "--dynamic-range-threshold",
        type=float,
        default=0.10,
        help=(
            "Rango mínimo diario de temperatura, en grados, usado solo "
            "como cribado descriptivo de posible señal dinámica."
        ),
    )
    parser.add_argument(
        "--dynamic-unique-threshold",
        type=int,
        default=4,
        help=(
            "Número mínimo de valores diarios únicos, tras redondeo, usado "
            "solo como cribado descriptivo de posible señal dinámica."
        ),
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_matching_columns(
    columns: list[Any],
    tokens: set[str],
) -> list[str]:
    matches: list[str] = []

    for column in columns:
        normalized = normalize_text(column)
        words = set(normalized.split())

        if normalized in tokens:
            matches.append(str(column))
            continue

        if words & tokens:
            matches.append(str(column))
            continue

        if any(
            token in normalized
            for token in tokens
            if len(token) >= 4
        ):
            matches.append(str(column))

    return matches


def safe_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    text = (
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(text, errors="coerce")


def summarize_temperature(
    series: pd.Series,
    round_decimals: int,
) -> dict[str, Any]:
    values = safe_numeric(series).dropna()

    if values.empty:
        return {
            "valid_values": 0,
            "unique_raw": 0,
            "unique_rounded": 0,
            "min": np.nan,
            "max": np.nan,
            "range": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
        }

    rounded = values.round(round_decimals)

    return {
        "valid_values": int(len(values)),
        "unique_raw": int(values.nunique(dropna=True)),
        "unique_rounded": int(rounded.nunique(dropna=True)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(ddof=1))
        if len(values) > 1
        else 0.0,
    }


def audit_historical_excels(
    historical_dir: Path,
    round_decimals: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not historical_dir.exists():
        return pd.DataFrame(
            [
                {
                    "file": str(historical_dir),
                    "sheet": "",
                    "status": "historical_directory_not_found",
                }
            ]
        )

    files = sorted(
        list(historical_dir.glob("*.xlsx"))
        + list(historical_dir.glob("*.xls"))
    )

    if not files:
        return pd.DataFrame(
            [
                {
                    "file": str(historical_dir),
                    "sheet": "",
                    "status": "no_excel_files_found",
                }
            ]
        )

    for path in files:
        try:
            excel = pd.ExcelFile(path)
        except Exception as exc:
            rows.append(
                {
                    "file": path.name,
                    "sheet": "",
                    "status": "excel_open_error",
                    "error": repr(exc),
                }
            )
            continue

        for sheet_name in excel.sheet_names:
            try:
                frame = pd.read_excel(
                    path,
                    sheet_name=sheet_name,
                    nrows=None,
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": path.name,
                        "sheet": sheet_name,
                        "status": "sheet_read_error",
                        "error": repr(exc),
                    }
                )
                continue

            temperature_columns = find_matching_columns(
                list(frame.columns),
                TEMPERATURE_TOKENS,
            )
            thm_columns = find_matching_columns(
                list(frame.columns),
                THM_TOKENS,
            )

            base_row: dict[str, Any] = {
                "file": path.name,
                "sheet": sheet_name,
                "status": "ok",
                "rows": int(len(frame)),
                "columns": " | ".join(
                    str(column)
                    for column in frame.columns
                ),
                "temperature_columns": " | ".join(
                    temperature_columns
                ),
                "thm_columns": " | ".join(thm_columns),
                "has_temperature_column": bool(
                    temperature_columns
                ),
                "has_thm_column": bool(thm_columns),
            }

            if temperature_columns:
                temp_column = temperature_columns[0]
                base_row["temperature_column_audited"] = (
                    temp_column
                )
                base_row.update(
                    {
                        f"temperature_{key}": value
                        for key, value in summarize_temperature(
                            frame[temp_column],
                            round_decimals,
                        ).items()
                    }
                )

            rows.append(base_row)

    return pd.DataFrame(rows)


def parse_timestamp_column(
    frame: pd.DataFrame,
) -> pd.Series:
    candidates = [
        "timestamp",
        "datetime",
        "fecha_hora",
        "fecha hora",
    ]

    normalized_map = {
        normalize_text(column): column
        for column in frame.columns
    }

    for candidate in candidates:
        normalized_candidate = normalize_text(candidate)
        if normalized_candidate in normalized_map:
            source = normalized_map[normalized_candidate]
            return pd.to_datetime(
                frame[source],
                errors="coerce",
                dayfirst=True,
            )

    return pd.Series(
        pd.NaT,
        index=frame.index,
        dtype="datetime64[ns]",
    )


def find_single_column(
    frame: pd.DataFrame,
    tokens: set[str],
) -> str | None:
    matches = find_matching_columns(
        list(frame.columns),
        tokens,
    )
    return matches[0] if matches else None


def daily_sftp_temperature_summary(
    frame: pd.DataFrame,
    timestamp: pd.Series,
    temp_column: str,
    thm_column: str | None,
    round_decimals: int,
    range_threshold: float,
    unique_threshold: int,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "timestamp": timestamp,
            "temperatura": safe_numeric(frame[temp_column]),
        }
    )

    if thm_column is not None:
        work["thm"] = safe_numeric(frame[thm_column])
    else:
        work["thm"] = np.nan

    work = work.dropna(
        subset=["timestamp", "temperatura"]
    ).copy()

    if work.empty:
        return pd.DataFrame()

    work["date"] = work["timestamp"].dt.date
    work["temperature_rounded"] = (
        work["temperatura"].round(round_decimals)
    )

    rows: list[dict[str, Any]] = []

    for date, group in work.groupby("date", sort=True):
        temp = group["temperatura"]
        rounded = group["temperature_rounded"]
        supervised = group["thm"].notna()

        value_range = float(temp.max() - temp.min())
        unique_rounded = int(rounded.nunique())

        candidate_dynamic = (
            value_range >= range_threshold
            and unique_rounded >= unique_threshold
        )

        rows.append(
            {
                "date": str(date),
                "rows_with_temperature": int(len(group)),
                "rows_with_thm": int(supervised.sum()),
                "supervised_pct": (
                    100.0 * float(supervised.mean())
                ),
                "temperature_min": float(temp.min()),
                "temperature_median": float(temp.median()),
                "temperature_mean": float(temp.mean()),
                "temperature_max": float(temp.max()),
                "temperature_range": value_range,
                "temperature_std": (
                    float(temp.std(ddof=1))
                    if len(temp) > 1
                    else 0.0
                ),
                "temperature_unique_raw": int(
                    temp.nunique()
                ),
                "temperature_unique_rounded": (
                    unique_rounded
                ),
                "candidate_dynamic_day": (
                    candidate_dynamic
                ),
                "candidate_dynamic_supervised_day": (
                    candidate_dynamic
                    and int(supervised.sum()) > 0
                ),
            }
        )

    return pd.DataFrame(rows)


def audit_sftp(
    path: Path,
    round_decimals: int,
    range_threshold: float,
    unique_threshold: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    if not path.exists():
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "status": "sftp_timeseries_not_found",
                "path": str(path),
            },
        )

    frame = pd.read_csv(path)

    temp_column = find_single_column(
        frame,
        TEMPERATURE_TOKENS,
    )
    thm_column = find_single_column(
        frame,
        THM_TOKENS,
    )
    timestamp = parse_timestamp_column(frame)

    metadata: dict[str, Any] = {
        "status": "ok",
        "path": str(path),
        "rows": int(len(frame)),
        "temperature_column": temp_column,
        "thm_column": thm_column,
        "timestamp_valid": int(timestamp.notna().sum()),
    }

    if temp_column is None:
        metadata["status"] = "temperature_column_not_found"
        return frame, pd.DataFrame(), metadata

    all_summary = summarize_temperature(
        frame[temp_column],
        round_decimals,
    )

    supervised_mask = (
        safe_numeric(frame[thm_column]).notna()
        if thm_column is not None
        else pd.Series(
            False,
            index=frame.index,
        )
    )

    supervised_summary = summarize_temperature(
        frame.loc[
            supervised_mask,
            temp_column,
        ],
        round_decimals,
    )

    metadata["all_temperature"] = all_summary
    metadata["supervised_temperature"] = (
        supervised_summary
    )
    metadata["supervised_rows"] = int(
        supervised_mask.sum()
    )

    if timestamp.notna().any():
        metadata["timestamp_start"] = str(
            timestamp.min()
        )
        metadata["timestamp_end"] = str(
            timestamp.max()
        )

        if supervised_mask.any():
            supervised_timestamps = timestamp[
                supervised_mask
            ].dropna()

            if not supervised_timestamps.empty:
                metadata["supervised_start"] = str(
                    supervised_timestamps.min()
                )
                metadata["supervised_end"] = str(
                    supervised_timestamps.max()
                )

    daily = daily_sftp_temperature_summary(
        frame=frame,
        timestamp=timestamp,
        temp_column=temp_column,
        thm_column=thm_column,
        round_decimals=round_decimals,
        range_threshold=range_threshold,
        unique_threshold=unique_threshold,
    )

    return frame, daily, metadata


def determine_configuration_a_feasibility(
    historical: pd.DataFrame,
    daily: pd.DataFrame,
    sftp_metadata: dict[str, Any],
) -> dict[str, Any]:
    historical_temperature_sheets = 0
    historical_temperature_and_thm_sheets = 0

    if not historical.empty:
        if "has_temperature_column" in historical.columns:
            historical_temperature_sheets = int(
                historical[
                    "has_temperature_column"
                ].fillna(False).astype(bool).sum()
            )

        if (
            "has_temperature_column" in historical.columns
            and "has_thm_column" in historical.columns
        ):
            both = (
                historical[
                    "has_temperature_column"
                ].fillna(False).astype(bool)
                & historical[
                    "has_thm_column"
                ].fillna(False).astype(bool)
            )
            historical_temperature_and_thm_sheets = int(
                both.sum()
            )

    candidate_supervised_days = 0
    if (
        not daily.empty
        and "candidate_dynamic_supervised_day"
        in daily.columns
    ):
        candidate_supervised_days = int(
            daily[
                "candidate_dynamic_supervised_day"
            ].fillna(False).astype(bool).sum()
        )

    supervised_summary = sftp_metadata.get(
        "supervised_temperature",
        {},
    )

    supervised_dynamic_global = bool(
        supervised_summary
        and not math.isnan(
            float(
                supervised_summary.get(
                    "range",
                    np.nan,
                )
            )
        )
        and float(
            supervised_summary.get(
                "range",
                0.0,
            )
        ) >= 0.10
        and int(
            supervised_summary.get(
                "unique_rounded",
                0,
            )
        ) >= 4
    )

    possible = (
        historical_temperature_and_thm_sheets > 0
        or candidate_supervised_days > 0
        or supervised_dynamic_global
    )

    if possible:
        conclusion = (
            "Hay indicios de solapamiento THM-temperatura potencialmente "
            "dinámica. Debe revisarse el periodo candidato antes de validar "
            "la configuración con temperatura."
        )
    else:
        conclusion = (
            "Con los datos auditados no se identifica un periodo "
            "supervisado con temperatura dinámica suficiente para plantear "
            "la configuración con temperatura. La configuración sin "
            "bromuro ni temperatura sí puede validarse."
        )

    return {
        "configuration_a_with_temperature_candidate": possible,
        "historical_sheets_with_temperature": (
            historical_temperature_sheets
        ),
        "historical_sheets_with_temperature_and_thm": (
            historical_temperature_and_thm_sheets
        ),
        "sftp_candidate_dynamic_supervised_days": (
            candidate_supervised_days
        ),
        "sftp_supervised_dynamic_global_screen": (
            supervised_dynamic_global
        ),
        "conclusion": conclusion,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Auditoría histórica de temperatura - Acosol")
    print("=" * 52)

    historical = audit_historical_excels(
        historical_dir=args.historical_dir,
        round_decimals=args.round_decimals,
    )

    _, daily, sftp_metadata = audit_sftp(
        path=args.sftp_timeseries,
        round_decimals=args.round_decimals,
        range_threshold=args.dynamic_range_threshold,
        unique_threshold=args.dynamic_unique_threshold,
    )

    feasibility = determine_configuration_a_feasibility(
        historical=historical,
        daily=daily,
        sftp_metadata=sftp_metadata,
    )

    historical_path = (
        args.output_dir
        / "historical_excel_temperature_inventory.csv"
    )
    daily_path = (
        args.output_dir
        / "sftp_temperature_daily_summary.csv"
    )
    metadata_path = (
        args.output_dir
        / "temperature_audit_metadata.json"
    )

    historical.to_csv(
        historical_path,
        index=False,
        encoding="utf-8-sig",
    )

    daily.to_csv(
        daily_path,
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "version": VERSION,
        "historical_directory": str(
            args.historical_dir
        ),
        "sftp_timeseries": str(
            args.sftp_timeseries
        ),
        "screening_parameters": {
            "round_decimals": args.round_decimals,
            "dynamic_range_threshold": (
                args.dynamic_range_threshold
            ),
            "dynamic_unique_threshold": (
                args.dynamic_unique_threshold
            ),
        },
        "sftp": sftp_metadata,
        "configuration_a_feasibility": feasibility,
        "important_note": (
            "Los criterios de señal dinámica son únicamente un cribado "
            "descriptivo. No demuestran por sí solos que la temperatura "
            "sea físicamente correcta; cualquier periodo candidato debe "
            "ser revisado antes de incorporarlo al modelo."
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n1) Excel históricos")
    print("--------------------")

    if historical.empty:
        print("No se han podido inventariar hojas.")
    else:
        ok = historical[
            historical.get(
                "status",
                pd.Series(
                    "",
                    index=historical.index,
                ),
            )
            == "ok"
        ]

        print(
            f"Hojas inventariadas: "
            f"{len(ok)}"
        )

        if "has_temperature_column" in ok.columns:
            print(
                "Hojas con columna de temperatura: "
                f"{int(ok['has_temperature_column'].fillna(False).sum())}"
            )

        if (
            "has_temperature_column" in ok.columns
            and "has_thm_column" in ok.columns
        ):
            both = (
                ok["has_temperature_column"]
                .fillna(False)
                .astype(bool)
                & ok["has_thm_column"]
                .fillna(False)
                .astype(bool)
            )
            print(
                "Hojas con temperatura y THM:      "
                f"{int(both.sum())}"
            )

    print("\n2) Serie continua SFTP")
    print("----------------------")

    if sftp_metadata.get("status") != "ok":
        print(
            "Estado: "
            f"{sftp_metadata.get('status')}"
        )
    else:
        print(
            f"Periodo total:      "
            f"{sftp_metadata.get('timestamp_start')} "
            f"-> {sftp_metadata.get('timestamp_end')}"
        )
        print(
            f"Filas totales:      "
            f"{sftp_metadata.get('rows')}"
        )
        print(
            f"Filas con THM:      "
            f"{sftp_metadata.get('supervised_rows')}"
        )

        all_temp = sftp_metadata.get(
            "all_temperature",
            {},
        )
        supervised_temp = sftp_metadata.get(
            "supervised_temperature",
            {},
        )

        print("\nTemperatura - periodo total:")
        print(
            f"  Válidos:          "
            f"{all_temp.get('valid_values')}"
        )
        print(
            f"  Únicos redondeados: "
            f"{all_temp.get('unique_rounded')}"
        )
        print(
            f"  Min/Max:          "
            f"{all_temp.get('min')} / "
            f"{all_temp.get('max')}"
        )
        print(
            f"  Rango:            "
            f"{all_temp.get('range')}"
        )

        print("\nTemperatura - periodo supervisado con THM:")
        print(
            f"  Periodo:          "
            f"{sftp_metadata.get('supervised_start')} "
            f"-> {sftp_metadata.get('supervised_end')}"
        )
        print(
            f"  Válidos:          "
            f"{supervised_temp.get('valid_values')}"
        )
        print(
            f"  Únicos redondeados: "
            f"{supervised_temp.get('unique_rounded')}"
        )
        print(
            f"  Min/Max:          "
            f"{supervised_temp.get('min')} / "
            f"{supervised_temp.get('max')}"
        )
        print(
            f"  Rango:            "
            f"{supervised_temp.get('range')}"
        )

        if not daily.empty:
            candidate_days = daily[
                daily[
                    "candidate_dynamic_supervised_day"
                ]
                .fillna(False)
                .astype(bool)
            ]
            print(
                "\nDías supervisados que pasan el cribado "
                "de señal dinámica: "
                f"{len(candidate_days)}"
            )

            if not candidate_days.empty:
                print(
                    candidate_days[
                        [
                            "date",
                            "rows_with_thm",
                            "temperature_min",
                            "temperature_max",
                            "temperature_range",
                            "temperature_unique_rounded",
                        ]
                    ].to_string(index=False)
                )

    print("\n3) Conclusión para la petición de planta")
    print("----------------------------------------")
    print(
        "Configuración A - sin bromuro, CON temperatura: "
        + (
            "CANDIDATA A REVISIÓN"
            if feasibility[
                "configuration_a_with_temperature_candidate"
            ]
            else "NO VALIDABLE CON LOS DATOS AUDITADOS"
        )
    )
    print(
        "Configuración B - sin bromuro NI temperatura: "
        "VALIDABLE"
    )
    print(
        "\n"
        + feasibility["conclusion"]
    )

    print("\nSalidas:")
    print(f"- {historical_path}")
    print(f"- {daily_path}")
    print(f"- {metadata_path}")
    print(
        "\n[INFO] Esta auditoría no modifica ni elimina datos."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
