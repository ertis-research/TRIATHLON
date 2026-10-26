from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_DIR = Path("acosol_sftp_dataset_v0_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Comprueba calidad y continuidad únicamente en el periodo supervisado "
            "del dataset SFTP de Acosol."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Carpeta generada por build_acosol_training_dataset_v0_1.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR / "supervised_checks",
        help="Carpeta de salida para informes de comprobación.",
    )
    parser.add_argument(
        "--gap-threshold-minutes",
        type=float,
        default=7.5,
        help="Umbral para considerar un hueco temporal relevante.",
    )
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def nonempty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def main() -> int:
    args = parse_args()

    if args.gap_threshold_minutes <= 0:
        raise SystemExit("--gap-threshold-minutes debe ser > 0.")

    supervised_path = args.dataset_dir / "acosol_sftp_supervised_v0_1.csv"
    audit_path = args.dataset_dir / "acosol_sftp_snapshot_audit_v0_1.csv"

    if not supervised_path.exists():
        raise FileNotFoundError(f"No existe: {supervised_path}")
    if not audit_path.exists():
        raise FileNotFoundError(f"No existe: {audit_path}")

    supervised = pd.read_csv(supervised_path)
    audit = pd.read_csv(audit_path)

    required_supervised = {
        "timestamp",
        "thm",
        "source_file",
        "quality_flags",
    }
    missing_supervised = sorted(required_supervised - set(supervised.columns))
    if missing_supervised:
        raise ValueError(
            "Faltan columnas en el dataset supervisado: "
            + ", ".join(missing_supervised)
        )

    required_audit = {
        "source_file",
        "filename_timestamp",
        "parse_ok",
        "all_rows_same_timestamp",
        "filename_timestamp_matches_internal",
        "missing_tags",
        "unknown_tags",
        "duplicate_tags",
        "invalid_value_tags",
        "has_thm_tag",
        "thm_value_valid",
        "complete_expected_tags",
        "quality_flags",
    }
    missing_audit = sorted(required_audit - set(audit.columns))
    if missing_audit:
        raise ValueError(
            "Faltan columnas en el informe de auditoría: "
            + ", ".join(missing_audit)
        )

    supervised["timestamp"] = pd.to_datetime(
        supervised["timestamp"],
        errors="coerce",
    )
    supervised["thm"] = pd.to_numeric(
        supervised["thm"],
        errors="coerce",
    )
    supervised = supervised.dropna(subset=["timestamp"]).copy()
    supervised = supervised.sort_values("timestamp").reset_index(drop=True)

    first_supervised = supervised["timestamp"].min()
    last_supervised = supervised["timestamp"].max()

    audit["filename_timestamp"] = pd.to_datetime(
        audit["filename_timestamp"],
        errors="coerce",
    )

    bool_cols = [
        "parse_ok",
        "all_rows_same_timestamp",
        "filename_timestamp_matches_internal",
        "has_thm_tag",
        "thm_value_valid",
        "complete_expected_tags",
    ]
    for col in bool_cols:
        audit[col] = as_bool(audit[col])

    # Se usa filename_timestamp para incluir también posibles CSV no parseables,
    # ya que esos pueden no disponer de timestamp interno.
    supervised_audit = audit[
        audit["filename_timestamp"].between(
            first_supervised,
            last_supervised,
            inclusive="both",
        )
    ].copy()

    # Continuidad temporal del dataset supervisado real.
    supervised["previous_timestamp"] = supervised["timestamp"].shift(1)
    supervised["gap_minutes"] = (
        supervised["timestamp"] - supervised["previous_timestamp"]
    ).dt.total_seconds() / 60.0

    gap_mask = supervised["gap_minutes"] > args.gap_threshold_minutes
    gaps = supervised[gap_mask].copy()

    duplicate_mask = supervised["timestamp"].duplicated(keep=False)
    duplicates = supervised[duplicate_mask].copy()

    non_positive_mask = (
        supervised["gap_minutes"].notna()
        & (supervised["gap_minutes"] <= 0)
    )
    non_positive = supervised[non_positive_mask].copy()

    # Incidencias de estructura/contenido dentro del periodo supervisado.
    parse_errors = supervised_audit[~supervised_audit["parse_ok"]].copy()

    timestamp_inconsistent = supervised_audit[
        supervised_audit["parse_ok"]
        & ~supervised_audit["all_rows_same_timestamp"]
    ].copy()

    timestamp_mismatches = supervised_audit[
        supervised_audit["parse_ok"]
        & supervised_audit["all_rows_same_timestamp"]
        & ~supervised_audit["filename_timestamp_matches_internal"]
    ].copy()

    missing_tags = supervised_audit[
        nonempty(supervised_audit["missing_tags"])
    ].copy()

    unknown_tags = supervised_audit[
        nonempty(supervised_audit["unknown_tags"])
    ].copy()

    duplicate_tags = supervised_audit[
        nonempty(supervised_audit["duplicate_tags"])
    ].copy()

    invalid_values = supervised_audit[
        nonempty(supervised_audit["invalid_value_tags"])
    ].copy()

    incomplete_expected_tags = supervised_audit[
        ~supervised_audit["complete_expected_tags"]
    ].copy()

    missing_thm_tag = supervised_audit[
        ~supervised_audit["has_thm_tag"]
    ].copy()

    invalid_thm = supervised_audit[
        supervised_audit["has_thm_tag"]
        & ~supervised_audit["thm_value_valid"]
    ].copy()

    supervised_quality_flagged = supervised[
        supervised["quality_flags"].fillna("").astype(str).str.strip().ne("ok")
    ].copy()

    positive_gaps = supervised.loc[
        supervised["gap_minutes"] > 0,
        "gap_minutes",
    ].dropna()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gaps_path = args.output_dir / "supervised_temporal_gaps.csv"
    duplicates_path = args.output_dir / "supervised_duplicate_timestamps.csv"
    audit_issues_path = args.output_dir / "supervised_snapshot_issues.csv"
    quality_flags_path = args.output_dir / "supervised_quality_flagged_rows.csv"
    summary_path = args.output_dir / "supervised_quality_summary.csv"

    gaps.to_csv(gaps_path, index=False, encoding="utf-8-sig")
    duplicates.to_csv(duplicates_path, index=False, encoding="utf-8-sig")
    supervised_quality_flagged.to_csv(
        quality_flags_path,
        index=False,
        encoding="utf-8-sig",
    )

    issue_mask = (
        ~supervised_audit["parse_ok"]
        | ~supervised_audit["all_rows_same_timestamp"]
        | (
            supervised_audit["all_rows_same_timestamp"]
            & ~supervised_audit["filename_timestamp_matches_internal"]
        )
        | nonempty(supervised_audit["missing_tags"])
        | nonempty(supervised_audit["unknown_tags"])
        | nonempty(supervised_audit["duplicate_tags"])
        | nonempty(supervised_audit["invalid_value_tags"])
        | ~supervised_audit["complete_expected_tags"]
        | ~supervised_audit["has_thm_tag"]
        | (
            supervised_audit["has_thm_tag"]
            & ~supervised_audit["thm_value_valid"]
        )
    )
    audit_issues = supervised_audit[issue_mask].copy()
    audit_issues.to_csv(
        audit_issues_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = [
        {"metric": "supervised_rows", "value": len(supervised)},
        {"metric": "first_supervised_timestamp", "value": first_supervised},
        {"metric": "last_supervised_timestamp", "value": last_supervised},
        {"metric": "audit_files_in_supervised_period", "value": len(supervised_audit)},
        {"metric": "parse_errors", "value": len(parse_errors)},
        {"metric": "inconsistent_internal_timestamps", "value": len(timestamp_inconsistent)},
        {"metric": "filename_internal_timestamp_mismatches", "value": len(timestamp_mismatches)},
        {"metric": "files_with_missing_tags", "value": len(missing_tags)},
        {"metric": "files_with_unknown_tags", "value": len(unknown_tags)},
        {"metric": "files_with_duplicate_tags", "value": len(duplicate_tags)},
        {"metric": "files_with_invalid_values", "value": len(invalid_values)},
        {"metric": "files_without_complete_9_tags", "value": len(incomplete_expected_tags)},
        {"metric": "files_without_thm_tag", "value": len(missing_thm_tag)},
        {"metric": "files_with_invalid_thm", "value": len(invalid_thm)},
        {"metric": "supervised_rows_with_quality_flags", "value": len(supervised_quality_flagged)},
        {"metric": "duplicate_timestamps", "value": len(duplicates)},
        {"metric": "non_positive_intervals", "value": len(non_positive)},
        {"metric": "temporal_gaps_above_threshold", "value": len(gaps)},
        {
            "metric": "gap_threshold_minutes",
            "value": args.gap_threshold_minutes,
        },
        {
            "metric": "interval_median_minutes",
            "value": float(positive_gaps.median()) if not positive_gaps.empty else float("nan"),
        },
        {
            "metric": "interval_min_positive_minutes",
            "value": float(positive_gaps.min()) if not positive_gaps.empty else float("nan"),
        },
        {
            "metric": "interval_max_minutes",
            "value": float(positive_gaps.max()) if not positive_gaps.empty else float("nan"),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("Comprobación periodo supervisado Acosol")
    print("=======================================")
    print(f"Periodo:                             {first_supervised} -> {last_supervised}")
    print(f"Snapshots supervisados:             {len(supervised)}")
    print(f"Ficheros auditados en ese periodo:  {len(supervised_audit)}")

    print("\n1) Estructura y contenido")
    print("-------------------------")
    print(f"Problemas de parsing:                {len(parse_errors)}")
    print(f"Timestamp interno inconsistente:     {len(timestamp_inconsistent)}")
    print(f"Timestamp nombre != interno:         {len(timestamp_mismatches)}")
    print(f"Ficheros con tags ausentes:          {len(missing_tags)}")
    print(f"Ficheros con tags desconocidos:      {len(unknown_tags)}")
    print(f"Ficheros con tags duplicados:        {len(duplicate_tags)}")
    print(f"Ficheros con valores inválidos:      {len(invalid_values)}")
    print(f"Ficheros sin las 9 señales:          {len(incomplete_expected_tags)}")
    print(f"Ficheros sin tag THM:                {len(missing_thm_tag)}")
    print(f"Ficheros con THM inválido:           {len(invalid_thm)}")

    print("\n2) Dataset supervisado")
    print("-----------------------")
    print(f"Filas con quality_flags != ok:       {len(supervised_quality_flagged)}")
    print(f"Timestamps duplicados:               {len(duplicates)}")
    print(f"Intervalos <= 0:                     {len(non_positive)}")
    print(f"Huecos > {args.gap_threshold_minutes:g} min:                    {len(gaps)}")

    if not positive_gaps.empty:
        print(f"Intervalo mediano:                   {positive_gaps.median():.3f} min")
        print(f"Intervalo mínimo positivo:           {positive_gaps.min():.3f} min")
        print(f"Intervalo máximo:                    {positive_gaps.max():.3f} min")

    if not gaps.empty:
        print("\nHuecos detectados:")
        print(
            gaps[
                [
                    "previous_timestamp",
                    "timestamp",
                    "gap_minutes",
                    "source_file",
                ]
            ].to_string(index=False)
        )

    print("\n3) Resultado")
    print("------------")
    critical_counts = [
        len(parse_errors),
        len(timestamp_inconsistent),
        len(timestamp_mismatches),
        len(missing_tags),
        len(unknown_tags),
        len(duplicate_tags),
        len(invalid_values),
        len(incomplete_expected_tags),
        len(missing_thm_tag),
        len(invalid_thm),
        len(duplicates),
        len(non_positive),
        len(gaps),
    ]

    if all(count == 0 for count in critical_counts):
        print("[OK] El periodo supervisado no presenta incidencias estructurales ni temporales.")
    else:
        print("[REVISAR] Se han detectado incidencias dentro del periodo supervisado.")

    print("\nInformes generados:")
    print(f"- {summary_path}")
    print(f"- {audit_issues_path}")
    print(f"- {gaps_path}")
    print(f"- {duplicates_path}")
    print(f"- {quality_flags_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
