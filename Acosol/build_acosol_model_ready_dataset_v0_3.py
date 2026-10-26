from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


VERSION = "v0.3"
DEFAULT_INPUT = Path("acosol_sftp_dataset_v0_1") / "acosol_sftp_supervised_v0_1.csv"

INCIDENT_START = pd.Timestamp("2026-09-12 12:15:15")
INCIDENT_END = pd.Timestamp("2026-09-12 13:10:15")
INCIDENT_MIN_THM = 100.0

ACTIVE_SENSOR_FEATURES = [
    "cloro",
    "turbidez",
    "conductividad",
    "ph",
    "cot",
]

EXCLUDED_SENSOR_FEATURES = {
    "temperatura": "excluded_static_hach_signal",
    "bromuro": "excluded_unreliable_signal",
    "tic": "excluded_by_plant_decision",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye dataset model-ready Acosol v0.3 manteniendo trazabilidad "
            "y excluyendo solo la incidencia THM ~131 del 12/09/2026."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("acosol_sftp_dataset_v0_3"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"No existe: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = {
        "timestamp",
        "thm",
        *ACTIVE_SENSOR_FEATURES,
        *EXCLUDED_SENSOR_FEATURES.keys(),
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["thm"] = pd.to_numeric(df["thm"], errors="coerce")

    for col in [*ACTIVE_SENSOR_FEATURES, *EXCLUDED_SENSOR_FEATURES.keys()]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", "thm"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    in_incident_window = (
        (df["timestamp"] >= INCIDENT_START)
        & (df["timestamp"] <= INCIDENT_END)
    )

    df["thm_131_incident"] = (
        in_incident_window
        & (df["thm"] >= INCIDENT_MIN_THM)
    )

    df["model_eligible"] = ~df["thm_131_incident"]
    df["model_exclusion_reason"] = ""
    df.loc[
        df["thm_131_incident"],
        "model_exclusion_reason",
    ] = "confirmed_plant_incident_thm_131_2026_09_12"

    eligible = df[df["model_eligible"]].copy()
    excluded = df[~df["model_eligible"]].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotated_path = args.output_dir / "acosol_sftp_supervised_annotated_v0_3.csv"
    training_path = args.output_dir / "acosol_sftp_model_training_v0_3.csv"
    excluded_path = args.output_dir / "acosol_sftp_model_excluded_rows_v0_3.csv"
    metadata_path = args.output_dir / "acosol_sftp_model_metadata_v0_3.json"

    df.to_csv(annotated_path, index=False, encoding="utf-8-sig")
    eligible.to_csv(training_path, index=False, encoding="utf-8-sig")
    excluded.to_csv(excluded_path, index=False, encoding="utf-8-sig")

    metadata = {
        "version": VERSION,
        "input_csv": str(args.input_csv),
        "rows_input": int(len(df)),
        "rows_model_eligible": int(len(eligible)),
        "rows_excluded": int(len(excluded)),
        "target": "thm",
        "active_sensor_features": ACTIVE_SENSOR_FEATURES,
        "excluded_sensor_features": EXCLUDED_SENSOR_FEATURES,
        "thm_incident_rule": {
            "start": str(INCIDENT_START),
            "end": str(INCIDENT_END),
            "min_thm_ug_l": INCIDENT_MIN_THM,
            "reason": (
                "Specific THM ~131 ug/L incident on 12/09/2026, confirmed "
                "by plant on 14/10/2026. Exclude only the 12 affected rows; "
                "the following ~12.75 ug/L readings are valid and retained."
            ),
        },
        "plant_decisions": [
            "THM ~131 ug/L on 12/09/2026 is a confirmed incident; exclude 12 rows only.",
            "The following THM ~12.75 ug/L segment is valid and retained.",
            "TIC is excluded from the model by confirmed plant decision.",
            "Bromide is retained in raw/audit but excluded due to ongoing meter issues.",
            "Temperature is retained but excluded due to confirmed reading issues.",
            "No generic THM > 100 deletion rule is applied.",
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Construcción dataset model-ready Acosol v0.3")
    print("============================================")
    print(f"Entrada:                    {args.input_csv}")
    print(f"Periodo:                    {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Filas entrada:              {len(df)}")
    print(f"Filas model-eligible:       {len(eligible)}")
    print(f"Filas excluidas:            {len(excluded)}")
    print(
        "Incidencia THM ~131:        "
        f"{INCIDENT_START} -> {INCIDENT_END}"
    )

    if len(excluded):
        print(
            f"THM excluido min/max:        "
            f"{excluded['thm'].min():.5f} / {excluded['thm'].max():.5f}"
        )
        print("\nFilas excluidas:")
        cols = ["timestamp", "thm"]
        if "source_file" in excluded.columns:
            cols.append("source_file")
        print(excluded[cols].to_string(index=False))

    retained_10_13 = eligible[
        (eligible["thm"] >= 10.0)
        & (eligible["thm"] < 14.0)
    ]
    print(
        f"\nFilas THM 10-13.999 conservadas: {len(retained_10_13)}"
    )

    print("\nFeatures activas:")
    for feature in ACTIVE_SENSOR_FEATURES:
        print(f"- {feature}")

    print("\nFeatures conservadas pero excluidas del modelo:")
    for feature, reason in EXCLUDED_SENSOR_FEATURES.items():
        print(f"- {feature}: {reason}")

    print("\nSalidas:")
    print(f"- {annotated_path}")
    print(f"- {training_path}")
    print(f"- {excluded_path}")
    print(f"- {metadata_path}")

    if len(excluded) != 12:
        print(
            "\n[AVISO] El número de filas excluidas no es 12. "
            "Revisar la incidencia antes de considerar estable esta versión."
        )

    print("\n[OK] Dataset model-ready v0.3 construido.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
