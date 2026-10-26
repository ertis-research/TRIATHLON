from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("acosol_sftp_dataset_v0_1") / "acosol_sftp_supervised_v0_1.csv"

FEATURES = [
    "cloro",
    "turbidez",
    "conductividad",
    "ph",
    "temperatura",
    "cot",
    "tic",
    "bromuro",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audita episodios extremos de THM en el dataset supervisado SFTP de Acosol "
            "sin modificar ni excluir observaciones."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("acosol_sftp_dataset_v0_1") / "thm_extremes_audit",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Umbral descriptivo de THM para seleccionar episodios a revisar.",
    )
    parser.add_argument(
        "--event-gap-minutes",
        type=float,
        default=10.0,
        help=(
            "Dos filas altas pertenecen al mismo episodio si están separadas "
            "como máximo este número de minutos."
        ),
    )
    parser.add_argument(
        "--context-minutes",
        type=float,
        default=60.0,
        help="Contexto temporal antes y después de cada episodio.",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=5.0,
        help="Delta absoluto de THM usado para listar saltos bruscos.",
    )
    return parser.parse_args()


def robust_threshold(series: pd.Series) -> tuple[float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    median = float(values.median())
    mad = float(np.median(np.abs(values.to_numpy() - median)))

    if mad > 0:
        threshold = median + 8.0 * 1.4826 * mad
    else:
        threshold = float(values.quantile(0.999))

    return median, mad, threshold


def assign_events(selected: pd.DataFrame, event_gap_minutes: float) -> pd.DataFrame:
    selected = selected.copy().sort_values("timestamp").reset_index(drop=True)

    if selected.empty:
        selected["event_id"] = pd.Series(dtype="Int64")
        return selected

    gaps = selected["timestamp"].diff().dt.total_seconds() / 60.0
    new_event = gaps.isna() | (gaps > event_gap_minutes)
    selected["event_id"] = new_event.cumsum().astype(int)
    return selected


def summarize_events(
    df: pd.DataFrame,
    events: pd.DataFrame,
    context_minutes: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for event_id, event in events.groupby("event_id"):
        start = event["timestamp"].min()
        end = event["timestamp"].max()
        delta = pd.Timedelta(minutes=context_minutes)

        before = df[
            (df["timestamp"] >= start - delta)
            & (df["timestamp"] < start)
        ].copy()

        after = df[
            (df["timestamp"] > end)
            & (df["timestamp"] <= end + delta)
        ].copy()

        row: dict[str, object] = {
            "event_id": int(event_id),
            "start": start,
            "end": end,
            "duration_minutes": (
                (end - start).total_seconds() / 60.0 if end > start else 0.0
            ),
            "rows": len(event),
            "thm_min": event["thm"].min(),
            "thm_median": event["thm"].median(),
            "thm_max": event["thm"].max(),
            "thm_unique_values": int(event["thm"].nunique(dropna=True)),
            "thm_before_median": before["thm"].median() if len(before) else np.nan,
            "thm_before_max": before["thm"].max() if len(before) else np.nan,
            "thm_after_median": after["thm"].median() if len(after) else np.nan,
            "thm_after_max": after["thm"].max() if len(after) else np.nan,
        }

        for feature in FEATURES:
            row[f"{feature}_event_min"] = event[feature].min()
            row[f"{feature}_event_median"] = event[feature].median()
            row[f"{feature}_event_max"] = event[feature].max()
            row[f"{feature}_before_median"] = (
                before[feature].median() if len(before) else np.nan
            )
            row[f"{feature}_after_median"] = (
                after[feature].median() if len(after) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    if args.threshold <= 0:
        raise SystemExit("--threshold debe ser > 0.")
    if args.event_gap_minutes <= 0:
        raise SystemExit("--event-gap-minutes debe ser > 0.")
    if args.context_minutes <= 0:
        raise SystemExit("--context-minutes debe ser > 0.")
    if args.jump_threshold <= 0:
        raise SystemExit("--jump-threshold debe ser > 0.")

    if not args.input_csv.exists():
        raise FileNotFoundError(f"No existe: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = {"timestamp", "thm", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Faltan columnas obligatorias: " + ", ".join(missing))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["thm"] = pd.to_numeric(df["thm"], errors="coerce")

    for feature in FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

    df = (
        df.dropna(subset=["timestamp", "thm"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    df["previous_timestamp"] = df["timestamp"].shift(1)
    df["previous_thm"] = df["thm"].shift(1)
    df["delta_thm"] = df["thm"] - df["previous_thm"]
    df["abs_delta_thm"] = df["delta_thm"].abs()

    median, mad, robust_limit = robust_threshold(df["thm"])

    selected = df[
        (df["thm"] >= args.threshold)
        | (df["thm"] >= robust_limit)
    ].copy()

    selected = assign_events(
        selected,
        event_gap_minutes=args.event_gap_minutes,
    )

    event_summary = summarize_events(
        df=df,
        events=selected,
        context_minutes=args.context_minutes,
    )

    context_frames: list[pd.DataFrame] = []
    for event_id, event in selected.groupby("event_id"):
        start = event["timestamp"].min() - pd.Timedelta(minutes=args.context_minutes)
        end = event["timestamp"].max() + pd.Timedelta(minutes=args.context_minutes)

        context = df[
            (df["timestamp"] >= start)
            & (df["timestamp"] <= end)
        ].copy()
        context["event_id"] = int(event_id)
        context["inside_extreme_event"] = context["timestamp"].isin(event["timestamp"])
        context_frames.append(context)

    context_df = (
        pd.concat(context_frames, ignore_index=True)
        if context_frames
        else pd.DataFrame()
    )

    jumps = df[df["abs_delta_thm"] >= args.jump_threshold].copy()

    repeated = (
        df.assign(thm_rounded=df["thm"].round(3))
        .groupby("thm_rounded")
        .agg(
            count=("thm", "size"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
        .sort_values(["count", "thm_rounded"], ascending=[False, False])
    )

    repeated_high = repeated[
        repeated["thm_rounded"] >= args.threshold
    ].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = args.output_dir / "high_thm_rows.csv"
    event_summary_path = args.output_dir / "high_thm_event_summary.csv"
    context_path = args.output_dir / "high_thm_event_context.csv"
    jumps_path = args.output_dir / "thm_large_jumps.csv"
    repeated_path = args.output_dir / "high_thm_repeated_values.csv"

    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    event_summary.to_csv(event_summary_path, index=False, encoding="utf-8-sig")
    context_df.to_csv(context_path, index=False, encoding="utf-8-sig")
    jumps.to_csv(jumps_path, index=False, encoding="utf-8-sig")
    repeated_high.to_csv(repeated_path, index=False, encoding="utf-8-sig")

    print("Auditoría de extremos THM Acosol")
    print("================================")
    print(f"Dataset:              {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Filas:                {len(df)}")
    print(f"THM mediana:          {median:.6f}")
    print(f"THM MAD:              {mad:.6f}")
    print(f"Umbral robusto 8*MAD: {robust_limit:.6f}")
    print(f"Umbral descriptivo:   {args.threshold:.6f}")
    print(f"Filas seleccionadas:  {len(selected)}")
    print(f"Episodios:            {selected['event_id'].nunique() if len(selected) else 0}")

    if not event_summary.empty:
        print("\nResumen de episodios:")
        display_cols = [
            "event_id",
            "start",
            "end",
            "duration_minutes",
            "rows",
            "thm_min",
            "thm_median",
            "thm_max",
            "thm_unique_values",
            "thm_before_median",
            "thm_after_median",
        ]
        print(event_summary[display_cols].to_string(index=False))

    print(f"\nSaltos |ΔTHM| >= {args.jump_threshold:g}: {len(jumps)}")
    if not jumps.empty:
        cols = [
            "timestamp",
            "previous_timestamp",
            "previous_thm",
            "thm",
            "delta_thm",
        ]
        if "source_file" in jumps.columns:
            cols.append("source_file")
        print(jumps[cols].head(20).to_string(index=False))

    if not repeated_high.empty:
        print("\nValores altos repetidos (THM redondeado a 0.001):")
        print(repeated_high.head(20).to_string(index=False))

    print("\nSalidas:")
    print(f"- {selected_path}")
    print(f"- {event_summary_path}")
    print(f"- {context_path}")
    print(f"- {jumps_path}")
    print(f"- {repeated_path}")

    print(
        "\n[INFO] Esta auditoría es descriptiva: no elimina ni etiqueta "
        "automáticamente ninguna observación como errónea."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
