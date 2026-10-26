from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("acosol_sftp_dataset_v0_1") / "acosol_sftp_supervised_v0_1.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auditoria descriptiva de bromuro en Acosol SFTP.")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("acosol_sftp_dataset_v0_1") / "bromuro_audit",
    )
    p.add_argument("--near-zero-threshold", type=float, default=0.001)
    p.add_argument("--meaningful-threshold", type=float, default=0.05)
    p.add_argument("--gap-minutes", type=float, default=7.5)
    return p.parse_args()


def classify(values: pd.Series, near_zero: float, meaningful: float) -> pd.Series:
    state = pd.Series("low_nonzero", index=values.index, dtype="object")
    state[values.isna()] = "missing"
    state[values.abs() <= near_zero] = "near_zero"
    state[values >= meaningful] = "meaningful"
    return state


def build_runs(df: pd.DataFrame, gap_minutes: float) -> pd.DataFrame:
    work = df.copy()
    gap = work["timestamp"].diff().dt.total_seconds().div(60.0)
    change = work["bromuro_state"].ne(work["bromuro_state"].shift())
    work["run_id"] = (gap.isna() | (gap > gap_minutes) | change).cumsum()
    rows = []
    for run_id, g in work.groupby("run_id"):
        rows.append({
            "run_id": int(run_id),
            "state": g["bromuro_state"].iloc[0],
            "start": g["timestamp"].min(),
            "end": g["timestamp"].max(),
            "rows": len(g),
            "duration_minutes": (g["timestamp"].max() - g["timestamp"].min()).total_seconds() / 60.0 if len(g) > 1 else 0.0,
            "bromuro_min": g["bromuro"].min(),
            "bromuro_median": g["bromuro"].median(),
            "bromuro_mean": g["bromuro"].mean(),
            "bromuro_max": g["bromuro"].max(),
            "unique_values": int(g["bromuro"].nunique(dropna=True)),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    if args.near_zero_threshold < 0:
        raise SystemExit("--near-zero-threshold debe ser >= 0")
    if args.meaningful_threshold <= args.near_zero_threshold:
        raise SystemExit("--meaningful-threshold debe ser mayor que --near-zero-threshold")
    if args.gap_minutes <= 0:
        raise SystemExit("--gap-minutes debe ser > 0")
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)

    df = pd.read_csv(args.input_csv)
    missing = sorted({"timestamp", "bromuro"} - set(df.columns))
    if missing:
        raise ValueError("Faltan columnas: " + ", ".join(missing))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["bromuro"] = pd.to_numeric(df["bromuro"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset vacio")

    df["bromuro_state"] = classify(df["bromuro"], args.near_zero_threshold, args.meaningful_threshold)
    df["date"] = df["timestamp"].dt.date

    state_counts = df["bromuro_state"].value_counts(dropna=False).rename_axis("state").reset_index(name="rows")
    state_counts["pct"] = state_counts["rows"] / len(df) * 100.0

    daily = df.groupby("date").agg(
        rows=("bromuro", "size"),
        valid=("bromuro", "count"),
        bromuro_min=("bromuro", "min"),
        bromuro_mean=("bromuro", "mean"),
        bromuro_median=("bromuro", "median"),
        bromuro_max=("bromuro", "max"),
        bromuro_std=("bromuro", "std"),
        unique_values=("bromuro", "nunique"),
    ).reset_index()

    states_by_day = pd.crosstab(df["date"], df["bromuro_state"]).reset_index()
    daily = daily.merge(states_by_day, on="date", how="left")
    for col in ["near_zero", "low_nonzero", "meaningful", "missing"]:
        if col not in daily.columns:
            daily[col] = 0
    daily["near_zero_pct"] = daily["near_zero"] / daily["rows"] * 100.0
    daily["meaningful_pct"] = daily["meaningful"] / daily["rows"] * 100.0

    runs = build_runs(df, args.gap_minutes)
    meaningful = df[df["bromuro_state"].eq("meaningful")].copy()
    low_nonzero = df[df["bromuro_state"].eq("low_nonzero")].copy()

    transitions = df[df["bromuro_state"].ne(df["bromuro_state"].shift())].copy()
    transitions["previous_state"] = df["bromuro_state"].shift()
    transitions["previous_bromuro"] = df["bromuro"].shift()

    rounded = (
        df.assign(bromuro_rounded=df["bromuro"].round(5))
        .groupby("bromuro_rounded", dropna=False)
        .agg(count=("bromuro", "size"), first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"))
        .reset_index()
        .sort_values(["count", "bromuro_rounded"], ascending=[False, False])
    )

    summary = pd.DataFrame([{
        "rows": len(df),
        "valid_rows": int(df["bromuro"].notna().sum()),
        "missing_rows": int(df["bromuro"].isna().sum()),
        "start": df["timestamp"].min(),
        "end": df["timestamp"].max(),
        "bromuro_min": df["bromuro"].min(),
        "bromuro_median": df["bromuro"].median(),
        "bromuro_mean": df["bromuro"].mean(),
        "bromuro_max": df["bromuro"].max(),
        "bromuro_std": df["bromuro"].std(),
        "unique_values": int(df["bromuro"].nunique(dropna=True)),
        "near_zero_threshold": args.near_zero_threshold,
        "meaningful_threshold": args.meaningful_threshold,
        "near_zero_rows": int(df["bromuro_state"].eq("near_zero").sum()),
        "near_zero_pct": float(df["bromuro_state"].eq("near_zero").mean() * 100.0),
        "meaningful_rows": int(df["bromuro_state"].eq("meaningful").sum()),
        "meaningful_pct": float(df["bromuro_state"].eq("meaningful").mean() * 100.0),
        "first_meaningful_timestamp": meaningful["timestamp"].min() if len(meaningful) else pd.NaT,
        "last_meaningful_timestamp": meaningful["timestamp"].max() if len(meaningful) else pd.NaT,
    }])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": args.output_dir / "bromuro_summary.csv",
        "states": args.output_dir / "bromuro_state_counts.csv",
        "daily": args.output_dir / "bromuro_daily_summary.csv",
        "runs": args.output_dir / "bromuro_state_runs.csv",
        "meaningful": args.output_dir / "bromuro_meaningful_rows.csv",
        "low_nonzero": args.output_dir / "bromuro_low_nonzero_rows.csv",
        "transitions": args.output_dir / "bromuro_state_transitions.csv",
        "values": args.output_dir / "bromuro_value_counts.csv",
    }
    for key, frame in [
        ("summary", summary), ("states", state_counts), ("daily", daily),
        ("runs", runs), ("meaningful", meaningful), ("low_nonzero", low_nonzero),
        ("transitions", transitions), ("values", rounded),
    ]:
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")

    print("Auditoria de bromuro Acosol")
    print("===========================")
    print(f"Dataset:                {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Filas:                  {len(df)}")
    print(f"Valores validos:        {df['bromuro'].notna().sum()}")
    print(f"Valores ausentes:       {df['bromuro'].isna().sum()}")
    print(f"Valores unicos:         {df['bromuro'].nunique(dropna=True)}")
    print(f"Rango:                  {df['bromuro'].min():.6f} -> {df['bromuro'].max():.6f} mg/L")
    print(f"Mediana:                {df['bromuro'].median():.6f} mg/L")
    print(f"Media:                  {df['bromuro'].mean():.6f} mg/L")
    print(f"\nNear-zero: |bromuro| <= {args.near_zero_threshold:g} mg/L")
    print(f"Meaningful: bromuro >= {args.meaningful_threshold:g} mg/L")
    print("\nDistribucion por estado:")
    print(state_counts.to_string(index=False, formatters={"pct": lambda x: f"{x:.3f}%"}))

    if len(meaningful):
        print("\nPeriodo con valores meaningful:")
        print(f"Primero:                {meaningful['timestamp'].min()}")
        print(f"Ultimo:                 {meaningful['timestamp'].max()}")
        print(f"Maximo observado:       {meaningful['bromuro'].max():.6f} mg/L")
    else:
        print("\nNo se han encontrado valores meaningful.")

    print("\nResumen diario:")
    cols = ["date", "rows", "bromuro_min", "bromuro_median", "bromuro_max", "unique_values", "near_zero_pct", "meaningful_pct"]
    print(daily[cols].to_string(index=False, formatters={"near_zero_pct": lambda x: f"{x:.2f}%", "meaningful_pct": lambda x: f"{x:.2f}%"}))

    print("\nTransiciones de estado:")
    tcols = ["timestamp", "previous_state", "previous_bromuro", "bromuro_state", "bromuro"]
    if "source_file" in transitions.columns:
        tcols.append("source_file")
    print(transitions[tcols].head(30).to_string(index=False))

    print("\nRuns near-zero mas largos:")
    zr = runs[runs["state"].eq("near_zero")].sort_values(["duration_minutes", "rows"], ascending=False)
    print(zr.head(10).to_string(index=False) if len(zr) else "No hay runs near-zero.")

    print("\nValores mas frecuentes:")
    print(rounded.head(20).to_string(index=False))

    print("\nSalidas:")
    for path in outputs.values():
        print(f"- {path}")
    print("\n[INFO] Auditoria descriptiva: no modifica ni elimina observaciones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
