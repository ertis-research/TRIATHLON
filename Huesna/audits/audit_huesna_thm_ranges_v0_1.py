from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_SENSOR = "DEPOSITO ETAP MULTIPARAMETRICA"
DEFAULT_METRIC = "Trihalometanos"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auditoría histórica de THM para localizar periodos en rangos concretos sin modificar el dataset."
    )
    p.add_argument("--csv-dir", default="csv_descargados")
    p.add_argument("--output-dir", default="outputs_thm_range_audit_v0_1")
    p.add_argument("--sensor", default=DEFAULT_SENSOR)
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--range-low", type=float, default=30.0)
    p.add_argument("--range-high", type=float, default=35.0)
    p.add_argument(
        "--max-gap-minutes",
        type=float,
        default=5.0,
        help="Máximo hueco temporal para considerar consecutivos dos puntos dentro del rango.",
    )
    p.add_argument(
        "--sustained-min-minutes",
        type=float,
        default=30.0,
        help="Duración mínima para marcar un intervalo 30-35 como sostenido.",
    )
    return p.parse_args()


def parse_value(raw: str) -> float | None:
    text = str(raw).strip().replace(",", ".")
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in ".+-eE")
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def iter_target_rows(
    files: Iterable[Path], sensor: str, metric: str
) -> Iterable[tuple[pd.Timestamp, float, str]]:
    sensor_key = sensor.strip().casefold()
    metric_key = metric.strip().casefold()

    for path in files:
        try:
            fh = path.open("r", encoding="latin1", errors="replace", newline="")
        except OSError:
            continue
        with fh:
            reader = csv.reader(fh, delimiter=";")
            for row in reader:
                if len(row) < 4:
                    continue
                if str(row[1]).strip().casefold() != sensor_key:
                    continue
                if str(row[2]).strip().casefold() != metric_key:
                    continue
                ts = pd.to_datetime(row[0], errors="coerce")
                if pd.isna(ts):
                    continue
                value = parse_value(row[3])
                if value is None:
                    continue
                yield pd.Timestamp(ts), float(value), path.name


def build_intervals(
    df_range: pd.DataFrame,
    max_gap_minutes: float,
    sustained_min_minutes: float,
) -> pd.DataFrame:
    columns = [
        "start",
        "end",
        "duration_minutes",
        "samples",
        "mean",
        "median",
        "min",
        "max",
        "is_sustained",
    ]
    if df_range.empty:
        return pd.DataFrame(columns=columns)

    work = df_range.sort_values("timestamp").reset_index(drop=True).copy()
    gap = work["timestamp"].diff().dt.total_seconds().div(60.0)
    work["block"] = ((gap.isna()) | (gap > float(max_gap_minutes))).cumsum()

    rows = []
    for _, sub in work.groupby("block", sort=True):
        start = pd.Timestamp(sub["timestamp"].iloc[0])
        end = pd.Timestamp(sub["timestamp"].iloc[-1])
        duration = max(0.0, (end - start).total_seconds() / 60.0)
        values = sub["value"].astype(float)
        rows.append(
            {
                "start": start,
                "end": end,
                "duration_minutes": duration,
                "samples": int(len(sub)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "min": float(values.min()),
                "max": float(values.max()),
                "is_sustained": bool(duration >= float(sustained_min_minutes)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["duration_minutes", "samples"], ascending=[False, False]
    )


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {csv_dir}")

    rows = list(iter_target_rows(files, args.sensor, args.metric))
    if not rows:
        raise RuntimeError(
            f"No se encontraron datos para sensor='{args.sensor}' metric='{args.metric}'"
        )

    df = pd.DataFrame(rows, columns=["timestamp", "value", "source_file"])
    df = df.sort_values("timestamp").drop_duplicates(
        subset=["timestamp", "value", "source_file"]
    ).reset_index(drop=True)

    low = float(args.range_low)
    high = float(args.range_high)
    range_mask = df["value"].between(low, high, inclusive="both")
    df_range = df.loc[range_mask].copy()

    intervals_df = build_intervals(
        df_range,
        max_gap_minutes=float(args.max_gap_minutes),
        sustained_min_minutes=float(args.sustained_min_minutes),
    )

    daily = df.assign(date=df["timestamp"].dt.date.astype(str)).groupby("date")["value"].agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
    )
    daily_range = (
        df.assign(date=df["timestamp"].dt.date.astype(str), in_range=range_mask)
        .groupby("date")["in_range"]
        .agg(count_30_35="sum", pct_30_35="mean")
    )
    daily = daily.join(daily_range)
    daily["pct_30_35"] = 100.0 * daily["pct_30_35"]
    daily = daily.reset_index()

    buckets = [
        ("<=0.1", df["value"] <= 0.1),
        ("0.1-20", (df["value"] > 0.1) & (df["value"] < 20.0)),
        ("20-30", (df["value"] >= 20.0) & (df["value"] < 30.0)),
        (f"{low:g}-{high:g}", range_mask),
        ("35-40", (df["value"] > high) & (df["value"] <= 40.0)),
        (">40", df["value"] > 40.0),
        (">500", df["value"] > 500.0),
    ]
    bucket_rows = []
    for label, mask in buckets:
        bucket_rows.append(
            {
                "range": label,
                "count": int(mask.sum()),
                "pct_all": float(100.0 * mask.mean()),
            }
        )
    summary_df = pd.DataFrame(bucket_rows)

    df_range.to_csv(output_dir / "thm_values_30_35.csv", index=False)
    intervals_df.to_csv(output_dir / "thm_30_35_intervals.csv", index=False)
    daily.to_csv(output_dir / "thm_daily_summary.csv", index=False)
    summary_df.to_csv(output_dir / "thm_range_summary.csv", index=False)

    values = df["value"].astype(float)
    print("\n=== AUDITORÍA HISTÓRICA THM ===")
    print(f"CSV revisados:              {len(files)}")
    print(f"Muestras THM:               {len(df)}")
    print(f"Rango temporal:             {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Media / mediana:            {values.mean():.3f} / {values.median():.3f}")
    print(f"p25 / p75 / p95 / p99:     {values.quantile(.25):.3f} / {values.quantile(.75):.3f} / {values.quantile(.95):.3f} / {values.quantile(.99):.3f}")
    print(f"Min / max bruto:            {values.min():.3f} / {values.max():.3f}")
    print(f"Muestras entre {low:g}-{high:g}:      {int(range_mask.sum())} ({100.0 * range_mask.mean():.2f}%)")
    print(f"Intervalos detectados:      {len(intervals_df)}")
    print(f"Intervalos sostenidos >={args.sustained_min_minutes:g} min: {int(intervals_df['is_sustained'].sum()) if not intervals_df.empty else 0}")

    if not intervals_df.empty:
        print("\nTop 15 intervalos por duración:")
        print(
            intervals_df.head(15)[
                ["start", "end", "duration_minutes", "samples", "mean", "min", "max", "is_sustained"]
            ].to_string(index=False)
        )

    print("\nArchivos generados:")
    for name in [
        "thm_range_summary.csv",
        "thm_30_35_intervals.csv",
        "thm_daily_summary.csv",
        "thm_values_30_35.csv",
    ]:
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
