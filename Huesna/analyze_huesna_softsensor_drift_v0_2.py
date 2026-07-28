from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_VARIANTS = [
    "pred_selected",
    "pred_hgb_all_calendar_operational",
    "pred_hgb_no_doy_operational",
    "pred_hgb_no_calendar_operational",
    "pred_hgb_all_calendar_include",
    "pred_hgb_no_doy_include",
    "pred_hgb_no_calendar_include",
]

DEFAULT_KEY_FEATURES = [
    "doy_sin",
    "minute_day_sin",
    "Temperatura_rollmean_15",
    "Temperatura_rollmin_5",
    "conductividad_rollmean_240",
    "Temperatura_rollmean_30",
    "Temperatura_rollmean_5",
    "PH_rollrange_15",
    "conductividad_rollmean_180",
    "Temperatura_rollmax_60",
    "PH",
    "UVAS",
    "conductividad",
    "turbidez",
    "Cloro",
    "Temperatura",
]


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero: {path}")
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def parse_list(text: str, default: list[str]) -> list[str]:
    if not str(text).strip():
        return list(default)
    return [p.strip() for p in str(text).split(",") if p.strip()]


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def metrics(y_true: pd.Series, y_pred: pd.Series, label: str) -> dict[str, object]:
    y = to_num(y_true)
    p = to_num(y_pred)
    mask = y.notna() & p.notna()
    y = y[mask]
    p = p[mask]
    n = int(mask.sum())
    if n == 0:
        return {"variant": label, "n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "smape": np.nan, "r2": np.nan}

    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))

    denom = (np.abs(y) + np.abs(p)) / 2.0
    smape = float(np.mean(np.where(denom > 0, np.abs(err) / denom, 0.0)) * 100.0)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {"variant": label, "n": n, "mae": mae, "rmse": rmse, "bias": bias, "smape": smape, "r2": r2}


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def build_daily(pred: pd.DataFrame, target_col: str, variants: list[str]) -> pd.DataFrame:
    rows = []
    for date, g in pred.groupby("date", sort=True):
        row: dict[str, object] = {
            "date": date,
            "n": int(len(g)),
            f"{target_col}_mean": float(to_num(g[target_col]).mean()),
            f"{target_col}_min": float(to_num(g[target_col]).min()),
            f"{target_col}_max": float(to_num(g[target_col]).max()),
        }
        for col in variants:
            m = metrics(g[target_col], g[col], col)
            row[f"{col}_mean"] = float(to_num(g[col]).mean())
            row[f"{col}_mae"] = m["mae"]
            row[f"{col}_rmse"] = m["rmse"]
            row[f"{col}_bias"] = m["bias"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_daily(model_dir: Path, time_col: str, target_col: str, selected_col: str, key_features: list[str]) -> pd.DataFrame | None:
    path = model_dir / "test_features_debug.csv"
    if not path.exists():
        return None

    df = read_csv_flexible(path)
    if time_col not in df.columns:
        return None

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[df[time_col].notna()].copy()
    df["date"] = df[time_col].dt.strftime("%Y-%m-%d")

    cols = [c for c in [target_col, selected_col] + key_features if c in df.columns]
    for c in cols:
        df[c] = to_num(df[c])

    agg: dict[str, tuple[str, str]] = {"n": (time_col, "size")}
    for c in cols:
        agg[f"{c}_mean"] = (c, "mean")
        agg[f"{c}_min"] = (c, "min")
        agg[f"{c}_max"] = (c, "max")

    return df.groupby("date", as_index=False).agg(**agg)


def build_feature_drift(model_dir: Path, key_features: list[str]) -> pd.DataFrame | None:
    path = model_dir / "split_feature_distribution_debug.csv"
    if not path.exists():
        return None

    df = read_csv_flexible(path)
    required_base = {"split", "feature", "mean", "std", "min", "max"}
    if not required_base.issubset(df.columns):
        return None
    low_q = "p05" if "p05" in df.columns else ("p10" if "p10" in df.columns else None)
    high_q = "p95" if "p95" in df.columns else ("p90" if "p90" in df.columns else None)
    if low_q is None or high_q is None:
        return None

    df = df[df["feature"].isin(key_features)].copy()
    if df.empty:
        return None

    numeric_cols = ["count", "mean", "std", "min", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "max"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = to_num(df[c])

    rows = []
    for feat, g in df.groupby("feature"):
        parts = {str(r["split"]): r for _, r in g.iterrows()}
        train = parts.get("train")
        val = parts.get("val")
        test = parts.get("test")
        if train is None or val is None or test is None:
            continue

        train_mean = float(train["mean"])
        val_mean = float(val["mean"])
        test_mean = float(test["mean"])
        train_std = float(train["std"]) if not pd.isna(train["std"]) else np.nan
        val_std = float(val["std"]) if not pd.isna(val["std"]) else np.nan

        z_train = (test_mean - train_mean) / train_std if train_std and not math.isnan(train_std) else np.nan
        z_val = (test_mean - val_mean) / val_std if val_std and not math.isnan(val_std) else np.nan

        train_p05 = float(train[low_q])
        train_p95 = float(train[high_q])
        val_p05 = float(val[low_q])
        val_p95 = float(val[high_q])

        rows.append({
            "feature": feat,
            "train_mean": train_mean,
            "val_mean": val_mean,
            "test_mean": test_mean,
            "train_std": train_std,
            "val_std": val_std,
            "test_vs_train_std_z": z_train,
            "test_vs_val_std_z": z_val,
            "train_p05": train_p05,
            "train_p95": train_p95,
            "val_p05": val_p05,
            "val_p95": val_p95,
            "test_outside_train_p05_p95": bool(test_mean < train_p05 or test_mean > train_p95),
            "test_outside_val_p05_p95": bool(test_mean < val_p05 or test_mean > val_p95),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_test_vs_val_std_z"] = out["test_vs_val_std_z"].abs()
    return out.sort_values(["test_outside_val_p05_p95", "abs_test_vs_val_std_z"], ascending=[False, False])


def plot_daily_predictions(daily: pd.DataFrame, target_col: str, variants: list[str], outdir: Path) -> None:
    x = pd.to_datetime(daily["date"])
    plt.figure(figsize=(11, 5))
    plt.plot(x, daily[f"{target_col}_mean"], marker="o", label="THM real medio")
    for col in variants:
        c = f"{col}_mean"
        if c in daily.columns:
            plt.plot(x, daily[c], marker="o", label=col)
    plt.title("Media diaria: THM real frente a variantes del modelo")
    plt.xlabel("Fecha")
    plt.ylabel("THM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "01_daily_real_vs_predictions.png")


def plot_daily_bias(daily: pd.DataFrame, variants: list[str], outdir: Path) -> None:
    x = pd.to_datetime(daily["date"])
    plt.figure(figsize=(11, 5))
    for col in variants:
        c = f"{col}_bias"
        if c in daily.columns:
            plt.plot(x, daily[c], marker="o", label=col)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Sesgo diario: predicción - THM real")
    plt.xlabel("Fecha")
    plt.ylabel("Bias")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "02_daily_bias_by_variant.png")


def plot_doy(feature_daily: pd.DataFrame | None, outdir: Path, selected_col: str) -> None:
    if feature_daily is None or "doy_sin_mean" not in feature_daily.columns:
        return

    x = pd.to_datetime(feature_daily["date"])

    plt.figure(figsize=(11, 5))
    if "y_true_mean" in feature_daily.columns:
        plt.plot(x, feature_daily["y_true_mean"], marker="o", label="THM real medio")
    pred_col = f"{selected_col}_mean"
    if pred_col in feature_daily.columns:
        plt.plot(x, feature_daily[pred_col], marker="o", label="Predicción media")
    plt.title("Caída diaria de predicción")
    plt.xlabel("Fecha")
    plt.ylabel("THM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "03_daily_prediction_drop.png")

    plt.figure(figsize=(11, 4))
    plt.plot(x, feature_daily["doy_sin_mean"], marker="o", label="doy_sin medio")
    plt.title("Evolución diaria de doy_sin en test")
    plt.xlabel("Fecha")
    plt.ylabel("doy_sin")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "04_daily_doy_sin.png")


def plot_drift(drift: pd.DataFrame | None, outdir: Path) -> None:
    if drift is None or drift.empty or "test_vs_val_std_z" not in drift.columns:
        return

    top = drift.sort_values("abs_test_vs_val_std_z", ascending=False).head(12)
    plt.figure(figsize=(10, 5))
    plt.bar(top["feature"].astype(str), top["test_vs_val_std_z"])
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Drift de features: z-score test frente a validación")
    plt.xlabel("Feature")
    plt.ylabel("(media_test - media_val) / std_val")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    savefig(outdir / "05_feature_drift_vs_validation.png")


def write_report(
    outdir: Path,
    pred: pd.DataFrame,
    overall: pd.DataFrame,
    daily: pd.DataFrame,
    drift: pd.DataFrame | None,
    target_col: str,
    selected_col: str,
) -> None:
    start = pred["timestamp"].min() if "timestamp" in pred.columns else ""
    end = pred["timestamp"].max() if "timestamp" in pred.columns else ""

    selected = overall[overall["variant"] == selected_col]
    selected_text = ""
    if not selected.empty:
        r = selected.iloc[0]
        selected_text = f"- `{selected_col}`: MAE={r['mae']:.3f}, RMSE={r['rmse']:.3f}, bias={r['bias']:.3f}.\n"

    worst_text = ""
    bias_col = f"{selected_col}_bias"
    if bias_col in daily.columns:
        worst = daily.sort_values(bias_col).iloc[0]
        worst_text = (
            f"- Mayor subestimación diaria: {worst['date']} "
            f"(THM real medio={worst[f'{target_col}_mean']:.3f}, "
            f"predicción media={worst[f'{selected_col}_mean']:.3f}, "
            f"bias={worst[bias_col]:.3f}).\n"
        )

    drift_text = ""
    if drift is not None and not drift.empty:
        drift_text = "\n## Features con mayor indicio de deriva\n\n"
        for _, row in drift.head(8).iterrows():
            drift_text += (
                f"- `{row['feature']}`: train_mean={row['train_mean']:.4f}, "
                f"val_mean={row['val_mean']:.4f}, test_mean={row['test_mean']:.4f}, "
                f"z_test_vs_val={row['test_vs_val_std_z']:.2f}, "
                f"fuera_p05_p95_val={row['test_outside_val_p05_p95']}.\n"
            )

    text = f"""# Diagnóstico de deriva soft-sensor THM

## Ventana analizada

- Inicio test: {start}
- Fin test: {end}
- Filas test: {len(pred)}

## Métricas globales continuas

{selected_text}{worst_text}
Ver detalle completo en `variant_metrics_overall.csv` y `daily_prediction_summary.csv`.

## Interpretación

- Si la predicción cae de forma brusca mientras el THM real se mantiene alto, el problema no se resuelve únicamente con un margen fijo.
- Si las variantes alternativas no corrigen la caída, el problema apunta a deriva/cambio de régimen y no solo a selección de calendario.
- Si alguna feature importante queda fuera de la distribución de validación, conviene añadir control de deriva o incertidumbre operativa.

{drift_text}
## Salidas generadas

- `variant_metrics_overall.csv`
- `daily_prediction_summary.csv`
- `feature_daily_summary.csv`
- `feature_drift_summary.csv`
- `01_daily_real_vs_predictions.png`
- `02_daily_bias_by_variant.png`
- `03_daily_prediction_drop.png`
- `04_daily_doy_sin.png`
- `05_feature_drift_vs_validation.png`
"""
    (outdir / "drift_analysis_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analiza deriva del soft-sensor THM Huesna v0.2")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--target-col", default="y_true")
    parser.add_argument("--selected-col", default="pred_selected")
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--key-features", default=",".join(DEFAULT_KEY_FEATURES))
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pred_path = model_dir / "test_predictions.csv"
    pred = read_csv_flexible(pred_path)
    if args.time_col not in pred.columns:
        raise ValueError(f"No existe la columna {args.time_col} en {pred_path}")
    if args.target_col not in pred.columns:
        raise ValueError(f"No existe la columna {args.target_col} en {pred_path}")

    pred[args.time_col] = pd.to_datetime(pred[args.time_col], errors="coerce")
    pred = pred[pred[args.time_col].notna()].copy().sort_values(args.time_col).reset_index(drop=True)
    pred["date"] = pred[args.time_col].dt.strftime("%Y-%m-%d")
    pred["timestamp"] = pred[args.time_col]
    pred[args.target_col] = to_num(pred[args.target_col])

    variants = [c for c in parse_list(args.variants, DEFAULT_VARIANTS) if c in pred.columns]
    preferred_variants = []
    if "pred_selected" in variants:
        preferred_variants.append("pred_selected")
    for family in ("all_calendar", "no_doy", "no_calendar"):
        operational = f"pred_hgb_{family}_operational"
        include = f"pred_hgb_{family}_include"
        if operational in variants:
            preferred_variants.append(operational)
        elif include in variants:
            preferred_variants.append(include)
    variants = preferred_variants or variants
    if not variants:
        raise ValueError("No se encontró ninguna columna de predicción válida.")
    for c in variants:
        pred[c] = to_num(pred[c])

    key_features = parse_list(args.key_features, DEFAULT_KEY_FEATURES)

    overall = pd.DataFrame([metrics(pred[args.target_col], pred[c], c) for c in variants])
    overall = overall.sort_values(["rmse", "mae"], na_position="last")
    overall.to_csv(outdir / "variant_metrics_overall.csv", index=False, encoding="utf-8-sig")

    daily = build_daily(pred, args.target_col, variants)
    daily.to_csv(outdir / "daily_prediction_summary.csv", index=False, encoding="utf-8-sig")

    feature_daily = build_feature_daily(model_dir, args.time_col, args.target_col, args.selected_col, key_features)
    if feature_daily is not None:
        feature_daily.to_csv(outdir / "feature_daily_summary.csv", index=False, encoding="utf-8-sig")

    drift = build_feature_drift(model_dir, key_features)
    if drift is not None:
        drift.to_csv(outdir / "feature_drift_summary.csv", index=False, encoding="utf-8-sig")

    plot_daily_predictions(daily, args.target_col, variants, outdir)
    plot_daily_bias(daily, variants, outdir)
    plot_doy(feature_daily, outdir, args.selected_col)
    plot_drift(drift, outdir)

    write_report(outdir, pred, overall, daily, drift, args.target_col, args.selected_col)

    manifest = {
        "model_dir": str(model_dir),
        "output_dir": str(outdir),
        "variants": variants,
        "key_features": key_features,
    }
    (outdir / "drift_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Diagnóstico de deriva generado correctamente.")
    print(f"Salida: {outdir}")
    print("")
    print("Métricas globales:")
    print(overall.to_string(index=False))
    print("")
    print("Resumen diario:")
    cols = ["date", "n", f"{args.target_col}_mean"]
    for c in variants:
        for suffix in ("mean", "bias"):
            col = f"{c}_{suffix}"
            if col in daily.columns:
                cols.append(col)
    print(daily[cols].to_string(index=False))
    print("")
    print("Archivos principales:")
    for name in [
        "drift_analysis_report.md",
        "variant_metrics_overall.csv",
        "daily_prediction_summary.csv",
        "feature_daily_summary.csv",
        "feature_drift_summary.csv",
    ]:
        p = outdir / name
        if p.exists():
            print(f"  - {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
