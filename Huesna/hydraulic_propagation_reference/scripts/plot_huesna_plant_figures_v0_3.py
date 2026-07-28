from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_VERSION = "0.3.0"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"No existe el fichero: {path}")
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = lookup.get(candidate.strip().lower())
        if found is not None:
            return found
    return None


def finite_number(value: float, name: str, minimum: Optional[float] = None) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} debe ser un número finito")
    if minimum is not None and numeric < minimum:
        raise ValueError(f"{name} debe ser >= {minimum}")
    return numeric


def date_range_label(df: pd.DataFrame, time_col: str) -> str:
    timestamps = pd.to_datetime(df[time_col], errors="coerce").dropna()
    if timestamps.empty:
        return "intervalo sin fechas válidas"
    start = timestamps.min().strftime("%Y-%m-%d %H:%M")
    end = timestamps.max().strftime("%Y-%m-%d %H:%M")
    return f"{start} → {end}"


def plot_etap_prediction(
    predictions: pd.DataFrame,
    outdir: Path,
    safety_margin: float,
    prealert_threshold: float,
    alert_threshold: float,
) -> bool:
    time_col = find_col(predictions, ["timestamp"])
    control_col = find_col(predictions, ["y_true"])
    prediction_col = find_col(predictions, ["pred_selected"])
    if time_col is None or prediction_col is None:
        raise ValueError("El CSV de predicciones debe contener timestamp y pred_selected")

    df = predictions.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[prediction_col] = pd.to_numeric(df[prediction_col], errors="coerce")
    df = df[df[time_col].notna()].sort_values(time_col)
    if df.empty or df[prediction_col].notna().sum() == 0:
        raise ValueError("No hay predicciones válidas para generar la figura de ETAP")
    df["pred_upper"] = df[prediction_col] + safety_margin

    plt.figure(figsize=(12, 5))
    if control_col is not None:
        df[control_col] = pd.to_numeric(df[control_col], errors="coerce")
        if df[control_col].notna().any():
            plt.plot(df[time_col], df[control_col], label="THM real en ETAP")
    plt.plot(df[time_col], df[prediction_col], label="Predicción del soft-sensor en ETAP")
    plt.plot(
        df[time_col],
        df["pred_upper"],
        label=f"Predicción + margen conservador (+{safety_margin:g})",
    )
    plt.axhline(
        prealert_threshold,
        linestyle="--",
        linewidth=1,
        label=f"Prealerta ETAP {prealert_threshold:g}",
    )
    plt.axhline(
        alert_threshold,
        linestyle=":",
        linewidth=1,
        label=f"Alerta ETAP {alert_threshold:g}",
    )
    plt.title(f"Soft-sensor THM en ETAP — {date_range_label(df, time_col)}")
    plt.xlabel("Fecha")
    plt.ylabel("THM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "01_etap_prediccion_vs_real_y_margen_conservador.png")
    return True


def plot_error_time(predictions: pd.DataFrame, outdir: Path) -> bool:
    time_col = find_col(predictions, ["timestamp"])
    control_col = find_col(predictions, ["y_true"])
    prediction_col = find_col(predictions, ["pred_selected"])
    if time_col is None or control_col is None or prediction_col is None:
        return False

    df = predictions.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[control_col] = pd.to_numeric(df[control_col], errors="coerce")
    df[prediction_col] = pd.to_numeric(df[prediction_col], errors="coerce")
    df = df[df[time_col].notna()].sort_values(time_col)
    df["error"] = df[prediction_col] - df[control_col]
    if df["error"].notna().sum() == 0:
        return False

    plt.figure(figsize=(12, 4))
    plt.plot(df[time_col], df["error"], label="Error modelo - THM real")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title(f"Error temporal del soft-sensor — {date_range_label(df, time_col)}")
    plt.xlabel("Fecha")
    plt.ylabel("Error THM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "02_error_temporal_modelo.png")
    return True


def plot_regulator_operational(
    alerts: pd.DataFrame,
    outdir: Path,
    regulator_node_id: str,
) -> bool:
    required = [
        "timestamp",
        "node_id",
        "thm_predicted",
        "thm_predicted_upper",
        "prealert_threshold",
        "alert_threshold",
        "alert_level_conservative",
    ]
    missing = [column for column in required if column not in alerts.columns]
    if missing:
        raise ValueError(f"Faltan columnas en operational alerts: {missing}")

    node_id = str(regulator_node_id).strip()
    df = alerts[alerts["node_id"].astype(str).str.strip().eq(node_id)].copy()
    if df.empty:
        return False

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    numeric_columns = [
        "thm_predicted",
        "thm_predicted_upper",
        "prealert_threshold",
        "alert_threshold",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df[df["timestamp"].notna()].sort_values("timestamp")
    if df.empty:
        return False

    aggregate = (
        df.groupby("timestamp", as_index=False)
        .agg(
            thm_predicted=("thm_predicted", "max"),
            thm_predicted_upper=("thm_predicted_upper", "max"),
            prealert_threshold=("prealert_threshold", "min"),
            alert_threshold=("alert_threshold", "min"),
        )
    )
    prealert_values = aggregate["prealert_threshold"].dropna().unique()
    alert_values = aggregate["alert_threshold"].dropna().unique()
    if len(prealert_values) == 0 or len(alert_values) == 0:
        return False
    if len(prealert_values) > 1 or len(alert_values) > 1:
        raise ValueError(
            f"El nodo {node_id} presenta más de un par de umbrales; revisa el CSV operacional"
        )

    prealert_threshold = float(prealert_values[0])
    alert_threshold = float(alert_values[0])
    if prealert_threshold >= alert_threshold:
        raise ValueError(f"Umbrales inválidos para el nodo {node_id}")

    plt.figure(figsize=(12, 5))
    plt.plot(aggregate["timestamp"], aggregate["thm_predicted"], label="Predicción propagada")
    plt.plot(
        aggregate["timestamp"],
        aggregate["thm_predicted_upper"],
        label="Predicción propagada + margen conservador",
    )
    plt.axhline(prealert_threshold, linestyle="--", linewidth=1, label="Umbral prealerta")
    plt.axhline(alert_threshold, linestyle=":", linewidth=1, label="Umbral alerta")
    plt.title(
        f"Regulador El Viso ({node_id}): predicción propagada — "
        f"{date_range_label(aggregate, 'timestamp')}"
    )
    plt.xlabel("Fecha de llegada estimada")
    plt.ylabel("THM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "03_el_viso_prediccion_propagada_margen_conservador.png")
    return True


def plot_alert_counts(alerts: pd.DataFrame, outdir: Path) -> bool:
    level_col = "alert_level_conservative"
    if level_col not in alerts.columns:
        return False

    levels = ["normal", "prealerta", "alerta", "not_classified"]
    counts = alerts[level_col].value_counts().reindex(levels).fillna(0)
    if counts.sum() == 0:
        return False

    plt.figure(figsize=(7, 4))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Distribución de estados de alerta conservadora")
    plt.xlabel("Estado")
    plt.ylabel("Número de registros propagados")
    for index, value in enumerate(counts.values):
        plt.text(index, value, f"{int(value):,}".replace(",", "."), ha="center", va="bottom")
    plt.grid(True, axis="y", alpha=0.3)
    savefig(outdir / "04_conteo_estados_alerta_conservadora.png")
    return True


def plot_margin_tradeoff(
    margins: pd.DataFrame,
    outdir: Path,
    safety_margin: float,
) -> list[str]:
    if "margin" not in margins.columns:
        return []

    df = margins.copy()
    numeric_columns = [
        "margin",
        "precision_event_ge_prealerta",
        "recall_event_ge_prealerta",
        "n_false_prealert_or_alert",
        "n_missed_prealert_or_alert",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.sort_values("margin")

    generated: list[str] = []
    if {
        "precision_event_ge_prealerta",
        "recall_event_ge_prealerta",
    }.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(df["margin"], df["precision_event_ge_prealerta"], marker="o", label="Precisión")
        plt.plot(df["margin"], df["recall_event_ge_prealerta"], marker="o", label="Recall")
        plt.axvline(
            safety_margin,
            linestyle="--",
            linewidth=1,
            label=f"Margen histórico +{safety_margin:g}",
        )
        plt.title("Compromiso entre precisión y recuperación de prealertas")
        plt.xlabel("Margen conservador añadido a la predicción")
        plt.ylabel("Métrica")
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = "05_margenes_precision_recall.png"
        savefig(outdir / filename)
        generated.append(filename)

    if {
        "n_false_prealert_or_alert",
        "n_missed_prealert_or_alert",
    }.issubset(df.columns):
        plt.figure(figsize=(8, 5))
        plt.plot(
            df["margin"],
            df["n_missed_prealert_or_alert"],
            marker="o",
            label="Prealertas perdidas",
        )
        plt.plot(
            df["margin"],
            df["n_false_prealert_or_alert"],
            marker="o",
            label="Falsas prealertas",
        )
        plt.axvline(
            safety_margin,
            linestyle="--",
            linewidth=1,
            label=f"Margen histórico +{safety_margin:g}",
        )
        plt.title("Efecto del margen conservador en alertas")
        plt.xlabel("Margen conservador añadido a la predicción")
        plt.ylabel("Número de registros")
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = "06_margenes_falsas_y_perdidas.png"
        savefig(outdir / filename)
        generated.append(filename)
    return generated


def write_readme(outdir: Path, safety_margin: float, regulator_node_id: str) -> None:
    text = f"""# Figuras Huesna

- `01`: soft-sensor y umbrales de ETAP.
- `02`: error temporal en el intervalo disponible.
- `03`: predicción propagada para el regulador `{regulator_node_id}`.
- `04`: recuento de estados conservadores.
- `05` y `06`: evaluación offline de márgenes.

El margen `+{safety_margin:g}` se conserva como referencia histórica; no es un óptimo productivo.
La evaluación usa el control en origen y no sustituye una validación con THM medido en destino.
"""
    (outdir / "README_figuras.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Figuras THM Huesna v{SCRIPT_VERSION}")
    parser.add_argument("--predictions-csv", required=True, type=Path)
    parser.add_argument("--operational-alerts-csv", required=True, type=Path)
    parser.add_argument("--margin-global-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--safety-margin", type=float, default=4.0)
    parser.add_argument("--etap-prealert-threshold", type=float, default=20.0)
    parser.add_argument("--etap-alert-threshold", type=float, default=25.0)
    parser.add_argument("--regulator-node-id", default="n50")
    args = parser.parse_args()

    safety_margin = finite_number(args.safety_margin, "--safety-margin", 0.0)
    etap_prealert = finite_number(args.etap_prealert_threshold, "--etap-prealert-threshold")
    etap_alert = finite_number(args.etap_alert_threshold, "--etap-alert-threshold")
    if etap_prealert >= etap_alert:
        raise ValueError("El umbral de prealerta de ETAP debe ser menor que el de alerta")
    regulator_node_id = str(args.regulator_node_id).strip()
    if not regulator_node_id:
        raise ValueError("--regulator-node-id no puede estar vacío")

    predictions_path = args.predictions_csv.resolve()
    alerts_path = args.operational_alerts_csv.resolve()
    margins_path = args.margin_global_csv.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    predictions = read_csv_flexible(predictions_path)
    alerts = read_csv_flexible(alerts_path)
    margins = read_csv_flexible(margins_path)

    generated: list[str] = []
    if plot_etap_prediction(predictions, outdir, safety_margin, etap_prealert, etap_alert):
        generated.append("01_etap_prediccion_vs_real_y_margen_conservador.png")
    if plot_error_time(predictions, outdir):
        generated.append("02_error_temporal_modelo.png")
    if plot_regulator_operational(alerts, outdir, regulator_node_id):
        generated.append("03_el_viso_prediccion_propagada_margen_conservador.png")
    if plot_alert_counts(alerts, outdir):
        generated.append("04_conteo_estados_alerta_conservadora.png")
    generated.extend(plot_margin_tradeoff(margins, outdir, safety_margin))
    write_readme(outdir, safety_margin, regulator_node_id)

    report = {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "predictions_csv": str(predictions_path),
        "operational_alerts_csv": str(alerts_path),
        "margin_global_csv": str(margins_path),
        "output_dir": str(outdir),
        "safety_margin": safety_margin,
        "etap_prealert_threshold": etap_prealert,
        "etap_alert_threshold": etap_alert,
        "regulator_node_id": regulator_node_id,
        "generated_figures": generated,
        "evaluation_note": (
            "Las figuras de márgenes usan una evaluación offline del control en origen; "
            "no validan THM medido en los destinos hidráulicos."
        ),
    }
    write_json(outdir / "figures_report.json", report)

    print(f"Figuras generadas: {len(generated)}")
    for filename in generated:
        print(f"  - {outdir / filename}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
