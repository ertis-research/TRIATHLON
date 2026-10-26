from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


VERSION = "v0.5_formal_validation"
TIME_COL = "timestamp"
TARGET_COL = "thm"

FORMAL_TRAIN_END_EXCLUSIVE = pd.Timestamp("2026-09-30 00:00:00")
FORMAL_VAL_END_EXCLUSIVE = pd.Timestamp("2026-10-01 00:00:00")
FORMAL_TEST_END_EXCLUSIVE = pd.Timestamp("2026-10-08 00:00:00")

BASELINE_REFERENCE = {
    "version": "v0.3.2",
    "development_data_end": "2026-09-28 17:25:19",
    "pooled_mae": 0.430363,
    "pooled_rmse": 0.918989,
    "pooled_smape_pct": 11.027,
    "pooled_bias": 0.070932,
    "pooled_r2": 0.130496,
}

SENSOR_FEATURES = ["cloro", "turbidez", "conductividad", "ph", "cot"]

TIME_FEATURES = [
    "hour",
    "dayofweek",
    "dayofyear",
    "hour_sin",
    "hour_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]

FEATURES = [*SENSOR_FEATURES, *TIME_FEATURES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validación formal out-of-time del soft-sensor THM de Acosol. "
            "TRAIN: hasta 29/09 inclusive; VAL: 30/09; "
            "TEST formal: 01/10-07/10 completos."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("acosol_sftp_dataset_v0_3")
        / "acosol_sftp_model_training_v0_3.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_outputs_acosol_thm_formal_validation_v0_5"),
    )
    parser.add_argument("--candidate-max-iter", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--min-iter", type=int, default=20)
    parser.add_argument("--expected-cadence-minutes", type=float, default=5.0)
    parser.add_argument("--high-thm-threshold", type=float, default=10.0)
    parser.add_argument("--operational-threshold", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = (
        df[TIME_COL].dt.hour
        + df[TIME_COL].dt.minute / 60.0
        + df[TIME_COL].dt.second / 3600.0
    )
    df["dayofweek"] = df[TIME_COL].dt.dayofweek.astype(float)
    df["dayofyear"] = df[TIME_COL].dt.dayofyear.astype(float)
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["dayofyear_sin"] = np.sin(
        2.0 * np.pi * df["dayofyear"] / 365.25
    )
    df["dayofyear_cos"] = np.cos(
        2.0 * np.pi * df["dayofyear"] / 365.25
    )
    return df


def make_hgb(max_iter: int, random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.04,
        max_iter=max_iter,
        l2_regularization=0.05,
        min_samples_leaf=20,
        early_stopping=False,
        random_state=random_state,
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual = y_pred - y_true
    abs_error = np.abs(residual)

    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 0

    smape = (
        float(np.mean(2.0 * abs_error[mask] / denominator[mask]) * 100.0)
        if np.any(mask)
        else np.nan
    )

    r2 = (
        float(r2_score(y_true, y_pred))
        if len(y_true) >= 2 and len(np.unique(y_true)) > 1
        else np.nan
    )

    pearson_r = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if len(y_true) >= 2
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
        else np.nan
    )

    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "smape_pct": smape,
        "bias_pred_minus_true": float(np.mean(residual)),
        "median_abs_error": float(np.median(abs_error)),
        "p90_abs_error": float(np.percentile(abs_error, 90)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "max_abs_error": float(np.max(abs_error)),
        "r2": r2,
        "pearson_r": pearson_r,
        "y_true_mean": float(np.mean(y_true)),
        "y_true_median": float(np.median(y_true)),
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_pred_min": float(np.min(y_pred)),
        "y_pred_max": float(np.max(y_pred)),
    }


def select_iterations(
    train: pd.DataFrame,
    val: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[int, float, float, int, pd.DataFrame]:
    candidate = make_hgb(
        max_iter=args.candidate_max_iter,
        random_state=args.random_state,
    )
    candidate.fit(train[FEATURES], train[TARGET_COL])

    y_val = val[TARGET_COL].to_numpy(dtype=float)

    best_iter = 0
    best_mae = float("inf")
    best_rmse = float("inf")
    no_improve = 0
    stop_iter = args.candidate_max_iter
    rows: list[dict[str, Any]] = []

    for iteration, prediction in enumerate(
        candidate.staged_predict(val[FEATURES]),
        start=1,
    ):
        y_pred = np.asarray(prediction, dtype=float)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_val, y_pred)))

        significant_improvement = (
            best_iter == 0
            or mae < (best_mae - args.min_delta)
        )

        if significant_improvement:
            best_iter = iteration
            best_mae = mae
            best_rmse = rmse
            no_improve = 0
        else:
            no_improve += 1

        rows.append(
            {
                "iteration": iteration,
                "val_mae": mae,
                "val_rmse": rmse,
                "best_iteration_so_far": best_iter,
                "best_val_mae_so_far": best_mae,
                "no_improve_count": no_improve,
                "significant_improvement": significant_improvement,
            }
        )

        if iteration >= args.min_iter and no_improve >= args.patience:
            stop_iter = iteration
            break

    if best_iter <= 0:
        raise RuntimeError(
            "No se pudo seleccionar un número de iteraciones."
        )

    return (
        best_iter,
        best_mae,
        best_rmse,
        stop_iter,
        pd.DataFrame(rows),
    )


def split_formal(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df[TIME_COL] < FORMAL_TRAIN_END_EXCLUSIVE].copy()

    val = df[
        (df[TIME_COL] >= FORMAL_TRAIN_END_EXCLUSIVE)
        & (df[TIME_COL] < FORMAL_VAL_END_EXCLUSIVE)
    ].copy()

    test = df[
        (df[TIME_COL] >= FORMAL_VAL_END_EXCLUSIVE)
        & (df[TIME_COL] < FORMAL_TEST_END_EXCLUSIVE)
    ].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            "TRAIN, VAL o TEST formal están vacíos. "
            "Comprueba que el dataset esté actualizado."
        )

    if test[TIME_COL].min() <= pd.Timestamp(
        BASELINE_REFERENCE["development_data_end"]
    ):
        raise ValueError(
            "El TEST formal no es posterior al periodo de desarrollo."
        )

    return train, val, test


def coverage_summary(
    part: pd.DataFrame,
    expected_cadence_minutes: float,
) -> dict[str, float]:
    timestamps = part[TIME_COL].sort_values().reset_index(drop=True)
    gaps = timestamps.diff().dt.total_seconds().div(60.0).dropna()

    start = timestamps.min()
    end = timestamps.max()

    expected_rows = int(
        math.floor(
            (end - start).total_seconds()
            / 60.0
            / expected_cadence_minutes
        )
    ) + 1

    actual_rows = int(len(part))
    coverage = (
        100.0 * actual_rows / expected_rows
        if expected_rows > 0
        else np.nan
    )

    return {
        "actual_rows": actual_rows,
        "expected_rows_approx": expected_rows,
        "coverage_pct_approx": float(coverage),
        "median_interval_minutes": (
            float(gaps.median()) if len(gaps) else np.nan
        ),
        "min_positive_interval_minutes": (
            float(gaps[gaps > 0].min())
            if np.any(gaps > 0)
            else np.nan
        ),
        "max_interval_minutes": (
            float(gaps.max()) if len(gaps) else np.nan
        ),
        "gaps_gt_7_5_minutes": int((gaps > 7.5).sum()),
        "gaps_gt_10_minutes": int((gaps > 10.0).sum()),
    }


def feature_shift_table(
    train_val: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for feature in SENSOR_FEATURES:
        ref = pd.to_numeric(
            train_val[feature],
            errors="coerce",
        ).dropna()

        tst = pd.to_numeric(
            test[feature],
            errors="coerce",
        ).dropna()

        ref_mean = float(ref.mean())
        ref_std = float(ref.std(ddof=1))
        ref_median = float(ref.median())
        ref_q1 = float(ref.quantile(0.25))
        ref_q3 = float(ref.quantile(0.75))
        ref_iqr = ref_q3 - ref_q1

        test_mean = float(tst.mean())
        test_median = float(tst.median())

        rows.append(
            {
                "feature": feature,
                "train_val_n": int(len(ref)),
                "test_n": int(len(tst)),
                "train_val_mean": ref_mean,
                "test_mean": test_mean,
                "train_val_std": ref_std,
                "standardized_mean_shift": (
                    (test_mean - ref_mean) / ref_std
                    if ref_std > 0
                    else np.nan
                ),
                "train_val_median": ref_median,
                "test_median": test_median,
                "train_val_iqr": ref_iqr,
                "robust_median_shift_iqr": (
                    (test_median - ref_median) / ref_iqr
                    if ref_iqr > 0
                    else np.nan
                ),
                "train_val_min": float(ref.min()),
                "train_val_max": float(ref.max()),
                "test_min": float(tst.min()),
                "test_max": float(tst.max()),
            }
        )

    return pd.DataFrame(rows)


def daily_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    data = predictions.copy()
    data["date"] = data[TIME_COL].dt.date

    rows: list[dict[str, Any]] = []

    for date, group in data.groupby("date", sort=True):
        rows.append(
            {
                "date": str(date),
                **compute_metrics(
                    group["y_true"].to_numpy(dtype=float),
                    group["y_pred"].to_numpy(dtype=float),
                ),
            }
        )

    return pd.DataFrame(rows)


def save_figures(
    predictions: pd.DataFrame,
    daily: pd.DataFrame,
    shift: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        predictions[TIME_COL],
        predictions["y_true"],
        label="THM real",
        linewidth=1.2,
    )
    ax.plot(
        predictions[TIME_COL],
        predictions["y_pred"],
        label="THM estimado",
        linewidth=1.2,
    )
    ax.set_title("Validación formal Acosol - THM real vs estimado")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("THM (µg/L)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "01_thm_real_vs_estimado.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        predictions["y_true"],
        predictions["y_pred"],
        s=12,
        alpha=0.5,
    )
    low = float(
        min(
            predictions["y_true"].min(),
            predictions["y_pred"].min(),
        )
    )
    high = float(
        max(
            predictions["y_true"].max(),
            predictions["y_pred"].max(),
        )
    )
    ax.plot([low, high], [low, high], linewidth=1.0)
    ax.set_title("THM real vs estimado - TEST formal")
    ax.set_xlabel("THM real (µg/L)")
    ax.set_ylabel("THM estimado (µg/L)")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "02_scatter_real_vs_estimado.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        predictions[TIME_COL],
        predictions["residual_pred_minus_true"],
        linewidth=1.0,
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Residuos - TEST formal")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Predicción - real (µg/L)")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "03_residuos_temporales.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(daily["date"], daily["mae"])
    ax.set_title("MAE diario - TEST formal")
    ax.set_xlabel("Día")
    ax.set_ylabel("MAE (µg/L)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(
        figures_dir / "04_mae_diario.png",
        dpi=180,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        shift["feature"],
        shift["standardized_mean_shift"].fillna(0.0),
    )
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("Shift de media entre TRAIN+VAL y TEST")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Δ media / desviación estándar TRAIN+VAL")
    fig.tight_layout()
    fig.savefig(
        figures_dir / "05_shift_variables.png",
        dpi=180,
    )
    plt.close(fig)


def main() -> int:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"No existe: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = {
        TIME_COL,
        TARGET_COL,
        *SENSOR_FEATURES,
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Faltan columnas obligatorias: "
            + ", ".join(missing)
        )

    df[TIME_COL] = pd.to_datetime(
        df[TIME_COL],
        errors="coerce",
    )
    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce",
    )

    for feature in SENSOR_FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df = (
        df.dropna(subset=[TIME_COL, TARGET_COL])
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("Dataset vacío.")

    if df[TIME_COL].duplicated().any():
        raise ValueError("Hay timestamps duplicados.")

    df = add_time_features(df)

    train, val, test = split_formal(df)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("VALIDACIÓN FORMAL DEL SOFT-SENSOR THM - ACOSOL")
    print("=" * 64)
    print(
        f"Dataset disponible: "
        f"{df[TIME_COL].min()} -> {df[TIME_COL].max()}"
    )
    print(
        f"Baseline congelado: {BASELINE_REFERENCE['version']} "
        f"(desarrollo hasta "
        f"{BASELINE_REFERENCE['development_data_end']})"
    )

    print("\nSplit formal fijado ANTES de evaluar TEST:")
    print(
        f"  TRAIN: {train[TIME_COL].min()} -> "
        f"{train[TIME_COL].max()} "
        f"({len(train)} filas)"
    )
    print(
        f"  VAL:   {val[TIME_COL].min()} -> "
        f"{val[TIME_COL].max()} "
        f"({len(val)} filas)"
    )
    print(
        f"  TEST:  {test[TIME_COL].min()} -> "
        f"{test[TIME_COL].max()} "
        f"({len(test)} filas)"
    )
    print(
        "  TEST calendario: 01/10/2026 00:00 "
        "-> 08/10/2026 00:00 (fin exclusivo)"
    )

    (
        selected_iter,
        val_mae,
        val_rmse,
        stop_iter,
        validation_curve,
    ) = select_iterations(
        train=train,
        val=val,
        args=args,
    )

    print("\nSelección de iteraciones usando SOLO VAL:")
    print(f"  Iteraciones seleccionadas: {selected_iter}")
    print(f"  Stop iteration:            {stop_iter}")
    print(f"  VAL MAE:                   {val_mae:.6f} µg/L")
    print(f"  VAL RMSE:                  {val_rmse:.6f} µg/L")

    train_val = pd.concat(
        [train, val],
        ignore_index=True,
    )

    model = make_hgb(
        max_iter=selected_iter,
        random_state=args.random_state,
    )
    model.fit(
        train_val[FEATURES],
        train_val[TARGET_COL],
    )

    y_true = test[TARGET_COL].to_numpy(dtype=float)
    y_pred = np.asarray(
        model.predict(test[FEATURES]),
        dtype=float,
    )

    formal_metrics = compute_metrics(
        y_true,
        y_pred,
    )

    dummy = DummyRegressor(strategy="median")
    dummy.fit(
        train_val[FEATURES],
        train_val[TARGET_COL],
    )
    dummy_pred = np.asarray(
        dummy.predict(test[FEATURES]),
        dtype=float,
    )
    dummy_metrics = compute_metrics(
        y_true,
        dummy_pred,
    )

    predictions = pd.DataFrame(
        {
            TIME_COL: test[TIME_COL].values,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual_pred_minus_true": (
                y_pred - y_true
            ),
            "abs_error": np.abs(
                y_pred - y_true
            ),
        }
    )

    if "source_file" in test.columns:
        predictions["source_file"] = (
            test["source_file"].values
        )

    daily = daily_metrics(predictions)
    coverage = coverage_summary(
        test,
        args.expected_cadence_minutes,
    )
    shift = feature_shift_table(
        train_val,
        test,
    )

    high_mask = y_true >= args.high_thm_threshold
    operational_mask = y_true >= args.operational_threshold

    high_metrics = (
        compute_metrics(
            y_true[high_mask],
            y_pred[high_mask],
        )
        if np.any(high_mask)
        else None
    )

    operational_metrics = (
        compute_metrics(
            y_true[operational_mask],
            y_pred[operational_mask],
        )
        if np.any(operational_mask)
        else None
    )

    print("\n" + "=" * 64)
    print("RESULTADO TEST FORMAL")
    print("=" * 64)
    print(f"N:              {formal_metrics['n']}")
    print(f"MAE:            {formal_metrics['mae']:.6f} µg/L")
    print(f"RMSE:           {formal_metrics['rmse']:.6f} µg/L")
    print(f"sMAPE:          {formal_metrics['smape_pct']:.3f}%")
    print(
        f"Bias:           "
        f"{formal_metrics['bias_pred_minus_true']:+.6f} µg/L"
    )
    print(f"R²:             {formal_metrics['r2']:.6f}")
    print(f"Pearson r:      {formal_metrics['pearson_r']:.6f}")
    print(
        f"P95 abs error:  "
        f"{formal_metrics['p95_abs_error']:.6f} µg/L"
    )
    print(
        f"Max abs error:  "
        f"{formal_metrics['max_abs_error']:.6f} µg/L"
    )

    print("\nDummyRegressor (mediana de TRAIN+VAL):")
    print(f"  MAE:          {dummy_metrics['mae']:.6f} µg/L")
    print(f"  RMSE:         {dummy_metrics['rmse']:.6f} µg/L")

    print("\nCobertura temporal aproximada del TEST:")
    print(f"  Filas reales:       {coverage['actual_rows']}")
    print(
        f"  Filas esperadas:    "
        f"{coverage['expected_rows_approx']}"
    )
    print(
        f"  Cobertura:          "
        f"{coverage['coverage_pct_approx']:.2f}%"
    )
    print(
        f"  Intervalo mediano:  "
        f"{coverage['median_interval_minutes']:.3f} min"
    )
    print(
        f"  Intervalo máximo:   "
        f"{coverage['max_interval_minutes']:.3f} min"
    )
    print(
        f"  Gaps > 7.5 min:     "
        f"{coverage['gaps_gt_7_5_minutes']}"
    )

    print(
        f"\nTHM >= {args.high_thm_threshold:g} µg/L: "
        f"{int(high_mask.sum())} muestras"
    )
    if high_metrics is not None:
        print(
            f"  MAE:          "
            f"{high_metrics['mae']:.6f} µg/L"
        )
        print(
            f"  RMSE:         "
            f"{high_metrics['rmse']:.6f} µg/L"
        )

    print(
        f"\nTHM >= {args.operational_threshold:g} µg/L: "
        f"{int(operational_mask.sum())} muestras"
    )
    if operational_metrics is not None:
        print(
            f"  MAE:          "
            f"{operational_metrics['mae']:.6f} µg/L"
        )
        print(
            f"  RMSE:         "
            f"{operational_metrics['rmse']:.6f} µg/L"
        )

    print("\nComparación contextual con baseline de desarrollo v0.3.2:")
    print(
        f"  MAE desarrollo:     "
        f"{BASELINE_REFERENCE['pooled_mae']:.6f} µg/L"
    )
    print(
        f"  MAE TEST formal:    "
        f"{formal_metrics['mae']:.6f} µg/L"
    )
    print(
        f"  Δ MAE:              "
        f"{formal_metrics['mae'] - BASELINE_REFERENCE['pooled_mae']:+.6f} µg/L"
    )

    validation_curve.to_csv(
        args.output_dir / "validation_iteration_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        args.output_dir / "formal_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.to_csv(
        args.output_dir / "formal_test_daily_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shift.to_csv(
        args.output_dir / "feature_shift_trainval_vs_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {
                "model": "hgb_v0_3_2_protocol",
                **formal_metrics,
            },
            {
                "model": "dummy_median",
                **dummy_metrics,
            },
        ]
    ).to_csv(
        args.output_dir / "formal_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([coverage]).to_csv(
        args.output_dir / "formal_test_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    subset_rows: list[dict[str, Any]] = []

    if high_metrics is not None:
        subset_rows.append(
            {
                "subset": (
                    f"thm_ge_"
                    f"{args.high_thm_threshold:g}"
                ),
                **high_metrics,
            }
        )

    if operational_metrics is not None:
        subset_rows.append(
            {
                "subset": (
                    f"thm_ge_"
                    f"{args.operational_threshold:g}"
                ),
                **operational_metrics,
            }
        )

    if subset_rows:
        pd.DataFrame(subset_rows).to_csv(
            args.output_dir / "formal_test_subset_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    metadata = {
        "version": VERSION,
        "purpose": (
            "Validación formal out-of-time del soft-sensor THM Acosol."
        ),
        "baseline_reference": BASELINE_REFERENCE,
        "features": FEATURES,
        "excluded_features": [
            "temperatura",
            "bromuro",
            "tic",
            "source_file",
            "thm",
        ],
        "split": {
            "train": {
                "start": str(train[TIME_COL].min()),
                "end": str(train[TIME_COL].max()),
                "rows": int(len(train)),
            },
            "validation": {
                "start": str(val[TIME_COL].min()),
                "end": str(val[TIME_COL].max()),
                "rows": int(len(val)),
            },
            "formal_test": {
                "start": str(test[TIME_COL].min()),
                "end": str(test[TIME_COL].max()),
                "rows": int(len(test)),
                "calendar_interval": (
                    "2026-10-01 00:00:00 <= t < "
                    "2026-10-08 00:00:00"
                ),
            },
        },
        "selection": {
            "candidate_max_iter": args.candidate_max_iter,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "min_iter": args.min_iter,
            "selected_iterations": selected_iter,
            "selection_stop_iteration": stop_iter,
            "validation_mae": val_mae,
            "validation_rmse": val_rmse,
        },
        "formal_test_metrics": formal_metrics,
        "dummy_median_metrics": dummy_metrics,
        "coverage": coverage,
        "high_thm_threshold": args.high_thm_threshold,
        "high_thm_count": int(high_mask.sum()),
        "high_thm_metrics": high_metrics,
        "operational_threshold": args.operational_threshold,
        "operational_threshold_count": int(
            operational_mask.sum()
        ),
        "operational_threshold_metrics": operational_metrics,
        "important_note": (
            "El TEST formal no debe utilizarse posteriormente para ajustar "
            "features, hiperparámetros o reglas de selección del baseline."
        ),
    }

    (
        args.output_dir
        / "formal_validation_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_figures(
        predictions=predictions,
        daily=daily,
        shift=shift,
        output_dir=args.output_dir,
    )

    print(
        f"\n[OK] Validación formal completada. "
        f"Outputs: {args.output_dir}"
    )
    print(
        "[IMPORTANTE] A partir de este momento el TEST 01/10-07/10 "
        "queda consumido como TEST formal. No usarlo para reajustar v0.3.2."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
