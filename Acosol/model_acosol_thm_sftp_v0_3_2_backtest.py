from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


VERSION = "v0.3.2"
TIME_COL = "timestamp"
TARGET_COL = "thm"

SENSOR_FEATURES = [
    "cloro",
    "turbidez",
    "conductividad",
    "ph",
    "cot",
]

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
EXCLUDED_FEATURES = ["temperatura", "bromuro", "tic", "source_file", "thm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtesting temporal Acosol v0.3.2 con early stopping manual "
            "sobre validación temporal."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=(
            Path("acosol_sftp_dataset_v0_3")
            / "acosol_sftp_model_training_v0_3.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_outputs_acosol_sftp_v0_3_2_backtest"),
    )
    parser.add_argument("--val-days", type=float, default=1.0)
    parser.add_argument("--test-days", type=float, default=1.0)
    parser.add_argument("--min-train-days", type=float, default=3.0)
    parser.add_argument("--candidate-max-iter", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--min-iter", type=int, default=20)
    parser.add_argument("--max-folds", type=int, default=0)
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
    }


def build_folds(
    df: pd.DataFrame,
    val_days: float,
    test_days: float,
    min_train_days: float,
    max_folds: int,
) -> list[dict[str, Any]]:
    max_ts = df[TIME_COL].max()
    newest: list[dict[str, Any]] = []
    offset = 0

    while True:
        test_end = max_ts - pd.Timedelta(days=offset * test_days)
        test_start = test_end - pd.Timedelta(days=test_days)
        val_start = test_start - pd.Timedelta(days=val_days)

        train = df[df[TIME_COL] <= val_start].copy()
        val = df[
            (df[TIME_COL] > val_start)
            & (df[TIME_COL] <= test_start)
        ].copy()
        test = df[
            (df[TIME_COL] > test_start)
            & (df[TIME_COL] <= test_end)
        ].copy()

        if train.empty or val.empty or test.empty:
            break

        train_days = (
            train[TIME_COL].max() - train[TIME_COL].min()
        ).total_seconds() / 86400.0

        if train_days < min_train_days:
            break

        newest.append({"train": train, "val": val, "test": test})
        offset += 1

    if max_folds > 0:
        newest = newest[:max_folds]

    if not newest:
        raise ValueError("No se puede generar ningún fold.")

    folds = list(reversed(newest))
    for fold_id, fold in enumerate(folds, start=1):
        fold["fold"] = fold_id

    all_test = pd.concat(
        [
            fold["test"][[TIME_COL]].assign(fold=fold["fold"])
            for fold in folds
        ],
        ignore_index=True,
    )
    if all_test[TIME_COL].duplicated().any():
        raise RuntimeError("Los tests de los folds se solapan.")

    return folds


def select_iterations(
    train: pd.DataFrame,
    val: pd.DataFrame,
    candidate_max_iter: int,
    patience: int,
    min_delta: float,
    min_iter: int,
    random_state: int,
) -> tuple[int, float, float, int, pd.DataFrame]:
    model = make_hgb(candidate_max_iter, random_state)
    model.fit(train[FEATURES], train[TARGET_COL])

    y_val = val[TARGET_COL].to_numpy(dtype=float)

    best_iter = 0
    best_mae = float("inf")
    best_rmse = float("inf")
    no_improve = 0
    stop_iter = candidate_max_iter
    rows: list[dict[str, Any]] = []

    for iteration, prediction in enumerate(
        model.staged_predict(val[FEATURES]),
        start=1,
    ):
        y_pred = np.asarray(prediction, dtype=float)
        mae = float(mean_absolute_error(y_val, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_val, y_pred)))

        improved = best_iter == 0 or mae < (best_mae - min_delta)

        if improved:
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
                "significant_improvement": improved,
            }
        )

        if iteration >= min_iter and no_improve >= patience:
            stop_iter = iteration
            break

    return best_iter, best_mae, best_rmse, stop_iter, pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    if args.val_days <= 0 or args.test_days <= 0 or args.min_train_days <= 0:
        raise ValueError("val-days, test-days y min-train-days deben ser > 0.")
    if args.candidate_max_iter <= 0:
        raise ValueError("--candidate-max-iter debe ser > 0.")
    if args.patience <= 0:
        raise ValueError("--patience debe ser > 0.")
    if args.min_delta < 0:
        raise ValueError("--min-delta debe ser >= 0.")
    if args.min_iter <= 0 or args.min_iter > args.candidate_max_iter:
        raise ValueError("--min-iter no válido.")
    if args.max_folds < 0:
        raise ValueError("--max-folds debe ser >= 0.")

    if not args.input_csv.exists():
        raise FileNotFoundError(f"No existe: {args.input_csv}")

    df = pd.read_csv(args.input_csv)

    required = {TIME_COL, TARGET_COL, *SENSOR_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Faltan columnas obligatorias: " + ", ".join(missing)
        )

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    for feature in SENSOR_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")

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

    folds = build_folds(
        df,
        args.val_days,
        args.test_days,
        args.min_train_days,
        args.max_folds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = args.output_dir / "validation_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    result_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []

    print("Backtesting temporal Acosol SFTP v0.3.2")
    print("=======================================")
    print(f"Dataset: {df[TIME_COL].min()} -> {df[TIME_COL].max()}")
    print(f"Filas model-eligible: {len(df)}")
    print(f"Features: {', '.join(FEATURES)}")
    print(f"Excluidas: {', '.join(EXCLUDED_FEATURES)}")
    print(f"Folds generados: {len(folds)}")
    print(
        f"Selección: max_iter={args.candidate_max_iter}, "
        f"patience={args.patience}, min_delta={args.min_delta}, "
        f"min_iter={args.min_iter}"
    )
    print(
        "Protocolo: TRAIN -> seleccionar iteraciones en VAL -> "
        "reentrenar TRAIN+VAL -> evaluar TEST una sola vez"
    )

    for fold in folds:
        fold_id = int(fold["fold"])
        train = fold["train"]
        val = fold["val"]
        test = fold["test"]

        (
            best_iter,
            best_val_mae,
            best_val_rmse,
            stop_iter,
            curve,
        ) = select_iterations(
            train,
            val,
            args.candidate_max_iter,
            args.patience,
            args.min_delta,
            args.min_iter,
            args.random_state,
        )

        curve.insert(0, "fold", fold_id)
        curve.to_csv(
            curves_dir / f"fold_{fold_id:02d}_validation_curve.csv",
            index=False,
            encoding="utf-8-sig",
        )

        train_val = pd.concat([train, val], ignore_index=True)

        final_fold_model = make_hgb(best_iter, args.random_state)
        final_fold_model.fit(
            train_val[FEATURES],
            train_val[TARGET_COL],
        )

        y_test = test[TARGET_COL].to_numpy(dtype=float)
        y_pred = np.asarray(
            final_fold_model.predict(test[FEATURES]),
            dtype=float,
        )
        test_metrics = compute_metrics(y_test, y_pred)

        result_rows.append(
            {
                "fold": fold_id,
                "selected_iterations": best_iter,
                "selection_stop_iteration": stop_iter,
                "val_mae_at_selected_iterations": best_val_mae,
                "val_rmse_at_selected_iterations": best_val_rmse,
                **test_metrics,
            }
        )

        for split_name, part in (
            ("train", train),
            ("val", val),
            ("test", test),
        ):
            fold_rows.append(
                {
                    "fold": fold_id,
                    "split": split_name,
                    "rows": len(part),
                    "min_timestamp": part[TIME_COL].min(),
                    "max_timestamp": part[TIME_COL].max(),
                    "thm_mean": part[TARGET_COL].mean(),
                    "thm_std": part[TARGET_COL].std(),
                    "thm_min": part[TARGET_COL].min(),
                    "thm_median": part[TARGET_COL].median(),
                    "thm_max": part[TARGET_COL].max(),
                }
            )

        pred = pd.DataFrame(
            {
                "fold": fold_id,
                "timestamp": test[TIME_COL].values,
                "y_true": y_test,
                "y_pred": y_pred,
                "residual_pred_minus_true": y_pred - y_test,
                "abs_error": np.abs(y_pred - y_test),
                "selected_iterations": best_iter,
                "selection_stop_iteration": stop_iter,
            }
        )
        if "source_file" in test.columns:
            pred["source_file"] = test["source_file"].values
        prediction_frames.append(pred)

        print(
            f"Fold {fold_id:02d}/{len(folds)} | "
            f"train={len(train)} val={len(val)} test={len(test)} | "
            f"iter={best_iter} stop={stop_iter} | "
            f"val_MAE={best_val_mae:.6f} | "
            f"test_MAE={test_metrics['mae']:.6f}"
        )

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    definitions = pd.DataFrame(fold_rows)

    pooled = compute_metrics(
        predictions["y_true"].to_numpy(dtype=float),
        predictions["y_pred"].to_numpy(dtype=float),
    )

    macro = {
        "folds": len(results),
        "mae_mean": results["mae"].mean(),
        "mae_std": results["mae"].std(ddof=1),
        "mae_median": results["mae"].median(),
        "rmse_mean": results["rmse"].mean(),
        "rmse_median": results["rmse"].median(),
        "selected_iterations_mean": results["selected_iterations"].mean(),
        "selected_iterations_median": results["selected_iterations"].median(),
        "selected_iterations_min": results["selected_iterations"].min(),
        "selected_iterations_max": results["selected_iterations"].max(),
        "selection_stop_iteration_mean": results["selection_stop_iteration"].mean(),
        "selection_stop_iteration_median": results["selection_stop_iteration"].median(),
        "folds_hitting_candidate_ceiling": int(
            (
                results["selection_stop_iteration"]
                >= args.candidate_max_iter
            ).sum()
        ),
    }

    results.to_csv(
        args.output_dir / "metrics_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        args.output_dir / "test_predictions_by_fold.csv",
        index=False,
        encoding="utf-8-sig",
    )
    definitions.to_csv(
        args.output_dir / "fold_definitions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{"model": "hgb_v0_3_2", **pooled}]).to_csv(
        args.output_dir / "pooled_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([macro]).to_csv(
        args.output_dir / "macro_test_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "version": VERSION,
        "input_csv": str(args.input_csv),
        "target": TARGET_COL,
        "features": FEATURES,
        "excluded_features": EXCLUDED_FEATURES,
        "protocol": (
            "For each fold, candidate HGB is fit on TRAIN only. "
            "Temporal VAL selects the iteration count using manual early stopping "
            "with patience/min_delta. A new HGB is refit on TRAIN+VAL using the "
            "selected iteration count. TEST is evaluated exactly once and is never "
            "used for training or model selection."
        ),
        "candidate_max_iter": args.candidate_max_iter,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "min_iter": args.min_iter,
        "val_days": args.val_days,
        "test_days": args.test_days,
        "min_train_days": args.min_train_days,
        "random_state": args.random_state,
    }

    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("RESULTADOS TEST POR FOLD")
    print("=" * 100)
    print(
        results[
            [
                "fold",
                "selected_iterations",
                "selection_stop_iteration",
                "val_mae_at_selected_iterations",
                "mae",
                "rmse",
                "smape_pct",
                "bias_pred_minus_true",
                "r2",
                "max_abs_error",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("RESUMEN")
    print("=" * 100)
    print(f"MAE pooled:               {pooled['mae']:.6f} µg/L")
    print(f"RMSE pooled:              {pooled['rmse']:.6f} µg/L")
    print(f"sMAPE pooled:             {pooled['smape_pct']:.3f}%")
    print(
        f"Bias pooled:              "
        f"{pooled['bias_pred_minus_true']:+.6f} µg/L"
    )
    print(f"R2 pooled:                {pooled['r2']:.6f}")
    print(f"MAE macro mean:           {macro['mae_mean']:.6f} µg/L")
    print(f"MAE macro median:         {macro['mae_median']:.6f} µg/L")
    print(
        f"Iteraciones mediana:      "
        f"{macro['selected_iterations_median']:.1f}"
    )
    print(
        f"Iteraciones rango:        "
        f"{macro['selected_iterations_min']} -> "
        f"{macro['selected_iterations_max']}"
    )
    print(
        f"Stop mediano:             "
        f"{macro['selection_stop_iteration_median']:.1f}"
    )
    print(
        f"Folds que tocan techo:    "
        f"{macro['folds_hitting_candidate_ceiling']}"
    )

    if macro["folds_hitting_candidate_ceiling"] > 0:
        print(
            "\n[REVISAR] Algún fold agotó candidate-max-iter antes de "
            "cumplir patience."
        )
    else:
        print(
            "\n[OK] Todos los folds finalizaron por patience/min_delta "
            "antes del techo candidato."
        )

    print(
        "\n[OK] Backtesting v0.3.2 completado sin usar TEST para selección."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
