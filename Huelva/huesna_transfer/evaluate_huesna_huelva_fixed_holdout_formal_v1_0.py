from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
HUELVA_ROOT = SCRIPT_DIR.parent

TRANSFER_DIR = SCRIPT_DIR / "outputs" / "formal_v1_0"
INPUT_CSV = (
    TRANSFER_DIR
    / "zero_shot_predictions_huesna_v5_8_12.csv"
)

LOCAL_MODEL_DIR = (
    HUELVA_ROOT
    / "model_outputs_huelva_thm_softsensor_v0_4_formal"
)
LOCAL_TEST_PREDICTIONS = (
    LOCAL_MODEL_DIR
    / "final_test_predictions_all.csv"
)

OUTPUT_DIR = TRANSFER_DIR

FORMAL_TEST_FRACTION = 0.20
MIN_TEST = 5
MIN_CALIBRATION = 5

LOCAL_SELECTED_ID = "ridge_baseline_24h_alpha_10"
LOCAL_HGB_COMPARATOR_ID = "hgb_multiscale_means_cfg_3"


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(
        np.abs(y_true) + np.abs(y_pred),
        1e-7,
    )
    return float(
        100.0
        * np.mean(
            2.0
            * np.abs(y_pred - y_true)
            / denom
        )
    )


def pearson_safe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return np.nan

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan

    return float(
        np.corrcoef(y_true, y_pred)[0, 1]
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true

    return {
        "n": int(len(y_true)),
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "bias": float(np.mean(error)),
        "r2": (
            float(r2_score(y_true, y_pred))
            if len(y_true) >= 2
            else np.nan
        ),
        "pearson_r": pearson_safe(
            y_true,
            y_pred,
        ),
        "smape": smape(
            y_true,
            y_pred,
        ),
    }


def fit_bias(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    return float(np.mean(y - x))


def fit_affine(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or np.std(x) < 1e-12:
        return 1.0, fit_bias(x, y)

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )
    return float(slope), float(intercept)


def add_metric_row(
    rows: List[Dict[str, object]],
    label: str,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    rows.append(
        {
            "label": label,
            "method": method,
            **compute_metrics(
                y_true,
                y_pred,
            ),
        }
    )


def resolve_formal_split(
    n_rows: int,
) -> Tuple[int, int]:
    n_test = max(
        MIN_TEST,
        int(
            math.ceil(
                FORMAL_TEST_FRACTION
                * n_rows
            )
        ),
    )
    n_cal = n_rows - n_test

    if n_cal < MIN_CALIBRATION:
        raise ValueError(
            f"Calibracion insuficiente: "
            f"{n_cal} muestras."
        )

    return n_cal, n_test


def load_local_test_predictions() -> pd.DataFrame:
    if not LOCAL_TEST_PREDICTIONS.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        LOCAL_TEST_PREDICTIONS,
        encoding="utf-8-sig",
    )
    df["sample_timestamp"] = pd.to_datetime(
        df["sample_timestamp"],
        errors="coerce",
    )
    return df


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"No existe el fichero zero-shot: {INPUT_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_CSV,
        encoding="utf-8-sig",
    )
    df["sample_timestamp"] = pd.to_datetime(
        df["sample_timestamp"],
        errors="coerce",
    )
    df["thm_real"] = pd.to_numeric(
        df["thm_real"],
        errors="coerce",
    )
    df["huesna_v5_8_12_hgb_pred"] = pd.to_numeric(
        df["huesna_v5_8_12_hgb_pred"],
        errors="coerce",
    )

    # El split formal se define sobre TODAS las etiquetas Huelva,
    # independientemente de si Huesna puede emitir prediccion.
    df = (
        df[
            df["sample_timestamp"].notna()
            & df["thm_real"].notna()
        ]
        .sort_values("sample_timestamp")
        .reset_index(drop=True)
    )

    n_calibration, n_test = resolve_formal_split(
        len(df)
    )

    calibration = df.iloc[:n_calibration].copy()
    test = df.iloc[n_calibration:].copy()

    cal_available = calibration[
        calibration["huesna_v5_8_12_hgb_pred"].notna()
    ].copy()
    test_available = test[
        test["huesna_v5_8_12_hgb_pred"].notna()
    ].copy()

    if len(cal_available) < MIN_CALIBRATION:
        raise ValueError(
            "No hay suficientes predicciones Huesna en desarrollo "
            "para estimar la calibracion."
        )

    y_cal = cal_available["thm_real"].to_numpy(dtype=float)
    x_cal = cal_available[
        "huesna_v5_8_12_hgb_pred"
    ].to_numpy(dtype=float)

    bias_parameter = fit_bias(
        x_cal,
        y_cal,
    )
    affine_slope, affine_intercept = fit_affine(
        x_cal,
        y_cal,
    )

    y_test_huesna = test_available[
        "thm_real"
    ].to_numpy(dtype=float)
    x_test = test_available[
        "huesna_v5_8_12_hgb_pred"
    ].to_numpy(dtype=float)

    zero_pred = x_test.copy()
    bias_pred = x_test + bias_parameter
    affine_pred = (
        affine_slope
        * x_test
        + affine_intercept
    )

    y_test_full = test[
        "thm_real"
    ].to_numpy(dtype=float)

    local_mean_value = float(
        calibration["thm_real"].mean()
    )
    local_median_value = float(
        calibration["thm_real"].median()
    )
    local_mean_full = np.full(
        n_test,
        local_mean_value,
        dtype=float,
    )
    local_median_full = np.full(
        n_test,
        local_median_value,
        dtype=float,
    )

    metric_rows: List[Dict[str, object]] = []

    def append_metric(
        label: str,
        method: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        coverage_n: int,
        coverage_total: int,
        scope: str,
    ) -> None:
        row = {
            "label": label,
            "method": method,
            "evaluation_scope": scope,
            "coverage_n": int(coverage_n),
            "coverage_total": int(coverage_total),
            "coverage_ratio": (
                float(coverage_n / coverage_total)
                if coverage_total
                else np.nan
            ),
            **compute_metrics(y_true, y_pred),
        }
        metric_rows.append(row)

    # Huesna transfer methods: metrics only where a prediction exists.
    append_metric(
        "Huesna v5.8.12",
        "zero_shot",
        y_test_huesna,
        zero_pred,
        len(test_available),
        n_test,
        "available_huesna_test",
    )
    append_metric(
        "Huesna v5.8.12",
        "fixed_bias_calibration",
        y_test_huesna,
        bias_pred,
        len(test_available),
        n_test,
        "available_huesna_test",
    )
    append_metric(
        "Huesna v5.8.12",
        "fixed_affine_calibration",
        y_test_huesna,
        affine_pred,
        len(test_available),
        n_test,
        "available_huesna_test",
    )

    # Labels-only baselines retain the complete formal test.
    append_metric(
        "Huelva labels-only",
        "local_fixed_mean",
        y_test_full,
        local_mean_full,
        n_test,
        n_test,
        "full_formal_test",
    )
    append_metric(
        "Huelva labels-only",
        "local_fixed_median",
        y_test_full,
        local_median_full,
        n_test,
        n_test,
        "full_formal_test",
    )

    local_predictions = load_local_test_predictions()

    prediction_table = test[
        [
            "sample_id",
            "sample_timestamp",
            "sample_point",
            "thm_unit",
            "thm_real",
            "huesna_v5_8_12_hgb_pred",
        ]
    ].copy()

    prediction_table[
        "huesna_v5_8_12_zero_shot_pred"
    ] = prediction_table[
        "huesna_v5_8_12_hgb_pred"
    ]

    available_mask = prediction_table[
        "huesna_v5_8_12_hgb_pred"
    ].notna()

    prediction_table[
        "huesna_v5_8_12_fixed_bias_pred"
    ] = np.nan
    prediction_table[
        "huesna_v5_8_12_fixed_affine_pred"
    ] = np.nan

    prediction_table.loc[
        available_mask,
        "huesna_v5_8_12_fixed_bias_pred",
    ] = (
        prediction_table.loc[
            available_mask,
            "huesna_v5_8_12_hgb_pred",
        ].to_numpy(dtype=float)
        + bias_parameter
    )
    prediction_table.loc[
        available_mask,
        "huesna_v5_8_12_fixed_affine_pred",
    ] = (
        affine_slope
        * prediction_table.loc[
            available_mask,
            "huesna_v5_8_12_hgb_pred",
        ].to_numpy(dtype=float)
        + affine_intercept
    )

    prediction_table[
        "local_fixed_mean_pred"
    ] = local_mean_full
    prediction_table[
        "local_fixed_median_pred"
    ] = local_median_full

    expected_test_times = set(
        pd.to_datetime(test["sample_timestamp"])
    )

    local_candidate_predictions: Dict[str, np.ndarray] = {}

    if not local_predictions.empty:
        for candidate_id, method_name in [
            (
                LOCAL_SELECTED_ID,
                "local_selected_ridge_24h",
            ),
            (
                LOCAL_HGB_COMPARATOR_ID,
                "local_hgb_multiscale_comparator",
            ),
        ]:
            candidate = local_predictions[
                local_predictions[
                    "candidate_id"
                ].astype(str)
                == candidate_id
            ].copy()

            candidate = candidate[
                candidate[
                    "sample_timestamp"
                ].isin(expected_test_times)
            ].sort_values(
                "sample_timestamp"
            )

            merged = test[
                [
                    "sample_timestamp",
                    "thm_real",
                ]
            ].merge(
                candidate[
                    [
                        "sample_timestamp",
                        "y_pred",
                    ]
                ],
                on="sample_timestamp",
                how="left",
            ).sort_values("sample_timestamp")

            if merged["y_pred"].notna().sum() == n_test:
                pred_full = merged[
                    "y_pred"
                ].to_numpy(dtype=float)
                local_candidate_predictions[
                    method_name
                ] = pred_full
                prediction_table[
                    method_name
                ] = pred_full

                append_metric(
                    "Huelva local formal",
                    method_name,
                    y_test_full,
                    pred_full,
                    n_test,
                    n_test,
                    "full_formal_test",
                )

    metrics = pd.DataFrame(metric_rows).sort_values(
        [
            "evaluation_scope",
            "mae",
            "rmse",
        ],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    # Fair comparison on the exact common subset where Huesna has a prediction.
    common_rows: List[Dict[str, object]] = []
    common_times = set(
        pd.to_datetime(
            test_available["sample_timestamp"]
        )
    )
    common_mask = test[
        "sample_timestamp"
    ].isin(common_times).to_numpy()
    y_common = y_test_full[common_mask]

    def add_common(
        label: str,
        method: str,
        pred: np.ndarray,
    ) -> None:
        common_rows.append(
            {
                "label": label,
                "method": method,
                "n": int(len(y_common)),
                **compute_metrics(
                    y_common,
                    np.asarray(pred, dtype=float),
                ),
            }
        )

    add_common(
        "Huesna v5.8.12",
        "zero_shot",
        zero_pred,
    )
    add_common(
        "Huesna v5.8.12",
        "fixed_bias_calibration",
        bias_pred,
    )
    add_common(
        "Huesna v5.8.12",
        "fixed_affine_calibration",
        affine_pred,
    )
    add_common(
        "Huelva labels-only",
        "local_fixed_mean",
        local_mean_full[common_mask],
    )
    add_common(
        "Huelva labels-only",
        "local_fixed_median",
        local_median_full[common_mask],
    )

    for method_name, pred_full in local_candidate_predictions.items():
        add_common(
            "Huelva local formal",
            method_name,
            pred_full[common_mask],
        )

    common_metrics = pd.DataFrame(
        common_rows
    ).sort_values(
        ["mae", "rmse"],
        ascending=[True, True],
    ).reset_index(drop=True)

    params = pd.DataFrame(
        [
            {
                "source_model": "Huesna v5.8.12",
                "n_total_huelva_labels": len(df),
                "n_calibration_labels": n_calibration,
                "n_calibration_huesna_available": len(cal_available),
                "n_test_labels": n_test,
                "n_test_huesna_available": len(test_available),
                "test_huesna_coverage_ratio": (
                    len(test_available) / n_test
                ),
                "formal_test_fraction": FORMAL_TEST_FRACTION,
                "calibration_start": (
                    calibration[
                        "sample_timestamp"
                    ].min()
                ),
                "calibration_end": (
                    calibration[
                        "sample_timestamp"
                    ].max()
                ),
                "test_start": (
                    test[
                        "sample_timestamp"
                    ].min()
                ),
                "test_end": (
                    test[
                        "sample_timestamp"
                    ].max()
                ),
                "bias_parameter": bias_parameter,
                "affine_slope": affine_slope,
                "affine_intercept": affine_intercept,
            }
        ]
    )

    prediction_table.to_csv(
        OUTPUT_DIR
        / "formal_fixed_holdout_predictions_27_7.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics.to_csv(
        OUTPUT_DIR
        / "formal_fixed_holdout_metrics_27_7.csv",
        index=False,
        encoding="utf-8-sig",
    )
    common_metrics.to_csv(
        OUTPUT_DIR
        / "formal_common_subset_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    params.to_csv(
        OUTPUT_DIR
        / "formal_fixed_holdout_calibration_parameters.csv",
        index=False,
        encoding="utf-8-sig",
    )

    missing_test = test[
        test[
            "huesna_v5_8_12_hgb_pred"
        ].isna()
    ].copy()
    missing_test.to_csv(
        OUTPUT_DIR
        / "formal_huesna_missing_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(
        test["sample_timestamp"],
        y_test_full,
        marker="o",
        label="THM real Huelva",
    )
    ax.plot(
        prediction_table["sample_timestamp"],
        prediction_table[
            "huesna_v5_8_12_zero_shot_pred"
        ],
        marker="o",
        label="Huesna v5.8.12 zero-shot",
    )
    ax.plot(
        prediction_table["sample_timestamp"],
        prediction_table[
            "huesna_v5_8_12_fixed_bias_pred"
        ],
        marker="o",
        label="Huesna v5.8.12 + bias fijo",
    )

    if (
        "local_selected_ridge_24h"
        in prediction_table.columns
    ):
        ax.plot(
            prediction_table["sample_timestamp"],
            prediction_table[
                "local_selected_ridge_24h"
            ],
            marker="o",
            label="Huelva Ridge 24 h",
        )

    ax.set_xlabel("Fecha")
    ax.set_ylabel("THM (ug/L)")
    ax.set_title(
        "Holdout formal 27/7: "
        "Huesna v5.8.12 -> Huelva"
    )
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR
        / "formal_fixed_holdout_27_7_predictions.png",
        dpi=170,
    )
    plt.close(fig)

    report_lines = [
        "# Holdout formal 27/7: Huesna -> Huelva",
        "",
        "## Protocolo",
        "",
        f"- Muestras Huelva totales: **{len(df)}**.",
        (
            f"- Desarrollo/calibracion: **{n_calibration}** "
            f"etiquetas; Huesna disponible en "
            f"**{len(cal_available)}/{n_calibration}**."
        ),
        (
            f"- Test final congelado: **{n_test}** "
            f"etiquetas; Huesna disponible en "
            f"**{len(test_available)}/{n_test}**."
        ),
        (
            "- El split formal 27/7 no cambia por la cobertura "
            "del modelo transferido."
        ),
        (
            "- Ninguna etiqueta del test se utiliza para ajustar "
            "bias o transformacion affine."
        ),
        "",
        "## Metricas con su cobertura real",
        "",
        "```text",
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "```",
        "",
        "## Comparacion justa sobre subconjunto comun",
        "",
        "```text",
        common_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "```",
        "",
        "## Parametros de calibracion",
        "",
        "```text",
        params.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        "```",
        "",
        "## Interpretacion",
        "",
        (
            "La ausencia de prediccion Huesna en una muestra del "
            "test se contabiliza como limitacion de cobertura y no "
            "se imputa ni se elimina del test formal Huelva. Las "
            "metricas del modelo local permanecen 7/7; la comparacion "
            "directa entre modelos se realiza adicionalmente sobre "
            "el subconjunto comun."
        ),
    ]

    (
        OUTPUT_DIR
        / "formal_fixed_holdout_report_v0_2.md"
    ).write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("============================================")
    print("HOLDOUT FORMAL 27/7: HUESNA -> HUELVA")
    print("============================================")
    print(
        f"Desarrollo/calibracion: {n_calibration} etiquetas; "
        f"Huesna={len(cal_available)}/{n_calibration}"
    )
    print(
        f"Test congelado:         {n_test} etiquetas; "
        f"Huesna={len(test_available)}/{n_test}"
    )
    print("")
    print("Metricas con cobertura real:")
    print(
        metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("")
    print("Comparacion sobre subconjunto comun:")
    print(
        common_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("")
    if not missing_test.empty:
        print("Muestras test sin prediccion Huesna:")
        print(
            missing_test[
                [
                    "sample_id",
                    "sample_timestamp",
                    "thm_real",
                ]
            ].to_string(index=False)
        )
        print("")
    print("Parametros:")
    print(
        params[
            [
                "bias_parameter",
                "affine_slope",
                "affine_intercept",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("")
    print(f"Salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
