# Referencia estable actual - Soft-sensor THM Acosol

**Estado:** método de investigación congelado y **primera evaluación formal out-of-time completada** (octubre de 2026).
**Fecha de documentación:** 26/10/2026. **Último dato utilizado:** 08/10/2026 17:15:24.
**Workflow operativo:** [`workflow.md`](workflow.md).

## Scripts y artefactos de referencia

- Datos: `download_acosol_sftp_v0_1.py` → `build_acosol_training_dataset_v0_1.py` → `check_acosol_supervised_dataset_v0_1.py` → `build_acosol_model_ready_dataset_v0_3.py`.
- Dataset elegible actual: `acosol_sftp_dataset_v0_3/acosol_sftp_model_training_v0_3.csv`.
- Script del **protocolo de desarrollo congelado**: `model_acosol_thm_sftp_v0_3_2_backtest.py`.
- Outputs del desarrollo: `model_outputs_acosol_sftp_v0_3_2_backtest/`.
- Script del **TEST formal independiente**: `validate_acosol_thm_formal_v0_5.py`.
- Outputs del TEST: `model_outputs_acosol_thm_formal_validation_v0_5/`.
- Informe y figuras documentales: `Validation/`.
- Auditorías vigentes: `audits/audit_acosol_thm_extremes_v0_1.py`, `audits/audit_acosol_bromuro_v0_1.py`, `audits/audit_acosol_temperatura_history_v0_1.py`.
- Experimento de históricos (no seleccionado): `experiments/model_acosol_thm_sftp_v0_4_history_backtest.py`; outputs archivados en `archive/outputs_legacy/`.

## Configuración de referencia

- Objetivo: **THM instantáneo**, sin THM como entrada ni histórico de THM.
- Entradas del HGB: **cloro, turbidez, conductividad, pH, COT** y 7 variables de calendario (**12 features**).
- **Excluidos:** bromuro (medidor con incidencia), temperatura (lectura con incidencia), TIC (confirmación de planta de que no se necesita).
- Familia/modelo: `HistGradientBoostingRegressor`, parámetros `loss=squared_error`, `learning_rate=0.04`, `l2_regularization=0.05`, `min_samples_leaf=20`, `early_stopping=False`, `random_state=42`.
- Iteraciones seleccionadas por **VAL cronológica** con `candidate_max_iter=4000`, `patience=100`, `min_delta=0.0001`, `min_iter=20`; reentrenamiento sobre TRAIN+VAL y evaluación del TEST una sola vez.
- **El procedimiento está congelado; no hay un `.joblib` de despliegue definitivo.**

## Datos y calidad - Corte 08/10/2026

- CSV productivos: **16.139**; serie completa: **16.126** timestamps.
- THM continuo desde **03/09/2026**; filas supervisadas: **10.130**.
- Model-ready: **10.118** filas; excluidas **12** por incidencia confirmada de THM ~131 µg/L del **12/09, 12:15:15–13:10:15**.
- El tramo inmediatamente posterior **~12,75 µg/L es válido y se conserva**. No se aplica ninguna regla genérica `THM > 100` fuera de esa incidencia.
- Señal de bromuro no fiable; medida de temperatura prácticamente fija en ~19,7 °C durante el periodo con THM; TIC fuera por decisión de planta.
- Auditoría de 9 hojas Excel históricas: **ninguna contiene temperatura**; no existe periodo THM-temperatura dinámico suficiente para validar la variante «sin bromuro, con temperatura».
- Variante de planta **sí evaluable**: «sin bromuro ni temperatura» (además sin TIC), la usada por v0.3.2.

## Desarrollo: baseline HGB v0.3.2

- Datos de desarrollo hasta **28/09/2026 17:25:19**: **7.273 filas elegibles**.
- `walk-forward` cronológico, **21 folds**: entrenamiento expansivo, VAL 1 día, TEST 1 día en cada fold.
- MAE combinado: **0,430363 µg/L**; RMSE: **0,918989 µg/L**; sMAPE: **11,027 %**; bias: **+0,070932 µg/L**.
- Experimento `v0.4` con histórico de entradas: MAE 0,418111, pero RMSE 0,969973 y peor rendimiento en THM >= 10 µg/L; **no reemplaza a v0.3.2**.

## Evaluación formal aislada: v0.5 (TEST 01–07/10/2026)

| Conjunto | Periodo | Muestras |
|---|---|---:|
| TRAIN | 03/09 – 29/09/2026 | 7.606 |
| VAL | 30/09/2026 | 288 |
| TEST formal | 01/10 – 07/10/2026, 7 días completos | **2.016** |

- La VAL eligió **128 iteraciones**; se volvió a entrenar con TRAIN+VAL sin consultar TEST.
- TEST con cobertura **100 %** y 0 gaps > 7,5 min.

| Métrica en TEST | HGB | Dummy mediana |
|---|---:|---:|
| MAE (µg/L) | 0,501528 | **0,370706** |
| RMSE (µg/L) | 0,570800 | **0,414937** |
| Bias (µg/L) | −0,482409 | −0,312351 |
| sMAPE | 15,038 % | 10,845 % |

**Resultado:** el Dummy mediana **supera al HGB en esta semana de THM bajo y estable** (rango real ~2,82–4,25 µg/L). El HGB tiene un error absoluto medio ~0,50 µg/L, pero subestima de forma sistemática y **no demuestra ventaja sobre la referencia trivial**. No hay muestras `THM >= 10 µg/L`, por lo que el TEST no evalúa episodios elevados.

**No utilizar el TEST 01–07/10 para ajustar el modelo:** es un periodo formal ya consultado. Un nuevo modelo requiere desarrollo separado y otro periodo final posterior no visto.

## Restricciones y siguientes pasos

1. Conservar el protocolo v0.3.2 y las evidencias v0.5.
2. Actualizar exclusivamente etiquetas/metadatos de la incidencia confirmada del 12/09 en el builder v0.3, **sin cambiar las filas descartadas**.
3. Investigar los extremos de algunas variables de entrada y las variaciones temporales, sin retocar el test ya consumido.
4. Tras solucionar bromuro y temperatura, acumular datos validados. Comparar variantes con/sin temperatura sobre **el mismo periodo** y solo si existe un solapamiento suficiente.
5. Reservar datos nuevos para una futura evaluación formal; si se mejora el modelo, no reusar 01–07/10 para seleccionarlo.
6. Mantener `config_private.py`, `sftp_raw/`, datasets, outputs y `archive/` fuera de Git.
