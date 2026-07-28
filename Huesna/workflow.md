# Flujo de trabajo — Sensor virtual THM Huesna

Este flujo cubre únicamente el trabajo de la UMA sobre el sensor virtual de THM. La propagación hidráulica, EPANET y las alertas quedan fuera de este flujo.

## 1. Configurar el acceso SFTP

El script `sftp.py` obtiene la configuración desde `config_private.py`, que no se versiona.

Crear el fichero privado a partir de la plantilla:

```powershell
Copy-Item .\config_private.example.py .\config_private.py
```

Editar `config_private.py` y completar la contraseña real.

## 2. Descargar los CSV nuevos

```powershell
python .\sftp.py
```

Los CSV se guardan en `csv_descargados`. Los ficheros locales con el mismo nombre y tamaño que el fichero remoto se omiten. Antes de entrenar, la descarga debe terminar con `Fallidos: 0`.

## 3. Definir un identificador único

```powershell
$runId = Get-Date -Format "yyyyMMdd_HHmmss"
```

## 4. Entrenar el sensor virtual estable

**Script de referencia:** `model_v5_8_8_incident_intervals.py`

La versión estable:

- utiliza únicamente variables de la multiparamétrica como entrada;
- no utiliza THM actual ni histórico como feature;
- genera variables actuales, lags y estadísticas móviles;
- compara las variantes `all_calendar`, `no_doy` y `no_calendar`;
- mantiene HGB como modelo operativo y usa Ridge y Dummy como baselines;
- excluye incidencias confirmadas mediante `incident_intervals_huesna.csv`;
- no aplica un límite global superior de THM.

```powershell
$modelOut = ".\model_outputs_thm_softsensor_v5_8_8_incidents_$runId"

python .\model_v5_8_8_incident_intervals.py `
  --output-dir $modelOut `
  --incident-intervals-csv ".\incident_intervals_huesna.csv" `
  --incident-policy exclude `
  --log-level compact
```

No deben añadirse límites globales como `--max-valid-target 80` o `--max-valid-target 100`.

## 5. Ejecutar el diagnóstico temporal

**Script:** `analyze_huesna_softsensor_drift_v0_2.py`

```powershell
$driftOut = ".\outputs_softsensor_drift_analysis_v0_2_incidents_$runId"

python .\analyze_huesna_softsensor_drift_v0_2.py `
  --model-dir $modelOut `
  --output-dir $driftOut
```

## 6. Revisar los resultados

Artefactos principales del entrenamiento:

- `final_selected_model.pkl`
- `instant_softsensor_bundle.pkl`
- `test_predictions.csv`
- `model_selection_results.csv`
- `scenario_comparison.csv`
- `split_target_distribution.csv`
- `data_quality_target_anomalies.csv`
- `incident_intervals_applied.csv`
- `metadata.json`
- `training_verbose.log`

Artefactos principales del diagnóstico:

- `drift_analysis_report.md`
- `variant_metrics_overall.csv`
- `daily_prediction_summary.csv`
- `feature_daily_summary.csv`
- `feature_drift_summary.csv`

## 7. Incidencia documentada

Planta confirmó una avería en la red aductora el 30/06/2026 que provocó alteraciones fuera de lo normal en todos los parámetros de red. El intervalo se declara en `incident_intervals_huesna.csv` y se amplía con un buffer posterior de 240 minutos.

No se usa un umbral global de 80 o 100 µg/L, ya que podría eliminar futuros valores elevados que fueran reales.

## 8. Scripts activos

- `sftp.py`: descarga incremental mediante configuración privada externa.
- `model_v5_8_8_incident_intervals.py`: entrenamiento estable de referencia.
- `analyze_huesna_softsensor_drift_v0_2.py`: diagnóstico temporal.
- `incident_intervals_huesna.csv`: incidencias confirmadas excluidas del entrenamiento.
- `config_private.example.py`: plantilla pública de configuración SFTP.

## 9. Limitaciones conocidas

Esta versión es la referencia estable actualmente reproducida. Está pendiente:

- validación temporal múltiple sobre varias semanas;
- revisión del tratamiento de las features históricas cuando existen huecos temporales entre bloques de datos.
