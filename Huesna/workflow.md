# Flujo de trabajo — Sensor virtual THM Huesna

## 1. Descargar y verificar datos

**Script:** `sftp.py`

Descarga los CSV nuevos desde el SFTP a `csv_descargados` sin volver a descargar los archivos que ya existen con el mismo tamaño.

```powershell
python .\sftp.py
```

Antes de una validación formal, comprobar que el último CSV corresponde a la fecha esperada, que no hay fallos de descarga y que contiene filas válidas.

## 2. Definir un identificador único

```powershell
$runId = Get-Date -Format "yyyyMMdd_HHmmss"
```

Nunca reutilizar un directorio de una validación congelada.

## 3. Incidencias y reglas de calidad

Ficheros activos:

- `incident_intervals_huesna.csv`: incidencias globales confirmadas que invalidan el intervalo completo para entrenamiento/evaluación.
- `feature_incident_intervals_huesna.csv`: incidencias confirmadas que afectan únicamente a una variable concreta.

Incidencias actualmente documentadas:

- 30/06/2026: avería de red aductora, excluida globalmente y con buffer posterior de 240 minutos.
- 29/08/2026 03:00–18:00: anomalía UVAS confirmada por planta, excluida únicamente de UVAS antes de generar lags y rollings.

No aplicar límites globales superiores de THM sin confirmación técnica. Planta ha señalado un posible problema histórico de lectura en el rango 30–35 µg/L, aproximadamente entre el 12/06/2026 y el 14/08/2026. La exclusión indiscriminada de dichas lecturas se evaluó experimentalmente en octubre de 2026 y no mejoró el rendimiento del modelo. Sigue pendiente determinar qué mediciones concretas fueron incorrectas.

## 4. Auditoría histórica de rangos THM

**Script:** `audits/audit_huesna_thm_ranges_v0_1.py`

Se utiliza cuando es necesario localizar periodos históricos de un rango de THM concreto. No modifica los datos.

```powershell
$thmAuditOut = ".\outputs_thm_range_audit_v0_1_$runId"

python .\audits\audit_huesna_thm_ranges_v0_1.py `
  --csv-dir ".\csv_descargados" `
  --output-dir $thmAuditOut `
  --range-low 30 `
  --range-high 35
```

## 5. Entrenar el soft-sensor

**Script estable:** `model_v5_8_12_incident_intervals.py`

Configuración recomendada:

```powershell
$env:LOKY_MAX_CPU_COUNT = [Environment]::ProcessorCount

$modelOut = ".\model_outputs_thm_softsensor_v5_8_12_incidents_$runId"

python .\model_v5_8_12_incident_intervals.py `
  --output-dir $modelOut `
  --incident-intervals-csv ".\incident_intervals_huesna.csv" `
  --feature-incident-intervals-csv ".\feature_incident_intervals_huesna.csv" `
  --incident-policy exclude `
  --calendar-selection-mode multi `
  --calendar-multival-folds 4 `
  --calendar-multival-window-days 7 `
  --calendar-multival-min-folds 3 `
  --calendar-prefer-all-tolerance 0.05 `
  --log-level verbose
```

La selección del calendario utiliza cuatro ventanas históricas de validación y mantiene el test final completamente reservado.

## 6. Diagnóstico temporal y de deriva

**Script:** `analyze_huesna_softsensor_drift_v0_2.py`

```powershell
$driftOut = ".\outputs_softsensor_drift_analysis_v0_2_incidents_$runId"

python .\analyze_huesna_softsensor_drift_v0_2.py `
  --model-dir $modelOut `
  --output-dir $driftOut
```

Revisar especialmente:

- `variant_metrics_overall.csv`
- `daily_prediction_summary.csv`
- `feature_drift_summary.csv`
- `01_daily_real_vs_predictions.png`
- `02_daily_bias_by_variant.png`
- `05_feature_drift_vs_validation.png`

## 7. Artefactos principales del entrenamiento

Revisar como mínimo:

- `calendar_multival_summary.csv`
- `calendar_multival_folds.csv`
- `test_predictions.csv`
- `split_target_distribution.csv`
- `data_quality_summary.csv`
- `data_quality_target_anomalies.csv`
- `data_quality_feature_incident_values.csv`
- `data_quality_x_sentinel_rows.csv`
- `feature_importance_aggregated.csv`
- `metadata.json`
- `inference_contract.json`

## 8. Referencia formal actual

Consultar `CURRENT_STABLE.md` antes de cualquier nueva prueba.

La referencia congelada actual es la validación del 21/09/2026:

- modelo: `model_outputs_thm_softsensor_v5_8_12_incidents_20260921_135541`
- diagnóstico: `outputs_softsensor_drift_analysis_v0_2_incidents_20260921_135541`
- variante: `all_calendar`
- test reservado: 14/09/2026 13:35 → 21/09/2026 13:35
- MAE: 1,996 µg/L
- RMSE: 2,393 µg/L
- bias: +1,832 µg/L

La parte hidráulica/EPANET se mantiene en `hydraulic_propagation_reference` y queda fuera del entrenamiento habitual del soft-sensor.

## 9. Experimentos y comprobaciones posteriores

### 9.1. Experimento THM 30–35 µg/L — 07/10/2026

Se evaluó una versión experimental `v5.8.13` para estudiar el posible problema de lectura comunicado por planta.

- **Modificación:** exclusión únicamente de las etiquetas THM comprendidas entre 30 y 35 µg/L, ambos incluidos, desde el 12/06/2026 hasta el 14/08/2026.
- **Variables de entrada:** se conservaron todas las medidas multiparamétricas, incluso en los instantes cuyos valores de THM fueron excluidos.
- **Comparación:** mismo histórico, mismas ventanas temporales y mismo test reservado que la v5.8.12.
- **Efecto:** reducción de 228.931 a 189.225 muestras de entrenamiento (39.706 menos).
- **Resultado en test:** MAE de 1,996 a 2,239 µg/L (+12,2 %) y RMSE de 2,393 a 2,581 µg/L (+7,9 %).

**Decisión:** experimento no promovido. Se mantiene la v5.8.12 como referencia estable. La eliminación indiscriminada de todo el rango no resulta beneficiosa; queda pendiente una identificación más precisa de las lecturas afectadas por parte de planta.

Scripts y resultados conservados localmente en:

`legacy_thm_audits/20261007_v5_8_13_target_range_experiment/`

### 9.2. Auditoría de Huerto Reyes — 21/10/2026

Se verificaron las métricas recibidas por SFTP desde el 06/10/2026 para el sensor `Multiparamétrica Huerto Reyes`.

- **Variables disponibles:** PH, UVAS, Conductividad y CLORO.
- **Variables no recibidas:** turbidez y temperatura.
- **THM en Huerto Reyes:** no detectado en los CSV analizados.
- **Continuidad:** cadencia mediana aproximada de 36 segundos y gap máximo observado de unos 73 segundos.
- **Compatibilidad:** disponibles 4 de las 6 variables de entrada requeridas por el modelo v5.8.12.

**Decisión:** no modificar el modelo estable. Para aplicar directamente la v5.8.12 faltan dos variables de entrada. Para validar las predicciones en Huerto Reyes se necesitan también medidas reales de THM tomadas en ese mismo punto.

Pendiente de confirmar con planta la disponibilidad de turbidez y temperatura mediante otros sensores.

Script y resultados conservados localmente en:

`legacy_thm_audits/20261021_huerto_reyes_input_audit/`

### 9.3. Política de versiones

La versión **v5.8.12 continúa siendo la referencia formal estable del proyecto**.

- Los experimentos posteriores deben utilizar directorios de salida independientes.
- No sobrescribir el entrenamiento ni los artefactos congelados de septiembre.
- Comparar cualquier nuevo modelo con el mismo test de referencia cuando corresponda.
- Mantener los experimentos exploratorios archivados y documentar sus conclusiones.

La copia de los artefactos del modelo congelado se conserva localmente en:

`frozen_models/20260921_v5_8_12/`

Esta carpeta está excluida de Git y debe conservarse también mediante una copia de seguridad externa.

Consultar `CURRENT_STABLE.md` para identificar la versión vigente y sus métricas oficiales.
