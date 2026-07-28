# Versión estable actual — Soft-sensor THM Huesna

- Script de entrenamiento: `model_v5_8_8_incident_intervals.py`
- Modelo estable: `model_outputs_thm_softsensor_v5_8_8_incidents_20260723_125701`
- Diagnóstico: `outputs_softsensor_drift_analysis_v0_2_incidents_20260723_125701`
- Script de diagnóstico: `analyze_huesna_softsensor_drift_v0_2.py`
- Incidencias documentadas: `incident_intervals_huesna.csv`

## Configuración

- Política de incidencias: `exclude`
- Sin límite global superior de THM.
- Avería de red aductora del 30/06/2026 excluida.
- Buffer posterior aplicado: 240 minutos.
- Modelo seleccionado: HGB global, variante `no_calendar`.

## Métricas de test

- MAE: 1,381 µg/L
- RMSE: 1,685 µg/L
- Bias: +0,241 µg/L
- Ventana test: 16/07/2026 12:35 → 23/07/2026 12:35

Estado: versión estable actual, pendiente de futura validación temporal múltiple.
