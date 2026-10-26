# Versión estable actual - Soft-sensor THM Huesna

- Estado: primera validación formal congelada (septiembre de 2026).
- Fecha de corte de datos: 21/09/2026 13:35.
- Script de entrenamiento: `model_v5_8_12_incident_intervals.py`
- Modelo estable: `model_outputs_thm_softsensor_v5_8_12_incidents_20260921_135541`
- Script de diagnóstico: `analyze_huesna_softsensor_drift_v0_2.py`
- Diagnóstico asociado: `outputs_softsensor_drift_analysis_v0_2_incidents_20260921_135541`
- Auditoría histórica THM 30–35 µg/L: `outputs_thm_range_audit_v0_1_20260921_134903`
- Incidencias globales documentadas: `incident_intervals_huesna.csv`
- Incidencias por feature documentadas: `feature_incident_intervals_huesna.csv`

## Configuración estable

- Objetivo: estimación instantánea de `Trihalometanos` sin usar THM como entrada ni histórico de THM.
- Variables base: pH, UVAS, conductividad, turbidez, cloro y temperatura.
- Histórico multiparamétrico máximo: 240 minutos.
- Modelo operativo: HistGradientBoostingRegressor (HGB global).
- Selección de calendario: validación temporal múltiple, 4 folds consecutivos de 7 días con histórico expansivo.
- Variante seleccionada: `all_calendar`.
- Política de incidencias globales: `exclude`.
- Sin límite global superior de THM.

## Incidencias y calidad de datos

- Avería de red aductora del 30/06/2026: excluida como incidencia global, con buffer posterior de 240 minutos.
- UVAS del 29/08/2026, 03:00–18:00: incidencia confirmada por planta y aplicada únicamente a la feature UVAS antes de construir lags/rollings. Se invalidaron 901 lecturas de UVAS.
- Glitches extremos THM: 2 lecturas de 1999,8 µg/L eliminadas mediante detector causal de extremos.
- THM inválidos <= 0,1 µg/L: 383 lecturas, mayoritariamente asociadas a patrones técnicos/centinela.
- Filas centinela detectadas en X: 472.
- Posible problema histórico de lectura THM en el rango 30–35 µg/L: planta señaló como período potencialmente afectado el 12/06/2026–14/08/2026. Sigue pendiente identificar qué lecturas concretas fueron incorrectas; no se considera inválido todo el rango.
- Experimento v5.8.13 (07/10/2026): se probó la exclusión de todas las etiquetas THM entre 30 y 35 µg/L durante ese período, conservando las variables multiparamétricas. El resultado empeoró el test reservado (MAE: 1,996 → 2,239 µg/L; RMSE: 2,393 → 2,581 µg/L). Experimento descartado; se mantiene el tratamiento de la v5.8.12.

## Split de la validación formal

- Train: 24/03/2026 13:35 → 07/09/2026 13:34 - 228.931 filas.
- Validación: 07/09/2026 13:35 → 14/09/2026 13:34 - 9.228 filas.
- Test reservado: 14/09/2026 13:35 → 21/09/2026 13:35 - 9.177 filas utilizables.
- Cobertura temporal aproximada del test: 91,0 % de los minutos de la ventana nominal.

## Selección multitemporal del calendario

| Variante | RMSE media | RMSE mediana | Peor RMSE |
| --- | ---: | ---: | ---: |
| `all_calendar` | 2,811 | 2,381 | 4,694 |
| `no_doy` | 4,501 | 2,487 | 11,174 |
| `no_calendar` | 4,753 | 2,781 | 11,400 |

`all_calendar` es la mejor variante por RMSE media multitemporal; no se selecciona por la tolerancia del 5 % ni usando el test.

## Métricas del test reservado

- MAE: 1,996 µg/L.
- RMSE: 2,393 µg/L.
- Bias: +1,832 µg/L.
- SMAPE: 11,246 %.
- R²: -0,167.
- Dummy median RMSE: 3,000 µg/L.

El mayor error de la semana se concentra tras la caída de régimen observada el 17/09. El modelo sobreestima temporalmente el THM y reduce progresivamente ese sesgo durante los días posteriores.

## Estado

Esta ejecución queda congelada como referencia formal de septiembre de 2026. No modificar ni sobrescribir los directorios de modelo, diagnóstico o auditoría indicados arriba. Cualquier cambio posterior debe ejecutarse con un nuevo `runId` y compararse contra esta referencia.
