# Flujo de trabajo - Soft-sensor THM Acosol

**Documentación revisada:** 26/10/2026.
**Último corte de datos evaluado:** 08/10/2026 17:15:00.
**Referencia vigente:** consultar [`CURRENT_STABLE.md`](CURRENT_STABLE.md) antes de continuar el modelado o iniciar una nueva validación.

## 1. Objetivo y alcance

El objetivo es estimar la concentración instantánea de trihalometanos (THM, µg/L) a la salida de la ETAP de Acosol mediante un **soft-sensor** supervisado con los datos continuos enviados por SFTP.

- El modelo estima **THM(t)** a partir de otras señales observadas en ese instante y variables de calendario: **no realiza predicción futura**.
- THM es el **objetivo (Y)**, nunca una entrada; tampoco se utiliza histórico de THM como feature.
- La selección del método y su validación deben respetar el orden cronológico. No se debe ajustar el modelo mirando los resultados de un test formal ya consultado.
- En esta fase se trabaja en validación y caracterización, **no existe un artefacto final listo para despliegue**.
- La clasificación de prealertas/alertas, la propagación por la red y la integración hidráulica son etapas posteriores, no incluidas en este pipeline de estimación de THM en ETAP.

## 2. Estructura de trabajo actual

Los scripts del **pipeline activo** y los dos scripts de modelado se conservan en la raíz. Las auditorías especializadas, experimentos e históricos tienen carpetas propias.

```text
Acosol/
├── workflow.md
├── CURRENT_STABLE.md
├── config_private.py                              # privado; ignorado por Git
├── config_private.example.py
├── download_acosol_sftp_v0_1.py                  # SFTP -> raw
├── build_acosol_training_dataset_v0_1.py        # raw -> timeseries + supervisado
├── check_acosol_supervised_dataset_v0_1.py      # control de calidad
├── build_acosol_model_ready_dataset_v0_3.py     # exclusiones confirmadas
├── model_acosol_thm_sftp_v0_3_2_backtest.py      # desarrollo del método
├── validate_acosol_thm_formal_v0_5.py           # evaluación formal congelada
├── audits/
│   ├── audit_acosol_bromuro_v0_1.py
│   ├── audit_acosol_temperatura_history_v0_1.py
│   └── audit_acosol_thm_extremes_v0_1.py
├── experiments/
│   └── model_acosol_thm_sftp_v0_4_history_backtest.py
├── Validation/                                  # informe y gráficas documentales
├── archive/
│   ├── audits_legacy/                            # inspect/audit SFTP originales
│   ├── datos_historicos/
│   ├── modelado_legacy/
│   ├── outputs_legacy/                           # incluye outputs del experimento v0.4
│   └── referencias/
├── sftp_raw/                                     # local, ignorado
├── acosol_sftp_dataset_v0_1/                    # generado, ignorado
├── acosol_sftp_dataset_v0_3/                    # generado, ignorado
├── model_outputs_acosol_sftp_v0_3_2_backtest/    # diagnóstico congelado
└── model_outputs_acosol_thm_formal_validation_v0_5/  # test 01–07/10
```

`config_private.py` contiene credenciales locales y no debe compartirse ni versionarse. Los raw, datasets, outputs y `archive/` se conservan localmente y permanecen excluidos de Git.

## 3. Origen de los datos SFTP

- Servidor: `hotstoragetriathlon.blob.core.windows.net`, puerto `22`, usuario `hotstoragetriathlon.acosol`.
- Ficheros productivos: `ABASTCLI01_DDMMYYYY_HHMMSS.csv` (no incluir `test.csv`).
- CSV UTF-8-SIG, separador `;`, decimal con coma, columnas `tag;timestamp;value` y timestamp `DD/MM/YYYY HH:MM:SS`.
- Cadencia nominal aproximada: **5 minutos**, sin remuestreo artificial. Se utiliza el **timestamp interno** del CSV.

| Tag SFTP | Señal | Papel actual |
|---|---|---|
| `SC_81_16_CLT02_PV` | cloro | feature |
| `SC_81_16_TRB01_PV` | turbidez | feature |
| `SC_81_16_COND01_PV` | conductividad | feature |
| `SC_81_16_PH01_PV` | pH | feature |
| `SC_81_16_CARB01_TOC_PV` | COT | feature |
| `SC_81_16_THM01_PV` | THM | objetivo Y |
| `SC_81_16_TEMP01_PV` | temperatura | conservar, excluir por incidencia confirmada de lectura |
| `SC_81_16_CARB01_TIC_PV` | TIC | conservar, excluir por decisión de planta |
| `SC_81_16_BROM01_PV` | bromuro | conservar, excluir hasta reparación/validación |

## 4. Cuándo y cómo actualizar el dataset

Ejecutar el flujo cuando lleguen nuevos CSV productivos, se corrijan datos o cambien las reglas de calidad. **Situarse siempre en la raíz `Acosol`**:

```powershell
python .\download_acosol_sftp_v0_1.py
python .\build_acosol_training_dataset_v0_1.py
python .\check_acosol_supervised_dataset_v0_1.py
python .\build_acosol_model_ready_dataset_v0_3.py
```

1. **Descarga incremental:** conserva raw sin alterar, compara tamaños, descarga atómica (`.part`) y actualiza manifest.
2. **Construcción:** transforma los snapshots en serie y tabla supervisada con THM; conserva procedencia (`source_file`) y auditoría.
3. **Comprobación:** verifica formato, tags, timestamps, duplicados, valores inválidos y huecos de adquisición. `[REVISAR]` no implica descartar automáticamente todos los datos.
4. **Model-ready:** aplica las exclusiones puntuales confirmadas, mantiene las columnas descartadas por motivos de auditoría y genera filas elegibles.

**Última ejecución documentada (datos hasta 08/10/2026):**

| Magnitud | Resultado |
|---|---:|
| CSV productivos descargados | 16.139 |
| Serie completa | 16.126 timestamps |
| Periodo continuo con THM | 03/09/2026 10:13:12 – 08/10/2026 17:15:24 |
| Filas supervisadas | 10.130 |
| Filas excluidas del model-ready | 12 |
| Filas elegibles | **10.118** |
| Huecos > 7,5 min en supervisado | 5 |
| Mayor hueco | ~174,8 min, 29/09 |
| Filas con THM ausente tras inicio | 0 |

Salidas principales:

```text
sftp_raw/
acosol_sftp_dataset_v0_1/acosol_sftp_supervised_v0_1.csv
acosol_sftp_dataset_v0_1/supervised_checks/
acosol_sftp_dataset_v0_3/acosol_sftp_supervised_annotated_v0_3.csv
acosol_sftp_dataset_v0_3/acosol_sftp_model_training_v0_3.csv
acosol_sftp_dataset_v0_3/acosol_sftp_model_excluded_rows_v0_3.csv
acosol_sftp_dataset_v0_3/acosol_sftp_model_metadata_v0_3.json
```

## 5. Reglas de calidad confirmadas por planta

**THM, 12/09/2026:** planta confirmó el **14/10/2026** que el bloque de ~131 µg/L fue una incidencia y no debe utilizarse para entrenar. La regla está limitada al intervalo **12:15:15–13:10:15** y a `THM >= 100 µg/L`: se excluyen **exactamente 12 lecturas**. El tramo posterior de ~12,75 µg/L **es válido y se conserva**. No hay regla genérica de descarte `THM > 100` fuera de ese episodio.

**Bromuro:** el medidor online está averiado/inestable y pendiente de servicio técnico. La auditoría mostró lecturas alternando ~0 y varios mg/L (picos ~9,99). Se almacena como raw pero no es feature. Los datos quincenales de laboratorio, si se reciben, pueden servir para contrastar la medida online; **no** deben interpolarse como si fueran una señal de 5 minutos.

**Temperatura:** la lectura tiene una incidencia confirmada y permanece casi constante en ~19,7 °C durante todo el periodo supervisado. Se excluye hasta que se recupere una señal fiable.

**TIC:** planta confirmó que no es necesario para la formación de THM en este estudio. Se conserva como señal raw pero se excluye del modelo. **COT se mantiene**.

**Control de entradas:** las columnas `source_file` y `THM` nunca deben incluirse entre las variables predictoras; THM es la referencia supervisada, no un predictor. Los valores extremos observados en otras entradas (p. ej., pH 14 o conductividad 5000) están pendientes de investigación específica; no deben eliminarse retrospectivamente para mejorar el resultado de un test ya observado.

## 6. Auditorías especializadas

### 6.1 Incidencias THM

`python .\audits\audit_acosol_thm_extremes_v0_1.py` permite estudiar los episodios elevados y su continuidad temporal. La exclusión del bloque ~131 se ejecuta en `build_acosol_model_ready_dataset_v0_3.py`, no dentro de esta auditoría.

### 6.2 Bromuro

`python .\audits\audit_acosol_bromuro_v0_1.py` produjo, hasta el 08/10/2026, **10.130** lecturas, de las cuales **4.612 (45,53 %) están prácticamente a cero** (`|bromuro| <= 0,001 mg/L`). El último periodo prácticamente a cero empezó el **05/10 17:00** y alcanzó el corte del **08/10 17:15**. Los valores próximos a 10 mg/L se consideran sospechosos, no evidencia de concentración real validada.

### 6.3 Temperatura y viabilidad de una variante con esta feature

`python .\audits\audit_acosol_temperatura_history_v0_1.py` revisó:

- 9 hojas históricas Excel: **ninguna** con temperatura (ni con temperatura+THM).
- SFTP completo: 16.126 filas, del 13/08 al 08/10; solo 2 valores de temperatura únicos tras redondear.
- SFTP supervisado con THM: 10.130 filas, temperatura ~19,7 °C, rango ~0,00002 °C y **0 días** con señal dinámica.

Por ello **no se puede validar de manera rigurosa** el modelo «sin bromuro, con temperatura» con los datos actuales. Sí es evaluable la variante «sin bromuro ni temperatura», que corresponde a la configuración de referencia actual. Una comparación futura A/B requerirá un periodo común con THM y temperatura recuperada, fiables ambos.

## 7. Método HGB congelado de desarrollo: v0.3.2

Script: `model_acosol_thm_sftp_v0_3_2_backtest.py`. Dataset original de desarrollo: **7.273 filas elegibles**, hasta el **28/09/2026 17:25:19**. No se ha guardado un modelo operativo final.

**Entradas:** cloro, turbidez, conductividad, pH, COT y las siete features de calendario: `hour`, `dayofweek`, `dayofyear`, `hour_sin`, `hour_cos`, `dayofyear_sin`, `dayofyear_cos` (**12 features**). No se usa histórico de THM ni lags de entradas en la versión base.

**HGB:** `loss=squared_error`, `learning_rate=0.04`, `l2_regularization=0.05`, `min_samples_leaf=20`, `early_stopping=False`, `random_state=42`.

Validación de desarrollo **walk-forward temporal, 21 folds**, con entrenamiento expansivo, 1 día de VAL y 1 día de TEST por fold (mínimo inicial de entrenamiento: 3 días). Para cada fold:

1. Entrenar HGB candidato **solo con TRAIN**.
2. Seleccionar iteraciones sobre **VAL cronológica** mediante `staged_predict` (máximo 4.000; paciencia 100; mejora mínima de MAE 0,0001; mínimo 20 iteraciones antes de detener).
3. Entrenar una nueva instancia sobre **TRAIN+VAL** con el número de iteraciones elegido.
4. Evaluar **TEST una sola vez**, sin utilizarlo en selección.

Resultados combinados de desarrollo:

| Métrica | HGB v0.3.2 |
|---|---:|
| MAE | 0,430363 µg/L |
| RMSE | 0,918989 µg/L |
| sMAPE | 11,027 % |
| Bias (predicción − real) | +0,070932 µg/L |
| R² | 0,130496 |

Mediana de iteraciones elegidas: 113, rango 1–1.015; ningún fold agotó el máximo de 4.000. Estos resultados describen **desarrollo**, no constituyen un TEST final independiente.

## 8. Experimento complementario v0.4: histórico de entradas

Script: `experiments/model_acosol_thm_sftp_v0_4_history_backtest.py`. Salidas trasladadas a `archive/outputs_legacy/model_outputs_acosol_sftp_v0_4_history_backtest/`.

Se probaron lags de **15, 30 y 60 min** y sus deltas para cloro, turbidez, conductividad, pH y COT. Los lags no cruzan gaps de adquisición, y nunca se utiliza THM histórico como entrada.

| Métrica | Baseline v0.3.2 | Variante v0.4 |
|---|---:|---:|
| MAE pooled | 0,430363 | 0,418111 |
| RMSE pooled | **0,918989** | 0,969973 |
| Folds con menor MAE | 11/21 | 10/21 |
| MAE en THM >= 10 µg/L | **8,432224** | 8,601630 |

La ligera ventaja de MAE global de v0.4 no fue consistente por fold y empeoró RMSE y errores en THM elevado. **No reemplaza al método congelado v0.3.2.**

## 9. Evaluación formal out-of-time: v0.5

Script: `validate_acosol_thm_formal_v0_5.py`. Resultados: `model_outputs_acosol_thm_formal_validation_v0_5/`. Informe y gráficas: `Validation/`.

El protocolo HGB y la selección de iteraciones se congelaron **antes** de evaluar un periodo posterior a los datos de desarrollo:

| Conjunto | Periodo | Filas |
|---|---|---:|
| TRAIN | 03/09/2026 10:13:12 – 29/09/2026 23:55:28 | 7.606 |
| VAL | 30/09/2026 | 288 |
| TEST final | 01/10/2026 00:00 – 08/10/2026 00:00 (fin exclusivo) | 2.016 |

En VAL se eligieron **128 iteraciones** (se detuvo tras 228); después se reentrenó sobre TRAIN+VAL. El Dummy de referencia predice constantemente la **mediana de THM de TRAIN+VAL**.

| Métrica | HGB v0.3.2 evaluado | Dummy mediana |
|---|---:|---:|
| MAE (µg/L) | 0,501528 | **0,370706** |
| RMSE (µg/L) | 0,570800 | **0,414937** |
| sMAPE | 15,038 % | **10,845 %** |
| Bias (µg/L) | −0,482409 | −0,312351 |
| R² | −3,366891 | −1,307648 |

**Interpretación:** en el periodo del 01–07/10 el Dummy superó al HGB. Esto **no implica que la mediana sea un soft-sensor mejor en general**: en esa semana el THM real permaneció bajo (2,82–4,25 µg/L, media ~3,53) y con escasa variabilidad, mientras el HGB subestimó de forma sistemática. El error medio absoluto del HGB fue ~0,50 µg/L y su P95 de error absoluto ~1,05 µg/L; son magnitudes que conviene presentar junto a la limitación relativa al Dummy.

**Cobertura:** 2.016 de 2.016 observaciones, **100 %**, mediana 5 min, sin gaps >7,5 min. El TEST no contiene medidas `THM >= 10 µg/L`: **no valida** la respuesta ante episodios elevados.

**Cambios estadísticos de entradas TRAIN+VAL → TEST:** cloro +0,95 desviaciones estándar; turbidez −0,87; pH −0,45; conductividad +0,09; COT −0,12. En magnitudes absolutas, las diferencias de medias son mucho menores (p. ej., cloro +0,083 mg/L y pH −0,064 unidades); el desplazamiento estadístico **no prueba** por sí solo un cambio fisicoquímico importante ni una causa concreta del error.

**Protección del TEST:** la semana 01–07/10 **ya está consumida** como evaluación de esta metodología. No elegir nuevas features, umbrales, hiperparámetros ni ventanas basándose en su resultado. Para una nueva versión, planificar otro test cronológico posterior intacto.

## 10. Estado y próximos pasos

- **Referencia actual:** dataset `v0.3`, procedimiento HGB `v0.3.2`, evaluación formal `v0.5`.
- Conservar el TEST 01–07/10 y los outputs originales sin alterarlos; si se regenera solo el model-ready para actualizar **metadatos descriptivos** de la confirmación de planta, no reescribir los resultados congelados de entrenamiento ni de evaluación.
- Investigar valores extremos de entradas (como pH 14 o conductividad 5000) con criterios definidos previamente para un **nuevo** experimento, no para retocar retrospectivamente el TEST consumido.
- Acumular nuevos datos THM; evaluar cambios de régimen y, si procede, ventanas de entrenamiento recientes usando únicamente desarrollo o futuros splits nuevos.
- Tras la reparación del bromuro, exigir validación de calidad antes de reutilizarlo como feature. Cuando la temperatura recupere una lectura fiable, recoger suficientes datos con THM y comparar con/sin temperatura sobre un **mismo periodo**.
- Revisar si es viable incorporar **caudal** al SFTP: el feed actual no lo incluye y puede aportar contexto en cambios reales del THM.
- No ejecutar una nueva validación formal sobre el periodo 01–07/10 ni presentar sus métricas como garantía operacional.

El resumen de decisiones y resultados vigentes se encuentra en [`CURRENT_STABLE.md`](CURRENT_STABLE.md).
