# Flujo de trabajo - soft-sensor THM Huelva

## 1. Alcance

La responsabilidad de la UMA se limita a la prediccion de THM mediante un modelo soft-sensor.

Quedan fuera de este flujo:

- propagacion hidraulica;
- clasificacion de prealertas y alertas;
- integracion con EPANET;
- logica operativa de alertas por punto.

El modelo se construye con:

- muestras de THM obtenidas de los PDF de laboratorio;
- datos multiparametricos de los ficheros `par`;
- ventanas historicas anteriores a cada muestra THM.

Los ficheros `fp` quedan excluidos por confirmacion de planta.

El Excel `CLORIMETROS+CAUDALIMETROS.xlsx` queda fuera del pipeline hasta que planta confirme el mapeo de puntos, las unidades, el significado de las senales y posibles duplicidades con los datos SFTP.

---

## 2. Estructura principal

```text
Huelva/
+-- THM_Depositos/
+-- huelva_outputs_v0_2_thm_pdf/
+-- huelva_multiparam_sftp/
|   +-- logs/
|   +-- raw_downloads_window_matched/
|   +-- selected_inspection_v0_1_clean/
+-- huelva_par_etl_outputs_v0_1/
+-- model_outputs_huelva_thm_softsensor_v0_2/
+-- archive/
+-- config_private.py
+-- config_private.example.py
+-- extract_huelva_thm_pdfs_v0_2.py
+-- download_huelva_multiparam_sftp_v0_15_selective_cached.py
+-- inspect_huelva_selected_multiparam_v0_1.py
+-- build_huelva_par_feature_dataset_v0_1.py
+-- train_huelva_thm_softsensor_v0_2.py
+-- workflow.md
```

`config_private.py` contiene la password SFTP local y debe permanecer ignorado por Git. `config_private.example.py` es la plantilla publica que se versiona.

---

## 3. Regla de actualizacion

El flujo debe ejecutarse de forma event-driven, no necesariamente cada dia.

### Ejecutar el flujo completo cuando

- se anade un PDF de laboratorio;
- una muestra pasa a disponer de fecha y hora exactas;
- se corrige una muestra THM;
- cambia el esquema de los CSV `par`;
- se desea reconstruir o reentrenar el modelo.

### No es necesario consultar el SFTP cuando

- no existen nuevas muestras THM exactas;
- solo han aparecido nuevos CSV multiparametricos posteriores a la ultima muestra;
- el dataset supervisado sigue sin cambios.

En Huelva, nuevos CSV sin nuevos targets THM no anaden ejemplos supervisados.

---

## 4. Paso 1 - extraccion de THM desde PDF

```powershell
python .\extract_huelva_thm_pdfs_v0_2.py `
  --pdf-root ".\THM_Depositos" `
  --output-dir ".\huelva_outputs_v0_2_thm_pdf" `
  --history-hours 24
```

Comprobar en consola:

```text
PDFs encontrados
Muestras exactas para entrenar
Candidatas solo con fecha
Revisar/errores
```

Salidas principales:

```text
huelva_outputs_v0_2_thm_pdf\
+-- huelva_thm_exact_datetime_samples_v0_2.csv
+-- huelva_required_multiparametric_windows_exact_v0_2.csv
+-- huelva_thm_date_only_candidates_v0_2.csv
+-- huelva_thm_review_pdfs_v0_2.csv
+-- huelva_thm_pdf_extraction_all_v0_2.csv
+-- huelva_thm_pdf_extraction_report_v0_2.xlsx
```

Solo las muestras con fecha y hora exactas pasan al entrenamiento.

---

## 5. Paso 2 - seleccion y descarga de CSV par

El downloader obtiene la password en este orden:

1. variable de entorno `HUELVA_SFTP_PASSWORD`;
2. fichero local `config_private.py`;
3. entrada interactiva, solo si se activa en el script.

Ejecucion normal:

```powershell
python .\download_huelva_multiparam_sftp_v0_15_selective_cached.py `
  --mode download-window
```

`par` es el tipo incluido por defecto. Tambien puede indicarse de forma explicita:

```powershell
python .\download_huelva_multiparam_sftp_v0_15_selective_cached.py `
  --mode download-window `
  --include-kinds par
```

El script:

1. carga las ventanas THM;
2. lista los ficheros `par` del SFTP;
3. reutiliza el indice local para ficheros ya conocidos;
4. inspecciona los nuevos o modificados;
5. detecta solape temporal con las ventanas THM;
6. descarga unicamente los ficheros necesarios.

Salidas principales:

```text
huelva_multiparam_sftp\logs\
+-- huelva_sftp_remote_csv_index_v0_15.csv
+-- huelva_sftp_remote_selected_v0_15.csv
+-- huelva_sftp_selective_download_manifest_v0_15.csv
+-- huelva_sftp_selective_runtime_config_v0_15.json
+-- paramiko_huelva_sftp_selective_v0_15.log
```

CSV descargados:

```text
huelva_multiparam_sftp\raw_downloads_window_matched
```

El JSON de runtime no guarda la password real: la sustituye por `***`.

### Nota de eficiencia

La version v0.15 es correcta y auditable, pero el SFTP plano sigue acumulando miles de ficheros.

La mejora futura recomendada es un selector en dos niveles:

1. prefiltro por nombre y metadatos, aprovechando timestamps como `_YYMMDD_HHMMSS_par.csv`;
2. verificacion de timestamps internos solo para candidatos proximos a las ventanas THM o con nombre ambiguo.

---

## 6. Paso 3 - inspeccion de estructura par

Este paso es opcional. No necesita ejecutarse en cada actualizacion.

Ejecutarlo cuando aparezcan nuevos esquemas o cambien sensores o columnas:

```powershell
python .\inspect_huelva_selected_multiparam_v0_1.py `
  --manifest-csv ".\huelva_multiparam_sftp\logs\huelva_sftp_selective_download_manifest_v0_15.csv" `
  --output-dir ".\huelva_multiparam_sftp\selected_inspection_v0_1_clean"
```

Comprobar:

```text
Ficheros en manifest
Ficheros locales existentes
Resumen por tipo par
read_ok igual a True
```

---

## 7. Paso 4 - construccion del dataset de features

Ejecucion normal:

```powershell
python .\build_huelva_par_feature_dataset_v0_1.py
```

El manifest y el CSV de ventanas ya tienen rutas por defecto coherentes con el pipeline actual.

Ejecucion explicita equivalente:

```powershell
python .\build_huelva_par_feature_dataset_v0_1.py `
  --manifest-csv ".\huelva_multiparam_sftp\logs\huelva_sftp_selective_download_manifest_v0_15.csv" `
  --windows-csv ".\huelva_outputs_v0_2_thm_pdf\huelva_required_multiparametric_windows_exact_v0_2.csv"
```

Decisiones aplicadas:

- solo `par`;
- `fp` excluido;
- nombres naturales;
- valor corregido valido como principal;
- valor medido valido como fallback;
- valores marcados explicitamente como no validos excluidos de la senal final;
- medido, corregido y status conservados para auditoria;
- `spec_*` desactivado por defecto;
- indicadores diagnosticos `ana::tool` desactivados por defecto.

Variables principales:

```text
turbidity_ntu
conductivity_us_cm
uv254_abs_m
toc_mg_l
doc_mg_l
free_chlorine_mg_l
ph
temperature_c
color_hazen
flow
```

Salidas:

```text
huelva_par_etl_outputs_v0_1\
+-- huelva_par_timeseries_raw_wide_v0_1.csv
+-- huelva_par_timeseries_clean_v0_1.csv
+-- huelva_thm_par_feature_dataset_v0_1.csv
+-- huelva_thm_par_window_coverage_v0_1.csv
+-- huelva_par_column_mapping_v0_1.csv
+-- huelva_par_read_issues_v0_1.csv
+-- huelva_par_etl_report_v0_1.md
+-- huelva_par_etl_report_v0_1.xlsx
```

Comprobar especialmente:

```text
Ficheros par leidos OK
Muestras THM exactas
Timestamps unicos limpios
Cobertura media por variable
```

---

## 8. Paso 5 - entrenamiento del soft-sensor

```powershell
python .\train_huelva_thm_softsensor_v0_2.py
```

La version v0.2:

1. ordena las muestras temporalmente;
2. reserva las ultimas cinco muestras como test final aislado;
3. usa las primeras 21 muestras como desarrollo;
4. ejecuta walk-forward expanding-window sobre desarrollo;
5. compara Dummy, Ridge y HGB;
6. selecciona cada familia usando RMSE, MAE y bias del walk-forward;
7. evalua una vez las familias seleccionadas en el test final;
8. mantiene HGB como familia principal del proyecto;
9. guarda por separado los modelos de evaluacion y los reentrenados con todos los datos.

Features actuales:

```text
medias de 24 horas de las 10 variables principales
```

`sample_point` no se utiliza como feature. En el dataset actual todos los targets corresponden al mismo punto de muestra.

Salidas principales:

```text
model_outputs_huelva_thm_softsensor_v0_2\
+-- feature_inventory.csv
+-- walk_forward_candidate_metrics.csv
+-- walk_forward_predictions_all.csv
+-- walk_forward_feature_filter_audit.csv
+-- final_test_metrics.csv
+-- final_test_predictions_all.csv
+-- final_test_feature_filter_audit.csv
+-- feature_target_correlations_development.csv
+-- primary_hgb_permutation_importance_test_exploratory.csv
+-- primary_hgb_evaluation_bundle.joblib
+-- primary_hgb_full_data_bundle.joblib
+-- best_overall_evaluation_bundle.joblib
+-- best_overall_full_data_bundle.joblib
+-- metadata.json
+-- final_test_predictions_vs_real.png
+-- primary_hgb_walk_forward_predictions_vs_real.png
+-- training_report_v0_2.md
+-- training_report_v0_2.xlsx
```

Con 26 muestras, los resultados deben considerarse exploratorios.

El baseline actual muestra que HGB no supera al Dummy en el test final. Esto no invalida el pipeline, pero indica que aun no existe una senal predictiva temporal suficientemente estable con la representacion actual.

---

## 9. Flujo completo

Cuando exista una nueva muestra THM exacta:

```powershell
python .\extract_huelva_thm_pdfs_v0_2.py `
  --pdf-root ".\THM_Depositos" `
  --output-dir ".\huelva_outputs_v0_2_thm_pdf" `
  --history-hours 24

python .\download_huelva_multiparam_sftp_v0_15_selective_cached.py `
  --mode download-window

python .\build_huelva_par_feature_dataset_v0_1.py

python .\train_huelva_thm_softsensor_v0_2.py
```

El inspector se mantiene como paso opcional de auditoria.

---

## 10. Estado de referencia actual

```text
PDFs encontrados:             62
Muestras THM exactas:         26
CSV par seleccionados:        47
CSV par leidos correctamente: 47/47
Filas multiparametricas:      174871
Timestamps unicos limpios:    174871
Cobertura media variables:    0.997
```

Split temporal del modelo v0.2:

```text
Desarrollo: 21 muestras
Test final: 5 muestras
Predicciones walk-forward por candidato: 11
```

Resultado de referencia del test final:

```text
Dummy mediana:
MAE  = 2.6200
RMSE = 3.6581

HGB principal:
MAE  = 4.6250
RMSE = 5.5664

Ridge seleccionado:
MAE  = 7.4389
RMSE = 7.9590
```

---

## 11. Siguiente evolucion prevista

La siguiente version debe centrarse en la representacion temporal, no en aumentar la complejidad del modelo.

Trabajo previsto:

```text
ETL multiventana
+-- 6 h
+-- 12 h
+-- 24 h
+-- 48 h
+-- 72 h

Entrenamiento v0.3
+-- Dummy
+-- Ridge
+-- HGB regularizado
+-- grupos compactos de variables
+-- seleccion walk-forward
+-- test final aislado
+-- diagnostico de deriva entre desarrollo y test
```

El objetivo es comprobar si existe senal predictiva con otra ventana temporal o si las variables disponibles no son suficientes con el numero actual de muestras.
