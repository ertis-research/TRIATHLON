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

El Excel `CLORIMETROS+CAUDALIMETROS.xlsx` permanece fuera del pipeline hasta que planta confirme el mapeo de puntos, las unidades, el significado de las senales y posibles duplicidades con los datos SFTP.

Antes de iniciar una nueva validacion formal, consultar `CURRENT_STABLE.md`.

---

## 2. Estructura principal activa

```text
Huelva/
+-- THM_Depositos/
+-- huelva_outputs_v0_2_thm_pdf/
+-- huelva_outputs_v0_3_72h_thm_pdf/
+-- huelva_multiparam_sftp/
+-- huelva_par_etl_outputs_v0_1/
+-- model_outputs_huelva_thm_softsensor_v0_4_formal/
+-- experiments/
|   +-- no_color_v0_5/
|       +-- train_huelva_thm_softsensor_v0_5_no_color.py
|       +-- README.md
|       +-- outputs/
+-- huesna_transfer/
|   +-- README.md
|   +-- evaluate_huesna_huelva_zero_shot_formal_v1_0.py
|   +-- evaluate_huesna_huelva_fixed_holdout_formal_v1_0.py
|   +-- results/
|   +-- outputs/
+-- Validation/
|   +-- Validacion_Huelva_octubre_2026.docx
|   +-- Validacion_Huelva_octubre_2026.pdf
+-- archive/
+-- config_private.py
+-- config_private.example.py
+-- extract_huelva_thm_pdfs_v0_2.py
+-- download_huelva_multiparam_sftp_v0_15_selective_cached.py
+-- inspect_huelva_selected_multiparam_v0_1.py
+-- build_huelva_par_feature_dataset_v0_1.py
+-- train_huelva_thm_softsensor_v0_4_formal.py
+-- CURRENT_STABLE.md
+-- workflow.md
```

`config_private.py` contiene la password SFTP local y debe permanecer ignorado por Git. `config_private.example.py` es la plantilla publica que se versiona.

Los datos brutos, outputs de entrenamiento, outputs regenerables y `archive/` permanecen fuera de Git mediante `.gitignore`.

Los experimentos complementarios se organizan en `experiments/`. Sus scripts y documentos de referencia se versionan, mientras que `outputs/` contiene resultados regenerables ignorados por Git.

La carpeta `huesna_transfer/results/` contiene unicamente resultados simplificados y consolidados que interesa versionar. `huesna_transfer/outputs/` contiene salidas regenerables y se ignora.

---

## 3. Regla de actualizacion

El flujo se ejecuta de forma event-driven.

Ejecutar el flujo cuando:

- se anade un nuevo PDF de laboratorio;
- una muestra pasa a disponer de fecha y hora exactas;
- se corrige una muestra THM;
- cambia el esquema de los CSV `par`;
- se desea iniciar una nueva validacion formal.

No es necesario reconstruir el conjunto supervisado cuando solo aparecen nuevos CSV multiparametricos posteriores a la ultima muestra THM exacta.

En Huelva, nuevos CSV sin nuevos targets THM no anaden ejemplos supervisados.

---

## 4. Paso 1 - extraccion de THM desde PDF

Ventanas de 24 horas para el ETL actual:

```powershell
python .\extract_huelva_thm_pdfs_v0_2.py `
  --pdf-root ".\THM_Depositos" `
  --output-dir ".\huelva_outputs_v0_2_thm_pdf" `
  --history-hours 24
```

Comprobar:

```text
PDFs encontrados
Muestras exactas para entrenar
Candidatas solo con fecha
Revisar/errores
```

Solo las muestras con fecha y hora exactas pasan al conjunto supervisado.

Referencia formal 01/10/2026:

```text
PDFs encontrados:               75
Muestras THM exactas:           34
Candidatas solo con fecha:      40
Revisar/errores:                 1
```

El PDF revisado del 23/10/2025 08:15 no contiene un resultado THM y permanece excluido.

---

## 5. Paso 2 - ventanas ampliadas para descarga SFTP

Para disponer de historico suficiente para las features multiescala hasta 72 horas:

```powershell
python .\extract_huelva_thm_pdfs_v0_2.py `
  --pdf-root ".\THM_Depositos" `
  --output-dir ".\huelva_outputs_v0_3_72h_thm_pdf" `
  --history-hours 72
```

Estas ventanas de 72 horas se utilizan para descargar los datos necesarios.

No deben utilizarse directamente como entrada del ETL actual para las features denominadas `24h`.

---

## 6. Paso 3 - descarga incremental de CSV `par`

```powershell
python .\download_huelva_multiparam_sftp_v0_15_selective_cached.py `
  --mode download-window `
  --windows-csv ".\huelva_outputs_v0_3_72h_thm_pdf\huelva_required_multiparametric_windows_exact_v0_2.csv" `
  --include-kinds par
```

Referencia formal 01/10/2026:

```text
Ficheros par representados: 20.045
Errores de descarga:         0
```

---

## 7. Paso 4 - construccion del dataset multiparametrico

El ETL actual utiliza literalmente `download_start_dt` y `download_end_dt` para las features que denomina `24h`.

Por este motivo debe recibir el CSV original de ventanas de 24 horas:

```powershell
python .\build_huelva_par_feature_dataset_v0_1.py `
  --manifest-csv ".\huelva_multiparam_sftp\logs\huelva_sftp_selective_download_manifest_v0_15.csv" `
  --windows-csv ".\huelva_outputs_v0_2_thm_pdf\huelva_required_multiparametric_windows_exact_v0_2.csv"
```

Decisiones activas:

- solo `par`;
- `fp` excluido;
- valor corregido valido como principal;
- valor medido valido como fallback;
- estado original conservado para auditoria;
- `spec_*` desactivado por defecto;
- indicadores `ana::tool` desactivados por defecto.

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

Referencia formal:

```text
Ficheros par leidos correctamente: 20.045 / 20.045
Filas multiparametricas:           205.641
Timestamps unicos limpios:         205.641
Cobertura media por variable:      0,998
```

La serie limpia completa permite que el entrenamiento formal reconstruya de forma independiente ventanas de 6, 12, 24, 48 y 72 horas.

---

## 8. Paso 5 - validacion formal local

Script activo:

```powershell
python .\train_huelva_thm_softsensor_v0_4_formal.py
```

Protocolo:

- `test = max(5, ceil(20% de N))`;
- split estrictamente temporal;
- test final completamente aislado;
- walk-forward expansivo solo dentro de desarrollo;
- comparacion de Dummy, Ridge y HGB;
- grupos `baseline_24h`, `multiscale_means` y `compact_level_trend`;
- seleccion global por RMSE, MAE y valor absoluto del bias;
- el test no modifica la seleccion.

Referencia 01/10/2026:

```text
Muestras exactas: 34
Desarrollo:       27
Test final:        7
```

Modelo seleccionado:

```text
ridge_baseline_24h_alpha_10
```

Walk-forward de desarrollo:

```text
MAE  = 3,7698 ug/L
RMSE = 4,6268 ug/L
Bias = -1,2134 ug/L
```

Test final 7/7:

```text
MAE  = 11,6500 ug/L
RMSE = 18,2223 ug/L
Bias = +0,4941 ug/L
```

La degradacion del test se asocia a un fuerte cambio de dominio en las variables de entrada, especialmente color, UV254, TOC y DOC. La muestra del 16/09/2026 produce una extrapolacion extrema del Ridge y no debe eliminarse ni utilizarse para volver a seleccionar el modelo.

Esta ejecucion queda congelada como primera referencia formal de Huelva.

### 8.1. Experimento complementario sin la variable color

Se repitio el protocolo de la validacion v0.4 excluyendo exclusivamente `color_hazen` de las entradas del modelo. Se conservaron las 34 muestras, el split 27/7, los 23 candidatos y el criterio de seleccion original.

Script:

```powershell
python .\experiments\no_color_v0_5\train_huelva_thm_softsensor_v0_5_no_color.py
```

El modelo seleccionado sigue siendo `ridge_baseline_24h_alpha_10`.

| Metrica test Ridge | v0.4 con color | v0.5 sin color |
|---|---:|---:|
| MAE (ug/L) | 11,650 | 11,701 |
| RMSE (ug/L) | 18,222 | 14,344 |
| Bias (ug/L) | +0,494 | -4,417 |

La mejora del RMSE se concentra principalmente en la muestra del 16/09. Sin embargo, las restantes predicciones empeoran en conjunto y el modelo presenta mayor tendencia a infraestimar.

El HGB seleccionado dentro de su familia obtiene RMSE = 12,010 ug/L en el test de v0.5, pero no sustituye al Ridge, ya que la seleccion global se realizo exclusivamente en desarrollo.

Esta prueba es exploratoria y complementaria: no modifica la seleccion ni la referencia formal v0.4. El detalle se conserva en `experiments/no_color_v0_5/README.md` y en el anexo de `Validation/`.

---

## 9. Transferencia formal Huesna -> Huelva

La transferencia es un analisis complementario y no participa en la seleccion del modelo local.

Los scripts activos se mantienen separados en `huesna_transfer/`:

```powershell
python .\huesna_transfer\evaluate_huesna_huelva_zero_shot_formal_v1_0.py
python .\huesna_transfer\evaluate_huesna_huelva_fixed_holdout_formal_v1_0.py
```

Modelo fuente:

```text
Huesna v5.8.12
model_outputs_thm_softsensor_v5_8_12_incidents_20260921_135541
```

Zero-shot global:

```text
Cobertura = 33/34 = 97,1 %
MAE       = 14,9676 ug/L
RMSE      = 16,9732 ug/L
Bias      = -14,7683 ug/L
Pearson   = 0,5563
```

Holdout equivalente:

```text
Desarrollo/calibracion Huesna: 27/27
Test Huesna:                   6/7
```

Mejor adaptacion observada: correccion fija de bias.

```text
MAE  = 8,2852 ug/L
RMSE = 9,3319 ug/L
Bias = -1,9266 ug/L
```

Comparacion sobre las seis muestras comunes:

```text
Ridge local:
MAE  = 6,6182
RMSE = 9,7783

Huesna + bias fijo:
MAE  = 8,2852
RMSE = 9,3319
```

Con seis muestras no existe evidencia suficiente para declarar una ventaja estable de una alternativa sobre la otra. El estudio anterior de transferencia tampoco mostro una superioridad consistente del modelo transferido frente al modelo lineal local.

La muestra del 16/09/2026 queda sin prediccion Huesna porque el pipeline fuente actual mantiene una guarda tecnica sobre UVAS introducida para limpieza de glitches.

Pendiente de confirmar con planta para futuras versiones de Huesna:

```text
hard-invalid -> no predecir
OOD plausible -> predecir + warning
in-domain -> prediccion normal
```

La validacion formal actual reproduce el contrato Huesna vigente y no aplica retrospectivamente esta posible mejora.

---

## 10. Diagnostico y estado actual

El diagnostico de deriva forma parte de la validacion formal y no debe utilizarse para modificar retrospectivamente el split o la seleccion.

Variables con mayor desplazamiento entre desarrollo y test:

```text
Color
UV254
TOC
DOC
Temperatura
```

En la muestra del 16/09/2026:

```text
THM real:         45,1 ug/L
Prediccion Ridge: 86,94 ug/L
Error:            +41,84 ug/L
```

Posteriormente, planta confirmo que el 15/09/2026 se agoto el aporte de agua procedente del pantano de BEAS, despues de varios dias de disminucion. Este cambio pudo modificar la composicion del agua bruta y coincide con la muestra mas extrema. Planta tambien comunico que el seguimiento de la sonda habia sido limitado, por lo que no puede descartarse una incidencia instrumental. Estos hechos no permiten atribuir toda la deriva de julio a septiembre a una unica causa. El detalle se documenta en `Validation/Cambio_variables_entrada_Huelva_octubre_2026.pdf`.

El bundle reentrenado con las 34 muestras puede conservarse como modelo actualizado para futuras pruebas, pero no hereda las metricas formales 27/7. Su rendimiento debe medirse con nuevas muestras futuras independientes.

El modelo queda formalmente caracterizado, pero no se considera todavia listo para uso operacional.

---

## 11. Nueva validacion futura

Cuando existan suficientes nuevas muestras THM exactas:

1. actualizar la extraccion de PDF;
2. regenerar ventanas 24 h y 72 h;
3. descargar `par` incrementalmente;
4. regenerar el ETL;
5. definir el nuevo split temporal antes de observar el nuevo test;
6. ejecutar una nueva version/directorio sin sobrescribir la referencia actual;
7. repetir el diagnostico de deriva;
8. repetir la transferencia Huesna solo como evidencia complementaria;
9. actualizar `CURRENT_STABLE.md`;
10. actualizar el informe formal.

No reutilizar el test actual para seleccionar modificaciones del modelo.
