# Referencia formal actual - soft-sensor THM Huelva

- Estado: primera validacion formal congelada.
- Fecha de cierre: 01/10/2026.
- Ultima muestra THM exacta: 16/09/2026 08:00.
- Script local de referencia: `train_huelva_thm_softsensor_v0_4_formal.py`.
- Salida formal local: `model_outputs_huelva_thm_softsensor_v0_4_formal`.
- Modelo seleccionado en desarrollo: `ridge_baseline_24h_alpha_10`.
- Bundle de evaluacion: `best_overall_evaluation_bundle.joblib`.
- Bundle reentrenado con las 34 muestras: `best_overall_full_data_bundle.joblib`.
- Informe formal: `Validation/Validacion_Huelva_octubre_2026.pdf`.

## Datos congelados

- PDFs de laboratorio: 75.
- Muestras THM exactas: 34.
- Candidatas sin hora exacta: 40.
- PDF sin resultado THM: 1.
- Ficheros `par` leidos: 20.045/20.045.
- Timestamps multiparametricos limpios: 205.641.
- Cobertura media de las 10 variables: 0,998.

## Split formal

Regla: `test = max(5, ceil(20% de N))`.

- Desarrollo: 27 muestras.
- Test final aislado: 7 muestras.
- Walk-forward dentro de desarrollo: minimo inicial 10; 17 predicciones por candidato.
- El test no se usa para seleccionar familia, features ni hiperparametros.

## Seleccion local

Seleccionado: `ridge_baseline_24h_alpha_10`.

Walk-forward desarrollo:

- MAE: 3,7698 ug/L.
- RMSE: 4,6268 ug/L.
- Bias: -1,2134 ug/L.

Test final 7/7:

- MAE: 11,6500 ug/L.
- RMSE: 18,2223 ug/L.
- Bias: +0,4941 ug/L.
- R2: -14,2022.
- Pearson: 0,0425.

Comparadores:

- HGB multiescala cfg. 3: MAE 12,0533; RMSE 12,9827; bias -12,0533.
- Dummy mediana: MAE 14,4429; RMSE 15,1802; bias -14,4429.

## Diagnostico principal

Existe un covariate shift fuerte entre desarrollo y test, especialmente en color, UV254, TOC y DOC.

La muestra del 16/09/2026 presenta una extrapolacion extrema del Ridge:

- THM real: 45,1 ug/L.
- Prediccion Ridge: 86,94 ug/L.
- Error: +41,84 ug/L.

No eliminar esta muestra ni reutilizar el test para seleccionar cambios.

## Experimento complementario v0.5 - sin color

- Fecha de ejecucion: 07/10/2026.
- Script: `experiments/no_color_v0_5/train_huelva_thm_softsensor_v0_5_no_color.py`.
- Salida regenerable: `experiments/no_color_v0_5/outputs/`.
- Unico cambio respecto a v0.4: exclusion de `color_hazen` de las entradas del modelo.
- Datos, split temporal, ventanas, familias de modelos y criterio de seleccion sin cambios.
- Desarrollo: 27 muestras; test: las mismas 7 muestras de v0.4.
- Modelo seleccionado en desarrollo: `ridge_baseline_24h_alpha_10`.

Walk-forward de desarrollo:

- MAE: 3,7804 ug/L.
- RMSE: 4,5622 ug/L.
- Bias: -1,3005 ug/L.

Test final del Ridge 7/7:

- MAE: 11,7006 ug/L.
- RMSE: 14,3441 ug/L.
- Bias: -4,4166 ug/L.

Comparacion frente a v0.4:

- RMSE: 18,2223 -> 14,3441 ug/L.
- MAE: 11,6500 -> 11,7006 ug/L.
- Prediccion del 16/09: 86,94 -> 70,59 ug/L; THM real 45,1 ug/L.

La eliminacion de color reduce el error extremo del 16/09, pero no mejora el MAE global del Ridge y aumenta su sesgo negativo. Persisten desplazamientos importantes en UV254, TOC, DOC y temperatura.

Este experimento es un analisis de sensibilidad posterior a la validacion formal v0.4. Reutiliza el test ya conocido y no constituye una nueva validacion independiente ni sustituye la referencia formal congelada.

## Contexto operativo confirmado por planta

Planta ha confirmado que el 15/09/2026 se agoto completamente el aporte de agua procedente del pantano de BEAS, que ya llevaba varios dias disminuyendo. Este cambio operativo coincide temporalmente con el episodio extremo observado el 16/09.

Asimismo, planta ha comunicado un seguimiento limitado de la sonda durante ese periodo. Por ello, no puede descartarse una contribucion instrumental a los valores registrados.

Esta informacion proporciona una explicacion operativa plausible del episodio, pero no demuestra que toda la deriva observada desde julio tenga el mismo origen.

## Transferencia Huesna -> Huelva

Fuente: Huesna v5.8.12 formal de 21/09/2026.

Zero-shot global:

- Cobertura: 33/34 = 97,1%.
- MAE: 14,9676 ug/L.
- RMSE: 16,9732 ug/L.
- Bias: -14,7683 ug/L.

Holdout equivalente 27/7:

- Desarrollo/calibracion Huesna disponible: 27/27.
- Test Huesna disponible: 6/7.
- Mejor adaptacion: correccion fija de bias.
- MAE: 8,2852 ug/L.
- RMSE: 9,3319 ug/L.
- Bias: -1,9266 ug/L.

Subconjunto comun de 6 muestras:

- Ridge local: MAE 6,6182; RMSE 9,7783; bias -6,3971.
- Huesna + bias: MAE 8,2852; RMSE 9,3319; bias -1,9266.

No existe una ventaja estable de uno sobre el otro con el numero actual de muestras.

## Limitaciones y pendientes

- El modelo no esta validado para uso operacional.
- El cambio de regimen reciente debe cubrirse con nuevas muestras exactas.
- Las 40 muestras sin hora exacta permanecen fuera del entrenamiento hasta disponer de timestamp fiable.
- Confirmar con planta la politica futura de inferencia de Huesna:
  - `hard-invalid`: dato realmente invalido -> abstencion.
  - `OOD`: dato plausible fuera del dominio -> prediccion + warning.
- La guarda UVAS 0-20 del pipeline Huesna se mantiene en esta validacion para reproducir el contrato actual.
- El bundle entrenado con las 34 muestras no tiene todavia un test futuro independiente.

## Regla de continuidad

Esta ejecucion queda congelada. No sobrescribir sus salidas. Cualquier nueva validacion debe:

1. incorporar las nuevas muestras exactas disponibles;
2. definir el split temporal antes de mirar el nuevo test;
3. seleccionar exclusivamente dentro de desarrollo;
4. reservar un nuevo test futuro;
5. compararse contra esta referencia.
