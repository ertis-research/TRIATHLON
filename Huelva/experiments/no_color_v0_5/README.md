
# Experimento v0.5 - eliminacion de color

## Objetivo

Evaluar la sensibilidad del soft-sensor de THM de Huelva a la eliminacion de `color_hazen`.

## Metodologia

Replica de v0.4 con el unico cambio de excluir color de las entradas del modelo.

- Muestras: 34.
- Desarrollo/test: 27/7.
- Validacion temporal y criterio de seleccion sin cambios.
- Modelo seleccionado: `ridge_baseline_24h_alpha_10`.

## Resultados

| Metrica | v0.4 con color | v0.5 sin color |
|---|---:|---:|
| MAE (ug/L) | 11,650 | 11,701 |
| RMSE (ug/L) | 18,222 | 14,344 |
| Bias (ug/L) | +0,494 | -4,417 |

En la muestra del 16/09, la prediccion Ridge pasa de 86,94 a 70,59 ug/L, frente a un THM real de 45,1 ug/L.

## Interpretacion

Eliminar color reduce el error extremo y mejora el RMSE, pero no mejora el MAE y aumenta el sesgo negativo. La deriva de otras variables de entrada persiste.

El test utilizado ya era conocido tras la validacion v0.4. Por tanto, esta comparacion es un analisis de sensibilidad posterior y no una nueva validacion independiente.

## Reproducibilidad

Ejecutar desde la raiz Huelva:

```powershell
python .\experiments\no_color_v0_5\train_huelva_thm_softsensor_v0_5_no_color.py
```

Las salidas regenerables se encuentran en `outputs/`.

La referencia formal principal permanece en v0.4.
