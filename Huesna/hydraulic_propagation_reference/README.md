# Propagación hidráulica Huesna

Referencia histórica del desarrollo EPANET/WNTR. Esta parte queda fuera del flujo activo de la UMA.

Scripts:
- `huesna_epanet_baseline_v0_6_threshold_zones.py`: simulación, edad del agua, mapeo y zonas.
- `integrate_hgb_epanet_alerts_v0_5_operational_margin.py`: proyección temporal y clasificación.
- `evaluate_alert_safety_margins_v0_2.py`: evaluación offline de márgenes.
- `plot_huesna_plant_figures_v0_3.py`: figuras de análisis.

Mapeos: `mappings/`.

Recursos compartidos no versionados:
- `epanet_redAlta_03_04.inp`
- `20260416_tiempos recorrido.xlsx`

Dependencias: `wntr`, `pandas`, `numpy`, `openpyxl` y `matplotlib`.

Uso: ejecutar desde la raíz `SFTPs`, indicando explícitamente las rutas de entrada y `--output-dir`.

Notas:
- `n50` es la referencia de El Viso y el nodo `2` es la semilla hidráulica histórica para recorrer aguas abajo desde Regulador Huerto Reyes.
- THM no se simula como especie química en EPANET; la propagación usa tiempos de recorrido.
- Umbrales históricos por zona: 20/25, 35/40 y 75/80.
- El margen `+4` es histórico y no una configuración productiva definitiva.
