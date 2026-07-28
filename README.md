# TRIATHLON: TRIhAlomeTHanes controL thrOugh AI-based techNologies

ETL = Extract, Transform, Load

## Acosol

ETL actual: load_acosol_excels_v0_1.py

Modelo de referencia: model_acosol_thm_v0_2.py
- hgb_base principal
- hgb_history candidato en seguimiento

Análisis de condiciones especiales: model_acosol_thm_v0_2_1.py

## Benahavís

Plataforma Zeus: Habilitación uso API pendiente.

## Huelva

Extracción THM: extract_huelva_thm_pdfs_v0_2.py

Descarga selectiva SFTP: download_huelva_multiparam_sftp_v0_15_selective_cached.py (solo ficheros "par")

ETL actual: build_huelva_par_feature_dataset_v0_1.py

Modelo de referencia: train_huelva_thm_softsensor_v0_2.py
- HGB principal
- Ridge y Dummy como baselines

## Huesna

Descarga SFTP: sftp.py

Modelo de referencia: model_v5_8_8_incident_intervals.py
- HGB principal con selección entre variantes de calendario
- Ridge y Dummy como baselines
- Exclusión de incidencias confirmadas mediante incident_intervals_huesna.csv

Diagnóstico temporal: analyze_huesna_softsensor_drift_v0_2.py

Referencia de propagación hidráulica EPANET: Huesna/hydraulic_propagation_reference/
