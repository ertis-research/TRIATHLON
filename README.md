# TRIATHLON: TRIhAlomeTHanes controL thrOugh AI-based techNologies

ETL = Extract, Transform, Load.

## Acosol

Descarga SFTP: `Acosol/download_acosol_sftp_v0_1.py`

ETL y calidad: `Acosol/build_acosol_training_dataset_v0_1.py`, `Acosol/check_acosol_supervised_dataset_v0_1.py` y `Acosol/build_acosol_model_ready_dataset_v0_3.py`.

Modelo de referencia: `Acosol/model_acosol_thm_sftp_v0_3_2_backtest.py`
- HGB con cloro, turbidez, conductividad, pH y COT; bromuro, temperatura y TIC excluidos.
- Evaluación formal: `Acosol/validate_acosol_thm_formal_v0_5.py` (test 01–07/10/2026). El Dummy mediana supera al HGB en ese período; modelo aún no desplegado.

## Benahavís

Plataforma Zeus: habilitación de acceso a la API pendiente.

## Huelva

Extracción THM de laboratorio: `Huelva/extract_huelva_thm_pdfs_v0_2.py`

Descarga selectiva SFTP: `Huelva/download_huelva_multiparam_sftp_v0_15_selective_cached.py` (solo ficheros `par`).

ETL: `Huelva/build_huelva_par_feature_dataset_v0_1.py`

Modelo de referencia: `Huelva/train_huelva_thm_softsensor_v0_4_formal.py`
- Ridge con histórico de 24 h, seleccionado mediante validación temporal; HGB y Dummy como comparadores.
- Test formal: 7 muestras de laboratorio con fecha y hora exactas.

## Huesna

Descarga SFTP: `Huesna/sftp.py`

Modelo de referencia: `Huesna/model_v5_8_12_incident_intervals.py`
- HGB con histórico multiparamétrico y selección temporal de variables de calendario.
- Exclusión de incidencias confirmadas y tratamiento específico de señales afectadas.

Diagnóstico temporal: `Huesna/analyze_huesna_softsensor_drift_v0_2.py`

Referencia de propagación hidráulica EPANET: `Huesna/hydraulic_propagation_reference/`

## Documentación

El flujo de trabajo y el estado de referencia de cada planta con soft-sensor se documentan en sus archivos `workflow.md` y `CURRENT_STABLE.md`. Los informes se conservan en `Validation/`.