# Transferencia formal Huesna -> Huelva

Carpeta separada para el estudio de transferencia entre plantas.

- Modelo fuente: Huesna v5.8.12 formal.
- Huelva: referencia formal 01/10/2026.
- `results/`: resultados consolidados y versionables.
- `outputs/`: salidas regenerables e ignoradas por Git.

Ejecucion desde la raiz de `Huelva`:

```powershell
python .\huesna_transfer\evaluate_huesna_huelva_zero_shot_formal_v1_0.py
python .\huesna_transfer\evaluate_huesna_huelva_fixed_holdout_formal_v1_0.py
```
