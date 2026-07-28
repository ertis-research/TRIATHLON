from __future__ import annotations

import csv
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko

try:
    import config_private as private_config
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "No existe config_private.py. Copia config_private.example.py como "
        "config_private.py y completa las credenciales SFTP."
    ) from exc


HOST = str(private_config.HOST)
PORT = int(getattr(private_config, "PORT", 22))
USERNAME = str(private_config.USERNAME)
PASSWORD = str(private_config.PASSWORD)
ROOT_PATH = str(getattr(private_config, "ROOT_PATH", "."))

DOWNLOAD_DIR = Path("csv_descargados")
DOWNLOAD_ONLY_NEW = True
SHOW_CSV_PREVIEW_ROWS = 5
SHOW_ALL_REMOTE_FILES = False
REMOTE_FILE_PREVIEW_COUNT = 10
CSV_ENCODINGS_TO_TRY = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


@dataclass(frozen=True)
class DownloadSummary:
    downloaded: list[Path]
    skipped: int
    failed: list[str]


def validate_private_config() -> None:
    missing = [
        name
        for name, value in {
            "HOST": HOST,
            "USERNAME": USERNAME,
            "PASSWORD": PASSWORD,
        }.items()
        if not str(value).strip()
    ]
    if missing:
        raise ValueError(
            "Faltan valores obligatorios en config_private.py: " + ", ".join(missing)
        )
    if PASSWORD == "PASSWORD":
        raise ValueError(
            "La contraseña de config_private.py sigue siendo el valor de la plantilla."
        )


def fmt_dt(timestamp: int | float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def human_size(num_bytes: int | float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def open_transport() -> paramiko.Transport:
    """Abre la conexión SFTP usando las credenciales de config_private.py."""
    transport = paramiko.Transport((HOST, PORT))
    try:
        transport.connect(username=USERNAME, password=PASSWORD)
        return transport
    except Exception:
        transport.close()
        raise


def list_dir_flat(sftp: paramiko.SFTPClient, path: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in sftp.listdir_attr(path):
        full_path = f"{path}/{entry.filename}".replace("//", "/")
        is_dir = stat.S_ISDIR(entry.st_mode)
        entries.append(
            {
                "name": entry.filename,
                "path": full_path,
                "is_dir": is_dir,
                "size": None if is_dir else int(entry.st_size),
                "mtime": int(entry.st_mtime),
                "mode": entry.st_mode,
            }
        )
    return entries


def download_csv_files(
    sftp: paramiko.SFTPClient,
    files_sorted: list[dict[str, Any]],
    download_dir: Path,
    only_new: bool = True,
) -> DownloadSummary:
    """Descarga CSV de forma incremental usando un fichero temporal por transferencia."""
    download_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    skipped = 0
    failed: list[str] = []

    for remote_file in files_sorted:
        name = str(remote_file["name"])
        if not name.lower().endswith(".csv"):
            continue

        remote_path = str(remote_file["path"])
        expected_size = int(remote_file["size"])
        local_path = download_dir / name
        temporary_path = local_path.with_name(f"{local_path.name}.part")

        if only_new and local_path.exists():
            try:
                if local_path.stat().st_size == expected_size:
                    skipped += 1
                    continue
            except OSError:
                pass

        try:
            temporary_path.unlink(missing_ok=True)
            sftp.get(remote_path, str(temporary_path))
            downloaded_size = temporary_path.stat().st_size
            if downloaded_size != expected_size:
                raise IOError(
                    f"Tamaño descargado inesperado: {downloaded_size} != {expected_size} bytes"
                )
            os.replace(temporary_path, local_path)
            remote_mtime = int(remote_file["mtime"])
            os.utime(local_path, (remote_mtime, remote_mtime))
            downloaded.append(local_path)
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            failed.append(remote_path)
            print(f"[ERROR] Descargando {remote_path} -> {local_path}: {exc}")

    print("=== DESCARGA CSV ===")
    print(f"Carpeta local destino: {download_dir.resolve()}")
    print(f"Descargados: {len(downloaded)}")
    if only_new:
        print(f"Saltados (ya existían con mismo tamaño): {skipped}")
    print(f"Fallidos: {len(failed)}")
    print()

    return DownloadSummary(downloaded=downloaded, skipped=skipped, failed=failed)


def get_last_csv_local_path(
    files_sorted: list[dict[str, Any]],
    download_dir: Path,
) -> Path | None:
    csv_files = [
        remote_file
        for remote_file in files_sorted
        if str(remote_file["name"]).lower().endswith(".csv")
    ]
    if not csv_files:
        return None
    return download_dir / str(csv_files[-1]["name"])


def read_csv_with_pandas(path: Path):
    import pandas as pd

    column_names = ["timestamp", "sensor", "metric", "value", "unit"]
    errors: list[tuple[str, str]] = []

    for encoding in CSV_ENCODINGS_TO_TRY:
        try:
            dataframe = pd.read_csv(
                path,
                sep=";",
                engine="python",
                header=None,
                names=column_names,
                usecols=[0, 1, 2, 3, 4],
                dtype=str,
                na_filter=False,
                encoding=encoding,
            )
            for column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(str).str.strip()
            return dataframe, encoding
        except Exception as exc:
            errors.append((encoding, str(exc)))

    error_lines = [f"  - {encoding}: {message}" for encoding, message in errors]
    raise RuntimeError(
        "No se pudo leer el CSV con ninguna de las codificaciones probadas:\n"
        + "\n".join(error_lines)
    )


def show_last_csv_structure(last_file: Path | None, preview_rows: int = 5) -> None:
    print("=== ESTRUCTURA DEL ÚLTIMO CSV ===")

    if last_file is None:
        print("No hay CSV para analizar.\n")
        return

    print(f"Archivo: {last_file.name}")
    print(f"Ruta:   {last_file.resolve()}\n")

    if not last_file.exists():
        print("El archivo no existe en local. Revisa la descarga.\n")
        return

    try:
        dataframe, used_encoding = read_csv_with_pandas(last_file)
        print(f"Codificación usada por pandas: {used_encoding}")
        print(f"Filas: {len(dataframe)}")
        print(f"Columnas: {dataframe.shape[1]}")
        print("Columnas:", ", ".join(dataframe.columns.tolist()))
        print("\nTipos inferidos:")
        for column, dtype in dataframe.dtypes.items():
            print(f"  - {column}: {dtype}")
        print(f"\nPreview (primeras {preview_rows} filas):")
        print(dataframe.head(preview_rows).to_string(index=False))
    except Exception as exc:
        print("[WARN] pandas no pudo leer el CSV de forma estructurada.")
        print(str(exc))
        print("Intentando lectura simple...")
        show_last_csv_simple(last_file, preview_rows)
    print()


def show_last_csv_simple(path: Path, preview_rows: int) -> None:
    file_handle = None
    used_encoding = None

    try:
        for encoding in CSV_ENCODINGS_TO_TRY:
            try:
                file_handle = path.open("r", newline="", encoding=encoding, errors="strict")
                used_encoding = encoding
                break
            except UnicodeError:
                if file_handle is not None:
                    file_handle.close()
                    file_handle = None

        if file_handle is None:
            file_handle = path.open("r", newline="", encoding="utf-8", errors="replace")
            used_encoding = "utf-8 (errors=replace)"

        with file_handle:
            reader = csv.reader(file_handle, delimiter=";")
            print(f"Codificación usada en lectura simple: {used_encoding}")
            print("Columnas: timestamp, sensor, metric, value, unit")
            print(f"\nPreview (primeras {preview_rows} filas):")
            shown = 0
            for row in reader:
                if not row:
                    continue
                if len(row) > 5 and row[-1] == "":
                    row = row[:5]
                print("  ", row[:5])
                shown += 1
                if shown >= preview_rows:
                    break
    except Exception as exc:
        print(f"[ERROR] No se pudo leer el archivo con la lectura simple: {exc}")


def print_remote_files(files_sorted: list[dict[str, Any]]) -> None:
    if not files_sorted:
        print("(no hay archivos)")
        return

    visible = files_sorted if SHOW_ALL_REMOTE_FILES else files_sorted[-REMOTE_FILE_PREVIEW_COUNT:]
    if not SHOW_ALL_REMOTE_FILES and len(files_sorted) > len(visible):
        print(
            f"Mostrando los {len(visible)} archivos más recientes de {len(files_sorted)}. "
            "Cambia SHOW_ALL_REMOTE_FILES a True para mostrar todos."
        )
    print(f"{'Fecha (mtime)':19}  {'Tamaño':>10}  Nombre")
    print("-" * 60)
    for remote_file in visible:
        print(
            f"{fmt_dt(remote_file['mtime']):19}  "
            f"{human_size(remote_file['size']):>10}  {remote_file['name']}"
        )


def main() -> int:
    transport: paramiko.Transport | None = None
    sftp: paramiko.SFTPClient | None = None

    try:
        validate_private_config()
        print("Conectando al servidor SFTP...")
        transport = open_transport()
        sftp = paramiko.SFTPClient.from_transport(transport)
        print("Conexión establecida correctamente.\n")

        try:
            current_directory = sftp.getcwd()
        except Exception:
            current_directory = None

        print("=== UBICACIÓN REMOTA ===")
        print(
            "Directorio remoto actual (getcwd): "
            + (current_directory or "(no disponible en este servidor)")
        )
        print(f"Directorio listado: {ROOT_PATH}\n")

        entries = list_dir_flat(sftp, ROOT_PATH)
        directories = [entry for entry in entries if entry["is_dir"]]
        files = [entry for entry in entries if not entry["is_dir"]]
        files_sorted = sorted(files, key=lambda item: item["mtime"])

        print("=== RESUMEN ===")
        print(f"Subcarpetas encontradas: {len(directories)}")
        print(f"Archivos encontrados:   {len(files)}\n")

        print("=== ARCHIVOS REMOTOS ===")
        print_remote_files(files_sorted)
        print()

        if directories:
            print("=== SUBCARPETAS ===")
            for directory in sorted(directories, key=lambda item: str(item["name"]).lower()):
                print(f"[DIR ] {directory['name']}  (mtime: {fmt_dt(directory['mtime'])})")
            print()

        summary = download_csv_files(
            sftp,
            files_sorted,
            download_dir=DOWNLOAD_DIR,
            only_new=DOWNLOAD_ONLY_NEW,
        )

        last_local_csv = get_last_csv_local_path(files_sorted, DOWNLOAD_DIR)
        show_last_csv_structure(last_local_csv, preview_rows=SHOW_CSV_PREVIEW_ROWS)

        return 1 if summary.failed else 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if sftp is not None:
            sftp.close()
        if transport is not None:
            transport.close()
        print("Conexión cerrada.")


if __name__ == "__main__":
    raise SystemExit(main())
