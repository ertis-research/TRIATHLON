from __future__ import annotations

import argparse
import csv
import re
import stat
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

import paramiko

try:
    import config_private as config
except ImportError as exc:
    raise SystemExit("No se ha encontrado config_private.py en la misma carpeta que el script.") from exc

PRODUCTION_FILENAME_RE = re.compile(r"^ABASTCLI01_(\d{8})_(\d{6})\.csv$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Descarga incremental de los CSV productivos del SFTP de Acosol."
    )
    parser.add_argument("--root-path", default=getattr(config, "ROOT_PATH", "."), help="Ruta remota que contiene los CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("sftp_raw"), help="Carpeta local para los CSV originales.")
    parser.add_argument("--from-date", default=None, help="Fecha/hora mínima: YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS.")
    parser.add_argument("--to-date", default=None, help="Fecha/hora máxima: YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS.")
    parser.add_argument("--latest", type=int, default=None, help="Limita la operación a los N CSV más recientes.")
    parser.add_argument("--dry-run", action="store_true", help="Muestra qué se descargaría sin escribir ficheros.")
    parser.add_argument("--progress-every", type=int, default=500, help="Muestra progreso cada N ficheros.")
    return parser.parse_args()


def validate_config() -> None:
    required = ["HOST", "PORT", "USERNAME", "PASSWORD"]
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise SystemExit("Faltan variables obligatorias en config_private.py: " + ", ".join(missing))
    if not str(config.HOST).strip():
        raise SystemExit("HOST está vacío en config_private.py.")
    if not str(config.USERNAME).strip():
        raise SystemExit("USERNAME está vacío en config_private.py.")
    if not str(config.PASSWORD):
        raise SystemExit("PASSWORD está vacío en config_private.py.")


def parse_filter_datetime(value: str | None, end_of_day: bool = False) -> datetime | None:
    if value is None:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d" and end_of_day:
                return parsed.replace(hour=23, minute=59, second=59)
            return parsed
        except ValueError:
            pass
    raise ValueError(f"Fecha/hora no válida: {value!r}. Usa YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS.")


def parse_filename_timestamp(filename: str) -> datetime | None:
    match = PRODUCTION_FILENAME_RE.fullmatch(filename)
    if not match:
        return None
    date_part, time_part = match.groups()
    try:
        return datetime.strptime(date_part + time_part, "%d%m%Y%H%M%S")
    except ValueError:
        return None


def remote_join(base: str, name: str) -> str:
    if base in {"", "."}:
        return name
    if base == "/":
        return f"/{name}"
    return str(PurePosixPath(base) / name)


def download_atomic(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path, expected_size: int) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = local_path.with_name(local_path.name + ".part")

    if temp_path.exists():
        temp_path.unlink()

    try:
        sftp.get(remote_path, str(temp_path))
        downloaded_size = temp_path.stat().st_size
        if downloaded_size != expected_size:
            raise OSError(
                f"Tamaño descargado incorrecto para {remote_path}: "
                f"{downloaded_size} bytes, esperados {expected_size}."
            )
        temp_path.replace(local_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "filename",
        "file_timestamp",
        "remote_size",
        "remote_mtime",
        "local_path",
        "status",
        "message",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    validate_config()

    if args.latest is not None and args.latest < 1:
        raise SystemExit("--latest debe ser >= 1.")
    if args.progress_every < 1:
        raise SystemExit("--progress-every debe ser >= 1.")

    try:
        from_dt = parse_filter_datetime(args.from_date)
        to_dt = parse_filter_datetime(args.to_date, end_of_day=True)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if from_dt is not None and to_dt is not None and from_dt > to_dt:
        raise SystemExit("--from-date no puede ser posterior a --to-date.")

    host = str(config.HOST).strip()
    port = int(config.PORT)
    username = str(config.USERNAME).strip()
    password = str(config.PASSWORD)
    root_path = str(args.root_path)
    output_dir = args.output_dir

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    transport = None
    sftp = None

    print("Descarga incremental SFTP Acosol v0.1")
    print("======================================")
    print(f"Host:          {host}")
    print(f"Puerto:        {port}")
    print(f"Usuario:       {username}")
    print(f"Ruta remota:   {root_path}")
    print(f"Carpeta local: {output_dir}")
    print(f"Modo dry-run:  {'sí' if args.dry_run else 'no'}")

    try:
        transport = paramiko.Transport((host, port))
        transport.set_keepalive(30)
        transport.connect(username=username, password=password)

        if not transport.is_authenticated():
            raise RuntimeError("La autenticación SFTP no quedó activa.")

        sftp = paramiko.SFTPClient.from_transport(transport)

        production_files = []
        ignored_csv = 0

        for attr in sftp.listdir_attr(root_path):
            if stat.S_ISDIR(attr.st_mode) or not attr.filename.lower().endswith(".csv"):
                continue

            file_timestamp = parse_filename_timestamp(attr.filename)
            if file_timestamp is None:
                ignored_csv += 1
                continue

            if from_dt is not None and file_timestamp < from_dt:
                continue
            if to_dt is not None and file_timestamp > to_dt:
                continue

            production_files.append(
                {
                    "filename": attr.filename,
                    "file_timestamp": file_timestamp,
                    "remote_size": int(attr.st_size or 0),
                    "remote_mtime": datetime.fromtimestamp(attr.st_mtime).isoformat(sep=" ") if attr.st_mtime is not None else "",
                }
            )

        production_files.sort(key=lambda item: item["file_timestamp"])

        if args.latest is not None:
            production_files = production_files[-args.latest:]

        print(f"\nCSV productivos seleccionados: {len(production_files)}")
        print(f"CSV ignorados por no cumplir el patrón productivo: {ignored_csv}")

        if not production_files:
            print("No hay CSV productivos que descargar con los filtros indicados.")
            return 0

        print(
            f"Rango seleccionado: {production_files[0]['file_timestamp']} -> "
            f"{production_files[-1]['file_timestamp']}"
        )

        manifest_rows = []
        downloaded = repaired = skipped = failed = would_download = would_repair = 0
        total = len(production_files)

        for index, item in enumerate(production_files, start=1):
            filename = str(item["filename"])
            file_timestamp = item["file_timestamp"]
            remote_size = int(item["remote_size"])
            remote_path = remote_join(root_path, filename)
            local_path = output_dir / filename
            status = ""
            message = ""

            try:
                if local_path.exists():
                    local_size = local_path.stat().st_size
                    if local_size == remote_size:
                        status = "skipped_existing"
                        skipped += 1
                    elif args.dry_run:
                        status = "would_repair_size_mismatch"
                        message = f"local_size={local_size}; remote_size={remote_size}"
                        would_repair += 1
                    else:
                        download_atomic(sftp, remote_path, local_path, remote_size)
                        status = "repaired_size_mismatch"
                        message = f"local_size_before={local_size}; remote_size={remote_size}"
                        repaired += 1
                elif args.dry_run:
                    status = "would_download"
                    would_download += 1
                else:
                    download_atomic(sftp, remote_path, local_path, remote_size)
                    status = "downloaded"
                    downloaded += 1

            except Exception as exc:
                status = "failed"
                message = f"{type(exc).__name__}: {exc}"
                failed += 1
                print(f"[ERROR] {filename}: {message}", file=sys.stderr)

            manifest_rows.append(
                {
                    "filename": filename,
                    "file_timestamp": file_timestamp.isoformat(sep=" "),
                    "remote_size": remote_size,
                    "remote_mtime": item["remote_mtime"],
                    "local_path": str(local_path),
                    "status": status,
                    "message": message,
                }
            )

            if index % args.progress_every == 0 or index == total:
                print(f"Procesados {index}/{total} ficheros...")

        print("\n" + "=" * 100)
        print("RESUMEN")
        print("=" * 100)

        if args.dry_run:
            print(f"Ya existentes y correctos:          {skipped}")
            print(f"Se descargarían nuevos:             {would_download}")
            print(f"Se repararían por tamaño distinto:  {would_repair}")
            print(f"Fallidos:                            {failed}")
            print("\nNo se ha escrito ningún fichero en modo --dry-run.")
        else:
            manifest_path = output_dir / "sftp_download_manifest.csv"
            write_manifest(manifest_path, manifest_rows)
            print(f"Descargados nuevos:                 {downloaded}")
            print(f"Reparados por tamaño distinto:      {repaired}")
            print(f"Ya existentes y correctos:          {skipped}")
            print(f"Fallidos:                            {failed}")
            print(f"\nManifest: {manifest_path}")

        if failed:
            print(
                "\n[WARN] La descarga terminó con errores. "
                "Puedes volver a ejecutar el script; los ficheros correctos se omitirán."
            )
            return 4

        print("\n[OK] Descarga incremental completada.")
        return 0

    except paramiko.AuthenticationException:
        print("[ERROR] Autenticación rechazada.", file=sys.stderr)
        return 2
    except (paramiko.SSHException, OSError, RuntimeError) as exc:
        print(f"[ERROR] Fallo SFTP: {exc}", file=sys.stderr)
        return 3
    finally:
        if sftp is not None:
            sftp.close()
        if transport is not None:
            transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
