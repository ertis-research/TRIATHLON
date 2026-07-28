from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import posixpath
import re
import socket
import stat
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import paramiko

SCRIPT_DIR = Path(__file__).resolve().parent

SFTP_HOST = "hotstoragetriathlon.blob.core.windows.net"
SFTP_PORT = 22
SFTP_USERNAME = "hotstoragetriathlon.aguasdehuelva"

SFTP_PASSWORD_ENV = "HUELVA_SFTP_PASSWORD"
ASK_PASSWORD_IF_MISSING = False

try:
    from config_private import HUELVA_SFTP_PASSWORD as PRIVATE_SFTP_PASSWORD
except ModuleNotFoundError as exc:
    if exc.name != "config_private":
        raise
    PRIVATE_SFTP_PASSWORD = ""

SFTP_REMOTE_ROOT = "/"

LOCAL_ROOT = SCRIPT_DIR / "huelva_multiparam_sftp"
LOG_DIR = LOCAL_ROOT / "logs"
SELECTED_DOWNLOAD_DIR = LOCAL_ROOT / "raw_downloads_window_matched"
EXISTING_DOWNLOAD_DIRS = [
    LOCAL_ROOT / "raw_downloads",
    SELECTED_DOWNLOAD_DIR,
]

WINDOWS_CSV = SCRIPT_DIR / "huelva_outputs_v0_2_thm_pdf" / "huelva_required_multiparametric_windows_exact_v0_2.csv"

# Modos:
#   remote-index                  -> solo crea índice remoto ligero, sin descargar.
#   download-window               -> crea índice y descarga solo seleccionados.
#   download-selected-from-index  -> reutiliza índice remoto ya generado y descarga seleccionados.
MODE = "download-window"

# Planta ha confirmado que, por ahora, solo se utilizan ficheros par.
INCLUDE_FILE_KINDS = ["par"]
ALLOWED_SUFFIXES = [".csv"]

MAX_DEPTH = 8
MAX_FILES = 0
REMOTE_HEAD_BYTES = 32_768
REMOTE_TAIL_BYTES = 32_768
EXTRA_MARGIN_HOURS = 0.0

# Seguridad contra parseos ambiguos que generan fechas futuras irreales.
DROP_FUTURE_WINDOWS = True
FUTURE_WINDOW_TOLERANCE_DAYS = 7
DOWNLOAD_UNCERTAIN_WITHOUT_DATES = False
OVERWRITE = False

# v0.15: caché incremental del índice remoto.
# Evita releer por SFTP la cabecera/cola de miles de CSV si el fichero remoto
# conserva mismo path, tamaño y modified_time.
USE_INDEX_CACHE = True
FORCE_REINDEX = False
CACHE_SOURCE_INDEX_CSVS = [
    LOG_DIR / "huelva_sftp_remote_csv_index_v0_15.csv",
    LOG_DIR / "huelva_sftp_remote_csv_index_v0_14.csv",
    LOG_DIR / "huelva_sftp_remote_csv_index_v0_13.csv",
    LOG_DIR / "huelva_sftp_remote_csv_index_v0_12.csv",
    LOG_DIR / "huelva_sftp_remote_csv_index_v0_11.csv",
]

PARAMIKO_LOG_FILE = LOG_DIR / "paramiko_huelva_sftp_selective_v0_15.log"
CONNECT_TIMEOUT_SECONDS = 45
BANNER_TIMEOUT_SECONDS = 45
AUTH_TIMEOUT_SECONDS = 45
VERSION = "v0.15"

warnings.filterwarnings(
    "ignore",
    message="Could not infer format.*",
    category=UserWarning,
)

REMOTE_INDEX_CSV = LOG_DIR / "huelva_sftp_remote_csv_index_v0_15.csv"
REMOTE_SELECTED_CSV = LOG_DIR / "huelva_sftp_remote_selected_v0_15.csv"
DOWNLOAD_MANIFEST_CSV = LOG_DIR / "huelva_sftp_selective_download_manifest_v0_15.csv"
RUNTIME_CONFIG_JSON = LOG_DIR / "huelva_sftp_selective_runtime_config_v0_15.json"


@dataclass
class RemoteFile:
    remote_path: str
    filename: str
    extension: str
    size_bytes: int
    modified_time: str
    file_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga selectiva SFTP Huelva por solape con ventanas THM.")
    parser.add_argument("--mode", choices=["remote-index", "download-window", "download-selected-from-index"], default=None)
    parser.add_argument("--windows-csv", default=None)
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--include-kinds",
        default=None,
        help="Tipos separados por coma. Por defecto: par. Ejemplo opcional: par,fp,unknown.",
    )
    parser.add_argument("--download-uncertain", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-reindex", action="store_true", help="Ignora la caché y reindexa leyendo muestras remotas.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return re.sub(r"\s+", " ", text)


def normalize_column_name(name: Any) -> str:
    text = normalize_text(name).lower().replace("\ufeff", "")
    text = re.sub(r"[^a-z0-9áéíóúüñ]+", "_", text)
    return text.strip("_")


def classify_file_kind(filename: str) -> str:
    name = filename.lower()
    if "filetest" in name:
        return "filetest"
    if re.search(r"(^|_)par(\.|_|$)", name):
        return "par"
    if re.search(r"(^|_)log(\.|_|$)", name):
        return "log"
    if re.search(r"(^|_)fp(\.|_|$)", name):
        return "fp"
    return "unknown"


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    mode = args.mode or MODE
    windows_csv = Path(args.windows_csv).resolve() if args.windows_csv else Path(WINDOWS_CSV).resolve()
    remote_root = args.remote_root or SFTP_REMOTE_ROOT
    max_files = args.max_files if args.max_files is not None else MAX_FILES
    include_kinds = [x.strip().lower() for x in args.include_kinds.split(",") if x.strip()] if args.include_kinds else list(INCLUDE_FILE_KINDS)

    password_env = SFTP_PASSWORD_ENV.strip()
    password = os.environ.get(password_env, "").strip() if password_env else ""
    password_source = f"env:{password_env}" if password else ""

    if not password:
        password = PRIVATE_SFTP_PASSWORD.strip()
        if password:
            password_source = "config_private.py"

    if not password and ASK_PASSWORD_IF_MISSING:
        password_source = "interactive_prompt"

    return {
        "version": VERSION,
        "sftp": {
            "host": SFTP_HOST,
            "port": SFTP_PORT,
            "username": SFTP_USERNAME,
            "password_source": password_source or "missing",
            "password": password,
            "ask_password_if_missing": ASK_PASSWORD_IF_MISSING,
            "remote_root": remote_root,
        },
        "paths": {
            "local_root": str(LOCAL_ROOT),
            "log_dir": str(LOG_DIR),
            "selected_download_dir": str(SELECTED_DOWNLOAD_DIR),
            "existing_download_dirs": [str(p) for p in EXISTING_DOWNLOAD_DIRS],
            "windows_csv": str(windows_csv),
            "remote_index_csv": str(REMOTE_INDEX_CSV),
            "remote_selected_csv": str(REMOTE_SELECTED_CSV),
            "download_manifest_csv": str(DOWNLOAD_MANIFEST_CSV),
            "paramiko_log_file": str(PARAMIKO_LOG_FILE),
        },
        "selection": {
            "mode": mode,
            "include_file_kinds": include_kinds,
            "allowed_suffixes": ALLOWED_SUFFIXES,
            "max_depth": MAX_DEPTH,
            "max_files": max_files,
            "remote_head_bytes": REMOTE_HEAD_BYTES,
            "remote_tail_bytes": REMOTE_TAIL_BYTES,
            "extra_margin_hours": EXTRA_MARGIN_HOURS,
            "drop_future_windows": DROP_FUTURE_WINDOWS,
            "future_window_tolerance_days": FUTURE_WINDOW_TOLERANCE_DAYS,
            "download_uncertain_without_dates": bool(args.download_uncertain or DOWNLOAD_UNCERTAIN_WITHOUT_DATES),
            "overwrite": bool(args.overwrite or OVERWRITE),
            "use_index_cache": USE_INDEX_CACHE,
            "force_reindex": bool(args.force_reindex or FORCE_REINDEX),
        },
    }


def validate_config(config: Dict[str, Any]) -> None:
    if config["selection"]["mode"] not in ["remote-index", "download-window", "download-selected-from-index"]:
        raise ValueError("Modo no válido.")
    password = str(config["sftp"].get("password", "")).strip()
    if not password and not config["sftp"].get("ask_password_if_missing", False):
        raise ValueError(
            "Password SFTP no disponible. Define HUELVA_SFTP_PASSWORD en "
            "config_private.py o en la variable de entorno del mismo nombre."
        )
    if not Path(config["paths"]["windows_csv"]).exists():
        raise FileNotFoundError(f"No existe WINDOWS_CSV: {config['paths']['windows_csv']}")


def get_password(config: Dict[str, Any]) -> str:
    password = str(config["sftp"].get("password", "")).strip()
    if password:
        return password
    if config["sftp"].get("ask_password_if_missing", False):
        return getpass.getpass("Contraseña SFTP: ")
    raise ValueError(
        "Password SFTP no disponible. Revisa config_private.py o la variable "
        "de entorno HUELVA_SFTP_PASSWORD."
    )


def configure_paramiko_logging(config: Dict[str, Any]) -> None:
    log_file = Path(config["paths"]["paramiko_log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    paramiko.util.log_to_file(str(log_file), level="INFO")


def connect_sftp(config: Dict[str, Any]) -> Tuple[paramiko.Transport, paramiko.SFTPClient]:
    sock = socket.create_connection((config["sftp"]["host"], int(config["sftp"]["port"])), timeout=CONNECT_TIMEOUT_SECONDS)
    transport = paramiko.Transport(sock)
    transport.banner_timeout = BANNER_TIMEOUT_SECONDS
    transport.auth_timeout = AUTH_TIMEOUT_SECONDS
    transport.start_client(timeout=CONNECT_TIMEOUT_SECONDS)
    transport.auth_password(username=config["sftp"]["username"], password=get_password(config), fallback=False)
    if not transport.is_authenticated():
        raise paramiko.AuthenticationException("Autenticación SFTP no completada.")
    return transport, paramiko.SFTPClient.from_transport(transport)


def remote_join(parent: str, child: str) -> str:
    if parent in ["", "/"]:
        return "/" + child.lstrip("/")
    return posixpath.join(parent, child)


def list_remote_files(sftp: paramiko.SFTPClient, config: Dict[str, Any]) -> List[RemoteFile]:
    allowed_suffixes = {s.lower() for s in config["selection"]["allowed_suffixes"]}
    include_kinds = {k.lower() for k in config["selection"]["include_file_kinds"]}
    max_depth = int(config["selection"]["max_depth"])
    max_files = int(config["selection"]["max_files"] or 0)
    results: List[RemoteFile] = []

    def walk(path: str, depth: int) -> None:
        if max_files and len(results) >= max_files:
            return
        if depth > max_depth:
            return
        try:
            entries = sftp.listdir_attr(path)
        except Exception as exc:
            print(f"[WARN] No se puede listar {path}: {type(exc).__name__}: {exc}")
            return
        for entry in entries:
            if max_files and len(results) >= max_files:
                return
            name = entry.filename
            if name in [".", ".."]:
                continue
            remote_path = remote_join(path, name)
            mode = entry.st_mode or 0
            if stat.S_ISDIR(mode):
                walk(remote_path, depth + 1)
                continue
            if not stat.S_ISREG(mode):
                continue
            extension = Path(name).suffix.lower()
            if extension not in allowed_suffixes:
                continue
            kind = classify_file_kind(name)
            if kind not in include_kinds:
                continue
            mtime = datetime.fromtimestamp(entry.st_mtime).isoformat(sep=" ") if entry.st_mtime else ""
            results.append(RemoteFile(remote_path, name, extension, int(entry.st_size or 0), mtime, kind))

    walk(config["sftp"]["remote_root"], 0)
    return results


def read_remote_sample_text(sftp: paramiko.SFTPClient, remote_path: str, size_bytes: int, head_bytes: int, tail_bytes: int) -> Tuple[str, str, Optional[str]]:
    try:
        with sftp.open(remote_path, "rb") as f:
            head = f.read(min(head_bytes, max(size_bytes, 0)))
            tail = b""
            if size_bytes > head_bytes and tail_bytes > 0:
                try:
                    f.seek(max(0, size_bytes - tail_bytes))
                    tail = f.read(tail_bytes)
                except Exception:
                    tail = b""
        raw = head + b"\n" + tail
        for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
            try:
                return raw.decode(enc), enc, None
            except UnicodeDecodeError:
                pass
        return raw.decode("latin1", errors="replace"), "latin1", None
    except Exception as exc:
        return "", "", f"{type(exc).__name__}: {exc}"


DATETIME_REGEX_PATTERNS = [
    re.compile(r"\b(?P<y>20\d{2})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})[ T;,_-]+(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?\b"),
    re.compile(r"\b(?P<d>\d{1,2})[-/](?P<m>\d{1,2})[-/](?P<y>20\d{2})[ T;,_-]+(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?\b"),
    re.compile(r"\b(?P<y>20\d{2})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})\b"),
    re.compile(r"\b(?P<d>\d{1,2})[-/](?P<m>\d{1,2})[-/](?P<y>20\d{2})\b"),
]


def parse_datetime_match(match: re.Match[str]) -> Optional[datetime]:
    try:
        return datetime(
            int(match.group("y")),
            int(match.group("m")),
            int(match.group("d")),
            int(match.groupdict().get("h") or 0),
            int(match.groupdict().get("mi") or 0),
            int(match.groupdict().get("s") or 0),
        )
    except Exception:
        return None


def extract_datetimes_from_text(text: str) -> List[datetime]:
    found: List[datetime] = []
    for pattern in DATETIME_REGEX_PATTERNS:
        count = 0
        for match in pattern.finditer(text):
            dt = parse_datetime_match(match)
            if dt is not None:
                found.append(dt)
                count += 1
            if count >= 200:
                break
    unique: List[datetime] = []
    seen = set()
    for dt in found:
        key = dt.isoformat()
        if key not in seen:
            unique.append(dt)
            seen.add(key)
    return unique


def parse_window_datetime_value(value: Any) -> Optional[pd.Timestamp]:
    """
    Parser robusto para download_start/download_end.
    Evita el parseo vectorial de pandas con formatos mixtos, que estaba omitiendo ventanas.
    """
    raw = normalize_text(value)

    if not raw or raw.lower() in ["nan", "none", "nat"]:
        return None

    raw = raw.replace("T", " ").replace("Z", "")
    raw = re.sub(r"\.\d+$", "", raw).strip()

    # Orden importante:
    # - Si empieza por año: ISO/año-mes-día.
    # - Si empieza por día: formato español día/mes/año.
    if re.match(r"^20\d{2}[-/]", raw):
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d",
        ]
    else:
        formats = [
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y",
        ]

    for fmt in formats:
        try:
            return pd.Timestamp(datetime.strptime(raw, fmt))
        except ValueError:
            continue

    # Fallback último recurso, elemento a elemento.
    for dayfirst in [False, True]:
        try:
            parsed = pd.to_datetime(raw, errors="coerce", dayfirst=dayfirst)
            if pd.notna(parsed):
                return pd.Timestamp(parsed)
        except Exception:
            continue

    return None


def parse_window_datetime_series(series: pd.Series) -> pd.Series:
    return series.map(parse_window_datetime_value)


def load_windows(windows_csv: Path, extra_margin_hours: float) -> Tuple[pd.DataFrame, datetime, datetime]:
    df = pd.read_csv(windows_csv, dtype=str, encoding="utf-8-sig")

    if df.empty:
        raise ValueError(f"WINDOWS_CSV vacío: {windows_csv}")

    normalized_map = {normalize_column_name(col): col for col in df.columns}

    start_candidates = [
        "download_start",
        "window_start",
        "window_start_datetime",
        "start",
        "start_datetime",
        "inicio",
        "fecha_inicio",
        "datetime_start",
        "from",
    ]

    end_candidates = [
        "download_end",
        "window_end",
        "window_end_datetime",
        "end",
        "end_datetime",
        "fin",
        "fecha_fin",
        "datetime_end",
        "to",
    ]

    start_col = None
    end_col = None

    for candidate in start_candidates:
        if candidate in normalized_map:
            start_col = normalized_map[candidate]
            break

    for candidate in end_candidates:
        if candidate in normalized_map:
            end_col = normalized_map[candidate]
            break

    # Fallback: si no existen columnas inicio/fin, usar sample_timestamp como centro.
    if start_col is None or end_col is None:
        sample_col = None

        for candidate in ["sample_timestamp", "timestamp", "sample_datetime", "fecha_muestra", "datetime"]:
            if candidate in normalized_map:
                sample_col = normalized_map[candidate]
                break

        if sample_col is None:
            raise ValueError(
                f"No se pudieron inferir columnas de inicio/fin en {windows_csv}. Columnas: {list(df.columns)}"
            )

        parsed_sample = parse_window_datetime_series(df[sample_col])
        df["_window_start_auto"] = parsed_sample - pd.Timedelta(hours=24)
        df["_window_end_auto"] = parsed_sample
        start_col = "_window_start_auto"
        end_col = "_window_end_auto"

    df["_window_start_dt"] = parse_window_datetime_series(df[start_col])
    df["_window_end_dt"] = parse_window_datetime_series(df[end_col])

    bad_mask = df["_window_start_dt"].isna() | df["_window_end_dt"].isna()
    bad_count = int(bad_mask.sum())

    if bad_count:
        print(f"[WARN] Ventanas con fechas no parseables: {bad_count}. Se omiten.")

    df = df[~bad_mask].copy()

    if DROP_FUTURE_WINDOWS:
        max_reasonable = pd.Timestamp(datetime.now() + timedelta(days=FUTURE_WINDOW_TOLERANCE_DAYS))
        future_mask = df["_window_end_dt"] > max_reasonable
        future_count = int(future_mask.sum())

        if future_count:
            print(
                f"[WARN] Ventanas futuras sospechosas: {future_count}. "
                f"Se omiten para evitar parseo ambiguo. Límite: {max_reasonable}."
            )
            df = df[~future_mask].copy()

    if df.empty:
        raise ValueError("No queda ninguna ventana válida tras parsear fechas.")

    if extra_margin_hours:
        margin = pd.Timedelta(hours=extra_margin_hours)
        df["_window_start_dt"] = df["_window_start_dt"] - margin
        df["_window_end_dt"] = df["_window_end_dt"] + margin

    df["_window_start_dt"] = pd.to_datetime(df["_window_start_dt"])
    df["_window_end_dt"] = pd.to_datetime(df["_window_end_dt"])

    print(f"Columna inicio ventana: {start_col}")
    print(f"Columna fin ventana:    {end_col}")
    print(f"Ventanas cargadas:      {len(df)}")

    return (
        df,
        df["_window_start_dt"].min().to_pydatetime(),
        df["_window_end_dt"].max().to_pydatetime(),
    )


def count_overlapping_windows(file_min: Optional[datetime], file_max: Optional[datetime], windows_df: pd.DataFrame) -> int:
    if file_min is None or file_max is None:
        return 0
    mask = (windows_df["_window_start_dt"] <= pd.Timestamp(file_max)) & (windows_df["_window_end_dt"] >= pd.Timestamp(file_min))
    return int(mask.sum())


def overlaps(file_min: Optional[datetime], file_max: Optional[datetime], global_start: datetime, global_end: datetime) -> bool:
    return file_min is not None and file_max is not None and file_min <= global_end and file_max >= global_start


def cache_key_from_values(remote_path: Any, size_bytes: Any, modified_time: Any, file_kind: Any) -> Tuple[str, str, str, str]:
    return (
        normalize_text(remote_path),
        normalize_text(size_bytes),
        normalize_text(modified_time),
        normalize_text(file_kind),
    )


def cache_key_from_remote_file(rf: RemoteFile) -> Tuple[str, str, str, str]:
    return cache_key_from_values(rf.remote_path, rf.size_bytes, rf.modified_time, rf.file_kind)


def parse_cached_datetime(value: Any) -> Optional[datetime]:
    text_value = normalize_text(value)

    if not text_value or text_value.lower() in ["nan", "none", "nat"]:
        return None

    try:
        parsed = pd.to_datetime(text_value, errors="coerce", dayfirst=False)
        if pd.notna(parsed):
            return pd.Timestamp(parsed).to_pydatetime()
    except Exception:
        return None

    return None


def load_index_cache(config: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str, str, str], Dict[str, Any]], str, int]:
    if not config["selection"].get("use_index_cache", True):
        return {}, "", 0

    if config["selection"].get("force_reindex", False):
        return {}, "", 0

    cache_paths = []

    # Primero el índice de esta versión, después versiones anteriores compatibles.
    cache_paths.append(Path(config["paths"]["remote_index_csv"]))

    for path in CACHE_SOURCE_INDEX_CSVS:
        if path not in cache_paths:
            cache_paths.append(path)

    required_cols = {
        "remote_path",
        "filename",
        "extension",
        "file_kind",
        "size_bytes",
        "modified_time",
        "sample_min_datetime",
        "sample_max_datetime",
    }

    for path in cache_paths:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
        except Exception as exc:
            print(f"[WARN] No se pudo leer caché {path}: {type(exc).__name__}: {exc}")
            continue

        if df.empty or not required_cols.issubset(set(df.columns)):
            continue

        cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

        for _, row in df.iterrows():
            key = cache_key_from_values(
                row.get("remote_path", ""),
                row.get("size_bytes", ""),
                row.get("modified_time", ""),
                row.get("file_kind", ""),
            )

            cache[key] = row.to_dict()

        return cache, str(path), len(df)

    return {}, "", 0


def build_selection_from_minmax(
    file_min: Optional[datetime],
    file_max: Optional[datetime],
    windows_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Tuple[bool, str, int]:
    overlapping_windows = count_overlapping_windows(file_min, file_max, windows_df)

    if file_min is not None and file_max is not None:
        selected = overlapping_windows > 0
        reason = (
            f"sample_datetime_overlap_{overlapping_windows}_windows"
            if selected
            else "sample_datetime_no_window_overlap"
        )
    else:
        selected = bool(config["selection"]["download_uncertain_without_dates"])
        reason = (
            "uncertain_no_datetime_download_enabled"
            if selected
            else "uncertain_no_datetime_detected"
        )

    return selected, reason, overlapping_windows


def row_from_cache(
    rf: RemoteFile,
    cached_row: Dict[str, Any],
    windows_df: pd.DataFrame,
    global_start: datetime,
    global_end: datetime,
    config: Dict[str, Any],
    cache_source_file: str,
) -> Dict[str, Any]:
    file_min = parse_cached_datetime(cached_row.get("sample_min_datetime", ""))
    file_max = parse_cached_datetime(cached_row.get("sample_max_datetime", ""))

    selected, reason, overlapping_windows = build_selection_from_minmax(
        file_min=file_min,
        file_max=file_max,
        windows_df=windows_df,
        config=config,
    )

    row = dict(cached_row)

    # Actualizamos atributos remotos por si hay pequeñas diferencias de tipo/formato.
    row.update({
        "remote_path": rf.remote_path,
        "filename": rf.filename,
        "extension": rf.extension,
        "file_kind": rf.file_kind,
        "size_bytes": rf.size_bytes,
        "modified_time": rf.modified_time,
        "sample_min_datetime": file_min.isoformat(sep=" ") if file_min else "",
        "sample_max_datetime": file_max.isoformat(sep=" ") if file_max else "",
        "global_window_start": global_start.isoformat(sep=" "),
        "global_window_end": global_end.isoformat(sep=" "),
        "overlapping_windows": overlapping_windows,
        "selected_for_download": selected,
        "selection_reason": reason,
        "index_cache_status": "reused_same_path_size_mtime",
        "index_cache_source": cache_source_file,
    })

    return row


def row_from_remote_sample(
    sftp: paramiko.SFTPClient,
    rf: RemoteFile,
    windows_df: pd.DataFrame,
    global_start: datetime,
    global_end: datetime,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    text, encoding, read_error = read_remote_sample_text(
        sftp,
        rf.remote_path,
        rf.size_bytes,
        int(config["selection"]["remote_head_bytes"]),
        int(config["selection"]["remote_tail_bytes"]),
    )

    dts = extract_datetimes_from_text(text)
    file_min = min(dts) if dts else None
    file_max = max(dts) if dts else None

    selected, reason, overlapping_windows = build_selection_from_minmax(
        file_min=file_min,
        file_max=file_max,
        windows_df=windows_df,
        config=config,
    )

    return {
        "remote_path": rf.remote_path,
        "filename": rf.filename,
        "extension": rf.extension,
        "file_kind": rf.file_kind,
        "size_bytes": rf.size_bytes,
        "modified_time": rf.modified_time,
        "sample_encoding": encoding,
        "sample_read_error": read_error,
        "sample_datetime_count": len(dts),
        "sample_min_datetime": file_min.isoformat(sep=" ") if file_min else "",
        "sample_max_datetime": file_max.isoformat(sep=" ") if file_max else "",
        "global_window_start": global_start.isoformat(sep=" "),
        "global_window_end": global_end.isoformat(sep=" "),
        "overlapping_windows": overlapping_windows,
        "selected_for_download": selected,
        "selection_reason": reason,
        "sample_text_preview": normalize_text(text[:500]),
        "index_cache_status": "sampled_remote_new_or_changed",
        "index_cache_source": "",
    }


def index_remote_files(
    sftp: paramiko.SFTPClient,
    config: Dict[str, Any],
    windows_df: pd.DataFrame,
    global_start: datetime,
    global_end: datetime,
) -> pd.DataFrame:
    remote_files = list_remote_files(sftp, config)
    print(f"Ficheros remotos candidatos por extensión/tipo: {len(remote_files)}")

    cache, cache_source_file, cache_rows = load_index_cache(config)

    if cache:
        print(f"Caché índice usada: {cache_source_file} ({cache_rows} filas)")
        print("Solo se leerán remotamente ficheros nuevos o modificados.")
    else:
        if config["selection"].get("force_reindex", False):
            print("Caché ignorada por --force-reindex.")
        else:
            print("Sin caché compatible. Se leerán muestras remotas ligeras.")

    rows: List[Dict[str, Any]] = []
    reused = 0
    sampled = 0

    for idx, rf in enumerate(remote_files, start=1):
        key = cache_key_from_remote_file(rf)
        cached_row = cache.get(key)

        if cached_row is not None:
            row = row_from_cache(
                rf=rf,
                cached_row=cached_row,
                windows_df=windows_df,
                global_start=global_start,
                global_end=global_end,
                config=config,
                cache_source_file=cache_source_file,
            )
            reused += 1
        else:
            if sampled == 0 or sampled % 50 == 0:
                print(f"[nuevo/modificado {sampled + 1}] {rf.remote_path}")

            row = row_from_remote_sample(
                sftp=sftp,
                rf=rf,
                windows_df=windows_df,
                global_start=global_start,
                global_end=global_end,
                config=config,
            )
            sampled += 1

        if idx == 1 or idx % 1000 == 0 or idx == len(remote_files):
            print(f"[{idx}/{len(remote_files)}] reutilizados={reused} muestreados={sampled}")

        rows.append(row)

    print("")
    print("Resumen caché índice:")
    print(f"  Reutilizados desde caché: {reused}")
    print(f"  Muestreados por SFTP:     {sampled}")

    return pd.DataFrame(rows)


def safe_local_path_for_remote(base_dir: Path, remote_path: str) -> Path:
    parts = [p for p in remote_path.lstrip("/").split("/") if p and p not in [".", ".."]]
    if not parts:
        digest = hashlib.sha1(remote_path.encode("utf-8")).hexdigest()[:12]
        parts = [f"remote_file_{digest}.csv"]
    return base_dir.joinpath(*parts)


def find_existing_local(remote_path: str, size_bytes: int, config: Dict[str, Any]) -> Optional[Path]:
    for root_text in config["paths"]["existing_download_dirs"]:
        root = Path(root_text)
        for candidate in [safe_local_path_for_remote(root, remote_path), root / posixpath.basename(remote_path)]:
            if candidate.exists() and candidate.is_file() and candidate.stat().st_size == size_bytes:
                return candidate
    return None


def download_selected_files(sftp: paramiko.SFTPClient, selected_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    overwrite = bool(config["selection"]["overwrite"])
    target_root = Path(config["paths"]["selected_download_dir"])
    target_root.mkdir(parents=True, exist_ok=True)
    df = selected_df[selected_df["selected_for_download"].astype(bool)].copy()
    print(f"Ficheros seleccionados para descarga: {len(df)}")
    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        remote_path = str(row["remote_path"])
        size_bytes = int(row.get("size_bytes", 0))
        local_path = safe_local_path_for_remote(target_root, remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        existing = None if overwrite else find_existing_local(remote_path, size_bytes, config)
        error = ""
        if existing is not None:
            status = "skipped_exists_same_size"
            final_path = existing
        else:
            final_path = local_path
            try:
                sftp.get(remote_path, str(final_path))
                status = "downloaded"
            except Exception as exc:
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
        print(f"[{status}] {remote_path} -> {final_path}")
        rows.append({
            "remote_path": remote_path,
            "local_path": str(final_path),
            "download_status": status,
            "download_error": error,
            "size_bytes_remote": size_bytes,
            "size_bytes_local": final_path.stat().st_size if final_path.exists() else "",
            "file_kind": row.get("file_kind", ""),
            "sample_min_datetime": row.get("sample_min_datetime", ""),
            "sample_max_datetime": row.get("sample_max_datetime", ""),
            "overlapping_windows": row.get("overlapping_windows", ""),
            "selection_reason": row.get("selection_reason", ""),
        })
    return pd.DataFrame(rows)


def print_selection_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Índice remoto vacío.")
        return
    print("\nResumen índice remoto:")
    print(df["file_kind"].value_counts(dropna=False).to_string())
    print("\nResumen selección:")
    print(df["selection_reason"].value_counts(dropna=False).to_string())
    selected = df[df["selected_for_download"].astype(bool)]
    print(f"\nSeleccionados: {len(selected)}")
    print(f"Tamaño seleccionado aprox.: {selected['size_bytes'].sum() / (1024 * 1024):.2f} MB")


def write_runtime_config(config: Dict[str, Any]) -> None:
    cfg = json.loads(json.dumps(config, default=str))
    cfg["sftp"]["password"] = "***"
    path = Path(config["paths"]["log_dir"]) / RUNTIME_CONFIG_JSON.name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = build_config(args)
    validate_config(config)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SELECTED_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    configure_paramiko_logging(config)
    write_runtime_config(config)

    mode = config["selection"]["mode"]
    windows_csv = Path(config["paths"]["windows_csv"])
    windows_df, global_start, global_end = load_windows(windows_csv, float(config["selection"]["extra_margin_hours"]))

    print("========================================")
    print("SFTP Huelva descarga selectiva por ventanas THM")
    print("========================================")
    print(f"Versión:             {VERSION}")
    print(f"Host:                {config['sftp']['host']}")
    print(f"Usuario:             {config['sftp']['username']}")
    print(f"Remote root:         {config['sftp']['remote_root']}")
    print(f"Modo:                {mode}")
    print(f"Windows CSV:         {windows_csv}")
    print(f"Ventanas THM válidas:{len(windows_df)}")
    print(f"Rango ventanas:      {global_start} -> {global_end}")
    print(f"Tipos incluidos:     {', '.join(config['selection']['include_file_kinds'])}")
    print(f"Target download dir: {config['paths']['selected_download_dir']}")
    print(f"Password source:     {config['sftp']['password_source']}")
    print("")

    transport = None
    sftp = None
    try:
        if mode == "download-selected-from-index":
            if not REMOTE_INDEX_CSV.exists():
                raise FileNotFoundError(f"No existe índice remoto previo: {REMOTE_INDEX_CSV}")
            index_df = pd.read_csv(REMOTE_INDEX_CSV, dtype=str, encoding="utf-8-sig")
            index_df["selected_for_download"] = index_df["selected_for_download"].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            transport, sftp = connect_sftp(config)
            print("Conexión SFTP OK.")
            index_df = index_remote_files(sftp, config, windows_df, global_start, global_end)
            index_df.to_csv(REMOTE_INDEX_CSV, index=False, encoding="utf-8-sig")
            print(f"Índice remoto guardado en: {REMOTE_INDEX_CSV}")

        print_selection_summary(index_df)
        selected_df = index_df[index_df["selected_for_download"].astype(bool)].copy()
        selected_df.to_csv(REMOTE_SELECTED_CSV, index=False, encoding="utf-8-sig")
        print(f"Selección guardada en:     {REMOTE_SELECTED_CSV}")

        if mode in ["download-window", "download-selected-from-index"]:
            if sftp is None:
                transport, sftp = connect_sftp(config)
                print("Conexión SFTP OK.")
            manifest_df = download_selected_files(sftp, index_df, config)
            manifest_df.to_csv(DOWNLOAD_MANIFEST_CSV, index=False, encoding="utf-8-sig")
            print(f"Manifest descarga:         {DOWNLOAD_MANIFEST_CSV}")
            if not manifest_df.empty:
                print("\nResumen descarga:")
                print(manifest_df["download_status"].value_counts(dropna=False).to_string())
    finally:
        try:
            if sftp is not None:
                sftp.close()
        except Exception:
            pass
        try:
            if transport is not None:
                transport.close()
        except Exception:
            pass

    print("\n========================================")
    print("Proceso selectivo completado")
    print("========================================")


if __name__ == "__main__":
    main()
