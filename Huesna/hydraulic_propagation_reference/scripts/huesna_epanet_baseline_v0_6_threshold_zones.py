from __future__ import annotations
import argparse
import json
import math
import re
import sys
import traceback
import unicodedata
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


SCRIPT_VERSION = "0.6.0"

# Mapeo inicial conservador basado en comentarios existentes dentro del .inp.
# Este mapeo se puede corregir/ampliar después con --mapping-csv.
DEFAULT_REFERENCE_NODE_MAPPING: Dict[str, str] = {
    "ALCOLEA": "n99",
    "BRENES": "n61",
    "CANTILLANA": "n44",
    "EL CORONIL": "n98",
    "CORONIL": "n98",
    "EL CUERVO": "n48",
    "LOS MOLARES": "n95",
    "MOLARES": "n95",
    "MARISMILLAS": "n89",
    "PARQUE LOGISTICO": "n55",
    "TOCINA": "n86",
    "TRAJANO": "n80",
    "UTRERA": "n52",
    # Depósitos/tanks comentados en el .inp. No todos aparecen directamente en la tabla.
    "REGULADOR HUERTO REYES": "2",
    "EL RUBIO": "3",
    "ROSARIO": "4",
}

# Alias que conviene evaluar antes que otros más generales.
# Ejemplo: "Utrera (Trajano)" debe mapear a TRAJANO antes que a UTRERA.
DEFAULT_MAPPING_PRIORITY: List[str] = [
    "REGULADOR HUERTO REYES",
    "PARQUE LOGISTICO",
    "LOS MOLARES",
    "EL CORONIL",
    "EL CUERVO",
    "MARISMILLAS",
    "CANTILLANA",
    "ALCOLEA",
    "TRAJANO",
    "BRENES",
    "TOCINA",
    "UTRERA",
    "CORONIL",
    "MOLARES",
    "ROSARIO",
    "EL RUBIO",
]

# Umbrales absolutos definidos por planta para clasificar estado THM por zona.
# La salida final sigue la convención de planta: normal / prealerta / alerta.
ALERT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ETAP": {"prealert_threshold": 20.0, "alert_threshold": 25.0},
    "REGULADOR_EL_VISO": {"prealert_threshold": 35.0, "alert_threshold": 40.0},
    "ENTRE_ETAP_Y_REGULADOR": {"prealert_threshold": 75.0, "alert_threshold": 80.0},
    "AGUAS_ABAJO_REGULADOR": {"prealert_threshold": 75.0, "alert_threshold": 80.0},
}

# Trazabilidad: planta indicó que el tramo entre ETAP y regulador usa el mismo
# criterio que aguas abajo del regulador. Por eso mantenemos una zona propia
# pero declaramos la fuente de umbral.
THRESHOLD_SOURCE_BY_ZONE: Dict[str, str] = {
    "ETAP": "ETAP",
    "REGULADOR_EL_VISO": "REGULADOR_EL_VISO",
    "ENTRE_ETAP_Y_REGULADOR": "AGUAS_ABAJO_REGULADOR",
    "AGUAS_ABAJO_REGULADOR": "AGUAS_ABAJO_REGULADOR",
    "UNKNOWN": "UNKNOWN",
}

ALERT_ZONE_LABELS: Dict[str, str] = {
    "ETAP": "ETAP",
    "REGULADOR_EL_VISO": "Regulador El Viso",
    "ENTRE_ETAP_Y_REGULADOR": "Entre ETAP y regulador",
    "AGUAS_ABAJO_REGULADOR": "Aguas abajo del regulador",
    "UNKNOWN": "Sin zona asignada",
}

ALERT_ZONE_ALIASES: Dict[str, str] = {
    "ETAP": "ETAP",
    "CABECERA": "ETAP",
    "DEP CABECERA": "ETAP",
    "DEP.CABECERA": "ETAP",
    "REGULADOR EL VISO": "REGULADOR_EL_VISO",
    "EL VISO": "REGULADOR_EL_VISO",
    "REGULADOR_EL_VISO": "REGULADOR_EL_VISO",
    "ENTRE ETAP Y REGULADOR": "ENTRE_ETAP_Y_REGULADOR",
    "ENTRE_ETAP_Y_REGULADOR": "ENTRE_ETAP_Y_REGULADOR",
    "TRAMO ETAP REGULADOR": "ENTRE_ETAP_Y_REGULADOR",
    "AGUAS ARRIBA REGULADOR": "ENTRE_ETAP_Y_REGULADOR",
    "AGUAS_ARRIBA_REGULADOR": "ENTRE_ETAP_Y_REGULADOR",
    "AGUAS ABAJO": "AGUAS_ABAJO_REGULADOR",
    "AGUAS ABAJO DEL REGULADOR": "AGUAS_ABAJO_REGULADOR",
    "AGUAS_ABAJO_REGULADOR": "AGUAS_ABAJO_REGULADOR",
    "DOWNSTREAM": "AGUAS_ABAJO_REGULADOR",
    "UNKNOWN": "UNKNOWN",
    "": "UNKNOWN",
}


@dataclass
class SimulationOutput:
    ok: bool
    results: Any = None
    error: Optional[str] = None
    traceback_text: Optional[str] = None


def log(message: str) -> None:
    print(message, flush=True)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def clean_text(value: Any) -> str:
    """Convierte una celda o valor opcional a texto limpio sin producir 'nan'."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).replace("\xa0", " ").strip()


def normalize_text(value: Any) -> str:
    """Normaliza texto para matching: mayúsculas, sin acentos y espacios limpios."""
    text = clean_text(value).upper()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()



def clean_number(value: Any) -> Optional[float]:
    """Convierte números tipo ' 21,36\xa0' a float. Devuelve None si no procede."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)

    text = str(value).replace("\xa0", " ").strip()
    if not text:
        return None
    text = text.replace(".", "") if re.match(r"^-?\d{1,3}(\.\d{3})+,\d+$", text) else text
    text = text.replace(",", ".")
    text = re.sub(r"[^0-9.\-eE+]", "", text)
    if text in {"", ".", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def import_wntr():
    try:
        import wntr  # type: ignore
        return wntr
    except Exception as exc:
        raise RuntimeError(
            "No se pudo importar WNTR. Instálalo con:\n"
            "  python -m pip install wntr pandas numpy openpyxl\n\n"
            f"Error original: {exc}"
        ) from exc


def split_data_and_comment(line: str) -> Tuple[str, str]:
    if ";" in line:
        data, comment = line.split(";", 1)
        return data.rstrip(), comment.strip()
    return line.rstrip(), ""


def parse_inp_sections(inp_path: Path) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None

    with inp_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[") and stripped.endswith("]"):
                current = stripped.strip("[]").upper()
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(line)
    return sections


def records_from_section(sections: Dict[str, List[str]], section: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in sections.get(section.upper(), []):
        stripped = line.strip()
        if not stripped or stripped.startswith(";"):
            continue
        data_part, comment = split_data_and_comment(line)
        tokens = data_part.split()
        if not tokens:
            continue
        rows.append({"id": tokens[0], "tokens": tokens, "comment": comment})
    return rows


def parse_inp_metadata(inp_path: Path) -> Dict[str, Any]:
    sections = parse_inp_sections(inp_path)
    return {
        "sections": {name: len(lines) for name, lines in sections.items()},
        "junction_comments": {r["id"]: r["comment"] for r in records_from_section(sections, "JUNCTIONS") if r["comment"]},
        "reservoir_comments": {r["id"]: r["comment"] for r in records_from_section(sections, "RESERVOIRS") if r["comment"]},
        "tank_comments": {r["id"]: r["comment"] for r in records_from_section(sections, "TANKS") if r["comment"]},
        "pipe_comments": {r["id"]: r["comment"] for r in records_from_section(sections, "PIPES") if r["comment"]},
        "raw_records": {
            "junctions": records_from_section(sections, "JUNCTIONS"),
            "reservoirs": records_from_section(sections, "RESERVOIRS"),
            "tanks": records_from_section(sections, "TANKS"),
            "pipes": records_from_section(sections, "PIPES"),
        },
    }


def get_attr(obj: Any, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            try:
                return getattr(obj, name)
            except Exception:
                pass
    return default


def get_node_coordinates(node: Any) -> Tuple[Optional[float], Optional[float]]:
    coords = get_attr(node, ["coordinates"], None)
    if coords is None:
        return None, None
    try:
        return float(coords[0]), float(coords[1])
    except Exception:
        return None, None


def get_junction_base_demand(node: Any) -> Optional[float]:
    # WNTR moderno: demand_timeseries_list[0].base_value
    try:
        dts = getattr(node, "demand_timeseries_list")
        if len(dts) > 0:
            value = getattr(dts[0], "base_value", None)
            return None if value is None else float(value)
    except Exception:
        pass

    # Fallback para versiones antiguas.
    value = get_attr(node, ["base_demand", "demand"], None)
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def network_summary(wn: Any, inp_metadata: Dict[str, Any], inp_path: Path) -> Dict[str, Any]:
    time_options = getattr(wn.options, "time", None)
    hydraulic_options = getattr(wn.options, "hydraulic", None)
    quality_options = getattr(wn.options, "quality", None)

    return {
        "script_version": SCRIPT_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inp_file": str(inp_path),
        "counts": {
            "junctions": len(getattr(wn, "junction_name_list", [])),
            "reservoirs": len(getattr(wn, "reservoir_name_list", [])),
            "tanks": len(getattr(wn, "tank_name_list", [])),
            "pipes": len(getattr(wn, "pipe_name_list", [])),
            "pumps": len(getattr(wn, "pump_name_list", [])),
            "valves": len(getattr(wn, "valve_name_list", [])),
            "nodes_total": len(getattr(wn, "node_name_list", [])),
            "links_total": len(getattr(wn, "link_name_list", [])),
        },
        "options_seconds": {
            "duration": get_attr(time_options, ["duration"], None),
            "hydraulic_timestep": get_attr(time_options, ["hydraulic_timestep"], None),
            "quality_timestep": get_attr(time_options, ["quality_timestep"], None),
            "report_timestep": get_attr(time_options, ["report_timestep"], None),
            "pattern_timestep": get_attr(time_options, ["pattern_timestep"], None),
        },
        "options_hours": {
            "duration": seconds_to_hours(get_attr(time_options, ["duration"], None)),
            "hydraulic_timestep": seconds_to_hours(get_attr(time_options, ["hydraulic_timestep"], None)),
            "quality_timestep": seconds_to_hours(get_attr(time_options, ["quality_timestep"], None)),
            "report_timestep": seconds_to_hours(get_attr(time_options, ["report_timestep"], None)),
            "pattern_timestep": seconds_to_hours(get_attr(time_options, ["pattern_timestep"], None)),
        },
        "hydraulic_options": {
            "inpfile_units": get_attr(hydraulic_options, ["inpfile_units"], None),
            "headloss": get_attr(hydraulic_options, ["headloss"], None),
            "demand_model": get_attr(hydraulic_options, ["demand_model"], None),
        },
        "quality_options": {
            "parameter": get_attr(quality_options, ["parameter"], None),
            "chemical_name": get_attr(quality_options, ["chemical_name"], None),
            "diffusivity": get_attr(quality_options, ["diffusivity"], None),
        },
        "inp_sections_line_count": inp_metadata.get("sections", {}),
        "known_named_junctions_from_inp_comments": inp_metadata.get("junction_comments", {}),
        "known_named_tanks_from_inp_comments": inp_metadata.get("tank_comments", {}),
        "units_note": (
            "WNTR lee el INP de EPANET y convierte internamente parámetros/resultados a SI. "
            "En los CSV se exportan valores SI y algunas conversiones prácticas cuando procede."
        ),
    }


def seconds_to_hours(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value) / 3600.0
    except Exception:
        return None


def build_nodes_summary(wn: Any, inp_metadata: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    comments: Dict[str, str] = {}
    comments.update(inp_metadata.get("junction_comments", {}))
    comments.update(inp_metadata.get("reservoir_comments", {}))
    comments.update(inp_metadata.get("tank_comments", {}))

    for name in getattr(wn, "node_name_list", []):
        node = wn.get_node(name)
        x, y = get_node_coordinates(node)
        node_type = node.__class__.__name__

        row: Dict[str, Any] = {
            "node_id": name,
            "node_type": node_type,
            "inp_comment": comments.get(name, ""),
            "x": x,
            "y": y,
            "elevation_or_head_m": None,
            "base_demand_m3s": None,
            "base_demand_Lps": None,
            "init_level_m": None,
            "min_level_m": None,
            "max_level_m": None,
            "diameter_m": None,
        }

        if node_type.lower().endswith("junction"):
            elev = get_attr(node, ["elevation"], None)
            demand = get_junction_base_demand(node)
            row["elevation_or_head_m"] = elev
            row["base_demand_m3s"] = demand
            row["base_demand_Lps"] = None if demand is None else demand * 1000.0
        elif node_type.lower().endswith("reservoir"):
            row["elevation_or_head_m"] = get_attr(node, ["base_head", "head"], None)
        elif node_type.lower().endswith("tank"):
            row["elevation_or_head_m"] = get_attr(node, ["elevation"], None)
            row["init_level_m"] = get_attr(node, ["init_level"], None)
            row["min_level_m"] = get_attr(node, ["min_level"], None)
            row["max_level_m"] = get_attr(node, ["max_level"], None)
            row["diameter_m"] = get_attr(node, ["diameter"], None)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["node_type", "node_id"]).reset_index(drop=True)


def build_links_summary(wn: Any, inp_metadata: Dict[str, Any]) -> pd.DataFrame:
    pipe_comments = inp_metadata.get("pipe_comments", {})
    rows: List[Dict[str, Any]] = []

    for name in getattr(wn, "link_name_list", []):
        link = wn.get_link(name)
        link_type = link.__class__.__name__
        row: Dict[str, Any] = {
            "link_id": name,
            "link_type": link_type,
            "inp_comment": pipe_comments.get(name, ""),
            "start_node": get_attr(link, ["start_node_name"], None),
            "end_node": get_attr(link, ["end_node_name"], None),
            "length_m": get_attr(link, ["length"], None),
            "diameter_m": get_attr(link, ["diameter"], None),
            "diameter_mm": None,
            "roughness": get_attr(link, ["roughness"], None),
            "minor_loss": get_attr(link, ["minor_loss"], None),
            "initial_status": str(get_attr(link, ["initial_status"], "")),
        }
        if row["diameter_m"] is not None:
            try:
                row["diameter_mm"] = float(row["diameter_m"]) * 1000.0
            except Exception:
                pass
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["link_type", "link_id"]).reset_index(drop=True)


def build_named_elements_summary(inp_metadata: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for section_name, comments_key, element_type in [
        ("JUNCTIONS", "junction_comments", "junction"),
        ("RESERVOIRS", "reservoir_comments", "reservoir"),
        ("TANKS", "tank_comments", "tank"),
        ("PIPES", "pipe_comments", "pipe"),
    ]:
        for element_id, comment in inp_metadata.get(comments_key, {}).items():
            if comment:
                rows.append(
                    {
                        "element_id": element_id,
                        "element_type": element_type,
                        "section": section_name,
                        "inp_comment": comment,
                        "normalized_comment": normalize_text(comment),
                    }
                )
    return pd.DataFrame(rows).sort_values(["element_type", "element_id"]).reset_index(drop=True)



def run_epanet_simulation(
    wn: Any,
    wntr: Any,
    label: str,
    output_dir: Path,
) -> SimulationOutput:
    """Ejecuta EPANET en una carpeta temporal aislada y exige convergencia."""
    safe_label = re.sub(r"[^A-Za-z0-9_-]+", "_", normalize_text(label).lower()).strip("_") or "simulation"
    try:
        with tempfile.TemporaryDirectory(prefix=f"wntr_{safe_label}_", dir=str(output_dir)) as temp_dir:
            file_prefix = str(Path(temp_dir) / safe_label)
            simulator = wntr.sim.EpanetSimulator(wn)
            try:
                results = simulator.run_sim(
                    file_prefix=file_prefix,
                    save_hyd=False,
                    convergence_error=True,
                )
            except TypeError:
                # Compatibilidad con versiones de WNTR sin convergence_error.
                results = simulator.run_sim(file_prefix=file_prefix, save_hyd=False)

            error_code = getattr(results, "error_code", None)
            if error_code not in (None, 0):
                raise RuntimeError(f"EPANET devolvió error_code={error_code}")
            return SimulationOutput(ok=True, results=results)
    except Exception as exc:
        return SimulationOutput(
            ok=False,
            error=f"Error ejecutando simulación {label}: {exc}",
            traceback_text=traceback.format_exc(),
        )



def result_keys(result_group: Any) -> List[str]:
    try:
        return list(result_group.keys())
    except Exception:
        return []


def get_result_dataframe(result_group: Any, key: str) -> Optional[pd.DataFrame]:
    try:
        df = result_group[key]
        if isinstance(df, pd.DataFrame):
            return df
        return pd.DataFrame(df)
    except Exception:
        return None


def timeseries_to_long(df: pd.DataFrame, variable: str, element_kind: str, value_column: str = "value_si") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    out = df.copy()
    out.index.name = "time_s"
    out = out.reset_index()
    out["time_h"] = pd.to_numeric(out["time_s"], errors="coerce") / 3600.0

    id_vars = ["time_s", "time_h"]
    long_df = out.melt(id_vars=id_vars, var_name="element_id", value_name=value_column)
    long_df.insert(2, "element_kind", element_kind)
    long_df.insert(3, "variable", variable)
    return long_df


def export_hydraulic_results(results: Any, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    node_frames: List[pd.DataFrame] = []
    link_frames: List[pd.DataFrame] = []
    exported: Dict[str, Any] = {"node_keys": [], "link_keys": []}

    node_keys = result_keys(results.node)
    link_keys = result_keys(results.link)
    exported["node_keys"] = node_keys
    exported["link_keys"] = link_keys

    for key in node_keys:
        df = get_result_dataframe(results.node, key)
        if df is None:
            continue
        long_df = timeseries_to_long(df, key, "node")
        if long_df.empty:
            continue
        if key == "demand":
            long_df["value_Lps"] = long_df["value_si"] * 1000.0
        node_frames.append(long_df)

    for key in link_keys:
        df = get_result_dataframe(results.link, key)
        if df is None:
            continue
        long_df = timeseries_to_long(df, key, "link")
        if long_df.empty:
            continue
        if key == "flowrate":
            long_df["value_Lps"] = long_df["value_si"] * 1000.0
        link_frames.append(long_df)

    node_results = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()
    link_results = pd.concat(link_frames, ignore_index=True) if link_frames else pd.DataFrame()

    safe_to_csv(node_results, output_dir / "hydraulic_node_results_long.csv")
    safe_to_csv(link_results, output_dir / "hydraulic_link_results_long.csv")

    # También exportamos algunas matrices wide útiles para inspección rápida.
    for key in ["pressure", "head", "demand", "quality"]:
        df = get_result_dataframe(results.node, key)
        if df is not None:
            df2 = df.copy()
            df2.index.name = "time_s"
            df2.insert(0, "time_h", pd.to_numeric(df2.index, errors="coerce") / 3600.0)
            df2.to_csv(output_dir / f"node_{key}_wide.csv", encoding="utf-8-sig")

    for key in ["flowrate", "velocity", "status", "headloss"]:
        df = get_result_dataframe(results.link, key)
        if df is not None:
            df2 = df.copy()
            df2.index.name = "time_s"
            df2.insert(0, "time_h", pd.to_numeric(df2.index, errors="coerce") / 3600.0)
            df2.to_csv(output_dir / f"link_{key}_wide.csv", encoding="utf-8-sig")

    return node_results, link_results, exported


def export_water_age_results(results: Any, output_dir: Path) -> pd.DataFrame:
    quality_df = get_result_dataframe(results.node, "quality")
    if quality_df is None or quality_df.empty:
        empty = pd.DataFrame(
            columns=[
                "time_s",
                "time_h",
                "node_id",
                "water_age_s",
                "water_age_h",
                "note",
            ]
        )
        safe_to_csv(empty, output_dir / "water_age_by_node_long.csv")
        return empty

    out = quality_df.copy()
    out.index.name = "time_s"
    out = out.reset_index()
    out["time_h"] = pd.to_numeric(out["time_s"], errors="coerce") / 3600.0

    # En EPANET, quality=AGE se expresa en segundos. WNTR conserva ese valor.
    # Para comparar con la hoja de tiempos de recorrido, convertimos a horas.
    long_df = out.melt(id_vars=["time_s", "time_h"], var_name="node_id", value_name="water_age_s")
    long_df["water_age_s"] = pd.to_numeric(long_df["water_age_s"], errors="coerce")
    long_df["water_age_h"] = long_df["water_age_s"] / 3600.0
    long_df["note"] = "EPANET/WNTR quality=AGE; water_age_s en segundos; water_age_h en horas"

    safe_to_csv(long_df, output_dir / "water_age_by_node_long.csv")

    # Resumen por nodo: último valor y promedio de las últimas 24 h.
    max_time = long_df["time_h"].max()
    last_df = long_df.loc[long_df["time_h"] == max_time, ["node_id", "water_age_s", "water_age_h"]].copy()
    last_df = last_df.rename(
        columns={
            "water_age_s": "water_age_s_final",
            "water_age_h": "water_age_h_final",
        }
    )

    if pd.notna(max_time):
        last24 = long_df[long_df["time_h"] >= max_time - 24.0]
    else:
        last24 = long_df

    mean24 = (
        last24.groupby("node_id", as_index=False)[["water_age_s", "water_age_h"]]
        .mean()
        .rename(
            columns={
                "water_age_s": "water_age_s_mean_last_24h",
                "water_age_h": "water_age_h_mean_last_24h",
            }
        )
    )

    summary = last_df.merge(mean24, on="node_id", how="outer")
    safe_to_csv(summary, output_dir / "water_age_by_node_summary.csv")

    return long_df


def read_reference_times(xlsx_path: Optional[Path]) -> pd.DataFrame:
    if xlsx_path is None or not xlsx_path.exists():
        return pd.DataFrame()

    # El fichero recibido tiene la hoja TIEMPOS_RECORRIDO. Si en otra versión no existe,
    # pandas lanzará error y se usará la primera hoja disponible.
    try:
        df = pd.read_excel(xlsx_path, sheet_name="TIEMPOS_RECORRIDO", engine="openpyxl")
    except ValueError:
        df = pd.read_excel(xlsx_path, sheet_name=0, engine="openpyxl")

    df = df.dropna(how="all").copy()
    df.columns = [str(c).replace("\xa0", " ").strip() for c in df.columns]

    rename = {
        "ORDEN": "orden",
        "ORDEN ": "orden",
        "RED": "red",
        "CÓDIGO SINAC DEPÓSITO": "codigo_sinac_deposito",
        "CÓDIGO SINAC RED": "codigo_sinac_red",
        "Volumen suministrado (m3/d)": "volumen_suministrado_m3d",
        "Volumen regulado municipal (m3)": "volumen_regulado_municipal_m3",
        "Tiempo de recorrido (h)": "tiempo_recorrido_h",
        "Tiempo almacenamiento depósito municipal (h)": "tiempo_almacenamiento_deposito_municipal_h",
        "Tiempo almacenamiento Regulador (h)": "tiempo_almacenamiento_regulador_h",
        "Tiempo total (h)": "tiempo_total_h",
        "Tiempo total (d)": "tiempo_total_d",
    }
    df = df.rename(columns={c: rename.get(c, c) for c in df.columns})

    numeric_cols = [
        "orden",
        "codigo_sinac_deposito",
        "codigo_sinac_red",
        "volumen_suministrado_m3d",
        "volumen_regulado_municipal_m3",
        "tiempo_recorrido_h",
        "tiempo_almacenamiento_deposito_municipal_h",
        "tiempo_almacenamiento_regulador_h",
        "tiempo_total_h",
        "tiempo_total_d",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].map(clean_number)

    if "red" in df.columns:
        df["red_normalizada"] = df["red"].map(normalize_text)
    else:
        df["red"] = ""
        df["red_normalizada"] = ""

    return df



def read_mapping_csv(mapping_csv: Optional[Path]) -> Dict[str, str]:
    """Compatibilidad: lee un CSV red,node_id y lo combina con el mapping interno."""
    mapping, _sources = read_reference_node_mapping(
        mapping_csv=mapping_csv,
        mapping_xlsx=None,
        output_dir=None,
        available_nodes=None,
    )
    return mapping


def find_column_by_keywords(columns: Iterable[str], required_keywords: Iterable[str]) -> Optional[str]:
    required = [normalize_text(k) for k in required_keywords]
    for col in columns:
        norm = normalize_text(col)
        if all(k in norm for k in required):
            return col
    return None



def read_mapping_xlsx_rows(mapping_xlsx: Optional[Path]) -> pd.DataFrame:
    columns = [
        "red",
        "red_normalizada",
        "node_id",
        "cod_esquema_pdf",
        "mapping_status",
        "mapping_source",
    ]
    if mapping_xlsx is None:
        return pd.DataFrame(columns=columns)
    if not mapping_xlsx.is_file():
        raise FileNotFoundError(f"No existe el Excel de mapeo locked: {mapping_xlsx}")

    workbook = pd.ExcelFile(mapping_xlsx)
    sheet = "Mapeo EPANET" if "Mapeo EPANET" in workbook.sheet_names else workbook.sheet_names[0]
    df = pd.read_excel(mapping_xlsx, sheet_name=sheet, engine="openpyxl")
    df.columns = [clean_text(column) for column in df.columns]

    red_col = (
        find_column_by_keywords(df.columns, ["nombre", "planta"])
        or find_column_by_keywords(df.columns, ["nombre", "tabla"])
        or find_column_by_keywords(df.columns, ["red"])
    )
    node_col = find_column_by_keywords(df.columns, ["nodo", "epanet"]) or find_column_by_keywords(df.columns, ["node"])
    cod_col = find_column_by_keywords(df.columns, ["cod"]) or find_column_by_keywords(df.columns, ["codigo"])
    if red_col is None or node_col is None:
        raise ValueError(
            "No se han detectado las columnas de referencia y nodo EPANET en el Excel. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        red = clean_text(row.get(red_col))
        node_raw = clean_text(row.get(node_col))
        if not red:
            continue
        pending = not node_raw or "PENDIENTE" in normalize_text(node_raw)
        rows.append(
            {
                "red": red,
                "red_normalizada": normalize_text(red),
                "node_id": "" if pending else node_raw,
                "cod_esquema_pdf": clean_text(row.get(cod_col)) if cod_col else "",
                "mapping_status": "pending" if pending else "mapped",
                "mapping_source": "mapping_xlsx_locked",
            }
        )
    return pd.DataFrame(rows, columns=columns)




def read_mapping_csv_rows(mapping_csv: Optional[Path]) -> pd.DataFrame:
    columns = [
        "red",
        "red_normalizada",
        "node_id",
        "cod_esquema_pdf",
        "mapping_status",
        "mapping_source",
    ]
    if mapping_csv is None:
        return pd.DataFrame(columns=columns)
    if not mapping_csv.is_file():
        raise FileNotFoundError(f"No existe el fichero de mapeo: {mapping_csv}")

    df: Optional[pd.DataFrame] = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(mapping_csv, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        df = pd.read_csv(mapping_csv)

    required = {"red", "node_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV de mapeo debe tener columnas {required}. Faltan: {missing}")

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        red = clean_text(row.get("red"))
        node_id = clean_text(row.get("node_id"))
        if not red:
            continue
        pending = not node_id or "PENDIENTE" in normalize_text(node_id)
        rows.append(
            {
                "red": red,
                "red_normalizada": normalize_text(red),
                "node_id": "" if pending else node_id,
                "cod_esquema_pdf": clean_text(
                    row.get("cod_esquema_pdf", row.get("COD esquema/PDF", ""))
                ),
                "mapping_status": "pending" if pending else "mapped",
                "mapping_source": "mapping_csv_manual",
            }
        )
    return pd.DataFrame(rows, columns=columns)




def read_reference_node_mapping(
    mapping_csv: Optional[Path],
    mapping_xlsx: Optional[Path],
    output_dir: Optional[Path] = None,
    available_nodes: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Combina mapeos por prioridad; una fila pending superior invalida el fallback inferior."""
    available_set = (
        {clean_text(node) for node in available_nodes if clean_text(node)}
        if available_nodes is not None
        else None
    )

    source_rows: List[Dict[str, Any]] = []
    source_order = 0

    def append_row(row: Dict[str, Any], priority: int) -> None:
        nonlocal source_order
        source_order += 1
        record = dict(row)
        record["red"] = clean_text(record.get("red"))
        record["red_normalizada"] = normalize_text(
            record.get("red_normalizada") or record.get("red")
        )
        record["node_id"] = clean_text(record.get("node_id"))
        record["mapping_status"] = clean_text(record.get("mapping_status")).lower() or "pending"
        record["priority"] = int(priority)
        record["source_order"] = int(source_order)
        source_rows.append(record)

    for red_norm, node_id in DEFAULT_REFERENCE_NODE_MAPPING.items():
        append_row(
            {
                "red": red_norm,
                "red_normalizada": red_norm,
                "node_id": node_id,
                "cod_esquema_pdf": "",
                "mapping_status": "mapped",
                "mapping_source": "default_internal",
            },
            priority=1,
        )

    for _, row in read_mapping_xlsx_rows(mapping_xlsx).iterrows():
        append_row(row.to_dict(), priority=2)
    for _, row in read_mapping_csv_rows(mapping_csv).iterrows():
        append_row(row.to_dict(), priority=3)

    columns = [
        "red",
        "red_normalizada",
        "node_id",
        "cod_esquema_pdf",
        "mapping_status",
        "mapping_source",
        "priority",
        "source_order",
    ]
    sources = pd.DataFrame(source_rows, columns=columns)
    if sources.empty:
        sources = pd.DataFrame(columns=columns)

    sources["node_exists_in_current_inp"] = False
    if available_set is not None:
        sources["node_exists_in_current_inp"] = sources["node_id"].map(
            lambda node: bool(node) and node in available_set
        )

    # Conflictos en la misma prioridad no deben resolverse por orden accidental.
    mapped_nonempty = sources[
        sources["mapping_status"].eq("mapped") & sources["node_id"].ne("")
    ]
    conflict_rows: List[pd.DataFrame] = []
    for (_, priority), group in mapped_nonempty.groupby(["red_normalizada", "priority"]):
        if group["node_id"].nunique() > 1:
            conflict_rows.append(group)
    conflicts = pd.concat(conflict_rows, ignore_index=True) if conflict_rows else pd.DataFrame()
    if not conflicts.empty:
        if output_dir is not None:
            safe_to_csv(conflicts, output_dir / "reference_node_mapping_conflicts.csv")
        raise ValueError(
            "Hay referencias con más de un nodo distinto en la misma prioridad de mapeo. "
            "Revisa reference_node_mapping_conflicts.csv."
        )

    choices = (
        sources.sort_values(
            ["red_normalizada", "priority", "source_order"],
            ascending=[True, False, False],
        )
        .drop_duplicates("red_normalizada", keep="first")
        .copy()
    )
    choices["is_effective_choice"] = True
    valid_effective = choices["mapping_status"].eq("mapped") & choices["node_id"].ne("")
    if available_set is not None:
        valid_effective &= choices["node_exists_in_current_inp"]
    choices["is_effective_mapping"] = valid_effective

    effective = choices[choices["is_effective_mapping"]].copy()
    mapping = dict(zip(effective["red_normalizada"], effective["node_id"]))

    sources = sources.merge(
        choices[
            [
                "red_normalizada",
                "source_order",
                "node_id",
                "mapping_status",
                "mapping_source",
                "is_effective_choice",
                "is_effective_mapping",
            ]
        ].rename(
            columns={
                "source_order": "effective_source_order",
                "node_id": "effective_node_id",
                "mapping_status": "effective_mapping_status",
                "mapping_source": "effective_mapping_source",
            }
        ),
        on="red_normalizada",
        how="left",
    )
    sources["is_effective_choice"] = (
        sources["source_order"] == sources["effective_source_order"]
    )
    sources["is_effective_mapping"] = (
        sources["is_effective_choice"]
        & sources["effective_mapping_status"].eq("mapped")
        & sources["effective_node_id"].ne("")
    )
    if available_set is not None:
        sources["is_effective_mapping"] &= sources["effective_node_id"].map(
            lambda node: bool(node) and node in available_set
        )

    if output_dir is not None:
        safe_to_csv(sources, output_dir / "reference_node_mapping_sources.csv")
        safe_to_csv(effective, output_dir / "reference_node_mapping_effective.csv")
        pending_choices = choices[~choices["is_effective_mapping"]].copy()
        safe_to_csv(pending_choices, output_dir / "reference_node_mapping_pending_or_invalid.csv")

    return mapping, sources



def map_reference_to_node(red_normalizada: str, mapping: Dict[str, str], available_nodes: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    available = set(str(n) for n in available_nodes)
    if not red_normalizada:
        return None, None

    # Coincidencia exacta primero.
    if red_normalizada in mapping and mapping[red_normalizada] in available:
        return mapping[red_normalizada], red_normalizada

    # Coincidencia por alias priorizados.
    for alias in DEFAULT_MAPPING_PRIORITY:
        alias_norm = normalize_text(alias)
        node_id = mapping.get(alias_norm)
        if node_id and node_id in available and alias_norm in red_normalizada:
            return node_id, alias_norm

    # Coincidencia general por longitud descendente para evitar alias demasiado cortos.
    for alias_norm in sorted(mapping.keys(), key=len, reverse=True):
        node_id = mapping[alias_norm]
        if node_id in available and alias_norm and alias_norm in red_normalizada:
            return node_id, alias_norm

    return None, None


def create_mapping_template(reference_df: pd.DataFrame, nodes_df: pd.DataFrame, output_dir: Path) -> None:
    if reference_df.empty:
        return

    known_nodes = nodes_df[["node_id", "node_type", "inp_comment"]].copy() if not nodes_df.empty else pd.DataFrame()
    known_nodes["hint"] = known_nodes.apply(
        lambda r: f"{r.get('node_id', '')} | {r.get('node_type', '')} | {r.get('inp_comment', '')}", axis=1
    ) if not known_nodes.empty else ""

    template = reference_df[["red", "red_normalizada"]].drop_duplicates().copy()
    template["node_id"] = ""
    template["comentario"] = "Rellenar node_id si se conoce. Columnas obligatorias para --mapping-csv: red,node_id"
    safe_to_csv(template, output_dir / "node_mapping_template_to_edit.csv")

    if not known_nodes.empty:
        safe_to_csv(known_nodes, output_dir / "available_model_nodes_for_mapping.csv")


def write_demand_mapping_suggestions(
    reference_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 5,
) -> None:
    """
    Genera candidatos de mapeo comparando:
      - volumen_suministrado_m3d / 86.4 = demanda equivalente en L/s
      - base_demand_Lps de cada nodo EPANET

    Es una ayuda para revisión manual. No sustituye al criterio hidráulico/topológico.
    """
    if reference_df.empty or nodes_df.empty:
        return
    if "volumen_suministrado_m3d" not in reference_df.columns or "base_demand_Lps" not in nodes_df.columns:
        return

    positive_nodes = nodes_df.copy()
    positive_nodes["base_demand_Lps"] = pd.to_numeric(positive_nodes["base_demand_Lps"], errors="coerce")
    positive_nodes = positive_nodes[positive_nodes["base_demand_Lps"] > 0].copy()

    if positive_nodes.empty:
        return

    records: List[Dict[str, Any]] = []
    for _, row in reference_df.iterrows():
        volume = clean_number(row.get("volumen_suministrado_m3d"))
        if volume is None:
            continue

        target_lps = volume / 86.4
        candidates = positive_nodes.copy()
        candidates["abs_diff_Lps"] = (candidates["base_demand_Lps"] - target_lps).abs()
        candidates["rel_diff_pct"] = np.where(
            target_lps > 0,
            100.0 * candidates["abs_diff_Lps"] / target_lps,
            np.nan,
        )
        candidates = candidates.sort_values(["abs_diff_Lps", "node_id"]).head(top_n)

        for rank, cand in enumerate(candidates.to_dict("records"), start=1):
            records.append(
                {
                    "orden": row.get("orden"),
                    "red": row.get("red"),
                    "red_normalizada": row.get("red_normalizada"),
                    "volumen_suministrado_m3d": volume,
                    "demanda_referencia_Lps": target_lps,
                    "candidate_rank": rank,
                    "candidate_node_id": cand.get("node_id"),
                    "candidate_comment": cand.get("inp_comment", ""),
                    "candidate_base_demand_Lps": cand.get("base_demand_Lps"),
                    "abs_diff_Lps": cand.get("abs_diff_Lps"),
                    "rel_diff_pct": cand.get("rel_diff_pct"),
                }
            )

    if records:
        suggestions = pd.DataFrame(records)
        safe_to_csv(suggestions, output_dir / "mapping_suggestions_by_demand.csv")



def compare_reference_times(
    reference_df: pd.DataFrame,
    water_age_long_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    mapping: Dict[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    if reference_df.empty:
        empty = pd.DataFrame()
        safe_to_csv(empty, output_dir / "comparison_travel_times.csv")
        return empty

    available_nodes = set(nodes_df["node_id"].astype(str)) if not nodes_df.empty else set()
    ref = reference_df.copy()

    mapped_pairs = ref["red_normalizada"].map(lambda x: map_reference_to_node(str(x), mapping, available_nodes))
    ref["mapped_node_id"] = mapped_pairs.map(lambda x: x[0])
    ref["matched_alias"] = mapped_pairs.map(lambda x: x[1])

    # Resumen de water age por nodo.
    if not water_age_long_df.empty:
        max_time = water_age_long_df["time_h"].max()
        final_age = water_age_long_df[water_age_long_df["time_h"] == max_time][["node_id", "water_age_h"]].copy()
        final_age = final_age.rename(columns={"water_age_h": "sim_water_age_h_final"})

        last24 = water_age_long_df[water_age_long_df["time_h"] >= max_time - 24.0]
        mean24 = (
            last24.groupby("node_id", as_index=False)["water_age_h"]
            .mean()
            .rename(columns={"water_age_h": "sim_water_age_h_mean_last_24h"})
        )
        age_summary = final_age.merge(mean24, on="node_id", how="outer")
    else:
        age_summary = pd.DataFrame(columns=["node_id", "sim_water_age_h_final", "sim_water_age_h_mean_last_24h"])

    comp = ref.merge(age_summary, left_on="mapped_node_id", right_on="node_id", how="left")

    # Comparaciones contra tiempo de recorrido y tiempo total de la tabla.
    for sim_col in ["sim_water_age_h_final", "sim_water_age_h_mean_last_24h"]:
        if sim_col in comp.columns:
            comp[f"diff_vs_tiempo_recorrido_h__{sim_col}"] = comp[sim_col] - comp.get("tiempo_recorrido_h")
            comp[f"diff_vs_tiempo_total_h__{sim_col}"] = comp[sim_col] - comp.get("tiempo_total_h")

    comp["mapping_status"] = np.where(comp["mapped_node_id"].notna(), "mapped", "unmapped")

    safe_to_csv(comp, output_dir / "comparison_travel_times.csv")
    safe_to_csv(comp[comp["mapping_status"] == "unmapped"], output_dir / "unmapped_reference_rows.csv")
    safe_to_csv(comp[comp["mapping_status"] == "mapped"], output_dir / "mapped_reference_rows.csv")

    return comp



def normalize_alert_zone(value: Any) -> str:
    """Normaliza zona de alerta a los códigos internos usados por el pipeline."""
    raw = normalize_text(value)
    raw = raw.replace(".", " ")
    raw = re.sub(r"\s+", " ", raw).strip()
    if raw in ALERT_THRESHOLDS or raw == "UNKNOWN":
        return raw
    if raw in ALERT_ZONE_ALIASES:
        return ALERT_ZONE_ALIASES[raw]
    raw_underscore = raw.replace(" ", "_")
    if raw_underscore in ALERT_THRESHOLDS:
        return raw_underscore
    return "UNKNOWN"



def read_alert_zones_csv(alert_zones_csv: Optional[Path]) -> Dict[str, str]:
    if alert_zones_csv is None:
        return {}
    if not alert_zones_csv.is_file():
        raise FileNotFoundError(f"No existe el CSV de zonas de alerta: {alert_zones_csv}")

    df = pd.read_csv(alert_zones_csv, encoding="utf-8-sig")
    required = {"node_id", "alert_zone"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"El CSV de zonas debe tener columnas {required}. Faltan: {missing}")

    zones: Dict[str, str] = {}
    for _, row in df.iterrows():
        node_id = clean_text(row.get("node_id"))
        zone = normalize_alert_zone(row.get("alert_zone"))
        if node_id:
            zones[node_id] = zone
    return zones





def guess_regulator_seed_nodes(
    nodes_df: pd.DataFrame,
    regulator_node_id: str,
    downstream_seed_node_id: Optional[str] = None,
) -> List[str]:
    """Obtiene nodos de la zona regulador a partir de anclajes explícitos y El Viso."""
    available = {clean_text(node) for node in nodes_df.get("node_id", pd.Series(dtype=str))}
    seeds: List[str] = []
    for configured_id in [regulator_node_id, downstream_seed_node_id]:
        node_id = clean_text(configured_id)
        if node_id and node_id in available:
            seeds.append(node_id)

    for _, row in nodes_df.iterrows():
        node_id = clean_text(row.get("node_id"))
        name_norm = normalize_text(row.get("inp_comment"))
        if node_id and "EL VISO" in name_norm:
            seeds.append(node_id)

    output: List[str] = []
    seen: Set[str] = set()
    for seed in seeds:
        if seed and seed not in seen:
            output.append(seed)
            seen.add(seed)
    return output




def build_flow_directed_links(
    links_df: pd.DataFrame,
    link_results: Optional[pd.DataFrame] = None,
    min_abs_flow_lps: float = 1e-6,
) -> pd.DataFrame:
    """
    Orienta enlaces según flujo medio simulado de las últimas 24 h si está disponible.

    Convención WNTR/EPANET: flowrate positivo va de start_node a end_node.
    Si no hay resultados hidráulicos, se usa la orientación del INP como fallback topológico.
    """
    if links_df.empty:
        return pd.DataFrame()

    links = links_df.copy()
    for col in ["link_id", "start_node", "end_node"]:
        if col not in links.columns:
            links[col] = ""
        links[col] = links[col].astype(str).str.strip()

    flow_summary = pd.DataFrame(columns=["link_id", "mean_flow_m3s", "mean_flow_Lps", "n_flow_samples"])
    if link_results is not None and not link_results.empty:
        lr = link_results.copy()
        if {"variable", "element_id", "value_si"}.issubset(lr.columns):
            flow = lr[lr["variable"].astype(str).eq("flowrate")].copy()
            if not flow.empty:
                flow["value_si"] = pd.to_numeric(flow["value_si"], errors="coerce")
                if "time_h" in flow.columns and flow["time_h"].notna().any():
                    max_h = pd.to_numeric(flow["time_h"], errors="coerce").max()
                    flow = flow[pd.to_numeric(flow["time_h"], errors="coerce") >= max_h - 24.0]
                flow_summary = (
                    flow.groupby("element_id", as_index=False)
                    .agg(mean_flow_m3s=("value_si", "mean"), n_flow_samples=("value_si", "count"))
                    .rename(columns={"element_id": "link_id"})
                )
                flow_summary["link_id"] = flow_summary["link_id"].astype(str).str.strip()
                flow_summary["mean_flow_Lps"] = flow_summary["mean_flow_m3s"] * 1000.0

    directed = links.merge(flow_summary, on="link_id", how="left")
    rows: List[Dict[str, Any]] = []
    for _, row in directed.iterrows():
        start = str(row.get("start_node", "")).strip()
        end = str(row.get("end_node", "")).strip()
        mean_flow_lps = pd.to_numeric(pd.Series([row.get("mean_flow_Lps")]), errors="coerce").iloc[0]
        has_flow = pd.notna(mean_flow_lps)

        if has_flow and abs(float(mean_flow_lps)) > min_abs_flow_lps:
            if float(mean_flow_lps) >= 0:
                from_node, to_node = start, end
                direction = "start_to_end"
            else:
                from_node, to_node = end, start
                direction = "end_to_start"
            method = "hydraulic_mean_flow_last_24h"
            confidence = "high"
        elif has_flow:
            from_node, to_node = "", ""
            direction = "near_zero_or_ambiguous"
            method = "hydraulic_mean_flow_last_24h"
            confidence = "low"
        else:
            from_node, to_node = start, end
            direction = "inp_start_to_end_fallback"
            method = "inp_topology_fallback"
            confidence = "medium_low"

        d = row.to_dict()
        d.update(
            {
                "flow_direction": direction,
                "directed_from_node": from_node,
                "directed_to_node": to_node,
                "direction_method": method,
                "direction_confidence": confidence,
            }
        )
        rows.append(d)

    return pd.DataFrame(rows)


def downstream_nodes_from_directed_links(directed_links_df: pd.DataFrame, seed_nodes: List[str]) -> Set[str]:
    if directed_links_df.empty or not seed_nodes:
        return set()
    adjacency: Dict[str, List[str]] = {}
    for _, row in directed_links_df.iterrows():
        src = str(row.get("directed_from_node", "")).strip()
        dst = str(row.get("directed_to_node", "")).strip()
        if not src or not dst:
            continue
        adjacency.setdefault(src, []).append(dst)

    seeds = [str(s).strip() for s in seed_nodes if str(s).strip()]
    seed_set = set(seeds)
    visited: Set[str] = set()
    queue: deque[str] = deque(seeds)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor in visited or neighbor in seed_set:
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    return visited


def build_alert_thresholds_table() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for zone, thresholds in ALERT_THRESHOLDS.items():
        rows.append(
            {
                "alert_zone": zone,
                "alert_zone_label": ALERT_ZONE_LABELS.get(zone, zone),
                "threshold_source_zone": THRESHOLD_SOURCE_BY_ZONE.get(zone, zone),
                "threshold_source_label": ALERT_ZONE_LABELS.get(THRESHOLD_SOURCE_BY_ZONE.get(zone, zone), THRESHOLD_SOURCE_BY_ZONE.get(zone, zone)),
                "prealert_threshold": thresholds["prealert_threshold"],
                "alert_threshold": thresholds["alert_threshold"],
            }
        )
    return pd.DataFrame(rows).sort_values("alert_zone").reset_index(drop=True)



def locked_references_by_node(mapping_sources_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    if mapping_sources_df is None or mapping_sources_df.empty:
        return {}
    df = mapping_sources_df.copy()
    if "is_effective_mapping" in df.columns:
        df = df[df["is_effective_mapping"].eq(True)]
    elif "mapping_status" in df.columns:
        df = df[df["mapping_status"].astype(str).eq("mapped")]
    if "mapping_source" in df.columns:
        df = df[
            df["mapping_source"].astype(str).isin(
                ["mapping_xlsx_locked", "mapping_csv_manual"]
            )
        ]

    output: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        node_id = clean_text(row.get("node_id"))
        red = clean_text(row.get("red"))
        if node_id and red:
            output.setdefault(node_id, []).append(red)
    return {node: " | ".join(sorted(set(names))) for node, names in output.items()}




def infer_node_alert_zone(
    node_id: str,
    node_name: str,
    explicit_zones: Dict[str, str],
    downstream_nodes: Set[str],
    etap_node_id: str,
    regulator_seed_nodes: List[str],
    locked_refs: Dict[str, str],
) -> Tuple[str, str, str, bool, str]:
    """Devuelve zona, método, confianza, necesidad de revisión y motivo."""
    node_id_clean = clean_text(node_id)
    name_norm = normalize_text(node_name)
    locked_text = locked_refs.get(node_id_clean, "")
    locked_norm = normalize_text(locked_text)
    etap_id = clean_text(etap_node_id)
    regulator_set = set(regulator_seed_nodes)

    if node_id_clean in explicit_zones:
        return (
            explicit_zones[node_id_clean],
            "alert_zones_csv",
            "manual",
            False,
            "Zona fijada manualmente mediante --alert-zones-csv",
        )

    if node_id_clean == etap_id:
        return "ETAP", "etap_explicit_id", "high", False, "Nodo ETAP fijado por --etap-node-id"
    if "DEP CABECERA" in name_norm or "DEP.CABECERA" in name_norm or name_norm == "ETAP":
        return "ETAP", "auto_etap_name", "medium", True, "ETAP inferida por comentario del INP; revisar"

    if node_id_clean in regulator_set:
        return (
            "REGULADOR_EL_VISO",
            "regulator_configured_anchor",
            "high",
            False,
            "Nodo configurado como anclaje de la zona regulador o semilla hidráulica aguas abajo",
        )
    if "EL VISO" in name_norm or "EL VISO" in locked_norm:
        return (
            "REGULADOR_EL_VISO",
            "auto_regulator_el_viso_name_or_mapping",
            "medium",
            True,
            "Regulador El Viso inferido por nombre o mapeo; revisar",
        )

    if node_id_clean in downstream_nodes:
        confidence = "high" if locked_text else "medium"
        return (
            "AGUAS_ABAJO_REGULADOR",
            "auto_flow_downstream",
            confidence,
            True,
            "Nodo alcanzable aguas abajo de la semilla hidráulica configurada; revisar",
        )

    if locked_text:
        return (
            "ENTRE_ETAP_Y_REGULADOR",
            "auto_not_downstream_locked_reference",
            "medium",
            True,
            "Referencia locked no alcanzable aguas abajo; se asigna provisionalmente al tramo 75/80",
        )

    return (
        "ENTRE_ETAP_Y_REGULADOR",
        "auto_fallback_not_downstream",
        "low",
        True,
        "Fallback automático para nodo interno no clasificado; revisar antes de uso operativo",
    )




def build_node_alert_zones(
    nodes_df: pd.DataFrame,
    links_df: pd.DataFrame,
    alert_zones_csv: Optional[Path],
    etap_node_id: str,
    regulator_node_id: str,
    auto_downstream_zones: bool,
    downstream_seed_node_id: Optional[str] = None,
    link_results: Optional[pd.DataFrame] = None,
    mapping_sources_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Construye node_id -> alert_zone usando:
      - zonas manuales si existen,
      - nodo ETAP/cabecera,
      - semillas del regulador/El Viso,
      - descendientes hidráulicos del regulador por flujo medio simulado,
      - mapeo locked para mejorar trazabilidad/confianza.
    """
    explicit_zones = read_alert_zones_csv(alert_zones_csv)
    regulator_seed_nodes = guess_regulator_seed_nodes(
        nodes_df, regulator_node_id, downstream_seed_node_id
    )
    locked_refs = locked_references_by_node(mapping_sources_df)

    available_nodes = {clean_text(node) for node in nodes_df.get("node_id", pd.Series(dtype=str))}
    configured_downstream_seed = clean_text(downstream_seed_node_id)
    if configured_downstream_seed and configured_downstream_seed in available_nodes:
        downstream_seed_nodes = [configured_downstream_seed]
    else:
        downstream_seed_nodes = list(regulator_seed_nodes)

    directed_links_df = build_flow_directed_links(links_df, link_results=link_results)
    downstream_nodes = (
        downstream_nodes_from_directed_links(directed_links_df, downstream_seed_nodes)
        if auto_downstream_zones
        else set()
    )

    if output_dir is not None and not directed_links_df.empty:
        safe_to_csv(directed_links_df, output_dir / "flow_directed_links.csv")

    rows: List[Dict[str, Any]] = []
    for _, row in nodes_df.iterrows():
        node_id = str(row.get("node_id", "")).strip()
        node_name = str(row.get("inp_comment", "")).strip()
        zone, method, confidence, review_needed, reason = infer_node_alert_zone(
            node_id=node_id,
            node_name=node_name,
            explicit_zones=explicit_zones,
            downstream_nodes=downstream_nodes,
            etap_node_id=etap_node_id,
            regulator_seed_nodes=regulator_seed_nodes,
            locked_refs=locked_refs,
        )
        thresholds = ALERT_THRESHOLDS.get(zone, {})
        rows.append(
            {
                "node_id": node_id,
                "node_type": row.get("node_type", ""),
                "node_name": node_name,
                "locked_reference_names": locked_refs.get(node_id, ""),
                "is_locked_reference_node": bool(locked_refs.get(node_id, "")),
                "alert_zone": zone,
                "alert_zone_label": ALERT_ZONE_LABELS.get(zone, "Sin zona asignada"),
                "threshold_source_zone": THRESHOLD_SOURCE_BY_ZONE.get(zone, zone),
                "threshold_source_label": ALERT_ZONE_LABELS.get(THRESHOLD_SOURCE_BY_ZONE.get(zone, zone), THRESHOLD_SOURCE_BY_ZONE.get(zone, zone)),
                "prealert_threshold": thresholds.get("prealert_threshold", np.nan),
                "alert_threshold": thresholds.get("alert_threshold", np.nan),
                "alert_zone_assignment_method": method,
                "alert_zone_confidence": confidence,
                "alert_zone_review_needed": bool(review_needed),
                "alert_zone_reason": reason,
            }
        )

    out = pd.DataFrame(rows)
    review = out[out["alert_zone_review_needed"].eq(True)].copy()
    if output_dir is not None:
        safe_to_csv(review, output_dir / "node_alert_zones_review.csv")

    summary = {
        "regulator_seed_nodes": regulator_seed_nodes,
        "downstream_seed_nodes": downstream_seed_nodes,
        "configured_downstream_seed_node_id": configured_downstream_seed or None,
        "n_downstream_nodes": int(len(downstream_nodes)),
        "downstream_nodes": sorted(downstream_nodes),
        "auto_downstream_zones": bool(auto_downstream_zones),
        "link_direction_method_counts": directed_links_df["direction_method"].value_counts(dropna=False).to_dict() if not directed_links_df.empty and "direction_method" in directed_links_df.columns else {},
        "link_direction_counts": directed_links_df["flow_direction"].value_counts(dropna=False).to_dict() if not directed_links_df.empty and "flow_direction" in directed_links_df.columns else {},
        "zone_counts": out["alert_zone"].value_counts(dropna=False).to_dict(),
        "zone_review_needed_rows": int(out["alert_zone_review_needed"].sum()),
        "locked_reference_nodes_count": int(out["is_locked_reference_node"].sum()),
    }
    if output_dir is not None:
        write_json(output_dir / "zone_inference_summary.json", summary)

    return out.sort_values(["alert_zone", "node_id"]).reset_index(drop=True), directed_links_df, summary


def add_alert_zone_to_nodes(nodes_df: pd.DataFrame, alert_zones_df: pd.DataFrame) -> pd.DataFrame:
    if nodes_df.empty or alert_zones_df.empty:
        return nodes_df.copy()
    keep_cols = [
        "node_id",
        "node_name",
        "alert_zone",
        "alert_zone_label",
        "threshold_source_zone",
        "threshold_source_label",
        "prealert_threshold",
        "alert_threshold",
        "alert_zone_assignment_method",
        "alert_zone_confidence",
        "alert_zone_review_needed",
        "alert_zone_reason",
        "locked_reference_names",
        "is_locked_reference_node",
    ]
    return nodes_df.merge(alert_zones_df[[c for c in keep_cols if c in alert_zones_df.columns]], on="node_id", how="left")


def classify_thm_alert_value(thm_value: Any, alert_zone: Any) -> str:
    """
    Clasifica un valor THM según zona:
      normal / prealerta / alerta / unknown_zone / missing_value
    """
    value = clean_number(thm_value)
    if value is None:
        return "missing_value"

    zone = normalize_alert_zone(alert_zone)
    thresholds = ALERT_THRESHOLDS.get(zone)
    if thresholds is None:
        return "unknown_zone"

    if value >= thresholds["alert_threshold"]:
        return "alerta"
    if value >= thresholds["prealert_threshold"]:
        return "prealerta"
    return "normal"



def classify_thm_alerts_dataframe(
    thm_df: pd.DataFrame,
    alert_zones_df: pd.DataFrame,
    node_col: str,
    thm_col: str,
) -> pd.DataFrame:
    if node_col not in thm_df.columns:
        raise ValueError(f"El CSV de THM no contiene la columna de nodo indicada: {node_col}")
    if thm_col not in thm_df.columns:
        raise ValueError(f"El CSV de THM no contiene la columna THM indicada: {thm_col}")

    out = thm_df.copy()
    zones = alert_zones_df[
        [
            "node_id",
            "node_name",
            "alert_zone",
            "alert_zone_label",
            "threshold_source_zone",
            "threshold_source_label",
            "prealert_threshold",
            "alert_threshold",
            "alert_zone_assignment_method",
            "alert_zone_review_needed",
        ]
    ].copy()

    out[node_col] = out[node_col].map(clean_text)
    out = out.merge(zones, left_on=node_col, right_on="node_id", how="left")
    out["alert_zone"] = out["alert_zone"].fillna("UNKNOWN")
    out["alert_level"] = out.apply(
        lambda row: classify_thm_alert_value(row.get(thm_col), row.get("alert_zone")),
        axis=1,
    )
    out["thm_margin_to_prealert"] = (
        pd.to_numeric(out[thm_col], errors="coerce")
        - pd.to_numeric(out["prealert_threshold"], errors="coerce")
    )
    out["thm_margin_to_alert"] = (
        pd.to_numeric(out[thm_col], errors="coerce")
        - pd.to_numeric(out["alert_threshold"], errors="coerce")
    )

    ordered_front = [
        column
        for column in [
            "timestamp",
            node_col,
            "node_name",
            "alert_zone",
            "alert_zone_label",
            "threshold_source_zone",
            "threshold_source_label",
            thm_col,
            "prealert_threshold",
            "alert_threshold",
            "alert_level",
            "thm_margin_to_prealert",
            "thm_margin_to_alert",
        ]
        if column in out.columns
    ]
    rest = [column for column in out.columns if column not in ordered_front]
    return out[ordered_front + rest]



def build_basic_diagnostics(
    nodes_df: pd.DataFrame,
    links_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    hydraulic_output: Optional[SimulationOutput],
    age_output: Optional[SimulationOutput],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "nodes_with_positive_demand": None,
        "total_base_demand_Lps": None,
        "pipes_total_length_km": None,
        "diameter_min_mm": None,
        "diameter_max_mm": None,
        "mapped_reference_rows": None,
        "unmapped_reference_rows": None,
        "hydraulic_simulation_ok": None,
        "water_age_simulation_ok": None,
    }

    if not nodes_df.empty:
        demand = pd.to_numeric(nodes_df.get("base_demand_Lps"), errors="coerce")
        diagnostics["nodes_with_positive_demand"] = int((demand.fillna(0) > 0).sum())
        diagnostics["total_base_demand_Lps"] = float(demand.sum(skipna=True))

    if not links_df.empty:
        length = pd.to_numeric(links_df.get("length_m"), errors="coerce")
        diam = pd.to_numeric(links_df.get("diameter_mm"), errors="coerce")
        diagnostics["pipes_total_length_km"] = float(length.sum(skipna=True) / 1000.0)
        diagnostics["diameter_min_mm"] = float(diam.min(skipna=True)) if diam.notna().any() else None
        diagnostics["diameter_max_mm"] = float(diam.max(skipna=True)) if diam.notna().any() else None

    if not comparison_df.empty and "mapping_status" in comparison_df.columns:
        diagnostics["mapped_reference_rows"] = int((comparison_df["mapping_status"] == "mapped").sum())
        diagnostics["unmapped_reference_rows"] = int((comparison_df["mapping_status"] == "unmapped").sum())

    if hydraulic_output is not None:
        diagnostics["hydraulic_simulation_ok"] = hydraulic_output.ok
        diagnostics["hydraulic_error"] = hydraulic_output.error
    if age_output is not None:
        diagnostics["water_age_simulation_ok"] = age_output.ok
        diagnostics["water_age_error"] = age_output.error

    return diagnostics



def validate_alert_thresholds() -> None:
    for zone, thresholds in ALERT_THRESHOLDS.items():
        prealert = float(thresholds["prealert_threshold"])
        alert = float(thresholds["alert_threshold"])
        if not math.isfinite(prealert) or not math.isfinite(alert) or prealert >= alert:
            raise ValueError(
                f"Umbrales inválidos para {zone}: prealerta={prealert}, alerta={alert}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Auditoría hidráulica base de Huesna con EPANET/WNTR v{SCRIPT_VERSION}"
    )
    parser.add_argument("--inp", required=True, type=Path)
    parser.add_argument("--times-xlsx", type=Path, default=None)
    parser.add_argument("--mapping-csv", type=Path, default=None)
    parser.add_argument("--mapping-xlsx", type=Path, default=None)
    parser.add_argument("--alert-zones-csv", type=Path, default=None)
    parser.add_argument("--etap-node-id", default="1")
    parser.add_argument("--regulator-node-id", default="n50")
    parser.add_argument(
        "--downstream-seed-node-id",
        default="2",
        help="Nodo hidráulico desde el que se recorre la red aguas abajo; en el modelo compartido es el tanque Regulador Huerto Reyes.",
    )
    parser.add_argument("--no-auto-downstream-zones", action="store_true")
    parser.add_argument("--thm-results-csv", type=Path, default=None)
    parser.add_argument("--thm-node-col", default="node_id")
    parser.add_argument("--thm-value-col", default="thm_estimated")
    parser.add_argument("--thm-output-name", default="thm_alerts_classified.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_epanet_baseline"))
    parser.add_argument("--skip-simulation", action="store_true")
    args = parser.parse_args()

    validate_alert_thresholds()
    inp_path = args.inp.resolve()
    times_xlsx = args.times_xlsx.resolve() if args.times_xlsx else None
    mapping_csv = args.mapping_csv.resolve() if args.mapping_csv else None
    mapping_xlsx = args.mapping_xlsx.resolve() if args.mapping_xlsx else None
    alert_zones_csv = args.alert_zones_csv.resolve() if args.alert_zones_csv else None
    thm_results_csv = args.thm_results_csv.resolve() if args.thm_results_csv else None
    output_dir = args.output_dir.resolve()
    ensure_output_dir(output_dir)

    if not inp_path.is_file():
        raise FileNotFoundError(f"No existe el fichero .inp: {inp_path}")
    for label, path in [
        ("--times-xlsx", times_xlsx),
        ("--mapping-csv", mapping_csv),
        ("--mapping-xlsx", mapping_xlsx),
        ("--alert-zones-csv", alert_zones_csv),
        ("--thm-results-csv", thm_results_csv),
    ]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"No existe {label}: {path}")

    etap_node_id = clean_text(args.etap_node_id)
    regulator_node_id = clean_text(args.regulator_node_id)
    downstream_seed_node_id = clean_text(args.downstream_seed_node_id)
    if not etap_node_id or not regulator_node_id or not downstream_seed_node_id:
        raise ValueError(
            "--etap-node-id, --regulator-node-id y --downstream-seed-node-id no pueden estar vacíos"
        )
    if etap_node_id in {regulator_node_id, downstream_seed_node_id}:
        raise ValueError("ETAP no puede coincidir con un anclaje de la zona regulador")
    if Path(args.thm_output_name).name != args.thm_output_name:
        raise ValueError("--thm-output-name debe ser solo un nombre de fichero")

    run_log: List[str] = []

    def log_and_keep(message: str) -> None:
        run_log.append(message)
        log(message)

    log_and_keep("==============================")
    log_and_keep(f"HUESNA EPANET BASELINE v{SCRIPT_VERSION}")
    log_and_keep("==============================")
    log_and_keep(f"INP:        {inp_path}")
    log_and_keep(f"Excel:      {times_xlsx or '(no indicado)'}")
    log_and_keep(f"Salida:     {output_dir}")

    try:
        wntr = import_wntr()
        log_and_keep(f"WNTR:       {getattr(wntr, '__version__', 'versión no disponible')}")
    except Exception as exc:
        log_and_keep(str(exc))
        (output_dir / "run_log.txt").write_text("\n".join(run_log) + "\n", encoding="utf-8")
        return 2

    log_and_keep("\n[1/7] Parseando metadatos del .inp...")
    inp_metadata = parse_inp_metadata(inp_path)

    log_and_keep("[2/7] Cargando WaterNetworkModel...")
    try:
        wn = wntr.network.WaterNetworkModel(str(inp_path))
    except Exception as exc:
        log_and_keep(f"[ERROR] No se pudo cargar el .inp con WNTR: {exc}")
        log_and_keep(traceback.format_exc())
        (output_dir / "run_log.txt").write_text("\n".join(run_log) + "\n", encoding="utf-8")
        return 2

    summary = network_summary(wn, inp_metadata, inp_path)
    write_json(output_dir / "resumen_modelo.json", summary)
    nodes_df = build_nodes_summary(wn, inp_metadata)
    links_df = build_links_summary(wn, inp_metadata)
    named_elements_df = build_named_elements_summary(inp_metadata)
    safe_to_csv(nodes_df, output_dir / "nodes_summary.csv")
    safe_to_csv(links_df, output_dir / "links_summary.csv")
    safe_to_csv(named_elements_df, output_dir / "inp_named_elements.csv")

    available_nodes = {clean_text(node) for node in nodes_df.get("node_id", pd.Series(dtype=str))}
    if etap_node_id not in available_nodes:
        log_and_keep(f"[AVISO] --etap-node-id={etap_node_id} no aparece en el INP; se intentará inferir por nombre.")
    if regulator_node_id not in available_nodes:
        log_and_keep(f"[AVISO] --regulator-node-id={regulator_node_id} no aparece en el INP; se intentará inferir por nombre.")
    if downstream_seed_node_id not in available_nodes:
        log_and_keep(
            f"[AVISO] --downstream-seed-node-id={downstream_seed_node_id} no aparece en el INP; "
            "se usarán los demás anclajes como fallback."
        )

    log_and_keep(
        f"Modelo cargado: {len(nodes_df)} nodos, {len(links_df)} enlaces. "
        f"Elementos con comentario: {len(named_elements_df)}"
    )

    log_and_keep("[3/7] Leyendo tiempos de recorrido y mapeos...")
    reference_df = read_reference_times(times_xlsx)
    if not reference_df.empty:
        safe_to_csv(reference_df, output_dir / "reference_travel_times_clean.csv")
        log_and_keep(f"Filas de referencia leídas: {len(reference_df)}")
    else:
        log_and_keep("No se ha indicado o no contiene datos una tabla de tiempos de recorrido.")

    create_mapping_template(reference_df, nodes_df, output_dir)
    write_demand_mapping_suggestions(reference_df, nodes_df, output_dir)
    mapping, mapping_sources_df = read_reference_node_mapping(
        mapping_csv=mapping_csv,
        mapping_xlsx=mapping_xlsx,
        output_dir=output_dir,
        available_nodes=available_nodes,
    )
    if mapping_xlsx:
        log_and_keep(f"Mapeo locked cargado desde: {mapping_xlsx}")
    if mapping_csv:
        log_and_keep(f"Mapeo manual adicional cargado desde: {mapping_csv}")
    log_and_keep(f"Mapeos efectivos válidos en el INP: {len(mapping)}")

    hydraulic_output: Optional[SimulationOutput] = None
    age_output: Optional[SimulationOutput] = None
    water_age_long_df = pd.DataFrame()
    comparison_df = pd.DataFrame()
    link_results = pd.DataFrame()

    if args.skip_simulation:
        log_and_keep("[4/7] Simulación hidráulica omitida por --skip-simulation.")
        log_and_keep("[5/7] Simulación de edad del agua omitida por --skip-simulation.")
    else:
        log_and_keep("[4/7] Ejecutando simulación hidráulica EPANET...")
        hydraulic_output = run_epanet_simulation(wn, wntr, "hydraulic", output_dir)
        if hydraulic_output.ok:
            node_results, link_results, exported_keys = export_hydraulic_results(
                hydraulic_output.results, output_dir
            )
            write_json(output_dir / "simulation_result_keys.json", exported_keys)
            log_and_keep(
                f"Simulación hidráulica OK. Filas: node={len(node_results)}, link={len(link_results)}"
            )
        else:
            log_and_keep(f"[ERROR] {hydraulic_output.error}")
            if hydraulic_output.traceback_text:
                (output_dir / "hydraulic_simulation_error.txt").write_text(
                    hydraulic_output.traceback_text, encoding="utf-8"
                )

        log_and_keep("[5/7] Ejecutando simulación de edad del agua...")
        try:
            wn_age = wntr.network.WaterNetworkModel(str(inp_path))
            wn_age.options.quality.parameter = "AGE"
            age_output = run_epanet_simulation(wn_age, wntr, "water_age", output_dir)
        except Exception as exc:
            age_output = SimulationOutput(
                ok=False,
                error=f"Error preparando simulación water age: {exc}",
                traceback_text=traceback.format_exc(),
            )

        if age_output.ok:
            water_age_long_df = export_water_age_results(age_output.results, output_dir)
            log_and_keep(f"Water age OK. Filas exportadas: {len(water_age_long_df)}")
        else:
            log_and_keep(f"[ERROR] {age_output.error}")
            if age_output.traceback_text:
                (output_dir / "water_age_simulation_error.txt").write_text(
                    age_output.traceback_text, encoding="utf-8"
                )

    log_and_keep("[6/7] Comparando con tiempos de recorrido...")
    comparison_df = compare_reference_times(
        reference_df, water_age_long_df, nodes_df, mapping, output_dir
    )
    if not comparison_df.empty:
        mapped = int((comparison_df["mapping_status"] == "mapped").sum())
        unmapped = int((comparison_df["mapping_status"] == "unmapped").sum())
        log_and_keep(f"Comparación generada. Mapeadas={mapped}, sin mapear={unmapped}")
    else:
        log_and_keep("No se generó comparación por falta de tiempos de referencia o resultados de edad.")

    log_and_keep("[7/7] Inferiendo zonas de alerta...")
    alert_zones_df, _directed_links_df, zone_summary = build_node_alert_zones(
        nodes_df=nodes_df,
        links_df=links_df,
        alert_zones_csv=alert_zones_csv,
        etap_node_id=etap_node_id,
        regulator_node_id=regulator_node_id,
        auto_downstream_zones=not args.no_auto_downstream_zones,
        downstream_seed_node_id=downstream_seed_node_id,
        link_results=link_results,
        mapping_sources_df=mapping_sources_df,
        output_dir=output_dir,
    )
    alert_thresholds_df = build_alert_thresholds_table()
    nodes_with_alert_zones_df = add_alert_zone_to_nodes(nodes_df, alert_zones_df)
    safe_to_csv(alert_thresholds_df, output_dir / "alert_thresholds_by_zone.csv")
    safe_to_csv(alert_zones_df, output_dir / "node_alert_zones.csv")
    safe_to_csv(nodes_with_alert_zones_df, output_dir / "nodes_summary_with_alert_zones.csv")

    log_and_keep(
        "Zonas generadas: "
        + ", ".join(
            f"{zone}={count}"
            for zone, count in alert_zones_df["alert_zone"].value_counts(dropna=False).sort_index().items()
        )
    )
    review_count = int(alert_zones_df["alert_zone_review_needed"].sum())
    if review_count:
        log_and_keep(
            f"[AVISO] {review_count} nodos requieren revisión: "
            f"{output_dir / 'node_alert_zones_review.csv'}"
        )

    if not comparison_df.empty:
        zone_columns = [
            "node_id",
            "alert_zone",
            "alert_zone_label",
            "threshold_source_zone",
            "threshold_source_label",
            "prealert_threshold",
            "alert_threshold",
            "alert_zone_assignment_method",
            "alert_zone_confidence",
            "alert_zone_review_needed",
            "alert_zone_reason",
        ]
        comp_with_zone = comparison_df.merge(
            alert_zones_df[zone_columns].rename(columns={"node_id": "mapped_node_id"}),
            on="mapped_node_id",
            how="left",
        )
        safe_to_csv(comp_with_zone, output_dir / "comparison_travel_times_with_alert_zones.csv")
        safe_to_csv(
            comp_with_zone[comp_with_zone["mapping_status"].eq("mapped")],
            output_dir / "mapped_reference_rows_with_alert_zone.csv",
        )
        safe_to_csv(
            comp_with_zone[comp_with_zone["mapping_status"].eq("unmapped")],
            output_dir / "unmapped_reference_rows_with_alert_zone.csv",
        )

    if thm_results_csv:
        thm_df = pd.read_csv(thm_results_csv, encoding="utf-8-sig")
        thm_alerts_df = classify_thm_alerts_dataframe(
            thm_df,
            alert_zones_df,
            args.thm_node_col,
            args.thm_value_col,
        )
        thm_output_path = output_dir / args.thm_output_name
        safe_to_csv(thm_alerts_df, thm_output_path)
        alert_counts = {
            str(key): int(value)
            for key, value in thm_alerts_df["alert_level"].value_counts(dropna=False).to_dict().items()
        }
        write_json(
            output_dir / "thm_alerts_summary.json",
            {
                "input_csv": str(thm_results_csv),
                "output_csv": str(thm_output_path),
                "node_col": args.thm_node_col,
                "thm_value_col": args.thm_value_col,
                "alert_level_counts": alert_counts,
            },
        )
        log_and_keep(f"Clasificación THM exportada: {thm_output_path}")

    diagnostics = build_basic_diagnostics(
        nodes_df, links_df, comparison_df, hydraulic_output, age_output
    )
    diagnostics.update(
        {
            "alert_thresholds_by_zone": ALERT_THRESHOLDS,
            "threshold_source_by_zone": THRESHOLD_SOURCE_BY_ZONE,
            "alert_zone_counts": {
                str(key): int(value)
                for key, value in alert_zones_df["alert_zone"].value_counts(dropna=False).to_dict().items()
            },
            "alert_zone_review_needed_rows": review_count,
            "alert_zones_csv": str(alert_zones_csv) if alert_zones_csv else None,
            "mapping_xlsx": str(mapping_xlsx) if mapping_xlsx else None,
            "mapping_csv": str(mapping_csv) if mapping_csv else None,
            "etap_node_id": etap_node_id,
            "regulator_node_id": regulator_node_id,
            "downstream_seed_node_id": downstream_seed_node_id,
            "auto_downstream_zones": not args.no_auto_downstream_zones,
            "zone_inference_summary": zone_summary,
            "regulator_anchor_note": (
                "n50 representa El Viso; el nodo 2 se usa explícitamente como semilla hidráulica "
                "aguas abajo. No se infiere equivalencia solo por el nombre del comentario INP."
            ),
        }
    )
    write_json(output_dir / "diagnostico_baseline.json", diagnostics)

    return_code = 0
    if diagnostics.get("hydraulic_simulation_ok") is False or diagnostics.get("water_age_simulation_ok") is False:
        log_and_keep("[AVISO] Alguna simulación falló; revisa *_simulation_error.txt.")
        return_code = 1

    log_and_keep("Finalizado.")
    log_and_keep(f"Revisión de zonas pendiente: {review_count} nodos")
    (output_dir / "run_log.txt").write_text("\n".join(run_log) + "\n", encoding="utf-8")

    output_files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    if "manifest_outputs.json" not in output_files:
        output_files.append("manifest_outputs.json")
        output_files.sort()
    write_json(
        output_dir / "manifest_outputs.json",
        {
            "script_version": SCRIPT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "output_dir": str(output_dir),
            "files": output_files,
        },
    )
    return return_code



if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise
