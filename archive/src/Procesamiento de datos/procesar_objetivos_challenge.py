import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astroquery.ipac.irsa import Irsa
from astroquery.simbad import Simbad

from generador_interpolacion import generar_datos_interpolados
from pipeline_spherex_vectorizado import ejecutar_pipeline_completo
from test_identificacion_gaia_alerce import (
    bautizar_fuentes_con_gaia,
    consultar_clasificacion_alerce,
)

OBJETIVOS_CHALLENGE = [
    {
        "nombre_display": "Barnard's Star",
        "slug": "barnards_star",
        "query_simbad": "Barnard's Star",
        "ra_j2000": 269.452076,
        "dec_j2000": 4.693365,
        "pmra_mas_yr": -801.55,
        "pmdec_mas_yr": 10362.39,
        "perfil_espectral": "DEFAULT",
        "fov_deg": 0.04,
        "descripcion": "Estrella enana roja con el mayor movimiento propio conocido (~10.3''/año).",
    },
    {
        "nombre_display": "Epsilon Indi B",
        "slug": "epsilon_indi_b",
        "query_simbad": "eps Ind B",
        "ra_j2000": 331.076250,
        "dec_j2000": -56.782500,
        "pmra_mas_yr": 3960.93,
        "pmdec_mas_yr": -2536.19,
        "perfil_espectral": "YSO",
        "fov_deg": 0.04,
        "descripcion": "Par binario de enanas marrones orbitando a Epsilon Indi.",
    },
    {
        "nombre_display": "WISE 0855",
        "slug": "wise_0855",
        "query_simbad": None,
        "coord_exacta_str": "08h55m10.831s -07d14m42.53s",
        "ra_j2000": 133.795129,
        "dec_j2000": -7.245147,
        "pmra_mas_yr": -8118.0,
        "pmdec_mas_yr": 680.0,
        "perfil_espectral": "YSO",
        "fov_deg": 0.05,
        "descripcion": "Enana marrón tipo Y ultr fría (~250 K) de movimiento ultrarrápido.",
    },
    {
        "nombre_display": "RW Aurigae",
        "slug": "rw_aurigae",
        "query_simbad": "RW Aur",
        "ra_j2000": 76.956529,
        "dec_j2000": 30.401442,
        "pmra_mas_yr": 4.5,
        "pmdec_mas_yr": -24.1,
        "perfil_espectral": "YSO",
        "fov_deg": 0.04,
        "descripcion": "Estrella variable joven tipo T Tauri con episodios de oscurecimiento por polvo.",
    },
    {
        "nombre_display": "R Draconis",
        "slug": "r_draconis",
        "query_simbad": "R Dra",
        "ra_j2000": 248.165033,
        "dec_j2000": 66.755344,
        "pmra_mas_yr": -14.2,
        "pmdec_mas_yr": 22.8,
        "perfil_espectral": "DEFAULT",
        "fov_deg": 0.04,
        "descripcion": "Estrella gigante roja variable pulsante tipo Mira de largo período.",
    },
]


def resolver_coordenadas_2026(obj_cfg: dict) -> tuple[float, float]:
    """Propaga las coordenadas J2000 hasta el año 2026 usando su movimiento propio."""
    if obj_cfg.get("coord_exacta_str"):
        c_base = SkyCoord(obj_cfg["coord_exacta_str"], unit=(u.hourangle, u.deg), frame="icrs")
        ra_0, dec_0 = float(c_base.ra.deg), float(c_base.dec.deg)
    else:
        ra_0, dec_0 = float(obj_cfg["ra_j2000"]), float(obj_cfg["dec_j2000"])

    c_con_mov = SkyCoord(
        ra=ra_0 * u.deg,
        dec=dec_0 * u.deg,
        pm_ra_cosdec=obj_cfg.get("pmra_mas_yr", 0.0) * u.mas / u.yr,
        pm_dec=obj_cfg.get("pmdec_mas_yr", 0.0) * u.mas / u.yr,
        obstime=Time("J2000"),
    )
    c_2026 = c_con_mov.apply_space_motion(new_obstime=Time("2026-06-01"))
    return float(c_2026.ra.deg), float(c_2026.dec.deg)


def enriquecer_nombres_simbad(resultado: dict, nombre_objetivo: str, ra_obj: float, dec_obj: float) -> dict:
    """Asigna el nombre común del objeto central y consulta nombres limpios en SIMBAD."""
    c_obj = SkyCoord(ra=ra_obj * u.deg, dec=dec_obj * u.deg, frame="icrs")
    centro = resultado["centro_campo"]

    # Consultar catálogo SIMBAD en la región para reemplazar IDs largos por nombres legibles
    tabla_simbad = None
    try:
        tabla_simbad = Simbad.query_region(
            SkyCoord(ra=centro["ra"] * u.deg, dec=centro["dec"] * u.deg, frame="icrs"),
            radius=centro["fov_deg"] * 0.75 * u.deg,
        )
    except Exception:
        tabla_simbad = None

    coords_simbad = None
    if tabla_simbad is not None and len(tabla_simbad) > 0:
        col_ra = "ra" if "ra" in tabla_simbad.colnames else "RA"
        col_dec = "dec" if "dec" in tabla_simbad.colnames else "DEC"
        coords_simbad = SkyCoord(ra=tabla_simbad[col_ra], dec=tabla_simbad[col_dec], unit=(u.deg, u.deg), frame="icrs")

    for ep in resultado["epochs"]:
        for fuente in ep.get("catalogo_campo", []):
            c_f = SkyCoord(ra=fuente["ra"] * u.deg, dec=fuente["dec"] * u.deg, frame="icrs")
            # 1. Si está a menos de 12'' de la posición 2026 del objetivo, bautizarlo con su nombre oficial
            if c_f.separation(c_obj).arcsec <= 12.0:
                fuente["nombre_oficial"] = nombre_objetivo
                fuente["clasificacion_objeto"] = f"Objetivo Principal ({nombre_objetivo})"
                ep["objeto_objetivo"] = fuente
                continue

            # 2. Intentar traducir el Gaia ID largo a un nombre legible de SIMBAD
            if coords_simbad is not None:
                idx, sep2d, _ = match_coordinates_sky(c_f, coords_simbad)
                if float(np.asarray(sep2d.arcsec).item()) <= 4.0:
                    col_id = "main_id" if "main_id" in tabla_simbad.colnames else "MAIN_ID"
                    nombre_limpio = str(tabla_simbad[int(np.asarray(idx).item())][col_id]).strip()
                    if nombre_limpio and not nombre_limpio.startswith("Gaia DR3"):
                        fuente["nombre_oficial"] = nombre_limpio
                        continue

            # 3. Acortar el ID de Gaia de 19 dígitos a los últimos 6 para que no sature el mapa
            if fuente.get("nombre_oficial", "").startswith("Gaia DR3 "):
                digitos = fuente["nombre_oficial"].replace("Gaia DR3 ", "")
                fuente["nombre_oficial"] = f"Gaia #{digitos[-6:]}"

    return resultado


def consultar_cuadros_irsa_seguro(ra: float, dec: float, radio_deg: float = 0.03, timeout_sec: int = 25) -> list[dict]:
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    for col in ["spherex_qr3", "spherex_dr1"]:
        print(f"[IRSA SIA2] Buscando cobertura en '{col}' para RA={ra:.5f}, Dec={dec:.5f}...")
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(Irsa.query_sia, pos=(coord, radio_deg * u.deg), collection=col)
        try:
            tab = fut.result(timeout=timeout_sec)
            if hasattr(tab, "to_table"):
                tab = tab.to_table()
            if len(tab) == 0:
                continue

            cols = tab.colnames
            cuadros = []
            for row in tab:
                url = str(row["access_url"])
                m = re.search(r"2D(\d)", url) or re.search(r"/(\d)/level2", url)
                banda = f"SPHEREx-D{m.group(1)}" if m else "SPHEREx"
                mjd = float(row["t_min"]) if "t_min" in cols else 61180.0
                em_min = float(row["em_min"]) * 1e6 if "em_min" in cols else 0.75
                em_max = float(row["em_max"]) * 1e6 if "em_max" in cols else 1.10

                cuadros.append({
                    "fecha_utc": str(mjd),
                    "mjd": mjd,
                    "lambda_um": round((em_min + em_max) / 2.0, 4),
                    "banda": banda,
                    "url": url,
                })

            print(f"[IRSA SIA2] Encontrados {len(cuadros)} cuadros disponibles en '{col}'.")
            return cuadros
        except FuturesTimeoutError:
            print(f"[IRSA SIA2] Timeout ({timeout_sec}s) consultando '{col}'.")
        except Exception as exc:
            print(f"[IRSA SIA2] Error en '{col}': {exc}")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return []


def procesar_un_objetivo(obj_cfg: dict, dir_salida: Path) -> dict | None:
    nombre = obj_cfg["nombre_display"]
    slug = obj_cfg["slug"]
    print("\n" + "=" * 75)
    print(f"PROCESANDO OBJETIVO DEL CHALLENGE: {nombre}")
    print("=" * 75)

    ra, dec = resolver_coordenadas_2026(obj_cfg)
    print(f"[Coordenadas 2026] RA = {ra:.6f} deg | Dec = {dec:.6f} deg")

    cuadros = consultar_cuadros_irsa_seguro(ra, dec, radio_deg=0.03)
    if not cuadros:
        print(f"[ADVERTENCIA] IRSA no devolvió cuadros para {nombre}.")
        return None

    ruta_in = dir_salida / f"input_{slug}.json"
    with open(ruta_in, "w", encoding="utf-8") as f:
        json.dump({"nombre_objeto": nombre, "ra": ra, "dec": dec, "fov_deg": obj_cfg["fov_deg"], "cuadros": cuadros}, f, indent=2)

    info_alerce = consultar_clasificacion_alerce(ra, dec, radio_arcsec=6.0)
    clase_usar = info_alerce["clase"] if info_alerce["oid"] else obj_cfg["perfil_espectral"]

    resultado_pipeline = ejecutar_pipeline_completo(str(ruta_in), clase_alerce=clase_usar)
    resultado_pipeline["nombre_objeto"] = nombre
    resultado_pipeline["descripcion_cientifica"] = obj_cfg["descripcion"]
    resultado_pipeline["alerce_info"] = info_alerce

    resultado_bautizado = bautizar_fuentes_con_gaia(resultado_pipeline, radio_match_arcsec=4.0)
    resultado_limpio = enriquecer_nombres_simbad(resultado_bautizado, nombre, ra, dec)
    resultado_final = generar_datos_interpolados(resultado_limpio)

    ruta_out = dir_salida / f"final_{slug}.json"
    with open(ruta_out, "w", encoding="utf-8") as f:
        json.dump(resultado_final, f, indent=2)

    print(f"[OK] {nombre} procesado y guardado en: {ruta_out.name}")
    return {"nombre": nombre, "slug": slug, "ra": ra, "dec": dec, "archivo_json": ruta_out.name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procesador de los 5 objetivos oficiales del Challenge SPHEREx")
    parser.add_argument("--objeto", type=str, default=None, help="Ej. 'Barnard', 'WISE', 'RW Aurigae'")
    args = parser.parse_args()

    dir_actual = Path(__file__).resolve().parent
    carpeta_challenge = dir_actual / "resultados_challenge"
    carpeta_challenge.mkdir(exist_ok=True)

    objetivos_a_correr = OBJETIVOS_CHALLENGE
    if args.objeto:
        filtro = args.objeto.lower()
        objetivos_a_correr = [o for o in OBJETIVOS_CHALLENGE if filtro in o["nombre_display"].lower() or filtro in o["slug"]]

    resumen_maestro = []
    for obj in objetivos_a_correr:
        res = procesar_un_objetivo(obj, carpeta_challenge)
        if res:
            resumen_maestro.append(res)

    ruta_indice = carpeta_challenge / "indice_objetivos_challenge.json"
    with open(ruta_indice, "w", encoding="utf-8") as f:
        json.dump({"objetivos": resumen_maestro}, f, indent=2)