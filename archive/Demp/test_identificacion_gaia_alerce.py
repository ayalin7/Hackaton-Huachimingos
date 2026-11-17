"""
================================================================================
MÓDULO: test_identificacion_gaia_alerce.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================

DESCRIPCIÓN GENERAL:
    Script independiente de validación para enriquecer el catálogo vectorizado
    de SPHEREx con identidad astrofísica real mediante dos APIs externas:
      1. ALeRCE (API REST ZTF): Se consulta ANTES del pipeline para clasificar
         el objeto central (AGN, SN, YSO, etc.) y seleccionar las 6 bandas
         infrarrojas óptimas.
      2. Gaia DR3 (Servicio TAP ESA/VizieR): Se consulta DESPUÉS de extraer las
         fuentes con SEP para bautizar cada estrella del campo con su ID
         oficial ('Gaia DR3 ...'), paralaje y movimiento propio.

USO DESDE TERMINAL:
    python3 test_identificacion_gaia_alerce.py
================================================================================
"""

import json
from pathlib import Path

import numpy as np
import astropy.units as u
import requests
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.gaia import Gaia

# Importamos tu pipeline principal
from pipeline_spherex_vectorizado import (
    ejecutar_pipeline_completo,
    renderizar_control_local,
)


# =====================================================================
# PASO PREVIO: CONSULTA A ALERCE (ANTES DE PROCESAR FITS)
# =====================================================================
def consultar_clasificacion_alerce(ra: float, dec: float, radio_arcsec: float = 5.0) -> dict:
    """
    Consulta la API pública de ALeRCE por cono de búsqueda para obtener el ID
    de ZTF y la clase astrofísica dominante que guiará la selección de bandas.
    """
    url_alerce = "https://api.alerce.online/ztf/v1/objects/"
    params = {
        "ra": ra,
        "dec": dec,
        "radius": radio_arcsec,
        "page_size": 5,
    }

    print(f"[ALeRCE] Buscando transientes/variables en RA={ra:.4f}, Dec={dec:.4f} (r={radio_arcsec}'')...")
    try:
        resp = requests.get(url_alerce, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("items", [])

        if not items:
            print("[ALeRCE] Sin coincidencias en ZTF. Usando perfil espectral 'DEFAULT'.")
            return {"oid": None, "clase": "DEFAULT", "probabilidad": 0.0}

        # Tomar el objeto más cercano / con mayor probabilidad de clasificación
        mejor = items[0]
        oid = mejor.get("oid")
        clase_raw = str(mejor.get("classrf") or mejor.get("classearly") or "DEFAULT").upper()

        # Mapear clases detalladas de ALeRCE a los perfiles infrarrojos de SPHEREx
        mapa_clases = {
            "AGN": "AGN",
            "QSO": "AGN",
            "BLAZAR": "AGN",
            "SNIA": "SN",
            "SNIBC": "SN",
            "SNII": "SN",
            "SLSN": "SN",
            "YSO": "YSO",
            "ASTEROID": "ASTEROID",
        }
        clase_spherex = mapa_clases.get(clase_raw, "DEFAULT")
        prob = float(mejor.get("pclassrf") or 0.0)

        print(f"[ALeRCE] ¡Encontrado! ID={oid} | Clase={clase_raw} -> Perfil SPHEREx='{clase_spherex}' (p={prob:.2f})")
        return {"oid": oid, "clase": clase_spherex, "clase_original": clase_raw, "probabilidad": prob}

    except Exception as exc:
        print(f"[ALeRCE] Advertencia al consultar API ({exc}). Usando 'DEFAULT'.")
        return {"oid": None, "clase": "DEFAULT", "probabilidad": 0.0}


# =====================================================================
# PASO POSTERIOR: COTEJO Y NOMBRAMIENTO CON GAIA DR3
# =====================================================================
def consultar_catalogo_gaia_campo(ra_centro: float, dec_centro: float, fov_deg: float):
    """
    Descarga en una sola consulta TAP todas las estrellas de Gaia DR3 dentro del
    campo de visión (fov_deg) para cruzarlas localmente con las detecciones de SEP.
    """
    radio_deg = (fov_deg * 0.75)  # Cubre toda la diagonal del recorte cuadrado
    print(f"[Gaia DR3] Consultando estrellas de referencia en radio={radio_deg:.3f}°...")

    query = f"""
    SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_centro}, {dec_centro}, {radio_deg})
    )
    AND phot_g_mean_mag < 20.5
    """
    job = Gaia.launch_job(query)
    tabla_gaia = job.get_results()
    print(f"[Gaia DR3] Recibidas {len(tabla_gaia)} fuentes catalogadas.")
    return tabla_gaia

def bautizar_fuentes_con_gaia(resultado_pipeline: dict, radio_match_arcsec: float = 3.5) -> dict:
    """
    Toma el JSON generado por pipeline_spherex_vectorizado.py y asigna nombre oficial
    ('Gaia DR3 ...'), paralaje y movimiento propio a cada fuente detectada.
    """
    centro = resultado_pipeline["centro_campo"]
    tabla_gaia = consultar_catalogo_gaia_campo(centro["ra"], centro["dec"], centro["fov_deg"])

    if len(tabla_gaia) == 0:
        print("[Gaia DR3] No se encontraron contrapartes en el catálogo.")
        return resultado_pipeline

    coords_gaia = SkyCoord(ra=tabla_gaia["ra"], dec=tabla_gaia["dec"], unit="deg", frame="icrs")

    def valor_seguro(val, default=0.0) -> float:
        """Convierte valores enmascarados (MaskedConstant) o NaN de Astropy a float limpio."""
        if val is None or np.ma.is_masked(val):
            return default
        v = float(val)
        return v if np.isfinite(v) else default

    for epoch in resultado_pipeline["epochs"]:
        emparejados = 0
        for fuente in epoch["catalogo_campo"]:
            c_sep = SkyCoord(ra=fuente["ra"] * u.deg, dec=fuente["dec"] * u.deg, frame="icrs")
            idx, sep2d, _ = match_coordinates_sky(c_sep, coords_gaia)

            # Extraer escalar puro sin importar si sep2d.arcsec es array 0D o 1D
            dist_arcsec = float(np.asarray(sep2d.arcsec).item())
            idx_escalar = int(np.asarray(idx).item())

            if dist_arcsec <= radio_match_arcsec:
                fila_gaia = tabla_gaia[idx_escalar]
                gaia_id = f"Gaia DR3 {fila_gaia['source_id']}"
                plx = valor_seguro(fila_gaia["parallax"], 0.0)

                fuente["nombre_oficial"] = gaia_id
                fuente["distancia_match_arcsec"] = round(dist_arcsec, 3)
                fuente["gaia_mag_g"] = round(valor_seguro(fila_gaia["phot_g_mean_mag"], 99.0), 2)
                fuente["paralaje_mas"] = round(plx, 3)
                fuente["pmra_mas_yr"] = round(valor_seguro(fila_gaia["pmra"], 0.0), 3)
                fuente["pmdec_mas_yr"] = round(valor_seguro(fila_gaia["pmdec"], 0.0), 3)
                fuente["clasificacion_objeto"] = "Estrella (Gaia)" if plx > 0.5 else "Fuente Extragaláctica / Lejana"
                emparejados += 1
            else:
                fuente["nombre_oficial"] = f"SPHEREx-IR {fuente['fuente_id']}"
                fuente["clasificacion_objeto"] = "Fuente Infrarroja (Sin contraparte óptica en Gaia)"

        print(f"[Cotejo Epoch {epoch['epoch_id']}] {emparejados}/{len(epoch['catalogo_campo'])} fuentes identificadas con nombre en Gaia DR3.")

    return resultado_pipeline

# =====================================================================
# EJECUCIÓN DE PRUEBA
# =====================================================================
if __name__ == "__main__":
    dir_actual = Path(__file__).resolve().parent

    # Puedes usar input_s3_real_grande.json (si ya corriste el test en vivo) o spherex.json
    ruta_entrada = dir_actual / "input_s3_real_grande.json"
    if not ruta_entrada.exists():
        ruta_entrada = dir_actual / "spherex.json"

    with open(ruta_entrada, "r", encoding="utf-8") as f:
        datos_in = json.load(f)

    # 1. ANTES del pipeline: Consultar ALeRCE para saber qué objeto es y elegir las 6 bandas
    info_alerce = consultar_clasificacion_alerce(datos_in["ra"], datos_in["dec"])

    # 2. DURANTE el pipeline: Ejecutar extracción de fuentes desde S3 con las bandas de ALeRCE
    resultado = ejecutar_pipeline_completo(str(ruta_entrada), clase_alerce=info_alerce["clase"])
    resultado["alerce_info"] = info_alerce

    # 3. DESPUÉS del pipeline: Cotejar contra Gaia DR3 para bautizar todas las estrellas/galaxias
    resultado_identificado = bautizar_fuentes_con_gaia(resultado)

    ruta_salida = dir_actual / "salida_identificada_gaia_alerce.json"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultado_identificado, f, indent=2)

    print(f"\n¡Listo! Catálogo completo con nombres oficiales guardado en:\n{ruta_salida}")