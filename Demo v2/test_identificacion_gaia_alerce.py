"""
================================================================================
MÓDULO: test_identificacion_gaia_alerce.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================
"""

import json
from pathlib import Path

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.time import Time
from astroquery.gaia import Gaia

from pipeline_spherex_vectorizado import ejecutar_pipeline_completo


def consultar_clasificacion_alerce(ra: float, dec: float, radio_arcsec: float = 5.0) -> dict:
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

        mejor = items[0]
        oid = mejor.get("oid")
        clase_raw = str(mejor.get("classrf") or mejor.get("classearly") or "DEFAULT").upper()

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


def consultar_catalogo_gaia_campo(ra_centro: float, dec_centro: float, fov_deg: float):
    radio_deg = fov_deg * 0.75
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


def bautizar_fuentes_con_gaia(resultado_pipeline: dict, radio_match_arcsec: float = 4.5) -> dict:
    centro = resultado_pipeline["centro_campo"]
    tabla_gaia = consultar_catalogo_gaia_campo(centro["ra"], centro["dec"], centro["fov_deg"])

    if len(tabla_gaia) == 0:
        print("[Gaia DR3] No se encontraron contrapartes en el catálogo.")
        return resultado_pipeline

    def valor_seguro(val, default=0.0) -> float:
        if val is None or np.ma.is_masked(val):
            return default
        v = float(val)
        return v if np.isfinite(v) else default

    pmra_arr = np.array([valor_seguro(v, 0.0) for v in tabla_gaia["pmra"]])
    pmdec_arr = np.array([valor_seguro(v, 0.0) for v in tabla_gaia["pmdec"]])
    dec_rad = np.radians(np.array(tabla_gaia["dec"], dtype=float))

    for epoch in resultado_pipeline["epochs"]:
        emparejados = 0
        t_obs = Time(float(epoch["mjd_promedio"]), format="mjd")
        dt_yr = (t_obs - Time("J2016.0")).to(u.yr).value

        # Propagar posiciones de Gaia DR3 (época J2016.0) a la época real de observación de SPHEREx
        ra_prop = np.array(tabla_gaia["ra"], dtype=float) + ((pmra_arr * dt_yr) / (3600.0 * 1000.0)) / np.maximum(np.cos(dec_rad), 1e-6)
        dec_prop = np.array(tabla_gaia["dec"], dtype=float) + ((pmdec_arr * dt_yr) / (3600.0 * 1000.0))
        coords_gaia_ep = SkyCoord(ra=ra_prop * u.deg, dec=dec_prop * u.deg, frame="icrs")

        for fuente in epoch["catalogo_campo"]:
            c_sep = SkyCoord(ra=fuente["ra"] * u.deg, dec=fuente["dec"] * u.deg, frame="icrs")
            idx, sep2d, _ = match_coordinates_sky(c_sep, coords_gaia_ep)

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
                fuente["clasificacion_objeto"] = "Estrella (Gaia DR3)" if plx > 0.5 else "Fuente Extragaláctica / Lejana"
                emparejados += 1
            else:
                fuente["nombre_oficial"] = f"SPHEREx-IR {fuente['fuente_id']}"
                fuente["clasificacion_objeto"] = "Fuente Infrarroja (Sin contraparte óptica en Gaia)"

        print(f"[Cotejo Epoch {epoch['epoch_id']}] {emparejados}/{len(epoch['catalogo_campo'])} fuentes identificadas con nombre en Gaia DR3.")

    return resultado_pipeline


if __name__ == "__main__":
    dir_actual = Path(__file__).resolve().parent
    ruta_entrada = dir_actual / "input_s3_real_grande.json"
    if not ruta_entrada.exists():
        ruta_entrada = dir_actual / "spherex.json"

    with open(ruta_entrada, "r", encoding="utf-8") as f:
        datos_in = json.load(f)

    info_alerce = consultar_clasificacion_alerce(datos_in["ra"], datos_in["dec"])
    resultado = ejecutar_pipeline_completo(str(ruta_entrada), clase_alerce=info_alerce["clase"])
    resultado["alerce_info"] = info_alerce
    resultado_identificado = bautizar_fuentes_con_gaia(resultado)

    ruta_salida = dir_actual / "salida_identificada_gaia_alerce.json"
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultado_identificado, f, indent=2)