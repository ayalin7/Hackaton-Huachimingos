"""
================================================================================
MÓDULO: generador_json_definitivo.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

DIR_ACTUAL = Path(__file__).resolve().parent
if str(DIR_ACTUAL) not in sys.path:
    sys.path.insert(0, str(DIR_ACTUAL))

from generador_interpolacion import generar_datos_interpolados
from pipeline_spherex_vectorizado import (
    BANDAS_OBJETIVO_UM,
    ejecutar_pipeline_completo,
    lambda_a_color_hex,
)
from test_identificacion_gaia_alerce import (
    bautizar_fuentes_con_gaia,
    consultar_clasificacion_alerce,
)

CARPETA_DATOS_WEB = DIR_ACTUAL / "datos_web"
CARPETA_DATOS_WEB.mkdir(exist_ok=True)

OBJETIVOS_CHALLENGE = [
    {
        "nombre_display": "Barnard's Star",
        "slug": "barnards_star",
        "alias": ["barnard", "gliese 699", "gj 699", "hip 87937", "v2500 oph"],
        "query_simbad": "Barnard's Star",
        "ra": 269.452076,
        "dec": 4.693365,
        "perfil_espectral": "DEFAULT",
        "fov_deg": 0.04,
        "pmra_ref": -801.55,
        "pmdec_ref": 10362.39,
        "paralaje_ref": 546.98,
        "fuente_literatura": "Gaia DR3 / Ribas et al. (2018) Nature 563",
        "astrofisica": {
            "tipo_cuerpo": "Estrella Enana Roja (Tipo espectral M4V)",
            "masa": "0.144 M☉ (~151 M_Jup)",
            "densidad": "42.5 g/cm³",
            "temperatura_eff_k": 3134,
            "composicion_quimica": "Hidrógeno (71%), Helio (27%), bandas moleculares IR de H₂O, TiO, VO y FeH",
            "descripcion": "Estrella individual más cercana al Sol (5.96 años luz) y poseedora del mayor movimiento propio conocido (~10.36''/año).",
        },
    },
    {
        "nombre_display": "Epsilon Indi B",
        "slug": "epsilon_indi_b",
        "alias": ["epsilon indi b", "eps ind b", "epsilon indi ba", "epsilon indi bb", "wise j220410.52-564657.7"],
        "query_simbad": "eps Ind B",
        "ra": 331.076250,
        "dec": -56.782500,
        "perfil_espectral": "BROWN_DWARF",
        "fov_deg": 0.04,
        "pmra_ref": 3960.93,
        "pmdec_ref": -2539.23,
        "paralaje_ref": 274.84,
        "fuente_literatura": "King et al. (2010) A&A 510 / SIMBAD",
        "astrofisica": {
            "tipo_cuerpo": "Sistema Binario de Enanas Marrones (T1V + T6V)",
            "masa": "0.072 M☉ + 0.045 M☉ (~75 + 47 M_Jup)",
            "densidad": "~68.0 g/cm³ (promedio del par subestelar)",
            "temperatura_eff_k": 1280,
            "composicion_quimica": "H₂ (74%), He (24%), fuerte absorción infrarroja de Metano (CH₄), Vapor de Agua (H₂O) y Amoníaco (NH₃)",
            "descripcion": "Par binario de enanas marrones tipo T orbitando alrededor de la estrella naranja Epsilon Indi a 11.87 años luz.",
        },
    },
    {
        "nombre_display": "WISE 0855",
        "slug": "wise_0855",
        "alias": ["wise 0855", "wise 0855-0714", "wise j085510.83-071442.5", "j085510.83-071442.5"],
        "query_simbad": None,
        "coord_exacta_str": "08h55m10.831s -07d14m42.53s",
        "ra": 133.795129,
        "dec": -7.245147,
        "perfil_espectral": "BROWN_DWARF",
        "fov_deg": 0.05,
        "pmra_ref": -8118.0,
        "pmdec_ref": 680.0,
        "paralaje_ref": 438.9,
        "fuente_literatura": "Luhman (2014) ApJ 786, L18 / SIMBAD",
        "astrofisica": {
            "tipo_cuerpo": "Enana Marrón Ultrafría (Tipo espectral Y4)",
            "masa": "3 a 10 M_Jup (rango de modelos evolutivos)",
            "densidad": "~1.5 – 5.0 g/cm³ (modelo atmosférico tipo Y)",
            "temperatura_eff_k": 250,
            "composicion_quimica": "H₂, He, nubes de hielo de agua (H₂O), metano (CH₄), fosfina (PH₃) y sulfuro de hidrógeno",
            "descripcion": "El objeto subestelar más frío conocido fuera del Sistema Solar (~250 K / -23 °C), ubicado a 7.43 años luz con desplazamiento ultrarrápido.",
        },
    },
    {
        "nombre_display": "RW Aurigae",
        "slug": "rw_aurigae",
        "alias": ["rw aurigae", "rw aur", "hd 240764", "hip 23873"],
        "query_simbad": "RW Aur",
        "ra": 76.956529,
        "dec": 30.401442,
        "perfil_espectral": "YSO",
        "fov_deg": 0.04,
        "pmra_ref": 4.21,
        "pmdec_ref": -24.15,
        "paralaje_ref": 6.12,
        "fuente_literatura": "Dodson-Robinson et al. (2016) / Gaia DR3 / SIMBAD",
        "astrofisica": {
            "tipo_cuerpo": "Estrella Variable Joven T Tauri Clásica (Sistema Binario)",
            "masa": "1.35 M☉ (componente A) + 0.90 M☉ (componente B)",
            "densidad": "0.38 g/cm³ (estrella pre-secuencia principal expandida)",
            "temperatura_eff_k": 4950,
            "composicion_quimica": "H, He, disco circunestelar rico en silicatos amorfos, hielos de H₂O (3.05 µm), CO₂ (4.25 µm) y gas CO (4.65 µm)",
            "descripcion": "Estrella joven en formación (~5 millones de años) con dramáticos episodios de oscurecimiento provocados por nubes de polvo en su disco.",
        },
    },
    {
        "nombre_display": "R Draconis",
        "slug": "r_draconis",
        "alias": ["r draconis", "r dra", "hd 149880", "hip 81014"],
        "query_simbad": "R Dra",
        "ra": 248.165033,
        "dec": 66.755344,
        "perfil_espectral": "DEFAULT",
        "fov_deg": 0.04,
        "pmra_ref": -15.42,
        "pmdec_ref": 22.84,
        "paralaje_ref": 1.48,
        "fuente_literatura": "Gaia DR3 / GCVS / SIMBAD",
        "astrofisica": {
            "tipo_cuerpo": "Estrella Gigante Roja Pulsante Variable Tipo Mira (M5e-M9e)",
            "masa": "~1.15 M☉ (núcleo C/O + envoltura extendida)",
            "densidad": "~2.4e-6 g/cm³ (atmósfera estelar ultra-expandida)",
            "temperatura_eff_k": 2850,
            "composicion_quimica": "Envoltura de H/He enriquecida con carbono, monóxido de carbono (CO), vapor de agua (H₂O), TiO y granos de polvo de silicato",
            "descripcion": "Gigante roja en etapa tardía (rama asintótica gigante) que pulsa radialmente cada ~245 días cambiando drásticamente su brillo infrarrojo.",
        },
    },
]


def slugify(texto: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", texto.strip().lower()).strip("_")
    return s or "region_celeste"


def buscar_target_challenge(query: str | None = None, ra: float | None = None, dec: float | None = None) -> dict | None:
    if query:
        q_norm = query.strip().lower()
        for obj in OBJETIVOS_CHALLENGE:
            if q_norm == obj["nombre_display"].lower() or q_norm == obj["slug"] or any(a in q_norm for a in obj["alias"]):
                return obj

    if ra is not None and dec is not None:
        c_in = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        for obj in OBJETIVOS_CHALLENGE:
            c_obj = SkyCoord(ra=obj["ra"] * u.deg, dec=obj["dec"] * u.deg, frame="icrs")
            if c_in.separation(c_obj).arcsec <= 45.0:
                return obj
    return None


def resolver_coordenadas_entrada(query: str | None = None, ra: float | None = None, dec: float | None = None) -> tuple[float, float, str, dict | None]:
    target_ch = buscar_target_challenge(query=query, ra=ra, dec=dec)
    if target_ch:
        if target_ch.get("coord_exacta_str"):
            c = SkyCoord(target_ch["coord_exacta_str"], unit=(u.hourangle, u.deg), frame="icrs")
            return float(c.ra.deg), float(c.dec.deg), target_ch["nombre_display"], target_ch
        if target_ch.get("query_simbad"):
            try:
                c = SkyCoord.from_name(target_ch["query_simbad"])
                return float(c.icrs.ra.deg), float(c.icrs.dec.deg), target_ch["nombre_display"], target_ch
            except Exception:
                pass
        return float(target_ch["ra"]), float(target_ch["dec"]), target_ch["nombre_display"], target_ch

    if ra is not None and dec is not None:
        return float(ra), float(dec), f"Campo ({ra:.4f}°, {dec:.4f}°)", None

    if query:
        q_clean = query.strip()
        parts = q_clean.replace(",", " ").split()
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1]), f"Campo ({float(parts[0]):.4f}°, {float(parts[1]):.4f}°)", None
            except ValueError:
                pass
        try:
            c = SkyCoord.from_name(q_clean)
            return float(c.icrs.ra.deg), float(c.icrs.dec.deg), q_clean, None
        except Exception as exc:
            raise ValueError(f"No se pudo resolver el identificador o coordenada '{query}': {exc}") from exc

    raise ValueError("Debes proporcionar un nombre/ID de objeto o coordenadas (ra, dec).")


def consultar_cuadros_irsa_sia2(ra: float, dec: float, radio_deg: float = 0.03, timeout_sec: int = 20) -> list[dict]:
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    for col in ("spherex_qr3", "spherex_dr1"):
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
                mjd = float(row["t_min"]) if "t_min" in cols else 60820.0
                em_min = float(row["em_min"]) * 1e6 if "em_min" in cols else 0.75
                em_max = float(row["em_max"]) * 1e6 if "em_max" in cols else 1.10
                cuadros.append({
                    "fecha_utc": str(mjd),
                    "mjd": mjd,
                    "lambda_um": round((em_min + em_max) / 2.0, 4),
                    "banda": banda,
                    "url": url,
                })
            return cuadros
        except (FuturesTimeoutError, Exception):
            continue
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return []


def inferir_propiedades_fisicas_cuerpo(
    fuente: dict,
    ecuacion: dict | None = None,
    target_challenge: dict | None = None,
    es_objetivo_central: bool = False,
    clase_alerce: str = "DEFAULT",
) -> dict:
    """
    Entrega los parámetros astrofísicos distinguiendo explícitamente entre:
      1) Valores de referencia publicados en la literatura (para los 5 targets oficiales).
      2) Cantidades derivadas directamente de astrometría Gaia DR3 + fotometría SPHEREx
         (sin inventar masas, densidades ni velocidades cosmológicas por umbral).
    """
    pmra = float(ecuacion["mu_ra_cosdec_mas_yr"]) if ecuacion else float(fuente.get("pmra_mas_yr") or 0.0)
    pmdec = float(ecuacion["mu_dec_mas_yr"]) if ecuacion else float(fuente.get("pmdec_mas_yr") or 0.0)
    pm_total = float(np.hypot(pmra, pmdec))

    if es_objetivo_central and target_challenge:
        plx = float(target_challenge.get("paralaje_ref") or 0.0)
        dist_pc = (1000.0 / plx) if plx > 0 else None
        vel_tan_kms = round(4.74047 * (pm_total / 1000.0) * dist_pc, 2) if dist_pc else None
        info_af = target_challenge["astrofisica"]
        fuente_lit = target_challenge.get("fuente_literatura", "SIMBAD")
        return {
            "origen_parametros": f"Valor de referencia de la literatura ({fuente_lit})",
            "tipo_cuerpo": info_af["tipo_cuerpo"],
            "velocidad_relativa_kms": vel_tan_kms,
            "movimiento_propio_total_mas_yr": round(pm_total, 2),
            "distancia_pc": round(dist_pc, 2) if dist_pc else None,
            "distancia_ly": round(dist_pc * 3.26156, 2) if dist_pc else None,
            "masa": f"{info_af['masa']} [Ref. literatura]",
            "densidad": f"{info_af['densidad']} [Ref. literatura]",
            "temperatura_eff_k": info_af["temperatura_eff_k"],
            "composicion_quimica": info_af["composicion_quimica"],
            "descripcion_cientifica": f"{info_af['descripcion']} (Fuente física: {fuente_lit}).",
        }

    plx = float(fuente.get("paralaje_mas") or 0.0)
    dist_pc = (1000.0 / plx) if plx > 0.5 else None
    vel_tan_kms = round(4.74047 * (pm_total / 1000.0) * dist_pc, 2) if (dist_pc and pm_total > 0) else None
    morf = fuente.get("tipo_morfologia", "puntual")

    return {
        "origen_parametros": "Medición astrométrica Gaia DR3 + Fotometría SPHEREx",
        "tipo_cuerpo": "Fuente extendida / Galaxia candidata" if morf == "extendido" else fuente.get("clasificacion_objeto", "Tipo no determinado"),
        "velocidad_relativa_kms": vel_tan_kms,
        "movimiento_propio_total_mas_yr": round(pm_total, 2) if pm_total > 0 else None,
        "distancia_pc": round(dist_pc, 1) if dist_pc else None,
        "distancia_ly": round(dist_pc * 3.26156, 1) if dist_pc else None,
        "masa": "No determinada (requiere espectroscopía individual)",
        "densidad": "No determinada",
        "temperatura_eff_k": None,
        "composicion_quimica": "Fotometría calibrada SPHEREx (0.75–5.0 µm); sin ajuste químico individual",
        "descripcion_cientifica": "Fuente extraída del campo SPHEREx. Velocidad tangencial calculada vía v_tan = 4.74 · μ · d cuando existe paralaje Gaia DR3 > 0.5 mas.",
    }


def asegurar_4_mediciones_satelite(datos_bautizados: dict, target_challenge: dict | None = None) -> dict:
    """
    Marca las épocas reales de SPHEREx como 'observada_spherex'. Si la muestra
    tiene menos de 4 épocas, proyecta las faltantes usando la cinemática real de
    Gaia DR3 etiquetándolas explícitamente como 'proyectada_gaia' (sin ruido ni
    variabilidad sinusoidal inventada).
    """
    epochs = datos_bautizados.get("epochs", [])
    for ep in epochs:
        ep.setdefault("tipo_medicion", "observada_spherex")

    if len(epochs) >= 4:
        return datos_bautizados

    ep_base = epochs[-1]
    mjd_0 = float(ep_base["mjd_promedio"])
    paso_mjd = 182.6

    while len(epochs) < 4:
        idx_nuevo = len(epochs) + 1
        dt_dias = (idx_nuevo - len(epochs)) * paso_mjd
        dt_yr = dt_dias / 365.25
        mjd_nuevo = mjd_0 + dt_dias
        mjd_0 = mjd_nuevo

        nuevo_catalogo = []
        for f in ep_base.get("catalogo_campo", []):
            pmra = float(f.get("pmra_mas_yr") or 0.0)
            pmdec = float(f.get("pmdec_mas_yr") or 0.0)
            cos_dec = max(float(np.cos(np.radians(f["dec"]))), 1e-6)

            ra_n = f["ra"] + ((pmra * dt_yr) / (3600.0 * 1000.0)) / cos_dec
            dec_n = f["dec"] + ((pmdec * dt_yr) / (3600.0 * 1000.0))

            nuevo_esp = [{**b_pt, "mjd": mjd_nuevo} for b_pt in f.get("espectro", [])]

            nuevo_catalogo.append({
                **f,
                "ra": float(ra_n),
                "dec": float(dec_n),
                "tipo_medicion": "proyectada_gaia",
                "espectro": nuevo_esp,
            })

        epochs.append({
            "epoch_id": idx_nuevo,
            "mjd_promedio": mjd_nuevo,
            "tipo_medicion": "proyectada_gaia",
            "es_reconstruida_cadencia": True,
            "objeto_objetivo": nuevo_catalogo[0] if nuevo_catalogo else None,
            "catalogo_campo": nuevo_catalogo,
            "bandas": ep_base.get("bandas", []),
        })
        ep_base = epochs[-1]

    datos_bautizados["epochs"] = epochs
    return datos_bautizados


def construir_catalogo_respaldo_s3_offline(
    ra: float, dec: float, fov_deg: float, nombre_objetivo: str, clase_alerce: str, target_challenge: dict | None
) -> dict:
    """
    Respaldo de emergencia cuando IRSA SIA2 o AWS S3 no responden.
    Usa identificadores SIM-001 (jamás IDs Gaia DR3 falsos) y marca
    explícitamente 'es_simulado': True.
    """
    objetivos_lam = BANDAS_OBJETIVO_UM.get(clase_alerce.upper(), BANDAS_OBJETIVO_UM["DEFAULT"])
    mjd_base = 60810.5
    rng = np.random.default_rng(seed=int(abs(ra * 100 + dec * 100)))

    pmra_obj = target_challenge["pmra_ref"] if target_challenge else -14.5
    pmdec_obj = target_challenge["pmdec_ref"] if target_challenge else 8.2
    plx_obj = target_challenge["paralaje_ref"] if target_challenge else 12.4

    estrellas_base = [
        {
            "fuente_id": "SIM_001",
            "nombre_oficial": nombre_objetivo,
            "es_objetivo": True,
            "ra0": ra,
            "dec0": dec,
            "pmra": pmra_obj,
            "pmdec": pmdec_obj,
            "plx": plx_obj,
            "mag_g": 19.8 if (target_challenge and target_challenge["slug"] == "wise_0855") else 11.4,
            "morfologia": "puntual",
            "a_arcsec": 6.4,
            "b_arcsec": 6.1,
            "angulo": 18.0,
            "flujo_base": 480.0,
            "pendiente_ir": 2.4 if clase_alerce in ("YSO", "BROWN_DWARF") else -0.45,
        }
    ]

    for idx in range(2, 16):
        d_ra = rng.uniform(-fov_deg * 0.45, fov_deg * 0.45)
        d_dec = rng.uniform(-fov_deg * 0.45, fov_deg * 0.45)
        es_gal = idx in (5, 11)
        estrellas_base.append({
            "fuente_id": f"SIM_{idx:03d}",
            "nombre_oficial": f"SIM-GAL-{idx:03d}" if es_gal else f"SIM-{idx:03d}",
            "es_objetivo": False,
            "ra0": ra + d_ra,
            "dec0": dec + d_dec,
            "pmra": 0.0 if es_gal else float(rng.normal(0, 12.0)),
            "pmdec": 0.0 if es_gal else float(rng.normal(0, 12.0)),
            "plx": 0.05 if es_gal else float(rng.uniform(0.6, 15.0)),
            "mag_g": float(rng.uniform(13.2, 19.2)),
            "morfologia": "extendido" if es_gal else "puntual",
            "a_arcsec": float(rng.uniform(14.0, 22.0)) if es_gal else 6.2,
            "b_arcsec": float(rng.uniform(8.0, 13.0)) if es_gal else 6.0,
            "angulo": float(rng.uniform(0, 180)),
            "flujo_base": float(rng.uniform(45.0, 620.0)),
            "pendiente_ir": 0.85 if es_gal else float(rng.uniform(-0.6, 0.35)),
        })

    epochs = []
    for ep_id in range(1, 5):
        dt_dias = (ep_id - 1) * 182.6
        dt_yr = dt_dias / 365.25
        mjd_ep = mjd_base + dt_dias
        catalogo_ep = []

        for st in estrellas_base:
            cos_dec = max(float(np.cos(np.radians(st["dec0"]))), 1e-6)
            ra_t = st["ra0"] + ((st["pmra"] * dt_yr) / (3600.0 * 1000.0)) / cos_dec
            dec_t = st["dec0"] + ((st["pmdec"] * dt_yr) / (3600.0 * 1000.0))

            espectro = []
            for lam in objetivos_lam:
                factor_lam = max((lam / objetivos_lam[0]) ** st["pendiente_ir"], 0.15)
                flux = float(st["flujo_base"] * factor_lam)
                err = float(flux * 0.035 + 1.2)
                mag_ab = float(-2.5 * np.log10(max(flux, 1e-3) / 3.631e9))
                espectro.append({
                    "mjd": mjd_ep,
                    "lambda_um": lam,
                    "color_hex": lambda_a_color_hex(lam, min(objetivos_lam), max(objetivos_lam)),
                    "flux_ujy": round(flux, 3),
                    "flux_err_ujy": round(err, 3),
                    "mag_ab": round(mag_ab, 3),
                })

            catalogo_ep.append({
                "fuente_id": st["fuente_id"],
                "nombre_oficial": st["nombre_oficial"],
                "clasificacion_objeto": "Objeto Simulado (Modo Offline)",
                "tipo_medicion": "simulada",
                "es_objetivo": st["es_objetivo"],
                "ra": float(ra_t),
                "dec": float(dec_t),
                "ra_err_mas": 22.0,
                "dec_err_mas": 22.0,
                "gaia_mag_g": round(st["mag_g"], 2),
                "paralaje_mas": round(st["plx"], 3),
                "pmra_mas_yr": round(st["pmra"], 3),
                "pmdec_mas_yr": round(st["pmdec"], 3),
                "tipo_morfologia": st["morfologia"],
                "semieje_a_arcsec": st["a_arcsec"],
                "semieje_b_arcsec": st["b_arcsec"],
                "angulo_deg": st["angulo"],
                "distancia_centro_arcsec": 0.0 if st["es_objetivo"] else round(float(np.hypot(ra_t - ra, dec_t - dec) * 3600.0), 2),
                "espectro": espectro,
            })

        epochs.append({
            "epoch_id": ep_id,
            "mjd_promedio": mjd_ep,
            "tipo_medicion": "simulada",
            "objeto_objetivo": catalogo_ep[0],
            "catalogo_campo": catalogo_ep,
            "bandas": [
                {
                    "banda_idx": b_i,
                    "mjd": mjd_ep,
                    "lambda_um": lam,
                    "banda": f"SPHEREx-D{min(b_i + 1, 6)}",
                    "color_hex": lambda_a_color_hex(lam, min(objetivos_lam), max(objetivos_lam)),
                    "total_fuentes": len(catalogo_ep),
                }
                for b_i, lam in enumerate(objetivos_lam)
            ],
        })

    return {
        "centro_campo": {"ra": ra, "dec": dec, "fov_deg": fov_deg},
        "clase_alerce": clase_alerce,
        "origen_datos": "SIMULADO_SIN_CONEXION_IRSA",
        "es_simulado": True,
        "epochs": epochs,
    }


def construir_json_definitivo(
    query: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    fov_deg: float | None = None,
    clase_forzada: str | None = None,
    guardar_disco: bool = True,
) -> dict:
    ra_res, dec_res, nombre_display, target_ch = resolver_coordenadas_entrada(query=query, ra=ra, dec=dec)
    fov_usar = float(fov_deg) if fov_deg else (target_ch["fov_deg"] if target_ch else 0.04)
    slug = target_ch["slug"] if target_ch else slugify(nombre_display)

    info_alerce = consultar_clasificacion_alerce(ra_res, dec_res, radio_arcsec=6.0)
    if clase_forzada and clase_forzada.upper() != "AUTO":
        clase_usar = clase_forzada.upper()
    elif info_alerce.get("oid"):
        clase_usar = info_alerce["clase"]
    elif target_ch:
        clase_usar = target_ch["perfil_espectral"]
    else:
        clase_usar = "DEFAULT"

    cuadros = consultar_cuadros_irsa_sia2(ra_res, dec_res, radio_deg=max(fov_usar * 0.75, 0.025))
    datos_pipeline = None

    if cuadros:
        ruta_tmp_in = CARPETA_DATOS_WEB / f"_tmp_in_{slug}.json"
        with open(ruta_tmp_in, "w", encoding="utf-8") as f:
            json.dump({"ra": ra_res, "dec": dec_res, "fov_deg": fov_usar, "cuadros": cuadros}, f)
        try:
            datos_pipeline = ejecutar_pipeline_completo(str(ruta_tmp_in), clase_alerce=clase_usar)
            datos_pipeline["origen_datos"] = "NASA_SPHEREx_S3_Vivo"
            datos_pipeline["es_simulado"] = False
        except Exception as exc:
            print(f"[Pipeline S3] Advertencia ejecutando FITS en vivo ({exc}), usando respaldo offline.")
        finally:
            ruta_tmp_in.unlink(missing_ok=True)

    if not datos_pipeline or not datos_pipeline.get("epochs"):
        datos_pipeline = construir_catalogo_respaldo_s3_offline(
            ra=ra_res,
            dec=dec_res,
            fov_deg=fov_usar,
            nombre_objetivo=nombre_display,
            clase_alerce=clase_usar,
            target_challenge=target_ch,
        )
    else:
        datos_pipeline = bautizar_fuentes_con_gaia(datos_pipeline, radio_match_arcsec=4.5)

    coord_centro = SkyCoord(ra=ra_res * u.deg, dec=dec_res * u.deg, frame="icrs")
    for ep in datos_pipeline["epochs"]:
        cat = ep.get("catalogo_campo", [])
        if not cat:
            continue
        mejor_idx = 0
        min_sep = 1e9
        for i, f in enumerate(cat):
            sep = SkyCoord(ra=f["ra"] * u.deg, dec=f["dec"] * u.deg).separation(coord_centro).arcsec
            if sep < min_sep:
                min_sep = sep
                mejor_idx = i
        if min_sep <= 20.0:
            cat[mejor_idx]["es_objetivo"] = True
            if target_ch:
                cat[mejor_idx]["nombre_oficial"] = target_ch["nombre_display"]
                cat[mejor_idx]["pmra_mas_yr"] = target_ch["pmra_ref"]
                cat[mejor_idx]["pmdec_mas_yr"] = target_ch["pmdec_ref"]
                cat[mejor_idx]["paralaje_mas"] = target_ch["paralaje_ref"]
            ep["objeto_objetivo"] = cat[mejor_idx]

    datos_4_epochs = asegurar_4_mediciones_satelite(datos_pipeline, target_challenge=target_ch)
    datos_4_epochs["alerce_info"] = info_alerce

    datos_interpolados = generar_datos_interpolados(datos_4_epochs)

    mapa_ecuaciones = {
        item["nombre_oficial"]: item["ecuacion"]
        for item in datos_interpolados.get("ecuaciones_trayectoria", [])
    }

    meta_gaia_por_nombre = {}
    for f in datos_interpolados["epochs"][0].get("catalogo_campo", []):
        meta_gaia_por_nombre[f.get("nombre_oficial")] = {
            "gaia_mag_g": f.get("gaia_mag_g"),
            "paralaje_mas": f.get("paralaje_mas", 0.0),
            "pmra_mas_yr": f.get("pmra_mas_yr", 0.0),
            "pmdec_mas_yr": f.get("pmdec_mas_yr", 0.0),
            "es_objetivo": bool(f.get("es_objetivo", False)),
        }

    for ep in datos_interpolados["epochs"]:
        for f in ep.get("catalogo_campo", []):
            nombre_f = f.get("nombre_oficial") or f["fuente_id"]
            meta_g = meta_gaia_por_nombre.get(nombre_f, {})
            for k, v in meta_g.items():
                if k not in f or f[k] is None:
                    f[k] = v
            eq_f = mapa_ecuaciones.get(nombre_f)
            f["astrofisica"] = inferir_propiedades_fisicas_cuerpo(
                fuente=f,
                ecuacion=eq_f,
                target_challenge=target_ch,
                es_objetivo_central=bool(f.get("es_objetivo", False)),
                clase_alerce=clase_usar,
            )
        if ep.get("objeto_objetivo"):
            nombre_obj = ep["objeto_objetivo"].get("nombre_oficial")
            for f in ep.get("catalogo_campo", []):
                if f.get("nombre_oficial") == nombre_obj:
                    ep["objeto_objetivo"] = f
                    break

    for item_eq in datos_interpolados.get("ecuaciones_trayectoria", []):
        nombre_f = item_eq["nombre_oficial"]
        meta_g = meta_gaia_por_nombre.get(nombre_f, {})
        item_eq["es_objetivo"] = meta_g.get("es_objetivo", False)
        fuente_ref = {
            "nombre_oficial": nombre_f,
            "tipo_morfologia": item_eq.get("tipo_morfologia", "puntual"),
            "clasificacion_objeto": item_eq.get("clasificacion_objeto", ""),
            "espectro": item_eq.get("espectro_referencia", []),
            **meta_g,
        }
        item_eq["astrofisica"] = inferir_propiedades_fisicas_cuerpo(
            fuente=fuente_ref,
            ecuacion=item_eq["ecuacion"],
            target_challenge=target_ch,
            es_objetivo_central=meta_g.get("es_objetivo", False),
            clase_alerce=clase_usar,
        )

    bandas_maestras = []
    esp_ref = datos_interpolados["epochs"][0]["catalogo_campo"][0].get("espectro", []) if datos_interpolados["epochs"][0].get("catalogo_campo") else []
    for idx_b, pt in enumerate(esp_ref):
        bandas_maestras.append({
            "banda_idx": idx_b,
            "lambda_um": pt["lambda_um"],
            "color_hex": pt["color_hex"],
            "etiqueta": f"B{idx_b + 1} ({pt['lambda_um']:.2f} µm)",
        })

    datos_interpolados["es_simulado"] = bool(datos_pipeline.get("es_simulado", False))
    datos_interpolados["meta"] = {
        "nombre_objetivo": nombre_display,
        "slug": slug,
        "es_challenge_oficial": target_ch is not None,
        "origen_datos": datos_pipeline.get("origen_datos", "NASA_SPHEREx_Pipeline"),
        "es_simulado": bool(datos_pipeline.get("es_simulado", False)),
        "total_mediciones": len(datos_interpolados["epochs"]),
        "bandas_disponibles": bandas_maestras,
    }

    if guardar_disco:
        ruta_salida = CARPETA_DATOS_WEB / f"definitivo_{slug}.json"
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(datos_interpolados, f, indent=2, ensure_ascii=False)
        print(f"[JSON Definitivo] Guardado en: {ruta_salida}")

    return datos_interpolados


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador del JSON Definitivo SPHEREx + Gaia + Interpolación")
    parser.add_argument("--objeto", type=str, default=None, help="Nombre del objeto")
    parser.add_argument("--ra", type=float, default=None, help="Ascensión Recta en grados")
    parser.add_argument("--dec", type=float, default=None, help="Declinación en grados")
    args = parser.parse_args()

    if args.objeto or (args.ra is not None and args.dec is not None):
        construir_json_definitivo(query=args.objeto, ra=args.ra, dec=args.dec)
    else:
        indice = []
        for obj in OBJETIVOS_CHALLENGE:
            print(f"\n>>> Generando JSON definitivo para: {obj['nombre_display']}...")
            res = construir_json_definitivo(query=obj["nombre_display"])
            indice.append({
                "nombre": obj["nombre_display"],
                "slug": obj["slug"],
                "ra": res["centro_campo"]["ra"],
                "dec": res["centro_campo"]["dec"],
                "descripcion": obj["astrofisica"]["descripcion"],
                "archivo": f"definitivo_{obj['slug']}.json",
            })
        ruta_idx = CARPETA_DATOS_WEB / "indice_challenge.json"
        with open(ruta_idx, "w", encoding="utf-8") as f:
            json.dump({"objetivos": indice}, f, indent=2, ensure_ascii=False)
        print(f"\n¡Catálogo definitivo completo generado en {CARPETA_DATOS_WEB}!")