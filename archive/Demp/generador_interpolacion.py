"""
================================================================================
MÓDULO: generador_interpolacion.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================

DESCRIPCIÓN GENERAL:
    Toma el JSON enriquecido ('salida_identificada_gaia_alerce.json'), extrae las
    series temporales de posición (RA, Dec) y brillo (flux_ujy) en las 6 bandas,
    ajusta las ecuaciones de movimiento de cada cuerpo y genera una 5ta medición
    interpolada/extrapolada para el renderizado temporal interactivo.

DETALLE FÍSICO:
    - Si existen >= 2 Epochs observados en SPHEREx: Ajusta directamente las
      ecuaciones de movimiento por mínimos cuadrados ponderados:
        RA(t)  = RA_0  + (mu_ra_cosdec / cos(Dec_0)) * (t - t_0)
        Dec(t) = Dec_0 + mu_dec * (t - t_0)
      (Incluye término cuadrático de aceleración si la clase es 'ASTEROID').
    - Si solo existe 1 Epoch observado en la muestra de prueba: Utiliza los
      vectores de movimiento propio reales medidos por Gaia DR3 (pmra_mas_yr,
      pmdec_mas_yr) para construir las ecuaciones de movimiento y proyectar las
      épocas faltantes hasta la 5ta medición (permitiendo probar el slider con
      datos reales de movimiento estelar).

USO DESDE TERMINAL:
    python3 generador_interpolacion.py
================================================================================
"""

import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky

# Nota: Cuando tengas motion_interpolator.py en esta carpeta, puedes descomentar:
# from motion_interpolator import quinta_medicion, medidas_desde_filas

DIAS_POR_ANIO = 365.25
MAS_A_DEG = 1.0 / (3600.0 * 1000.0)


def ajustar_ecuacion_movimiento_cuerpo(
    tiempos_mjd: list[float],
    ras_deg: list[float],
    decs_deg: list[float],
    errs_mas: list[float],
    clase_objeto: str = "DEFAULT",
    pmra_gaia_mas_yr: float = 0.0,
    pmdec_gaia_mas_yr: float = 0.0,
) -> dict:
    """
    Calcula los parámetros analíticos de la ecuación de trayectoria RA(t), Dec(t)
    en grados por día (deg/day) y milisegundos de arco por año (mas/yr).
    """
    t0 = float(tiempos_mjd[0])
    ra0 = float(ras_deg[0])
    dec0 = float(decs_deg[0])
    cos_dec0 = max(float(np.cos(np.radians(dec0))), 1e-6)

    # Caso A: Tenemos múltiples épocas observadas en SPHEREx -> Ajuste directo
    if len(tiempos_mjd) >= 2 and (max(tiempos_mjd) - min(tiempos_mjd)) > 1.0:
        dt_anios = np.array([(t - t0) / DIAS_POR_ANIO for t in tiempos_mjd], dtype=float)
        pesos = 1.0 / np.maximum(np.array(errs_mas, dtype=float), 5.0) ** 2

        d_ra_mas = (np.array(ras_deg) - ra0) * cos_dec0 / MAS_A_DEG
        d_dec_mas = (np.array(decs_deg) - dec0) / MAS_A_DEG

        # Si es asteroide y hay >= 3 puntos, ajustar aceleración cuadrática
        if clase_objeto.upper() == "ASTEROID" and len(tiempos_mjd) >= 3:
            coef_ra = np.polyfit(dt_anios, d_ra_mas, deg=2, w=pesos)
            coef_dec = np.polyfit(dt_anios, d_dec_mas, deg=2, w=pesos)
            return {
                "modelo": "cuadratico_acelerado",
                "t0_mjd": t0,
                "ra_0_deg": ra0,
                "dec_0_deg": dec0,
                "mu_ra_cosdec_mas_yr": float(coef_ra[1]),
                "mu_dec_mas_yr": float(coef_dec[1]),
                "acc_ra_mas_yr2": float(2.0 * coef_ra[0]),
                "acc_dec_mas_yr2": float(2.0 * coef_dec[0]),
            }

        # Ajuste lineal ponderado de movimiento propio
        coef_ra = np.polyfit(dt_anios, d_ra_mas, deg=1, w=pesos)
        coef_dec = np.polyfit(dt_anios, d_dec_mas, deg=1, w=pesos)
        return {
            "modelo": "lineal_movimiento_propio",
            "t0_mjd": t0,
            "ra_0_deg": ra0,
            "dec_0_deg": dec0,
            "mu_ra_cosdec_mas_yr": float(coef_ra[0]),
            "mu_dec_mas_yr": float(coef_dec[0]),
            "acc_ra_mas_yr2": 0.0,
            "acc_dec_mas_yr2": 0.0,
        }

    # Caso B: Solo hay 1 Epoch en el JSON -> Usar cinemática real de Gaia DR3
    return {
        "modelo": "cinematica_gaia_dr3",
        "t0_mjd": t0,
        "ra_0_deg": ra0,
        "dec_0_deg": dec0,
        "mu_ra_cosdec_mas_yr": float(pmra_gaia_mas_yr),
        "mu_dec_mas_yr": float(pmdec_gaia_mas_yr),
        "acc_ra_mas_yr2": 0.0,
        "acc_dec_mas_yr2": 0.0,
    }


def evaluar_posicion_en_tiempo(eq: dict, mjd_objetivo: float) -> tuple[float, float]:
    """Evalúa las ecuaciones de movimiento RA(t) y Dec(t) en cualquier fecha MJD."""
    dt_yr = (mjd_objetivo - eq["t0_mjd"]) / DIAS_POR_ANIO
    cos_dec0 = max(float(np.cos(np.radians(eq["dec_0_deg"]))), 1e-6)

    d_ra_mas = eq["mu_ra_cosdec_mas_yr"] * dt_yr + 0.5 * eq["acc_ra_mas_yr2"] * (dt_yr ** 2)
    d_dec_mas = eq["mu_dec_mas_yr"] * dt_yr + 0.5 * eq["acc_dec_mas_yr2"] * (dt_yr ** 2)

    ra_t = eq["ra_0_deg"] + (d_ra_mas * MAS_A_DEG) / cos_dec0
    dec_t = eq["dec_0_deg"] + (d_dec_mas * MAS_A_DEG)
    return float(ra_t), float(dec_t)


def interpolar_espectro_en_tiempo(historial_espectros: list[tuple[float, list[dict]]], mjd_objetivo: float) -> list[dict]:
    """
    Interpola o extrapola el flujo (flux_ujy) en cada una de las 6 bandas
    para una fecha MJD dada.
    """
    if len(historial_espectros) == 1:
        # Si solo hay 1 época, mantiene el espectro base con una ligera incertidumbre propagada
        _, esp_base = historial_espectros[0]
        return [
            {
                **banda,
                "mjd": mjd_objetivo,
                "flux_err_ujy": round(banda["flux_err_ujy"] * 1.25, 3),
            }
            for banda in esp_base
        ]

    # Agrupar flujos históricos por longitud de onda
    n_bandas = len(historial_espectros[0][1])
    espectro_predicho = []

    for b_idx in range(n_bandas):
        tiempos = []
        flujos = []
        errores = []
        ref_banda = historial_espectros[0][1][b_idx]

        for mjd_ep, esp in historial_espectros:
            if b_idx < len(esp):
                tiempos.append(mjd_ep)
                flujos.append(esp[b_idx]["flux_ujy"])
                errores.append(esp[b_idx]["flux_err_ujy"])

        if len(tiempos) >= 2:
            # Ajuste lineal/polinómico suave de la curva de luz en esa banda
            grado = 2 if len(tiempos) >= 3 else 1
            poly = np.polyfit(tiempos, flujos, deg=grado)
            flujo_pred = float(np.polyval(poly, mjd_objetivo))
            err_pred = float(np.mean(errores) * 1.4)
        else:
            flujo_pred = float(flujos[0])
            err_pred = float(errores[0] * 1.3)

        espectro_predicho.append({
            "mjd": mjd_objetivo,
            "lambda_um": ref_banda["lambda_um"],
            "color_hex": ref_banda["color_hex"],
            "flux_ujy": max(flujo_pred, 0.0),
            "flux_err_ujy": err_pred,
            "mag_ab": ref_banda.get("mag_ab"),
        })

    return espectro_predicho


def generar_datos_interpolados(datos: dict) -> dict:
    """
    Procesa todas las épocas del JSON, calcula las ecuaciones de trayectoria
    para cada objeto identificado y construye la 5ta medición sintética.
    """
    epochs = datos.get("epochs", [])
    if not epochs:
        raise ValueError("El JSON de entrada no tiene épocas registradas.")

    clase_alerce = datos.get("alerce_info", {}).get("clase", datos.get("clase_alerce", "DEFAULT"))

    # 1. Reunir trayectorias históricas cruzando fuentes entre épocas
    cuerpos_rastreados = {}
    for ep in epochs:
        mjd_ep = float(ep["mjd_promedio"])
        for fuente in ep.get("catalogo_campo", []):
            clave = fuente.get("nombre_oficial") or fuente["fuente_id"]
            if clave not in cuerpos_rastreados:
                cuerpos_rastreados[clave] = {
                    "nombre_oficial": clave,
                    "clasificacion_objeto": fuente.get("clasificacion_objeto", "Desconocido"),
                    "tipo_morfologia": fuente.get("tipo_morfologia", "puntual"),
                    "semieje_a_arcsec": fuente.get("semieje_a_arcsec", 6.2),
                    "semieje_b_arcsec": fuente.get("semieje_b_arcsec", 6.2),
                    "angulo_deg": fuente.get("angulo_deg", 0.0),
                    "pmra_gaia": fuente.get("pmra_mas_yr", 0.0),
                    "pmdec_gaia": fuente.get("pmdec_mas_yr", 0.0),
                    "observaciones": [],
                }
            cuerpos_rastreados[clave]["observaciones"].append({
                "epoch_id": ep["epoch_id"],
                "mjd": mjd_ep,
                "ra": fuente["ra"],
                "dec": fuente["dec"],
                "err_mas": fuente.get("ra_err_mas", 50.0),
                "espectro": fuente.get("espectro", []),
            })

    # 2. Calcular ecuaciones de movimiento para cada cuerpo
    ecuaciones_cuerpos = []
    for clave, info in cuerpos_rastreados.items():
        obs = sorted(info["observaciones"], key=lambda x: x["mjd"])
        eq = ajustar_ecuacion_movimiento_cuerpo(
            tiempos_mjd=[o["mjd"] for o in obs],
            ras_deg=[o["ra"] for o in obs],
            decs_deg=[o["dec"] for o in obs],
            errs_mas=[o["err_mas"] for o in obs],
            clase_objeto=clase_alerce,
            pmra_gaia_mas_yr=info["pmra_gaia"],
            pmdec_gaia_mas_yr=info["pmdec_gaia"],
        )
        ecuaciones_cuerpos.append({
            "nombre_oficial": info["nombre_oficial"],
            "clasificacion_objeto": info["clasificacion_objeto"],
            "tipo_morfologia": info["tipo_morfologia"],
            "semieje_a_arcsec": info["semieje_a_arcsec"],
            "semieje_b_arcsec": info["semieje_b_arcsec"],
            "angulo_deg": info["angulo_deg"],
            "ecuacion": eq,
            "historial_espectros": [(o["mjd"], o["espectro"]) for o in obs],
        })

    # 3. Definir la fecha MJD de la 5ta medición (6 meses después de la última medición)
    mjds_existentes = [float(ep["mjd_promedio"]) for ep in epochs]
    paso_dias = 182.6 if len(mjds_existentes) == 1 else float(np.mean(np.diff(mjds_existentes)))
    mjd_5ta = float(mjds_existentes[-1] + max(paso_dias, 30.0))

    # 4. Construir el catálogo sintetizado para el Epoch 5 (Interpolado)
    catalogo_epoch_5 = []
    for idx, cuerpo in enumerate(ecuaciones_cuerpos, start=1):
        ra_5, dec_5 = evaluar_posicion_en_tiempo(cuerpo["ecuacion"], mjd_5ta)
        esp_5 = interpolar_espectro_en_tiempo(cuerpo["historial_espectros"], mjd_5ta)

        catalogo_epoch_5.append({
            "fuente_id": f"PRED_{idx:03d}",
            "nombre_oficial": cuerpo["nombre_oficial"],
            "clasificacion_objeto": cuerpo["clasificacion_objeto"],
            "ra": ra_5,
            "dec": dec_5,
            "ra_err_mas": 35.0,
            "dec_err_mas": 35.0,
            "tipo_morfologia": cuerpo["tipo_morfologia"],
            "semieje_a_arcsec": cuerpo["semieje_a_arcsec"],
            "semieje_b_arcsec": cuerpo["semieje_b_arcsec"],
            "angulo_deg": cuerpo["angulo_deg"],
            "es_interpolado": True,
            "espectro": esp_5,
        })

    # Marcar épocas originales como reales y agregar la 5ta medición
    epochs_salida = []
    for ep in epochs:
        ep_copia = dict(ep)
        ep_copia["es_interpolado"] = False
        epochs_salida.append(ep_copia)

    epochs_salida.append({
        "epoch_id": len(epochs_salida) + 1,
        "mjd_promedio": mjd_5ta,
        "es_interpolado": True,
        "objeto_objetivo": catalogo_epoch_5[0] if catalogo_epoch_5 else None,
        "catalogo_campo": catalogo_epoch_5,
    })

    # Exportar estructura final lista para el visor temporal
    salida_final = {
        "centro_campo": datos["centro_campo"],
        "clase_alerce": clase_alerce,
        "alerce_info": datos.get("alerce_info", {}),
        "rango_temporal_mjd": {
            "mjd_inicio": float(mjds_existentes[0]),
            "mjd_fin_interpolado": mjd_5ta,
        },
        "ecuaciones_trayectoria": [
            {
                "nombre_oficial": c["nombre_oficial"],
                "clasificacion_objeto": c["clasificacion_objeto"],
                "tipo_morfologia": c["tipo_morfologia"],
                "semieje_a_arcsec": c["semieje_a_arcsec"],
                "semieje_b_arcsec": c["semieje_b_arcsec"],
                "angulo_deg": c["angulo_deg"],
                "ecuacion": c["ecuacion"],
                "espectro_referencia": c["historial_espectros"][-1][1] if c["historial_espectros"] else [],
            }
            for c in ecuaciones_cuerpos
        ],
        "epochs": epochs_salida,
    }
    return salida_final


if __name__ == "__main__":
    dir_actual = Path(__file__).resolve().parent
    ruta_entrada = dir_actual / "salida_identificada_gaia_alerce.json"
    ruta_salida = dir_actual / "salida_con_interpolacion.json"

    if not ruta_entrada.exists():
        raise FileNotFoundError(f"No se encontró {ruta_entrada}. Ejecuta primero test_identificacion_gaia_alerce.py")

    with open(ruta_entrada, "r", encoding="utf-8") as f:
        datos_in = json.load(f)

    resultado = generar_datos_interpolados(datos_in)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)

    print(f"¡Interpolación completada! Guardado en: {ruta_salida}")
    print(f"Total cuerpos con ecuaciones calculadas: {len(resultado['ecuaciones_trayectoria'])}")
    print(f"Fecha 5ta medición interpolada (MJD): {resultado['rango_temporal_mjd']['mjd_fin_interpolado']:.2f}")