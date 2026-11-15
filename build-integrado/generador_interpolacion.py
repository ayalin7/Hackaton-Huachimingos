"""
================================================================================
MÓDULO: generador_interpolacion.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================
"""

import json
from pathlib import Path

import numpy as np

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
    t0 = float(tiempos_mjd[0])
    ra0 = float(ras_deg[0])
    dec0 = float(decs_deg[0])
    cos_dec0 = max(float(np.cos(np.radians(dec0))), 1e-6)

    if len(tiempos_mjd) >= 2 and (max(tiempos_mjd) - min(tiempos_mjd)) > 1.0:
        dt_anios = np.array([(t - t0) / DIAS_POR_ANIO for t in tiempos_mjd], dtype=float)
        pesos = 1.0 / np.maximum(np.array(errs_mas, dtype=float), 5.0) ** 2

        d_ra_mas = (np.array(ras_deg) - ra0) * cos_dec0 / MAS_A_DEG
        d_dec_mas = (np.array(decs_deg) - dec0) / MAS_A_DEG

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
    dt_yr = (mjd_objetivo - eq["t0_mjd"]) / DIAS_POR_ANIO
    cos_dec0 = max(float(np.cos(np.radians(eq["dec_0_deg"]))), 1e-6)

    d_ra_mas = eq["mu_ra_cosdec_mas_yr"] * dt_yr + 0.5 * eq["acc_ra_mas_yr2"] * (dt_yr ** 2)
    d_dec_mas = eq["mu_dec_mas_yr"] * dt_yr + 0.5 * eq["acc_dec_mas_yr2"] * (dt_yr ** 2)

    ra_t = eq["ra_0_deg"] + (d_ra_mas * MAS_A_DEG) / cos_dec0
    dec_t = eq["dec_0_deg"] + (d_dec_mas * MAS_A_DEG)
    return float(ra_t), float(dec_t)


def interpolar_espectro_en_tiempo(historial_espectros: list[tuple[float, list[dict]]], mjd_objetivo: float) -> list[dict]:
    if len(historial_espectros) == 1:
        _, esp_base = historial_espectros[0]
        return [
            {
                **banda,
                "mjd": mjd_objetivo,
                "flux_err_ujy": round(banda["flux_err_ujy"] * 1.25, 3),
            }
            for banda in esp_base
        ]

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
    epochs = datos.get("epochs", [])
    if not epochs:
        raise ValueError("El JSON de entrada no tiene épocas registradas.")

    clase_alerce = datos.get("alerce_info", {}).get("clase", datos.get("clase_alerce", "DEFAULT"))

    # 1. Reunir trayectorias históricas cruzando por nombre oficial o celda espacial (10'')
    cuerpos_rastreados = {}
    for ep in epochs:
        mjd_ep = float(ep["mjd_promedio"])
        for fuente in ep.get("catalogo_campo", []):
            nombre_f = fuente.get("nombre_oficial") or ""
            if nombre_f and not nombre_f.startswith("SPHEREx-IR"):
                clave = nombre_f
            else:
                clave = f"POS_{round(fuente['ra'] * 360)}_{round(fuente['dec'] * 360)}"

            if clave not in cuerpos_rastreados:
                cuerpos_rastreados[clave] = {
                    "nombre_oficial": nombre_f or fuente["fuente_id"],
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
    for _, info in cuerpos_rastreados.items():
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

    # 3. Fecha MJD de la 5ta medición (extrapolación analítica +6 meses)
    mjds_existentes = [float(ep["mjd_promedio"]) for ep in epochs]
    paso_dias = 182.6 if len(mjds_existentes) == 1 else float(np.mean(np.diff(mjds_existentes)))
    mjd_5ta = float(mjds_existentes[-1] + max(paso_dias, 30.0))

    # 4. Construir catálogo extrapolado para el Epoch 5
    catalogo_epoch_5 = []
    for idx, cuerpo in enumerate(ecuaciones_cuerpos, start=1):
        ra_5, dec_5 = evaluar_posicion_en_tiempo(cuerpo["ecuacion"], mjd_5ta)
        esp_5 = interpolar_espectro_en_tiempo(cuerpo["historial_espectros"], mjd_5ta)

        catalogo_epoch_5.append({
            "fuente_id": f"EXTRAP_{idx:03d}",
            "nombre_oficial": cuerpo["nombre_oficial"],
            "clasificacion_objeto": cuerpo["clasificacion_objeto"],
            "tipo_medicion": "extrapolada",
            "ra": ra_5,
            "dec": dec_5,
            "ra_err_mas": 35.0,
            "dec_err_mas": 35.0,
            "tipo_morfologia": cuerpo["tipo_morfologia"],
            "semieje_a_arcsec": cuerpo["semieje_a_arcsec"],
            "semieje_b_arcsec": cuerpo["semieje_b_arcsec"],
            "angulo_deg": cuerpo["angulo_deg"],
            "es_interpolado": True,
            "es_extrapolado": True,
            "espectro": esp_5,
        })

    epochs_salida = []
    for ep in epochs:
        ep_copia = dict(ep)
        ep_copia["es_interpolado"] = False
        ep_copia["es_extrapolado"] = False
        ep_copia.setdefault("tipo_medicion", "observada_spherex")
        epochs_salida.append(ep_copia)

    epochs_salida.append({
        "epoch_id": len(epochs_salida) + 1,
        "mjd_promedio": mjd_5ta,
        "tipo_medicion": "extrapolada",
        "es_interpolado": True,
        "es_extrapolado": True,
        "objeto_objetivo": catalogo_epoch_5[0] if catalogo_epoch_5 else None,
        "catalogo_campo": catalogo_epoch_5,
    })

    salida_final = {
        "centro_campo": datos["centro_campo"],
        "clase_alerce": clase_alerce,
        "alerce_info": datos.get("alerce_info", {}),
        "es_simulado": bool(datos.get("es_simulado", False)),
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
        raise FileNotFoundError(f"No se encontró {ruta_entrada}.")

    with open(ruta_entrada, "r", encoding="utf-8") as f:
        datos_in = json.load(f)

    resultado = generar_datos_interpolados(datos_in)

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)