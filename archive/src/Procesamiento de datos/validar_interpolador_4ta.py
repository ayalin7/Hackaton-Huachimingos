"""
================================================================================
MÓDULO: validar_interpolador_4ta.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================

DESCRIPCIÓN GENERAL:
    Prueba de validación científica (Hindcasting / Leave-Last-Out):
    Toma las primeras N-1 mediciones reales observadas por SPHEREx (ej. Epochs 1, 2, 3),
    entrena las ecuaciones de movimiento y espectrometría con ellas, predice la
    posición y el flujo de la última medición real (ej. Epoch 4) y compara la
    predicción contra el dato real medido por el satélite.

MÉTRICAS DE ERROR QUE CALCULA:
    1. Error astrométrico en segundos de arco ('') y milisegundos de arco (mas)
       usando separación angular real en la esfera celeste (SkyCoord.separation).
    2. Error fotométrico porcentual (%) en las 6 bandas infrarrojas.
    3. Gráfico comparativo listo para presentar a tu equipo o al jurado.

USO DESDE TERMINAL:
    python3 validar_interpolador_4ta.py resultados_challenge/final_barnards_star.json
================================================================================
"""

import argparse
import json
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord

# Importamos las funciones matemáticas exactas de tu generador de interpolación
from generador_interpolacion import (
    ajustar_ecuacion_movimiento_cuerpo,
    evaluar_posicion_en_tiempo,
    interpolar_espectro_en_tiempo,
)


def validar_prediccion_retrospectiva(ruta_json: Path):
    if not ruta_json.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_json}")

    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    # Filtrar únicamente las épocas reales observadas (excluyendo la 5ta sintética si ya existe)
    epochs_reales = [ep for ep in datos.get("epochs", []) if not ep.get("es_interpolado", False)]

    if len(epochs_reales) < 2:
        print(f"[AVISO] {ruta_json.name} solo tiene {len(epochs_reales)} época real observada.")
        print("Se necesitan al menos 2 épocas reales en SPHEREx para ocultar la última y validar.")
        return

    # Separar: Entrenamiento = primeras N-1 mediciones | Prueba = última medición real
    epochs_entrenamiento = epochs_reales[:-1]
    epoch_objetivo_real = epochs_reales[-1]
    mjd_prueba = float(epoch_objetivo_real["mjd_promedio"])

    nombre_obj = datos.get("nombre_objeto", "Objeto Objetivo")
    clase = datos.get("clase_alerce", "DEFAULT")

    print("\n" + "=" * 75)
    print(f"VALIDACIÓN RETROSPECTIVA DEL INTERPOLADOR: {nombre_obj}")
    print(f"Entrenando con Epochs 1..{len(epochs_entrenamiento)} -> Prediciendo Epoch {epoch_objetivo_real['epoch_id']} (MJD={mjd_prueba:.2f})")
    print("=" * 75)

    # Extraer el historial del objeto objetivo en cada época
    tiempos, ras, decs, errs = [], [], [], []
    historial_espectros = []
    pmra_gaia, pmdec_gaia = 0.0, 0.0

    for ep in epochs_entrenamiento:
        obj = ep.get("objeto_objetivo") or (ep["catalogo_campo"][0] if ep.get("catalogo_campo") else None)
        if not obj:
            continue
        tiempos.append(float(ep["mjd_promedio"]))
        ras.append(float(obj["ra"]))
        decs.append(float(obj["dec"]))
        errs.append(float(obj.get("ra_err_mas", 50.0)))
        pmra_gaia = float(obj.get("pmra_mas_yr", pmra_gaia))
        pmdec_gaia = float(obj.get("pmdec_mas_yr", pmdec_gaia))
        if obj.get("espectro"):
            historial_espectros.append((float(ep["mjd_promedio"]), obj["espectro"]))

    # Dato real de la última época (el que queremos ver si calza)
    obj_real_final = epoch_objetivo_real.get("objeto_objetivo") or epoch_objetivo_real["catalogo_campo"][0]
    ra_real = float(obj_real_final["ra"])
    dec_real = float(obj_real_final["dec"])
    esp_real = obj_real_final.get("espectro", [])

    # 1. Ajustar ecuaciones SOLO con las primeras N-1 mediciones
    eq_prueba = ajustar_ecuacion_movimiento_cuerpo(
        tiempos_mjd=tiempos,
        ras_deg=ras,
        decs_deg=decs,
        errs_mas=errs,
        clase_objeto=clase,
        pmra_gaia_mas_yr=pmra_gaia,
        pmdec_gaia_mas_yr=pmdec_gaia,
    )

    # 2. Predecir posición y espectro en la fecha de la última medición real
    ra_pred, dec_pred = evaluar_posicion_en_tiempo(eq_prueba, mjd_prueba)
    esp_pred = interpolar_espectro_en_tiempo(historial_espectros, mjd_prueba) if historial_espectros else []

    # 3. Calcular error angular real en la esfera celeste
    c_real = SkyCoord(ra=ra_real * u.deg, dec=dec_real * u.deg, frame="icrs")
    c_pred = SkyCoord(ra=ra_pred * u.deg, dec=dec_pred * u.deg, frame="icrs")
    error_arcsec = float(c_real.separation(c_pred).arcsec)
    error_pixeles_spherex = error_arcsec / 6.2  # 1 píxel SPHEREx = 6.2 arcsec

    print(f"\n[ASTROMETRÍA]")
    print(f"  Posición REAL medida (Epoch {epoch_objetivo_real['epoch_id']}):  RA = {ra_real:.6f}°, Dec = {dec_real:.6f}°")
    print(f"  Posición PREDICHA por modelo:     RA = {ra_pred:.6f}°, Dec = {dec_pred:.6f}°")
    print(f"  -> Error de predicción: {error_arcsec:.3f} arcsec ({error_pixeles_spherex:.2f} píxeles de SPHEREx)")

    # 4. Comparar fotometría banda por banda
    if esp_real and esp_pred:
        print(f"\n[FOTOMETRÍA / ESPECTRO]")
        errores_pct = []
        for b_r, b_p in zip(esp_real, esp_pred):
            fl_r = b_r["flux_ujy"]
            fl_p = b_p["flux_ujy"]
            err_pct = abs(fl_p - fl_r) / max(abs(fl_r), 1e-6) * 100.0
            errores_pct.append(err_pct)
            print(f"  λ={b_r['lambda_um']:.3f} µm | Real: {fl_r:8.2f} µJy | Predicho: {fl_p:8.2f} µJy | Dif: {err_pct:5.1f}%")
        print(f"  -> Error fotométrico medio: {np.mean(errores_pct):.1f}%")

    # 5. Generar gráfico comparativo Real vs. Predicho
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="#090c10")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")

    # Panel izquierdo: Trayectoria espacial
    ax1.plot(ras + [ra_pred], decs + [dec_pred], "--", color="#4cc9f0", alpha=0.6, label="Trayectoria Interpolada")
    ax1.scatter(ras, decs, s=70, color="#2ec4b6", edgecolor="white", zorder=3, label=f"Entrenamiento (Epochs 1..{len(epochs_entrenamiento)})")
    ax1.scatter([ra_real], [dec_real], s=130, marker="*", color="#ffd166", edgecolor="black", zorder=5, label=f"REAL (Epoch {epoch_objetivo_real['epoch_id']})")
    ax1.scatter([ra_pred], [dec_pred], s=110, marker="X", color="#ff3b30", edgecolor="white", zorder=4, label=f"PREDICHO (Error: {error_arcsec:.2f}'')")

    ax1.set_title(f"Validación Astrométrica: {nombre_obj}\nError = {error_arcsec:.2f}'' ({error_pixeles_spherex:.2f} px SPHEREx)", color="white")
    ax1.set_xlabel("Ascensión Recta (RA) [deg]", color="#cccccc")
    ax1.set_ylabel("Declinación (Dec) [deg]", color="#cccccc")
    ax1.invert_xaxis()
    ax1.tick_params(colors="#cccccc")
    ax1.grid(True, color="white", alpha=0.1, linestyle=":")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    # Panel derecho: Espectro Real vs Predicho
    if esp_real and esp_pred:
        lams_r = [b["lambda_um"] for b in esp_real]
        fl_r = [b["flux_ujy"] for b in esp_real]
        lams_p = [b["lambda_um"] for b in esp_pred]
        fl_p = [b["flux_ujy"] for b in esp_pred]

        ax2.plot(lams_r, fl_r, "-o", color="#ffd166", linewidth=2, markersize=8, label=f"Espectro REAL (Epoch {epoch_objetivo_real['epoch_id']})")
        ax2.plot(lams_p, fl_p, "--s", color="#ff3b30", linewidth=2, markersize=7, label="Espectro PREDICHO")
        ax2.set_title("Comparación Espectral en las 6 Bandas", color="white")
        ax2.set_xlabel("Longitud de Onda λ [µm]", color="#cccccc")
        ax2.set_ylabel("Flujo Calibrado [µJy]", color="#cccccc")
        ax2.tick_params(colors="#cccccc")
        ax2.grid(True, color="white", alpha=0.1, linestyle=":")
        ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=9)

    plt.tight_layout()
    ruta_png = ruta_json.with_name(f"validacion_4ta_{ruta_json.stem}.png")
    plt.savefig(ruta_png, dpi=180)
    print(f"\nGráfico de validación guardado en: {ruta_png}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida el interpolador ocultando la última época real observada.")
    parser.add_argument("archivo", nargs="?", default="resultados_challenge/final_barnards_star.json", help="Ruta al JSON del objeto.")
    args = parser.parse_args()

    dir_actual = Path(__file__).resolve().parent
    ruta = Path(args.archivo)
    if not ruta.is_absolute():
        ruta = dir_actual / ruta

    validar_prediccion_retrospectiva(ruta)