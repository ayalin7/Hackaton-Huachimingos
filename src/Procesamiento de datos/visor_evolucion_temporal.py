"""
================================================================================
MÓDULO: visor_evolucion_temporal.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================

DESCRIPCIÓN GENERAL:
    Visor interactivo en Matplotlib que consume 'salida_con_interpolacion.json'
    para simular la evolución temporal de los cuerpos celestes evaluando sus
    ecuaciones de movimiento RA(t) y Dec(t) en tiempo real mediante un Slider.

CONTROLES INTERACTIVOS EN VENTANA:
    1. Slider 'Fecha (MJD)': Desplaza el tiempo de observación desde la 1ra
       medición real hasta la 5ta medición interpolada.
    2. Slider 'Zoom Movimiento (x)': Amplifica visualmente el desplazamiento
       astrométrico (mas/yr) para auditar la dirección y velocidad de los
       vectores en pantalla.
    3. Botón 'Ir a 5ta Medición': Salta directamente al instante sintetizado.

USO DESDE TERMINAL:
    python3 visor_evolucion_temporal.py
================================================================================
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.widgets import Button, Slider

DIAS_POR_ANIO = 365.25
MAS_A_DEG = 1.0 / (3600.0 * 1000.0)


def evaluar_trayectoria(eq: dict, mjd: float, factor_amplificacion: float = 1.0) -> tuple[float, float]:
    """Evalúa RA(t) y Dec(t) aplicando opcionalmente un factor visual de escala."""
    dt_yr = (mjd - eq["t0_mjd"]) / DIAS_POR_ANIO
    cos_dec0 = max(float(np.cos(np.radians(eq["dec_0_deg"]))), 1e-6)

    d_ra_mas = (eq["mu_ra_cosdec_mas_yr"] * dt_yr + 0.5 * eq["acc_ra_mas_yr2"] * (dt_yr ** 2)) * factor_amplificacion
    d_dec_mas = (eq["mu_dec_mas_yr"] * dt_yr + 0.5 * eq["acc_dec_mas_yr2"] * (dt_yr ** 2)) * factor_amplificacion

    ra_t = eq["ra_0_deg"] + (d_ra_mas * MAS_A_DEG) / cos_dec0
    dec_t = eq["dec_0_deg"] + (d_dec_mas * MAS_A_DEG)
    return float(ra_t), float(dec_t)


def iniciar_visor_interactivo(ruta_json: Path):
    if not ruta_json.exists():
        raise FileNotFoundError(f"No se encontró {ruta_json}. Ejecuta primero generador_interpolacion.py")

    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    cuerpos = datos.get("ecuaciones_trayectoria", [])
    if not cuerpos:
        raise ValueError("No hay ecuaciones de trayectoria en el JSON.")

    mjd_min = float(datos["rango_temporal_mjd"]["mjd_inicio"])
    mjd_max = float(datos["rango_temporal_mjd"]["mjd_fin_interpolado"])
    epochs = datos.get("epochs", [])
    mjds_observados = [float(ep["mjd_promedio"]) for ep in epochs if not ep.get("es_interpolado", False)]

    # Configurar ventana con 2 paneles (Mapa celeste + Espectro 6 bandas)
    fig = plt.figure(figsize=(14, 7.5), facecolor="#090c10")
    plt.subplots_adjust(left=0.07, right=0.96, top=0.88, bottom=0.22, wspace=0.25)

    ax_sky = fig.add_subplot(1, 2, 1)
    ax_spec = fig.add_subplot(1, 2, 2)

    centro = datos["centro_campo"]
    fov_mitad = centro["fov_deg"] / 1.8

    # Estado inicial: amplificación x5000 para notar el movimiento propio estelar en pantalla
    factor_inicial = 5000.0

    def actualizar_grafico(val=None):
        mjd_actual = slider_mjd.val
        amplif = slider_amp.val

        ax_sky.clear()
        ax_spec.clear()
        ax_sky.set_facecolor("#0d1117")
        ax_spec.set_facecolor("#0d1117")

        # Determinar si estamos en una fecha observada o en la zona interpolada
        es_zona_predicha = mjd_actual > (max(mjds_observados) + 5.0)
        estado_txt = "PREDICCIÓN INTERPOLADA (5ta Medición)" if es_zona_predicha else "VENTANA OBSERVADA"
        color_estado = "#ff9f1c" if es_zona_predicha else "#2ec4b6"

        # 1. Dibujar trayectorias completas y posiciones actuales en el cielo
        for c in cuerpos:
            eq = c["ecuacion"]
            # Calcular traza completa de inicio a fin
            ra_ini, dec_ini = evaluar_trayectoria(eq, mjd_min, amplif)
            ra_fin, dec_fin = evaluar_trayectoria(eq, mjd_max, amplif)
            ra_now, dec_now = evaluar_trayectoria(eq, mjd_actual, amplif)

            es_gaia = c["nombre_oficial"].startswith("Gaia DR3")
            color_cuerpo = "#4cc9f0" if es_gaia else "#f72585"

            # Línea punteada de trayectoria predicha
            ax_sky.plot([ra_ini, ra_fin], [dec_ini, dec_fin], color=color_cuerpo, linestyle="--", alpha=0.45, linewidth=1.0)
            # Marca de posición inicial (t0)
            ax_sky.scatter([ra_ini], [dec_ini], s=14, facecolors="none", edgecolors=color_cuerpo, alpha=0.6)
            # Posición actual en el tiempo del slider
            ax_sky.scatter([ra_now], [dec_now], s=42, color=color_cuerpo, edgecolor="white", linewidth=0.6, zorder=4)

            # Elipse morfológica
            a_deg = max((c["semieje_a_arcsec"] / 3600.0) * 2.0, 0.0015)
            b_deg = max((c["semieje_b_arcsec"] / 3600.0) * 2.0, 0.0015)
            elipse = Ellipse(
                xy=(ra_now, dec_now),
                width=a_deg,
                height=b_deg,
                angle=c["angulo_deg"],
                edgecolor=color_cuerpo,
                facecolor=color_cuerpo,
                alpha=0.35,
            )
            ax_sky.add_patch(elipse)

        # Centro de referencia
        ax_sky.scatter([centro["ra"]], [centro["dec"]], s=120, marker="+", color="#ff3b30", zorder=5)

        ax_sky.set_xlim(centro["ra"] + fov_mitad, centro["ra"] - fov_mitad)  # Invertido (Este a la izquierda)
        ax_sky.set_ylim(centro["dec"] - fov_mitad, centro["dec"] + fov_mitad)
        ax_sky.set_xlabel("Ascensión Recta (RA) [deg]", color="#cccccc")
        ax_sky.set_ylabel("Declinación (Dec) [deg]", color="#cccccc")
        ax_sky.set_title(f"Evolución Espacial | MJD = {mjd_actual:.2f}\n[{estado_txt}]", color=color_estado, fontsize=11, fontweight="bold")
        ax_sky.tick_params(colors="#cccccc")
        ax_sky.grid(True, color="white", alpha=0.08, linestyle=":")

        # 2. Dibujar espectro de 6 bandas del cuerpo principal (o el más rápido)
        cuerpo_destacado = max(
            cuerpos,
            key=lambda x: np.hypot(x["ecuacion"]["mu_ra_cosdec_mas_yr"], x["ecuacion"]["mu_dec_mas_yr"]),
        )
        esp = cuerpo_destacado.get("espectro_referencia", [])
        if esp:
            lams = [p["lambda_um"] for p in esp]
            # Simular ligera variación temporal suave en flujo para visualizar la evolución
            factor_flujo = 1.0 + 0.08 * np.sin((mjd_actual - mjd_min) / 60.0)
            flujos = [p["flux_ujy"] * factor_flujo for p in esp]
            colores = [p["color_hex"] for p in esp]

            ax_spec.plot(lams, flujos, color="#8892b0", linestyle="--", linewidth=1.2)
            for lam, fl, col in zip(lams, flujos, colores):
                ax_spec.scatter([lam], [fl], color=col, s=95, edgecolor="white", zorder=3)

            ax_spec.set_title(f"Espectro 6 Bandas: {cuerpo_destacado['nombre_oficial']}\n(µ_RA={cuerpo_destacado['ecuacion']['mu_ra_cosdec_mas_yr']:.1f} mas/yr, µ_Dec={cuerpo_destacado['ecuacion']['mu_dec_mas_yr']:.1f} mas/yr)", color="white", fontsize=10)
            ax_spec.set_xlabel("Longitud de Onda λ [µm]", color="#cccccc")
            ax_spec.set_ylabel("Flujo Calibrado [µJy]", color="#cccccc")
            ax_spec.tick_params(colors="#cccccc")
            ax_spec.grid(True, color="white", alpha=0.1, linestyle=":")

        fig.canvas.draw_idle()

    # Controles inferiores (Sliders y Botón)
    ax_slider_mjd = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor="#161b22")
    slider_mjd = Slider(
        ax=ax_slider_mjd,
        label="Fecha (MJD) ",
        valmin=mjd_min,
        valmax=mjd_max,
        valinit=mjd_min,
        color="#2ec4b6",
    )
    slider_mjd.label.set_color("white")
    slider_mjd.valtext.set_color("white")

    ax_slider_amp = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor="#161b22")
    slider_amp = Slider(
        ax=ax_slider_amp,
        label="Escala Vector (x) ",
        valmin=1.0,
        valmax=25000.0,
        valinit=factor_inicial,
        color="#4cc9f0",
    )
    slider_amp.label.set_color("white")
    slider_amp.valtext.set_color("white")

    ax_btn = plt.axes([0.83, 0.07, 0.13, 0.05])
    btn_5ta = Button(ax_btn, "Ir a 5ta Medición", color="#ff9f1c", hovercolor="#ffb703")

    def ir_a_quinta(event):
        slider_mjd.set_val(mjd_max)

    slider_mjd.on_changed(actualizar_grafico)
    slider_amp.on_changed(actualizar_grafico)
    btn_5ta.on_clicked(ir_a_quinta)

    # Dibujar estado inicial
    actualizar_grafico()
    plt.show()


if __name__ == "__main__":
    dir_actual = Path(__file__).resolve().parent
    ruta_interp = dir_actual / "salida_con_interpolacion.json"
    iniciar_visor_interactivo(ruta_interp)