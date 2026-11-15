"""
================================================================================
MÓDULO: graficador_universal.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================

DESCRIPCIÓN GENERAL:
    Visualizador independiente y genérico para auditar cualquiera de los JSON
    generados durante el desarrollo del pipeline (desde las primeras salidas
    cruda de Fase 2 hasta el catálogo calibrado e identificado con Gaia DR3 y
    ALeRCE).

CARACTERÍSTICAS:
    - Autodetección de esquema JSON:
        * Modo Catálogo Consolidado ('catalogo_campo'): Dibuja todas las fuentes
          cruzadas en las 6 bandas, resalta el 'objeto_objetivo' y etiqueta en
          el gráfico las estrellas identificadas con nombre oficial ('Gaia DR3').
        * Modo Banda Individual ('bandas[i]'): Dibuja una banda espectral
          específica usando su color real ('color_hex') o por morfología.
        * Compatibilidad retroactiva: Soporta tanto llaves antiguas ('flujo',
          'semieje_a_px', 'angulo_rad') como las nuevas calibradas ('flux_ujy',
          'semieje_a_arcsec', 'angulo_deg').
    - Modo Interactivo / CLI: Permite pasar cualquier ruta de JSON, época y
      banda por argumentos de terminal sin tocar el código.

USO DESDE TERMINAL:
    # 1. Graficar automáticamente el último JSON identificado con Gaia/ALeRCE:
    python3 graficador_universal.py

    # 2. Pasar cualquier JSON específico:
    python3 graficador_universal.py salida_s3_real_grande.json

    # 3. Ver una banda específica (0 a 5) de un archivo:
    python3 graficador_universal.py salida_vectorizada_completa.json --banda 2

    # 4. Guardar sin abrir ventana emergente:
    python3 graficador_universal.py salida_identificada_gaia_alerce.json --no-show
================================================================================
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def normalizar_fuente(obj: dict, color_banda: str = "#ffd166") -> dict:
    """
    Adapta cualquier diccionario de fuente (esquema antiguo o nuevo) a un
    formato visual uniforme para Matplotlib.
    """
    ra = float(obj["ra"])
    dec = float(obj["dec"])

    # Flujo (soporta 'flux_ujy' calibrado o 'flujo' crudo)
    flujo = float(obj.get("flux_ujy") or obj.get("flujo") or 1.0)

    # Semiejes en grados (soporta arcsec calibrados o píxeles antiguos * 6.2'')
    if "semieje_a_arcsec" in obj:
        a_deg = (float(obj["semieje_a_arcsec"]) / 3600.0) * 2.0
        b_deg = (float(obj["semieje_b_arcsec"]) / 3600.0) * 2.0
    else:
        escala_px_deg = 6.2 / 3600.0
        a_deg = float(obj.get("semieje_a_px", 1.5)) * escala_px_deg * 2.0
        b_deg = float(obj.get("semieje_b_px", 1.5)) * escala_px_deg * 2.0

    # Garantizar un tamaño mínimo visible en pantalla
    a_deg = max(a_deg, 0.0012)
    b_deg = max(b_deg, 0.0012)

    # Ángulo en grados (soporta 'angulo_deg' o 'angulo_rad')
    if "angulo_deg" in obj:
        angulo_deg = float(obj["angulo_deg"])
    else:
        angulo_deg = float(np.degrees(obj.get("angulo_rad", 0.0)))

    # Color según tipo de objeto o color espectral de la banda
    morf = obj.get("tipo_morfologia", "puntual")
    clasif = obj.get("clasificacion_objeto", "")

    if "Estrella (Gaia)" in clasif:
        color = "#4cc9f0"  # Azul celeste para estrellas confirmadas por Gaia
    elif morf == "extendido":
        color = "#f72585"  # Magenta para objetos extendidos / galaxias
    else:
        color = color_banda

    nombre = obj.get("nombre_oficial") or obj.get("fuente_id") or ""

    return {
        "ra": ra,
        "dec": dec,
        "flujo": flujo,
        "a_deg": a_deg,
        "b_deg": b_deg,
        "angulo_deg": angulo_deg,
        "color": color,
        "nombre": nombre,
        "es_gaia": nombre.startswith("Gaia DR3"),
        "morfologia": morf,
    }


def graficar_json(
    ruta_json: Path,
    epoch_idx: int = 0,
    banda_idx: int | None = None,
    mostrar_nombres: bool = True,
    mostrar_ventana: bool = True,
):
    if not ruta_json.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_json}")

    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    epochs = datos.get("epochs", [])
    if not epochs:
        raise ValueError("El JSON no contiene el bloque 'epochs' o está vacío.")

    epoch_idx = min(max(epoch_idx, 0), len(epochs) - 1)
    epoch = epochs[epoch_idx]

    # Decidir qué lista de fuentes graficar:
    # - Si el usuario pide una banda específica (--banda N), usa epoch['bandas'][N]['fuentes']
    # - Si existe 'catalogo_campo' (JSON consolidado + Gaia) y no pidió banda, usa ese catálogo completo
    # - Si es un JSON antiguo sin 'catalogo_campo', usa epoch['bandas'][0]['fuentes']
    fuentes_raw = []
    color_base = "#ffd166"
    subtitulo = ""

    if banda_idx is not None and "bandas" in epoch and len(epoch["bandas"]) > 0:
        b_idx = min(max(banda_idx, 0), len(epoch["bandas"]) - 1)
        banda = epoch["bandas"][b_idx]
        fuentes_raw = banda.get("fuentes", [])
        color_base = banda.get("color_hex", "#ffd166")
        subtitulo = f"Banda {b_idx} ({banda.get('banda', 'SPHEREx')}) | λ = {banda.get('lambda_um', 0):.3f} µm"
    elif "catalogo_campo" in epoch and epoch["catalogo_campo"]:
        fuentes_raw = epoch["catalogo_campo"]
        subtitulo = "Catálogo Consolidado Multi-banda + Cotejo Gaia DR3"
    elif "bandas" in epoch and len(epoch["bandas"]) > 0:
        banda = epoch["bandas"][0]
        fuentes_raw = banda.get("fuentes", [])
        color_base = banda.get("color_hex", "#ffd166")
        subtitulo = f"Banda 0 ({banda.get('banda', 'SPHEREx')}) | λ = {banda.get('lambda_um', 0):.3f} µm"
    else:
        raise ValueError("No se encontraron fuentes graficables en el JSON.")

    if not fuentes_raw:
        print(f"Advertencia: El archivo {ruta_json.name} tiene 0 fuentes detectadas en esta selección.")
        return

    fuentes_norm = [normalizar_fuente(obj, color_base) for obj in fuentes_raw]

    # Configurar lienzo astronómico
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#090c10")
    ax.set_facecolor("#090c10")

    for f_obj in fuentes_norm:
        elipse = Ellipse(
            xy=(f_obj["ra"], f_obj["dec"]),
            width=f_obj["a_deg"],
            height=f_obj["b_deg"],
            angle=f_obj["angulo_deg"],
            edgecolor=f_obj["color"],
            facecolor=f_obj["color"],
            alpha=0.72,
            linewidth=1.1,
        )
        ax.add_patch(elipse)

    ras = [f_obj["ra"] for f_obj in fuentes_norm]
    decs = [f_obj["dec"] for f_obj in fuentes_norm]
    ax.scatter(ras, decs, s=8, c="white", alpha=0.85, zorder=3)

    # Marcar el centro de búsqueda / objeto objetivo
    centro = datos.get("centro_campo")
    if centro:
        ax.scatter(
            [centro["ra"]],
            [centro["dec"]],
            s=140,
            marker="+",
            c="#ff3b30",
            linewidths=1.5,
            zorder=4,
            label="Centro / Objetivo",
        )

    # Etiquetar fuentes identificadas con Gaia DR3 (máximo 15 más brillantes para no saturar)
    if mostrar_nombres:
        gaia_objs = [f_obj for f_obj in fuentes_norm if f_obj["es_gaia"]]
        gaia_objs.sort(key=lambda x: x["flujo"], reverse=True)
        for g_obj in gaia_objs[:15]:
            # Mostrar solo los últimos dígitos del ID de Gaia para legibilidad limpia
            id_corto = g_obj["nombre"].replace("Gaia DR3 ", "G3:")
            ax.annotate(
                id_corto,
                xy=(g_obj["ra"], g_obj["dec"]),
                xytext=(4, 4),
                textcoords="offset points",
                color="#90e0ef",
                fontsize=7.5,
                alpha=0.9,
            )

    # Información de ALeRCE si existe en el JSON
    alerce_info = datos.get("alerce_info", {})
    tag_alerce = f" | ALeRCE: {alerce_info['oid']} ({alerce_info['clase']})" if alerce_info.get("oid") else ""

    ax.set_title(
        f"{ruta_json.name} — Epoch {epoch.get('epoch_id', 1)} ({len(fuentes_norm)} fuentes){tag_alerce}\n{subtitulo}",
        color="white",
        fontsize=11,
        pad=12,
    )
    ax.set_xlabel("Ascensión Recta (RA) [deg]", color="#cccccc")
    ax.set_ylabel("Declinación (Dec) [deg]", color="#cccccc")
    ax.tick_params(colors="#cccccc")
    ax.grid(True, color="white", alpha=0.1, linestyle=":")
    ax.invert_xaxis()  # Convención astronómica: Este hacia la izquierda

    # Leyenda visual compacta
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0], [0], marker="o", color="w", label="Estrella (Gaia DR3)", markerfacecolor="#4cc9f0", markersize=8, linestyle="None"),
        Line2D([0], [0], marker="o", color="w", label="Extendido / Galaxia", markerfacecolor="#f72585", markersize=8, linestyle="None"),
        Line2D([0], [0], marker="o", color="w", label="Fuente IR / Banda", markerfacecolor=color_base, markersize=8, linestyle="None"),
        Line2D([0], [0], marker="+", color="#ff3b30", label="Centro de búsqueda", markersize=10, linestyle="None"),
    ]
    ax.legend(handles=leyenda, loc="lower left", facecolor="#161b22", edgecolor="#30363d", labelcolor="white", fontsize=8.5)

    plt.tight_layout()
    ruta_png = ruta_json.with_name(f"grafico_{ruta_json.stem}_ep{epoch.get('epoch_id', 1)}.png")
    plt.savefig(ruta_png, dpi=200)
    print(f"Gráfico guardado exitosamente en: {ruta_png}")

    if mostrar_ventana:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graficador Universal de JSONs Vectorizados SPHEREx")
    parser.add_argument(
        "archivo",
        nargs="?",
        default=None,
        help="Nombre o ruta del archivo JSON a graficar. Si se omite, busca automáticamente el más reciente.",
    )
    parser.add_argument("--epoch", type=int, default=0, help="Índice del Epoch a graficar (0 = primero).")
    parser.add_argument("--banda", type=int, default=None, help="Índice de banda específica (0 a 5). Si se omite, usa el catálogo consolidado.")
    parser.add_argument("--sin-nombres", action="store_true", help="Oculta las etiquetas de texto de Gaia DR3 en el mapa.")
    parser.add_argument("--no-show", action="store_true", help="Solo guarda el PNG sin abrir ventana interactiva.")
    args = parser.parse_args()

    dir_actual = Path(__file__).resolve().parent

    if args.archivo:
        ruta_objetivo = Path(args.archivo)
        if not ruta_objetivo.is_absolute():
            ruta_objetivo = dir_actual / ruta_objetivo
    else:
        # Prioridad automática de archivos generados en tu carpeta
        candidatos = [
            "salida_identificada_gaia_alerce.json",
            "salida_s3_real_grande.json",
            "salida_vectorizada_completa.json",
            "salida_fase2_vectorizada.json",
        ]
        ruta_objetivo = next((dir_actual / c for c in candidatos if (dir_actual / c).exists()), None)
        if ruta_objetivo is None:
            raise FileNotFoundError("No se encontró ningún JSON de salida en esta carpeta. Pasa la ruta como argumento.")

    graficar_json(
        ruta_json=ruta_objetivo,
        epoch_idx=args.epoch,
        banda_idx=args.banda,
        mostrar_nombres=not args.sin_nombres,
        mostrar_ventana=not args.no_show,
    )