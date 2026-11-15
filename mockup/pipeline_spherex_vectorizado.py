"""
================================================================================
MÓDULO: pipeline_spherex_vectorizado.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================

DESCRIPCIÓN GENERAL:
    Pipeline principal de procesamiento astrofísico y vectorización de imágenes
    infrarrojas de la misión SPHEREx. Transforma arreglos de píxeles FITS en un
    catálogo estructurado, calibrado y ligero (JSON) listo para alimentar tanto
    el interpolador de ecuaciones de movimiento (motion_interpolator.py) como
    el motor de renderizado web tipo Stellarium (Aladin Lite / WebGL).

ARQUITECTURA Y FASES QUE EJECUTA:
    1. Selección Espectral y Temporal (Fase 1):
       - Agrupa las observaciones por ventanas temporales (Epochs) usando la
         fecha juliana modificada (MJD).
       - Filtra los cuadros disponibles para conservar únicamente las 6 bandas
         infrarrojas más diagnósticas según la clasificación astrofísica del
         objeto entregada por ALeRCE (AGN, SN, YSO, ASTEROID o DEFAULT).
       - Traduce las URLs HTTPS de IRSA a rutas directas del bucket público de
         Amazon S3 (s3://nasa-irsa-spherex/) para lectura en streaming sin
         descargas locales ni saturación de servidores.

    2. Recorte Astrométrico y Extracción de Fuentes (Fase 2):
       - Lee en streaming la extensión científica 'IMAGE' y su sistema de
         coordenadas mundiales (WCS) mediante astropy y s3fs.
       - Realiza un recorte espacial exacto (Cutout2D) centrado en (RA, Dec)
         con el campo de visión solicitado (fov_deg).
       - Modela y sustrae el fondo infrarrojo del cielo y detecta fuentes
         puntuales y extendidas mediante Source Extractor (sep).
       - Calibra la fotometría de brillo superficial (MJy/sr) a flujo integrado
         en microJanskys (flux_ujy) y magnitud AB (mag_ab), estimando además
         el error astrométrico en milisegundos de arco (ra_err_mas, dec_err_mas).

    3. Consolidación Multi-banda y Espectrometría (Fase 3):
       - Cruza las detecciones de las 6 bandas por proximidad celeste
         (match_coordinates_sky) para asignar un identificador único (fuente_id)
         a cada estrella o galaxia del campo.
       - Construye la curva espectral de 6 puntos (espectro) y aísla el
         objeto objetivo más cercano al centro de búsqueda.
       - Inyecta metadatos visuales para el frontend: color hexadecimal por
         longitud de onda (color_hex) y semiejes en segundos de arco.

    4. Control Visual Local (Opcional / Auditoría):
       - Permite renderizar localmente en Matplotlib un mosaico de las 6 bandas
         (o una banda individual) junto con la curva espectral calibrada del
         objeto sin necesidad de volver a consultar AWS S3.

ENTRADAS Y SALIDAS:
    - Entrada: Archivo JSON con 'ra', 'dec', 'fov_deg' y lista 'cuadros'.
    - Salida:  'salida_vectorizada_completa.json' (catálogo vectorizado) y
               gráficos PNG opcionales de control local.

USO DESDE TERMINAL:
    # Ejecutar procesamiento completo desde S3 y graficar las 6 bandas:
    python3 pipeline_spherex_vectorizado.py --clase DEFAULT

    # Cambiar rápidamente la banda graficada leyendo el JSON ya procesado:
    python3 pipeline_spherex_vectorizado.py --solo-graficar --banda 2
================================================================================
"""

import argparse
import json
from pathlib import Path

import astropy.units as u
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import sep
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from astropy.nddata.utils import NoOverlapError

# =====================================================================
# CONFIGURACIÓN DE BANDAS POR TIPO DE OBJETO (ALeRCE -> SPHEREx)
# =====================================================================
BANDAS_OBJETIVO_UM = {
    "AGN": [0.80, 1.05, 1.65, 2.20, 3.40, 4.60],
    "SN": [0.75, 0.90, 1.08, 1.60, 2.15, 3.60],
    "YSO": [0.85, 1.25, 2.20, 3.05, 4.25, 4.65],
    "ASTEROID": [0.78, 0.92, 1.05, 1.50, 2.00, 3.00],
    "DEFAULT": [0.75, 0.82, 0.90, 0.97, 1.03, 1.10]
}

# Constantes instrumentales de SPHEREx
ESCALA_PIXEL_ARCSEC = 6.2
SR_POR_PIXEL = (ESCALA_PIXEL_ARCSEC / 206265.0) ** 2  # ~9.03e-10 estereorradianes


def irsa_url_a_s3(url_irsa: str) -> str:
    """Convierte la URL HTTPS de IRSA a la ruta directa en el bucket S3 de la NASA."""
    prefijo_irsa = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
    prefijo_s3 = "s3://nasa-irsa-spherex/"
    if url_irsa.startswith(prefijo_irsa):
        return url_irsa.replace(prefijo_irsa, prefijo_s3, 1)
    return url_irsa


def lambda_a_color_hex(lambda_um: float, lam_min: float = 0.75, lam_max: float = 5.0) -> str:
    """Mapea una longitud de onda infrarroja a un color HEX continuo (azul -> cian -> amarillo -> rojo)."""
    norm = np.clip((lambda_um - lam_min) / max(lam_max - lam_min, 1e-6), 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(norm)
    return mcolors.to_hex(rgba)


def calibrar_fotometria(flujo_mjy_sr: float, err_mjy_sr: float) -> tuple[float, float, float | None]:
    """Convierte brillo superficial (MJy/sr) a flujo integrado (µJy) y magnitud AB."""
    # 1 MJy = 1e12 µJy
    flux_ujy = float(flujo_mjy_sr * SR_POR_PIXEL * 1e12)
    flux_err_ujy = float(err_mjy_sr * SR_POR_PIXEL * 1e12)

    if flux_ujy > 0:
        # Punto cero AB: 3631 Jy = 3.631e9 µJy
        mag_ab = float(-2.5 * np.log10(flux_ujy / 3.631e9))
    else:
        mag_ab = None

    return flux_ujy, flux_err_ujy, mag_ab


# =====================================================================
# FASE 1: AGRUPACIÓN TEMPORAL (EPOCHS) Y SELECCIÓN DE 6 BANDAS
# =====================================================================
def agrupar_por_epochs(cuadros: list, umbral_dias: float = 15.0) -> list:
    if not cuadros:
        return []
    cuadros_ordenados = sorted(cuadros, key=lambda x: x["mjd"])
    epochs = [[cuadros_ordenados[0]]]
    for cuadro in cuadros_ordenados[1:]:
        if (cuadro["mjd"] - epochs[-1][-1]["mjd"]) <= umbral_dias:
            epochs[-1].append(cuadro)
        else:
            epochs.append([cuadro])
    return epochs


def seleccionar_cuadros_fase1(datos_json: dict, clase_alerce: str = "DEFAULT") -> list:
    objetivos = BANDAS_OBJETIVO_UM.get(clase_alerce.upper(), BANDAS_OBJETIVO_UM["DEFAULT"])
    epochs_crudos = agrupar_por_epochs(datos_json["cuadros"])
    plan = []

    for idx_epoch, cuadros_epoch in enumerate(epochs_crudos, start=1):
        seleccionados = []
        urls_usadas = set()

        for idx_banda, lam_obj in enumerate(objetivos):
            candidatos = [c for c in cuadros_epoch if c["url"] not in urls_usadas]
            if not candidatos:
                break
            mejor = min(candidatos, key=lambda c: abs(c["lambda_um"] - lam_obj))
            urls_usadas.add(mejor["url"])

            seleccionados.append({
                "banda_idx": idx_banda,
                "fecha_utc": mejor["fecha_utc"],
                "mjd": mejor["mjd"],
                "lambda_objetivo_um": lam_obj,
                "lambda_real_um": mejor["lambda_um"],
                "banda": mejor["banda"],
                "color_hex": lambda_a_color_hex(mejor["lambda_um"], min(objetivos), max(objetivos)),
                "s3_uri": irsa_url_a_s3(mejor["url"])
            })

        plan.append({
            "epoch_id": idx_epoch,
            "mjd_promedio": float(np.mean([c["mjd"] for c in seleccionados])),
            "cuadros_seleccionados": seleccionados
        })

    return plan


# =====================================================================
# FASE 2: RECORTE ASTROMÉTRICO Y EXTRACCIÓN VECTORIZADA CALIBRADA
# =====================================================================
def procesar_fits_s3_fase2(s3_uri: str, ra_centro: float, dec_centro: float, fov_deg: float, umbral_sigma: float = 3.0) -> list:
    with fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        imagen_completa = hdul["IMAGE"].data
        wcs_completo = WCS(hdul["IMAGE"].header)

    centro = SkyCoord(ra=ra_centro * u.deg, dec=dec_centro * u.deg, frame="icrs")
    try:
        cutout = Cutout2D(
            imagen_completa,
            position=centro,
            size=(fov_deg * u.deg, fov_deg * u.deg),
            wcs=wcs_completo,
            mode="partial",
            fill_value=np.nan
        )
    except NoOverlapError:
        # El cuadro FITS está cerca en el catálogo pero su sensor no cubre exactamente (ra, dec)
        return []

    sub_img = np.ascontiguousarray(cutout.data, dtype=np.float64)
    mascara_invalida = np.isnan(sub_img)
    if np.all(mascara_invalida):
        return []

    bkg = sep.Background(sub_img, mask=mascara_invalida)
    sub_img_limpia = sub_img - bkg.back()

    objetos = sep.extract(
        sub_img_limpia,
        thresh=umbral_sigma,
        err=bkg.globalrms,
        mask=mascara_invalida
    )

    if len(objetos) == 0:
        return []

    radios_apertura = np.maximum(2.5 * objetos["a"], 2.0)
    flujo_ap, flujo_err_ap, _ = sep.sum_circle(
        sub_img_limpia,
        objetos["x"],
        objetos["y"],
        radios_apertura,
        err=bkg.globalrms,
        mask=mascara_invalida
    )

    coords_sky = cutout.wcs.pixel_to_world(objetos["x"], objetos["y"])
    fuentes = []

    for i, obj in enumerate(objetos):
        flux_ujy, flux_err_ujy, mag_ab = calibrar_fotometria(flujo_ap[i], flujo_err_ap[i])
        snr = max(flux_ujy / max(flux_err_ujy, 1e-6), 1.0)
        fwhm_mas = (2.355 * obj["a"]) * ESCALA_PIXEL_ARCSEC * 1000.0
        pos_err_mas = float(np.clip(fwhm_mas / snr, 5.0, 2000.0))

        es_extendido = bool(obj["a"] > 2.5)
        fuentes.append({
            "ra": float(coords_sky[i].ra.deg),
            "dec": float(coords_sky[i].dec.deg),
            "ra_err_mas": pos_err_mas,
            "dec_err_mas": pos_err_mas,
            "flux_ujy": flux_ujy,
            "flux_err_ujy": flux_err_ujy,
            "mag_ab": mag_ab,
            "semieje_a_arcsec": float(obj["a"] * ESCALA_PIXEL_ARCSEC),
            "semieje_b_arcsec": float(obj["b"] * ESCALA_PIXEL_ARCSEC),
            "angulo_deg": float(np.degrees(obj["theta"])),
            "tipo_morfologia": "extendido" if es_extendido else "puntual"
        })

    return fuentes
# =====================================================================
# FASE 3: EMPAREJAMIENTO MULTI-BANDA Y CONSTRUCCIÓN DE ESPECTROS
# =====================================================================
def consolidar_catalogo_epoch(bandas_procesadas: list, ra_objetivo: float, dec_objetivo: float, radio_match_arcsec: float = 4.0) -> tuple[list, dict | None]:
    """
    Cruza las detecciones de las 6 bandas por coordenadas celestes para construir
    el espectro de cada objeto y aislar el objeto objetivo más cercano al centro.
    """
    catalogo_maestro = []
    coord_centro = SkyCoord(ra=ra_objetivo * u.deg, dec=dec_objetivo * u.deg, frame="icrs")

    for b in bandas_procesadas:
        lam = b["lambda_um"]
        color = b["color_hex"]
        mjd = b["mjd"]

        for f in b["fuentes"]:
            c_fuente = SkyCoord(ra=f["ra"] * u.deg, dec=f["dec"] * u.deg, frame="icrs")
            punto_espectral = {
                "mjd": mjd,
                "lambda_um": lam,
                "color_hex": color,
                "flux_ujy": f["flux_ujy"],
                "flux_err_ujy": f["flux_err_ujy"],
                "mag_ab": f["mag_ab"]
            }

            if not catalogo_maestro:
                catalogo_maestro.append({
                    "fuente_id": "SRC_001",
                    "ra": f["ra"],
                    "dec": f["dec"],
                    "ra_err_mas": f["ra_err_mas"],
                    "dec_err_mas": f["dec_err_mas"],
                    "tipo_morfologia": f["tipo_morfologia"],
                    "semieje_a_arcsec": f["semieje_a_arcsec"],
                    "semieje_b_arcsec": f["semieje_b_arcsec"],
                    "angulo_deg": f["angulo_deg"],
                    "distancia_centro_arcsec": float(c_fuente.separation(coord_centro).arcsec),
                    "espectro": [punto_espectral]
                })
                continue

            coords_existentes = SkyCoord(
                ra=[item["ra"] for item in catalogo_maestro] * u.deg,
                dec=[item["dec"] for item in catalogo_maestro] * u.deg,
                frame="icrs"
            )
            idx, sep2d, _ = match_coordinates_sky(c_fuente, coords_existentes)

            if sep2d.arcsec <= radio_match_arcsec:
                catalogo_maestro[int(idx)]["espectro"].append(punto_espectral)
            else:
                catalogo_maestro.append({
                    "fuente_id": f"SRC_{len(catalogo_maestro) + 1:03d}",
                    "ra": f["ra"],
                    "dec": f["dec"],
                    "ra_err_mas": f["ra_err_mas"],
                    "dec_err_mas": f["dec_err_mas"],
                    "tipo_morfologia": f["tipo_morfologia"],
                    "semieje_a_arcsec": f["semieje_a_arcsec"],
                    "semieje_b_arcsec": f["semieje_b_arcsec"],
                    "angulo_deg": f["angulo_deg"],
                    "distancia_centro_arcsec": float(c_fuente.separation(coord_centro).arcsec),
                    "espectro": [punto_espectral]
                })

    # Ordenar espectros por longitud de onda y detectar el objeto objetivo (más cercano al centro)
    objeto_objetivo = None
    if catalogo_maestro:
        for item in catalogo_maestro:
            item["espectro"].sort(key=lambda x: x["lambda_um"])
        catalogo_maestro.sort(key=lambda x: x["distancia_centro_arcsec"])
        if catalogo_maestro[0]["distancia_centro_arcsec"] <= 15.0:
            objeto_objetivo = catalogo_maestro[0]

    return catalogo_maestro, objeto_objetivo


def ejecutar_pipeline_completo(ruta_json: str, clase_alerce: str = "DEFAULT") -> dict:
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    ra = datos["ra"]
    dec = datos["dec"]
    fov_deg = datos["fov_deg"]

    epochs_filtrados = seleccionar_cuadros_fase1(datos, clase_alerce=clase_alerce)
    resultado = {
        "centro_campo": {"ra": ra, "dec": dec, "fov_deg": fov_deg},
        "clase_alerce": clase_alerce,
        "epochs": []
    }

    for epoch in epochs_filtrados:
        bandas_procesadas = []
        for cuadro in epoch["cuadros_seleccionados"]:
            print(f"Procesando Epoch {epoch['epoch_id']} | Banda {cuadro['banda_idx']} (λ={cuadro['lambda_real_um']:.3f} µm)...")
            fuentes = procesar_fits_s3_fase2(
                s3_uri=cuadro["s3_uri"],
                ra_centro=ra,
                dec_centro=dec,
                fov_deg=fov_deg
            )
            bandas_procesadas.append({
                "banda_idx": cuadro["banda_idx"],
                "mjd": cuadro["mjd"],
                "lambda_um": cuadro["lambda_real_um"],
                "banda": cuadro["banda"],
                "color_hex": cuadro["color_hex"],
                "total_fuentes": len(fuentes),
                "fuentes": fuentes
            })

        catalogo_consolidado, objeto_objetivo = consolidar_catalogo_epoch(bandas_procesadas, ra, dec)

        resultado["epochs"].append({
            "epoch_id": epoch["epoch_id"],
            "mjd_promedio": epoch["mjd_promedio"],
            "objeto_objetivo": objeto_objetivo,
            "catalogo_campo": catalogo_consolidado,
            "bandas": bandas_procesadas
        })

    return resultado


# =====================================================================
# HERRAMIENTA DE CONTROL VISUAL LOCAL (REEMPLAZA A imagenes.py)
# =====================================================================
def renderizar_control_local(datos: dict, epoch_idx: int = 0, banda_idx: int | None = None, carpeta_salida: Path = Path(".")):
    """
    Permite visualizar una banda específica (--banda 0..5), una grilla con las 6 bandas
    (--banda None) y la curva espectral calibrada del objeto central.
    """
    epoch = datos["epochs"][epoch_idx]
    bandas = epoch["bandas"]
    if not bandas:
        print("No hay bandas procesadas para graficar.")
        return

    # 1. Graficar una banda específica o las 6 bandas en mosaico
    bandas_a_dibujar = [bandas[banda_idx]] if (banda_idx is not None and 0 <= banda_idx < len(bandas)) else bandas
    cols = 1 if len(bandas_a_dibujar) == 1 else 3
    rows = int(np.ceil(len(bandas_a_dibujar) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), facecolor="#0b0e14", squeeze=False)
    for idx, b in enumerate(bandas_a_dibujar):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("#0b0e14")
        fuentes = b["fuentes"]

        for obj in fuentes:
            # Convertir arcsec a grados para graficar en el plano RA/Dec
            a_deg = max(obj["semieje_a_arcsec"] / 3600.0, 0.0008)
            b_deg = max(obj["semieje_b_arcsec"] / 3600.0, 0.0008)
            color = b["color_hex"] if obj["tipo_morfologia"] == "puntual" else "#00f5d4"

            elipse = Ellipse(
                xy=(obj["ra"], obj["dec"]),
                width=a_deg * 2,
                height=b_deg * 2,
                angle=obj["angulo_deg"],
                edgecolor=color,
                facecolor=color,
                alpha=0.75
            )
            ax.add_patch(elipse)

        ras = [obj["ra"] for obj in fuentes]
        decs = [obj["dec"] for obj in fuentes]
        ax.scatter(ras, decs, s=6, c="white", alpha=0.85)
        ax.scatter([datos["centro_campo"]["ra"]], [datos["centro_campo"]["dec"]], s=90, marker="+", c="red", label="Objetivo")

        ax.set_title(f"Banda {b['banda_idx']} | λ = {b['lambda_um']:.3f} µm ({len(fuentes)} fuentes)", color="white", fontsize=11)
        ax.set_xlabel("RA [deg]", color="#cccccc")
        ax.set_ylabel("Dec [deg]", color="#cccccc")
        ax.tick_params(colors="#cccccc")
        ax.invert_xaxis()

    # Apagar subplots sobrantes si hay menos de 6 bandas
    for idx in range(len(bandas_a_dibujar), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    sufijo = f"banda_{banda_idx}" if banda_idx is not None else "6_bandas"
    ruta_mapa = carpeta_salida / f"render_campo_epoch{epoch['epoch_id']}_{sufijo}.png"
    plt.tight_layout()
    plt.savefig(ruta_mapa, dpi=180)
    plt.close(fig)
    print(f"Mapa vectorial guardado en: {ruta_mapa}")

    # 2. Graficar la curva espectral (espectrometría de 6 bandas) del objeto objetivo
    obj_central = epoch.get("objeto_objetivo")
    if obj_central and len(obj_central["espectro"]) > 1:
        esp = obj_central["espectro"]
        lams = [p["lambda_um"] for p in esp]
        flujos = [p["flux_ujy"] for p in esp]
        errores = [p["flux_err_ujy"] for p in esp]
        colores = [p["color_hex"] for p in esp]

        fig_esp, ax_esp = plt.subplots(figsize=(8, 4.5), facecolor="#0b0e14")
        ax_esp.set_facecolor("#0b0e14")
        ax_esp.plot(lams, flujos, color="#8892b0", linestyle="--", linewidth=1.2, zorder=1)
        ax_esp.errorbar(lams, flujos, yerr=errores, fmt="none", ecolor="#ffffff", alpha=0.5, zorder=2)

        for lam, fl, col in zip(lams, flujos, colores):
            ax_esp.scatter([lam], [fl], color=col, s=90, edgecolor="white", zorder=3)

        ax_esp.set_title(f"Espectro Infrarrojo SPHEREx (6 Bandas) - {obj_central['fuente_id']}", color="white")
        ax_esp.set_xlabel("Longitud de Onda λ [µm]", color="#cccccc")
        ax_esp.set_ylabel("Flujo Calibrado [µJy]", color="#cccccc")
        ax_esp.grid(True, alpha=0.15, color="white")
        ax_esp.tick_params(colors="#cccccc")

        ruta_esp = carpeta_salida / f"render_espectro_epoch{epoch['epoch_id']}.png"
        plt.tight_layout()
        plt.savefig(ruta_esp, dpi=180)
        plt.close(fig_esp)
        print(f"Curva espectral guardada en: {ruta_esp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Vectorizado SPHEREx + Visualizador por Banda")
    parser.add_argument("--clase", type=str, default="DEFAULT", help="Clase ALeRCE: AGN, SN, YSO, ASTEROID, DEFAULT")
    parser.add_argument("--banda", type=int, default=None, help="Índice de banda a graficar (0 a 5). Si se omite, grafica las 6.")
    parser.add_argument("--solo-graficar", action="store_true", help="No descarga FITS de S3; lee el JSON ya generado y cambia la banda graficada.")
    args = parser.parse_args()

    dir_actual = Path(__file__).resolve().parent
    ruta_entrada = dir_actual / "spherex.json"
    ruta_salida = dir_actual / "salida_vectorizada_completa.json"

    if args.solo_graficar and ruta_salida.exists():
        print("Cargando JSON existente para renderizar...")
        with open(ruta_salida, "r", encoding="utf-8") as f:
            salida = json.load(f)
    else:
        salida = ejecutar_pipeline_completo(str(ruta_entrada), clase_alerce=args.clase)
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(salida, f, indent=2)
        print(f"JSON vectorizado guardado en: {ruta_salida}")

    renderizar_control_local(salida, epoch_idx=0, banda_idx=args.banda, carpeta_salida=dir_actual)