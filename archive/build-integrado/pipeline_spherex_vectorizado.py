"""
================================================================================
MÓDULO: pipeline_spherex_vectorizado.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================
Pipeline principal de procesamiento astrofísico y vectorización de imágenes
infrarrojas de la misión SPHEREx. Transforma arreglos de píxeles FITS en un
catálogo estructurado, calibrado y ligero (JSON).
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
from astropy.nddata.utils import NoOverlapError
from astropy.wcs import WCS
from matplotlib.patches import Ellipse

BANDAS_OBJETIVO_UM = {
    "AGN": [0.80, 1.05, 1.65, 2.20, 3.40, 4.60],
    "SN": [0.75, 0.90, 1.08, 1.60, 2.15, 3.60],
    "YSO": [0.85, 1.25, 2.20, 3.05, 4.25, 4.65],
    "BROWN_DWARF": [0.85, 1.15, 1.60, 2.20, 3.35, 4.65],
    "ASTEROID": [0.78, 0.92, 1.05, 1.50, 2.00, 3.00],
    "DEFAULT": [0.75, 0.82, 0.90, 0.97, 1.03, 1.10],
}

ESCALA_PIXEL_ARCSEC = 6.2
SR_POR_PIXEL = (ESCALA_PIXEL_ARCSEC / 206265.0) ** 2


def irsa_url_a_s3(url_irsa: str) -> str:
    """Convierte la URL HTTPS de IRSA a la ruta directa en el bucket S3 de la NASA."""
    prefijo_irsa = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
    prefijo_s3 = "s3://nasa-irsa-spherex/"
    if url_irsa.startswith(prefijo_irsa):
        return url_irsa.replace(prefijo_irsa, prefijo_s3, 1)
    return url_irsa


def lambda_a_color_hex(lambda_um: float, lam_min: float = 0.75, lam_max: float = 5.0) -> str:
    """Mapea una longitud de onda infrarroja a un color HEX continuo."""
    norm = np.clip((lambda_um - lam_min) / max(lam_max - lam_min, 1e-6), 0.0, 1.0)
    rgba = plt.get_cmap("turbo")(norm)
    return mcolors.to_hex(rgba)


def calibrar_fotometria(flujo_mjy_sr: float, err_mjy_sr: float) -> tuple[float, float, float | None]:
    """
    Convierte brillo superficial integrado en apertura (MJy/sr) a flujo (µJy)
    y magnitud AB de apertura fija (sin corrección de PSF).
    """
    flux_ujy = float(flujo_mjy_sr * SR_POR_PIXEL * 1e12)
    flux_err_ujy = float(err_mjy_sr * SR_POR_PIXEL * 1e12)

    if flux_ujy > 0:
        mag_ab = float(-2.5 * np.log10(flux_ujy / 3.631e9))
    else:
        mag_ab = None

    return flux_ujy, flux_err_ujy, mag_ab


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
                "s3_uri": irsa_url_a_s3(mejor["url"]),
            })

        plan.append({
            "epoch_id": idx_epoch,
            "mjd_promedio": float(np.mean([c["mjd"] for c in seleccionados])),
            "cuadros_seleccionados": seleccionados,
        })

    return plan


def procesar_fits_s3_fase2(s3_uri: str, ra_centro: float, dec_centro: float, fov_deg: float, umbral_sigma: float = 3.0) -> list:
    with fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        imagen_completa = hdul["IMAGE"].data
        wcs_completo = WCS(hdul["IMAGE"].header)
        varianza_completa = hdul["VARIANCE"].data if "VARIANCE" in [h.name.upper() for h in hdul] else None

    centro = SkyCoord(ra=ra_centro * u.deg, dec=dec_centro * u.deg, frame="icrs")
    try:
        cutout = Cutout2D(
            imagen_completa,
            position=centro,
            size=(fov_deg * u.deg, fov_deg * u.deg),
            wcs=wcs_completo,
            mode="partial",
            fill_value=np.nan,
        )
    except NoOverlapError:
        return []

    sub_img = np.ascontiguousarray(cutout.data, dtype=np.float64)
    mascara_invalida = np.isnan(sub_img)
    if np.all(mascara_invalida):
        return []

    bkg = sep.Background(sub_img, mask=mascara_invalida)
    sub_img_limpia = sub_img - bkg.back()

    err_mapa = bkg.globalrms
    if varianza_completa is not None:
        try:
            cut_var = Cutout2D(
                varianza_completa,
                position=centro,
                size=(fov_deg * u.deg, fov_deg * u.deg),
                wcs=wcs_completo,
                mode="partial",
                fill_value=np.nan,
            )
            var_arr = np.ascontiguousarray(cut_var.data, dtype=np.float64)
            err_mapa = np.sqrt(np.where(np.isfinite(var_arr) & (var_arr > 0), var_arr, bkg.globalrms**2))
        except Exception:
            err_mapa = bkg.globalrms

    objetos = sep.extract(
        sub_img_limpia,
        thresh=umbral_sigma,
        err=err_mapa,
        mask=mascara_invalida,
    )

    if len(objetos) == 0:
        return []

    radios_apertura = np.maximum(2.5 * objetos["a"], 2.0)
    flujo_ap, flujo_err_ap, _ = sep.sum_circle(
        sub_img_limpia,
        objetos["x"],
        objetos["y"],
        radios_apertura,
        err=err_mapa,
        mask=mascara_invalida,
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
            "tipo_morfologia": "extendido" if es_extendido else "puntual",
        })

    return fuentes


def consolidar_catalogo_epoch(bandas_procesadas: list, ra_objetivo: float, dec_objetivo: float, radio_match_arcsec: float = 8.0) -> tuple[list, dict | None]:
    catalogo_maestro = []
    coord_centro = SkyCoord(ra=ra_objetivo * u.deg, dec=dec_objetivo * u.deg, frame="icrs")

    for b in bandas_procesadas:
        fuentes = b["fuentes"]
        if not fuentes:
            continue
        lam, color, mjd = b["lambda_um"], b["color_hex"], b["mjd"]
        coords_banda = SkyCoord(ra=[f["ra"] for f in fuentes] * u.deg, dec=[f["dec"] for f in fuentes] * u.deg, frame="icrs")
        seps_centro = coords_banda.separation(coord_centro).arcsec

        if not catalogo_maestro:
            for idx_f, f in enumerate(fuentes):
                catalogo_maestro.append({
                    "fuente_id": f"SRC_{idx_f + 1:03d}",
                    "ra": f["ra"],
                    "dec": f["dec"],
                    "ra_err_mas": f["ra_err_mas"],
                    "dec_err_mas": f["dec_err_mas"],
                    "tipo_morfologia": f["tipo_morfologia"],
                    "semieje_a_arcsec": f["semieje_a_arcsec"],
                    "semieje_b_arcsec": f["semieje_b_arcsec"],
                    "angulo_deg": f["angulo_deg"],
                    "distancia_centro_arcsec": float(seps_centro[idx_f]),
                    "espectro": [{"mjd": mjd, "lambda_um": lam, "color_hex": color, "flux_ujy": f["flux_ujy"], "flux_err_ujy": f["flux_err_ujy"], "mag_ab": f["mag_ab"]}],
                })
            continue

        coords_existentes = SkyCoord(ra=[item["ra"] for item in catalogo_maestro] * u.deg, dec=[item["dec"] for item in catalogo_maestro] * u.deg, frame="icrs")
        idxs, sep2d, _ = match_coordinates_sky(coords_banda, coords_existentes)

        for idx_f, f in enumerate(fuentes):
            punto = {"mjd": mjd, "lambda_um": lam, "color_hex": color, "flux_ujy": f["flux_ujy"], "flux_err_ujy": f["flux_err_ujy"], "mag_ab": f["mag_ab"]}
            if float(sep2d[idx_f].arcsec) <= radio_match_arcsec:
                catalogo_maestro[int(idxs[idx_f])]["espectro"].append(punto)
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
                    "distancia_centro_arcsec": float(seps_centro[idx_f]),
                    "espectro": [punto],
                })

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
        "epochs": [],
    }

    for epoch in epochs_filtrados:
        bandas_procesadas = []
        for cuadro in epoch["cuadros_seleccionados"]:
            print(f"Procesando Epoch {epoch['epoch_id']} | Banda {cuadro['banda_idx']} (λ={cuadro['lambda_real_um']:.3f} µm)...")
            fuentes = procesar_fits_s3_fase2(
                s3_uri=cuadro["s3_uri"],
                ra_centro=ra,
                dec_centro=dec,
                fov_deg=fov_deg,
            )
            bandas_procesadas.append({
                "banda_idx": cuadro["banda_idx"],
                "mjd": cuadro["mjd"],
                "lambda_um": cuadro["lambda_real_um"],
                "banda": cuadro["banda"],
                "color_hex": cuadro["color_hex"],
                "total_fuentes": len(fuentes),
                "fuentes": fuentes,
            })

        catalogo_consolidado, objeto_objetivo = consolidar_catalogo_epoch(bandas_procesadas, ra, dec)

        resultado["epochs"].append({
            "epoch_id": epoch["epoch_id"],
            "mjd_promedio": epoch["mjd_promedio"],
            "tipo_medicion": "observada_spherex",
            "objeto_objetivo": objeto_objetivo,
            "catalogo_campo": catalogo_consolidado,
            "bandas": bandas_procesadas,
        })

    return resultado


def renderizar_control_local(datos: dict, epoch_idx: int = 0, banda_idx: int | None = None, carpeta_salida: Path = Path(".")):
    epoch = datos["epochs"][epoch_idx]
    bandas = epoch["bandas"]
    if not bandas:
        print("No hay bandas procesadas para graficar.")
        return

    bandas_a_dibujar = [bandas[banda_idx]] if (banda_idx is not None and 0 <= banda_idx < len(bandas)) else bandas
    cols = 1 if len(bandas_a_dibujar) == 1 else 3
    rows = int(np.ceil(len(bandas_a_dibujar) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows), facecolor="#0b0e14", squeeze=False)
    for idx, b in enumerate(bandas_a_dibujar):
        ax = axes[idx // cols][idx % cols]
        ax.set_facecolor("#0b0e14")
        fuentes = b["fuentes"]

        for obj in fuentes:
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
                alpha=0.75,
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

    for idx in range(len(bandas_a_dibujar), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    sufijo = f"banda_{banda_idx}" if banda_idx is not None else "6_bandas"
    ruta_mapa = carpeta_salida / f"render_campo_epoch{epoch['epoch_id']}_{sufijo}.png"
    plt.tight_layout()
    plt.savefig(ruta_mapa, dpi=180)
    plt.close(fig)
    print(f"Mapa vectorial guardado en: {ruta_mapa}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Vectorizado SPHEREx + Visualizador por Banda")
    parser.add_argument("--clase", type=str, default="DEFAULT", help="Clase ALeRCE: AGN, SN, YSO, BROWN_DWARF, ASTEROID, DEFAULT")
    parser.add_argument("--banda", type=int, default=None, help="Índice de banda a graficar (0 a 5)")
    parser.add_argument("--solo-graficar", action="store_true", help="Lee el JSON ya generado y cambia la banda graficada")
    args = parser.parse_args()

    dir_actual = Path(__file__).resolve().parent
    ruta_entrada = dir_actual / "spherex.json"
    ruta_salida = dir_actual / "salida_vectorizada_completa.json"

    if args.solo_graficar and ruta_salida.exists():
        with open(ruta_salida, "r", encoding="utf-8") as f:
            salida = json.load(f)
    else:
        salida = ejecutar_pipeline_completo(str(ruta_entrada), clase_alerce=args.clase)
        with open(ruta_salida, "w", encoding="utf-8") as f:
            json.dump(salida, f, indent=2)

    renderizar_control_local(salida, epoch_idx=0, banda_idx=args.banda, carpeta_salida=dir_actual)