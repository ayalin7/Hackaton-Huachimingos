import json
import numpy as np
import sep
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

# =====================================================================
# CONFIGURACIÓN DE BANDAS POR TIPO DE OBJETO (ALeRCE -> SPHEREx)
# =====================================================================
# Longitudes de onda objetivo (en micras) según la física del objeto.
# SPHEREx cubre de 0.75 a 5.0 um; aquí definimos las 6 más diagnósticas.
BANDAS_OBJETIVO_UM = {
    "AGN": [0.80, 1.05, 1.65, 2.20, 3.40, 4.60],       # Continuo + líneas H-alfa/Pa-beta + polvo caliente
    "SN": [0.75, 0.90, 1.08, 1.60, 2.15, 3.60],        # Evolución térmica y líneas de hierro/helio
    "YSO": [0.85, 1.25, 2.20, 3.05, 4.25, 4.65],       # Hielos (H2O a 3.05 um, CO2 a 4.25 um, CO a 4.65 um)
    "ASTEROID": [0.78, 0.92, 1.05, 1.50, 2.00, 3.00],  # Silicatos (bandas de 0.9 y 1.9 um) + hidratación
    "DEFAULT": [0.75, 0.82, 0.90, 0.97, 1.03, 1.10]    # Muestreo uniforme en detector D1 (ideal para tu JSON actual)
}


def irsa_url_a_s3(url_irsa: str) -> str:
    """Convierte la URL HTTPS de IRSA a la ruta directa en el bucket S3 de la NASA."""
    prefijo_irsa = "https://irsa.ipac.caltech.edu/ibe/data/spherex/"
    prefijo_s3 = "s3://nasa-irsa-spherex/"
    if url_irsa.startswith(prefijo_irsa):
        return url_irsa.replace(prefijo_irsa, prefijo_s3, 1)
    return url_irsa


# =====================================================================
# FASE 1: AGRUPACIÓN TEMPORAL (EPOCHS) Y SELECCIÓN DE 6 BANDAS
# =====================================================================
def agrupar_por_epochs(cuadros: list, umbral_dias: float = 15.0) -> list:
    """
    Agrupa los cuadros por ventana de observación (Epoch).
    Los barridos cercanos en días (ej. MJD 61180 a 61182) pertenecen al mismo Epoch;
    visitas separadas por meses crean un nuevo Epoch.
    """
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
    """
    Filtra el JSON de entrada para quedarse como máximo con las 6 bandas más cercanas
    a las longitudes de onda diagnósticas en cada Epoch disponible.
    """
    objetivos = BANDAS_OBJETIVO_UM.get(clase_alerce.upper(), BANDAS_OBJETIVO_UM["DEFAULT"])
    epochs_crudos = agrupar_por_epochs(datos_json["cuadros"])
    plan_procesamiento = []

    for idx_epoch, cuadros_epoch in enumerate(epochs_crudos, start=1):
        seleccionados = []
        urls_usadas = set()

        for lam_obj in objetivos:
            # Filtrar candidatos aún no elegidos para no repetir el mismo archivo FITS
            candidatos = [c for c in cuadros_epoch if c["url"] not in urls_usadas]
            if not candidatos:
                break

            mejor_cuadro = min(candidatos, key=lambda c: abs(c["lambda_um"] - lam_obj))
            urls_usadas.add(mejor_cuadro["url"])

            seleccionados.append({
                "fecha_utc": mejor_cuadro["fecha_utc"],
                "mjd": mejor_cuadro["mjd"],
                "lambda_objetivo_um": lam_obj,
                "lambda_real_um": mejor_cuadro["lambda_um"],
                "banda": mejor_cuadro["banda"],
                "s3_uri": irsa_url_a_s3(mejor_cuadro["url"])
            })

        mjd_promedio = float(np.mean([c["mjd"] for c in seleccionados]))
        plan_procesamiento.append({
            "epoch_id": idx_epoch,
            "mjd_promedio": mjd_promedio,
            "cuadros_seleccionados": seleccionados
        })

    return plan_procesamiento


# =====================================================================
# FASE 2: RECORTE ASTROMÉTRICO (Cutout2D) Y EXTRACCIÓN DE FUENTES
# =====================================================================
def procesar_fits_s3_fase2(s3_uri: str, ra_centro: float, dec_centro: float, fov_deg: float, umbral_sigma: float = 3.0) -> list:
    """
    Lee el FITS en streaming desde AWS S3, recorta exactamente alrededor de (RA, Dec)
    y extrae todas las fuentes astronómicas vectorizadas con coordenadas celestes reales.
    """
    with fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        imagen_completa = hdul["IMAGE"].data
        wcs_completo = WCS(hdul["IMAGE"].header)

    # Recorte astrométrico exacto usando coordenadas del cielo (reemplaza el recorte manual)
    centro = SkyCoord(ra=ra_centro * u.deg, dec=dec_centro * u.deg, frame="icrs")
    tamanio = (fov_deg * u.deg, fov_deg * u.deg)
    cutout = Cutout2D(
        imagen_completa,
        position=centro,
        size=tamanio,
        wcs=wcs_completo,
        mode="partial",
        fill_value=np.nan
    )

    # sep exige memoria contigua en float64 nativo (los FITS suelen venir en big-endian)
    sub_img = np.ascontiguousarray(cutout.data, dtype=np.float64)
    mascara_invalida = np.isnan(sub_img)

    # Sustracción del fondo infrarrojo del cielo
    bkg = sep.Background(sub_img, mask=mascara_invalida)
    sub_img_limpia = sub_img - bkg.back()

    # Detección de objetos sobre el umbral de ruido
    objetos = sep.extract(
        sub_img_limpia,
        thresh=umbral_sigma,
        err=bkg.globalrms,
        mask=mascara_invalida
    )

    if len(objetos) == 0:
        return []

    # Transformar coordenadas de píxel (x, y) del recorte a coordenadas celestes (RA, Dec)
    coords_sky = cutout.wcs.pixel_to_world(objetos["x"], objetos["y"])

    fuentes_vectorizadas = []
    for i, obj in enumerate(objetos):
        # Si el semieje mayor supera ~2.5 píxeles, probablemente es una galaxia/objeto extendido
        es_extendido = bool(obj["a"] > 2.5)
        fuentes_vectorizadas.append({
            "ra": float(coords_sky[i].ra.deg),
            "dec": float(coords_sky[i].dec.deg),
            "flujo": float(obj["flux"]),
            "pico_brillo": float(obj["peak"]),
            "semieje_a_px": float(obj["a"]),
            "semieje_b_px": float(obj["b"]),
            "angulo_rad": float(obj["theta"]),
            "tipo_morfologia": "extendido" if es_extendido else "puntual"
        })

    return fuentes_vectorizadas


# =====================================================================
# EJECUCIÓN DE PRUEBA CON TU SAMPLE (spherex.json)
# =====================================================================
def ejecutar_pipeline_fases_1_y_2(ruta_json: str, clase_alerce: str = "DEFAULT") -> dict:
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    ra = datos["ra"]
    dec = datos["dec"]
    fov_deg = datos["fov_deg"]

    # Paso 1: Filtrar de N cuadros a máximo 6 por Epoch
    epochs_filtrados = seleccionar_cuadros_fase1(datos, clase_alerce=clase_alerce)

    resultado = {
        "centro_campo": {"ra": ra, "dec": dec, "fov_deg": fov_deg},
        "clase_alerce": clase_alerce,
        "epochs": []
    }

    # Paso 2: Extraer fuentes vectorizadas de cada FITS seleccionado
    for epoch in epochs_filtrados:
        bandas_procesadas = []
        for cuadro in epoch["cuadros_seleccionados"]:
            print(f"Procesando Epoch {epoch['epoch_id']} | λ={cuadro['lambda_real_um']:.3f} µm -> {cuadro['s3_uri']}")
            fuentes = procesar_fits_s3_fase2(
                s3_uri=cuadro["s3_uri"],
                ra_centro=ra,
                dec_centro=dec,
                fov_deg=fov_deg
            )
            bandas_procesadas.append({
                "mjd": cuadro["mjd"],
                "lambda_um": cuadro["lambda_real_um"],
                "banda": cuadro["banda"],
                "total_fuentes": len(fuentes),
                "fuentes": fuentes
            })

        resultado["epochs"].append({
            "epoch_id": epoch["epoch_id"],
            "mjd_promedio": epoch["mjd_promedio"],
            "bandas": bandas_procesadas
        })

    return resultado


if __name__ == "__main__":
    salida = ejecutar_pipeline_fases_1_y_2("spherex.json", clase_alerce="DEFAULT")
    with open("salida_fase2_vectorizada.json", "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2)
    print("Listo. Guardado en salida_fase2_vectorizada.json")