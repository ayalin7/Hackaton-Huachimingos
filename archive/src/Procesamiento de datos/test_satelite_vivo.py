"""
================================================================================
MÓDULO: test_satelite_vivo.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================

DESCRIPCIÓN GENERAL:
    Script de pruebas de integración y estrés en vivo contra el repositorio
    público de la NASA en Amazon S3 (s3://nasa-irsa-spherex/). Permite validar
    el funcionamiento de 'pipeline_spherex_vectorizado.py' sobre campos
    astronómicos reales y amplios (fov_deg = 0.25°) sin depender de archivos
    de muestra estáticos ni sufrir timeouts del servidor web de IRSA.

FLUJO DE EJECUCIÓN:
    1. Exploración en Nube (S3):
       - Conecta de forma anónima al bucket público de SPHEREx mediante s3fs.
       - Lista los archivos FITS de una campaña de barrido real (level2) y lee
         únicamente sus cabeceras astrométricas (hdul["IMAGE"].header) en
         streaming para extraer coordenadas (CRVAL1, CRVAL2), fecha (MJD-OBS)
         y detector sin descargar los datos de imagen completos.
       - Agrupa automáticamente los cuadros que solapan dentro de un radio de
         1.5 grados alrededor de una coordenada de referencia.

    2. Generación de Entrada Dinámica:
       - Construye y guarda 'input_s3_real_grande.json' replicando exactamente
         el contrato de entrada que entregará el backend FastAPI en producción.

    3. Ejecución del Pipeline y Renderizado:
       - Invoca 'ejecutar_pipeline_completo()' sobre el campo amplio (0.25°),
         exporta el catálogo vectorizado en 'salida_s3_real_grande.json' y
         genera las gráficas de control (mosaico de 6 bandas y espectro).

DEPENDENCIAS:
    - Requiere que 'pipeline_spherex_vectorizado.py' se encuentre en el mismo
      directorio.
    - Librerías externas: s3fs, astropy.

USO DESDE TERMINAL:
    python3 test_satelite_vivo.py
================================================================================
"""
import json
from pathlib import Path

import astropy.units as u
import s3fs
from astropy.coordinates import SkyCoord
from astropy.io import fits

from pipeline_spherex_vectorizado import (
    ejecutar_pipeline_completo,
    renderizar_control_local,
)


def explorar_spherex_directo_s3(carpeta_s3: str = "nasa-irsa-spherex/qr3/level2/2026W21_1A/l2b-v27-2026-268/1/", max_archivos: int = 30) -> dict:
    """
    Lista archivos FITS reales directamente desde el bucket público de AWS S3 de la NASA
    (evitando los timeouts de irsa.ipac.caltech.edu) y extrae sus coordenadas y bandas.
    """
    fs = s3fs.S3FileSystem(anon=True)
    print(f"Listando archivos en s3://{carpeta_s3} ...")
    archivos = sorted(fs.glob(f"{carpeta_s3}*.fits"))

    if not archivos:
        raise RuntimeError(f"No se encontraron FITS en {carpeta_s3}")

    print(f"Total de FITS en esa carpeta: {len(archivos)}. Inspeccionando los primeros {max_archivos}...")

    cuadros = []
    ra_ref, dec_ref = None, None

    for ruta_s3 in archivos[:max_archivos]:
        uri = f"s3://{ruta_s3}"
        # Leer solo la cabecera (header) en streaming: tarda milisegundos y descarga pocos KB
        with fits.open(uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
            hdr = hdul["IMAGE"].header
            ra_cuadro = float(hdr.get("CRVAL1", 0.0))
            dec_cuadro = float(hdr.get("CRVAL2", 0.0))
            mjd_obs = float(hdr.get("MJD-OBS", 61180.0))
            banda = str(hdr.get("DETECTOR", "SPHEREx-D1"))

            # Fijar el centro del campo en el primer cuadro válido
            if ra_ref is None:
                ra_ref, dec_ref = ra_cuadro, dec_cuadro

            # Verificar si este cuadro solapa con nuestra región de prueba (< 1.5 grados)
            c1 = SkyCoord(ra=ra_ref * u.deg, dec=dec_ref * u.deg)
            c2 = SkyCoord(ra=ra_cuadro * u.deg, dec=dec_cuadro * u.deg)
            if c1.separation(c2).deg > 1.5:
                continue

            # Extraer longitud de onda central de la cabecera (si no viene explícita, estimar por archivo)
            lam_um = float(hdr.get("WAVELEN", 0.75 + len(cuadros) * 0.06))

            cuadros.append({
                "fecha_utc": str(hdr.get("DATE-OBS", mjd_obs)),
                "mjd": mjd_obs,
                "lambda_um": lam_um,
                "banda": banda,
                "url": uri,
            })

    print(f"¡Listo! Se armó un campo real con {len(cuadros)} cuadros alrededor de RA={ra_ref:.4f}, Dec={dec_ref:.4f}")
    return {
        "ra": ra_ref,
        "dec": dec_ref,
        "fov_deg": 0.25,  # Campo amplio (2.5x más grande que tu prueba inicial)
        "cuadros": cuadros,
    }


if __name__ == "__main__":
    dir_actual = Path(__file__).resolve().parent
    ruta_json_vivo = dir_actual / "input_s3_real_grande.json"
    ruta_salida = dir_actual / "salida_s3_real_grande.json"

    # 1. Consultar cabeceras directo desde AWS S3 (sin timeouts de IRSA)
    datos_entrada = explorar_spherex_directo_s3(max_archivos=25)
    with open(ruta_json_vivo, "w", encoding="utf-8") as f:
        json.dump(datos_entrada, f, indent=2)

    # 2. Ejecutar tu pipeline completo sobre ese campo real de 0.25°
    resultado = ejecutar_pipeline_completo(str(ruta_json_vivo), clase_alerce="DEFAULT")

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)
    print(f"Procesamiento completado. Catálogo vectorizado en: {ruta_salida}")

    # 3. Renderizar las 6 bandas y la curva espectral
    renderizar_control_local(resultado, epoch_idx=0, banda_idx=None, carpeta_salida=dir_actual)