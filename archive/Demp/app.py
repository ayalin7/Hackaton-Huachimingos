"""
================================================================================
BACKEND WEB: app.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================
Servidor FastAPI que conecta el pipeline vectorizado de SPHEREx, la
identificación con ALeRCE / Gaia DR3 y el interpolador de movimiento con la
interfaz web tipo Stellarium (index.html).

Ejecución:
    uvicorn app:app --reload --port 8000
================================================================================
"""

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval
from astropy.wcs import WCS
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

DIR_APP = Path(__file__).resolve().parent
if str(DIR_APP) not in sys.path:
    sys.path.insert(0, str(DIR_APP))

from generador_interpolacion import evaluar_posicion_en_tiempo
from generador_json_definitivo import (
    CARPETA_DATOS_WEB,
    OBJETIVOS_CHALLENGE,
    buscar_target_challenge,
    construir_json_definitivo,
    slugify,
)

# Localizar index.html tanto si está en la misma carpeta como en ../frontend
RUTA_INDEX = DIR_APP / "index.html"
if not RUTA_INDEX.exists():
    RUTA_INDEX = DIR_APP.parent / "frontend" / "index.html"

app = FastAPI(title="SPHEREx Stellarium Explorer - NASA Space Apps 2026")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
def _no_content():
    return Response(status_code=204)


# -----------------------------------------------------------------
# ENDPOINTS DE ESTADO Y OBJETIVOS OFICIALES DEL CHALLENGE
# -----------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "ok": True,
        "servicio": "SPHEREx Vectorial + Gaia DR3 + ALeRCE + Interpolador",
        "archivos_precalculados": len(list(CARPETA_DATOS_WEB.glob("definitivo_*.json"))),
    }


@app.get("/api/challenge-targets")
def challenge_targets():
    """Entrega la lista de los 5 objetivos oficiales del desafío para acceso rápido."""
    lista = []
    for obj in OBJETIVOS_CHALLENGE:
        archivo = CARPETA_DATOS_WEB / f"definitivo_{obj['slug']}.json"
        lista.append({
            "nombre": obj["nombre_display"],
            "slug": obj["slug"],
            "ra": obj["ra"],
            "dec": obj["dec"],
            "fov_deg": obj["fov_deg"],
            "tipo_cuerpo": obj["astrofisica"]["tipo_cuerpo"],
            "descripcion": obj["astrofisica"]["descripcion"],
            "precalculado": archivo.exists(),
        })
    return {"objetivos": lista}


# -----------------------------------------------------------------
# ENDPOINT PRINCIPAL VECTORIAL: PIPELINE + GAIA + INTERPOLADOR
# -----------------------------------------------------------------
@app.get("/api/timeline")
def api_timeline(
    q: str | None = Query(default=None, description="Nombre o ID del objeto (ej. 'WISE 0855', 'Barnard's Star')"),
    ra: float | None = Query(default=None, description="Ascensión Recta en grados"),
    dec: float | None = Query(default=None, description="Declinación en grados"),
    fov: float | None = Query(default=None, description="Campo de visión en grados"),
    clase: str = Query(default="AUTO", description="Perfil espectral: AUTO, DEFAULT, YSO, AGN, SN, ASTEROID"),
    forzar_vivo: bool = Query(default=False, description="Si es True, ignora caché y recalcula desde S3/IRSA"),
):
    """
    Entrega el JSON definitivo con las 4 mediciones del satélite, la 5ta medición
    interpolada, ecuaciones de movimiento, espectro de 6 bandas y telemetría física.
    """
    target_ch = buscar_target_challenge(query=q, ra=ra, dec=dec)
    slug = target_ch["slug"] if target_ch else (slugify(q) if q else f"coord_{round(ra or 0, 3)}_{round(dec or 0, 3)}")
    ruta_cache = CARPETA_DATOS_WEB / f"definitivo_{slug}.json"

    # Si existe el JSON definitivo ya generado en disco y no se fuerza recálculo en vivo, entregar al instante
    if ruta_cache.exists() and not forzar_vivo and clase.upper() == "AUTO":
        with open(ruta_cache, "r", encoding="utf-8") as f:
            return json.load(f)

    try:
        datos_definitivos = construir_json_definitivo(
            query=q,
            ra=ra,
            dec=dec,
            fov_deg=fov,
            clase_forzada=clase,
            guardar_disco=True,
        )
        return datos_definitivos
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error ejecutando pipeline astrofísico: {exc}") from exc


@app.get("/api/posicion-tiempo")
def api_posicion_tiempo(
    slug: str = Query(..., description="Slug del objeto cargado"),
    mjd: float = Query(..., description="Fecha Juliana Modificada (MJD) objetivo"),
):
    """Evalúa las ecuaciones de movimiento de generador_interpolacion.py en un instante MJD."""
    ruta_cache = CARPETA_DATOS_WEB / f"definitivo_{slug}.json"
    if not ruta_cache.exists():
        raise HTTPException(status_code=404, detail="No se encontró el catálogo para ese slug.")

    with open(ruta_cache, "r", encoding="utf-8") as f:
        datos = json.load(f)

    posiciones = []
    for item in datos.get("ecuaciones_trayectoria", []):
        ra_t, dec_t = evaluar_posicion_en_tiempo(item["ecuacion"], mjd)
        posiciones.append({
            "nombre_oficial": item["nombre_oficial"],
            "ra": ra_t,
            "dec": dec_t,
        })
    return {"mjd": mjd, "posiciones": posiciones}


# -----------------------------------------------------------------
# IMÁGENES IR POR BANDA (RECORTES PNG DESDE LOS FITS DE SPHEREx EN S3)
# -----------------------------------------------------------------
CARPETA_CUTOUTS = CARPETA_DATOS_WEB / "cutouts"
CARPETA_CUTOUTS.mkdir(exist_ok=True)


def generar_png_cutout(s3_uri: str, ra: float, dec: float, fov_deg: float, ruta_png: Path) -> None:
    """Recorta el FITS de SPHEREx alrededor de (ra, dec) y lo guarda como PNG con estiramiento asinh."""
    centro = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    with fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        imagen = hdul["IMAGE"].data
        wcs = WCS(hdul["IMAGE"].header)
        cutout = Cutout2D(imagen, position=centro, size=(fov_deg * u.deg, fov_deg * u.deg),
                          wcs=wcs, mode="partial", fill_value=np.nan)
        datos = np.array(cutout.data, dtype=float)

    validos = np.isfinite(datos)
    if not validos.any():
        raise ValueError("El recorte no contiene píxeles válidos.")
    datos = np.where(validos, datos, np.nanmedian(datos))
    norm = ImageNormalize(datos, interval=PercentileInterval(99.5), stretch=AsinhStretch(0.1))
    plt.imsave(ruta_png, np.asarray(norm(datos)), cmap="inferno", origin="lower", vmin=0, vmax=1)


@app.get("/api/cutout/{slug}/{epoch_idx}/{banda_idx}.png")
def api_cutout(slug: str, epoch_idx: int, banda_idx: int):
    """Entrega la imagen infrarroja real de SPHEREx para una medición y banda del objeto cargado."""
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", slug):
        raise HTTPException(status_code=400, detail="Slug inválido.")

    ruta_png = CARPETA_CUTOUTS / f"{slug}_e{epoch_idx}_b{banda_idx}.png"
    if ruta_png.exists():
        return FileResponse(ruta_png, media_type="image/png")

    ruta_json = CARPETA_DATOS_WEB / f"definitivo_{slug}.json"
    if not ruta_json.exists():
        raise HTTPException(status_code=404, detail="No se encontró el catálogo para ese slug.")
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)

    try:
        banda = datos["epochs"][epoch_idx]["bandas"][banda_idx]
    except (IndexError, KeyError) as exc:
        raise HTTPException(status_code=404, detail="Esa medición o banda no existe.") from exc

    s3_uri = banda.get("s3_uri")
    if not s3_uri:
        raise HTTPException(status_code=404, detail="Esta banda no tiene imagen SPHEREx asociada.")

    centro = datos["centro_campo"]
    try:
        generar_png_cutout(s3_uri, centro["ra"], centro["dec"], centro["fov_deg"], ruta_png)
    except NoOverlapError as exc:
        raise HTTPException(status_code=404, detail="El detector no cubre este punto en esa banda.") from exc
    except Exception as exc:
        ruta_png.unlink(missing_ok=True)
        raise HTTPException(status_code=502, detail=f"No se pudo leer el FITS desde S3: {exc}") from exc

    return FileResponse(ruta_png, media_type="image/png")


# -----------------------------------------------------------------
# RESOLUCIÓN DE NOMBRES Y AUTOCOMPLETADO (CHALLENGE + SIMBAD)
# -----------------------------------------------------------------
@lru_cache(maxsize=512)
def _simbad_names(prefix: str):
    q = prefix.replace("'", "''").lower()
    adql = f"SELECT TOP 8 id FROM ident WHERE LOWER(id) LIKE '{q}%'"
    r = requests.get(
        "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
        timeout=5,
        params={"request": "doQuery", "lang": "adql", "format": "json", "query": adql},
    )
    r.raise_for_status()
    names = []
    for row in r.json().get("data", []):
        n = re.sub(r"^NAME\s+", "", str(row[0])).strip()
        if n and n not in names:
            names.append(n)
    return tuple(names)


@app.get("/api/suggest")
def suggest(q: str = Query(min_length=2)):
    q_low = q.strip().lower()
    coincidencias_ch = [
        o["nombre_display"]
        for o in OBJETIVOS_CHALLENGE
        if q_low in o["nombre_display"].lower() or any(q_low in a for a in o["alias"])
    ]
    try:
        simbad = list(_simbad_names(" ".join(q.split())))
    except Exception:
        simbad = []
    combinados = list(dict.fromkeys(coincidencias_ch + simbad))[:8]
    return {"names": combinados}


@app.get("/api/resolve")
def resolve(name: str):
    target_ch = buscar_target_challenge(query=name)
    if target_ch:
        return {"name": target_ch["nombre_display"], "ra": target_ch["ra"], "dec": target_ch["dec"]}

    parts = name.strip().replace(",", " ").split()
    if len(parts) == 2:
        try:
            return {"name": name, "ra": float(parts[0]), "dec": float(parts[1])}
        except ValueError:
            pass

    try:
        c = SkyCoord.from_name(name.strip())
        return {"name": name.strip(), "ra": float(c.icrs.ra.deg), "dec": float(c.icrs.dec.deg)}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"No se encontró «{name}» en SIMBAD/Challenge") from exc


@app.get("/", include_in_schema=False)
def serve_index():
    if RUTA_INDEX.exists():
        return FileResponse(RUTA_INDEX)
    return Response("<h1>No se encontró index.html</h1>", media_type="text/html", status_code=404)


if RUTA_INDEX.parent.exists() and RUTA_INDEX.parent != DIR_APP:
    app.mount("/", StaticFiles(directory=RUTA_INDEX.parent, html=True), name="frontend")