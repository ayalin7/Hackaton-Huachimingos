"""
================================================================================
BACKEND WEB: app.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================
"""

import io
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.nddata.utils import NoOverlapError
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from fastapi import FastAPI, HTTPException, Query
import base64
import io
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from PIL import Image
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
import threading
from collections import OrderedDict

import objetivos
import spherex_video as sv

# Activa caché en RAM para el servidor web (no escribe carpetas en disco)
sv.usar_memoria(True)

from astropy.table import Table

def _buscar_imagenes_s3_local(ra, dec, coleccion="spherex_qr3", detector=None):
    cuadros = buscar_cuadros_locales_o_irsa(ra, dec, fov_deg=0.08)
    if not cuadros:
        raise SystemExit("No encontré cuadros en índices locales ni en IRSA.")
    tabla = Table({
        "access_url": [c["url"] for c in cuadros],
        "t_min": [float(c["mjd"]) for c in cuadros],
        "energy_bandpassname": [c.get("banda", "SPHEREx-D1") for c in cuadros],
        "em_min": [float(c.get("lambda_um", 1.0)) * 1e-6 for c in cuadros],
        "em_max": [float(c.get("lambda_um", 1.0)) * 1e-6 for c in cuadros],
    })
    tabla.sort("t_min")
    return sv.filtrar_detector(tabla, detector)

def _bajar_recorte_s3(access_url, ra, dec, fov_deg):
    s3_uri = irsa_url_a_s3(access_url)
    centro = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    size = (fov_deg * u.deg, fov_deg * u.deg)
    with fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        raw_img = hdul["IMAGE"].data.astype(float)
        raw_zodi = hdul["ZODI"].data.astype(float) if "ZODI" in hdul else np.zeros_like(raw_img)
        raw_flags = hdul["FLAGS"].data.astype(np.int64) if "FLAGS" in hdul else np.zeros(raw_img.shape, dtype=np.int64)
        header = sv.arreglar_sip(hdul["IMAGE"].header.copy())
        wcs = WCS(header)

        cut_img = Cutout2D(raw_img - raw_zodi, position=centro, size=size, wcs=wcs, mode="partial", fill_value=np.nan)
        cut_flags = Cutout2D(raw_flags, position=centro, size=size, wcs=wcs, mode="partial", fill_value=0)

        lam = np.nan
        try:
            wcs_w = WCS(hdul["IMAGE"].header, fobj=hdul, key="W")
            ny, nx = raw_img.shape
            lam_val, _ = wcs_w.pixel_to_world(nx / 2, ny / 2)
            lam = float(getattr(lam_val, "value", lam_val))
        except Exception:
            pass

    return cut_img.data, cut_flags.data.astype(np.int32), cut_img.wcs.to_header(relax=True), lam

sv.buscar_imagenes = _buscar_imagenes_s3_local
sv._bajar_recorte = _bajar_recorte_s3

_GIF_CACHE: OrderedDict[tuple, bytes] = OrderedDict()
_GIF_LOCKS: dict[tuple, threading.Lock] = {}
_GIF_GLOBAL_LOCK = threading.Lock()

def _lock_gif(clave: tuple) -> threading.Lock:
    with _GIF_GLOBAL_LOCK:
        return _GIF_LOCKS.setdefault(clave, threading.Lock())
        

DIR_APP = Path(__file__).resolve().parent
if str(DIR_APP) not in sys.path:
    sys.path.insert(0, str(DIR_APP))

from generador_interpolacion import evaluar_posicion_en_tiempo
from generador_json_definitivo import (
    CARPETA_DATOS_WEB,
    OBJETIVOS_CHALLENGE,
    buscar_target_challenge,
    construir_json_definitivo,
    consultar_cuadros_irsa_sia2,
    slugify,
)
from pipeline_spherex_vectorizado import irsa_url_a_s3

RUTA_INDEX = DIR_APP / "index.html"
if not RUTA_INDEX.exists():
    RUTA_INDEX = DIR_APP.parent / "frontend" / "index.html"

app = FastAPI(title="SPHEREx Stellarium Explorer - NASA Space Apps 2026")

def generar_png_base64_desde_s3(s3_uri: str, ra: float, dec: float, fov_deg: float = 0.04) -> dict | None:
    """Extrae los píxeles reales de SPHEREx en S3 y los devuelve como PNG Base64 + límites WCS."""
    with fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
        data = hdul["IMAGE"].data.astype(np.float32)
        wcs = WCS(hdul["IMAGE"].header)

    centro = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    cutout = Cutout2D(
        data, position=centro, size=(fov_deg * u.deg, fov_deg * u.deg),
        wcs=wcs, mode="partial", fill_value=np.nan
    )
    img = cutout.data

    validos = img[~np.isnan(img)]
    if len(validos) == 0:
        return None
    vmin, vmax = ZScaleInterval().get_limits(validos)
    norm = np.clip((np.nan_to_num(img, nan=vmin) - vmin) / max(vmax - vmin, 1e-8), 0.0, 1.0)

    img_uint8 = np.flipud((norm * 255).astype(np.uint8))
    pil_img = Image.fromarray(img_uint8, mode="L").resize((180, 180), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    ny, nx = img.shape
    esquinas = cutout.wcs.pixel_to_world([0, nx, nx, 0], [0, 0, ny, ny])
    bounds = [[float(c.ra.deg), float(c.dec.deg)] for c in esquinas]

    return {"png_base64": f"data:image/png;base64,{b64}", "bounds": bounds}

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
def _no_content():
    return Response(status_code=204)


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "servicio": "SPHEREx Vectorial + Gaia DR3 + ALeRCE + Interpolador",
        "archivos_precalculados": len(list(CARPETA_DATOS_WEB.glob("definitivo_*.json"))),
    }


@app.get("/api/challenge-targets")
def challenge_targets():
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


@app.get("/api/timeline")
def api_timeline(
    q: str | None = Query(default=None, description="Nombre o ID del objeto"),
    ra: float | None = Query(default=None, description="Ascensión Recta en grados"),
    dec: float | None = Query(default=None, description="Declinación en grados"),
    fov: float | None = Query(default=None, description="Campo de visión en grados"),
    clase: str = Query(default="AUTO", description="Perfil espectral"),
    forzar_vivo: bool = Query(default=False, description="Si es True, ignora caché"),
):
    target_ch = buscar_target_challenge(query=q, ra=ra, dec=dec)
    slug = target_ch["slug"] if target_ch else (slugify(q) if q else f"coord_{round(ra or 0, 3)}_{round(dec or 0, 3)}")
    ruta_cache = CARPETA_DATOS_WEB / f"definitivo_{slug}.json"

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
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise HTTPException(status_code=400, detail="Slug inválido.")

    ruta_cache = (CARPETA_DATOS_WEB / f"definitivo_{slug}.json").resolve()
    if ruta_cache.parent != CARPETA_DATOS_WEB.resolve() or not ruta_cache.exists():
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


@app.get("/api/cutout-png")
def api_cutout_png(
    ra: float = Query(...),
    dec: float = Query(...),
    fov: float = Query(default=0.04, ge=0.005, le=0.5),
    epoch_idx: int = Query(default=0, ge=0, le=1),
):
    """Extrae un recorte de píxeles reales FITS de SPHEREx desde AWS S3 y lo sirve como PNG."""
    cuadros = consultar_cuadros_irsa_sia2(ra, dec, radio_deg=max(fov * 0.75, 0.025), timeout_sec=15)
    if not cuadros:
        raise HTTPException(status_code=404, detail="Sin recortes FITS reales de SPHEREx disponibles para esta región.")

    cuadros.sort(key=lambda c: c["mjd"])
    candidatos = cuadros if epoch_idx == 0 else list(reversed(cuadros))
    centro = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    for cuadro in candidatos[:6]:
        s3_uri = irsa_url_a_s3(cuadro["url"])
        try:
            with fits.open(s3_uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
                data = hdul["IMAGE"].data
                wcs = WCS(hdul["IMAGE"].header)
            cutout = Cutout2D(data, position=centro, size=(fov * u.deg, fov * u.deg), wcs=wcs, mode="partial", fill_value=np.nan)
            arr = np.asarray(cutout.data, dtype=float)
            good = np.isfinite(arr)
            if not good.any():
                continue

            lo, hi = ZScaleInterval().get_limits(arr[good])
            norm = np.clip((arr - lo) / (hi - lo + 1e-30), 0, 1)
            norm[~good] = 0
            img = Image.fromarray((norm * 255).astype(np.uint8)[::-1], "L").resize((420, 420), Image.NEAREST)
            buf = io.BytesIO()
            img.save(buf, "PNG")
            return Response(buf.getvalue(), media_type="image/png", headers={"X-SPHEREx-MJD": str(cuadro["mjd"])})
        except (NoOverlapError, Exception):
            continue

    raise HTTPException(status_code=404, detail="Ningún cuadro FITS solapó con el centro del campo.")


@lru_cache(maxsize=512)
def _simbad_names(prefix: str):
    q = (
        prefix.replace("\\", "\\\\")
        .replace("'", "''")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .lower()
    )
    adql = f"SELECT TOP 8 id FROM ident WHERE LOWER(id) LIKE '{q}%' ESCAPE '\\'"
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

def buscar_cuadros_locales_o_irsa(ra: float, dec: float, fov_deg: float = 0.04) -> list[dict]:
    """Busca los índices S3 en datos_web/ o en resultados_challenge/ sin depender de IRSA."""
    centro = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    carpetas_busqueda = [
        CARPETA_DATOS_WEB,
        Path.home() / "Desktop/Hackaton-Huachimingos/src/Procesamiento de datos/resultados_challenge",
    ]
    for carpeta in carpetas_busqueda:
        if not carpeta.exists():
            continue
        for ruta_in in carpeta.glob("input_*.json"):
            try:
                with open(ruta_in, "r", encoding="utf-8") as f:
                    data_in = json.load(f)
                c_archivo = SkyCoord(ra=float(data_in["ra"]) * u.deg, dec=float(data_in["dec"]) * u.deg, frame="icrs")
                if centro.separation(c_archivo).deg <= 0.5 and data_in.get("cuadros"):
                    print(f"[S3 Índice Local] Usando {ruta_in.name} ({len(data_in['cuadros'])} cuadros)")
                    return data_in["cuadros"]
            except Exception:
                continue

    return consultar_cuadros_irsa_sia2(ra, dec, radio_deg=max(fov_deg * 0.75, 0.03), timeout_sec=10)


@app.get("/api/inspect_pixels")
def inspect_pixels(ra: float, dec: float, fov_deg: float = 0.05):
    """Extrae la 1ª y última época real desde AWS S3."""
    cuadros = buscar_cuadros_locales_o_irsa(ra, dec, max(fov_deg, 0.05))
    if not cuadros:
        raise HTTPException(status_code=404, detail="Sin índices S3 para esta coordenada.")

    cuadros.sort(key=lambda c: c["mjd"])
    ep1, ep2 = None, None
    fov_recorte = max(fov_deg, 0.05)

    for c in cuadros:
        try:
            uri = irsa_url_a_s3(c["url"])
            res = generar_png_base64_desde_s3(uri, ra, dec, fov_recorte)
            if res:
                res["mjd"] = round(c["mjd"], 2)
                ep1 = res
                print(f"[S3 Recorte Época 1 OK] {uri}")
                break
        except Exception as e:
            continue

    for c in reversed(cuadros):
        try:
            uri = irsa_url_a_s3(c["url"])
            res = generar_png_base64_desde_s3(uri, ra, dec, fov_recorte)
            if res:
                res["mjd"] = round(c["mjd"], 2)
                ep2 = res
                print(f"[S3 Recorte Última Época OK] {uri}")
                break
        except Exception:
            continue

    if not ep1:
        raise HTTPException(status_code=404, detail="Ningún cuadro S3 solapó con esta coordenada.")

    return {"epoch_1": ep1, "epoch_last": ep2 or ep1}

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
        info = objetivos.resolver(name.strip())
        return {"name": info["nombre"], "ra": float(info["ra"]), "dec": float(info["dec"])}
    except (SystemExit, Exception) as exc:
        raise HTTPException(status_code=404, detail=f"No se encontró «{name}» en SIMBAD/Challenge") from exc

@app.get("/api/gif")
def api_gif(
    ra: float | None = Query(default=None, ge=0, lt=360),
    dec: float | None = Query(default=None, ge=-90, le=90),
    objeto: str | None = Query(default=None, max_length=80),
    fov: float = Query(default=0.08, ge=0.02, le=0.5),
    detector: int = Query(default=1, ge=1, le=6),
    modo: str = Query(default="resta", pattern="^(directo|resta|anterior)$"),
    identificar: bool = Query(default=True),
    asteroides: bool = Query(default=False),
    max_cuadros: int = Query(default=35, ge=2, le=100),
    fps: int = Query(default=3, ge=1, le=10),
):
    """Genera el GIF animado de SPHEREx (directo o resta) en memoria usando spherex_video.py."""
    if objeto is None and (ra is None or dec is None):
        raise HTTPException(status_code=400, detail="Indica 'ra' y 'dec' o 'objeto'.")

    info_obj = None
    if objeto:
        try:
            info_obj = objetivos.resolver(objeto.strip(), ra, dec)
            ra, dec = info_obj["ra"], info_obj["dec"]
        except (SystemExit, Exception) as exc:
            raise HTTPException(status_code=404, detail=f"No se encontró '{objeto}': {exc}") from exc

    clave = (round(ra, 5), round(dec, 5), (objeto or "").lower(), fov, detector, modo, identificar, asteroides, fps)

    with _lock_gif(clave):
        if clave in _GIF_CACHE:
            _GIF_CACHE.move_to_end(clave)
            return Response(_GIF_CACHE[clave], media_type="image/gif", headers={"Cache-Control": "public, max-age=3600"})

        try:
            res = sv.hacer_video(
                ra=ra,
                dec=dec,
                fov_deg=fov,
                detector=detector,
                coleccion="spherex_dr1,spherex_qr3",
                modo=modo,
                salida=None,
                max_cuadros=max_cuadros,
                fps=fps,
                usar_alerce=identificar,
                usar_asteroides=asteroides,
                objetivo=info_obj,
                tam=520,
            )
        except SystemExit as exc:
            raise HTTPException(status_code=503, detail=f"IRSA/SPHEREx sin imágenes disponibles: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error generando GIF: {exc}") from exc

        if not isinstance(res, dict) or not res.get("gif"):
            raise HTTPException(status_code=404, detail="No se pudo construir el GIF para esta región.")

        _GIF_CACHE[clave] = res["gif"]
        while len(_GIF_CACHE) > 30:
            _GIF_CACHE.popitem(last=False)

        return Response(res["gif"], media_type="image/gif", headers={"Cache-Control": "public, max-age=3600"})

@app.get("/", include_in_schema=False)
def serve_index():
    if RUTA_INDEX.exists():
        return FileResponse(RUTA_INDEX)
    return Response("<h1>No se encontró index.html</h1>", media_type="text/html", status_code=404)


if RUTA_INDEX.parent.exists() and RUTA_INDEX.parent != DIR_APP:
    app.mount("/", StaticFiles(directory=RUTA_INDEX.parent, html=True), name="frontend")