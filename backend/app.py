"""SPHEREx Timeline - backend MVP.

Endpoints
  GET /api/health
  GET /api/images?ra&dec&fov            -> SPHEREx spectral images covering a position (SIA2, IRSA)
  GET /api/render?mode=single|stack|diff -> PNG built from IRSA cutouts (co-add / difference / single)
  GET /api/pixel?ra&dec&fov&mode&xf&yf  -> pixel value, sky coordinates, quality flags & FITS metadata
  GET /api/suggest?q                    -> identifier autocomplete via SIMBAD TAP
  GET /api/resolve?name                 -> object name / coordinate resolver
  GET /api/spectrum?ra&dec              -> on-the-fly spectrum at a position
The frontend (../frontend) is served from "/".
"""
import hashlib
import io
import os
import re
import socket
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
from pathlib import Path

import astropy.units as u
import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astroquery.ipac.irsa import Irsa
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from scipy.ndimage import map_coordinates

# Collection names are documented at the SPHEREx data access page of IRSA.
# Change with: set SPHEREX_COLLECTION=spherex_dr1  (or a newer one) before starting.
COLLECTION = os.getenv("SPHEREX_COLLECTION", "spherex_dr1")
ALLOWED_PREFIX = "https://irsa.ipac.caltech.edu/"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

SURVEY_START_MJD = 60796.0   # ~2025-05-01, approximate start of survey 1
SURVEY_LEN_DAYS = 182.6      # ~6 months per all-sky pass
SIA_TIMEOUT_SEC = 25
MAX_SIA_RADIUS_DEG = 0.05
MIN_SIA_RADIUS_DEG = 0.01

socket.setdefaulttimeout(30)

app = FastAPI(title="SPHEREx Timeline")


@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
def _no_content():
    return Response(status_code=204)


# ----------------------------------------------------------------- helpers
def _check_url(url: str) -> str:
    if not url.startswith(ALLOWED_PREFIX):
        raise HTTPException(400, "Only IRSA URLs are allowed")
    return url


def _run_sia(ra: float, dec: float, radius: float):
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    return Irsa.query_sia(pos=(coord, radius * u.deg), collection=COLLECTION)


def _parse_sia_row(row, cols) -> dict:
    url = str(row["access_url"])
    m = re.search(r"2D(\d)", url) or re.search(r"/(\d)/level2", url)
    det = int(m.group(1)) if m else 0
    mjd = float(row["t_min"]) if "t_min" in cols else float("nan")
    em_min = float(row["em_min"]) * 1e6 if "em_min" in cols else float("nan")
    em_max = float(row["em_max"]) * 1e6 if "em_max" in cols else float("nan")
    survey = int((mjd - SURVEY_START_MJD) // SURVEY_LEN_DAYS) + 1 if np.isfinite(mjd) else 0
    return {
        "url": url,
        "detector": det,
        "mjd": mjd,
        "survey": survey,
        "wl_min": em_min,
        "wl_max": em_max,
        "wl": (em_min + em_max) / 2,
    }


@lru_cache(maxsize=256)
def _query_images(ra: float, dec: float, radius: float):
    print(f"[IRSA] Consultando {COLLECTION} en RA={ra}, Dec={dec}, r={radius}°...", flush=True)
    ex = ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(_run_sia, ra, dec, radius)
    try:
        tab = fut.result(timeout=SIA_TIMEOUT_SEC)
    except FuturesTimeoutError as exc:
        print(f"[IRSA] Tiempo agotado ({SIA_TIMEOUT_SEC}s) en RA={ra}, Dec={dec}", flush=True)
        raise TimeoutError(f"IRSA SIA2 tardó más de {SIA_TIMEOUT_SEC} segundos en responder") from exc
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    if hasattr(tab, "to_table"):
        tab = tab.to_table()
    cols = tab.colnames
    out = [_parse_sia_row(row, cols) for row in tab]
    out.sort(key=lambda r: (r["mjd"], r["detector"]))
    print(f"[IRSA] Respuesta recibida: {len(out)} filas", flush=True)
    return out


def _download_cutout(full_url: str, path: Path) -> None:
    r = requests.get(full_url, timeout=60)
    if not r.ok:
        raise HTTPException(502, f"IRSA cutout HTTP {r.status_code}: {r.text[:200]}")
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(path)


def _read_fits_cutout(path: Path):
    with fits.open(path) as hdul:
        names = [h.name.upper() for h in hdul]
        img_hdu = (
            hdul["IMAGE"]
            if "IMAGE" in names
            else next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
        )
        data = np.asarray(img_hdu.data, dtype=float)
        if data.ndim > 2:
            data = data[0]
        header = img_hdu.header.copy()
        try:
            wcs = WCS(header, fobj=hdul, naxis=2)
        except Exception:  # noqa: BLE001
            wcs = WCS(header, naxis=2)
        flags = np.asarray(hdul["FLAGS"].data) if "FLAGS" in names else None
        if flags is not None and flags.ndim > 2:
            flags = flags[0]
    return data, flags, wcs, header


def _fetch_cutout(url: str, ra: float, dec: float, fov: float):
    """Download a cutout FITS (cached on disk) and return (image, flags, wcs, header)."""
    _check_url(url)
    full = f"{url}?center={ra:.6f},{dec:.6f}&size={fov:.5f}"
    path = CACHE / (hashlib.md5(full.encode()).hexdigest() + ".fits")
    if not path.exists():
        _download_cutout(full, path)
    try:
        return _read_fits_cutout(path)
    except OSError:
        path.unlink(missing_ok=True)
        _download_cutout(full, path)
        return _read_fits_cutout(path)


def _reproject(data, wcs_src, wcs_ref, shape_ref):
    """Resample `data` (with wcs_src) onto the reference grid, using only astropy + scipy."""
    yy, xx = np.mgrid[0:shape_ref[0], 0:shape_ref[1]]
    world = wcs_ref.pixel_to_world(xx, yy)
    xs, ys = wcs_src.world_to_pixel(world)
    return map_coordinates(data, [ys, xs], order=1, mode="constant", cval=np.nan)


def _png(arr, symmetric=False, min_side=512):
    a = np.array(arr, dtype=float)
    good = np.isfinite(a)
    if not good.any():
        raise HTTPException(422, "Cutout has no valid pixels here")
    if symmetric:
        med = np.nanmedian(a)
        sig = 1.4826 * np.nanmedian(np.abs(a[good] - med)) or np.nanstd(a)
        lo, hi = -5 * sig, 5 * sig
        a = a - med
    else:
        lo, hi = ZScaleInterval().get_limits(a[good])
    scaled = np.clip((a - lo) / (hi - lo + 1e-30), 0, 1)
    scaled[~good] = 0
    img = Image.fromarray((scaled * 255).astype(np.uint8)[::-1], "L")  # flip: origin lower-left
    k = max(1, int(np.ceil(min_side / max(img.size))))
    if k > 1:
        img = img.resize((img.width * k, img.height * k), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


# --------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"ok": True, "collection": COLLECTION}


@app.get("/api/images")
def images(ra: float, dec: float, fov: float = 0.05):
    sia_radius = round(min(max(fov / 2, MIN_SIA_RADIUS_DEG), MAX_SIA_RADIUS_DEG), 4)
    try:
        res = _query_images(round(ra, 5), round(dec, 5), sia_radius)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"IRSA query failed ({COLLECTION}): {e}")
    return {"collection": COLLECTION, "count": len(res), "images": res}


def _build(mode, ra, dec, fov, urls, a, b):
    """Return (array, wcs, flags, header) for the product that the viewer shows."""
    if mode == "single":
        target = a or (urls[0] if urls else None)
        if not target:
            raise HTTPException(400, "single needs a or u")
        d, f, w, h = _fetch_cutout(target, ra, dec, fov)
        return d, w, f, h
    if mode == "diff":
        if not a or not b:
            raise HTTPException(400, "diff needs a and b")
        da, fa, wa, ha = _fetch_cutout(a, ra, dec, fov)
        db, _, wb, _ = _fetch_cutout(b, ra, dec, fov)
        return da - _reproject(db, wb, wa, da.shape), wa, fa, ha
    if mode == "stack":
        urls = urls[:40]
        if not urls:
            raise HTTPException(400, "No images to stack")
        ref, fref, wref, href = _fetch_cutout(urls[0], ra, dec, fov)
        layers = [ref]
        for url in urls[1:]:
            d, _, w, _ = _fetch_cutout(url, ra, dec, fov)
            layers.append(_reproject(d, w, wref, ref.shape))
        return np.nanmedian(np.stack(layers), axis=0), wref, fref, href
    raise HTTPException(400, "mode must be single, diff or stack")


@app.get("/api/render")
def render(
    ra: float,
    dec: float,
    fov: float = 0.05,
    mode: str = "single",
    u_: list[str] = Query(default=[], alias="u"),
    a: str | None = None,
    b: str | None = None,
):
    try:
        arr, *_ = _build(mode, ra, dec, fov, u_, a, b)
        return _png(arr, symmetric=(mode == "diff"))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(502, f"Render failed: {e}")


@app.get("/api/pixel")
def pixel(
    ra: float,
    dec: float,
    fov: float = 0.05,
    mode: str = "single",
    xf: float = 0.5,
    yf: float = 0.5,
    u_: list[str] = Query(default=[], alias="u"),
    a: str | None = None,
    b: str | None = None,
):
    """Details of the pixel under a click. xf, yf are fractions (0-1) from the top-left of the displayed PNG."""
    try:
        arr, wcs, flags, header = _build(mode, ra, dec, fov, u_, a, b)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(502, f"Pixel lookup failed: {e}")

    ny, nx = arr.shape
    x = min(max(int(xf * nx), 0), nx - 1)
    y = min(max(ny - 1 - int(yf * ny), 0), ny - 1)   # the PNG is flipped vertically
    sky = wcs.pixel_to_world(x, y)
    val = float(arr[y, x])
    flag = None
    bits = []
    if flags is not None and flags.shape == arr.shape:
        flag = int(flags[y, x])
        bits = [k for k in range(32) if (flag >> k) & 1]
    keys = ["DATE-OBS", "MJD-OBS", "EXPTIME", "DETECTOR", "BUNIT"]
    return {
        "ra": float(sky.ra.deg),
        "dec": float(sky.dec.deg),
        "x": x,
        "y": y,
        "nx": nx,
        "ny": ny,
        "value": val if np.isfinite(val) else None,
        "flags": flag,
        "flag_bits": bits,
        "header": {k: str(header[k]) for k in keys if k in header},
    }


# ----------------------------------------------------- name search / resolve
@lru_cache(maxsize=512)
def _simbad_names(prefix: str):
    q = prefix.replace("'", "''").lower()
    adql = f"SELECT TOP 8 id FROM ident WHERE LOWER(id) LIKE '{q}%'"
    r = requests.get(
        "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
        timeout=6,
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
    """Identifier suggestions (SIMBAD). Returns an empty list if SIMBAD is unreachable."""
    try:
        return {"names": list(_simbad_names(" ".join(q.split())))}
    except Exception:  # noqa: BLE001
        return {"names": []}


def _parse_coord(text: str):
    parts = text.replace(",", " ").split()
    try:
        if len(parts) == 2:
            return SkyCoord(float(parts[0]) * u.deg, float(parts[1]) * u.deg)
    except ValueError:
        pass
    if re.search(r"\d", text) and (":" in text or re.search(r"\d\s*h", text)):
        try:
            return SkyCoord(text, unit=(u.hourangle, u.deg))
        except Exception:  # noqa: BLE001
            return None
    return None


@app.get("/api/resolve")
def resolve(name: str):
    c = _parse_coord(name.strip())
    if c is None:
        try:
            c = SkyCoord.from_name(name.strip())
        except Exception:  # noqa: BLE001
            raise HTTPException(404, f"No encontré «{name}» en SIMBAD/Sesame")
    return {"name": name, "ra": float(c.icrs.ra.deg), "dec": float(c.icrs.dec.deg)}


def _flux_at(url, ra, dec):
    data, flags, wcs, _ = _fetch_cutout(url, ra, dec, 0.02)
    x, y = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    x, y = int(round(float(x))), int(round(float(y)))
    y0, y1 = max(y - 1, 0), y + 2
    x0, x1 = max(x - 1, 0), x + 2
    box = data[y0:y1, x0:x1]
    fl = None if flags is None else bool(np.any(flags[y0:y1, x0:x1]))
    return float(np.nanmedian(box)) if box.size else float("nan"), fl


@app.get("/api/spectrum")
def spectrum(ra: float, dec: float, n: int = 40):
    try:
        res = _query_images(round(ra, 5), round(dec, 5), MIN_SIA_RADIUS_DEG)[: max(1, min(n, 100))]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"IRSA query failed ({COLLECTION}): {e}")

    def work(r):
        try:
            flux, fl = _flux_at(r["url"], ra, dec)
            return {
                "wl": r["wl"],
                "flux": flux,
                "flagged": fl,
                "mjd": r["mjd"],
                "detector": r["detector"],
            }
        except Exception:  # noqa: BLE001
            return None

    with ThreadPoolExecutor(8) as ex:
        pts = [p for p in ex.map(work, res) if p and np.isfinite(p["flux"])]
    pts.sort(key=lambda p: p["wl"])
    return {"unit": "MJy/sr", "points": pts}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")