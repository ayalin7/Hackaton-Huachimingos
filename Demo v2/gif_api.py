"""
gif_api.py — API mínima que SOLO entrega el GIF de SPHEREx de un parche de cielo.

Usa el motor de spherex_video.py (imágenes reales de IRSA, alineadas, con fecha
y longitud de onda en cada cuadro). Nada se escribe en disco: los GIF se guardan
en memoria mientras el servidor está encendido.

Ejecutar:
    python -m uvicorn gif_api:app --port 8001

Endpoints:
    GET /api/gif?ra=269.4463&dec=4.7675          → image/gif
    GET /api/gif?objeto=Barnard's Star&modo=resta → image/gif
    GET /api/health                              → estado
    GET /                                        → página gif.html
"""

import sys
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

import objetivos            # noqa: E402
import spherex_video as sv  # noqa: E402

# Modo servidor: búsquedas y recortes se guardan en RAM, no en disco
sv.usar_memoria(True)

app = FastAPI(title="SPHEREx GIF API")

# Permite que la página se abra desde otro sitio o como archivo local (file://)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

COLECCIONES = "spherex_dr1,spherex_qr3"
MAX_GIFS_EN_MEMORIA = 30

_gifs: "OrderedDict[tuple, bytes]" = OrderedDict()
_candados: dict = {}
_candado_global = threading.Lock()


def _candado_para(clave):
    """Un candado por pedido: si dos personas piden el mismo GIF a la vez, se calcula una sola vez."""
    with _candado_global:
        return _candados.setdefault(clave, threading.Lock())


@app.get("/api/health")
def health():
    return {"ok": True, "gifs_en_memoria": len(_gifs)}


@app.get("/api/gif")
def api_gif(
    ra: float | None = Query(default=None, ge=0, lt=360, description="Ascensión recta en grados"),
    dec: float | None = Query(default=None, ge=-90, le=90, description="Declinación en grados"),
    objeto: str | None = Query(default=None, max_length=80,
                               description="Nombre (SIMBAD). Corrige el movimiento propio del objeto."),
    fov: float = Query(default=0.1, ge=0.02, le=0.5, description="Campo de visión en grados"),
    detector: int = Query(default=1, ge=1, le=6, description="1=0.75–1.09 µm … 6=4.42–5.00 µm"),
    modo: str = Query(default="directo", pattern="^(directo|resta|anterior)$",
                      description="directo, resta (cada cuadro menos el promedio) o anterior (menos el cuadro previo)"),
    identificar: bool = Query(default=False, description="Numerar y etiquetar fuentes (ALeRCE, Gaia, SIMBAD, WISE)"),
    max_cuadros: int = Query(default=40, ge=2, le=120),
    fps: int = Query(default=3, ge=1, le=10, description="Cuadros por segundo"),
):
    if objeto is None and (ra is None or dec is None):
        raise HTTPException(status_code=400, detail="Indica 'ra' y 'dec' (grados) o 'objeto'.")

    # Resolver el objetivo: con nombre se pide a SIMBAD posición + movimiento propio,
    # así el centro sigue al objeto aunque se mueva rápido (Barnard, WISE 0855...)
    info = None
    if objeto:
        try:
            # Ojo: si además vienen ra/dec, se toman como la posición J2000 del objeto
            info = objetivos.resolver(objeto.strip(), ra, dec)
        except (SystemExit, Exception) as exc:
            raise HTTPException(status_code=404, detail=f"No encontré '{objeto}': {exc}") from exc
        if not info:
            raise HTTPException(status_code=404, detail=f"No encontré '{objeto}' en SIMBAD.")
        ra, dec = info["ra"], info["dec"]

    clave = (round(ra, 6), round(dec, 6), (objeto or "").lower(), fov, detector, modo,
             identificar, max_cuadros, fps)

    with _candado_para(clave):
        if clave in _gifs:
            _gifs.move_to_end(clave)
            return Response(_gifs[clave], media_type="image/gif",
                            headers={"Cache-Control": "public, max-age=3600"})
        try:
            r = sv.hacer_video(ra, dec, fov, detector=detector, coleccion=COLECCIONES,
                               modo=modo, salida=None, max_cuadros=max_cuadros, fps=fps,
                               usar_alerce=identificar, objetivo=info, tam=600)
        except SystemExit as exc:
            # El motor termina así cuando IRSA no responde o no hay imágenes
            raise HTTPException(status_code=503, detail=f"Sin imágenes SPHEREx: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Error al armar el GIF: {exc!r}") from exc

        if not isinstance(r, dict) or not r.get("gif"):
            raise HTTPException(status_code=404, detail="No hay imágenes SPHEREx para ese pedido.")

        _gifs[clave] = r["gif"]
        while len(_gifs) > MAX_GIFS_EN_MEMORIA:
            _gifs.popitem(last=False)
        return Response(r["gif"], media_type="image/gif",
                        headers={"Cache-Control": "public, max-age=3600"})


@app.get("/", include_in_schema=False)
@app.get("/gif.html", include_in_schema=False)
def pagina():
    ruta = DIR / "gif.html"
    if ruta.exists():
        return FileResponse(ruta)
    return Response("<h1>Falta gif.html</h1>", media_type="text/html", status_code=404)
