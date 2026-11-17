"""
asteroides_capa.py — Dónde estaban los asteroides y cometas CONOCIDOS en el
momento exacto de cada cuadro SPHEREx.

Usa SkyBoT (IMCCE, Observatorio de París), que calcula posiciones a partir de
las órbitas conocidas. A diferencia de ALeRCE (que dice qué vio ZTF en ese lugar
en OTRAS fechas), aquí obtienes la posición en CADA cuadro: si un punto del video
se mueve justo sobre una marca, es ese asteroide; si se mueve y no hay marca,
podría ser algo no catalogado.

Nota: se usa el geocentro como observador ('500'). SPHEREx está en órbita baja;
la diferencia (paralaje) es de algunos segundos de arco para un asteroide del
cinturón principal, menos de 1-2 píxeles de SPHEREx.
"""
import json
import os
import time
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

CACHE_SKYBOT = "cache_skybot"
EN_MEMORIA = False   # True en el servidor: no se escribe en disco
_RAM = {}
COLOR_ASTEROIDE = (255, 220, 0)
COLOR_COMETA = (255, 140, 0)


def _consulta_skybot(ra, dec, radio_deg, mjd, reintentos=2):
    """Lista de cuerpos en el cono para un instante. Guarda en caché."""
    archivo = os.path.join(CACHE_SKYBOT,
                           f"{ra:.4f}_{dec:+.4f}_{radio_deg:.3f}_{mjd:.5f}.json")
    if EN_MEMORIA:
        if archivo in _RAM:
            return _RAM[archivo]
    else:
        os.makedirs(CACHE_SKYBOT, exist_ok=True)
    if not EN_MEMORIA and os.path.exists(archivo):
        with open(archivo) as f:
            return json.load(f)

    from astroquery.imcce import Skybot
    epoca = Time(mjd, format="mjd", scale="utc")
    for i in range(reintentos + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t = Skybot.cone_search(SkyCoord(ra, dec, unit="deg"),
                                       radio_deg * u.deg, epoca,
                                       find_planets=False, cache=False)
            break
        except RuntimeError as e:
            if "No solar system object" in str(e):  # no hay nada: respuesta válida
                t = []
                break
            if i == reintentos:
                print(f"  SkyBoT no respondió ({e})")
                return None
            time.sleep(2)
        except Exception as e:
            if i == reintentos:
                print(f"  SkyBoT no respondió ({type(e).__name__})")
                return None
            time.sleep(2)

    def num(v):
        try:
            return float(getattr(v, "value", v))
        except (TypeError, ValueError):
            return float("nan")

    cuerpos = [{
        "nombre": str(f["Name"]),
        "numero": int(num(f["Number"])) if np.isfinite(num(f["Number"])) else -1,
        "tipo": str(f["Type"]),
        "ra": num(f["RA"]),
        "dec": num(f["DEC"]),
        "V": num(f["V"]),
        "error_arcsec": num(f["posunc"]),
        "vel_ra_arcsec_h": num(f["RA_rate"]),
        "vel_dec_arcsec_h": num(f["DEC_rate"]),
    } for f in t]
    if EN_MEMORIA:
        if len(_RAM) > 5000:
            _RAM.clear()
        _RAM[archivo] = cuerpos
    else:
        with open(archivo, "w") as fh:
            json.dump(cuerpos, fh, indent=1)
    return cuerpos


def asteroides_por_cuadro(ra, dec, fov_deg, meta, wcs_out, forma, v_max=21.0):
    """
    Agrega a cada cuadro de `meta` la clave "asteroides": lista de marcas con
    posición en píxeles de la grilla común. Devuelve el resumen por objeto.
    """
    radio = fov_deg / 2 * np.sqrt(2) * 1.05
    resumen = {}
    for m in meta:
        cuerpos = _consulta_skybot(ra, dec, radio, m["mjd"])
        m["asteroides"] = []
        if not cuerpos:
            continue
        cuerpos = [c for c in cuerpos if np.isfinite(c["V"]) and c["V"] <= v_max]
        if not cuerpos:
            continue
        c_sky = SkyCoord([c["ra"] for c in cuerpos], [c["dec"] for c in cuerpos],
                         unit="deg")
        x, y = wcs_out.world_to_pixel(c_sky)
        for c, xi, yi in zip(cuerpos, np.atleast_1d(x), np.atleast_1d(y)):
            if 0 <= xi < forma[1] and 0 <= yi < forma[0]:
                marca = {**c, "x": float(xi), "y": float(yi)}
                m["asteroides"].append(marca)
                r = resumen.setdefault(c["nombre"], {**c, "n_cuadros": 0})
                r["n_cuadros"] += 1
    return sorted(resumen.values(), key=lambda r: r["V"])


def imprimir_tabla(resumen):
    if not resumen:
        print("  SkyBoT: ningún asteroide conocido (hasta el brillo límite) "
              "pasó por el parche en esas fechas.")
        return
    print(f"  SkyBoT: {len(resumen)} cuerpos conocidos pasaron por el parche")
    print("  nombre                 tipo           V     vel (\"/h)  cuadros")
    for r in resumen[:20]:
        v = np.hypot(r["vel_ra_arcsec_h"], r["vel_dec_arcsec_h"])
        print(f"  {r['nombre'][:21]:<21}  {r['tipo'][:13]:<13}  {r['V']:4.1f}  "
              f"{v:8.1f}  {r['n_cuadros']:>4}")
    if len(resumen) > 20:
        print(f"  ... y {len(resumen) - 20} más (ver el JSON)")


def dibujar(d, marcas, n_pix, zoom, fuente):
    """Rombos sobre la posición predicha en ESE cuadro."""
    r = max(6, 2.5 * zoom)
    for m in marcas:
        X = (m["x"] + 0.5) * zoom
        Y = (n_pix - 1 - m["y"] + 0.5) * zoom
        color = COLOR_COMETA if "comet" in m["tipo"].lower() else COLOR_ASTEROIDE
        ancho = 3 if m["V"] <= 18 else 1  # los más brillantes, línea gruesa
        d.polygon([(X, Y - r), (X + r, Y), (X, Y + r), (X - r, Y)],
                  outline=color, width=ancho)
        if len(marcas) <= 10:
            d.text((X + r + 2, Y - 7), m["nombre"], fill=color, font=fuente)
