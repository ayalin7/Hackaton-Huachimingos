"""
objetivos.py — Resolver un objeto por nombre y calcular dónde está HOY.

Los catálogos dan la posición en el año 2000 (época J2000). Las estrellas
cercanas se mueven ("movimiento propio"): Barnard's Star avanza ~10.4"/año, así
que en 2026 está ~4.5' (¡~45 píxeles de SPHEREx!) lejos de su posición J2000.
Este módulo pide a SIMBAD posición + movimiento propio + paralaje y propaga la
posición a la fecha de cada imagen SPHEREx.
"""
import json
import os
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time

CACHE_OBJ = "cache_objetos"
EN_MEMORIA = False   # True en el servidor: no se escribe en disco
_RAM = {}

# Respaldo si SIMBAD no responde (movimiento propio en mas/año, paralaje en mas).
# Fuentes: Gaia DR3 vía Wikipedia; ε Indi B comparte el movimiento de ε Indi A.
RESPALDO_PM = {
    "barnard's star": (-801.551, 10362.394, 546.9759),
    "eps ind b": (3966.661, -2536.192, 274.8431),
    "wise 0855-0714": (-8123.7, 673.2, 439.0),
}


def _clave(nombre):
    return nombre.strip().lower().replace("epsilon indi", "eps ind").replace("ε", "eps ")


def resolver(nombre, ra=None, dec=None):
    """
    Devuelve dict con ra, dec (J2000, grados), pmra, pmdec (mas/año), plx (mas).
    Si das ra/dec, se usan esas coordenadas (y SIMBAD solo para el movimiento).
    """
    archivo = os.path.join(CACHE_OBJ, _clave(nombre).replace(" ", "_").replace("'", "") + ".json")
    if EN_MEMORIA:
        if archivo not in _RAM:
            _RAM[archivo] = _desde_simbad(nombre) or _desde_sesame(nombre) or {}
        info = dict(_RAM[archivo])
    elif os.path.exists(archivo):
        with open(archivo) as f:
            info = json.load(f)
    else:
        os.makedirs(CACHE_OBJ, exist_ok=True)
        info = _desde_simbad(nombre) or _desde_sesame(nombre) or {}
        if info:
            with open(archivo, "w") as f:
                json.dump(info, f, indent=1)

    if ra is not None and dec is not None:
        info["ra"], info["dec"] = float(ra), float(dec)
    if "ra" not in info:
        raise SystemExit(f"No pude encontrar '{nombre}'. Dame sus coordenadas.")

    # Completar movimiento propio con el respaldo si falta
    k = _clave(nombre)
    for nombre_r, (pmra, pmdec, plx) in RESPALDO_PM.items():
        if nombre_r in k and not info.get("pmra"):
            info.update(pmra=pmra, pmdec=pmdec, plx=plx, fuente_pm="respaldo")
    info.setdefault("pmra", 0.0)
    info.setdefault("pmdec", 0.0)
    info.setdefault("plx", None)
    info["nombre"] = nombre
    return info


def _desde_simbad(nombre):
    try:
        import catalogos
        # Buscar por nombre con la tabla de identificadores (funciona en ambos
        # servidores: Estrasburgo y el espejo de Harvard)
        nom = nombre.replace("'", "''")
        q = ("SELECT TOP 1 b.main_id, b.ra, b.dec, b.pmra, b.pmdec, b.plx_value "
             "FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
             f"WHERE i.id = '{nom}'")
        b, _ = catalogos.consulta_simbad(q)
        if b is None or len(b) == 0:
            # SIMBAD normaliza los nombres (p. ej. "R Dra" → "V* R Dra"):
            # en ese caso usamos su resolvedor de nombres
            from astroquery.simbad import SimbadClass
            for srv in catalogos.SERVIDORES_SIMBAD:
                try:
                    cli = SimbadClass()
                    cli.server = srv
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        t = cli.query_object(nombre)
                    break
                except Exception:
                    t = None
            if t is None or len(t) == 0:
                return None
            cols = {c.lower(): c for c in t.colnames}
            main_id = str(t[cols["main_id"]][0]).replace("'", "''")
            b, _ = catalogos.consulta_simbad(
                "SELECT main_id, ra, dec, pmra, pmdec, plx_value FROM basic "
                f"WHERE main_id = '{main_id}'")
        f = b[0]
        main_id = str(f["main_id"])

        def num(v):
            return None if np.ma.is_masked(v) else float(v)
        return {"ra": num(f["ra"]), "dec": num(f["dec"]), "pmra": num(f["pmra"]),
                "pmdec": num(f["pmdec"]), "plx": num(f["plx_value"]),
                "id_simbad": main_id, "fuente": "SIMBAD"}
    except Exception as e:
        print(f"  SIMBAD no respondió para '{nombre}' ({type(e).__name__})")
        return None


def _desde_sesame(nombre):
    try:
        c = SkyCoord.from_name(nombre)
        return {"ra": c.ra.deg, "dec": c.dec.deg, "fuente": "Sesame (sin mov. propio)"}
    except Exception:
        return None


def posicion_en(info, mjd):
    """Posición del objeto en la fecha `mjd` (aplica el movimiento propio)."""
    pmra, pmdec = info.get("pmra") or 0.0, info.get("pmdec") or 0.0
    if pmra == 0 and pmdec == 0:
        return SkyCoord(info["ra"], info["dec"], unit="deg")
    plx = info.get("plx")
    dist = Distance(parallax=plx * u.mas) if plx and plx > 0 else 1 * u.kpc
    c = SkyCoord(ra=info["ra"] * u.deg, dec=info["dec"] * u.deg,
                 pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                 distance=dist, obstime=Time("J2000"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return c.apply_space_motion(new_obstime=Time(mjd, format="mjd"))


def describir(info, mjd):
    c0 = SkyCoord(info["ra"], info["dec"], unit="deg")
    c1 = posicion_en(info, mjd)
    sep = c0.separation(SkyCoord(c1.ra, c1.dec)).arcsec
    mu = np.hypot(info.get("pmra") or 0, info.get("pmdec") or 0) / 1000
    print(f"  {info['nombre']}: J2000 = ({info['ra']:.5f}, {info['dec']:+.5f}), "
          f"mov. propio = {mu:.2f}\"/año")
    print(f"  en {Time(mjd, format='mjd').iso[:10]} está en ({c1.ra.deg:.5f}, "
          f"{c1.dec.deg:+.5f}) → se movió {sep:.0f}\" ({sep / 60:.1f}') desde J2000")
