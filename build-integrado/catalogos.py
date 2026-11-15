"""
catalogos.py — SIMBAD y WISE para identificar las fuentes del campo.

SIMBAD (CDS, Estrasburgo): nombre y TIPO de objetos ya estudiados
    (RR Lyrae, galaxia, enana marrón, estrella joven, cuásar...).
WISE (catálogo AllWISE, servido por IRSA): fuentes infrarrojas a 3.4, 4.6, 12 y
    22 µm (W1..W4). Encuentra objetos muy rojos o fríos que Gaia no ve, y su
    color W1−W2 da pistas:
        W1−W2 ≳ 0.8  → típico de AGN/cuásares (criterio de Stern et al. 2012)
        W1−W2 ≳ 2    → muy rojo: candidato a enana marrón fría
    ext_flg > 0 → la fuente se ve extendida (posible galaxia)
    cc_flags ≠ 0 → WISE avisa que puede estar contaminada por un artefacto

Las búsquedas se guardan en cache_catalogos/ (no se repiten al re-ejecutar).
"""
import json
import os
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

CACHE = "cache_catalogos"
LIMITE_SEG = 90      # tiempo máximo de espera por consulta a cada servicio
EN_MEMORIA = False
_RAM = {}


def _cache(nombre, calcular):
    archivo = os.path.join(CACHE, nombre + ".json")
    if EN_MEMORIA:
        if nombre not in _RAM:
            _RAM[nombre] = calcular()
        return _RAM[nombre]
    if os.path.exists(archivo):
        with open(archivo) as f:
            return json.load(f)
    datos = calcular()
    if datos is not None:
        os.makedirs(CACHE, exist_ok=True)
        with open(archivo, "w") as f:
            json.dump(datos, f)
    return datos


class SinRespuesta(Exception):
    pass


def con_limite(funcion, segundos=None):
    """
    Ejecuta `funcion()` en un hilo aparte y espera como máximo `segundos`.
    Si no termina a tiempo lanza SinRespuesta y el programa sigue (el hilo
    queda como 'daemon', así que no impide que el programa termine).
    """
    import threading
    import queue
    segundos = segundos or LIMITE_SEG
    q = queue.Queue()

    def correr():
        try:
            q.put(("ok", funcion()))
        except Exception as e:
            q.put(("error", e))

    threading.Thread(target=correr, daemon=True).start()
    try:
        estado, valor = q.get(timeout=segundos)
    except queue.Empty:
        raise SinRespuesta(f"sin respuesta en {segundos} s")
    if estado == "error":
        raise valor
    return valor


def _aplanar(v):
    if np.ma.is_masked(v):
        return None
    if isinstance(v, bytes):
        v = v.decode()
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


# ------------------------------------------------------------------ SIMBAD
SERVIDORES_SIMBAD = ("simbad.cds.unistra.fr", "simbad.harvard.edu")


def consulta_simbad(adql):
    """
    Ejecuta una consulta ADQL en SIMBAD probando los dos servidores:
    primero el principal (CDS, Estrasburgo) y, si falla, el espejo de Harvard
    (mismos datos). Devuelve (tabla, servidor) o lanza el último error.
    """
    from astroquery.simbad import SimbadClass
    ultimo = None
    for srv in SERVIDORES_SIMBAD:
        try:
            cliente = SimbadClass()
            cliente.server = srv

            def pedir():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return cliente.query_tap(adql)
            return con_limite(pedir), srv
        except Exception as e:
            ultimo = e
            print(f"  SIMBAD ({srv}) no respondió ({type(e).__name__}: {str(e)[:40]})"
                  + ("; pruebo el espejo de Harvard" if srv != SERVIDORES_SIMBAD[-1] else ""))
    raise ultimo


def simbad_campo(ra, dec, radio_arcsec):
    """Objetos SIMBAD en el cono, con su tipo. Devuelve lista de dicts o None."""
    def pedir():
        r_deg = radio_arcsec / 3600
        cono = f"CONTAINS(POINT('ICRS', {{p}}ra, {{p}}dec), CIRCLE('ICRS', {ra}, {dec}, {r_deg})) = 1"
        consultas = [
            # con la descripción del tipo (tabla otypedef)
            "SELECT TOP 5000 b.main_id, b.ra, b.dec, b.otype, d.description "
            "FROM basic AS b LEFT JOIN otypedef AS d ON b.otype = d.otype WHERE "
            + cono.format(p="b."),
            # respaldo sin descripción
            "SELECT TOP 5000 main_id, ra, dec, otype FROM basic WHERE " + cono.format(p=""),
        ]
        for q in consultas:
            try:
                t, srv = consulta_simbad(q)
            except Exception:
                continue
            cols = {c.lower(): c for c in t.colnames}
            filas = []
            for f in t:
                a, b = _aplanar(f[cols["ra"]]), _aplanar(f[cols["dec"]])
                if a is None or b is None:
                    continue
                filas.append({"nombre": str(_aplanar(f[cols["main_id"]]) or ""),
                              "ra": float(a), "dec": float(b),
                              "otype": str(_aplanar(f[cols["otype"]]) or ""),
                              "descripcion": str(_aplanar(f[cols["description"]]) or "")
                              if "description" in cols else ""})
            if srv != SERVIDORES_SIMBAD[0]:
                print(f"  (SIMBAD respondió desde {srv})")
            return filas
        print("  SIMBAD: ningún servidor respondió; se sigue sin SIMBAD")
        return None

    datos = _cache(f"simbad_{ra:.4f}_{dec:+.4f}_{radio_arcsec:.0f}", pedir)
    if datos is not None:
        print(f"  SIMBAD: {len(datos)} objetos en el campo")
    return datos


# Tipo SIMBAD → (familia, nombre en español). Se revisa en orden.
_TIPOS = [
    (("RR*", "RRLyr"), "variable", "RR Lyrae"),
    (("Mi*", "Mira"), "variable", "variable Mira"),
    (("LP*", "LongPeriodV"), "variable", "variable de periodo largo"),
    (("EB*", "EclBin"), "variable", "binaria eclipsante"),
    (("Ce*", "cC*", "Cepheid", "classicalCep"), "variable", "cefeida"),
    (("dS*", "delSctV"), "variable", "delta Scuti"),
    (("TT*", "TTauri", "Y*O", "YSO", "Or*", "Ae*", "Y*?"), "variable", "estrella joven"),
    (("CV*", "No*", "Nova"), "variable", "variable cataclísmica"),
    (("V*", "Variable", "Pu*", "Ro*", "BY*", "RS*", "Er*", "Fl*"), "variable", "estrella variable"),
    (("BD*", "BrownD", "BD?"), "especial", "enana marrón"),
    (("WD*", "WhiteDwarf"), "estrella", "enana blanca"),
    (("PM*", "HighPM*"), "estrella", "estrella (alto mov. propio)"),
    (("QSO", "Q?"), "agn", "cuásar"),
    (("AGN", "Sy1", "Sy2", "Seyfert", "Bla", "BLL", "LIN"), "agn", "AGN"),
    (("SN*", "SN"), "transiente", "supernova"),
    (("G", "GiG", "GiC", "GiP", "BiC", "IG", "PaG", "SBG", "bCG", "EmG", "H2G", "LSB", "rG",
      "Galaxy", "G?"), "galaxia", "galaxia"),
    (("Cl*", "GlC", "OpC", "C?*"), "otro", "cúmulo"),
    (("HII", "PN", "RNe", "Cld", "ISM", "MoC", "DNe"), "otro", "nebulosa / medio interestelar"),
    (("**", "SB*", "El*"), "estrella", "estrella doble"),
    (("*", "Star", "MS*", "Ev*", "RG*", "HB*", "s*r", "s*y", "s*b", "LM*", "Em*",
      "C*", "S*", "OH*", "WR*", "Be*"), "estrella", "estrella"),
]


def clasificar_simbad(otype, descripcion=""):
    """Devuelve (familia, nombre_es, informativo)."""
    t = (otype or "").strip()
    for codigos, fam, nombre in _TIPOS:
        if t in codigos:
            return fam, nombre, nombre not in ("estrella",)
    d = (descripcion or "").lower()
    for clave, fam, nombre in (("galaxy", "galaxia", "galaxia"), ("quasar", "agn", "cuásar"),
                               ("variable", "variable", "estrella variable"),
                               ("brown", "especial", "enana marrón"),
                               ("young", "variable", "estrella joven"), ("star", "estrella", "estrella")):
        if clave in d:
            return fam, nombre, nombre != "estrella"
    return "otro", (descripcion or t or "objeto SIMBAD"), bool(t)


# -------------------------------------------------------------------- WISE
def wise_campo(ra, dec, radio_arcsec):
    """Fuentes AllWISE en el cono (vía IRSA). Devuelve lista de dicts o None."""
    def pedir():
        from astroquery.ipac.irsa import Irsa
        cols = "designation,ra,dec,w1mpro,w2mpro,w3mpro,w4mpro,cc_flags,ext_flg,ph_qual"
        def consulta():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return Irsa.query_region(SkyCoord(ra, dec, unit="deg"),
                                         catalog="allwise_p3as_psd", spatial="Cone",
                                         radius=radio_arcsec * u.arcsec, columns=cols)
        print(f"  WISE: consultando IRSA (espera máx. {LIMITE_SEG} s)...")
        for i in range(2):
            try:
                t = con_limite(consulta)
                break
            except Exception as e:
                print(f"  WISE (IRSA) no respondió ({type(e).__name__}: {str(e)[:40]})")
                if i == 1 or isinstance(e, SinRespuesta):
                    print("  WISE: se sigue sin WISE (prueba de nuevo más tarde o usa --sin-wise)")
                    return None
        filas = []
        for f in t:
            d = {k: _aplanar(f[k]) for k in t.colnames}
            if d.get("ra") is None or d.get("dec") is None:
                continue
            filas.append({"nombre": f"AllWISE {d.get('designation') or ''}".strip(),
                          "ra": float(d["ra"]), "dec": float(d["dec"]),
                          "W1": d.get("w1mpro"), "W2": d.get("w2mpro"),
                          "W3": d.get("w3mpro"), "W4": d.get("w4mpro"),
                          "cc_flags": str(d.get("cc_flags") or "0000"),
                          "ext_flg": int(d.get("ext_flg") or 0),
                          "ph_qual": str(d.get("ph_qual") or "")})
        return filas

    datos = _cache(f"wise_{ra:.4f}_{dec:+.4f}_{radio_arcsec:.0f}", pedir)
    if datos is not None:
        print(f"  WISE: {len(datos)} fuentes en el campo")
    return datos


def clasificar_wise(w):
    """Pista a partir del color y la forma en WISE. Devuelve (familia, texto)."""
    w1, w2 = w.get("W1"), w.get("W2")
    color = (w1 - w2) if (w1 is not None and w2 is not None) else None
    if w.get("ext_flg", 0) > 0:
        return "galaxia", "extendida en WISE (¿galaxia?)"
    if color is not None and color >= 2.0:
        return "especial", "muy roja en WISE (¿enana marrón?)"
    if color is not None and color >= 0.8:
        return "agn", "roja en WISE (¿AGN?)"
    return "estrella", "fuente WISE"


def contaminada_wise(w):
    """WISE marca contaminación por artefacto (difracción, halo, fantasma...)."""
    return any(c not in "0" for c in (w.get("cc_flags") or "0000")[:2])


# ------------------------------------------------------------ cruce común
def a_arrays(lista):
    if not lista:
        return None
    return {"ra": np.array([o["ra"] for o in lista]), "dec": np.array([o["dec"] for o in lista]),
            "lista": lista,
            "coord": SkyCoord([o["ra"] for o in lista], [o["dec"] for o in lista], unit="deg")}


# ---------------------------------------------------------------- Gaia DR3
def gaia_dr3_campo(ra, dec, radio_arcsec, mjd_epoca=None, g_max=20.5):
    """
    Estrellas Gaia DR3 del cono (archivo de la ESA), CON paralaje y movimiento
    propio. Si se da `mjd_epoca`, las posiciones (época J2016.0) se llevan a esa
    fecha con su movimiento propio: indispensable para estrellas rápidas como
    Barnard's Star (~10"/año). Devuelve el mismo formato que
    identificar.gaia_campo(), o None si la ESA no responde.
    """
    def pedir():
        from astroquery.gaia import Gaia
        q = (f"SELECT source_id, ra, dec, parallax, parallax_error, pmra, pmdec, "
             f"phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1 = CONTAINS("
             f"POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {radio_arcsec / 3600})) "
             f"AND phot_g_mean_mag < {g_max}")

        def correr():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return Gaia.launch_job_async(q, verbose=False).get_results()
        try:
            t = con_limite(correr)
        except Exception as e:
            print(f"  Gaia DR3 (ESA) no respondió ({type(e).__name__}: {str(e)[:40]})")
            return None
        cols = {c.lower(): c for c in t.colnames}
        filas = []
        for f in t:
            d = {k: _aplanar(f[cols[k]]) for k in
                 ("source_id", "ra", "dec", "parallax", "parallax_error", "pmra", "pmdec",
                  "phot_g_mean_mag") if k in cols}
            if d.get("ra") is None or d.get("dec") is None:
                continue
            filas.append(d)
        return filas

    filas = _cache(f"gaiadr3_{ra:.4f}_{dec:+.4f}_{radio_arcsec:.0f}", pedir)
    if filas is None:
        return None

    def col(nombre):
        return np.array([f.get(nombre) if f.get(nombre) is not None else np.nan
                         for f in filas], float)
    g = {"ra": col("ra"), "dec": col("dec"), "G": col("phot_g_mean_mag"),
         "plx": col("parallax"), "eplx": col("parallax_error"),
         "pmra": col("pmra"), "pmdec": col("pmdec"),
         "source_id": [str(f.get("source_id")) for f in filas],
         "radio_arcsec": float(radio_arcsec), "version": "DR3"}
    if mjd_epoca is not None and len(filas):
        # Gaia DR3 entrega posiciones en la época J2016.0 (MJD 57388.0)
        dt_anios = (mjd_epoca - 57388.0) / 365.25
        g["dec"] = g["dec"] + np.nan_to_num(g["pmdec"]) * dt_anios / 3.6e6
        g["ra"] = g["ra"] + (np.nan_to_num(g["pmra"]) * dt_anios / 3.6e6
                             / np.cos(np.radians(g["dec"])))
    print(f"  Gaia DR3 (ESA): {len(filas)} estrellas en el campo"
          + (" (posiciones llevadas a la fecha de SPHEREx)" if mjd_epoca is not None else ""))
    return g
