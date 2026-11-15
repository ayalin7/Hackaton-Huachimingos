"""
alerce_capa.py — Capa de ALeRCE para la herramienta SPHEREx.

Busca en ALeRCE (broker chileno de alertas de ZTF y LSST/Rubin) los objetos
conocidos dentro del parche de cielo, y entrega:
  - una tabla con su clase (asteroide, supernova, estrella variable, AGN...),
  - si estuvieron "activos" durante las fechas de las imágenes SPHEREx,
  - marcas (círculos con etiqueta) para dibujar sobre el video.

Instalación:  pip install alerce
"""
import json
import os
import time
import warnings

import numpy as np

CACHE_ALERCE = "cache_alerce"
EN_MEMORIA = False   # True en el servidor: no se escribe en disco
_RAM = {}

# Color de cada familia de clases (R, G, B)
COLORES = {
    "asteroide": (255, 220, 0),     # amarillo
    "transiente": (255, 70, 70),    # rojo: supernovas y otros explosivos
    "variable": (0, 220, 255),      # celeste: estrellas variables
    "agn": (255, 0, 255),           # magenta: núcleos activos de galaxias
    "otro": (120, 255, 120),        # verde
}

FAMILIAS = {
    "asteroide": {"asteroid", "ast", "sso"},
    "transiente": {"sn", "snia", "snibc", "snii", "slsn", "tde", "kn", "cv/nova",
                   "sn-like", "sn-like/other", "fast-other", "long-other"},
    "variable": {"lpv", "rrl", "cep", "e", "eb", "dsct", "periodic-other",
                 "yso", "cv", "varstar", "periodic", "stochastic", "vs"},
    "agn": {"agn", "qso", "blazar", "agn-like"},
    "descartar": {"bogus", "satellite"},
}


# Nombre en español de cada clase de ALeRCE (para el panel del video)
NOMBRES = {
    "vs": "estrella variable", "sn": "supernova", "agn": "AGN", "asteroid": "asteroide",
    "rrl": "RR Lyrae", "lpv": "variable periodo largo", "cep": "cefeida",
    "e": "binaria eclipsante", "eb": "binaria eclipsante", "dsct": "delta Scuti",
    "periodic-other": "variable periódica", "yso": "estrella joven (YSO)",
    "cv/nova": "variable cataclísmica", "cv": "variable cataclísmica",
    "qso": "cuásar", "blazar": "blázar", "snia": "supernova Ia", "snibc": "supernova Ibc",
    "snii": "supernova II", "slsn": "supernova superluminosa", "tde": "disrupción de marea",
}


def nombre_clase(clase):
    return NOMBRES.get(str(clase or "").strip().lower(), str(clase or "objeto ALeRCE"))


def familia(clase):
    c = str(clase or "").strip().lower()
    for fam, nombres in FAMILIAS.items():
        if c in nombres:
            return fam
    return "otro"


def _primera(fila, nombres, defecto=None):
    """Devuelve el primer campo que exista y no esté vacío (los nombres cambian
    entre ZTF y LSST)."""
    for n in nombres:
        v = fila.get(n)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return defecto


def _normalizar(fila, survey, clasificador):
    ra = _primera(fila, ["meanra", "ra", "mean_ra"])
    dec = _primera(fila, ["meandec", "dec", "mean_dec"])
    if ra is None or dec is None:
        return None
    clase = _primera(fila, ["class", "class_name", "classname"], "")
    oid = str(_primera(fila, ["oid", "objectId", "diaObjectId"], ""))
    return {
        "oid": oid,
        "survey": survey,
        "ra": float(ra),
        "dec": float(dec),
        "clase": str(clase),
        "clasificador": clasificador,
        "probabilidad": _primera(fila, ["probability", "prob"]),
        "firstmjd": _primera(fila, ["firstmjd", "first_mjd"]),
        "lastmjd": _primera(fila, ["lastmjd", "last_mjd"]),
        "ndet": _primera(fila, ["ndet", "n_det"]),
        "familia": familia(clase),
        "link": f"https://alerce.online/object/{oid}" if survey == "ztf" else "",
    }


def buscar_alerce(ra, dec, radio_arcsec, surveys=("ztf", "lsst"), reintentos=2):
    """
    Busca objetos de ALeRCE en un círculo. Guarda el resultado en disco: si
    ALeRCE no responde, usa la última búsqueda guardada.
    """
    archivo = os.path.join(CACHE_ALERCE, f"{ra:.4f}_{dec:+.4f}_{radio_arcsec:.0f}.json")
    if EN_MEMORIA and archivo in _RAM:
        return [dict(o) for o in _RAM[archivo]]
    if not EN_MEMORIA:
        os.makedirs(CACHE_ALERCE, exist_ok=True)

    try:
        from alerce.core import Alerce
    except ImportError:
        print("  ALeRCE: falta la librería. Instálala con:  pip install alerce")
        return [] if EN_MEMORIA else _leer_cache(archivo)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cliente = Alerce()

    objetos, algo_respondio = {}, False
    # Para ZTF consultamos dos clasificadores: el de curva de luz (variables,
    # supernovas...) y el de "stamps" (que es el que reconoce asteroides).
    consultas = [("ztf", None), ("ztf", "stamp_classifier"), ("lsst", None)]
    for survey, clasif in consultas:
        if survey not in surveys:
            continue
        params = dict(survey=survey, ra=ra, dec=dec, radius=radio_arcsec,
                      page_size=500, format="pandas")
        if clasif:
            params["classifier"] = clasif
        for i in range(reintentos + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = cliente.query_objects(**params)
                algo_respondio = True
                break
            except Exception as e:
                if i == reintentos:
                    print(f"  ALeRCE {survey}{'/' + clasif if clasif else ''}: "
                          f"no respondió ({type(e).__name__})")
                    df = None
                else:
                    time.sleep(2)
        if df is None or len(df) == 0:
            continue
        for fila in df.to_dict("records"):
            obj = _normalizar(fila, survey, clasif or "lc_classifier")
            if obj is None:
                continue
            # Guardamos las DOS clasificaciones por separado; la elección final
            # (preferir la de curva de luz) se hace en filtrar().
            previo = objetos.setdefault(obj["oid"], obj)
            clave = "temprana" if obj["clasificador"] == "stamp_classifier" else "lc"
            previo[clave] = {"clase": obj["clase"], "probabilidad": obj["probabilidad"],
                             "familia": obj["familia"]}

    lista = list(objetos.values())
    if algo_respondio:
        if EN_MEMORIA:
            if len(_RAM) > 300:
                _RAM.clear()
            _RAM[archivo] = [dict(o) for o in lista]
        else:
            with open(archivo, "w") as f:
                json.dump(lista, f, indent=2, default=str)
        return lista
    return [] if EN_MEMORIA else _leer_cache(archivo)


def _leer_cache(archivo):
    if os.path.exists(archivo):
        print("  ALeRCE: uso la búsqueda guardada", archivo)
        with open(archivo) as f:
            return json.load(f)
    return []


def _elegir_clase(o, prob_min, prob_temprana):
    """
    ALeRCE tiene dos clasificadores:
      - curva de luz (lc): usa MUCHAS observaciones → más confiable;
      - "stamps" (temprano): usa solo la PRIMERA alerta → es una pista.
    Se prefiere la de curva de luz; la temprana solo se acepta con más
    probabilidad. Devuelve el objeto con clase/familia/probabilidad finales,
    o None si ninguna clasificación es suficientemente confiable.
    """
    # Compatibilidad con búsquedas guardadas por versiones anteriores
    if "lc" not in o and "temprana" not in o:
        clave = "temprana" if o.get("clasificador") == "stamp_classifier" else "lc"
        o[clave] = {"clase": o["clase"], "probabilidad": o.get("probabilidad"),
                    "familia": o["familia"]}
    lc, tp = o.get("lc"), o.get("temprana")

    def ok(c, pmin):
        if not c or c["familia"] in ("otro", "descartar") or not c["clase"]:
            return False
        p = c.get("probabilidad")
        return p is None or float(p) >= pmin

    if ok(lc, prob_min):
        c, temprana = lc, False
    elif ok(tp, prob_temprana):
        c, temprana = tp, True
    else:
        return None
    return {**o, "clase": c["clase"], "familia": c["familia"],
            "probabilidad": c.get("probabilidad"), "temprana": temprana,
            "nombre": nombre_clase(c["clase"])}


def filtrar(objetos, prob_min=0.5, prob_temprana=0.7, incluir_asteroides=False,
            max_objetos=200):
    """
    Limpia la lista:
      - quita detecciones falsas ('bogus') y asteroides (ZTF los vio en OTRAS
        fechas; para eso usa --asteroides),
      - prefiere la clasificación de curva de luz; la temprana exige prob >= 0.7,
      - ordena: activos en las fechas SPHEREx primero, luego más probables.
    """
    salida = []
    for o in objetos:
        o = _elegir_clase(dict(o), prob_min, prob_temprana)
        if o is None:
            continue
        if o["familia"] == "asteroide" and not incluir_asteroides:
            continue
        salida.append(o)
    salida.sort(key=lambda o: (not o.get("activo_en_spherex", False),
                               -float(o.get("probabilidad") or 0)))
    return salida[:max_objetos]


def marcar_actividad(objetos, mjd_min, mjd_max, margen_dias=30):
    """Marca 'activo_en_spherex' si ALeRCE lo detectó cerca de las fechas SPHEREx."""
    for o in objetos:
        f0, f1 = o.get("firstmjd"), o.get("lastmjd")
        o["activo_en_spherex"] = bool(
            f0 is not None and f1 is not None
            and float(f0) <= mjd_max + margen_dias and float(f1) >= mjd_min - margen_dias)
    return objetos


def marcas_en_grilla(objetos, wcs_out, forma):
    """Convierte RA/Dec de cada objeto a píxeles de la grilla común."""
    from astropy.coordinates import SkyCoord
    if not objetos:
        return []
    c = SkyCoord([o["ra"] for o in objetos], [o["dec"] for o in objetos], unit="deg")
    x, y = wcs_out.world_to_pixel(c)
    marcas = []
    for o, xi, yi in zip(objetos, np.atleast_1d(x), np.atleast_1d(y)):
        if 0 <= xi < forma[1] and 0 <= yi < forma[0]:
            marcas.append({**o, "x": float(xi), "y": float(yi)})
    return marcas


def imprimir_tabla(objetos, total_sin_filtro=None):
    if not objetos:
        print("  ALeRCE: no hay objetos (que pasen los filtros) en este parche.")
        return
    extra = f" (de {total_sin_filtro} antes de filtrar)" if total_sin_filtro else ""
    print(f"  ALeRCE: {len(objetos)} objetos marcados{extra}")
    conteo = {}
    for o in objetos:
        conteo[o["familia"]] = conteo.get(o["familia"], 0) + 1
    print("  por familia:", ", ".join(f"{k}={v}" for k, v in conteo.items()))
    print("  survey  oid                    clase            activo  link")
    for o in objetos[:15]:
        act = "sí" if o.get("activo_en_spherex") else "-"
        print(f"  {o['survey']:<6}  {o['oid']:<21}  {o['clase'][:15]:<15}  "
              f"{act:<6}  {o['link']}")
