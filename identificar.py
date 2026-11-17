"""
identificar.py — ¿Qué es cada punto del video? Respuesta usando ALeRCE.

1. Detecta las fuentes en la imagen apilada (todas las fechas combinadas, con
   menos ruido) y las numera de la más brillante a la más débil.
2. Las cruza con:
   - los objetos clasificados por ALeRCE (ZTF/LSST): estrella variable,
     supernova, AGN/quásar, etc., con su probabilidad;
   - si ALeRCE no tiene nada ahí, con el catálogo Gaia DR2 que ALeRCE sirve a
     través de catsHTM: si Gaia mide paralaje, es una estrella de nuestra galaxia.
3. Lo que no aparece en ninguno queda como "sin identificar" (¡lo interesante!).
"""
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

COLORES = {
    "variable": (0, 220, 255),      # celeste
    "transiente": (255, 70, 70),    # rojo
    "agn": (255, 0, 255),           # magenta
    "asteroide": (255, 220, 0),     # amarillo
    "estrella": (140, 255, 140),    # verde
    "gaia": (170, 200, 170),        # verde grisáceo: fuente Gaia sin paralaje
    "otro": (200, 200, 200),
    "sin_identificar": (255, 150, 0),  # naranjo
    "galaxia": (255, 190, 120),     # durazno
    "especial": (190, 120, 255),    # lila: enanas marrones y otros raros
}


# --------------------------------------------------------------- detección
def detectar_fuentes(img, n_max=20, sigma=5.0, sep_px=3, ventana=None):
    """
    Picos sobre el fondo. Devuelve lista de dicts con x, y, señal/ruido.

    1) máximo local en una ventana ESTRECHA (≈ 1 píxel nativo), así una
       estrella junto a otra más brillante no queda tapada por su ala;
    2) luego, de la más brillante a la más débil, se acepta cada pico solo si
       está a más de `sep_px` de los ya aceptados (evita dos picos por estrella).
    """
    from scipy.ndimage import maximum_filter
    datos = np.where(np.isfinite(img), img, np.nanmedian(img))
    med = np.nanmedian(datos)
    ruido = 1.4826 * np.nanmedian(np.abs(datos - med)) or np.nanstd(datos) or 1.0
    ventana = ventana or max(3, sep_px // 2 * 2 + 1 - 2)
    picos = (datos == maximum_filter(datos, size=ventana)) & (datos > med + sigma * ruido)
    borde = max(2, sep_px // 2)
    picos[:borde, :] = picos[-borde:, :] = picos[:, :borde] = picos[:, -borde:] = False
    ys, xs = np.nonzero(picos)
    snr = (datos[ys, xs] - med) / ruido
    aceptados = []
    for k in np.argsort(-snr):
        if all((xs[k] - xs[j]) ** 2 + (ys[k] - ys[j]) ** 2 > sep_px ** 2 for j in aceptados):
            aceptados.append(k)
        if len(aceptados) >= n_max:
            break
    return [{"num": i + 1, "x": float(xs[k]), "y": float(ys[k]), "snr": float(snr[k])}
            for i, k in enumerate(aceptados)]


# ------------------------------------------------------- verificación
def _pico_en(img, x, y, caja):
    """S/R del máximo local en una caja alrededor de (x, y). None si no hay datos."""
    ny, nx = img.shape
    x0, x1 = int(round(x)) - caja, int(round(x)) + caja + 1
    y0, y1 = int(round(y)) - caja, int(round(y)) + caja + 1
    if x0 < 0 or y0 < 0 or x1 > nx or y1 > ny:
        return None
    sub = img[y0:y1, x0:x1]
    if np.isfinite(sub).sum() < sub.size / 2:
        return None
    med = np.nanmedian(img)
    ruido = 1.4826 * np.nanmedian(np.abs(img - med))
    if not np.isfinite(ruido) or ruido == 0:
        return None
    return float((np.nanmax(sub) - med) / ruido)


def verificar(fuentes, cubo, avisos, apilar, caja=1, umbral_mitad=3.0,
              frac_aviso=0.3, min_cuadros=4):
    """
    Decide si cada fuente detectada es REAL. Tres pruebas:

    1. Prueba de las dos mitades: los cuadros se reparten en dos grupos
       independientes (pares e impares en el tiempo) y se apila cada grupo.
       Una estrella real aparece en ambos (S/R >= umbral_mitad); un rayo
       cósmico, un píxel malo o un reflejo de UNA exposición aparece en uno solo.
    2. Cobertura: la posición debe tener datos en al menos `min_cuadros` cuadros.
    3. Avisos: si en >= frac_aviso de los cuadros la fuente cae sobre píxeles
       marcados como fantasma óptico o crosstalk, se marca "posible fantasma".

    Además cuenta en cuántos cuadros individuales se ve (útil en las brillantes).
    Agrega a cada fuente: confirmada (bool), motivo, snr_mitades, n_cuadros_visto, ...
    """
    n = len(cubo)
    if n >= 2:
        pares, impares = list(range(0, n, 2)), list(range(1, n, 2))
        mitad_a, mitad_b = apilar(cubo[pares]), apilar(cubo[impares])
    else:
        mitad_a = mitad_b = None
    for f in fuentes:
        x, y = f["x"], f["y"]
        cubiertos = [k for k in range(n) if _pico_en(cubo[k], x, y, caja) is not None]
        vistos = [k for k in cubiertos if _pico_en(cubo[k], x, y, caja) >= 3]
        f["n_cuadros_cubre"] = len(cubiertos)
        f["n_cuadros_visto"] = len(vistos)
        if avisos is not None and len(avisos) == n and cubiertos:
            xi, yi = int(round(x)), int(round(y))
            con_aviso = sum(bool(avisos[k][max(0, yi - caja):yi + caja + 1,
                                           max(0, xi - caja):xi + caja + 1].any())
                            for k in cubiertos)
            f["frac_aviso"] = round(con_aviso / len(cubiertos), 2)
        else:
            f["frac_aviso"] = 0.0
        sa = _pico_en(mitad_a, x, y, caja) if mitad_a is not None else None
        sb = _pico_en(mitad_b, x, y, caja) if mitad_b is not None else None
        f["snr_mitades"] = [None if v is None else round(v, 1) for v in (sa, sb)]

        if len(cubiertos) < min_cuadros:
            f["confirmada"], f["motivo"] = False, f"pocos cuadros con datos ({len(cubiertos)})"
        elif sa is None or sb is None or min(sa, sb) < umbral_mitad:
            f["confirmada"], f["motivo"] = False, "no aparece en las dos mitades"
        else:
            f["confirmada"], f["motivo"] = True, "aparece en las dos mitades"
        f["posible_fantasma"] = f["frac_aviso"] >= frac_aviso
    return fuentes


# ------------------------------------------------------- Gaia vía ALeRCE
def _columna(df, *candidatos):
    cols = {c.lower(): c for c in df.columns}
    for c in candidatos:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def gaia_campo(ra, dec, radio_arcsec):
    """Estrellas Gaia DR2 del campo, desde el servicio catsHTM de ALeRCE."""
    try:
        from alerce.core import Alerce
        import catalogos

        def pedir():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return Alerce().catshtm_conesearch(ra, dec, radio_arcsec, "GAIA/DR2",
                                                   format="pandas")
        df = catalogos.con_limite(pedir)
    except Exception as e:
        print(f"  Gaia (catsHTM de ALeRCE) no respondió ({type(e).__name__})")
        return None
    if df is None or len(df) == 0:
        return None
    if not hasattr(df, "columns"):          # una sola fila llega como Serie
        df = df.to_frame().T
    c_ra, c_dec = _columna(df, "ra", "RA"), _columna(df, "dec", "Dec", "DEC")
    if c_ra is None or c_dec is None:
        return None
    a = np.asarray(df[c_ra], float)
    b = np.asarray(df[c_dec], float)
    # catsHTM podría entregar radianes. Solo se prueba esa lectura si los valores
    # caben en el rango de radianes; se elige la que cae más cerca del campo.
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    df = df[ok]
    cen = SkyCoord(ra, dec, unit="deg")
    puede_ser_rad = (np.all(np.abs(a) <= 2 * np.pi + 1e-6)
                     and np.all(np.abs(b) <= np.pi / 2 + 1e-6))
    puede_ser_deg = np.all(np.abs(b) <= 90)
    if puede_ser_rad:
        sep_rad = np.nanmedian(cen.separation(
            SkyCoord(np.degrees(a), np.degrees(b), unit="deg")).arcsec)
        sep_deg = (np.nanmedian(cen.separation(SkyCoord(a, b, unit="deg")).arcsec)
                   if puede_ser_deg else np.inf)
        if sep_rad < sep_deg:
            a, b = np.degrees(a), np.degrees(b)
    elif not puede_ser_deg:
        print("  Gaia: coordenadas en un formato desconocido; se omite Gaia")
        return None
    c_g = _columna(df, "Mag_G", "phot_g_mean_mag", "Gmag", "G")
    c_plx, c_eplx = _columna(df, "Plx", "parallax"), _columna(df, "ErrPlx", "parallax_error")
    out = {"ra": a, "dec": b,
           "G": np.asarray(df[c_g], float) if c_g else np.full(len(a), np.nan),
           "plx": np.asarray(df[c_plx], float) if c_plx else np.full(len(a), np.nan),
           "eplx": np.asarray(df[c_eplx], float) if c_eplx else np.full(len(a), np.nan),
           "radio_arcsec": float(radio_arcsec), "version": "DR2"}
    print(f"  Gaia (vía ALeRCE/catsHTM): {len(a)} fuentes en el campo")
    return out


# ------------------------------------------------------------ cruce
def _id_gaia(m_ga):
    """'Gaia DR3 123…' si se conoce el identificador; si no, 'Gaia DR2'/'Gaia DR3'."""
    v = m_ga.get("version", "DR2")
    return f"Gaia {v} {m_ga['source_id']}" if m_ga.get("source_id") else f"Gaia {v}"


def p_azar(sep_arcsec, densidad_por_arcsec2):
    """
    Probabilidad de que un objeto SIN relación caiga a esa distancia o menos
    por pura casualidad:  P = 1 - exp(-π r² ρ).
    """
    return float(1 - np.exp(-np.pi * sep_arcsec ** 2 * densidad_por_arcsec2))


def identificar(fuentes, wcs, objetos_alerce, gaia, radio_arcsec=9.0,
                area_campo_arcsec2=None, cobertura=None, p_max=0.05,
                simbad=None, wise=None, radio_cono_arcsec=None):
    """
    Asigna a cada fuente detectada lo que es, combinando 4 catálogos:

      prioridad de la ETIQUETA:
        1. ALeRCE  — clase por variabilidad (curva de luz > temprana)
        2. SIMBAD  — tipo de objeto, si es informativo (RR Lyrae, galaxia, ...)
        3. Gaia    — estrella (si mide paralaje) o fuente Gaia
        4. SIMBAD  — "estrella" genérica
        5. WISE    — fuente infrarroja; su color da una pista (¿AGN?, ¿enana marrón?)
        6. sin identificar

    Además, cada fuente guarda lo que encontró en CADA catálogo (nombre SIMBAD,
    magnitudes WISE, G de Gaia...), y la probabilidad de coincidencia casual
    del cruce que dio la etiqueta. Gaia y WISE eligen la fuente MÁS BRILLANTE
    dentro del radio (la que SPHEREx más probablemente ve).

    Devuelve (identificadas, no_detectados).
    """
    import catalogos as cat
    al = objetos_alerce or []
    c_al = SkyCoord([o["ra"] for o in al], [o["dec"] for o in al], unit="deg") if al else None
    rho_al = len(al) / area_campo_arcsec2 if (al and area_campo_arcsec2) else 0.0
    area_cono = np.pi * (radio_cono_arcsec or 1.0) ** 2
    sb = cat.a_arrays(simbad)
    wi = cat.a_arrays(wise)
    c_ga = None
    if gaia is not None and len(gaia["ra"]):
        c_ga = SkyCoord(gaia["ra"], gaia["dec"], unit="deg")
        area_ga = np.pi * gaia["radio_arcsec"] ** 2
        G_ord = np.sort(gaia["G"][np.isfinite(gaia["G"])])
    W1_ord = (np.sort([w["W1"] for w in wise if w.get("W1") is not None]) if wise else [])

    resultado, usados = [], set()
    for f in fuentes:
        c = wcs.pixel_to_world(f["x"], f["y"])
        r = {**f, "ra": float(c.ra.deg), "dec": float(c.dec.deg)}

        # --- ALeRCE
        m_al = None
        if c_al is not None:
            sep = c.separation(c_al).arcsec
            k = int(np.argmin(sep))
            if sep[k] <= radio_arcsec:
                m_al = (k, float(sep[k]))
                usados.add(k)
        # --- SIMBAD (el más cercano)
        m_sb = None
        if sb is not None:
            sep = c.separation(sb["coord"]).arcsec
            k = int(np.argmin(sep))
            if sep[k] <= radio_arcsec:
                o = sb["lista"][k]
                fam, nombre_es, informativo = cat.clasificar_simbad(o["otype"], o["descripcion"])
                m_sb = {"nombre": o["nombre"], "otype": o["otype"], "tipo_es": nombre_es,
                        "familia": fam, "informativo": informativo,
                        "sep_arcsec": round(float(sep[k]), 1),
                        "p_azar": round(p_azar(sep[k], len(sb["lista"]) / area_cono), 3)}
                r["simbad"] = m_sb
        # --- Gaia (la más brillante dentro del radio)
        m_ga = None
        if c_ga is not None:
            sep = c.separation(c_ga).arcsec
            dentro = np.where(sep <= radio_arcsec)[0]
            if len(dentro):
                G = gaia["G"][dentro]
                k = dentro[np.nanargmin(np.where(np.isfinite(G), G, 99))]
                plx, eplx, g = gaia["plx"][k], gaia["eplx"][k], gaia["G"][k]
                estrella = bool(np.isfinite(plx) and np.isfinite(eplx) and eplx > 0 and plx / eplx > 5)
                n_b = np.searchsorted(G_ord, g, side="right") if np.isfinite(g) and len(G_ord) else len(gaia["ra"])
                m_ga = {"G": float(g) if np.isfinite(g) else None, "estrella": estrella,
                        "distancia_pc": float(1000 / plx) if estrella else None,
                        "sep_arcsec": round(float(sep[k]), 1),
                        "p_azar": round(p_azar(sep[k], max(n_b, 1) / area_ga), 3)}
                # Gaia DR3 trae además identificador, paralaje y movimiento propio
                for clave in ("source_id", "pmra", "pmdec"):
                    if gaia.get(clave) is not None:
                        v = gaia[clave][k]
                        if clave == "source_id":
                            m_ga[clave] = str(v)
                        elif np.isfinite(v):
                            m_ga[clave] = float(v)
                if np.isfinite(plx):
                    m_ga["paralaje_mas"] = float(plx)
                m_ga["version"] = gaia.get("version", "DR2")
                r["gaia"] = m_ga
                r["G"] = m_ga["G"]
                if estrella:
                    r["distancia_pc"] = m_ga["distancia_pc"]
        # --- WISE (la más brillante en W1 dentro del radio)
        m_wi = None
        if wi is not None:
            sep = c.separation(wi["coord"]).arcsec
            dentro = np.where(sep <= radio_arcsec)[0]
            if len(dentro):
                w1 = np.array([wi["lista"][k]["W1"] if wi["lista"][k]["W1"] is not None else 99
                               for k in dentro])
                k = dentro[int(np.argmin(w1))]
                w = wi["lista"][k]
                fam_w, txt_w = cat.clasificar_wise(w)
                n_b = (np.searchsorted(W1_ord, w["W1"], side="right")
                       if w.get("W1") is not None and len(W1_ord) else len(wi["lista"]))
                m_wi = {"nombre": w["nombre"], "W1": w.get("W1"), "W2": w.get("W2"),
                        "W1_W2": (round(w["W1"] - w["W2"], 2)
                                  if w.get("W1") is not None and w.get("W2") is not None else None),
                        "familia": fam_w, "pista": txt_w,
                        "contaminada": cat.contaminada_wise(w),
                        "sep_arcsec": round(float(sep[k]), 1),
                        "p_azar": round(p_azar(sep[k], max(n_b, 1) / area_cono), 3)}
                r["wise"] = m_wi

        # --- etiqueta final, por prioridad
        if m_al is not None:
            o = al[m_al[0]]
            pa = p_azar(m_al[1], rho_al)
            r.update(tipo=o["familia"], etiqueta=o.get("nombre") or o["clase"],
                     clase=o["clase"], fuente_id=o["oid"], catalogo="ALeRCE",
                     survey=o["survey"], link=o.get("link", ""),
                     temprana=o.get("temprana", False),
                     probabilidad=None if o.get("probabilidad") is None else float(o["probabilidad"]),
                     activo=o.get("activo_en_spherex", False),
                     sep_arcsec=round(m_al[1], 1), p_azar=round(pa, 3))
        elif m_sb is not None and m_sb["informativo"]:
            r.update(tipo=m_sb["familia"], etiqueta=m_sb["tipo_es"], fuente_id=m_sb["nombre"],
                     catalogo="SIMBAD", sep_arcsec=m_sb["sep_arcsec"], p_azar=m_sb["p_azar"])
        elif m_ga is not None and m_ga["p_azar"] <= p_max:
            r.update(tipo="estrella" if m_ga["estrella"] else "gaia",
                     etiqueta="estrella" if m_ga["estrella"] else "fuente Gaia",
                     fuente_id=(m_sb["nombre"] if m_sb else _id_gaia(m_ga)), catalogo="Gaia",
                     sep_arcsec=m_ga["sep_arcsec"], p_azar=m_ga["p_azar"])
        elif m_sb is not None:
            r.update(tipo=m_sb["familia"], etiqueta=m_sb["tipo_es"], fuente_id=m_sb["nombre"],
                     catalogo="SIMBAD", sep_arcsec=m_sb["sep_arcsec"], p_azar=m_sb["p_azar"])
        elif m_wi is not None and (m_wi["p_azar"] <= p_max or m_ga is None):
            # (si el cruce con Gaia era probablemente casual, WISE dice más)
            r.update(tipo=m_wi["familia"], etiqueta=m_wi["pista"], fuente_id=m_wi["nombre"],
                     catalogo="WISE", sep_arcsec=m_wi["sep_arcsec"], p_azar=m_wi["p_azar"])
        elif m_ga is not None:  # cruce con Gaia dudoso, pero es lo único que hay
            r.update(tipo="estrella" if m_ga["estrella"] else "gaia",
                     etiqueta="estrella" if m_ga["estrella"] else "fuente Gaia",
                     fuente_id=_id_gaia(m_ga), catalogo="Gaia",
                     sep_arcsec=m_ga["sep_arcsec"], p_azar=m_ga["p_azar"])
        else:
            r.update(tipo="sin_identificar", etiqueta="sin identificar", catalogo=None)
        r["cruce_dudoso"] = bool(r.get("p_azar") is not None and r["p_azar"] > p_max)
        # WISE avisa de contaminación por artefacto: si SOLO WISE la ve, es dudosa
        if r["catalogo"] == "WISE" and m_wi["contaminada"]:
            r["cruce_dudoso"] = True
        resultado.append(r)

    # Objetos de ALeRCE sin fuente asociada: aparte, y solo donde hay datos
    no_det = []
    for k, o in enumerate(al):
        if k in usados:
            continue
        x, y = wcs.world_to_pixel(SkyCoord(o["ra"], o["dec"], unit="deg"))
        x, y = float(x), float(y)
        if cobertura is not None:
            yi, xi = int(round(y)), int(round(x))
            if not (0 <= yi < cobertura.shape[0] and 0 <= xi < cobertura.shape[1]
                    and cobertura[yi, xi]):
                continue
        no_det.append({"x": x, "y": y, "ra": o["ra"], "dec": o["dec"],
                       "tipo": o["familia"], "etiqueta": o.get("nombre") or o["clase"],
                       "clase": o["clase"], "fuente_id": o["oid"], "link": o.get("link", ""),
                       "temprana": o.get("temprana", False),
                       "probabilidad": o.get("probabilidad"), "no_detectado": True})
    return resultado, no_det


def descartar_alas(ident, radio_px=8, factor=5.0):
    """
    Una fuente SIN IDENTIFICAR (o identificada solo por un cruce dudoso con Gaia
    o WISE) a menos de `radio_px` (≈ 4 píxeles de SPHEREx) de
    otra al menos `factor` veces más brillante (en S/R) es casi siempre parte de
    esa estrella: su ala, un halo o un pico de difracción que el detector separó.
    Se quita de la lista (se devuelve aparte) y se renumera el resto.
    """
    quedan, fuera = [], []
    for f in ident:
        dudosa = (f["tipo"] == "sin_identificar"
                  or (f.get("cruce_dudoso") and f.get("catalogo") in ("Gaia", "WISE")))
        if dudosa and f.get("snr") and not f.get("es_objetivo"):
            vecina = [g for g in ident if g is not f and g.get("snr")
                      and g["snr"] >= factor * f["snr"]
                      and np.hypot(g["x"] - f["x"], g["y"] - f["y"]) <= radio_px]
            if vecina:
                v = max(vecina, key=lambda g: g["snr"])
                fuera.append({**f, "confirmada": False,
                              "motivo": f"ala/halo de la fuente #{v['num']} "
                                        f"({v['snr'] / f['snr']:.0f} veces más brillante)"})
                continue
        quedan.append(f)
    for i, f in enumerate(quedan, 1):
        f["num"] = i
    return quedan, fuera


# ------------------------------------------------- variabilidad con SPHEREx
def _flujo_apertura(img, x, y, r_ap, r_in, r_out):
    ny, nx = img.shape
    x0, x1 = int(max(0, x - r_out - 1)), int(min(nx, x + r_out + 2))
    y0, y1 = int(max(0, y - r_out - 1)), int(min(ny, y + r_out + 2))
    sub = img[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    ap, an = r <= r_ap, (r >= r_in) & (r <= r_out)
    if ap.sum() == 0 or not np.isfinite(sub[ap]).all() or np.isfinite(sub[an]).sum() < 5:
        return np.nan
    return float(np.sum(sub[ap] - np.nanmedian(sub[an])))


def variabilidad(ident, cubo, lams, escala=1, umbral=3.0, min_cuadros=6, snr_min=15):
    """
    ¿La fuente cambia de brillo en los propios datos de SPHEREx?

    Problema: cada cuadro tiene una λ distinta, y el brillo de cualquier
    estrella cambia con λ. Solución:
      1. Se divide el flujo de la fuente por el de estrellas de comparación
         (estrellas Gaia brillantes, no clasificadas como variables) en el MISMO
         cuadro → se cancela la mayor parte del efecto de la λ y de la
         calibración de cada cuadro.
      2. A ese cociente se le ajusta una curva suave en λ (parábola, por la
         diferencia de color entre la fuente y las comparación).
      3. Se mide cuánto se dispersan los puntos alrededor de la curva y se
         compara con la dispersión típica de las estrellas de comparación.
    índice = dispersión de la fuente / dispersión ESPERADA para su brillo.
    > umbral → "varía". Fuentes con S/R < snr_min: "débil para medir".
    """
    lams = np.asarray(lams, float)
    n = len(cubo)
    if n < min_cuadros or not np.isfinite(lams).sum() >= min_cuadros:
        return ident
    r_ap, r_in, r_out = 1.5 * escala, 3.0 * escala, 5.0 * escala
    flujos = {f["num"]: np.array([_flujo_apertura(c, f["x"], f["y"], r_ap, r_in, r_out)
                                  for c in cubo]) for f in ident}
    comp = [f["num"] for f in ident
            if f["tipo"] in ("estrella", "gaia") and (f.get("snr") or 0) > 15
            and not f.get("posible_fantasma")]
    if len(comp) < 3:
        for f in ident:
            f["variabilidad"] = {"estado": "sin estrellas de comparación"}
        return ident
    ref = np.nansum([flujos[k] for k in comp], axis=0)

    def dispersion(fl, excluir=None):
        r = ref - (fl if excluir else 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = fl / r
        ok = np.isfinite(q) & np.isfinite(lams) & (q > 0)
        if ok.sum() < min_cuadros:
            return np.nan, int(ok.sum())
        y = np.log(q[ok])
        coef = np.polyfit(lams[ok], y, 2)
        res = y - np.polyval(coef, lams[ok])
        return float(1.4826 * np.median(np.abs(res - np.median(res)))), int(ok.sum())

    # Ruido esperado según el brillo: σ² = σ_sis² + (k / SR)², ajustado con las
    # estrellas de comparación (una débil se dispersa más solo por ruido).
    snr_de = {f["num"]: f.get("snr") or 0 for f in ident}
    d_c, s_c = [], []
    for k in comp:
        d = dispersion(flujos[k], excluir=True)[0]
        if np.isfinite(d):
            d_c.append(d)
            s_c.append(snr_de[k])
    if len(d_c) < 3:
        return ident
    d_c, s_c = np.array(d_c), np.array(s_c)
    A = np.vstack([np.ones_like(s_c), 1 / s_c ** 2]).T
    try:
        from scipy.optimize import nnls
        (sis2, k2), _ = nnls(A, d_c ** 2)
    except Exception:
        sis2, k2 = np.median(d_c) ** 2, 0.0
    # piso: nunca esperar menos que la mejor estrella de comparación
    piso = float(np.min(d_c))

    def esperada(snr):
        return max(np.sqrt(sis2 + k2 / max(snr, 1) ** 2), piso)

    for f in ident:
        if (f.get("snr") or 0) < snr_min:
            f["variabilidad"] = {"estado": "débil para medir"}
            continue
        d, nok = dispersion(flujos[f["num"]], excluir=f["num"] in comp)
        if not np.isfinite(d):
            f["variabilidad"] = {"estado": "no medible", "n": nok}
            continue
        idx = d / esperada(f["snr"])
        f["variabilidad"] = {"indice": round(idx, 1), "n": nok,
                             "dispersion_pct": round(100 * d, 1),
                             "esperada_pct": round(100 * esperada(f["snr"]), 1),
                             "estado": "varía" if idx > umbral else "estable"}
    return ident


# ------------------------------------------------------------ tabla
def _detalle(r):
    det = []
    if r.get("catalogo"):
        det.append(f"[{r['catalogo']}]")
    sbd = r.get("simbad")
    if sbd and r.get("catalogo") != "SIMBAD":
        det.append(f"SIMBAD: {sbd['nombre']} ({sbd['otype']})")
    w = r.get("wise")
    if w and w.get("W1") is not None:
        txt = f"W1={w['W1']:.1f}"
        if w.get("W1_W2") is not None:
            txt += f" W1−W2={w['W1_W2']:+.2f}"
        if r.get("catalogo") != "WISE" and w["familia"] in ("agn", "especial"):
            txt += f" ({w['pista']})"
        det.append(txt)
    if r.get("temprana"):
        det.append("clasif. temprana")
    if r.get("probabilidad") is not None:
        det.append(f"prob={r['probabilidad']:.2f}")
    if r.get("activo"):
        det.append("activo en fechas SPHEREx")
    if r.get("G") is not None:
        det.append(f"G={r['G']:.1f}")
    if r.get("distancia_pc"):
        det.append(f"d≈{r['distancia_pc']:.0f} pc")
    if r.get("p_azar") is not None:
        det.append(f"P_azar={r['p_azar']:.0%}")
    return det


def imprimir(ident, no_det=None):
    if not ident:
        print("  No se detectaron fuentes.")
        return
    print(f"\n  {'#':>3}  {'qué es':<24} {'id':<22} {'S/R':>5} {'visto':>6} {'¿varía?':>12}  detalle")
    for r in ident:
        det = _detalle(r)
        if r.get("cruce_dudoso"):
            det.insert(0, "⚠ cruce dudoso")
        if r.get("posible_fantasma"):
            det.insert(0, "⚠ posible fantasma/crosstalk")
        if r.get("link"):
            det.append(r["link"])
        v = r.get("variabilidad", {})
        var = (f"{v['estado']} x{v['indice']}" if "indice" in v else v.get("estado", "-"))
        snr = f"{r['snr']:.0f}" if r.get("snr") else "-"
        visto = (f"{r['n_cuadros_visto']}/{r['n_cuadros_cubre']}"
                 if r.get("n_cuadros_cubre") else "-")
        print(f"  {r['num']:>3}  {r['etiqueta'][:24]:<24} {str(r.get('fuente_id', ''))[:22]:<22} "
              f"{snr:>5} {visto:>6} {var[:12]:>12}  {'  '.join(det)}")
    tipos = {}
    for r in ident:
        tipos[r["etiqueta"]] = tipos.get(r["etiqueta"], 0) + 1
    print("  resumen:", ", ".join(f"{k}: {v}" for k, v in tipos.items()))
    var = [r["num"] for r in ident if r.get("variabilidad", {}).get("estado") == "varía"]
    if var:
        print("  varían en los datos SPHEREx: #" + ", #".join(map(str, var)))
    if no_det:
        print(f"  ALeRCE además tiene {len(no_det)} objetos en el campo que SPHEREx no "
              "detecta (en el JSON, 'no_detectados')")


# ------------------------------------------------------------ CSV
COLUMNAS_CSV = [
    ("num", "#"), ("estado", "estado"), ("etiqueta", "qué es"), ("catalogo", "según"),
    ("fuente_id", "id"), ("ra", "AR (°)"), ("dec", "Dec (°)"), ("snr", "S/R"),
    ("visto", "visto en cuadros"), ("var_estado", "¿varía en SPHEREx?"),
    ("var_indice", "índice variabilidad"), ("probabilidad", "prob. ALeRCE"),
    ("temprana", "clasif. temprana"), ("G", "Gaia G"), ("distancia_pc", "distancia (pc)"),
    ("sb_nombre", "SIMBAD nombre"), ("sb_tipo", "SIMBAD tipo"), ("W1", "WISE W1"),
    ("W1_W2", "WISE W1−W2"), ("w_pista", "pista WISE"), ("sep_arcsec", "separación (\")"),
    ("p_azar", "P azar"), ("avisos", "avisos"), ("link", "link"),
]


def _fila_csv(f, estado):
    v = f.get("variabilidad") or {}
    sb = f.get("simbad") or {}
    w = f.get("wise") or {}
    avisos = []
    if f.get("posible_fantasma"):
        avisos.append("posible fantasma/crosstalk")
    if f.get("cruce_dudoso"):
        avisos.append("cruce dudoso")
    if f.get("motivo") and estado == "descartada":
        avisos.append(f["motivo"])
    return {
        "num": f.get("num", ""), "estado": estado, "etiqueta": f.get("etiqueta", ""),
        "catalogo": f.get("catalogo") or "", "fuente_id": f.get("fuente_id", ""),
        "ra": f.get("ra"), "dec": f.get("dec"), "snr": f.get("snr"),
        "visto": (f"{f['n_cuadros_visto']}/{f['n_cuadros_cubre']}"
                  if f.get("n_cuadros_cubre") else ""),
        "var_estado": v.get("estado", ""), "var_indice": v.get("indice"),
        "probabilidad": f.get("probabilidad"), "temprana": "sí" if f.get("temprana") else "",
        "G": f.get("G"), "distancia_pc": f.get("distancia_pc"),
        "sb_nombre": sb.get("nombre", ""), "sb_tipo": sb.get("tipo_es", ""),
        "W1": w.get("W1"), "W1_W2": w.get("W1_W2"), "w_pista": w.get("pista", ""),
        "sep_arcsec": f.get("sep_arcsec"), "p_azar": f.get("p_azar"),
        "avisos": "; ".join(avisos), "link": f.get("link", ""),
    }


def guardar_csv(ruta, ident, descartadas=None, no_det=None, excel_es=True):
    """
    Tabla de identificación en CSV. Por defecto en formato de Excel en español
    (separador ';' y coma decimal), con BOM para que se vean bien las tildes.
    Incluye las fuentes descartadas y los objetos de ALeRCE no detectados,
    con una columna 'estado' para filtrarlas.
    """
    import csv
    sep, dec = (";", ",") if excel_es else (",", ".")

    def fmt(clave, x):
        if x is None or x == "":
            return ""
        if isinstance(x, float):
            if not np.isfinite(x):
                return ""
            nd = 6 if clave in ("ra", "dec") else (3 if clave == "p_azar" else 2)
            if clave in ("snr", "distancia_pc"):
                nd = 0
            return f"{x:.{nd}f}".replace(".", dec)
        return str(x)

    filas = [_fila_csv(f, "identificada") for f in ident]
    filas += [_fila_csv(f, "descartada") for f in (descartadas or [])]
    filas += [_fila_csv(f, "ALeRCE no detectado") for f in (no_det or [])]
    with open(ruta, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh, delimiter=sep)
        w.writerow([titulo for _, titulo in COLUMNAS_CSV])
        for fila in filas:
            w.writerow([fmt(c, fila.get(c)) for c, _ in COLUMNAS_CSV])
    return ruta


# ------------------------------------------------------------ dibujo
def fuente_tt(tam=15):
    """Fuente con tildes, ñ, λ y µ (DejaVu, que viene con matplotlib)."""
    from PIL import ImageFont
    try:
        from matplotlib import font_manager
        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), tam)
    except Exception:
        try:
            return ImageFont.load_default(size=tam)
        except TypeError:
            return ImageFont.load_default()


def _relevante(f):
    return f["tipo"] not in ("estrella", "gaia")


def dibujar(d, ident, n_pix, zoom, fuente, no_det=None, alto_cabecera=24):
    """Círculos numerados; color según lo que es. Etiquetas que no se salen."""
    ancho = alto = n_pix * zoom
    r = max(7, 1.8 * zoom)
    for f in (no_det or []):  # opcionales: pequeños y finos
        X, Y = (f["x"] + 0.5) * zoom, (n_pix - 1 - f["y"] + 0.5) * zoom
        color = COLORES.get(f["tipo"], COLORES["otro"])
        for a in range(0, 360, 45):  # círculo punteado
            d.arc([X - r * .7, Y - r * .7, X + r * .7, Y + r * .7], a, a + 22, fill=color)
    for f in ident:
        X, Y = (f["x"] + 0.5) * zoom, (n_pix - 1 - f["y"] + 0.5) * zoom
        color = COLORES.get(f["tipo"], COLORES["otro"])
        grosor = 3 if (f.get("activo") or f.get("variabilidad", {}).get("estado") == "varía") else 2
        if not _relevante(f):
            grosor = 1
        d.ellipse([X - r, Y - r, X + r, Y + r], outline=color, width=grosor)
        txt = str(f["num"])
        tx, ty = X + r * 0.75, Y - r - 14
        if ty < alto_cabecera + 2:          # no tapar la barra de la fecha
            ty = Y + r - 2
        if tx + 8 * len(txt) > ancho - 2:   # no salirse por la derecha
            tx = X - r * 0.75 - 8 * len(txt)
        d.text((tx, ty), txt, fill=color, font=fuente)


def panel_leyenda(ident, alto, ancho=330, fuente=None, n_no_det=0):
    """Panel lateral agrupado: ALeRCE, sin identificar, estrellas."""
    from PIL import Image, ImageDraw
    fuente = fuente or fuente_tt(13)
    titulo = fuente_tt(14)
    img = Image.new("RGB", (ancho, alto), (12, 14, 20))
    d = ImageDraw.Draw(img)
    y, paso, limite = 8, 17, alto - 22

    def linea(txt, color, f=fuente):
        nonlocal y
        if y > limite:
            return False
        d.text((10, y), txt[:46], fill=color, font=f)
        y += paso
        return True

    comunes = ("estrella", "gaia")
    grupos = [
        ("Clasificados (ALeRCE/SIMBAD/WISE)",
         [f for f in ident if f["tipo"] not in comunes + ("sin_identificar",)]),
        ("Sin identificar", [f for f in ident if f["tipo"] == "sin_identificar"]),
        ("Estrellas (Gaia/SIMBAD/WISE)", [f for f in ident if f["tipo"] in comunes]),
    ]
    linea("Identificación", (255, 255, 255), titulo)
    y += 4
    for nombre, lista in grupos:
        if not lista:
            continue
        if not linea(f"{nombre} ({len(lista)})", (150, 155, 170)):
            break
        for i, f in enumerate(lista):
            extra = []
            if f.get("probabilidad") is not None:
                extra.append(f"{f['probabilidad']:.2f}")
            if f.get("temprana"):
                extra.append("temprana")
            if f.get("G") is not None and f["tipo"] in ("estrella", "gaia"):
                extra.append(f"G={f['G']:.1f}")
            elif f.get("catalogo") == "SIMBAD" and f.get("fuente_id"):
                extra.append(str(f["fuente_id"])[:18])
            if f.get("variabilidad", {}).get("estado") == "varía":
                extra.append("¡varía!")
            marca = " (?)" if (f.get("posible_fantasma") or f.get("cruce_dudoso")) else ""
            txt = f"{f['num']:>2} {f['etiqueta']}{marca}" + (f"  {', '.join(extra)}" if extra else "")
            if y > limite - paso and i < len(lista) - 1:
                linea(f"   … y {len(lista) - i} más", (150, 150, 150))
                break
            if not linea(txt, COLORES.get(f["tipo"], COLORES["otro"])):
                break
        y += 5
    if n_no_det:
        d.text((10, alto - 18), f"+{n_no_det} de ALeRCE no detectados (ver JSON)",
               fill=(120, 120, 130), font=fuente)
    return np.asarray(img)
