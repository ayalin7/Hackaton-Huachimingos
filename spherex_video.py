"""
spherex_video.py — De coordenadas (RA, Dec) a un video de un parche de cielo SPHEREx.

Pipeline:
  1. Buscar en IRSA (SIA2) todas las Spectral Images que cubren (RA, Dec)
  2. Descargar solo un recorte (cutout) de cada una
  3. Limpiar: restar zodiacal, enmascarar FLAGS
  4. Reproyectar todos los recortes a una misma grilla (norte arriba)
  5. Ordenar por tiempo, escalar con límites comunes y armar un GIF
     (modo "directo" o "resta" = cada cuadro menos la mediana → resalta lo que se mueve)

Instalación:
  pip install astroquery astropy reproject imageio numpy

Uso:
  python spherex_video.py 337.8783 -5.1892 --fov 0.1 --detector 1 --modo resta
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.time import Time
from astropy.table import Table
from astroquery.ipac.irsa import Irsa
from reproject import reproject_interp
from astropy.visualization import ZScaleInterval
import imageio.v3 as iio

warnings.simplefilter("ignore", FITSFixedWarning)
warnings.filterwarnings("ignore", message="All-NaN slice")

# Dar más tiempo a cada descarga (por defecto son 10 s y IRSA a veces tarda más)
from astropy.utils.data import conf as _data_conf
_data_conf.remote_timeout = 60

# Silenciar el aviso largo de SIP (es solo informativo)
import logging
from astropy import log as _astropy_log
_astropy_log.setLevel(logging.WARNING)

PIX_SCALE_ARCSEC = 6.15

# Bits de la extensión FLAGS (Tabla 16 del SPHEREx Explanatory Supplement)
FLAGS_BITS = {
    "TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "NONFUNC": 6, "DICHROIC": 7,
    "MISSING_DATA": 9, "HOT": 10, "COLD": 11, "FULLSAMPLE": 12, "PHANMISS": 14,
    "NONLINEAR": 15, "PERSIST": 17, "OUTLIER": 19, "CROSSTALK": 20, "SOURCE": 21,
    "GHOST": 22, "GHOST_FPA": 23, "GHOST_EXT": 24, "STREAK": 25, "BLOOM": 26,
    "SNOWBALL": 27, "HALO": 28, "SATELLITE_HALO": 29,
}


def _bits(*nombres):
    return sum(1 << FLAGS_BITS[n] for n in nombres)


# Píxeles que se BORRAN (NaN): la lista que el Suplemento recomienda excluir de
# la fotometría, más OUTLIER y STREAK (trazos de satélites) para no detectarlos
# como fuentes.
MASCARA_RECOMENDADA = _bits("SUR_ERROR", "NONFUNC", "MISSING_DATA", "HOT", "COLD",
                            "NONLINEAR", "PERSIST", "OUTLIER", "STREAK", "BLOOM",
                            "SNOWBALL", "HALO", "SATELLITE_HALO")
# Píxeles que NO se borran pero hacen dudar de una fuente: fantasmas ópticos y
# crosstalk pueden parecer estrellas que no existen.
BITS_AVISO = _bits("GHOST", "GHOST_FPA", "GHOST_EXT", "CROSSTALK")

# Modo en memoria (para el servidor web): no se escribe NADA en disco.
# Las búsquedas y recortes se guardan solo en RAM, con un límite de tamaño.
EN_MEMORIA = False
_RAM_RECORTES, _RAM_BUSQUEDAS = None, None


class _LRU:
    """Diccionario con tamaño máximo: al llenarse, olvida lo más antiguo."""
    def __init__(self, maximo):
        from collections import OrderedDict
        import threading
        self.d, self.max, self.lock = OrderedDict(), maximo, threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.max:
                self.d.popitem(last=False)


def usar_memoria(activar=True, max_recortes=3000, max_busquedas=200):
    """Activa el modo en memoria en este módulo y en las capas."""
    global EN_MEMORIA, _RAM_RECORTES, _RAM_BUSQUEDAS
    EN_MEMORIA = activar
    _RAM_RECORTES = _LRU(max_recortes)
    _RAM_BUSQUEDAS = _LRU(max_busquedas)
    import alerce_capa, asteroides_capa, objetivos, catalogos
    for m in (alerce_capa, asteroides_capa, objetivos, catalogos):
        m.EN_MEMORIA = activar  # tamaño de píxel aproximado de SPHEREx


# Columnas de la búsqueda que usamos (ver explicación de cada una en el chat)
COLS_UTILES = (
    "obs_id",              # nombre único de la observación
    "access_url",          # link al FITS (para pedir el recorte)
    "access_estsize",      # tamaño del archivo completo en kB
    "t_min", "t_max",      # inicio y fin de la exposición (MJD)
    "t_exptime",           # tiempo de exposición (s)
    "energy_bandpassname", # detector: SPHEREx-D1 ... D6
    "em_min", "em_max",    # rango de λ de la imagen completa (metros)
    "s_ra", "s_dec",       # centro de la imagen completa
    "s_region",            # contorno de la imagen en el cielo
    "s_pixel_scale",       # tamaño del píxel
    "obs_release_date",    # fecha de publicación en IRSA
    "dataproduct_type",    # image / cube / ...
)


def valor(fila, col, defecto=None):
    """Lee una columna de una fila sin romperse si no existe o viene vacía."""
    try:
        v = fila[col]
    except (KeyError, IndexError):
        return defecto
    if np.ma.is_masked(v):
        return defecto
    if isinstance(v, bytes):
        v = v.decode()
    return v.item() if hasattr(v, "item") else v


def escala_pixel_arcsec(tabla):
    """Tamaño del píxel en arcsec, leído de la tabla si se puede."""
    if "s_pixel_scale" in tabla.colnames and len(tabla):
        v = valor(tabla[0], "s_pixel_scale")
        if v:
            v = float(v)
            return v * 3600 if v < 0.1 else v  # puede venir en grados o en arcsec
    return PIX_SCALE_ARCSEC


def resumen(tabla):
    """Imprime cuántas imágenes hay por detector, su rango de λ y de fechas."""
    if "energy_bandpassname" not in tabla.colnames:
        return
    bandas = np.asarray(tabla["energy_bandpassname"], dtype=str)
    print("  Detector   N   λ (µm)          fechas")
    for b in sorted(set(bandas)):
        sub = tabla[bandas == b]
        rango_l = ""
        if "em_min" in sub.colnames:
            rango_l = f"{np.nanmin(sub['em_min']) * 1e6:.2f}-{np.nanmax(sub['em_max']) * 1e6:.2f}"
        t0 = Time(float(np.min(sub["t_min"])), format="mjd").iso[:10]
        t1 = Time(float(np.max(sub["t_min"])), format="mjd").iso[:10]
        print(f"  {b:<12} {len(sub):>2}   {rango_l:<14}  {t0} → {t1}")


# ---------------------------------------------------------------- 1. Búsqueda
def buscar_imagenes(ra, dec, coleccion="spherex_qr3", detector=None):
    """
    Devuelve una tabla con las Spectral Images que contienen (ra, dec).
    `coleccion` puede ser una lista separada por comas, p. ej.
    "spherex_dr1,spherex_qr3": se juntan (más fechas = más tiempo para ver
    movimientos). Si una misma exposición está en dos colecciones, se queda la
    de la colección que aparece primero.
    """
    from astropy.table import vstack
    nombres = [c.strip() for c in str(coleccion).split(",") if c.strip()]
    tablas = []
    for col in nombres:
        try:
            t = _buscar_una(ra, dec, col)
        except SystemExit as e:
            if len(nombres) == 1:
                raise
            print(f"  {col}: {e}")
            continue
        if len(t):
            t["coleccion"] = col
            tablas.append(t)
    if not tablas:
        raise SystemExit("No encontré imágenes en ninguna colección.")
    tabla = vstack(tablas, metadata_conflicts="silent") if len(tablas) > 1 else tablas[0]

    # Quitar exposiciones repetidas entre colecciones (mismo instante y detector)
    if len(tablas) > 1 and "energy_bandpassname" in tabla.colnames:
        clave = [f"{float(t):.4f}|{b}" for t, b in
                 zip(tabla["t_min"], np.asarray(tabla["energy_bandpassname"], dtype=str))]
        _, idx = np.unique(clave, return_index=True)
        tabla = tabla[np.sort(idx)]

    tabla.sort("t_min")  # t_min = inicio de la exposición en MJD
    return filtrar_detector(tabla, detector)


def _buscar_una(ra, dec, coleccion):
    coord = SkyCoord(ra, dec, unit="deg")
    if EN_MEMORIA:
        clave = f"{coleccion}|{ra:.4f}|{dec:.4f}"
        t = _RAM_BUSQUEDAS.get(clave)
        if t is None:
            t = _consultar_sia(coord, coleccion)
            cols = [c for c in COLS_UTILES if c in t.colnames]
            t = t[cols]
            _RAM_BUSQUEDAS.put(clave, t)
        return _limpiar_tabla(t.copy())

    # Guardamos cada búsqueda en disco: si IRSA está caído, usamos la última copia
    os.makedirs("cache_busquedas", exist_ok=True)
    archivo = os.path.join("cache_busquedas", f"{coleccion}_{ra:.4f}_{dec:+.4f}.ecsv")
    tabla = None
    for i in range(3):
        try:
            tabla = Irsa.query_sia(pos=(coord, 5 * u.arcsec), collection=coleccion)
        except Exception as e:
            print(f"  búsqueda falló ({type(e).__name__}), intento {i + 1}/3")
            time.sleep(5)
            continue
        try:  # guardar solo las columnas útiles (evita tipos raros)
            cols = [c for c in COLS_UTILES if c in tabla.colnames]
            tabla = tabla[cols]
            tabla.write(archivo, overwrite=True)
        except Exception as e:
            print("  (no pude guardar la búsqueda:", e, ")")
        break
    if tabla is None:
        if os.path.exists(archivo):  # noqa
            print("  IRSA no responde: uso la búsqueda guardada", archivo)
            tabla = Table.read(archivo)
        else:
            raise SystemExit("IRSA no responde y no hay búsqueda guardada para "
                             "estas coordenadas. Intenta de nuevo en unos minutos.")
    return _limpiar_tabla(tabla)


def _limpiar_tabla(tabla):
    # Revisa qué columnas trae (cambian entre releases):  print(tabla.colnames)

    # Quedarnos solo con imágenes (por si la colección trae otros productos)
    if "dataproduct_type" in tabla.colnames:
        es_img = np.asarray(tabla["dataproduct_type"], dtype=str) == "image"
        if es_img.any():
            tabla = tabla[es_img]

    # Una fila por archivo
    _, idx = np.unique(np.asarray(tabla["access_url"]), return_index=True)
    tabla = tabla[np.sort(idx)]

    return tabla


def _consultar_sia(coord, coleccion):
    for i in range(3):
        try:
            return Irsa.query_sia(pos=(coord, 5 * u.arcsec), collection=coleccion)
        except Exception as e:
            print(f"  búsqueda falló ({type(e).__name__}), intento {i + 1}/3")
            time.sleep(3)
    raise SystemExit("IRSA no responde. Intenta de nuevo en unos minutos.")


def filtrar_detector(tabla, detector):
    """Deja solo las imágenes del detector pedido (1..6). None = todos."""
    if detector is None or "energy_bandpassname" not in tabla.colnames:
        return tabla
    banda = np.char.find(np.asarray(tabla["energy_bandpassname"], dtype=str),
                         f"D{detector}") >= 0
    return tabla[banda]


# ----------------------------------------------------------- 2. y 3. Cutouts
def abrir_con_reintentos(url, intentos=5):
    """
    Descarga el recorte a un archivo TEMPORAL (no al caché de astropy, que crece
    sin límite) y lo abre en memoria. IRSA a veces responde 503: reintentamos.
    """
    from astropy.utils.data import download_file
    import io
    for i in range(intentos):
        try:
            if EN_MEMORIA:  # directo a RAM, sin archivo temporal
                import urllib.request
                with urllib.request.urlopen(url, timeout=60) as r:
                    contenido = r.read()
                return fits.open(io.BytesIO(contenido))
            tmp = download_file(url, cache=False, show_progress=False)
            try:
                with open(tmp, "rb") as fh:
                    contenido = fh.read()
            finally:
                os.remove(tmp)
            return fits.open(io.BytesIO(contenido))
        except Exception as e:
            if "No space left" in str(e) or getattr(e, "errno", None) == 28:
                raise SystemExit("\nDisco lleno. Libera espacio o usa un --fov más chico.")
            temporal = "503" in str(e) or "timed out" in str(e).lower()
            if temporal and i < intentos - 1:
                espera = 3 * (i + 1)
                print(f"  servidor ocupado, reintento en {espera} s...")
                time.sleep(espera)
            else:
                raise


def arreglar_sip(header):
    """Los headers traen coeficientes SIP pero sin el sufijo '-SIP' en CTYPE."""
    if "A_ORDER" in header:
        for k in ("CTYPE1", "CTYPE2"):
            if k in header and not header[k].endswith("-SIP"):
                header[k] = header[k] + "-SIP"
    return header


CACHE_RECORTES = "cache_recortes"


def descargar_cutout(access_url, ra, dec, fov_deg, mascara_bits=0, devolver_flags=False):
    """
    Devuelve (imagen_limpia, wcs, longitud_de_onda_um).
    Guarda en cache_recortes/ SOLO lo que usamos (imagen - zodi, flags, header, λ)
    en formato comprimido: ~10 veces menos espacio que el FITS completo.
    """
    import hashlib
    if EN_MEMORIA:
        clave = f"{access_url}|{ra:.5f}|{dec:.5f}|{fov_deg:.4f}"
        guardado = _RAM_RECORTES.get(clave)
        if guardado is None:
            guardado = _bajar_recorte(access_url, ra, dec, fov_deg)
            img_, flags_, header_, lam_ = guardado
            guardado = (img_.astype(np.float32), flags_, header_, lam_)
            _RAM_RECORTES.put(clave, guardado)
        img, flags, header, lam = guardado
        img = img.astype(float)
        if mascara_bits:
            img[(flags & mascara_bits) != 0] = np.nan
        if devolver_flags:
            return img, WCS(header), lam, flags
        return img, WCS(header), lam
    os.makedirs(CACHE_RECORTES, exist_ok=True)
    clave = hashlib.md5(f"{access_url}|{ra:.5f}|{dec:.5f}|{fov_deg:.4f}".encode()).hexdigest()
    archivo = os.path.join(CACHE_RECORTES, clave + ".npz")

    if os.path.exists(archivo):
        d = np.load(archivo, allow_pickle=False)
        img = d["img"].astype(float)
        header = fits.Header.fromstring(str(d["header"]))
        flags, lam = d["flags"], float(d["lam"])
    else:
        img, flags, header, lam = _bajar_recorte(access_url, ra, dec, fov_deg)
        time.sleep(0.2)  # no saturar el servidor de IRSA
        np.savez_compressed(archivo, img=img.astype(np.float32), flags=flags,
                            header=header.tostring(), lam=lam)

    if mascara_bits:
        img = img.copy()
        img[(flags & mascara_bits) != 0] = np.nan
    if devolver_flags:
        return img, WCS(header), lam, flags
    return img, WCS(header), lam


def _bajar_recorte(access_url, ra, dec, fov_deg):
    url = f"{access_url}?center={ra},{dec}&size={fov_deg}"
    with abrir_con_reintentos(url) as hdul:
        img = hdul["IMAGE"].data.astype(float)        # MJy/sr
        zodi = hdul["ZODI"].data.astype(float)        # fondo zodiacal modelado
        flags = hdul["FLAGS"].data.astype(np.int64)   # máscara de bits
        header = arreglar_sip(hdul["IMAGE"].header.copy())
        wcs = WCS(header)

        img = img - zodi
        # (Las máscaras de FLAGS se aplican en descargar_cutout: muchos bits son
        #  solo informativos; revisa la tabla del Explanatory Supplement.)

        # Longitud de onda en el centro del recorte (cada píxel ve una λ distinta)
        lam = np.nan
        try:
            wcs_w = WCS(hdul["IMAGE"].header, fobj=hdul, key="W")
            ny, nx = img.shape
            lam, _bw = wcs_w.pixel_to_world(nx / 2, ny / 2)
            lam = float(getattr(lam, "value", lam))
        except Exception:
            pass

    return img, flags.astype(np.int32), header, lam


# ----------------------------------------------------------- 4. Reproyección
def grilla_comun(ra, dec, fov_deg, pix_arcsec=PIX_SCALE_ARCSEC):
    """WCS tangente centrada en (ra, dec), norte arriba, este a la izquierda."""
    n = int(np.ceil(fov_deg * 3600 / pix_arcsec))
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [n / 2 + 0.5, n / 2 + 0.5]
    s = pix_arcsec / 3600
    w.wcs.cdelt = [-s, s]
    return w, (n, n)


# --------------------------------------------------------------- 5. Video
def a_8bits(cubo, vmin, vmax):
    x = np.clip((cubo - vmin) / (vmax - vmin), 0, 1)
    x = np.nan_to_num(x, nan=0.5)  # píxeles sin dato en gris, no negro
    return (x * 255).astype(np.uint8)


def reproyectar(img, wcs_in, wcs_out, forma, sobremuestreo=1):
    """
    Lleva un recorte a la grilla común.
    - sobremuestreo = 1: interpolación bilineal (rápida, píxel nativo).
    - sobremuestreo > 1: interpolación bicúbica en una grilla más fina.
    Los NaN se rellenan antes de interpolar (la bicúbica los "contagia") y se
    vuelven a marcar después.
    """
    malos = ~np.isfinite(img)
    relleno = np.where(malos, np.nanmedian(img), img)
    orden = "bicubic" if sobremuestreo > 1 else "bilinear"
    out, cobertura = reproject_interp((relleno, wcs_in), wcs_out, shape_out=forma,
                                      order=orden)
    if malos.any():
        m, _ = reproject_interp((malos.astype(float), wcs_in), wcs_out,
                                shape_out=forma, order="bilinear")
        out[m > 0.5] = np.nan
    out[cobertura == 0] = np.nan
    return out


def apilar(cubo, sigma=3.0):
    """
    Combina todos los cuadros en una imagen "profunda":
      1. resta el fondo de cada cuadro,
      2. iguala su brillo con el de un cuadro de referencia,
      3. promedia con recorte sigma (descarta lo que no se repite:
         rayos cósmicos, objetos en movimiento, píxeles malos).
    El ruido baja aproximadamente como 1/sqrt(N).
    """
    from astropy.stats import sigma_clip
    cubo = np.array([quitar_fondo(c) for c in cubo])
    ref = np.nanmedian(cubo, axis=0)
    norm = np.array([c / ajustar_escala(c, ref) for c in cubo])
    recortado = sigma_clip(np.ma.masked_invalid(norm), sigma=sigma, axis=0, masked=True)
    return np.ma.filled(recortado.mean(axis=0).astype(float), np.nan)


def hacer_video(ra, dec, fov_deg=0.1, detector=None, coleccion="spherex_qr3",
                modo="directo", salida="spherex.gif", max_cuadros=200, fps=3,
                mascara_bits=0, sep_min_h=1.0, zoom=None, solo_listar=False,
                sobremuestreo=1, suave=False, hacer_apilado=False, usar_alerce=False,
                usar_asteroides=False, prob_min=0.5, v_max=21.0,
                objetivo=None, hacer_fotometria=False, hilos=4, descartar_malos=True,
                tam=720, n_fuentes=25, prob_temprana=0.7,
                mostrar_no_detectados=False, usar_simbad=True, usar_wise=True,
                csv_excel_es=True):
    todas = buscar_imagenes(ra, dec, coleccion)
    print(f"{len(todas)} imágenes encontradas en total:")
    resumen(todas)
    tabla = filtrar_detector(todas, detector)
    if detector is not None:
        print(f"Usando {len(tabla)} imágenes del detector D{detector}")
    if solo_listar:
        return None, []
    if len(tabla) > max_cuadros:
        print(f"Uso los primeros {max_cuadros} de {len(tabla)} (cambia con --max)")
    tabla = tabla[:max_cuadros]
    if len(tabla) == 0:
        raise SystemExit("No hay imágenes para ese detector.")

    # Si hay un objetivo con movimiento propio, centramos donde está en la
    # fecha mediana de las imágenes (no en su posición del año 2000).
    if objetivo is not None:
        import objetivos
        mjd_med = float(np.median(np.asarray(tabla["t_min"], dtype=float)))
        objetivos.describir(objetivo, mjd_med)
        c = objetivos.posicion_en(objetivo, mjd_med)
        ra, dec = float(c.ra.deg), float(c.dec.deg)

    pix = escala_pixel_arcsec(tabla)
    n_lado = fov_deg * 3600 / pix * max(1, int(sobremuestreo))
    ram_mb = len(tabla) * n_lado ** 2 * 8 / 1e6
    if ram_mb > 1500:
        raise SystemExit(f"Con --fov {fov_deg} y --sobremuestreo {sobremuestreo} el cubo "
                         f"ocuparía ~{ram_mb:.0f} MB de RAM. Baja el FOV o el sobremuestreo.")
    import shutil
    libre_gb = shutil.disk_usage(".").free / 1e9
    if libre_gb < 1:
        print(f"⚠ Atención: solo quedan {libre_gb:.1f} GB libres en el disco.")
    sobremuestreo = max(1, int(sobremuestreo))
    wcs_out, forma = grilla_comun(ra, dec, fov_deg, pix / sobremuestreo)
    if zoom is None:  # agrandar hasta ~tam píxeles de lado
        zoom = max(1, int(round(tam / forma[0])))

    # Estimar cuánto se va a descargar (recorte ≈ fracción del archivo completo)
    if "access_estsize" in tabla.colnames:
        frac = min(1.0, (forma[0] / sobremuestreo / 2040) ** 2)
        mb = np.nansum(np.asarray(tabla["access_estsize"], dtype=float)) * frac / 1024
        print(f"Descarga estimada: ~{max(mb, 0.1 * len(tabla)):.1f} MB "
              f"({forma[0]}×{forma[1]} píxeles por cuadro)")
    cuadros, meta, avisos = [], [], []
    # 1) Descargar en paralelo (varias a la vez). Las que ya están en el caché
    #    se leen del disco al instante.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    filas = list(tabla)
    n = len(filas)
    resultados = {}
    t_ini = time.time()

    def bajar(i):
        return i, descargar_cutout(filas[i]["access_url"], ra, dec, fov_deg, mascara_bits,
                                   devolver_flags=True)

    with ThreadPoolExecutor(max_workers=max(1, hilos)) as ex:
        futuros = [ex.submit(bajar, i) for i in range(n)]
        for k, fut in enumerate(as_completed(futuros), 1):
            try:
                i, res = fut.result()
                resultados[i] = res
            except SystemExit:
                raise
            except Exception as e:
                print(f"  falló una imagen: {e}")
            if k % 10 == 0 or k == n:
                seg = time.time() - t_ini
                resta = seg / k * (n - k)
                print(f"  descargadas {k}/{n}   (faltan ~{resta / 60:.1f} min)")

    # 2) Reproyectar en orden de fecha
    for i, fila in enumerate(filas):
        if i not in resultados:
            continue
        res = resultados[i]
        img, wcs_in, lam = res[:3]
        rep = reproyectar(img, wcs_in, wcs_out, forma, sobremuestreo)
        cuadros.append(rep)
        # Mapa de "avisos" (fantasma/crosstalk) llevado a la grilla común
        if len(res) > 3 and res[3] is not None:
            av = ((np.asarray(res[3]) & BITS_AVISO) != 0).astype(float)
            av_rep, _ = reproject_interp((av, wcs_in), wcs_out, shape_out=forma,
                                         order="bilinear")
            avisos.append(np.nan_to_num(av_rep) > 0.25)
        else:
            avisos.append(np.zeros(forma, bool))
        t0 = float(fila["t_min"])
        t1 = valor(fila, "t_max")
        t_medio = (t0 + float(t1)) / 2 if t1 else t0  # tiempo medio de la exposición
        em_min, em_max = valor(fila, "em_min"), valor(fila, "em_max")
        meta.append({
            "obs_id": valor(fila, "obs_id", ""),
            "fecha_utc": Time(t_medio, format="mjd").isot,
            "mjd": t_medio,
            "t_exptime_s": valor(fila, "t_exptime"),
            "lambda_um": lam,
            "rango_lambda_imagen_um": [em_min * 1e6, em_max * 1e6] if em_min and em_max else None,
            "banda": valor(fila, "energy_bandpassname", ""),
            "publicado": str(valor(fila, "obs_release_date", "")),
            "s_region": str(valor(fila, "s_region", "")),
            "url": str(fila["access_url"]),
        })

    print(f"Cuadros descargados: {len(cuadros)} de {len(tabla)}")
    if not cuadros:
        print("No se descargó ninguna imagen. Intenta de nuevo en unos minutos.")
        return None, meta

    cubo = np.array(cuadros)
    suave = suave or sobremuestreo > 1
    png_espectro = png_apilado = None

    if objetivo is not None:
        import objetivos
        for m in meta:
            c = objetivos.posicion_en(objetivo, m["mjd"])
            x, y = wcs_out.world_to_pixel(SkyCoord(c.ra, c.dec))
            m["objetivo_xy"] = [float(x), float(y)]
            m["objetivo_radec"] = [float(c.ra.deg), float(c.dec.deg)]
        if hacer_fotometria:
            import fotometria
            flujos = fotometria.medir(cubo, [m["objetivo_xy"] for m in meta],
                                      pix / sobremuestreo, pix)
            for m, fl in zip(meta, flujos):
                m["flujo_mJy"] = None if not np.isfinite(fl) else float(fl)
            png_espectro = fotometria.guardar(
                salida.rsplit(".", 1)[0] if salida else None, meta, flujos,
                titulo=f"{objetivo['nombre']} — SPHEREx")

    marcas = []
    if usar_alerce:
        import alerce_capa
        radio = fov_deg / 2 * np.sqrt(2) * 3600  # círculo que cubre todo el cuadrado
        objetos = alerce_capa.buscar_alerce(ra, dec, radio)
        mjds = [m["mjd"] for m in meta]
        alerce_capa.marcar_actividad(objetos, min(mjds), max(mjds))
        en_campo = alerce_capa.marcas_en_grilla(objetos, wcs_out, forma)
        marcas = alerce_capa.filtrar(en_campo, prob_min=prob_min,
                                     prob_temprana=prob_temprana)
        print(f"  ALeRCE: {len(marcas)} objetos clasificados en el campo "
              f"(de {len(en_campo)} antes de filtrar)")

    resumen_ast = []
    if usar_asteroides:
        import asteroides_capa
        print("Calculando posiciones de asteroides conocidos (SkyBoT)...")
        resumen_ast = asteroides_capa.asteroides_por_cuadro(
            ra, dec, fov_deg, meta, wcs_out, forma, v_max=v_max)
        asteroides_capa.imprimir_tabla(resumen_ast)

    # Fondo 2D: quita gradientes y marca cuadros con fondo anómalo
    cubo, calidad, metrica = limpiar_fondo(cubo, caja=9 * sobremuestreo)
    for m, q, v in zip(meta, calidad, metrica):
        m["calidad"], m["gradiente_fondo"] = q, round(float(v), 2)
    malos = [i for i, q in enumerate(calidad) if q == "mala"]
    if malos:
        print(f"  {len(malos)} cuadro(s) con fondo anómalo: "
              + ", ".join(meta[i]["fecha_utc"][:16] for i in malos[:6])
              + (" ..." if len(malos) > 6 else ""))
        if descartar_malos and len(malos) < len(meta):
            buenos = [i for i in range(len(meta)) if calidad[i] == "buena"]
            cubo = cubo[buenos]
            avisos = [avisos[i] for i in buenos]
            meta_video = [meta[i] for i in buenos]
            print("  (se excluyen del video y del apilado; siguen en la fotometría y el JSON)")
        else:
            meta_video = meta
    else:
        meta_video = meta
    meta_completa = meta

    ident = None
    descartadas = None
    no_det = []
    if usar_alerce:
        import identificar
        profunda_id = apilar(cubo)
        # Detectar de más y quedarse con las que pasan la verificación
        candidatas = identificar.detectar_fuentes(profunda_id, n_max=3 * n_fuentes,
                                                  sep_px=2 * sobremuestreo + 1)
        identificar.verificar(candidatas, cubo, avisos, apilar, caja=sobremuestreo)
        fuentes = [f for f in candidatas if f["confirmada"]][:n_fuentes]
        descartadas = [f for f in candidatas if not f["confirmada"]]
        for i, f in enumerate(fuentes, 1):
            f["num"] = i
        motivos = {}
        for f in descartadas:
            motivos[f["motivo"].split(" (")[0]] = motivos.get(f["motivo"].split(" (")[0], 0) + 1
        print(f"  Detección: {len(candidatas)} candidatas → {len(fuentes)} confirmadas, "
              f"{len(descartadas)} descartadas"
              + (" (" + ", ".join(f"{v} {k}" for k, v in motivos.items()) + ")" if motivos else ""))
        radio = fov_deg / 2 * np.sqrt(2) * 3600
        gaia = identificar.gaia_campo(ra, dec, radio)
        import catalogos
        simbad = catalogos.simbad_campo(ra, dec, radio) if usar_simbad else None
        wise = catalogos.wise_campo(ra, dec, radio) if usar_wise else None
        ident, no_det = identificar.identificar(
            fuentes, wcs_out, marcas, gaia, radio_arcsec=1.5 * pix,
            area_campo_arcsec2=(fov_deg * 3600) ** 2,
            cobertura=np.isfinite(profunda_id),
            simbad=simbad, wise=wise, radio_cono_arcsec=radio)
        # Fuentes "sin identificar" pegadas a una estrella MUCHO más brillante
        # suelen ser su ala, halo o picos de difracción: se descartan.
        ident, ruido_brillante = identificar.descartar_alas(
            ident, radio_px=4 * sobremuestreo, factor=5.0)
        for f in ruido_brillante:
            descartadas.append(f)
        if ruido_brillante:
            print(f"  {len(ruido_brillante)} fuente(s) dudosa(s) descartada(s) por estar "
                  "en el ala/halo de una estrella brillante")
        identificar.variabilidad(ident, cubo, [m["lambda_um"] for m in meta_video],
                                 escala=sobremuestreo)
        identificar.imprimir(ident, no_det)

    procesado, meta = procesar(cubo, meta_video, modo, sep_min_h)
    frames = renderizar(procesado, meta, modo, zoom, suave,
                        None if ident is not None else marcas, ident=ident,
                        no_det=no_det if mostrar_no_detectados else None,
                        n_no_det=len(no_det))

    info = {"ra": ra, "dec": dec, "fov_deg": fov_deg, "modo": modo,
            "coleccion": coleccion, "detector": detector,
            "pixel_arcsec": pix, "n_pixeles": forma[0], "zoom": zoom,
            "cuadros": meta_completa, "cuadros_video": [m["fecha_utc"] for m in meta],
            "alerce": marcas, "asteroides_skybot": resumen_ast,
            "identificacion": ident,
            "descartadas": descartadas if usar_alerce else None,
            "no_detectados": no_det,
            "mascara_flags": int(mascara_bits)}

    if hacer_apilado:
        profunda = apilar(cubo)
        etiqueta = {"fecha_utc": f"apilado de {len(cubo)} cuadros", "lambda_um": np.nan}
        img_ap = renderizar(profunda[None], [etiqueta], "directo", zoom, suave,
                            None if ident is not None else marcas, ident=ident,
                            no_det=no_det if mostrar_no_detectados else None,
                            n_no_det=len(no_det))[0]
        if salida:
            nombre = salida.rsplit(".", 1)[0] + "_apilado.png"
            iio.imwrite(nombre, img_ap)
            fits.writeto(nombre.replace(".png", ".fits"), profunda,
                         wcs_out.to_header(), overwrite=True)
            print("Imagen apilada:", nombre, "(y su .fits con coordenadas)")
        else:
            png_apilado = iio.imwrite("<bytes>", img_ap, extension=".png")

    if not salida:
        # Modo servidor: todo se devuelve en memoria, nada se escribe en disco
        gif = iio.imwrite("<bytes>", frames, extension=".gif",
                          duration=1000 / fps, loop=0)
        return {"gif": gif, "apilado_png": png_apilado,
                "espectro_png": png_espectro, "info": info}

    iio.imwrite(salida, frames, duration=1000 / fps, loop=0)
    if ident is not None:
        import identificar
        ruta_csv = identificar.guardar_csv(
            salida.rsplit(".", 1)[0] + "_identificacion.csv", ident,
            descartadas, no_det, excel_es=csv_excel_es)
        print("Tabla de identificación:", ruta_csv)
    with open(salida.rsplit(".", 1)[0] + ".json", "w") as f:
        json.dump(info, f, indent=2, default=str)
    print("Listo:", salida)
    return frames, meta


# ------------------------------------------------- 6. Procesamiento (mejoras)
def fondo_2d(img, caja=9):
    """
    Estima el fondo SUAVE de una imagen (gradientes, escalones, luz parásita,
    residuos del zodiacal) en dos pasadas:
      1) filtro de mediana grande sobre la imagen completa (fondo aproximado),
      2) se tapan las fuentes que sobresalen de ese fondo y se repite.
    Devuelve (fondo, ruido_robusto).
    """
    from scipy.ndimage import median_filter, binary_dilation
    med = np.nanmedian(img)
    base = np.where(np.isfinite(img), img, med)
    fondo = median_filter(base, size=caja, mode="nearest")
    resto = base - fondo
    ruido = 1.4826 * np.nanmedian(np.abs(resto - np.nanmedian(resto)))
    if not np.isfinite(ruido) or ruido == 0:
        ruido = np.nanstd(resto) or 1.0
    tapar = ~np.isfinite(img) | (resto > 3 * ruido)
    tapar = binary_dilation(tapar, iterations=max(1, caja // 6))
    fondo = median_filter(np.where(tapar, fondo, base), size=caja, mode="nearest")
    return fondo, ruido


def limpiar_fondo(cubo, caja=9, umbral=4.0):
    """
    Resta el fondo 2D de cada cuadro y detecta cuadros con fondo anómalo
    (gradientes o escalones fuertes, típicos cerca del borde del detector).
    Devuelve (cubo_limpio, lista_de_calidad, metrica).
    """
    limpios, metrica = [], []
    for c in cubo:
        fondo, ruido = fondo_2d(c, caja)
        limpios.append(c - fondo)
        rango = np.nanpercentile(fondo, 98) - np.nanpercentile(fondo, 2)
        metrica.append(float(rango / ruido))
    metrica = np.array(metrica)
    corte = max(2.0, umbral * np.nanmedian(metrica))
    calidad = ["mala" if m > corte else "buena" for m in metrica]
    return np.array(limpios), calidad, metrica


def quitar_fondo(img):
    """Resta el nivel de fondo (mediana) para que todos los cuadros partan de ~0."""
    return img - np.nanmedian(img)


def ajustar_escala(f, ref):
    """
    Busca el factor s que mejor cumple f ≈ s·ref usando solo los píxeles brillantes
    (las estrellas). Compensa que una misma estrella se ve más o menos brillante
    según la longitud de onda de cada cuadro.
    """
    ok = np.isfinite(f) & np.isfinite(ref)
    if ok.sum() < 20:
        return 1.0
    umbral = np.nanpercentile(ref[ok], 90)
    b = ok & (ref > umbral)
    den = np.sum(ref[b] ** 2)
    s = float(np.sum(f[b] * ref[b]) / den) if den > 0 else 1.0
    # Un factor fuera de rango indica que no hay estrellas útiles: no escalar
    return s if 0.2 <= s <= 5 else 1.0


def procesar(cubo, meta, modo, sep_min_h=1.0, k_vecinos=3):
    """
    directo  : imagen con el fondo restado.
    resta    : cada cuadro menos una referencia hecha con los k cuadros de λ más
               parecida PERO tomados al menos sep_min_h horas antes o después
               (así un objeto que se mueve no se cancela a sí mismo).
    anterior : cada cuadro menos el cuadro anterior (escalado). Resalta lo que
               cambió entre dos fechas seguidas.
    """
    cubo = np.array([quitar_fondo(c) for c in cubo])
    n = len(cubo)
    mjd = np.array([m["mjd"] for m in meta])
    lam = np.array([m["lambda_um"] for m in meta], dtype=float)

    if modo == "directo":
        return cubo, meta

    salida, meta_out = [], []
    for i in range(n):
        if modo == "anterior":
            if i == 0:
                continue
            ref = cubo[i - 1]
            info = {"referencia": [meta[i - 1]["fecha_utc"]]}
        else:  # resta
            lejos = np.abs(mjd - mjd[i]) * 24 >= sep_min_h
            cand = np.where(lejos & np.isfinite(lam))[0]
            if len(cand) == 0:
                cand = np.where(np.arange(n) != i)[0]
            cand = cand[np.argsort(np.abs(lam[cand] - lam[i]))][:k_vecinos]
            ref = np.nanmedian(cubo[cand], axis=0)
            info = {"referencia": [meta[j]["fecha_utc"] for j in cand]}

        s = ajustar_escala(cubo[i], ref)
        salida.append(cubo[i] - s * ref)
        meta_out.append({**meta[i], **info, "escala": s})

    return np.array(salida), meta_out


# --------------------------------------------------------- 7. Renderizado
def dibujar_marcas(d, marcas, n_pix, zoom, fuente):
    """Círculos de ALeRCE: color según la clase; línea gruesa si estuvo activo
    en las fechas de SPHEREx. Etiquetas solo si hay pocos objetos."""
    from alerce_capa import COLORES
    r = max(6, 2.5 * zoom)
    for m in marcas:
        X = (m["x"] + 0.5) * zoom
        Y = (n_pix - 1 - m["y"] + 0.5) * zoom
        color = COLORES.get(m["familia"], COLORES["otro"])
        ancho = 3 if m.get("activo_en_spherex") else 1
        d.ellipse([X - r, Y - r, X + r, Y + r], outline=color, width=ancho)
        if len(marcas) <= 12:
            d.text((X + r + 2, Y - 7), m["clase"] or m["oid"][-6:], fill=color, font=fuente)


def _rellenar_huecos(img, caja=5):
    """
    SOLO PARA MOSTRAR: rellena los píxeles enmascarados que quedan DENTRO de la
    zona con datos (huecos chicos), con la mediana de sus vecinos. Los bordes sin
    datos se dejan tal cual. No se usa para detectar ni medir.
    """
    from scipy.ndimage import binary_fill_holes, median_filter
    nan = ~np.isfinite(img)
    if not nan.any():
        return img
    huecos = nan & binary_fill_holes(~nan)
    if not huecos.any():
        return img
    base = np.where(nan, np.nanmedian(img), img)
    out = img.copy()
    out[huecos] = median_filter(base, size=caja)[huecos]
    return out


def a_8bits_asinh(cubo, vmin, vmax, a=8.0):
    """Escala asinh: muestra a la vez estrellas brillantes y fuentes débiles."""
    x = np.clip((cubo - vmin) / (vmax - vmin), 0, 1)
    x = np.arcsinh(a * x) / np.arcsinh(a)
    x = np.nan_to_num(x, nan=0.0)
    return (x * 255).astype(np.uint8)


def renderizar(cubo, meta, modo, zoom=None, suave=False, marcas=None, ident=None,
               no_det=None, n_no_det=0):
    """Convierte el cubo a cuadros RGB de 8 bits, agrandados y con etiqueta."""
    from PIL import Image, ImageDraw, ImageFont

    cubo = np.array([_rellenar_huecos(c) for c in cubo])
    datos = cubo[np.isfinite(cubo)]
    if modo == "directo":
        # Límites comunes a todos los cuadros: fondo un poco bajo la mediana,
        # tope en las estrellas brillantes. Con asinh no se "queman".
        vmin = np.nanpercentile(datos, 35)
        vmax = np.nanpercentile(datos, 99.8)
        grises = a_8bits_asinh(cubo, vmin, vmax)[:, ::-1, :]  # FITS: origen abajo
    else:
        # Diferencias: escala simétrica en torno a 0 (gris = sin cambio)
        lim = np.nanpercentile(np.abs(datos), 99.5)
        grises = a_8bits(cubo, -lim, lim)[:, ::-1, :]
    n_pix = grises.shape[1]
    if zoom is None:
        zoom = max(1, int(round(400 / n_pix)))

    import identificar
    fuente = identificar.fuente_tt(15)   # con tildes, λ y µ

    panel = None
    if ident:
        panel = identificar.panel_leyenda(ident, n_pix * zoom, n_no_det=n_no_det)

    frames = []
    for g, m in zip(grises, meta):
        img = Image.fromarray(g).resize((n_pix * zoom, n_pix * zoom),
                                        Image.LANCZOS if suave else Image.NEAREST
                                        ).convert("RGB")
        d = ImageDraw.Draw(img)
        if np.isfinite(m.get("lambda_um", np.nan)):
            texto = f"{m['fecha_utc'][:16].replace('T', ' ')} UTC    λ = {m['lambda_um']:.3f} µm"
        else:
            texto = m["fecha_utc"]
        d.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        d.text((6, 3), texto, fill=(255, 255, 255), font=fuente)
        etiqueta = {"directo": "directo", "resta": "resta (ref. λ cercana)",
                    "anterior": "menos cuadro anterior"}[modo]
        d.text((6, img.height - 20), etiqueta, fill=(255, 220, 0), font=fuente)
        dibujar_marcas(d, marcas or [], n_pix, zoom, fuente)
        if ident:
            identificar.dibujar(d, ident, n_pix, zoom, fuente, no_det=no_det)
        if m.get("objetivo_xy"):
            X = (m["objetivo_xy"][0] + 0.5) * zoom
            Y = (n_pix - 1 - m["objetivo_xy"][1] + 0.5) * zoom
            a, b = max(5, 2 * zoom), max(10, 4 * zoom)  # hueco central y largo
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                d.line([(X + dx * a, Y + dy * a), (X + dx * b, Y + dy * b)],
                       fill=(255, 0, 255), width=2)
        if m.get("asteroides"):
            import asteroides_capa
            asteroides_capa.dibujar(d, m["asteroides"], n_pix, zoom, fuente)
        cuadro = np.asarray(img)
        if panel is not None:
            cuadro = np.concatenate([cuadro, panel], axis=1)
        frames.append(cuadro)
    return frames


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ra", type=float, nargs="?",
                   help="AR en grados (opcional si usas --objeto)")
    p.add_argument("dec", type=float, nargs="?")
    p.add_argument("--objeto", help="nombre del objeto (SIMBAD); corrige su movimiento propio")
    p.add_argument("--fotometria", action="store_true",
                   help="medir el brillo del objetivo en cada cuadro (espectro + curva de luz)")
    p.add_argument("--max", type=int, default=200, help="máximo de cuadros a usar")
    p.add_argument("--incluir-malos", action="store_true",
                   help="no excluir del video los cuadros con fondo anómalo")
    p.add_argument("--hilos", type=int, default=4,
                   help="descargas simultáneas (más = más rápido, pero IRSA puede rechazar)")
    p.add_argument("--fov", type=float, default=0.1, help="campo de visión en grados")
    p.add_argument("--detector", type=int, choices=range(1, 7))
    p.add_argument("--coleccion", default="spherex_qr3")
    p.add_argument("--modo", choices=["directo", "resta", "anterior"], default="directo")
    p.add_argument("--sep", type=float, default=1.0,
                   help="modo resta: horas mínimas entre un cuadro y su referencia")
    p.add_argument("--zoom", type=int, help="factor de agrandado (por defecto se calcula con --tam)")
    p.add_argument("--sobremuestreo", type=int, default=2,
                   help="grilla N veces más fina que el píxel nativo (2 o 3 recomendado)")
    p.add_argument("--suave", action="store_true",
                   help="agrandar con interpolación suave en vez de píxeles cuadrados")
    p.add_argument("--apilar", action="store_true",
                   help="además, guardar una imagen profunda combinando todos los cuadros")
    p.add_argument("--alerce", "--identificar", dest="alerce", action="store_true",
                   help="identificar cada fuente (ALeRCE, SIMBAD, Gaia y WISE) y numerarla")
    p.add_argument("--tam", type=int, default=720, help="tamaño del video en píxeles")
    p.add_argument("--prob-temprana", type=float, default=0.7,
                   help="probabilidad mínima para aceptar la clasificación TEMPRANA de ALeRCE")
    p.add_argument("--csv-punto", action="store_true",
                   help="CSV con coma y punto decimal (formato inglés) en vez del de Excel en español")
    p.add_argument("--sin-simbad", action="store_true", help="no consultar SIMBAD")
    p.add_argument("--sin-wise", action="store_true", help="no consultar WISE")
    p.add_argument("--mostrar-no-detectados", action="store_true",
                   help="dibujar (punteados) los objetos de ALeRCE que SPHEREx no detecta")
    p.add_argument("--fuentes", type=int, default=25, help="máximo de fuentes a identificar")
    p.add_argument("--asteroides", action="store_true",
                   help="marcar dónde estaban los asteroides conocidos en cada cuadro (SkyBoT)")
    p.add_argument("--prob", type=float, default=0.5,
                   help="ALeRCE: probabilidad mínima de la clasificación (0-1)")
    p.add_argument("--vmax", type=float, default=21.0,
                   help="asteroides: magnitud V máxima (más grande = más débiles)")
    p.add_argument("--listar", action="store_true",
                   help="solo mostrar qué imágenes hay, sin descargar nada")
    p.add_argument("--salida", default="spherex.gif")
    p.add_argument("--fps", type=float, default=3)
    p.add_argument("--mascara", type=int, default=MASCARA_RECOMENDADA,
                   help="bits de FLAGS a enmascarar (por defecto la máscara recomendada "
                        "por el Explanatory Supplement; 0 = ninguno)")
    a = p.parse_args()
    objetivo = None
    if a.objeto:
        import objetivos
        objetivo = objetivos.resolver(a.objeto, a.ra, a.dec)
        a.ra, a.dec = objetivo["ra"], objetivo["dec"]
    elif a.ra is None or a.dec is None:
        p.error("da RA y Dec, o usa --objeto NOMBRE")
    hacer_video(a.ra, a.dec, a.fov, a.detector, a.coleccion, a.modo, a.salida,
                fps=a.fps, mascara_bits=a.mascara, sep_min_h=a.sep, zoom=a.zoom,
                solo_listar=a.listar, sobremuestreo=a.sobremuestreo,
                suave=a.suave, hacer_apilado=a.apilar, usar_alerce=a.alerce,
                usar_asteroides=a.asteroides, prob_min=a.prob, v_max=a.vmax,
                objetivo=objetivo, hacer_fotometria=a.fotometria, max_cuadros=a.max,
                hilos=a.hilos, descartar_malos=not a.incluir_malos,
                tam=a.tam, n_fuentes=a.fuentes, prob_temprana=a.prob_temprana,
                mostrar_no_detectados=a.mostrar_no_detectados,
                usar_simbad=not a.sin_simbad, usar_wise=not a.sin_wise,
                csv_excel_es=not a.csv_punto)
