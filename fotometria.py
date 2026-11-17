"""
fotometria.py — Brillo del objetivo en cada cuadro → espectro y curva de luz.

Como cada cuadro SPHEREx ve una longitud de onda distinta, medir el brillo del
objeto en todos los cuadros da, "gratis", su ESPECTRO (brillo vs λ). Y si el
objeto es variable, el brillo vs fecha da su CURVA DE LUZ.

Método: fotometría de apertura simple
  flujo = Σ (I - fondo_local) · Ω_píxel
con I en MJy/sr y Ω_píxel el ángulo sólido de un píxel de la grilla (sr).
Es una estimación (no corrige por la parte de la PSF que cae fuera de la
apertura), suficiente para comparar formas de espectro y variaciones.
"""
import csv

import numpy as np

ARCSEC_RAD = np.pi / (180 * 3600)


def medir(cubo, posiciones, pix_grilla_arcsec, pix_nativo_arcsec=6.15,
          r_ap=1.5, r_in=3.0, r_out=5.0):
    """
    cubo: (N, ny, nx) en MJy/sr. posiciones: lista de (x, y) en píxeles de la grilla.
    Radios en PÍXELES NATIVOS de SPHEREx (se escalan a la grilla).
    Devuelve flujos en mJy (NaN si el objeto cae fuera o en una zona sin datos).
    """
    escala = pix_nativo_arcsec / pix_grilla_arcsec
    ra_, ri, ro = r_ap * escala, r_in * escala, r_out * escala
    omega = (pix_grilla_arcsec * ARCSEC_RAD) ** 2
    ny, nx = cubo.shape[1:]
    yy, xx = np.mgrid[:ny, :nx]
    flujos = []
    for img, (x, y) in zip(cubo, posiciones):
        r = np.hypot(xx - x, yy - y)
        ap, anillo = r <= ra_, (r >= ri) & (r <= ro)
        if not np.isfinite(img[ap]).all() or np.isfinite(img[anillo]).sum() < 5:
            flujos.append(np.nan)
            continue
        fondo = np.nanmedian(img[anillo])
        flujos.append(float(np.sum(img[ap] - fondo) * 1e6 * omega * 1e3))  # mJy
    return np.array(flujos)


def marcar_atipicos(meta, flujos, tolerancia=0.4, gap_dias=30, ventana=7):
    """
    Marca puntos que se alejan >40% de sus vecinos en λ dentro de la misma época
    (rayos cósmicos, píxeles malos, borde del detector). No los borra: los
    señala para que el gráfico y el análisis puedan ignorarlos.
    """
    flujos = np.asarray(flujos, dtype=float)
    lam = np.array([m["lambda_um"] for m in meta], dtype=float)
    mjd = np.array([m["mjd"] for m in meta], dtype=float)
    orden_t = np.argsort(mjd)
    epoca = np.zeros(len(mjd), dtype=int)
    e = 0
    for a, b in zip(orden_t[:-1], orden_t[1:]):
        if mjd[b] - mjd[a] > gap_dias:
            e += 1
        epoca[b] = e
    atipico = np.zeros(len(flujos), dtype=bool)
    for e in np.unique(epoca):
        idx = np.where(epoca == e)[0]
        idx = idx[np.argsort(lam[idx])]
        for k, i in enumerate(idx):
            vec = idx[max(0, k - ventana // 2): k + ventana // 2 + 1]
            vec = vec[vec != i]
            ref = np.nanmedian(flujos[vec]) if len(vec) >= 2 else np.nan
            if np.isfinite(ref) and ref != 0 and abs(flujos[i] - ref) / abs(ref) > tolerancia:
                atipico[i] = True
    malos = np.array([m.get("calidad") == "mala" for m in meta])
    return atipico | malos | ~np.isfinite(flujos), epoca


def guardar(nombre_base, meta, flujos, titulo=""):
    """
    Escribe <base>_fotometria.csv y <base>_espectro.png.
    Si nombre_base es None no escribe nada: devuelve el PNG como bytes.
    """
    flujos = np.asarray(flujos, dtype=float)
    atipico, epoca = marcar_atipicos(meta, flujos)
    for m, a in zip(meta, atipico):
        m["atipico"] = bool(a)
    if nombre_base is None:
        return _figura(meta, flujos, atipico, epoca, titulo, None)
    with open(nombre_base + "_fotometria.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fecha_utc", "mjd", "lambda_um", "banda", "flujo_mJy", "epoca", "atipico"])
        for m, fl, ep, at in zip(meta, flujos, epoca, atipico):
            w.writerow([m["fecha_utc"], f"{m['mjd']:.5f}", f"{m['lambda_um']:.4f}",
                        m.get("banda", ""), f"{fl:.4f}", int(ep), int(at)])
    _figura(meta, flujos, atipico, epoca, titulo, nombre_base + "_espectro.png")


def _figura(meta, flujos, atipico, epoca, titulo, ruta):
    """Dibuja espectro, curva de luz y cambio entre épocas. ruta=None → bytes PNG."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (para el gráfico instala matplotlib:  pip install matplotlib)")
        return

    lam = np.array([m["lambda_um"] for m in meta], dtype=float)
    mjd = np.array([m["mjd"] for m in meta], dtype=float)
    ok = np.isfinite(flujos) & np.isfinite(lam) & ~atipico
    malo = np.isfinite(flujos) & np.isfinite(lam) & atipico
    if ok.sum() < 2:
        print("  Fotometría: muy pocos puntos válidos para graficar.")
        return

    n_ep = len(np.unique(epoca[ok]))
    fig, ejes = plt.subplots(1, 3 if n_ep > 1 else 2, figsize=(16 if n_ep > 1 else 11, 4.2))
    a1, a2 = ejes[0], ejes[1]
    sc = a1.scatter(lam[ok], flujos[ok], c=mjd[ok] - mjd[ok].min(), cmap="viridis", s=18)
    if malo.any():
        a1.scatter(lam[malo], flujos[malo], marker="x", c="0.6", s=22, label="atípico")
        a1.legend(loc="best", fontsize=8)
    a1.set_xlabel("Longitud de onda λ (µm)")
    a1.set_ylabel("Flujo (mJy)")
    a1.set_title("Espectro (cada punto = un cuadro)")
    fig.colorbar(sc, ax=a1, label="días desde el primer cuadro")

    sc2 = a2.scatter(mjd[ok] - mjd[ok].min(), flujos[ok], c=lam[ok], cmap="plasma", s=18)
    if malo.any():
        a2.scatter(mjd[malo] - mjd[ok].min(), flujos[malo], marker="x", c="0.6", s=22)
    a2.set_xlabel("Días desde el primer cuadro")
    a2.set_ylabel("Flujo (mJy)")
    a2.set_title("Curva de luz (color = λ)")
    fig.colorbar(sc2, ax=a2, label="λ (µm)")
    if n_ep > 1:
        # Cambio de brillo de cada época respecto de la primera, por λ
        a3 = ejes[2]
        bordes = np.arange(0.7, 5.15, 0.3)
        centros = (bordes[:-1] + bordes[1:]) / 2
        eps = sorted(np.unique(epoca[ok]))

        def perfil(e):
            sel = ok & (epoca == e)
            return np.array([np.nanmedian(flujos[sel & (lam >= a) & (lam < b)])
                             if (sel & (lam >= a) & (lam < b)).sum() >= 2 else np.nan
                             for a, b in zip(bordes[:-1], bordes[1:])])
        ref = perfil(eps[0])
        f_ref = mjd[ok & (epoca == eps[0])].min()
        for e in eps[1:]:
            with np.errstate(invalid="ignore", divide="ignore"):
                dm = -2.5 * np.log10(perfil(e) / ref)
            dias = mjd[ok & (epoca == e)].min() - f_ref
            a3.plot(centros, dm, "o-", ms=4, label=f"+{dias:.0f} días")
        a3.axhline(0, color="0.5", lw=0.8)
        a3.invert_yaxis()
        a3.set_xlabel("Longitud de onda λ (µm)")
        a3.set_ylabel("Δ magnitud vs. 1ª época  (arriba = más brillante)")
        a3.set_title("Cambio entre épocas")
        a3.legend(fontsize=8)
    fig.suptitle(titulo)
    fig.tight_layout()
    if ruta is None:
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110)
        plt.close(fig)
        return buf.getvalue()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"  Fotometría: {ruta} y su .csv")
