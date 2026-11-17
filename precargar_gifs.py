"""
================================================================================
MÓDULO: precargar_gifs.py
PROYECTO: NASA Space Apps Challenge 2026 - Visor Vectorial SPHEREx + ALeRCE
================================================================================
Genera una sola vez los GIF de SPHEREx de los 5 objetivos del desafío (modo
resta y modo directo) y los guarda en datos_web/gifs/<slug>_<modo>.gif.

Así la demo los muestra al instante:
  - app.py los entrega sin volver a pedir imágenes a IRSA/S3 cuando la página
    pide el GIF del objeto central con los parámetros por defecto;
  - la versión publicada en GitHub Pages los lee directamente de datos_web/gifs/.

Uso (con el servidor apagado o encendido, da igual):
    python precargar_gifs.py                     # los 5 objetivos, ambos modos
    python precargar_gifs.py --solo wise_0855    # uno solo (usa el slug)
    python precargar_gifs.py --modos resta       # solo un modo
    python precargar_gifs.py --forzar            # rehace aunque ya existan

Cada objetivo tarda unos minutos la primera vez (descarga los recortes); el
segundo modo del mismo objetivo es mucho más rápido porque reutiliza lo
descargado.
================================================================================
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DIR = Path(__file__).resolve().parent
if str(DIR) not in sys.path:
    sys.path.insert(0, str(DIR))

# Importar app aplica los mismos ajustes que usa el servidor: índices locales o
# IRSA para buscar cuadros, lectura directa desde S3 y caché en memoria.
import app  # noqa: E402
import objetivos  # noqa: E402
import spherex_video as sv  # noqa: E402
from generador_json_definitivo import OBJETIVOS_CHALLENGE  # noqa: E402

# Nombres que SIMBAD resuelve bien (los mismos que usa gif.html)
NOMBRE_SIMBAD = {
    "barnards_star": "Barnard's Star",
    "epsilon_indi_b": "eps Ind B",
    "wise_0855": "WISE J085510.83-071442.5",
    "rw_aurigae": "RW Aur",
    "r_draconis": "R Dra",
}


def resolver_objetivo(obj: dict) -> dict:
    """
    Posición J2000 y movimiento propio del objetivo. Primero SIMBAD; si no
    responde o no trae movimiento propio, usa los valores de referencia del
    catálogo del desafío (OBJETIVOS_CHALLENGE).
    """
    nombre = NOMBRE_SIMBAD.get(obj["slug"], obj["nombre_display"])
    try:
        info = objetivos.resolver(nombre)
    except (SystemExit, Exception) as exc:
        print(f"  SIMBAD no resolvió '{nombre}' ({exc}); uso los valores de referencia.")
        info = {}
    if not info.get("pmra") and not info.get("pmdec"):
        info.update(ra=float(obj["ra"]), dec=float(obj["dec"]),
                    pmra=float(obj["pmra_ref"]), pmdec=float(obj["pmdec_ref"]),
                    plx=float(obj["paralaje_ref"]), fuente="OBJETIVOS_CHALLENGE")
    info["nombre"] = obj["nombre_display"]
    return info


def generar(obj: dict, modo: str, info: dict) -> bytes:
    """Mismo llamado que hace app.py en /api/gif, con los parámetros por defecto."""
    p = app.GIF_PRECARGA_PARAMS
    res = sv.hacer_video(
        ra=info["ra"],
        dec=info["dec"],
        fov_deg=p["fov"],
        detector=p["detector"],
        coleccion="spherex_dr1,spherex_qr3",
        modo=modo,
        salida=None,
        max_cuadros=p["max_cuadros"],
        fps=p["fps"],
        usar_alerce=p["identificar"],
        usar_asteroides=p["asteroides"],
        objetivo=info,
        tam=520,
    )
    if not isinstance(res, dict) or not res.get("gif"):
        raise RuntimeError("el motor no devolvió un GIF")
    return res["gif"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Precalcula los GIF de los 5 objetivos del desafío.")
    ap.add_argument("--solo", nargs="+", metavar="SLUG",
                    help="slugs a generar, p. ej. barnards_star wise_0855")
    ap.add_argument("--modos", nargs="+", choices=app.GIF_PRECARGA_MODOS,
                    default=list(app.GIF_PRECARGA_MODOS))
    ap.add_argument("--forzar", action="store_true", help="rehacer aunque el GIF ya exista")
    args = ap.parse_args(argv)

    objetivos_a_usar = [o for o in OBJETIVOS_CHALLENGE if not args.solo or o["slug"] in args.solo]
    if args.solo:
        desconocidos = set(args.solo) - {o["slug"] for o in OBJETIVOS_CHALLENGE}
        if desconocidos:
            print("Slugs desconocidos:", ", ".join(sorted(desconocidos)))
            print("Válidos:", ", ".join(o["slug"] for o in OBJETIVOS_CHALLENGE))
            return 2

    app.CARPETA_GIFS.mkdir(parents=True, exist_ok=True)
    ruta_indice = app.CARPETA_GIFS / "indice.json"
    indice = json.loads(ruta_indice.read_text(encoding="utf-8")) if ruta_indice.exists() else {}

    hechos, saltados, fallidos = [], [], []
    for obj in objetivos_a_usar:
        pendientes = [m for m in args.modos
                      if args.forzar or not app.ruta_gif_precargado(obj["slug"], m).exists()]
        saltados += [f"{obj['slug']}_{m}" for m in args.modos if m not in pendientes]
        if not pendientes:
            continue

        print(f"\n>>> {obj['nombre_display']}")
        info = resolver_objetivo(obj)
        for modo in pendientes:
            clave = f"{obj['slug']}_{modo}"
            print(f"  Generando modo {modo}...")
            t0 = time.time()
            try:
                gif = generar(obj, modo, info)
            except (SystemExit, Exception) as exc:
                # El motor termina con SystemExit cuando IRSA/S3 no entregan imágenes
                print(f"  ✗ {clave}: {exc}")
                fallidos.append(clave)
                continue
            ruta = app.ruta_gif_precargado(obj["slug"], modo)
            ruta.write_bytes(gif)
            seg = time.time() - t0
            print(f"  ✓ {ruta.name}  ({len(gif) / 1e6:.1f} MB, {seg:.0f} s)")
            hechos.append(clave)
            indice[clave] = {
                "objetivo": obj["nombre_display"],
                "modo": modo,
                "archivo": ruta.name,
                "bytes": len(gif),
                "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "parametros": app.GIF_PRECARGA_PARAMS,
                "posicion_fuente": info.get("fuente", "SIMBAD"),
            }
            ruta_indice.write_text(json.dumps(indice, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nResumen")
    print(f"  generados: {len(hechos)}" + (f"  ({', '.join(hechos)})" if hechos else ""))
    print(f"  ya existían: {len(saltados)}" + ("  (usa --forzar para rehacerlos)" if saltados else ""))
    if fallidos:
        print(f"  fallaron: {len(fallidos)}  ({', '.join(fallidos)})")
        print("  Vuelve a ejecutar más tarde: solo se generan los que faltan.")
    print(f"  carpeta: {app.CARPETA_GIFS}")
    return 1 if fallidos else 0


if __name__ == "__main__":
    sys.exit(main())
