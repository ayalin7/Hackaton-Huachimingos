"""Quick check that IRSA data access works. Run:  python test_irsa.py"""
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.ipac.irsa import Irsa

c = SkyCoord("08h55m10.831s", "-07d14m42.53s")   # WISE 0855
for coll in ("spherex_qr2", "spherex_qr"):
    try:
        t = Irsa.query_sia(pos=(c, 0.02 * u.deg), collection=coll)
        print(coll, "->", len(t), "images; columns:", t.colnames[:12])
        if len(t):
            print(t["access_url"][0])
    except Exception as e:
        print(coll, "failed:", e)
