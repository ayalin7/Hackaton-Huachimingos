import math
from generador_interpolacion import DIAS_POR_ANIO, MAS_A_DEG, evaluar_posicion_en_tiempo


def evaluar_espejo_js(eq: dict, mjd_objetivo: float) -> tuple[float, float]:
    dt_yr = (mjd_objetivo - eq["t0_mjd"]) / DIAS_POR_ANIO
    cos_dec0 = max(math.cos(math.radians(eq["dec_0_deg"])), 1e-6)
    d_ra_mas = eq["mu_ra_cosdec_mas_yr"] * dt_yr + 0.5 * eq.get("acc_ra_mas_yr2", 0.0) * (dt_yr ** 2)
    d_dec_mas = eq["mu_dec_mas_yr"] * dt_yr + 0.5 * eq.get("acc_dec_mas_yr2", 0.0) * (dt_yr ** 2)
    return eq["ra_0_deg"] + (d_ra_mas * MAS_A_DEG) / cos_dec0, eq["dec_0_deg"] + d_dec_mas * MAS_A_DEG


if __name__ == "__main__":
    eq_test = {
        "t0_mjd": 60810.5,
        "ra_0_deg": 269.452076,
        "dec_0_deg": 4.693365,
        "mu_ra_cosdec_mas_yr": -801.55,
        "mu_dec_mas_yr": 10362.39,
        "acc_ra_mas_yr2": 0.0,
        "acc_dec_mas_yr2": 0.0,
    }
    ra_py, dec_py = evaluar_posicion_en_tiempo(eq_test, 61175.75)
    ra_js, dec_js = evaluar_espejo_js(eq_test, 61175.75)

    assert math.isclose(ra_py, ra_js, rel_tol=1e-12)
    assert math.isclose(dec_py, dec_js, rel_tol=1e-12)
    print(f"OK: Consistencia Python vs JS verificada (ΔRA={abs(ra_py - ra_js):.2e} deg)")