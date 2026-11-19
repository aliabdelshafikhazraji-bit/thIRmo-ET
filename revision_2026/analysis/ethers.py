"""Ether probe wavenumbers: published calibration correlations (Adil and Farooq) versus thIRmo-ET, and
recalculation of the published shock-tube mole-fraction histories with Eq. (9)."""
from common import *
from config import DIGITIZED
import broad as B
NA = 6.02214076e23
PROBE = {'di-n-propyl ether': 1121.82, '1,2-dimethoxyethane': 1120.5}
RANGE = {'di-n-propyl ether': (550.0, 1500.0), '1,2-dimethoxyethane': (660.0, 1350.0)}
SHOCK = {'di-n-propyl ether': [(1151.0, 1.451), (1228.0, 1.382), (1306.0, 1.392)], '1,2-dimethoxyethane': [(1184.0, 1.806), (1226.0, 1.915), (1249.0, 1.895)]}
def correlation(species, T):
    """Published calibration correlations converted from m2/mol to cm2/molecule."""
    T = np.asarray(T, float)
    s = (-7.27e-8 * T ** 3 + 2.438e-4 * T ** 2 - 0.2856 * T + 128.03) if species == 'di-n-propyl ether' else (-0.0251 * T + 48.9)
    return s * 1e4 / NA
def window(species): return next(g for g in groups() if g['species'] == species and g['split'] == 'testing')
def sigma_probe(model, species, T, **kw):
    g = window(species); return float(np.interp(PROBE[species], g['x'], B.predict(model, g, T, **kw)))
if __name__ == '__main__':
    mdl = B.main_model(); out = {}
    for sp in PROBE:
        g = window(sp); lo, hi = RANGE[sp]; Ts = np.unique(np.r_[np.arange(lo, hi + 1, 10.0), [t for t, _ in SHOCK[sp]]])
        curve = [dict(T=float(T), pred=sigma_probe(mdl, sp, T), phys=sigma_probe(None, sp, T), corr=float(correlation(sp, T))) for T in Ts]
        # measured scanned spectra of the same studies at the probe wavenumber (withheld test records)
        meas = []
        for i, r in enumerate(g['records']):
            T = r['temperature_K']; ref = float(np.interp(PROBE[sp], g['x'], g['references'][i])); ok = bool(np.interp(PROBE[sp], g['x'], g['masks'][i].astype(float)) > 0.99)
            meas.append(dict(T=T, P=r['pressure_atm'], covered=ok, measured=ref, pred=sigma_probe(mdl, sp, T), phys=sigma_probe(None, sp, T), corr=float(correlation(sp, T)) if lo <= T <= hi else None))
        shock = [dict(T=T, P=P, pred=sigma_probe(mdl, sp, T), corr=float(correlation(sp, T))) for T, P in SHOCK[sp]]
        for s in shock: s['sigma_ratio_pred_over_corr'] = s['pred'] / s['corr']; s['concentration_change_percent'] = 100 * (s['corr'] / s['pred'] - 1)
        out[sp] = dict(probe=PROBE[sp], curve=curve, measured=meas, shock=shock)
        print(sp); [print('   shock T %.0f  pred %.3e corr %.3e  ratio %.3f  conc change %.1f%%' % (s['T'], s['pred'], s['corr'], s['sigma_ratio_pred_over_corr'], s['concentration_change_percent'])) for s in shock]
        [print('   meas  T %.0f  measured %.3e pred %.3e (%.1f%%) phys %.3e (%.1f%%) corr %s' % (m['T'], m['measured'], m['pred'], 100 * (m['pred'] / m['measured'] - 1), m['phys'], 100 * (m['phys'] / m['measured'] - 1), '%.3e (%.1f%%)' % (m['corr'], 100 * (m['corr'] / m['measured'] - 1)) if m['corr'] else '-')) for m in meas]
        print('   curve ratio pred/corr at', [(int(c['T']), round(c['pred'] / c['corr'], 2)) for c in curve if c['T'] % 100 == 0 or c['T'] in (lo, hi)])
    # recalculated histories from the digitized published curves
    import csv as _csv
    rows = list(_csv.DictReader(open(DIGITIZED / 'digitized_ether_histories.csv')))
    hist = {}
    for r in rows:
        key = (r['species'], float(r['T5_K']), float(r['P5_atm'])); hist.setdefault(key, []).append((float(r['time_ms']), float(r['published_mole_fraction_pct'])))
    res = []
    for (sp, T5, P5), v in sorted(hist.items()):
        v = np.asarray(sorted(v)); t, xr = v[:, 0], v[:, 1]; ratio = float(correlation(sp, T5)) / sigma_probe(mdl, sp, T5); xh = xr * ratio
        Ex = 100 * np.trapezoid(np.abs(xh - xr), t) / np.trapezoid(np.maximum(xr, 0), t)
        res.append(dict(species=sp, T5=T5, P5=P5, ratio=ratio, Ex=float(Ex), signed=float(100 * (ratio - 1)), t=t.tolist(), x_pub=xr.tolist(), x_new=xh.tolist()))
        print(f'   history {sp:20s} {T5:.0f} K {P5:.3f} atm  sigma_corr/sigma_pred {ratio:.3f}  E_x {Ex:.1f}%  signed {100*(ratio-1):+.1f}%')
    out['histories'] = res; jdump(out, 'ethers.json')
