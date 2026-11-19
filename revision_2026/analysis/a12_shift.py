"""Band displacement: extrapolated shift and broadening versus the values fitted between the warmest input and each
withheld measured spectrum (experimental test windows), and the n-propyl acetate band maximum of Figure 6."""
from common import *
import broad as B, physics as ph, pickle
from scipy.ndimage import gaussian_filter1d
tr_meas, tr_hitemp, hitran, test = split_groups(); mdl = B.main_model(); pr = {r['record_id'] + '|' + r['window']: r for r in published_rows()}; rows = []
def fwhm(x, y):
    h = y.max() / 2; idx = np.flatnonzero(y >= h); return float(x[idx[-1]] - x[idx[0]])
for g in test:
    if g['kind'] != 'measurement': continue
    cp = B.cold_params(g); r2, nue, rho = ph.shift_law(cp)
    for i, r in enumerate(g['records']):
        T = r['temperature_K']; m = g['masks'][i]; ref = g['references'][i]; d_fit, w_fit, e_fit = ph.fit_deformation(g['x'], g['warm'], ref, m)
        rows.append(dict(species=g['species'], window=g['window'], temperature_K=T, shift_pred=float(ph.shift(cp, T)), shift_linear=float(r2 * (T - cp['T'][2])), shift_fit=d_fit, broad_pred=float(ph.broadening(cp, T)), broad_fit=w_fit,
                         best_deformation_error=e_fit, physics_error=E(B.predict(None, g, T), ref, m), model_error=E(B.predict(mdl, g, T), ref, m), unchanged_error=E(g['warm'], ref, m), published_error=pr[r['record_id'] + '|' + g['window']]['final_error_percent'],
                         r2_per100K=100 * r2, rate_ratio=rho, nu_e=nue, warm_fwhm=fwhm(g['x'], g['warm'])))
# n-propyl acetate band maximum (same procedure as the original Figure 6)
g = next(g for g in test if g['species'] == 'n-propyl acetate'); x = g['x']; dx = x[1] - x[0]; sm = lambda y: gaussian_filter1d(y, (8 / 2.3548) / dx, mode='nearest'); win = (x >= 1210) & (x <= 1255); peak = []
def pk(y):
    ys = sm(y); idx = np.flatnonzero(win); j = idx[np.argmax(ys[idx])]; a, b, c = ys[j - 1], ys[j], ys[j + 1]; return float(x[j] + 0.5 * (a - c) / (a - 2 * b + c) * dx)
for i, r in enumerate(g['records']):
    T = r['temperature_K']; ref = np.where(g['masks'][i], g['references'][i], 0)
    peak.append(dict(T=T, measured=pk(ref), predicted=pk(B.predict(mdl, g, T)), physics=pk(B.predict(None, g, T)), warm_input=pk(g['warm'])))
jdump(dict(rows=rows, npa_peak=peak), 'shift.json')
print(f"{'species':20s} {'T':>6s} {'d_pred':>7s} {'d_lin':>7s} {'d_fit':>7s} {'w_pred':>7s} {'w_fit':>7s} {'E_best':>7s} {'E_phys':>7s} {'E_model':>7s} {'E_pub':>7s} {'E_unch':>7s}")
for r in rows: print(f"{r['species'][:20]:20s} {r['temperature_K']:6.0f} {r['shift_pred']:7.2f} {r['shift_linear']:7.2f} {r['shift_fit']:7.2f} {r['broad_pred']:7.2f} {r['broad_fit']:7.2f} {r['best_deformation_error']:7.1f} {r['physics_error']:7.1f} {r['model_error']:7.1f} {r['published_error']:7.1f} {r['unchanged_error']:7.1f}  {r['window'][-18:]}")
for p in peak: print('nPA peak', p)
