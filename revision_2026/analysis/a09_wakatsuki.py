"""Temperature equation of Wakatsuki et al. (2005): (i) fitted to the three near-ambient PNNL inputs only, as in the
original submission; (ii) fitted in its intended regime, i.e. to the PNNL inputs plus the measured spectra of the
same molecule up to about 600 K, and extrapolated to the remaining higher-temperature measurements."""
from common import *
import broad as B, baselines as BL
tr_meas, tr_hitemp, hitran, test = split_groups(); mdl = B.main_model(); out = {}
# (i) three-input fits for every experimental test window
three = []
for g in test:
    if g['kind'] != 'measurement': continue
    coef, cond = BL.wakatsuki_fit(g['x'], g['inputs'], g['Tin'], bound_exponent=False); coefb, _ = BL.wakatsuki_fit(g['x'], g['inputs'], g['Tin'])
    for i, r in enumerate(g['records']):
        T = r['temperature_K']; m = g['masks'][i]; ref = g['references'][i]
        three.append(dict(species=g['species'], window=g['window'], temperature_K=T, wakatsuki_three_input=E(BL.wakatsuki_predict(g['x'], coef, T), ref, m), wakatsuki_three_input_bounded=E(BL.wakatsuki_predict(g['x'], coefb, T), ref, m), exponent_at_bound_fraction=float(np.mean((coefb[2] <= -2 + 1e-9) | (coefb[2] >= -1e-9))), thirmo=E(B.predict(mdl, g, T), ref, m), condition_number=cond))
s3 = {k: summarize([dict(r, error_percent=r[k]) for r in three]) for k in ('wakatsuki_three_input', 'wakatsuki_three_input_bounded', 'thirmo')}; out['three_input'] = dict(rows=three, molecule=s3)
print('three-input Wakatsuki, molecule means:', {k: round(v, 1) for k, v in s3['wakatsuki_three_input'].items()}); print('   bounded exponent:', {k: round(v, 1) for k, v in s3['wakatsuki_three_input_bounded'].items()}, 'mean', round(float(np.mean(list(s3['wakatsuki_three_input_bounded'].values()))), 2), ' thIRmo', {k: round(v, 1) for k, v in s3['thirmo'].items()})
for sp in ('n-propyl acetate', 'di-n-propyl ether'): print('  ', sp, [(r['temperature_K'], round(r['wakatsuki_three_input'], 1), round(r['wakatsuki_three_input_bounded'], 1), round(r['thirmo'], 1), round(r['exponent_at_bound_fraction'], 2)) for r in three if r['species'] == sp])
# (ii) intended regime: inputs spanning near-ambient to about 600 K
T_SPLIT = 615.0; wide = []; fits = []
for g in test:
    if g['kind'] != 'measurement': continue
    lo = [i for i, r in enumerate(g['records']) if r['temperature_K'] <= T_SPLIT]; hi = [i for i, r in enumerate(g['records']) if r['temperature_K'] > T_SPLIT]
    if not lo or not hi: continue
    valid = np.all([g['masks'][i] for i in lo], axis=0); spectra = np.vstack([g['inputs'], [g['references'][i] for i in lo]]); temps = np.r_[g['Tin'], [g['records'][i]['temperature_K'] for i in lo]]
    coef, cond = BL.wakatsuki_fit(g['x'][valid], spectra[:, valid], temps); coefu, _ = BL.wakatsuki_fit(g['x'][valid], spectra[:, valid], temps, bound_exponent=False)
    fits.append(dict(species=g['species'], window=g['window'], fit_temperatures=[float(t) for t in temps], n_inputs=len(temps), condition_number=cond))
    for i in hi:
        T = g['records'][i]['temperature_K']; m = g['masks'][i] & valid; ref = g['references'][i]; yw = np.zeros_like(ref); yw[valid] = BL.wakatsuki_predict(g['x'][valid], coef, T); yu = np.zeros_like(ref); yu[valid] = BL.wakatsuki_predict(g['x'][valid], coefu, T)
        wide.append(dict(species=g['species'], window=g['window'], temperature_K=T, n_inputs=len(temps), fit_Tmax=float(temps.max()), wakatsuki_wide_input=E(yw, ref, m), wakatsuki_wide_input_unbounded=E(yu, ref, m), thirmo=E(B.predict(mdl, g, T), ref, m), physics=E(B.predict(None, g, T), ref, m),
                         unchanged=E(g['warm'], ref, m), hottest_input_unchanged=E(spectra[-1], ref, m), common_points=int(m.sum())))
sw = {k: summarize([dict(r, error_percent=r[k]) for r in wide]) for k in ('wakatsuki_wide_input', 'wakatsuki_wide_input_unbounded', 'thirmo', 'physics', 'unchanged', 'hottest_input_unchanged')}; out['wide_input'] = dict(rows=wide, molecule=sw, fits=fits, T_split=T_SPLIT)
print('intended-regime Wakatsuki (inputs up to %.0f K):' % T_SPLIT)
for f in fits: print('   fit', f['species'], f['window'][-22:], np.round(f['fit_temperatures'], 0), 'cond %.0f' % f['condition_number'])
for r in wide: print(f"   {r['species']:20s} {r['window'][-18:]:18s} {r['temperature_K']:6.1f} K  Wakatsuki({r['n_inputs']} inputs to {r['fit_Tmax']:.0f} K) {r['wakatsuki_wide_input']:6.1f}% (unbounded {r['wakatsuki_wide_input_unbounded']:6.1f}%)   thIRmo-ET(3 PNNL) {r['thirmo']:5.1f}%   hottest input unchanged {r['hottest_input_unchanged']:5.1f}%")
print('   molecule means:', {k: {a: round(b, 1) for a, b in v.items()} for k, v in sw.items()})
jdump(out, 'wakatsuki.json')
