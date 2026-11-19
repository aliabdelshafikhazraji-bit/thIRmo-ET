"""Diisopropyl ether at 1121.82 cm-1: the shock-tube correlation of Adil and Farooq (J. Phys. Chem. A 127 (2023)
7764-7771) over its stated 500-1500 K range and the scanned test spectra of the same study, compared with the
broad-band prediction from the three PNNL spectra (Table G4 and Figure S5 of the supplementary material)."""
from common import *
import broad as B
from config import DIGITIZED
C = json.load(open(DIGITIZED / 'single_wavenumber_correlations.json'))['DiPE']; NA = 6.02214076e23
SP = jload('seed_probe.json'); Q_BROAD = SP['q']; BETA = SP['beta']
def corr(T):
    """Published correlation, m2/mol converted to cm2/molecule."""
    a = C['coeffs']; T = np.asarray(T, float); return (a[0] + a[1] * T + a[2] * T ** 2) * 1e4 / NA
def summary(rows):
    R = np.array([r['pred'] / r['corr'] for r in rows]); Tr = np.array([r['T'] for r in rows]); cov = np.array([r['lo'] <= r['corr'] <= r['hi'] for r in rows])
    return dict(ratio_min=float(R.min()), ratio_max=float(R.max()), ratio_mean=float(R.mean()), mean_abs_dev_percent=float(100 * np.mean(np.abs(R - 1))), ratio_at_ends=[float(R[0]), float(R[-1])],
                T_at_ratio_min=float(Tr[np.argmin(R)]), T_at_ratio_max=float(Tr[np.argmax(R)]), covered_fraction=float(cov.mean()), covered_T=[float(t) for t in Tr[cov]])
g = next(g for g in groups() if g['species'] == 'diisopropyl ether' and g['split'] == 'testing'); x = g['x']; nu0 = C['wavenumber_cm1']; mdl = B.main_model()
assert x[0] < nu0 < x[-1]; rows = []
for T in np.arange(C['T_range_K'][0], C['T_range_K'][1] + 1, 25.0):
    y = B.predict(mdl, g, T); tr = B.predict(mdl, g, T, per_tree=True); half = Q_BROAD * (tr.std(0) + BETA * y.max()); s_ = float(np.interp(nu0, x, y)); h = float(np.interp(nu0, x, half))
    rows.append(dict(T=float(T), corr=float(corr(T)), pred=s_, lo=max(s_ - h, 0), hi=s_ + h, phys=float(np.interp(nu0, x, B.predict(None, g, T)))))
meas = []
for i, r in enumerate(g['records']):
    if np.interp(nu0, x, g['masks'][i].astype(float)) > 0.99:
        T = r['temperature_K']; meas.append(dict(T=float(T), P=float(r['pressure_atm']), measured=float(np.interp(nu0, x, g['references'][i])), pred=float(np.interp(nu0, x, B.predict(mdl, g, T))), corr=float(corr(T))))
res = dict(branch='broad-band', window=[float(x[0]), float(x[-1])], probe=nu0, correlation=C, rows=rows, measured_spectra=meas, summary=summary(rows))
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res['summary'].items() if k != 'covered_T'})
for m in meas: print('  T %.0f  measured %.3e  predicted %.3e (%+.1f%%)  correlation %.3e (%+.1f%%)' % (m['T'], m['measured'], m['pred'], 100 * (m['pred'] / m['measured'] - 1), m['corr'], 100 * (m['corr'] / m['measured'] - 1)))
jdump(res, 'dipe_probe.json')
