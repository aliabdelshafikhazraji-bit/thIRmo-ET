"""Native-resolution test of the retained NO lines against HITEMP (Reviewer 2, comment 6).

The released line-resolved branch predicts the NO spectrum at native resolution (output_fwhm_cm1 = 0) from the three
PNNL NO spectra. HITEMP2019 cross sections are calculated line by line as in Section 2.3 of the manuscript (air
broadening, Voigt profiles, 25 cm-1 wing cutoff, HAPI/TIPS partition sums, terrestrial abundances) at the ten NO test
temperatures and at 0.7, 1.0, 1.5 and 1.9 atm. Each retained line is scored with Eq. (4) over +-1 cm-1 around its
centre (as in the leave-one-species-out test of the line ensemble) and by the cross section at the reference line
centre. The envelope alone (no retained-line term) is scored on the same points for comparison.

Inputs, not redistributed: NO_5T.TXT, NO_25T.TXT, NO_50T.TXT (PNNL; folder in THIRMO_PNNL) and 08_HITEMP2019.par.bz2
(HITEMP; path in THIRMO_HITEMP). Output: no_native.json."""
import os, bz2
from common import *
from scipy.special import wofz
import hapi
from spectral_model import ThreeSpectrumPredictor
from broad_model import frozen_core as core

PNNL_DIR = Path(os.environ.get('THIRMO_PNNL', REPO / 'data' / 'PNNL'))
HITEMP = Path(os.environ.get('THIRMO_HITEMP', REPO / 'data' / 'HITEMP' / '08_HITEMP2019.par.bz2'))
WINDOW = (1600.0, 2050.0); TIN = [278.13, 298.15, 323.13]      # as in the data set (dataset_manifest.json)
TEMPS = [373.15, 473.15, 573.15, 673.15, 773.15, 873.15, 973.15, 1173.15, 1373.15, 1573.15]
PRESSURES = [0.7, 1.0, 1.5, 1.9]; HALF = 1.0; STEP = 0.002; CUT = 25.0
C2 = 1.4387769; TREF = 296.0; KB = 1.380649e-16; AMU = 1.66053907e-24; CLIGHT = 2.99792458e10
MASS = {1: 29.997989, 2: 30.995023, 3: 32.002234}               # 14N16O, 15N16O, 14N18O

# ---- HITEMP lines of the window and its 25 cm-1 margins
rows = []
with bz2.open(HITEMP, 'rt') as f:
    for s in f:
        nu = float(s[3:15])
        if WINDOW[0] - CUT <= nu <= WINDOW[1] + CUT:
            rows.append((int(s[2]), nu, float(s[15:25]), float(s[35:40]), float(s[45:55]), float(s[55:59]), float(s[59:67])))
L = np.array(rows); ISO, NU0, S0, GAIR, ELOW, NAIR, DAIR = (L[:, k] for k in range(7)); ISO = ISO.astype(int)
Q296 = {i: hapi.partitionSum(8, i, TREF) for i in (1, 2, 3)}

def line_state(T, P):
    q = np.array([Q296[i] / hapi.partitionSum(8, i, T) for i in ISO])
    S = S0 * q * np.exp(-C2 * ELOW / T) / np.exp(-C2 * ELOW / TREF) * (1 - np.exp(-C2 * NU0 / T)) / (1 - np.exp(-C2 * NU0 / TREF))
    nu = NU0 + DAIR * P; gam = (TREF / T) ** NAIR * GAIR * P
    sig = nu / CLIGHT * np.sqrt(KB * T / (np.array([MASS[i] for i in ISO]) * AMU))
    o = np.argsort(nu)
    return nu[o], S[o], gam[o], sig[o]

def reference(x, T, P, extra_sigma=0.0, near=1.0):
    """Monochromatic HITEMP cross section (cm2 molecule-1) at points x; with extra_sigma, convolved with that Gaussian.
    Voigt (Faddeeva) within `near` cm-1 of a line, Lorentz beyond, nothing beyond 25 cm-1."""
    nu, S, gam, sig = line_state(T, P); sig = np.hypot(sig, extra_sigma); out = np.zeros_like(x)
    for a in range(0, len(x), 256):
        xs = x[a:a + 256]; lo, hi = np.searchsorted(nu, [xs[0] - CUT, xs[-1] + CUT])
        d = xs[None, :] - nu[lo:hi, None]; ad = np.abs(d)
        lor = gam[lo:hi, None] / np.pi / (d ** 2 + gam[lo:hi, None] ** 2)
        val = np.where(ad <= CUT, lor, 0.0)
        m = (np.abs(nu[lo:hi] - xs.mean()) <= near + np.ptp(xs) / 2)
        if m.any():
            z = (d[m] + 1j * gam[lo:hi][m, None]) / (sig[lo:hi][m, None] * np.sqrt(2))
            voi = wofz(z).real / (sig[lo:hi][m, None] * np.sqrt(2 * np.pi))
            val[m] = np.where(ad[m] <= near, voi, val[m])
        out[a:a + 256] = (S[lo:hi, None] * val).sum(0)
    return out

# ---- three PNNL inputs and the released line-resolved branch
spectra = [np.loadtxt(PNNL_DIR / f).T for f in ('NO_5T.TXT', 'NO_25T.TXT', 'NO_50T.TXT')]
model = ThreeSpectrumPredictor(); ctx = model.prepare(spectra, TIN, WINDOW, units='pnnl')
centers = np.array(sorted(r['center_cm1'] for r in ctx['line_records']))
G = [g for g in groups() if g['window'] == 'NO_1600_2050'][0]
input_check = float(np.max(np.abs(ctx['prepared_cold_spectra'] - G['inputs'])) / np.max(G['inputs']))

# check of the line-by-line calculation: 2 cm-1 response at 1 atm against the data-set references (Eq. 4)
ref_check = {}
for k, r in enumerate(G['records']):
    T = r['temperature_K']; m = G['masks'][k].astype(bool)
    mine = reference(G['x'], T, 1.0, extra_sigma=core.SIGMA_CM1, near=8.0)
    ref_check[f'{T:.2f}'] = E(mine, G['references'][k], m)

# ---- scores
seg = np.round(np.arange(-HALF, HALF + STEP / 2, STEP), 6); X = (centers[:, None] + seg[None, :]).ravel()
X = np.clip(X, WINDOW[0], WINDOW[1]); n = len(seg); env_ctx = dict(ctx, line_records=[])
records = []
for P in PRESSURES:
    for T in TEMPS:
        ref = reference(X, T, P)
        full = model.evaluate(ctx, [T], [P], X, output_fwhm_cm1=0.)[0]; env = model.evaluate(env_ctx, [T], [P], X, output_fwhm_cm1=0.)[0]
        for j, c in enumerate(centers):
            sl = slice(j * n, (j + 1) * n); r_, f_, e_ = ref[sl], full[sl], env[sl]; xs = X[sl]
            core_ = np.abs(xs - c) <= 0.05; k = np.flatnonzero(core_)[np.argmax(r_[core_])]
            records.append(dict(T=T, P=P, center=float(c), E_full=100 * float(np.abs(f_ - r_).sum() / np.maximum(r_, 0).sum()),
                                E_env=100 * float(np.abs(e_ - r_).sum() / np.maximum(r_, 0).sum()),
                                center_ratio=float(f_[k] / r_[k]), peak_ratio=float(f_[core_].max() / r_[core_].max()), ref_center=float(r_[k])))
        print(f'{P} atm {T:7.2f} K  E(+-1 cm-1) {np.mean([q["E_full"] for q in records[-len(centers):]]):6.1f}%  envelope alone '
              f'{np.mean([q["E_env"] for q in records[-len(centers):]]):6.1f}%  centre ratio median {np.median([q["center_ratio"] for q in records[-len(centers):]]):.2f}', flush=True)

def agg(sel):
    rr = [q for q in records if sel(q)]
    cr = np.array([q['center_ratio'] for q in rr])
    return dict(n=len(rr), E_full=float(np.mean([q['E_full'] for q in rr])), E_env=float(np.mean([q['E_env'] for q in rr])),
                center_median_abs_dev=float(100 * np.median(np.abs(cr - 1))), center_mean_dev=float(100 * np.mean(cr - 1)),
                center_p10=float(100 * (np.percentile(cr, 10) - 1)), center_p90=float(100 * (np.percentile(cr, 90) - 1)))
BANDS = {'373-773 K': (370, 780), '873-1173 K': (870, 1180), '1373-1573 K': (1370, 1580)}
summary = {f'{P:g} atm': dict(all=agg(lambda q, P=P: q['P'] == P), **{b: agg(lambda q, P=P, lo=lo, hi=hi: q['P'] == P and lo <= q['T'] <= hi) for b, (lo, hi) in BANDS.items()}) for P in PRESSURES}
summary['0.7-1.5 atm'] = agg(lambda q: q['P'] <= 1.5)
res = dict(lines=len(centers), centers_cm1=centers.tolist(), window=WINDOW, temperatures_K=TEMPS, pressures_atm=PRESSURES,
           half_width_cm1=HALF, step_cm1=STEP, input_check_max_rel_dev=input_check, check_2cm1_vs_dataset_E_percent=ref_check,
           hitemp_lines=int(len(L)), summary=summary, records=records)
json.dump(res, open(RESULTS / 'no_native.json', 'w'), indent=1)
print('inputs vs data set: max relative deviation', f'{input_check:.2e}')
print('2 cm-1 HITEMP check against the data set (E, %):', {k: round(v, 2) for k, v in ref_check.items()})
for k, v in summary.items(): print(k, json.dumps(v if 'all' not in v else v['all']))
