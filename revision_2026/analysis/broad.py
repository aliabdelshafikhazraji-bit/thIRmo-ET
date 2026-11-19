"""Broad-band branch of thIRmo-ET (revised manuscript, Section 2.4).

Stage 1 (physics.py): the band shift and broadening fitted between the three near-ambient inputs are
extrapolated to the target temperature with conserved band area, Eqs. (3a)-(3c).
Stage 2 (this file): an ensemble of extremely randomized trees, fitted on other molecules, predicts the
remaining change from that physical reference. Tree settings and per-identity weights are the published ones."""
from common import *
import physics as ph
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor

T_CAP = 1000.0                      # K; the learned correction is not extrapolated above this temperature
SEED = int(pub.PARAMETERS['random_state'])
PARAMETERS = dict(pub.PARAMETERS)   # 128 trees, depth 16, >=16 samples per leaf, 80% of predictors per split
FEATURE_NAMES = ['warm_input', 'cold_to_warm_slope_per_100K', 'relative_wavenumber_rms', 'window_position', 'first_derivative_scaled',
                 'second_derivative_scaled', 'local_contrast_4cm', 'target_T_per_1000K', 'delta_T_per_500K', 'inverse_T_difference',
                 'physical_reference', 'reference_minus_warm', 'rms_width_cm1', 'cold_shift_rate_per_100K', 'cold_broadening_rate', 'window_width_per_1000cm']

_CP = None
def cold_params(g):
    """Pairwise shift/broadening fits between the three inputs of a window (cached for dataset windows)."""
    global _CP
    if 'cp' in g: return g['cp']
    if _CP is None:
        f = CACHE / 'cold_params.json'
        if not f.exists():
            out = {}
            for h in groups(): out[str(h['gi'])] = dict(ph.cold_parameters(h['x'], h['inputs'], h['Tin']), species=h['species'], window=h['window'], split=h['split'], kind=h['kind'], window_cm1=[float(h['x'][0]), float(h['x'][-1])])
            f.write_text(json.dumps(out, indent=1))
        _CP = {int(k): v for k, v in json.loads(f.read_text()).items()}
    g['cp'] = _CP[g['gi']] if 'gi' in g else ph.cold_parameters(g['x'], g['inputs'], g['Tin'])
    return g['cp']

def prepare(x, inputs, temperatures):
    """Build a window dictionary from three spectra (cm2/molecule, 2 cm-1 response) on a uniform grid."""
    x = np.asarray(x, float); S = np.asarray(inputs, float); T = np.asarray(temperatures, float); o = np.argsort(T); S, T = S[o], T[o]
    W = np.ptp(x); area = float(np.trapezoid(np.maximum(S[-1], 0), x))
    g = dict(x=x, inputs=S, Tin=T, warm=S[-1], area=area, anchor=S[-1] * W / area)
    return g

def shape(g):
    """Position-free descriptors of the normalized warmest input (computed once per window)."""
    if 'shape' not in g:
        x = g['x']; dx = x[1] - x[0]; W = np.ptp(x); fw = g['anchor']; fc = g['inputs'][0] * W / g['area']
        w = np.maximum(g['warm'], 0); c = (x * w).sum() / w.sum(); r = np.sqrt((w * (x - c) ** 2).sum() / w.sum())
        g['shape'] = dict(cols=[fw, (fw - fc) / (g['Tin'][-1] - g['Tin'][0]) * 100, (x - c) / r, (x - x[0]) / W, np.gradient(fw, dx) * r,
                                np.gradient(np.gradient(fw, dx), dx) * r * r, fw - gaussian_filter1d(fw, 4.0 / dx, mode='nearest')], r=float(r), c=float(c), W=float(W))
    return g['shape']

def reference(g, T):
    """Normalized physical reference f_ref(nu, T) = sigma_ref W / S, Eq. (3c)."""
    return ph.reference_spectrum(g['x'], g['warm'], cold_params(g), T) * np.ptp(g['x']) / g['area']

def predictors(g, T):
    """Predictor matrix (n_points, 16) and normalized physical reference at target temperature T."""
    cp = cold_params(g); s = shape(g); n = len(g['x']); T1, T2, T3 = cp['T']; fr = reference(g, T)
    eta = -cp['d13'] / (T3 - T1) * 100; lam = cp['w13'] ** 2 / (T3 - T1)
    cols = s['cols'] + [np.full(n, T / 1000), np.full(n, (T - T3) / 500), np.full(n, 1000 / T3 - 1000 / T), fr, fr - g['anchor']] + [np.full(n, v) for v in (s['r'], eta, lam, s['W'] / 1000)]
    return np.column_stack(cols).astype('f4'), fr

def training_arrays(blocks):
    """blocks: list of (windows, total weight per molecular identity). Published weighting: each identity's weight is
    shared equally among its windows, source kinds, target records and valid wavenumber positions (Eq. 6)."""
    X = []; y = []; w = []
    for selected, total in blocks:
        if total == 0 or not selected: continue
        bands = {s: sum(g['species'] == s for g in selected) for s in {g['species'] for g in selected}}
        for g in selected:
            domains = {r['source_kind'] for r in g['records']}; counts = {d: sum(r['source_kind'] == d for r in g['records']) for d in domains}
            for i, r in enumerate(g['records']):
                m = g['masks'][i]; n = int(m.sum()); Xi, fr = predictors(g, r['temperature_K'])
                X.append(Xi[m]); y.append(g['references'][i, m] * np.ptp(g['x']) / g['area'] - fr[m])
                w.extend([total / bands[g['species']] / len(domains) / counts[r['source_kind']] / n] * n)
    w = np.asarray(w); w *= len(w) / w.sum(); return np.concatenate(X), np.concatenate(y), w

def fit(blocks, seed=SEED, n_jobs=2):
    X, y, w = training_arrays(blocks); p = dict(PARAMETERS, random_state=seed, n_jobs=n_jobs)
    return ExtraTreesRegressor(**p).fit(X, y, sample_weight=w)

def predict(model, g, T, per_tree=False, cap=T_CAP, weight=1.0):
    """Band-envelope cross section (cm2/molecule) at temperature T. model=None returns the physical stage alone.
    Above the cap, the spectrum predicted at the cap is carried to T by the additional physical shift and
    broadening, so the band area predicted at the cap is conserved."""
    cp = cold_params(g); Tq = min(T, cap) if cap else T; scale = g['area'] / np.ptp(g['x']); X, fr = predictors(g, Tq)
    if model is None: out = np.maximum(fr, 0) * scale
    elif per_tree: out = np.maximum(fr[None, :] + weight * np.asarray([e.predict(X) for e in model.estimators_]), 0) * scale
    else: out = np.maximum(fr + weight * model.predict(X), 0) * scale
    if cap and T > cap:
        dd = ph.shift(cp, T) - ph.shift(cp, cap); dw = float(np.sqrt(max(ph.broadening(cp, T) ** 2 - ph.broadening(cp, cap) ** 2, 0)))
        out = np.asarray([ph.deform(g['x'], o, dd, dw) for o in out]) if out.ndim == 2 else ph.deform(g['x'], out, dd, dw)
    return out

def main_blocks():
    tr_meas, tr_hitemp, hitran, test = split_groups(); return [(tr_meas + tr_hitemp, 1.0)]

_MAIN = None
def main_model():
    """The broad-band forest of the revised manuscript: experimental + HITEMP training windows, published seed."""
    global _MAIN
    if _MAIN is None:
        import pickle; f = CACHE / 'broad_forest.pkl'
        if f.exists(): _MAIN = pickle.loads(f.read_bytes())
        else: _MAIN = fit(main_blocks()); f.write_bytes(pickle.dumps(_MAIN))
    return _MAIN
