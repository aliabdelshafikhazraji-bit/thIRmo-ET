"""thIRmo-ET, broad-band branch: hot band envelope from three near-ambient spectra (manuscript Section 2.4).

Stage 1 extrapolates the band shift and broadening observed between the three inputs with conserved band area
(physics.py, Eqs. 3a-3c). Stage 2 adds a correction learned from other molecules by 128 extremely randomized
trees with 16 predictors that contain no absolute wavenumber. Above 1000 K the correction is not updated: the
spectrum predicted at 1000 K is carried to the requested temperature by the additional physical shift and
broadening. Output: cross section in cm2/molecule (Napierian) at a 2 cm-1 Gaussian response on a 512-point grid.

Inference needs NumPy and SciPy only; no molecular identity, mass, line list, or hot spectrum is read.

    python broad_band.py --spectrum cold.txt 278.15 --spectrum room.txt 298.15 --spectrum warm.txt 323.15 \
        --units pnnl --window 610 740 --temperature 700 800 900 --output hot_spectra.csv

Use this branch for molecules whose three spectra retain no qualifying isolated lines (cold_router.py); molecules
with retained lines (for example NO and CH4) use the line-resolved branch in spectral_model.py.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path: sys.path.insert(0, _p)
import physics as ph                                   # noqa: E402
from broad_model import frozen_core as core            # noqa: E402  (published 2 cm-1 preparation of the inputs)

VERSION = '5.0.0'
N_PREDICTORS = 16
T_CAP = 1000.0          # K, temperature above which the learned correction is no longer updated
Q_INTERVAL = 0.982      # multiplier of the calibrated 90% interval (supplementary Appendix G; 0.98 to two decimals)
BETA_INTERVAL = 0.02    # floor of the interval scale, as a fraction of the predicted band maximum
DEFAULT_MODEL = HERE / 'models' / 'thirmo_et_broad_band.npz'


class Forest:
    """Extremely randomized regression trees stored as numeric arrays (same layout as the published envelope)."""
    def __init__(self, path=DEFAULT_MODEL):
        path = Path(path); self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        if self.sha256 != self.metadata['weights_sha256']: raise ValueError('Model weights do not match the recorded SHA-256.')
        with np.load(path, allow_pickle=False) as z:
            self.offsets, self.feature, self.threshold = z['offsets'], z['feature'], z['threshold']
            self.left, self.right, self.value = z['left'], z['right'], z['value']
        size = len(self.feature)
        if not (self.offsets[0] == 0 and self.offsets[-1] == size and np.all(np.diff(self.offsets) > 0)): raise ValueError('Invalid tree offsets.')
        if np.any(self.feature >= N_PREDICTORS) or not np.all(np.isfinite(self.value)): raise ValueError('Invalid model arrays.')

    def predict_trees(self, X):
        """Per-tree predictions, shape (n_trees, n_rows)."""
        X = np.asarray(X, dtype='f4')
        if X.ndim != 2 or X.shape[1] != N_PREDICTORS or not np.all(np.isfinite(X)): raise ValueError('Prediction requires finite 16-column predictors.')
        rows = np.arange(len(X)); out = np.empty((len(self.offsets) - 1, len(X)))
        for k, start in enumerate(self.offsets[:-1]):
            nodes = np.zeros(len(X), dtype=np.int64)
            for _ in range(64):
                feats = self.feature[start + nodes]; active = feats >= 0
                if not np.any(active): break
                cur = start + nodes[active]
                nodes[active] = np.where(X[rows[active], feats[active]] <= self.threshold[cur], self.left[cur], self.right[cur])
            out[k] = self.value[start + nodes]
        return out

    def predict(self, X): return self.predict_trees(X).mean(0)


def prepare(spectra, temperatures_K, window_cm1, units='pnnl'):
    """spectra: three (wavenumber, value) arrays; units 'pnnl' (decadic absorbance at 1 ppm m) or 'cm2/molecule'.
    Returns the window on the 512-point grid at the 2 cm-1 response with its near-ambient shift and broadening fits."""
    T = np.asarray(temperatures_K, float); lo, hi = map(float, window_cm1)
    if len(spectra) != 3 or T.shape != (3,) or len(np.unique(T)) != 3 or np.any((T < 275) | (T > 350)):
        raise ValueError('Exactly three spectra at distinct temperatures within 275-350 K are required.')
    if not 0 < lo < hi: raise ValueError('Two increasing positive window boundaries are required.')
    canonical = 'pnnl' if units == 'pnnl' else 'cross-section'
    if units not in ('pnnl', 'cm2/molecule', 'cm2-molecule', 'cross-section'): raise ValueError('Units must be pnnl or cm2/molecule.')
    order = np.argsort(T); x = np.linspace(lo, hi, core.N); S = []
    for i in order:
        xs, ys = core.unique_sorted(*spectra[i])
        if xs[0] > lo or xs[-1] < hi: raise ValueError('Every input spectrum must cover the full window.')
        S.append(core.native_smooth(xs, core.to_cross_section(ys, canonical), x)[0])
    S = np.asarray(S); T = T[order]; area = float(np.trapezoid(np.maximum(S[-1], 0), x))
    if area <= 0: raise ValueError('The warmest input must have positive integrated absorption in the window.')
    W = float(x[-1] - x[0]); g = dict(x=x, inputs=S, Tin=T, warm=S[-1], area=area, width=W, anchor=S[-1] * W / area)
    g['cp'] = ph.cold_parameters(x, S, T); return g


def _shape(g):
    if 'shape' not in g:
        x = g['x']; dx = x[1] - x[0]; W = g['width']; fw = g['anchor']; fc = g['inputs'][0] * W / g['area']
        w = np.maximum(g['warm'], 0); c = (x * w).sum() / w.sum(); r = np.sqrt((w * (x - c) ** 2).sum() / w.sum())
        g['shape'] = dict(cols=[fw, (fw - fc) / (g['Tin'][-1] - g['Tin'][0]) * 100, (x - c) / r, (x - x[0]) / W, np.gradient(fw, dx) * r,
                                np.gradient(np.gradient(fw, dx), dx) * r * r, fw - gaussian_filter1d(fw, 4.0 / dx, mode='nearest')], r=float(r))
    return g['shape']


def predictors(g, T):
    """Predictor matrix (512, 16) and the normalized physical reference of Eq. (3c) at temperature T."""
    cp = g['cp']; s = _shape(g); n = len(g['x']); T1, T2, T3 = cp['T']
    fr = ph.reference_spectrum(g['x'], g['warm'], cp, T) * g['width'] / g['area']
    eta = -cp['d13'] / (T3 - T1) * 100; lam = cp['w13'] ** 2 / (T3 - T1)
    cols = s['cols'] + [np.full(n, T / 1000), np.full(n, (T - T3) / 500), np.full(n, 1000 / T3 - 1000 / T), fr, fr - g['anchor']] + [np.full(n, v) for v in (s['r'], eta, lam, g['width'] / 1000)]
    return np.column_stack(cols).astype('f4'), fr


class BroadBandPredictor:
    def __init__(self, model_path=DEFAULT_MODEL): self.forest = Forest(model_path)

    def predict(self, g, T, physical_stage_only=False, with_interval=False):
        """Cross section (cm2/molecule) on g['x'] at temperature T (K). With with_interval=True also returns the
        lower and upper bounds of the calibrated 90% prediction interval."""
        T = float(T)
        if T <= g['Tin'][-1]: raise ValueError('The requested temperature must exceed the warmest input.')
        cp = g['cp']; Tq = min(T, T_CAP); scale = g['area'] / g['width']; X, fr = predictors(g, Tq)
        if physical_stage_only: trees = None; y = np.maximum(fr, 0) * scale
        else:
            trees = np.maximum(fr[None, :] + self.forest.predict_trees(X), 0) * scale; y = np.maximum(fr + self.forest.predict_trees(X).mean(0), 0) * scale
        def carry(a):
            if T <= T_CAP: return a
            dd = ph.shift(cp, T) - ph.shift(cp, T_CAP); dw = float(np.sqrt(max(ph.broadening(cp, T) ** 2 - ph.broadening(cp, T_CAP) ** 2, 0)))
            return ph.deform(g['x'], a, dd, dw)
        y = carry(y)
        if not with_interval or trees is None: return y
        sd = np.asarray([carry(t) for t in trees]).std(0); half = Q_INTERVAL * (sd + BETA_INTERVAL * y.max())
        return y, np.maximum(y - half, 0), y + half


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--spectrum', nargs=2, action='append', required=True, metavar=('FILE', 'T_K'), help='two-column file (wavenumber, value) and its temperature; give three')
    p.add_argument('--units', default='pnnl', choices=['pnnl', 'cm2/molecule'])
    p.add_argument('--window', nargs=2, type=float, required=True, metavar=('LOW', 'HIGH'))
    p.add_argument('--temperature', nargs='+', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    g = prepare([core.read_spectrum(f) for f, _ in a.spectrum], [float(t) for _, t in a.spectrum], a.window, a.units); model = BroadBandPredictor()
    rows = []
    for T in a.temperature:
        y, lo, hi = model.predict(g, T, with_interval=True)
        rows += [(T, x, v, l, h) for x, v, l, h in zip(g['x'], y, lo, hi)]
    np.savetxt(a.output, np.asarray(rows), delimiter=',', header='temperature_K,wavenumber_cm1,cross_section_cm2_molecule,interval90_low,interval90_high', comments='')
    r2, nue, rho = ph.shift_law(g['cp'])
    meta = dict(tool='thIRmo-ET', branch='broad_band', version=VERSION, weights_sha256=model.forest.sha256, input_temperatures_K=g['Tin'].tolist(), window_cm1=list(a.window),
                cold_shift_cm1=dict(first_interval=g['cp']['d12'], second_interval=g['cp']['d23']), rate_ratio=None if rho != rho else rho, effective_wavenumber_cm1=nue, cold_broadening_cm1=g['cp']['w13'],
                correction_updated_up_to_K=T_CAP, response_fwhm_cm1=core.FWHM, validated_range='experimental test spectra up to 974 K; higher temperatures are extrapolations')
    a.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf8'); print(a.output)


if __name__ == '__main__': main()
