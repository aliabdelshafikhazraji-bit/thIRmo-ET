"""Prepare the benzene nu4 window (610-740 cm-1) from the three PNNL benzene spectra with the published 2 cm-1
preparation. The output is read by benzene.py, a14_seed_probe.py and a16_benzene_fig8.py.

The PNNL files BENZENE_5T.TXT, BENZENE_25T.TXT and BENZENE_50T.TXT are not redistributed. Obtain them from PNNL and
give their folder as the first argument or in THIRMO_PNNL (default: <repo>/data/PNNL). The script writes
benzene_inputs.npz, benzene_native.npz and benzene_cold_params.json to the cache folder of config.py."""
import os
from common import *
import physics as ph
PNNL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('THIRMO_PNNL', REPO / 'data' / 'PNNL'))
files = [(278.15, 'BENZENE_5T.TXT'), (298.15, 'BENZENE_25T.TXT'), (323.15, 'BENZENE_50T.TXT')]
W = (610.0, 740.0); x = np.linspace(*W, inference.N); P = []; native = {}
for T, f in files:
    d = np.loadtxt(PNNL / f); xs, ys = inference.unique_sorted(d[:, 0], d[:, 1]); ys = inference.to_cross_section(ys, 'pnnl')
    m = (xs > 560) & (xs < 790); native[T] = (xs[m], ys[m]); P.append(inference.native_smooth(xs[m], ys[m], x)[0])
P = np.array(P); Ts = np.array([t for t, _ in files])
feat, area, anchor, _, _ = inference.make_features(P, Ts, x)
cp = ph.cold_parameters(x, P, Ts); cp['structure'] = ph.structure_indices(x, P[-1])
np.savez(CACHE / 'benzene_inputs.npz', x=x, inputs=P, Tin=Ts, features=feat, area=area, anchor=anchor)
np.savez(CACHE / 'benzene_native.npz', **{f'x{int(T)}': v[0] for T, v in native.items()}, **{f'y{int(T)}': v[1] for T, v in native.items()})
json.dump(cp, open(CACHE / 'benzene_cold_params.json', 'w'), indent=1)
print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in cp.items() if k != 'structure'})
print('shift law (r2, nu_e, rate ratio):', ph.shift_law(cp))
