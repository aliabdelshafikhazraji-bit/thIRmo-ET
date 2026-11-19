"""Check that the broad-band weights and the NumPy-only inference reproduce the manuscript values.

    python three_input/broad_band/verify_release.py --dataset <prepared data set folder>

Recomputes the molecule-mean errors (Eq. 4) of the six test molecules with experimental references (Table 3 of the
manuscript) from the prepared inputs of the data set (group_*.npz files and dataset_manifest.json, built with
reproduction/pipeline.py from the licensed source spectra). Expected: 1,2-dimethoxyethane 17.83, di-n-propyl ether
15.64, diisopropyl ether 15.39, n-propyl acetate 11.08, toluene 12.61, acetone 12.87; mean 14.24 %."""
import argparse, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
p = argparse.ArgumentParser(); p.add_argument('--dataset', type=Path, required=True); a = p.parse_args()
for q in (str(HERE), str(REPO / 'three_input'), str(REPO)):
    if q not in sys.path: sys.path.insert(0, q)
import broad_band as bb, physics as ph                     # noqa: E402
from reproduction.pipeline import load_dataset             # noqa: E402
model = bb.BroadBandPredictor(); per = {}
for g in load_dataset(a.dataset):
    if g['split'] != 'testing' or g['records'][0]['source_kind'] != 'measurement': continue
    x = g['x']; S = g['inputs']; T = np.asarray(g['input_temperatures_K'], float); area = float(np.trapezoid(np.maximum(S[-1], 0), x)); W = float(x[-1] - x[0])
    w = dict(x=x, inputs=S, Tin=T, warm=S[-1], area=area, width=W, anchor=S[-1] * W / area); w['cp'] = ph.cold_parameters(x, S, T); errs = []
    for i, r in enumerate(g['records']):
        m = g['masks'][i]; ref = g['references'][i]; y = model.predict(w, r['temperature_K']); errs.append(100 * np.abs(y[m] - ref[m]).sum() / np.maximum(ref[m], 0).sum())
    per.setdefault(g['species'], []).append(float(np.mean(errs)))
mol = {s: float(np.mean(v)) for s, v in per.items()}
for s, v in mol.items(): print(f'{s:22s} {v:6.2f} %')
print(f"{'mean of six molecules':22s} {np.mean(list(mol.values())):6.2f} %")
