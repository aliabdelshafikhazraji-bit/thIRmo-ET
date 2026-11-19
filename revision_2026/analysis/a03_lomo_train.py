"""Leave-one-molecule-out over the 15 experimental training molecules (test molecules never used).
Stores out-of-fold predictions with per-tree spread (for interval calibration) and scans the temperature cap
and the weight of the learned correction."""
from common import *
import broad as B, time, pickle
tr_meas, tr_hitemp, hitran, test = split_groups(); mols = sorted({g['species'] for g in tr_meas}); t0 = time.time()
caps = [700.0, 800.0, 900.0, 1000.0, None]; lams = [0.0, 0.5, 0.75, 1.0]; rows = []; oof = []
for mol in mols:
    train = [g for g in tr_meas + tr_hitemp if g['species'] != mol]; mdl = B.fit([(train, 1.0)])
    for g in [g for g in tr_meas if g['species'] == mol]:
        for i, r in enumerate(g['records']):
            T = r['temperature_K']; m = g['masks'][i]; ref = g['references'][i]; row = dict(species=mol, window=g['window'], gi=g['gi'], rec=i, temperature_K=T)
            for cap in caps: row[f'cap_{cap}'] = E(B.predict(mdl, g, T, cap=cap), ref, m)
            for lam in lams: row[f'lam_{lam}'] = E(B.predict(mdl, g, T, weight=lam), ref, m)
            row['physics'] = E(B.predict(None, g, T), ref, m); row['unchanged'] = E(g['warm'], ref, m); rows.append(row)
            trees = B.predict(mdl, g, T, per_tree=True)
            oof.append(dict(species=mol, window=g['window'], gi=g['gi'], rec=i, T=T, x=g['x'], mask=m, ref=ref, pred=B.predict(mdl, g, T), phys=B.predict(None, g, T),
                            tree_sd=trees.std(0), q05=np.quantile(trees, .05, 0), q95=np.quantile(trees, .95, 0), q025=np.quantile(trees, .025, 0), q975=np.quantile(trees, .975, 0)))
    print(mol, '%.0f s' % (time.time() - t0), flush=True)
res = {}
for k in [f'cap_{c}' for c in caps] + [f'lam_{l}' for l in lams] + ['physics', 'unchanged']:
    s = summarize([dict(r, error_percent=r[k]) for r in rows]); res[k] = dict(mean=mean_of(s, mols), per_molecule=s)
jdump(dict(rows=rows, summary=res, molecules=mols), 'lomo_training.json'); (CACHE / 'lomo_training_oof.pkl').write_bytes(pickle.dumps(oof))
print({k: round(v['mean'], 2) for k, v in res.items()})
hi = [r for r in rows if r['temperature_K'] > 800]
print('records above 800 K:', len(hi), {k: round(float(np.mean([r[k] for r in hi])), 2) for k in [f'cap_{c}' for c in caps] + ['physics']})
