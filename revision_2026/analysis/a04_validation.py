"""Broader molecule-disjoint validation over the 21 molecules with experimental high-temperature spectra:
leave-one-molecule-out, leave-one-family-out, and repeated random 15/6 molecule-disjoint splits."""
from common import *
import broad as B, time
tr_meas, tr_hitemp, hitran, test = split_groups(); t0 = time.time()
meas = tr_meas + [g for g in test if g['kind'] == 'measurement']; mols = sorted({g['species'] for g in meas}); assert len(mols) == 21 and set(mols) == set(FAMILY)
def score(mdl, held):
    rows = []
    for g in held:
        for i, r in enumerate(g['records']):
            T = r['temperature_K']; m = g['masks'][i]; ref = g['references'][i]
            rows.append(dict(species=g['species'], window=g['window'], temperature_K=T, model=E(B.predict(mdl, g, T), ref, m), physics=E(B.predict(None, g, T), ref, m), unchanged=E(g['warm'], ref, m)))
    return rows
def msum(rows): return {k: summarize([dict(r, error_percent=r[k]) for r in rows]) for k in ('model', 'physics', 'unchanged')}
out = {}
# leave one molecule out
rows = []
for mol in mols:
    mdl = B.fit([([g for g in meas + tr_hitemp if g['species'] != mol], 1.0)]); rows += score(mdl, [g for g in meas if g['species'] == mol])
out['lomo'] = dict(rows=rows, molecule=msum(rows)); print('LOMO done %.0f s' % (time.time() - t0), {k: round(mean_of(v, mols), 2) for k, v in out['lomo']['molecule'].items()}, flush=True)
# leave one family out
rows = []; fams = sorted(set(FAMILY.values()))
for fam in fams:
    held = [g for g in meas if FAMILY[g['species']] == fam]; mdl = B.fit([([g for g in meas + tr_hitemp if FAMILY.get(g['species']) != fam], 1.0)])
    rows += [dict(r, family=fam) for r in score(mdl, held)]
ms = msum(rows); out['lofo'] = dict(rows=rows, molecule=ms, family={fam: {k: mean_of(ms[k], [m for m in mols if FAMILY[m] == fam]) for k in ms} for fam in fams})
print('LOFO done %.0f s' % (time.time() - t0), {k: round(mean_of(v, mols), 2) for k, v in ms.items()}, flush=True)
# repeated random molecule-disjoint splits (15 training / 6 test molecules, as in the fixed partition)
rng = np.random.default_rng(20261119); splits = []
for s in range(30):
    held_m = sorted(rng.choice(mols, 6, replace=False).tolist()); mdl = B.fit([([g for g in meas + tr_hitemp if g['species'] not in held_m], 1.0)])
    ms = msum(score(mdl, [g for g in meas if g['species'] in held_m])); splits.append(dict(test_molecules=held_m, **{k: mean_of(ms[k], held_m) for k in ms}))
    if s % 5 == 4: print('split', s + 1, '%.0f s' % (time.time() - t0), flush=True)
out['random_splits'] = splits
jdump(out, 'validation.json')
a = np.array([[s['model'], s['physics'], s['unchanged']] for s in splits]); print('random splits mean/sd/min/max (model, physics, unchanged):', a.mean(0).round(2), a.std(0, ddof=1).round(2), a.min(0).round(2), a.max(0).round(2))
print('LOMO per molecule (model / physics / unchanged)')
for m in mols: print(f"  {m:20s} {FAMILY[m]:22s} {out['lomo']['molecule']['model'][m]:6.1f} {out['lomo']['molecule']['physics'][m]:6.1f} {out['lomo']['molecule']['unchanged'][m]:6.1f}   LOFO {out['lofo']['molecule']['model'][m]:6.1f}")
print('LOFO family means:', {f: {k: round(v, 1) for k, v in d.items()} for f, d in out['lofo']['family'].items()})
