"""Main benchmark of the revised manuscript: the fixed 33/8 molecular partition, 84 test records.
Broad-band molecules: physical reference + forest (broad.py). NO and CH4: published line-resolved envelope."""
from common import *
import broad as B, linebranch as L, baselines as BL, physics as ph
tr_meas, tr_hitemp, hitran, test = split_groups(); mdl = B.main_model(); pub_rows = {r['record_id'] + '|' + r['window']: r for r in published_rows()}
rows = []; preds = {}
for g in test:
    lr = g['species'] in LINE_RESOLVED; wcoef, _ = BL.wakatsuki_fit(g['x'], g['inputs'], g['Tin'])
    for i, r in enumerate(g['records']):
        if not r.get('primary_evaluation', True): continue
        T = r['temperature_K']; m = g['masks'][i]; ref = g['references'][i]; p = pub_rows[r['record_id'] + '|' + g['window']]
        assert abs(p['temperature_K'] - T) < 1e-6 and abs(p['unchanged_error_percent'] - E(g['warm'], ref, m)) < 1e-6
        row = dict(species=g['species'], window=g['window'], gi=g['gi'], record_id=r['record_id'], temperature_K=T, pressure_atm=r['pressure_atm'], bath_gas=r.get('bath_gas'), source=r.get('source'),
                   source_kind=r['source_kind'], unchanged=E(g['warm'], ref, m), linear=E(BL.linear(g, T), ref, m), exponential=E(BL.exponential(g, T), ref, m), wakatsuki=E(BL.wakatsuki_predict(g['x'], wcoef, T), ref, m),
                   published_final=p['final_error_percent'], published_without_hitran=p['without_hitran_error_percent'])
        if lr:
            y = L.predict(g, T); row.update(model=p['final_error_percent'], physics=row['unchanged'], model_dataset_inputs=E(y, ref, m), branch='line-resolved')
        else:
            y = B.predict(mdl, g, T); yp = B.predict(None, g, T); row.update(model=E(y, ref, m), physics=E(yp, ref, m), branch='broad-band')
            preds[(g['gi'], i, 'physics')] = yp
        preds[(g['gi'], i, 'model')] = y; rows.append(row)
assert len(rows) == 84
def table(key, sel=None):
    rr = [dict(r, error_percent=r[key]) for r in rows if sel is None or sel(r)]; return summarize(rr)
keys = ['unchanged', 'linear', 'exponential', 'wakatsuki', 'physics', 'model', 'published_final', 'published_without_hitran']
mol = {k: table(k) for k in keys}
groupsum = {k: dict(all8=mean_of(mol[k], ORDER), experiment6=mean_of(mol[k], MEAS), hitemp2=mean_of(mol[k], LINE_RESOLVED), ethers3=mean_of(mol[k], ETHERS), others3=mean_of(mol[k], OTHERS)) for k in keys}
bins = [(323.15, 500), (500, 700), (700, 900), (900, 1100), (1100, 1300), (1300, 1600)]; tb = []
for kind, label in (('measurement', 'PNNL / experiment'), ('HITEMP', 'PNNL / HITEMP')):
    for lo, hi in bins:
        sel = [r for r in rows if r['source_kind'] == kind and lo < r['temperature_K'] <= hi]
        if not sel: continue
        e = dict(group=label, lo=lo, hi=hi, molecules=len({r['species'] for r in sel}), records=len(sel))
        for k in keys:
            s = summarize([dict(r, error_percent=r[k]) for r in sel]); e[k] = float(np.mean(list(s.values()))); e[k + '_record_mean'] = float(np.mean([r[k] for r in sel]))
        tb.append(e)
# Table 4 style info
t4 = []
for s in ORDER:
    rr = [r for r in rows if r['species'] == s]; g0 = next(g for g in test if g['species'] == s)
    t4.append(dict(species=s, input_T=[round(float(t), 2) for t in g0['Tin']], reference='Experiment' if rr[0]['source_kind'] == 'measurement' else 'HITEMP', windows=len({r['window'] for r in rr}), records=len(rr),
                   Tmin=min(r['temperature_K'] for r in rr), Tmax=max(r['temperature_K'] for r in rr), Pmin=min(r['pressure_atm'] for r in rr), Pmax=max(r['pressure_atm'] for r in rr), **{k: mol[k][s] for k in keys}))
jdump(dict(rows=rows, molecule=mol, groups=groupsum, temperature_bins=tb, table4=t4), 'main.json')
import pickle; (CACHE / 'main_preds.pkl').write_bytes(pickle.dumps(preds))
print('molecule-mean errors (%)'); print(f"{'':22s}" + ''.join(f'{k[:12]:>13s}' for k in keys))
for s in ORDER: print(f'{s:22s}' + ''.join(f'{mol[k][s]:13.2f}' for k in keys))
for gk in ('experiment6', 'hitemp2', 'all8', 'ethers3', 'others3'): print(f'{gk:22s}' + ''.join(f'{groupsum[k][gk]:13.2f}' for k in keys))
print('temperature bins (molecule-mean within bin)')
for e in tb: print(f"{e['group']:18s} ({e['lo']:.0f},{e['hi']:.0f}] mol {e['molecules']} rec {e['records']:2d} " + ' '.join(f"{k[:6]} {e[k]:7.2f}" for k in keys[:6]))
