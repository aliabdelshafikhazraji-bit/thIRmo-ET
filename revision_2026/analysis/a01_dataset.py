"""Dataset accounting (Table 1) and per-window source audit (SM Appendix) from the prepared dataset manifest."""
from common import *
import collections
G = groups(); out = {}; t1 = []
for split, kind, label in [('training', 'measurement', 'Training / Experiment'), ('training', 'HITEMP', 'Training / HITEMP'), ('testing', 'measurement', 'Testing / Experiment'), ('testing', 'HITEMP', 'Testing / HITEMP'), ('hitran', 'HITRAN', 'Training / HITRAN')]:
    sel = [g for g in G if g['split'] == split and g['kind'] == kind]; recs = [r for g in sel for r in g['records']]
    assert all(r['source_kind'] == kind for r in recs)
    t1.append(dict(row=label, molecules=len({g['species'] for g in sel}), windows=len(sel), records=len(recs), Tmin=min(r['temperature_K'] for r in recs), Tmax=max(r['temperature_K'] for r in recs),
                   Pmin=min(r['pressure_atm'] for r in recs), Pmax=max(r['pressure_atm'] for r in recs), bath=dict(collections.Counter(str(r.get('bath_gas')) for r in recs)), origin=dict(collections.Counter(r.get('data_origin') for r in recs))))
out['table1'] = t1
train_ids = {g['species'] for g in G if g['split'] in ('training', 'hitran')}; test_ids = {g['species'] for g in G if g['split'] == 'testing'}
orig_ids = {g['species'] for g in G if g['split'] == 'training'}; hit_ids = {g['species'] for g in G if g['split'] == 'hitran'}
out['identities'] = dict(training=len(train_ids), testing=len(test_ids), overlap=sorted(train_ids & test_ids), original_training=len(orig_ids), hitran=len(hit_ids), hitran_new=len(hit_ids - orig_ids), in_both_collections=sorted(orig_ids & hit_ids))
BG = {'Ar': 'Ar', 'argon': 'Ar', 'N2': 'N2', 'nitrogen': 'N2', 'air': 'air', 'not verified from accessible workbook': 'not stated in data file'}
aud = []
for g in sorted(G, key=lambda g: (g['split'] != 'testing', g['kind'] != 'measurement', g['species'].lower(), g['x'][0])):
    recs = g['records']; Ts = [r['temperature_K'] for r in recs]; Ps = [r['pressure_atm'] for r in recs]
    aud.append(dict(species=g['species'], role='Testing' if g['split'] == 'testing' else 'Training', reference={'measurement': 'Experiment', 'HITEMP': 'HITEMP', 'HITRAN': 'HITRAN'}[g['kind']], window=[float(g['x'][0]), float(g['x'][-1])], records=len(recs),
                    Tmin=min(Ts), Tmax=max(Ts), Pmin=min(Ps), Pmax=max(Ps), bath=sorted({BG.get(str(r.get('bath_gas')), str(r.get('bath_gas'))) for r in recs}), origin=dict(collections.Counter(r.get('data_origin') for r in recs)),
                    doi=sorted({str(r.get('source_doi') or r.get('source')) for r in recs}), mask_fraction=float(np.mean([m.mean() for m in g['masks']])), input_T=[float(t) for t in g['Tin']]))
out['audit'] = aud
dig = [(g['species'], r['record_id'], r['temperature_K'], r.get('source')) for g in G for r in g['records'] if r.get('data_origin') == 'digitized']; out['digitized_records'] = dig
jdump(out, 'dataset.json')
for r in t1: print(r['row'], r['molecules'], r['windows'], r['records'], round(r['Tmin'], 2), round(r['Tmax'], 2), round(r['Pmin'], 2), round(r['Pmax'], 2), r['bath'], r['origin'])
print(out['identities']); print('digitized:', len(dig)); [print('  ', d) for d in dig]
env = json.load(open(REPO / 'study/environment.json')); print(env)
