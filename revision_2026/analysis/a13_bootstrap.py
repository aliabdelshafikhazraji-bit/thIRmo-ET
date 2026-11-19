"""Molecule-level uncertainty of the reported means (percentile bootstrap over molecules, 20000 resamples)."""
from common import *
main = jload('main.json'); val = jload('validation.json'); rng = np.random.default_rng(7)
def ci(v, n=20000):
    v = np.asarray(v, float); m = v[rng.integers(0, len(v), (n, len(v)))].mean(1); return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), lo=float(np.quantile(m, .025)), hi=float(np.quantile(m, .975)), n=len(v))
out = {}
for k in ('model', 'physics', 'unchanged', 'published_final'):
    out['fixed_experiment6_' + k] = ci([main['molecule'][k][s] for s in MEAS]); out['fixed_all8_' + k] = ci([main['molecule'][k][s] for s in ORDER])
out['fixed_experiment6_model_minus_physics'] = ci([main['molecule']['model'][s] - main['molecule']['physics'][s] for s in MEAS])
out['fixed_experiment6_model_minus_published'] = ci([main['molecule']['model'][s] - main['molecule']['published_final'][s] for s in MEAS])
out['fixed_ethers_minus_others'] = dict(ethers=float(np.mean([main['molecule']['model'][s] for s in ETHERS])), others=float(np.mean([main['molecule']['model'][s] for s in OTHERS])))
mols = sorted(val['lomo']['molecule']['model'])
for name in ('lomo', 'lofo'):
    for k in ('model', 'physics', 'unchanged'): out[f'{name}21_{k}'] = ci([val[name]['molecule'][k][m] for m in mols])
    d = [val[name]['molecule']['model'][m] - val[name]['molecule']['physics'][m] for m in mols]; out[f'{name}21_model_minus_physics'] = dict(ci(d), improved=int(sum(x < 0 for x in d)), worsened=int(sum(x > 0 for x in d)))
    out[f'{name}21_median_model'] = float(np.median([val[name]['molecule']['model'][m] for m in mols])); out[f'{name}21_median_physics'] = float(np.median([val[name]['molecule']['physics'][m] for m in mols]))
sp = val['random_splits']; a = np.array([[s['model'], s['physics'], s['unchanged']] for s in sp])
out['random_splits'] = dict(n=len(sp), model=dict(mean=float(a[:, 0].mean()), sd=float(a[:, 0].std(ddof=1)), min=float(a[:, 0].min()), max=float(a[:, 0].max()), q10=float(np.quantile(a[:, 0], .1)), q90=float(np.quantile(a[:, 0], .9))),
                            physics=dict(mean=float(a[:, 1].mean()), sd=float(a[:, 1].std(ddof=1)), min=float(a[:, 1].min()), max=float(a[:, 1].max())), unchanged=dict(mean=float(a[:, 2].mean()), sd=float(a[:, 2].std(ddof=1))),
                            model_better_than_physics=int((a[:, 0] < a[:, 1]).sum()), fixed_partition_percentile_model=float((a[:, 0] < main['groups']['model']['experiment6']).mean() * 100))
jdump(out, 'bootstrap.json')
for k, v in out.items(): print(k, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else round(v, 2))
