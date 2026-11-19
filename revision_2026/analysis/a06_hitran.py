"""Role of the added HITRAN training collection, weight sensitivity, paired uncertainty, and seed variability."""
from common import *
import broad as B, linebranch as L, time
from sklearn.ensemble import ExtraTreesRegressor
tr_meas, tr_hitemp, hitran, test = split_groups(); t0 = time.time(); out = {}
def boot_ci(d, n=20000, seed=1):
    d = np.asarray(d, float); rng = np.random.default_rng(seed); m = d[rng.integers(0, len(d), (n, len(d)))].mean(1); return [float(np.quantile(m, .025)), float(np.quantile(m, .975))]
# ---- (0) original submission: paired difference with/without the HITRAN block (published per-record errors)
pr = published_rows(); w = summarize(pr, 'final_error_percent'); wo = summarize(pr, 'without_hitran_error_percent'); d = [w[s] - wo[s] for s in ORDER]
out['original_submission'] = dict(with_hitran=w, without_hitran=wo, difference={s: w[s] - wo[s] for s in ORDER}, mean_difference=float(np.mean(d)), ci95=boot_ci(d), n_increase=int(sum(x > 0 for x in d)), n_decrease=int(sum(x < 0 for x in d)),
                                  exact_sign_flip_p=float(np.mean([abs(np.mean(np.array(d) * s)) >= abs(np.mean(d)) - 1e-12 for s in np.array(np.meshgrid(*[[-1, 1]] * 8)).T.reshape(-1, 8)])))
print('original: mean diff %.2f CI %s, sign-flip p %.3f' % (out['original_submission']['mean_difference'], np.round(out['original_submission']['ci95'], 2), out['original_submission']['exact_sign_flip_p']), flush=True)
# ---- (1) broad-band forest: HITRAN weight scan (0 = revised thIRmo-ET)
def score_broad(mdl, sel):
    rows = []
    for g in sel:
        for i, r in enumerate(g['records']): rows.append(dict(species=g['species'], window=g['window'], error_percent=E(B.predict(mdl, g, r['temperature_K']), g['references'][i], g['masks'][i])))
    return summarize(rows)
bb = {}
for a in (0.0, 0.25, 0.5, 1.0):
    mdl = B.main_model() if a == 0 else B.fit([(tr_meas + tr_hitemp, 1.0), (hitran, a)]); s = score_broad(mdl, test); bb[str(a)] = dict(molecule=s, experiment6=mean_of(s, MEAS))
    print('broad alpha', a, {k: round(s[k], 2) for k in ORDER}, round(mean_of(s, MEAS), 2), '%.0f s' % (time.time() - t0), flush=True)
d = [bb['0.5']['molecule'][s] - bb['0.0']['molecule'][s] for s in MEAS]; out['broad_weight_scan'] = bb; out['broad_paired_0.5_vs_0'] = dict(difference=dict(zip(MEAS, d)), mean=float(np.mean(d)), ci95=boot_ci(d))
# ---- (2) line-resolved envelope: published pipeline refitted at several HITRAN weights (dataset inputs)
def fit_pub(G, a, seed=None):
    X, y, wts, _ = pub.training_arrays(G, a); p = dict(pub.PARAMETERS, n_jobs=2)
    if seed is not None: p['random_state'] = seed
    return ExtraTreesRegressor(**p).fit(X, y, sample_weight=wts)
def score_pub(mdl, sel):
    rows = []
    for g in sel:
        for i, r in enumerate(g['records']):
            X = inference.features_at(g['features'], g['Tin'][-1], r['temperature_K']); y = np.maximum((g['anchor'] + mdl.predict(X)) * g['area'] / np.ptp(g['x']), 0)
            rows.append(dict(species=g['species'], window=g['window'], error_percent=E(y, g['references'][i], g['masks'][i])))
    return summarize(rows)
G = groups(); lr = {}
for a in (0.0, 0.25, 0.5, 1.0):
    s = score_pub(fit_pub(G, a), test); lr[str(a)] = dict(molecule=s, hitemp2=mean_of(s, LINE_RESOLVED), experiment6=mean_of(s, MEAS))
    print('line-resolved alpha', a, {k: round(s[k], 2) for k in ORDER}, '%.0f s' % (time.time() - t0), flush=True)
fr = {}
for name in ('final', 'without'):
    rows = [dict(species=g['species'], window=g['window'], error_percent=E(L.predict(g, r['temperature_K'], 'final' if name == 'final' else 'noh'), g['references'][i], g['masks'][i])) for g in test for i, r in enumerate(g['records'])]
    fr[name] = summarize(rows)
out['line_weight_scan'] = lr; out['line_frozen_on_dataset_inputs'] = fr
# ---- (3) training-only evidence: hold out CO or CO2 (HITEMP references) with and without the HITRAN block
loo = {}; oof_line = []
for mol in ('CO', 'CO2'):
    held = [g for g in tr_hitemp if g['species'] == mol]; Gs = [g for g in G if g['split'] != 'testing' and g['species'] != mol]
    loo[mol] = {}
    for a in (0.0, 0.5):
        mdl_a = fit_pub(Gs, a); loo[mol][str(a)] = score_pub(mdl_a, held)[mol]
        if a == 0.5:
            for g in held:
                for i, r in enumerate(g['records']):
                    X = inference.features_at(g['features'], g['Tin'][-1], r['temperature_K']); sc = g['area'] / np.ptp(g['x'])
                    trees = np.maximum(g['anchor'][None, :] + np.asarray([e.predict(X) for e in mdl_a.estimators_]), 0) * sc
                    oof_line.append(dict(species=mol, window=g['window'], T=r['temperature_K'], x=g['x'], mask=g['masks'][i], ref=g['references'][i], pred=np.maximum(g['anchor'] + mdl_a.predict(X), 0) * sc,
                                         tree_sd=trees.std(0), q05=np.quantile(trees, .05, 0), q95=np.quantile(trees, .95, 0)))
    loo[mol]['unchanged'] = summarize([dict(species=g['species'], window=g['window'], error_percent=E(g['warm'], g['references'][i], g['masks'][i])) for g in held for i, r in enumerate(g['records'])])[mol]
    print('held-out', mol, {k: round(v, 2) for k, v in loo[mol].items()}, '%.0f s' % (time.time() - t0), flush=True)
out['line_training_only_holdout'] = loo
import pickle; (CACHE / 'line_oof_co_co2.pkl').write_bytes(pickle.dumps(oof_line))
# ---- (4) training-only evidence for the broad-band forest: leave-one-molecule-out with the HITRAN block added
mols = sorted({g['species'] for g in tr_meas}); rows = []
for mol in mols:
    mdl = B.fit([([g for g in tr_meas + tr_hitemp if g['species'] != mol], 1.0), ([g for g in hitran if g['species'] != mol], 0.5)])
    for g in [g for g in tr_meas if g['species'] == mol]:
        for i, r in enumerate(g['records']): rows.append(dict(species=mol, window=g['window'], error_percent=E(B.predict(mdl, g, r['temperature_K']), g['references'][i], g['masks'][i])))
s = summarize(rows); base = jload('lomo_training.json')['summary']['cap_1000.0']['per_molecule']; d = [s[m] - base[m] for m in mols]
out['broad_lomo_with_hitran'] = dict(per_molecule=s, mean=mean_of(s, mols), mean_without=mean_of(base, mols), paired_mean=float(np.mean(d)), ci95=boot_ci(d))
print('broad LOMO with HITRAN %.2f vs without %.2f; paired %.2f CI %s' % (mean_of(s, mols), mean_of(base, mols), np.mean(d), np.round(boot_ci(d), 2)), '%.0f s' % (time.time() - t0), flush=True)
# ---- (5) seed variability of the broad-band forest
seeds = []
for sd in range(10):
    s = score_broad(B.fit(B.main_blocks(), seed=sd), [g for g in test if g['kind'] == 'measurement']); seeds.append(dict(seed=sd, molecule=s, experiment6=mean_of(s, MEAS)))
v = np.array([x['experiment6'] for x in seeds]); out['seed_variability'] = dict(runs=seeds, mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max()))
print('seeds: mean %.2f sd %.2f range %.2f-%.2f' % (v.mean(), v.std(ddof=1), v.min(), v.max()), flush=True)
jdump(out, 'hitran.json')
