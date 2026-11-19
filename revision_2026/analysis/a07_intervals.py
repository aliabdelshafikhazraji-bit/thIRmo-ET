"""Per-wavenumber prediction intervals.
(i) Raw ensemble interval: 5th-95th percentile of the 128 per-tree predictions.
(ii) Calibrated interval: prediction +/- q (s_tree + beta * peak), with q the 90th percentile of the normalized
     out-of-fold residuals of training molecules held out one at a time (equal weight per molecule). The test
     molecules are not used for calibration."""
from common import *
import broad as B, linebranch as L, pickle
BETA = 0.02; NOMINAL = 0.90
def wquantile(v, w, q):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w) / w.sum(); return float(v[np.searchsorted(c, q)])
def calibrate(oof, bins=None):
    """Return q (global) or {bin: q}. Each molecule's valid points share unit weight."""
    by = {}
    for r in oof:
        m = r['mask']; sc = r['tree_sd'][m] + BETA * r['pred'].max(); res = np.abs(r['ref'][m] - r['pred'][m]) / sc
        key = None if bins is None else next(i for i, (lo, hi) in enumerate(bins) if lo < r['T'] <= hi)
        by.setdefault(key, []).append((r['species'], res))
    out = {}
    for key, items in by.items():
        sp = sorted({s for s, _ in items}); v = []; w = []
        for s in sp:
            arr = np.concatenate([a for t, a in items if t == s]); v.append(arr); w.append(np.full(len(arr), 1.0 / len(arr)))
        out[key] = wquantile(np.concatenate(v), np.concatenate(w), NOMINAL)
    return out
def evaluate(recs, q, bins=None):
    rows = []
    for r in recs:
        m = r['mask']; key = None if bins is None else next(i for i, (lo, hi) in enumerate(bins) if lo < r['T'] <= hi); qq = q.get(key, max(q.values()))
        half = qq * (r['tree_sd'] + BETA * r['pred'].max()); lo_, hi_ = np.maximum(r['pred'] - half, 0), r['pred'] + half; ref = r['ref']
        rows.append(dict(species=r['species'], window=r['window'], T=r['T'], cov_cal=float(np.mean((ref[m] >= lo_[m]) & (ref[m] <= hi_[m]))), cov_raw=float(np.mean((ref[m] >= r['q05'][m]) & (ref[m] <= r['q95'][m]))),
                         relwidth_cal=float(np.mean(half[m]) / ref[m].max()), relwidth_raw=float(np.mean(0.5 * (r['q95'][m] - r['q05'][m])) / ref[m].max()),
                         cov_cal_core=float(np.mean(((ref >= lo_) & (ref <= hi_))[m & (ref >= 0.5 * ref[m].max())])), relwidth_cal_core=float(np.mean(half[m & (ref >= 0.5 * ref[m].max())]) / ref[m].max())))
    return rows
def summarize_rows(rows, key):
    return summarize([dict(r, error_percent=100 * r[key]) for r in rows])
out = {}
# ---------------- broad-band branch
oof = pickle.loads((CACHE / 'lomo_training_oof.pkl').read_bytes()); tr_meas, tr_hitemp, hitran, test = split_groups(); mdl = B.main_model()
recs = []
for g in test:
    if g['species'] in LINE_RESOLVED: continue
    for i, r in enumerate(g['records']):
        T = r['temperature_K']; trees = B.predict(mdl, g, T, per_tree=True)
        recs.append(dict(species=g['species'], window=g['window'], gi=g['gi'], rec=i, T=T, x=g['x'], mask=g['masks'][i], ref=g['references'][i], pred=B.predict(mdl, g, T), tree_sd=trees.std(0), q05=np.quantile(trees, .05, 0), q95=np.quantile(trees, .95, 0)))
TB = [(0, 500), (500, 700), (700, 900), (900, 3000)]
for name, bins in (('global', None), ('temperature_bins', TB)):
    q = calibrate(oof, bins); rows = evaluate(recs, q, bins); cal_rows = evaluate(oof, q, bins)
    res = dict(q={str(k): v for k, v in q.items()}, test_rows=rows)
    for k in ('cov_cal', 'cov_raw', 'relwidth_cal', 'relwidth_raw', 'cov_cal_core', 'relwidth_cal_core'):
        s = summarize_rows(rows, k); res[k] = dict(molecule=s, mean=mean_of(s, MEAS)); c = summarize_rows(cal_rows, k); res[k]['calibration_molecule_mean'] = float(np.mean(list(c.values())))
    res['by_temperature'] = []
    for lo, hi in TB:
        sel = [r for r in rows if lo < r['T'] <= hi]
        if sel: res['by_temperature'].append(dict(lo=lo, hi=hi, records=len(sel), **{k: float(np.mean(list(summarize_rows(sel, k).values()))) for k in ('cov_cal', 'cov_raw', 'relwidth_cal', 'relwidth_raw')}))
    out['broad_' + name] = res
    print(name, 'q', {k: round(v, 2) for k, v in res['q'].items()}, '| test coverage calibrated %.1f%% (raw ensemble %.1f%%), mean half-width %.1f%% of peak (raw %.1f%%) | calibration-set coverage %.1f%%' % (
        res['cov_cal']['mean'], res['cov_raw']['mean'], res['relwidth_cal']['mean'], res['relwidth_raw']['mean'], res['cov_cal']['calibration_molecule_mean']))
    print('   per molecule cal:', {k: round(v, 1) for k, v in res['cov_cal']['molecule'].items()}, ' raw:', {k: round(v, 1) for k, v in res['cov_raw']['molecule'].items()})
    print('   by T:', [(b['lo'], b['hi'], b['records'], round(b['cov_cal'], 1), round(b['cov_raw'], 1), round(b['relwidth_cal'], 1)) for b in res['by_temperature']])
(CACHE / 'interval_test_records.pkl').write_bytes(pickle.dumps(recs))
# ---------------- line-resolved branch (published envelope; frozen forest evaluated on dataset inputs)
f = CACHE / 'line_oof_co_co2.pkl'
lrecs = []
for g in test:
    if g['species'] not in LINE_RESOLVED: continue
    for i, r in enumerate(g['records']):
        T = r['temperature_K']; trees = L.predict(g, T, per_tree=True)
        lrecs.append(dict(species=g['species'], window=g['window'], T=T, x=g['x'], mask=g['masks'][i], ref=g['references'][i], pred=L.predict(g, T), tree_sd=trees.std(0), q05=np.quantile(trees, .05, 0), q95=np.quantile(trees, .95, 0)))
if f.exists():
    loof = pickle.loads(f.read_bytes()); q = calibrate(loof); rows = evaluate(lrecs, q); res = dict(q={str(k): v for k, v in q.items()}, test_rows=rows)
    for k in ('cov_cal', 'cov_raw', 'relwidth_cal', 'relwidth_raw'):
        s = summarize_rows(rows, k); res[k] = dict(molecule=s, mean=mean_of(s, LINE_RESOLVED))
    res['by_temperature'] = []
    for lo, hi in [(0, 500), (500, 700), (700, 900), (900, 1100), (1100, 1600)]:
        sel = [r for r in rows if lo < r['T'] <= hi]
        if sel: res['by_temperature'].append(dict(lo=lo, hi=hi, records=len(sel), **{k: float(np.mean(list(summarize_rows(sel, k).values()))) for k in ('cov_cal', 'cov_raw', 'relwidth_cal', 'relwidth_raw')}))
    out['line_resolved'] = res
    print('line-resolved q', res['q'], '| coverage calibrated %.1f%% (raw %.1f%%), half-width %.1f%% (raw %.1f%%)' % (res['cov_cal']['mean'], res['cov_raw']['mean'], res['relwidth_cal']['mean'], res['relwidth_raw']['mean']))
    print('   per molecule cal:', {k: round(v, 1) for k, v in res['cov_cal']['molecule'].items()}, ' raw:', {k: round(v, 1) for k, v in res['cov_raw']['molecule'].items()})
    print('   by T:', [(b['lo'], b['hi'], b['records'], round(b['cov_cal'], 1), round(b['cov_raw'], 1), round(b['relwidth_cal'], 1)) for b in res['by_temperature']])
    (CACHE / 'interval_line_records.pkl').write_bytes(pickle.dumps(lrecs))
jdump(out, 'intervals.json')
