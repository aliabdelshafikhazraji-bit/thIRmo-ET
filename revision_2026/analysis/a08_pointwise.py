"""Wavelength-selection metrics: predicted versus reference optimum probing wavenumber,
the sensitivity lost by probing at the predicted optimum, and the pointwise cross-section error there."""
from common import *
import pickle
main = jload('main.json'); preds = pickle.loads((CACHE / 'main_preds.pkl').read_bytes()); G = {g['gi']: g for g in groups()}
rows = []; cnt = {}
for r in main['rows']:
    g = G[r['gi']]; i = next(k for k, rr in enumerate(g['records']) if rr['record_id'] == r['record_id']); m = g['masks'][i]; x = g['x']; ref = g['references'][i]; y = preds[(g['gi'], i, 'model')]
    idx = np.flatnonzero(m); jr = idx[np.argmax(ref[idx])]; jp = idx[np.argmax(y[idx])]; jw = idx[np.argmax(g['warm'][idx])]
    core = m & (ref >= 0.5 * ref[jr]); rel = (y[core] - ref[core]) / ref[core]
    rows.append(dict(species=r['species'], window=r['window'], temperature_K=r['temperature_K'], source_kind=r['source_kind'], nu_ref=float(x[jr]), nu_pred=float(x[jp]), nu_warm=float(x[jw]),
                     position_error_cm1=float(x[jp] - x[jr]), position_error_unchanged_cm1=float(x[jw] - x[jr]),
                     sensitivity_loss_percent=float(100 * (1 - ref[jp] / ref[jr])), sensitivity_loss_unchanged_percent=float(100 * (1 - ref[jw] / ref[jr])),
                     sigma_error_at_pred_opt_percent=float(100 * (y[jp] / ref[jp] - 1)), sigma_error_at_ref_peak_percent=float(100 * (y[jr] / ref[jr] - 1)), sigma_error_unchanged_at_ref_peak_percent=float(100 * (g['warm'][jr] / ref[jr] - 1)),
                     concentration_error_at_pred_opt_percent=float(100 * (ref[jp] / y[jp] - 1)), core_mean_abs_rel_error_percent=float(100 * np.mean(np.abs(rel))), core_points=int(core.sum()), grid_step=float(x[1] - x[0])))
def agg(key, f=np.abs):
    return summarize([dict(r, error_percent=float(f(r[key]))) for r in rows])
keys = ['position_error_cm1', 'position_error_unchanged_cm1', 'sensitivity_loss_percent', 'sensitivity_loss_unchanged_percent', 'sigma_error_at_pred_opt_percent', 'sigma_error_at_ref_peak_percent', 'sigma_error_unchanged_at_ref_peak_percent', 'concentration_error_at_pred_opt_percent', 'core_mean_abs_rel_error_percent']
mol = {k: agg(k) for k in keys}; signed = {k: agg(k, lambda v: v) for k in ('sigma_error_at_ref_peak_percent', 'position_error_cm1')}
out = dict(rows=rows, molecule_mean_abs=mol, molecule_mean_signed=signed, groups={k: dict(experiment6=mean_of(mol[k], MEAS), hitemp2=mean_of(mol[k], LINE_RESOLVED)) for k in keys})
# temperature breakdown for the experimental molecules
tb = []
for lo, hi in [(323.15, 500), (500, 700), (700, 900), (900, 1100)]:
    sel = [r for r in rows if r['source_kind'] == 'measurement' and lo < r['temperature_K'] <= hi]
    if sel: tb.append(dict(lo=lo, hi=hi, records=len(sel), **{k: float(np.mean(list(summarize([dict(r, error_percent=abs(r[k])) for r in sel]).values()))) for k in keys}))
out['experiment_by_temperature'] = tb; jdump(out, 'pointwise.json')
print(f"{'molecule':22s} |dnu| (unch)   sens.loss% (unch)   |err at pred opt|%   |err at ref peak|% (unch)   core mean |rel err|%")
for s in ORDER: print(f"{s:22s} {mol['position_error_cm1'][s]:5.1f} ({mol['position_error_unchanged_cm1'][s]:5.1f})   {mol['sensitivity_loss_percent'][s]:5.1f} ({mol['sensitivity_loss_unchanged_percent'][s]:5.1f})      {mol['sigma_error_at_pred_opt_percent'][s]:5.1f}            {mol['sigma_error_at_ref_peak_percent'][s]:5.1f} ({mol['sigma_error_unchanged_at_ref_peak_percent'][s]:5.1f})        {mol['core_mean_abs_rel_error_percent'][s]:5.1f}")
print({k: {a: round(b, 2) for a, b in v.items()} for k, v in out['groups'].items()})
for b in tb: print(b['lo'], b['hi'], b['records'], {k: round(b[k], 1) for k in keys})
