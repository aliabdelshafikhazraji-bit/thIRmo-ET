"""Why the HITEMP-reference error grows with temperature: share of the reference band area
that lies where the warmest input shows almost no absorption, band-area ratios and the error of the unchanged input."""
from common import *
import linebranch as L
tr_meas, tr_hitemp, hitran, test = split_groups(); pr = {r['record_id'] + '|' + r['window']: r for r in published_rows()}; rows = []
for g in test:
    if g['species'] not in LINE_RESOLVED: continue
    x = g['x']; warm = g['warm']; weak = warm < 0.02 * warm.max(); weak5 = warm < 0.05 * warm.max()
    for i, r in enumerate(g['records']):
        m = g['masks'][i]; ref = np.where(m, g['references'][i], 0); y = np.where(m, L.predict(g, r['temperature_K']), 0); p = pr[r['record_id'] + '|' + g['window']]
        rows.append(dict(species=g['species'], window=g['window'], T=r['temperature_K'], new_area_fraction=float(ref[weak].sum() / ref.sum()), new_area_fraction_5=float(ref[weak5].sum() / ref.sum()), ref_area_over_warm=float(ref.sum() / np.where(m, warm, 0).sum()),
                         pred_area_over_ref=float(y.sum() / ref.sum()), peak_ratio_ref_over_warm=float(ref.max() / warm.max()), unchanged=p['unchanged_error_percent'], model=p['final_error_percent'],
                         error_in_new_region_share=float(np.abs(y - ref)[weak].sum() / np.abs(y - ref).sum())))
jdump(rows, 'hot_growth.json')
for w in sorted({r['window'] for r in rows}):
    print(w)
    for r in [r for r in rows if r['window'] == w]: print(f"   {r['T']:7.1f} K  new-area {100*r['new_area_fraction']:5.1f}% (5%: {100*r['new_area_fraction_5']:5.1f}%)  ref/warm area {r['ref_area_over_warm']:.2f}  pred/ref area {r['pred_area_over_ref']:.2f}  peak ref/warm {r['peak_ratio_ref_over_warm']:.2f}  unchanged {r['unchanged']:6.1f}  model {r['model']:5.1f}  err share in new region {100*r['error_in_new_region_share']:4.0f}%")
