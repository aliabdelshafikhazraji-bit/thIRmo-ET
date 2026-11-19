"""Reference-data consistency: spectra of the same molecule and window measured at nearly the same temperature
in different facilities (and digitized versus numerical records). Quantifies the reference uncertainty floor."""
from common import *
G = groups(); out = []
for g in G:
    if g['kind'] != 'measurement': continue
    recs = g['records']
    for i in range(len(recs)):
        for j in range(i + 1, len(recs)):
            a, b = recs[i], recs[j]
            if str(a.get('source')) == str(b.get('source')): continue
            dT = abs(a['temperature_K'] - b['temperature_K'])
            if dT > 30: continue
            m = g['masks'][i] & g['masks'][j]
            if m.sum() < 30: continue
            ra, rb = g['references'][i], g['references'][j]; mean = 0.5 * (ra + rb)
            e = 100 * float(np.abs(ra[m] - rb[m]).sum() / np.maximum(mean[m], 0).sum())
            area = float(np.trapezoid(np.where(m, ra, 0), g['x']) / np.trapezoid(np.where(m, rb, 0), g['x']))
            out.append(dict(species=g['species'], split=g['split'], window=[float(g['x'][0]), float(g['x'][-1])], T_a=a['temperature_K'], T_b=b['temperature_K'], source_a=str(a.get('source')), source_b=str(b.get('source')),
                            origin_a=a.get('data_origin'), origin_b=b.get('data_origin'), bath_a=str(a.get('bath_gas')), bath_b=str(b.get('bath_gas')), P_a=a['pressure_atm'], P_b=b['pressure_atm'], difference_percent=e, area_ratio=area, common_points=int(m.sum())))
jdump(out, 'source_consistency.json')
for o in out: print(f"{o['species']:15s} {o['window'][0]:.0f}-{o['window'][1]:.0f}  {o['T_a']:.0f} K [{o['source_a'][:28]}, {o['origin_a']}, {o['bath_a'][:3]}] vs {o['T_b']:.0f} K [{o['source_b'][:28]}, {o['origin_b']}, {o['bath_b'][:3]}]  diff {o['difference_percent']:.1f}%  area ratio {o['area_ratio']:.3f}")

# pairs from different windows (different source studies) of the same molecule with overlapping wavenumber coverage
out2 = []; meas = [g for g in G if g['kind'] == 'measurement']
for a in range(len(meas)):
    for b in range(a + 1, len(meas)):
        ga, gb = meas[a], meas[b]
        if ga['species'] != gb['species']: continue
        lo, hi = max(ga['x'][0], gb['x'][0]), min(ga['x'][-1], gb['x'][-1])
        if hi - lo < 30: continue
        xc = np.arange(lo, hi + 1e-9, 0.5)
        for i, ra in enumerate(ga['records']):
            for j, rb in enumerate(gb['records']):
                if str(ra.get('source')) == str(rb.get('source')) or abs(ra['temperature_K'] - rb['temperature_K']) > 30: continue
                ma = np.interp(xc, ga['x'], ga['masks'][i].astype(float)) > 0.99; mb = np.interp(xc, gb['x'], gb['masks'][j].astype(float)) > 0.99; m = ma & mb
                if m.sum() < 60: continue
                ya = np.interp(xc, ga['x'], ga['references'][i]); yb = np.interp(xc, gb['x'], gb['references'][j]); mean = 0.5 * (ya + yb)
                out2.append(dict(species=ga['species'], range=[float(xc[m][0]), float(xc[m][-1])], T_a=ra['temperature_K'], T_b=rb['temperature_K'], source_a=str(ra.get('source')), source_b=str(rb.get('source')), bath_a=str(ra.get('bath_gas')), bath_b=str(rb.get('bath_gas')),
                                 P_a=ra['pressure_atm'], P_b=rb['pressure_atm'], origin_a=ra.get('data_origin'), origin_b=rb.get('data_origin'), difference_percent=100 * float(np.abs(ya[m] - yb[m]).sum() / np.maximum(mean[m], 0).sum()),
                                 area_ratio=float(np.trapezoid(np.where(m, ya, 0), xc) / np.trapezoid(np.where(m, yb, 0), xc)), peak_ratio=float(ya[m].max() / yb[m].max())))
jdump(dict(same_window=out, cross_window=out2), 'source_consistency.json')
print('cross-window pairs'); 
for o in out2: print(f"{o['species']:15s} {o['range'][0]:.0f}-{o['range'][1]:.0f}  {o['T_a']:.0f} K [{o['source_a'][:28]}, {o['bath_a'][:3]}, {o['P_a']:.2f} atm] vs {o['T_b']:.0f} K [{o['source_b'][:28]}, {o['bath_b'][:3]}, {o['P_b']:.2f} atm]  diff {o['difference_percent']:.1f}%  area {o['area_ratio']:.3f}  peak {o['peak_ratio']:.3f}")
