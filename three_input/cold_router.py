"""Frozen signal-only routing gate; no species names or hot references.

Passage means resolved, separated peaks dominate the supplied cold window.
It does NOT establish a diatomic identity, validate a partition function, or
prove that all hot-band absorption is represented by the observed lines.
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths

THRESHOLDS=dict(relative_peak_prominence=.005, minimum_peak_detection_separation_cm1=.18,
                maximum_width_cm1=.40, minimum_nearest_peak_cm1=1.30,
                minimum_prominence_over_peak_height=.50,
                minimum_qualified_peak_count=3, minimum_qualified_prominence_fraction=.90,
                maximum_native_sampling_cm1=.08)


def route_cold_spectra(curves, context_window):
    """curves is exactly three (ascending wavenumber, cross-section, T_K) tuples.

    Spectral units may be any common positive multiplicative scaling. The
    numerical inference API separately handles conversion to cross-section.
    No filename, species identity, molecular mass, or hot target is accessed.
    """
    if len(curves)!=3:raise ValueError('Exactly three cold spectra are required.')
    lo,hi=map(float,context_window)
    if not 0<lo<hi:raise ValueError('A positive increasing context window is required.')
    temperatures=[float(c[2]) for c in curves]
    if not np.all(np.isfinite(temperatures)) or len(set(temperatures))!=3 or min(temperatures)<=0:
        raise ValueError('Three distinct positive input temperatures are required.')
    details=[];all_pass=True
    for x,y,T in curves:
        x=np.asarray(x,float);y=np.asarray(y,float)
        if x.ndim!=1 or x.shape!=y.shape or len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x)<=0):
            raise ValueError('Finite, strictly ascending input spectral grids are required.')
        if x[0]>lo or x[-1]<hi:raise ValueError('Each input must cover the context window.')
        m=(x>=lo)&(x<=hi);x,y=x[m],y[m]
        if len(x)<3:raise ValueError('At least three grid samples within the window are required.')
        spacing=float(np.median(np.diff(x)));max_gap=float(np.max(np.diff(x)))
        reasons=[];peak_rows=[];fraction=0.;count=0
        if max_gap>THRESHOLDS['maximum_native_sampling_cm1']:
            reasons.append('Input grid is too coarse or contains gaps for the narrow-line fitting model.')
        height=float(y.max())
        if height<=0:
            reasons.append('No positive spectral peak.')
        else:
            peaks,p=find_peaks(y,prominence=THRESHOLDS['relative_peak_prominence']*height,
                    distance=max(1,int(THRESHOLDS['minimum_peak_detection_separation_cm1']/spacing)))
            widths=peak_widths(y,peaks,rel_height=.5)[0]*spacing
            gaps=np.minimum(np.r_[np.inf,np.diff(x[peaks])],np.r_[np.diff(x[peaks]),np.inf]) if len(peaks) else np.array([])
            contrast=np.divide(p['prominences'],y[peaks],out=np.zeros(len(peaks)),where=y[peaks]>0)
            qualified=(widths<=THRESHOLDS['maximum_width_cm1'])&(gaps>=THRESHOLDS['minimum_nearest_peak_cm1'])&(contrast>=THRESHOLDS['minimum_prominence_over_peak_height'])
            count=int(qualified.sum())
            if len(peaks):fraction=float(p['prominences'][qualified].sum()/p['prominences'].sum())
            for i,w,g,c,q,pr in zip(peaks,widths,gaps,contrast,qualified,p['prominences']):
                peak_rows.append(dict(center_cm1=float(x[i]),half_prominence_width_cm1=float(w),nearest_detected_peak_cm1=None if not np.isfinite(g) else float(g),prominence_over_peak_height=float(c),qualified=bool(q),prominence_fraction_of_max=float(pr/height)))
        if count<THRESHOLDS['minimum_qualified_peak_count']:reasons.append('Fewer than three isolated narrow peaks pass the shape criteria.')
        if fraction<THRESHOLDS['minimum_qualified_prominence_fraction']:reasons.append('Isolated narrow peaks account for less than 90% of detected peak prominence.')
        passed=not reasons;all_pass&=passed
        details.append(dict(temperature_K=float(T),native_median_step_cm1=spacing,maximum_gap_cm1=max_gap,
                     qualified_peak_count=count,qualified_prominence_fraction=fraction,passed=passed,reasons=reasons,peaks=peak_rows))
    return dict(route='resolved_line_candidate' if all_pass else 'broad',passed=all_pass,
                thresholds=THRESHOLDS.copy(),input_diagnostics=details,
                interpretation='Morphology only. The resolved branch still requires cold-fit quality checks and reports the component from observed cold lines; molecule class and unobserved hot bands are not inferred.')


def route_probe(curves,context_window,probe_wavenumber_cm1):
    """Local route: unchanged per-peak criteria at a specified probe frequency.

    The 90% whole-window and three-qualified-peak conditions are deliberately
    not used. A mixed window may contain a locally resolved observable feature.
    The nearest peak must satisfy the same cold shape criteria in all inputs,
    fall within the existing +/-0.65 cm-1 cold-fit half-window, and subsequently
    pass the line fitter's quality checks. The probe does not select parameters.
    """
    probe=float(probe_wavenumber_cm1)
    lo,hi=map(float,context_window)
    if not np.isfinite(probe) or not lo<=probe<=hi:
        raise ValueError('The finite probe wavenumber must lie within the context band.')
    diagnostics=route_cold_spectra(curves,context_window)
    selected=[];all_pass=True
    for item in diagnostics['input_diagnostics']:
        nearest=min(item['peaks'],key=lambda p:abs(p['center_cm1']-probe)) if item['peaks'] else None
        reasons=[]
        if item['maximum_gap_cm1']>THRESHOLDS['maximum_native_sampling_cm1']:
            reasons.append('Cold grid exceeds the fixed narrow-feature sampling limit.')
        if nearest is None:
            reasons.append('No detected cold peak near the probe.')
        else:
            if abs(nearest['center_cm1']-probe)>.65:
                reasons.append('Probe is outside the +/-0.65 cm-1 neighborhood of the nearest peak.')
            if not nearest['qualified']:
                reasons.append('Nearest peak fails the fixed width, separation, or prominence criteria.')
        passed=not reasons;all_pass&=passed
        selected.append(dict(temperature_K=item['temperature_K'],nearest_peak=nearest,
            maximum_gap_cm1=item['maximum_gap_cm1'],passed=passed,reasons=reasons))
    keys=['relative_peak_prominence','minimum_peak_detection_separation_cm1','maximum_width_cm1',
        'minimum_nearest_peak_cm1','minimum_prominence_over_peak_height','maximum_native_sampling_cm1']
    return dict(route='resolved_line_candidate' if all_pass else 'broad',passed=all_pass,
        probe_wavenumber_cm1=probe,probe_peak_distance_limit_cm1=.65,
        thresholds={key:THRESHOLDS[key] for key in keys},input_diagnostics=selected,
        global_prominence_fraction_gate_used=False,global_peak_count_gate_used=False,
        interpretation='Local cold morphology only; eligible cold line fit is also required. The narrow result is the contribution of retained observed lines, not a complete hot-band spectrum.')
