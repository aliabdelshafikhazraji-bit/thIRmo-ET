"""Train and select a narrow-line residual head using training species only.

The final NO test source and published NO calibration are never loaded here.
Model selection is leave-one-molecule-out over CO, HCl, and HF.
"""
from pathlib import Path
import json,csv,hashlib,pickle,argparse
import numpy as np
from scipy.special import voigt_profile
from sklearn.ensemble import ExtraTreesRegressor
from line_model.cold_line_representation import predict_diatomic_line,doppler_sigma

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'model_frozen'
DATA=ROOT/'training_data'
COLD=ROOT/'audit/training_cold_line_representations.json'
SPECIES=['CO','HCl','HF']
PARAMS=dict(n_estimators=128,max_depth=8,min_samples_leaf=4,max_features=1.,random_state=20261101,n_jobs=3)
FEATURE_NAMES=['effective_lower_energy_per1000cm1','cold_width_temperature_exponent',
 'log_gamma296_cm1','line_to_vibrational_wavenumber_ratio',
 'log_cold_anchor_line_area','cold_thermal_fit_log_residual',
 'target_to_anchor_temperature','anchor_to_target_temperature',
 'vibrational_temperature_to_target_temperature','inverse_temperature_change_per1000K']

def feature_rows(record,temperatures):
    T=np.atleast_1d(temperatures).astype(float);Ta=record['temperatures_K'][-1]
    constant=[record['effective_lower_energy_cm1']/1000,record['width_temperature_exponent'],
      np.log(record['gamma_at296_cm1']),record['center_cm1']/record['vibrational_wavenumber_proxy_cm1'],
      np.log(record['area_cm_molecule'][-1]),record['thermal_fit_residual_log']]
    return np.column_stack([np.full(len(T),v) for v in constant]+[T/Ta,Ta/T,
      1.438776877*record['vibrational_wavenumber_proxy_cm1']/T,(1/Ta-1/T)*1000])

def load_training():
    cold=json.loads(COLD.read_text())['records']
    gate_path=DATA/'cold_gate.json';gate=json.loads(gate_path.read_text())
    gd={(a['species'],a['line_index']):a for a in gate['line_decisions']}
    selected=[];decisions=[]
    for r in cold:
        assert r['species'] in SPECIES
        reasons=[]
        if not gd[(r['species'],r['line_index'])]['pass']:reasons.append('native_cold_mismatch_above20percent')
        if r['fit_relative_rms']>=.03:reasons.append('cold_fit_RMS_above3percent')
        if r['fit_at_bound']:reasons.append('optimizer_bound_active')
        if r['effective_lower_energy_cm1']<0:reasons.append('negative_effective_energy')
        if not 0<=r['width_temperature_exponent']<=2:reasons.append('width_exponent_outside0to2')
        if not r['fit_success']:reasons.append('optimizer_failure')
        decisions.append(dict(species=r['species'],line_index=r['line_index'],retained=not reasons,reasons=reasons,
              worst_cold_relative_mismatch=gd[(r['species'],r['line_index'])]['worst_cold_relative_mismatch']))
        if not reasons:selected.append(r)
    datasets={}
    needed=['temperatures_K','pressures_atm','line_strength_cm_molecule','lorentz_hwhm_cm1','wavenumber_cm1','spectra_native_cm2_molecule']
    for sp in SPECIES:
        with np.load(DATA/f'{sp}_native_training.npz') as z:
            datasets[sp]={k:z[k] for k in needed}
    return selected,datasets,decisions

def cases_for(records,datasets):
    result=[]
    for record in records:
        z=datasets[record['species']];j=record['line_index']
        for it,T in enumerate(z['temperatures_K']):
            if T<350:continue
            for ip,P in enumerate(z['pressures_atm']):
                area,gamma=predict_diatomic_line(record,T,P)
                result.append(dict(record=record,T=float(T),P=float(P),it=it,ip=ip,
                  area_base=float(area),gamma_base=float(gamma),
                  area_true=float(z['line_strength_cm_molecule'][j,it]),
                  gamma_true=float(z['lorentz_hwhm_cm1'][j,it,ip])))
    return result

def train(records,datasets):
    cases=cases_for(records,datasets)
    X=np.stack([feature_rows(c['record'],[c['T']])[0] for c in cases])
    y=np.array([[np.log(c['area_true']/c['area_base']),np.log(c['gamma_true']/c['gamma_base'])] for c in cases])
    ids={r['species'] for r in records};counts={s:sum(c['record']['species']==s for c in cases) for s in ids}
    weights=np.array([1/counts[c['record']['species']] for c in cases]);weights*=len(weights)/weights.sum()
    model=ExtraTreesRegressor(**PARAMS).fit(X,y,sample_weight=weights)
    return model

def evaluate(records,datasets,model=None):
    rows=[]
    cases=cases_for(records,datasets)
    all_corr=np.zeros((len(cases),2)) if model is None else model.predict(np.stack([feature_rows(c['record'],[c['T']])[0] for c in cases]))
    for case_index,c in enumerate(cases):
        r=c['record'];sp=r['species'];z=datasets[sp];j=r['line_index']
        corr=all_corr[case_index]
        area=c['area_base']*np.exp(corr[0]);gamma=c['gamma_base']*np.exp(corr[1])
        x=z['wavenumber_cm1'][j];ref=z['spectra_native_cm2_molecule'][j,c['it'],c['ip']]
        pred=area*voigt_profile(x-r['center_cm1'],doppler_sigma(r['center_cm1'],c['T'],r['mass_amu']),gamma)
        l1=100*np.trapezoid(abs(pred-ref),x)/np.trapezoid(np.maximum(ref,0),x)
        rows.append(dict(species=sp,line_index=j,T_K=c['T'],P_atm=c['P'],
          local_integrated_L1_percent=float(l1),area_percent_error=float(100*(area/c['area_true']-1)),
          width_percent_error=float(100*(gamma/c['gamma_true']-1)),
          predicted_area=float(area),reference_area=c['area_true'],predicted_width=float(gamma),reference_width=c['gamma_true']))
    return rows

def summarize(rows):
    per={}
    for sp in sorted({r['species'] for r in rows}):
        rr=[r for r in rows if r['species']==sp]
        per[sp]=dict(n_lines=len({r['line_index'] for r in rr}),n_cases=len(rr),
          mean_local_integrated_L1_percent=float(np.mean([r['local_integrated_L1_percent'] for r in rr])),
          mean_absolute_area_error_percent=float(np.mean([abs(r['area_percent_error']) for r in rr])),
          mean_absolute_width_error_percent=float(np.mean([abs(r['width_percent_error']) for r in rr])))
    return dict(per_species=per,mean_species_local_integrated_L1_percent=float(np.mean([r['mean_local_integrated_L1_percent'] for r in per.values()])),
      mean_species_absolute_area_error_percent=float(np.mean([r['mean_absolute_area_error_percent'] for r in per.values()])),
      mean_species_absolute_width_error_percent=float(np.mean([r['mean_absolute_width_error_percent'] for r in per.values()])))

def export_numeric(model,path):
    offsets=np.r_[0,np.cumsum([t.tree_.node_count for t in model.estimators_])]
    arrays={a:np.concatenate([getattr(t.tree_,b) for t in model.estimators_]) for a,b in [('left','children_left'),('right','children_right'),('feature','feature'),('threshold','threshold')]}
    arrays.update(offsets=offsets,value=np.concatenate([t.tree_.value[:,:,0] for t in model.estimators_]))
    np.savez_compressed(path,**arrays)
