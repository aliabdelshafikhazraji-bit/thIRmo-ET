"""Line learning from measured cold spectra and spectral hot references only.

Tabulated masses/energies/strengths/widths in the archival NPZ are not loaded.
The spectral-to-spectral supervision fits the reference ordinate arrays.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,csv
import numpy as np
from scipy.optimize import least_squares
from scipy.special import voigt_profile
from sklearn.ensemble import ExtraTreesRegressor
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'three_input'))
from line_model.cold_line_representation import fit_line,cold_band_centroid,add_cold_thermal_parameters,predict_diatomic_line,PNNL_COLUMN
from line_model.spectra_only_line import feature_rows,eligible
from cold_router import route_cold_spectra
DATA=None
OUT=None
MODEL_OUTPUT=None
SPECIES=['CO','HCl','HF']
WINDOWS={'CO':[1800,2300],'HCl':[2400,3150],'HF':[3500,4300]}
PARAMS=dict(n_estimators=128,max_depth=8,min_samples_leaf=4,max_features=1.,random_state=20261101,n_jobs=3)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
def hot_fit(x,y,guess):
    """Five profile parameters and a linear background, all fitted to ordinate."""
    scale=max(float(y.max()),1e-40); yy=y/scale; delta=x-guess
    a=max(float(np.trapezoid(yy,x)),.001)
    initial=[np.log(a),np.log(.035),np.log(.006),0.,0.,0.]
    lower=[-15,np.log(.0001),np.log(.00005),-.12,-.1,-.2]
    upper=[5,np.log(.4),np.log(.08),.12,.3,.2]
    def residual(p):return np.exp(p[0])*voigt_profile(delta-p[3],np.exp(p[2]),np.exp(p[1]))+p[4]+p[5]*delta-yy
    fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=250,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    p=fit.x
    return dict(area=float(np.exp(p[0])*scale),gamma=float(np.exp(p[1])),gaussian_sigma=float(np.exp(p[2])),center=float(guess+p[3]),
        relative_rms=float(np.sqrt(np.mean(residual(p)**2))),fit_success=bool(fit.success),at_bound=bool(np.any(p-np.array(lower)<1e-5)|np.any(np.array(upper)-p<1e-5)))
def load():
    select=json.loads((DATA/'training_inputs.json').read_text())
    gate=json.loads((DATA/'cold_gate.json').read_text())
    passed={(r['species'],r['line_index']) for r in gate['line_decisions'] if r['pass']}
    # Only file paths, temperatures and reference-grid alignment are read here.
    bundles={};curves={};fits=[]
    for entry in select:
        sp=entry['species']; assert sp in SPECIES
        curves[sp]=[]
        for item,Tinput in zip(entry['pnnl_inputs'],entry['input_temperatures_K']):
            a=np.loadtxt(DATA/item['file']);a=a[np.argsort(a[:,0])]
            curves[sp].append((a[:,0],a[:,1]*np.log(10)/PNNL_COLUMN,float(Tinput)))
        keys=['wavenumber_cm1','temperatures_K','pressures_atm','spectra_native_cm2_molecule']
        with np.load(DATA/f'{sp}_native_training.npz') as z:bundles[sp]={k:z[k] for k in keys}
        z=bundles[sp]
        for j,x in enumerate(z['wavenumber_cm1']):
            if (sp,j) not in passed:continue
            for it,T in enumerate(z['temperatures_K']):
                if T<350:continue
                for ip,P in enumerate(z['pressures_atm']):
                    r=hot_fit(x,z['spectra_native_cm2_molecule'][j,it,ip],float(x[len(x)//2]))
                    fits.append(dict(species=sp,line_index=j,it=it,ip=ip,T=float(T),P=float(P),**r))
        print('SPECTRAL TARGET FITS',sp,flush=True)
    write(OUT/'hot_spectral_fit_labels.json',fits)
    return curves,bundles,fits
def gaussian_coefficient(fits,ids):
    per={}
    for sp in ids:
        f=[r['gaussian_sigma']/(r['center']*np.sqrt(r['T'])) for r in fits if r['species']==sp and r['fit_success'] and not r['at_bound'] and r['relative_rms']<.03]
        if not f:raise ValueError('No spectral estimates for Gaussian width.')
        per[sp]=float(np.median(f))
    return float(np.mean(list(per.values()))),per
def cold_records(curves,bundles,d,tag):
    gate=json.loads((DATA/'cold_gate.json').read_text());gd={(r['species'],r['line_index']):r['pass'] for r in gate['line_decisions']}
    records=[];decisions=[]
    for sp in SPECIES:
        cold=curves[sp];diag=route_cold_spectra(cold,WINDOWS[sp]);vib=cold_band_centroid(cold,WINDOWS[sp])
        for j,x in enumerate(bundles[sp]['wavenumber_cm1']):
            if not gd[(sp,j)]:continue
            guess=float(x[len(x)//2]);warmx,warmy,_=cold[-1];m=abs(warmx-guess)<.12
            center=float(warmx[m][np.argmax(warmy[m])])
            morphology=all(any(p['qualified'] and abs(p['center_cm1']-center)<.12 for p in a['peaks']) for a in diag['input_diagnostics'])
            if not morphology:continue
            r=add_cold_thermal_parameters(fit_line(cold,center,d,'sinc_0.056'),vib)
            r.update(species=sp,line_index=j)
            decisions.append(dict(species=sp,line_index=j,retained=eligible(r),rms=r['fit_relative_rms']))
            if eligible(r): records.append(r)
        print('COLD',tag,sp,sum(r['species']==sp for r in records),flush=True)
    write(OUT/f'cold_{tag}.json',dict(gaussian_coefficient=d,records=records,decisions=decisions))
    return records
def cases(records,bundles,fits):
    by={(r['species'],r['line_index']):r for r in records};result=[]
    for f in fits:
        key=f['species'],f['line_index']
        if key not in by:continue
        # Reference profiles must be representable; no test spectrum is involved.
        if not f['fit_success'] or f['relative_rms']>=.03 or f['at_bound']:continue
        r=by[key];a,g=predict_diatomic_line(r,f['T'],f['P'])
        result.append(dict(f,r=r,area_base=float(a),gamma_base=float(g)))
    return result
def train(cc):
    X=np.stack([feature_rows(c['r'],[c['T']])[0] for c in cc]);y=np.array([[np.log(c['area']/c['area_base']),np.log(c['gamma']/c['gamma_base'])] for c in cc])
    ids={c['species'] for c in cc};count={s:sum(c['species']==s for c in cc) for s in ids}
    weights=np.array([1/count[c['species']]*(.5 if c['species'] in ['HCl','HF'] else 1.) for c in cc]);weights*=len(weights)/weights.sum()
    return ExtraTreesRegressor(**PARAMS).fit(X,y,sample_weight=weights)
def evaluate(cc,bundles,d,model=None):
    X=np.stack([feature_rows(c['r'],[c['T']])[0] for c in cc]);corr=np.zeros((len(cc),2)) if model is None else model.predict(X);rows=[]
    for c,q in zip(cc,corr):
        z=bundles[c['species']];x=z['wavenumber_cm1'][c['line_index']];ref=z['spectra_native_cm2_molecule'][c['line_index'],c['it'],c['ip']]
        A=c['area_base']*np.exp(q[0]);g=c['gamma_base']*np.exp(q[1]);pred=A*voigt_profile(x-c['r']['center_cm1'],d*c['r']['center_cm1']*np.sqrt(c['T']),g)
        rows.append(dict(species=c['species'],line_index=c['line_index'],T=c['T'],P=c['P'],error_percent=float(100*np.trapezoid(abs(pred-ref),x)/np.trapezoid(np.maximum(ref,0),x))))
    return rows
def mean(rows):return float(np.mean([np.mean([r['error_percent'] for r in rows if r['species']==sp]) for sp in {r['species'] for r in rows}]))
def export(model,path):
    offsets=np.r_[0,np.cumsum([t.tree_.node_count for t in model.estimators_])]
    a={x:np.concatenate([getattr(t.tree_,y) for t in model.estimators_]) for x,y in [('left','children_left'),('right','children_right'),('feature','feature'),('threshold','threshold')]}
    a.update(offsets=offsets,value=np.concatenate([t.tree_.value[:,:,0] for t in model.estimators_]))
    np.savez_compressed(path,**a)
def main():
    import argparse
    global DATA,OUT,MODEL_OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='Acquired native reference NPZs, cold_gate.json and training_inputs.json.')
    parser.add_argument('--output',type=Path,required=True,help='New weight/config directory.')
    args=parser.parse_args()
    DATA=args.data.resolve();MODEL_OUTPUT=args.output.resolve();OUT=MODEL_OUTPUT/'audit'
    OUT.mkdir(parents=True,exist_ok=True)
    cold,bundles,fits=load();base=[];corr=[];folds=[]
    for held in SPECIES:
        d,per=gaussian_coefficient(fits,[s for s in SPECIES if s!=held])
        rec=cold_records(cold,bundles,d,'hold_'+held);cc=cases(rec,bundles,fits)
        tr=[c for c in cc if c['species']!=held];te=[c for c in cc if c['species']==held]
        model=train(tr);a=evaluate(te,bundles,d);b=evaluate(te,bundles,d,model);base+=a;corr+=b
        folds.append(dict(held_species=held,gaussian_coefficient=d,physical_error=mean(a),residual_error=mean(b)))
        print('LOSO',folds[-1],flush=True)
    d,per=gaussian_coefficient(fits,SPECIES);rec=cold_records(cold,bundles,d,'final');cc=cases(rec,bundles,fits);model=train(cc)
    dest=MODEL_OUTPUT;dest.mkdir(parents=True,exist_ok=True)
    weights=dest/'narrow_residual_forest.npz';export(model,weights)
    choose=mean(corr)<mean(base)
    config=dict(schema_version=3,selected_branch='physical_plus_extra_trees_residual' if choose else 'physical_baseline',
        gaussian_coefficient_cm1_per_cm1_sqrtK=d,gaussian_coefficient_basis='Equal-species mean of median fitted hot-spectral Voigt Gaussian-width coefficients; no molecular properties used.',
        gaussian_coefficient_training_species=per,training_species=SPECIES,numeric_model_sha256=sha(weights),parameters=PARAMS,
        features_count=10,training_cases=len(cc),line_counts={sp:sum(r['species']==sp for r in rec) for sp in SPECIES},
        temperature_training_K=sorted({c['T'] for c in cc}),pressure_training_atm=sorted({c['P'] for c in cc}),
        validation=dict(physical_LOSO_error=mean(base),residual_LOSO_error=mean(corr),folds=folds),
        identity_is_feature=False,molecular_mass_used=False,reference_parameters_read=False,strict_three_input=True,
        target_basis='Area and effective Lorentz width fitted to native reference spectra, never read from tabulated line parameters.',
        scope='Developmental transfer evaluation; fixed NO/CH4 heldout from all fitting, previously inspected during earlier development.')
    write(dest/'narrow_model_config.json',config);write(OUT/'LOSO_records.json',dict(physical=base,residual=corr))
    write(OUT/'frozen_manifest.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),no_new_holdout_evaluation=True,weights_sha256=sha(weights),config_sha256=sha(dest/'narrow_model_config.json'),training_code_sha256=sha(__file__)))
    print('FROZEN',json.dumps(config),flush=True)
if __name__=='__main__':main()
