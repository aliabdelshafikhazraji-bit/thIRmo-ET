"""Experimental line branch requiring three spectra and their temperatures.

No species name, molecular mass, formula, hot target spectrum, or line list is
an inference input. One Doppler mass proxy is learned from training species.
The physics still assumes linear-rotor-like population behavior; this branch
is not validated for every molecule with resolved absorption features.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.special import voigt_profile
from .cold_line_representation import (detect_candidates,fit_line,cold_band_centroid,
    add_cold_thermal_parameters,predict_diatomic_line,doppler_sigma)

HERE=Path(__file__).resolve().parent

def eligible(r):
    return bool(r['fit_success'] and not r['fit_at_bound'] and r['fit_relative_rms']<.03
        and r['effective_lower_energy_cm1']>=0 and 0<=r['width_temperature_exponent']<=2)

def feature_rows(r,temperatures):
    T=np.atleast_1d(temperatures).astype(float);Ta=r['temperatures_K'][-1]
    c=[r['effective_lower_energy_cm1']/1000,r['width_temperature_exponent'],
       np.log(r['gamma_at296_cm1']),r['center_cm1']/r['vibrational_wavenumber_proxy_cm1'],
       np.log(r['area_cm_molecule'][-1]),r['thermal_fit_residual_log']]
    return np.column_stack([np.full(len(T),v) for v in c]+[T/Ta,Ta/T,
        1.438776877*r['vibrational_wavenumber_proxy_cm1']/T,(1/Ta-1/T)*1000])

class SpectraOnlyNarrowHead:
    def __init__(self,model_dir=None):
        path=Path(model_dir or HERE/'model_frozen')
        self.config=json.loads((path/'narrow_model_config.json').read_text())
        weights=path/'narrow_residual_forest.npz'
        self.sha256=hashlib.sha256(weights.read_bytes()).hexdigest()
        if self.sha256!=self.config['numeric_model_sha256']:raise ValueError('Model hash mismatch.')
        self.mass_proxy=self.config['doppler_mass_proxy_amu']
        self.use_head=self.config['selected_branch']=='physical_plus_extra_trees_residual'
        with np.load(weights,allow_pickle=False) as z:
            for name in z.files:setattr(self,name,z[name])
    def correction(self,X):
        X=np.asarray(X,dtype='f4');result=np.zeros((len(X),2));rows=np.arange(len(X))
        if X.shape[1]!=10 or not np.isfinite(X).all():raise ValueError('Ten finite features required.')
        if not self.use_head:return result
        for start,end in zip(self.offsets[:-1],self.offsets[1:]):
            nodes=np.zeros(len(X),dtype=int)
            for _ in range(64):
                feature=self.feature[start+nodes];active=feature>=0
                if not active.any():break
                current=start+nodes[active]
                left=X[rows[active],feature[active]]<=self.threshold[current]
                nodes[active]=np.where(left,self.left[current],self.right[current])
            else:raise ValueError('Tree traversal did not terminate.')
            result+=self.value[start+nodes]
        return result/(len(self.offsets)-1)
    def parameters(self,r,T,P_atm=1.):
        T=np.atleast_1d(T).astype(float);P=np.broadcast_to(np.asarray(P_atm,float),T.shape)
        if np.any(T<=0) or np.any(P<=0) or not np.isfinite(T).all() or not np.isfinite(P).all():
            raise ValueError('Finite positive gas conditions required.')
        if r['mass_amu']!=self.mass_proxy:raise ValueError('Cold representation must use the frozen training mass proxy.')
        corr=self.correction(feature_rows(r,T));area,gamma=predict_diatomic_line(r,T,P)
        return area*np.exp(corr[:,0]),gamma*np.exp(corr[:,1])
    def line(self,r,wavenumber,T,P_atm=1.):
        T=np.atleast_1d(T).astype(float);area,gamma=self.parameters(r,T,P_atm)
        return area*voigt_profile(np.asarray(wavenumber)-r['center_cm1'],
            doppler_sigma(r['center_cm1'],T,self.mass_proxy),gamma)

def fit_band(curves,context_window,input_pressure_atm=1.,model_dir=None):
    if len(curves)!=3:raise ValueError('Exactly three distinct-temperature near-ambient spectra are required.')
    curves=sorted(curves,key=lambda c:c[2]);ts=np.array([c[2] for c in curves])
    if np.any(np.diff(ts)<=0) or not np.isfinite(ts).all():raise ValueError('Temperatures must be distinct and finite.')
    if input_pressure_atm<=0:raise ValueError('Positive input pressure required.')
    for x,y,T in curves:
        if x[0]>context_window[0] or x[-1]<context_window[1]:raise ValueError('Spectra must cover the context band.')
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.diff(x)<=0):raise ValueError('Finite ascending spectra required.')
    config=json.loads((Path(model_dir or HERE/'model_frozen')/'narrow_model_config.json').read_text())
    mass_proxy=config['doppler_mass_proxy_amu']
    vib=cold_band_centroid(curves,context_window)
    records=[]
    for candidate in detect_candidates(curves,context_window,relative_prominence=.005,min_spacing=.8):
        r=fit_line(curves,candidate['candidate_center_cm1'],mass_proxy,instrument='sinc_0.056')
        r=add_cold_thermal_parameters(r,vib)
        r['gamma_at296_cm1']/=input_pressure_atm
        r['input_pressure_atm']=input_pressure_atm
        r['mass_basis']='Fixed equal-species mean from training set; not target molecular mass'
        r['retained']=eligible(r);records.append(r)
    if not any(r['retained'] for r in records):raise ValueError('No isolated lines satisfy cold-fit quality criteria.')
    return records

def predict_lines(head,records,wavenumber,T,P_atm=1.):
    result=np.zeros(np.broadcast(np.asarray(wavenumber),np.atleast_1d(T)).shape)
    for r in records:
        if eligible(r):result+=head.line(r,wavenumber,T,P_atm)
    return result
