"""Input-only isolated-line representation of native PNNL spectra.

No hot reference spectrum or line-list parameters are consumed. The cold line
model is a Voigt profile convolved with an explicitly specified instrument.
It is a representation prototype, not a claimed high-temperature predictor.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import voigt_profile
from numpy.polynomial.legendre import leggauss

KB=1.380649e-23; AMU=1.66053906660e-27; C=299792458.; NA=6.02214076e23
C2=1.438776877
PNNL_COLUMN=1e-6*101325/(KB*296)*1e-6*100
NOMINAL_RESOLUTION=.112
_q,_w=leggauss(96)

def load_cold(directory,prefix,temperatures):
    curves=[]
    for suffix,T in zip(['5T','25T','50T'],temperatures):
        a=np.loadtxt(Path(directory)/f'{prefix}_{suffix}.TXT')
        a=a[np.argsort(a[:,0])]
        curves.append((a[:,0],a[:,1]*np.log(10)/PNNL_COLUMN,float(T)))
    return curves

def doppler_sigma(nu,T,mass_amu):
    return nu*np.sqrt(KB*T/(mass_amu*AMU))/C

def profile(delta,gamma,sigma_doppler,instrument='sinc_0.112'):
    """Unit-area profile; sinc number means FIRST ZERO in cm^-1.

    Gaussian branch means FWHM .112cm^-1. These are explicit candidate
    conventions, because the PNNL metadata only gives nominal resolution.
    """
    delta=np.asarray(delta)
    if instrument=='gaussian_0.112':
        return voigt_profile(delta,np.hypot(sigma_doppler,.112/np.sqrt(8*np.log(2))),gamma)
    if instrument not in ('sinc_0.112','sinc_0.056'):
        raise ValueError(instrument)
    first_zero=float(instrument.split('_')[1]); cutoff=1/(2*first_zero)
    f=(_q+1)*cutoff/2; weights=_w*cutoff/2
    return 2*np.sum(weights*np.exp(-2*np.pi*gamma*f-2*np.pi**2*sigma_doppler**2*f*f)*np.cos(2*np.pi*delta[...,None]*f),axis=-1)

def detect_candidates(curves,window,relative_prominence=.005,min_spacing=.8):
    x,y,_=curves[-1]; m=(x>=window[0])&(x<=window[1]);x=x[m];y=y[m]
    dx=float(np.median(np.diff(x)))
    peaks,props=find_peaks(y,prominence=max(y.max()*relative_prominence,1e-30),distance=max(1,int(.18/dx)))
    result=[]
    for j,p in enumerate(peaks):
        nearest=min(x[p]-x[peaks[j-1]] if j>0 else np.inf,x[peaks[j+1]]-x[p] if j<len(peaks)-1 else np.inf)
        if nearest>=min_spacing and x[p]>=window[0]+.8 and x[p]<=window[1]-.8:
            result.append(dict(candidate_center_cm1=float(x[p]),prominence=float(props['prominences'][j]),nearest_detected_peak_distance_cm1=float(nearest)))
    return result

def fit_line(curves,center,mass_amu,instrument='sinc_0.112',half_window=.65):
    xx=[];yy=[];ts=[]
    for x,y,T in curves:
        m=abs(x-center)<=half_window
        xx.append(x[m]);yy.append(y[m]);ts.append(T)
    scale=max(np.max(a) for a in yy)
    yy=[y/scale for y in yy]
    area=[max(np.trapezoid(y,x),.005) for x,y in zip(xx,yy)]
    # Shared line center, separate integrated areas, Lorentz HWHMs, baselines,
    # and linear backgrounds. Joint location improves coarse-sampling stability.
    initial=np.r_[0.,np.log(area),np.log([.035]*3),[0.]*3,[0.]*3]
    lower=np.r_[-.09,[-12.]*3,[np.log(.0001)]*3,[-.1]*3,[-.15]*3]
    upper=np.r_[.09,[3.]*3,[np.log(.3)]*3,[.15]*3,[.15]*3]
    def render(p,j,x):
        c=center+p[0];A=np.exp(p[1+j]);g=np.exp(p[4+j]);s=doppler_sigma(c,ts[j],mass_amu)
        return A*profile(x-c,g,s,instrument)+p[7+j]+p[10+j]*(x-c)
    def residual(p):
        return np.concatenate([render(p,j,x)-y for j,(x,y) in enumerate(zip(xx,yy))])
    fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=450,ftol=1e-10,xtol=1e-10,gtol=1e-10)
    p=fit.x;rr=residual(p);gammas=np.exp(p[4:7]);areas=np.exp(p[1:4])*scale
    slopes=np.polyfit(np.log(np.array(ts)/296),np.log(gammas),1)
    area_slopes=np.polyfit(1/np.array(ts),np.log(areas),1)
    return dict(center_cm1=float(center+p[0]),candidate_center_cm1=float(center),temperatures_K=ts,
        area_cm_molecule=areas.tolist(),lorentz_hwhm_cm1=gammas.tolist(),
        gamma_at296_cm1=float(np.exp(slopes[1])),width_temperature_exponent=float(-slopes[0]),
        inverse_temperature_area_slope=float(area_slopes[0]),
        fit_relative_rms=float(np.sqrt(np.mean(rr**2))),fit_relative_max_error=float(np.max(abs(rr))),
        fit_success=bool(fit.success),optimizer_evaluations=int(fit.nfev),instrument=instrument,
        half_window_cm1=half_window,mass_amu=mass_amu,
        baseline_cm2_molecule=(p[7:10]*scale).tolist(),baseline_slope=(p[10:13]*scale).tolist(),
        fit_at_bound=bool(np.any((p-lower)<1e-5)|np.any((upper-p)<1e-5)),
        fitted_parameters=p.tolist(),scale=scale)

def reconstruct_cold(record,x,index):
    c=record['center_cm1'];T=record['temperatures_K'][index]
    return record['area_cm_molecule'][index]*profile(np.asarray(x)-c,record['lorentz_hwhm_cm1'][index],doppler_sigma(c,T,record['mass_amu']),record['instrument'])+record['baseline_cm2_molecule'][index]+record['baseline_slope'][index]*(np.asarray(x)-c)

def render_intrinsic(record,x,area,gamma,T):
    """Remove instrument response; area/gamma/T must be explicitly supplied.

    This function alone does not predict thermally populated new lines.
    """
    return area*voigt_profile(np.asarray(x)-record['center_cm1'],doppler_sigma(record['center_cm1'],T,record['mass_amu']),gamma)

def cold_band_centroid(curves,window):
    """Estimate a vibrational scale from the warmest measured band only."""
    x,y,_=curves[-1];m=(x>=window[0])&(x<=window[1]);x=x[m];y=np.maximum(y[m],0)
    return float(np.trapezoid(x*y,x)/np.trapezoid(y,x))

def diatomic_partition_proxy(T,vibrational_wavenumber):
    """Rigid linear rotor times one harmonic vibration, up to a constant."""
    T=np.asarray(T,float)
    return T/(-np.expm1(-C2*vibrational_wavenumber/T))

def add_cold_thermal_parameters(record,vibrational_wavenumber):
    """Infer an effective lower-state population scale from cold line areas.

    This is a physical linear-diatomic assumption, not a target-specific line
    list. No high-temperature area, strength, broadening, or concentration is
    consumed. Species with electronic/spin structure may violate the proxy.
    """
    T=np.array(record['temperatures_K']);S=np.array(record['area_cm_molecule'])
    Q=diatomic_partition_proxy(T,vibrational_wavenumber)
    stim=-np.expm1(-C2*record['center_cm1']/T)
    slope,intercept=np.polyfit(1/T,np.log(S*Q/stim),1)
    return dict(record,vibrational_wavenumber_proxy_cm1=float(vibrational_wavenumber),
                effective_lower_energy_cm1=float(-slope/C2),
                effective_lower_energy_scale_K=float(-slope),
                thermal_proxy='Linear rigid rotor + one harmonic vibrational mode; inferred from cold line areas only',
                thermal_fit_residual_log=float(np.sqrt(np.mean((np.log(S*Q/stim)-(slope/T+intercept))**2))))

def predict_diatomic_line(record,T,P_atm=1.,width_exponent=None,area_log_correction=0.,width_log_correction=0.):
    """Physical narrow-line baseline, with optional independently trained residuals.

    P is total pressure in atm. Cold width is fitted at one atmosphere in N2;
    absent new broadener data, its power law is carried to the target bath.
    This assumption must accompany predictions in Ar or reactive mixtures.
    """
    T=np.asarray(T,float);P_atm=np.asarray(P_atm,float)
    Ta=record['temperatures_K'][-1];Sa=record['area_cm_molecule'][-1]
    vib=record['vibrational_wavenumber_proxy_cm1'];nu=record['center_cm1']
    area=Sa*diatomic_partition_proxy(Ta,vib)/diatomic_partition_proxy(T,vib)*np.exp(record['effective_lower_energy_scale_K']*(1/Ta-1/T))*((-np.expm1(-C2*nu/T))/(-np.expm1(-C2*nu/Ta)))*np.exp(area_log_correction)
    n=record['width_temperature_exponent'] if width_exponent is None else width_exponent
    gamma=record['gamma_at296_cm1']*(296/T)**n*P_atm*np.exp(width_log_correction)
    return area,gamma
