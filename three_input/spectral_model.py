"""One three-spectrum calculation for band plots and sensing wavenumbers.

Broad envelopes and cold-observed isolated-line contrast are fitted separately.
The default fine-spectrum sum is independent of the requested grid. A requested
Gaussian response preserves the learned 2 cm-1 envelope; only cold-observed
eligible line contrast is restored at finer responses. The unresolved background
retains the learned envelope resolution.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.special import voigt_profile
from broad_model import ThreeInputBroadModel
from broad_model import frozen_core as core
from line_model import SpectraOnlyNarrowHead,eligible
from line_model.cold_line_representation import fit_line,cold_band_centroid,add_cold_thermal_parameters
from cold_router import route_cold_spectra

VERSION='4.0.1-experimental'
class ThreeSpectrumPredictor:
    def __init__(self,broad_model_path=None,line_model_dir=None):
        self.broad=ThreeInputBroadModel(broad_model_path)
        self.line=SpectraOnlyNarrowHead(line_model_dir)
        self._cache={}

    def prepare(self,spectra,input_temperatures_K,context_window_cm1,units='cm2/molecule'):
        ts=np.asarray(input_temperatures_K,float);bounds=np.asarray(context_window_cm1,float)
        if len(spectra)!=3 or ts.shape!=(3,) or not np.isfinite(ts).all() or len(np.unique(ts))!=3 or np.any((ts<275)|(ts>350)):
            raise ValueError('Exactly three measured spectra at distinct temperatures within 275-350 K are required.')
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<bounds[0]<bounds[1]:raise ValueError('Two increasing positive context boundaries are required.')
        if units=='pnnl':canonical='pnnl'
        elif units in ('cm2/molecule','cm2-molecule','cross-section'):canonical='cross-section'
        else:raise ValueError('Units must be pnnl or cm2/molecule.')
        curves=[]
        for i in np.argsort(ts):
            x,y=core.unique_sorted(*spectra[i])
            if x[0]>bounds[0] or x[-1]<bounds[1]:raise ValueError('Every cold spectrum must cover the full context band.')
            curves.append((x,core.to_cross_section(y,canonical),float(ts[i])))
        ts=np.sort(ts)
        h=hashlib.sha256(bounds.tobytes()+ts.tobytes())
        for x,y,T in curves:h.update(x.tobytes());h.update(y.tobytes())
        key=h.hexdigest()
        if key in self._cache:return self._cache[key]
        grid=np.linspace(*bounds,core.N)
        prepared=np.array([core.native_smooth(x,y,grid)[0] for x,y,T in curves])
        features,area,anchor,_,_=core.make_features(prepared,ts,grid)
        diagnostics=route_cold_spectra(curves,bounds)
        candidates=[];records=[];rejected=[]
        for peak in diagnostics['input_diagnostics'][-1]['peaks']:
            c=peak['center_cm1']
            if not peak['qualified'] or c<bounds[0]+.8 or c>bounds[1]-.8:continue
            passed=all(a['maximum_gap_cm1']<=.08 and any(p['qualified'] and abs(p['center_cm1']-c)<=.12 for p in a['peaks']) for a in diagnostics['input_diagnostics'])
            if passed:candidates.append(c)
        if candidates:
            vib=cold_band_centroid(curves,bounds)
            for c in candidates:
                r=add_cold_thermal_parameters(fit_line(curves,c,self.line.gaussian_coefficient,'sinc_0.056'),vib)
                if eligible(r):records.append(r)
                else:rejected.append(dict(candidate_center_cm1=c,rms=r['fit_relative_rms'],fit_at_bound=r['fit_at_bound'],effective_lower_energy_cm1=r['effective_lower_energy_cm1'],width_temperature_exponent=r['width_temperature_exponent']))
        context=dict(curves=curves,temperatures_K=ts,bounds=bounds,grid=grid,features=features,area=area,anchor=anchor,
            line_records=records,cold_diagnostics=diagnostics,rejected_cold_fits=rejected,input_digest=key,
            prepared_cold_spectra=prepared)
        self._cache[key]=context
        return context

    def evaluate(self,context,target_temperatures_K,target_pressures_atm,wavenumber_cm1,output_fwhm_cm1=0.):
        resolution=float(output_fwhm_cm1)
        if not np.isfinite(resolution) or resolution<0:raise ValueError('Output FWHM must be finite and nonnegative.')
        T=np.atleast_1d(np.asarray(target_temperatures_K,float));P=np.atleast_1d(np.asarray(target_pressures_atm,float))
        if T.ndim!=1 or not len(T) or not np.isfinite(T).all() or np.any(T<=context['temperatures_K'][-1]):raise ValueError('Target temperatures must exceed the warmest input.')
        if P.ndim!=1 or len(P) not in (1,len(T)) or not np.isfinite(P).all():raise ValueError('One pressure or one pressure per target temperature is required.')
        P=np.broadcast_to(P,T.shape)
        if np.any((P<.6)|(P>2)):raise ValueError('Target pressures must lie within 0.6-2.0 atm.')
        x=np.atleast_1d(np.asarray(wavenumber_cm1,float))
        if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any((x<context['bounds'][0])|(x>context['bounds'][1])):raise ValueError('Requested wavenumbers must lie within the supplied context band.')
        # There is deliberately no probe-specific route or model selection here.
        if len(x)<=4:
            right=np.clip(np.searchsorted(context['grid'],x),1,len(context['grid'])-1)
            positions=np.unique(np.r_[right-1,right]);n=len(positions)
            features=np.tile(context['features'][positions],(len(T),1))
            X=np.column_stack([features,np.repeat(T/1000,n),np.repeat((T-context['temperatures_K'][-1])/500,n)]).astype('f4')
            values=self.broad.forest.predict(X).reshape(len(T),n)
            envelope=np.maximum((values+context['anchor'][positions])*context['area']/np.ptp(context['bounds']),0)
            result=np.array([np.interp(x,context['grid'][positions],b) for b in envelope])
        else:
            envelope=np.array([np.maximum((context['anchor']+self.broad.forest.predict(core.features_at(context['features'],context['temperatures_K'][-1],t)))*context['area']/np.ptp(context['bounds']),0) for t in T])
            result=np.array([np.interp(x,context['grid'],b) for b in envelope])
        if not np.isfinite(result).all():
            raise ValueError('The supplied cold spectra do not constrain a finite extrapolation at these target conditions.')
        if resolution>core.FWHM:
            # B is learned at 2 cm-1, so only apply the additional variance.
            from scipy.ndimage import gaussian_filter1d
            native=np.linspace(*context['bounds'],int(np.ceil(np.ptp(context['bounds'])/.05))+1)
            coarse=self.evaluate(dict(context,line_records=[]),T,P,native,output_fwhm_cm1=0.)
            extra=np.sqrt(resolution**2-core.FWHM**2)/np.sqrt(8*np.log(2))
            coarse=gaussian_filter1d(coarse,extra/np.median(np.diff(native)),axis=1,mode='nearest',truncate=6)
            return np.array([np.interp(x,native,y) for y in coarse])
        if resolution==core.FWHM:return result
        output_sigma=resolution/np.sqrt(8*np.log(2))
        for r in context['line_records']:
            A,g=self.line.parameters(r,T,P)
            s=self.line.gaussian_coefficient*r['center_cm1']*np.sqrt(T)
            delta=x[None,:]-r['center_cm1']
            result+=A[:,None]*(voigt_profile(delta,np.hypot(s,output_sigma)[:,None],g[:,None])-voigt_profile(delta,np.hypot(s,core.SIGMA_CM1)[:,None],g[:,None]))
        if not np.isfinite(result).all():
            raise ValueError('The supplied cold spectra do not constrain a finite extrapolation at these target conditions.')
        return np.maximum(result,0)

    def predict(self,spectra,input_temperatures_K,target_temperatures_K,target_pressures_atm,
                context_window_cm1,units='cm2/molecule',probe_wavenumber_cm1=None,wavenumber_grid_cm1=None,output_fwhm_cm1=0.):
        c=self.prepare(spectra,input_temperatures_K,context_window_cm1,units)
        if probe_wavenumber_cm1 is not None and wavenumber_grid_cm1 is not None:raise ValueError('Supply a grid or a probe, not both.')
        if probe_wavenumber_cm1 is not None:x=np.array([float(probe_wavenumber_cm1)])
        elif wavenumber_grid_cm1 is not None:x=np.asarray(wavenumber_grid_cm1,float)
        elif float(output_fwhm_cm1)>=core.FWHM:x=c['grid']
        elif c['line_records']:x=np.linspace(*c['bounds'],int(np.ceil(np.ptp(c['bounds'])/.002))+1)
        else:x=c['grid']
        y=self.evaluate(c,target_temperatures_K,target_pressures_atm,x,output_fwhm_cm1=output_fwhm_cm1)
        T=np.atleast_1d(target_temperatures_K).astype(float);P=np.broadcast_to(np.atleast_1d(target_pressures_atm).astype(float),T.shape)
        warnings=[]
        if T.max()>1800:warnings.append('Some output temperatures exceed the isolated-line training maximum of 1800 K.')
        if P.max()>1.5:warnings.append('Some pressures exceed the isolated-line training maximum of 1.5 atm.')
        meta=dict(tool='thIRmo-ET',version=VERSION,input_count=3,input_temperatures_K=c['temperatures_K'].tolist(),
            target_temperatures_K=T.tolist(),target_pressures_atm=P.tolist(),context_window_cm1=c['bounds'].tolist(),
            branch='envelope_plus_resolved_contrast' if c['line_records'] else 'broad_band_envelope',
            molecular_identity_input=False,molecular_mass_input=False,molecular_properties_input=False,hot_reference_input=False,
            broad_weights_sha256=self.broad.forest.sha256,line_weights_sha256=self.line.sha256,
            gaussian_width_coefficient=self.line.gaussian_coefficient,
            output_units='cm2/molecule (natural-log convention)',input_units=units,
            reconstruction='For output FWHM r <= 2 cm-1: max(B + sum(G_r L - G_2 L), 0); for r > 2, B receives only the additional Gaussian variance.',
            output_gaussian_fwhm_cm1=float(output_fwhm_cm1),envelope_training_gaussian_fwhm_cm1=core.FWHM,
            unresolved_background_resolution_cm1=core.FWHM,
            resolution_note='Fine output restores only eligible cold-observed line contrast; it does not deconvolve the unresolved envelope background.',
            prediction_scope='Learned 2 cm-1 envelope with eligible cold-observed line contrast at the requested output response.',

            retained_cold_line_count=len(c['line_records']),retained_cold_line_centers_cm1=[r['center_cm1'] for r in c['line_records']],
            rejected_cold_fits=c['rejected_cold_fits'],cold_routing_thresholds={k:v for k,v in c['cold_diagnostics']['thresholds'].items() if k not in ['minimum_qualified_peak_count','minimum_qualified_prominence_fraction']},
            pressure_handling='Envelope independent of pressure; Lorentz widths of retained lines scale with target pressure/1 atm.',
            input_pressure_assumption_atm=1.,input_resolution_assumption='PNNL boxcar sinc, first zero 0.056 cm-1 for isolated-line fits.',
            same_function_for_spectrum_and_probe=True,warnings=warnings,
            assumptions=['Resolved correction predicts cold-observed lines; newly appearing hot transitions are represented only by the learned envelope.',
                'The line population baseline uses a linear-rotor/one-vibration proxy inferred from cold areas and band centroid.',
                'Bath-gas-specific widths and pressure shifts are not inferred from the three spectra.'])
        return dict(wavenumber_cm1=x,cross_section_cm2_molecule=y,metadata=meta)
