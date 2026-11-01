"""thIRmo-ET: three near-ambient spectra to a hot band or probe cross-section."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from broad_model import ThreeInputBroadModel
from broad_model import frozen_core as core
from line_model import SpectraOnlyNarrowHead,fit_band,predict_lines,eligible
from cold_router import route_probe

VERSION='3.0.0-experimental'

class ThreeSpectrumPredictor:
    """Common interface with no species identity or target molecular constants.

    Full spectra retain the frozen 512-point broad-band envelope. At a supplied
    probe frequency, the three cold spectra determine whether an observed-line
    component can be evaluated using the experimental narrow branch. Otherwise
    the probe result is interpolated from the same broad-band envelope.
    """
    def __init__(self):
        self.broad=ThreeInputBroadModel()
        self.line=SpectraOnlyNarrowHead()

    def predict(self,spectra,input_temperatures_K,target_temperatures_K,
                target_pressures_atm,context_window_cm1,units='cm2/molecule',
                probe_wavenumber_cm1=None):
        ts=np.asarray(input_temperatures_K,float)
        targets=np.atleast_1d(np.asarray(target_temperatures_K,float))
        if len(spectra)!=3 or ts.shape!=(3,):
            raise ValueError('Exactly three measured spectra and three input temperatures are required.')
        if not np.isfinite(ts).all() or len(np.unique(ts))!=3 or np.any((ts<275)|(ts>350)):
            raise ValueError('Input temperatures must be three distinct values within 275-350 K.')
        if targets.ndim!=1 or not len(targets) or not np.isfinite(targets).all() or np.any(targets<=ts.max()):
            raise ValueError('Every target temperature must exceed the warmest input temperature.')
        pressures=np.atleast_1d(np.asarray(target_pressures_atm,float))
        if pressures.ndim!=1 or len(pressures) not in (1,len(targets)) or not np.isfinite(pressures).all():
            raise ValueError('Supply one target pressure or one pressure per target temperature.')
        pressures=np.broadcast_to(pressures,targets.shape).copy()
        if np.any((pressures<.6)|(pressures>2)):
            raise ValueError('Target pressures must lie within the study range 0.6-2.0 atm.')
        bounds=np.asarray(context_window_cm1,float)
        if bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<bounds[0]<bounds[1]:
            raise ValueError('Supply two increasing positive context-band boundaries.')
        if units=='pnnl':canonical='pnnl'
        elif units in ('cm2/molecule','cm2-molecule','cross-section'):canonical='cross-section'
        else:raise ValueError('Units must be pnnl or cm2/molecule.')
        order=np.argsort(ts);ts=ts[order];curves=[]
        for i in order:
            x,y=core.unique_sorted(*spectra[i])
            if x[0]>bounds[0] or x[-1]<bounds[1]:
                raise ValueError('Each spectrum must cover the complete context band.')
            curves.append((x,core.to_cross_section(y,canonical),float(input_temperatures_K[i])))
        metadata=dict(tool='thIRmo-ET',version=VERSION,input_count=3,
            input_temperatures_K=ts.tolist(),target_temperatures_K=targets.tolist(),
            target_pressures_atm=pressures.tolist(),context_window_cm1=bounds.tolist(),
            input_units=units,output_units='cm2/molecule (natural-log convention)',
            molecular_identity_input=False,molecular_mass_input=False,hot_reference_input=False,
            broad_weights_sha256=self.broad.forest.sha256,line_weights_sha256=self.line.sha256,
            training_Doppler_mass_proxy_amu=self.line.mass_proxy,
            input_pressure_assumption_atm=1.0,
            input_resolution_assumption='Narrow fits use the PNNL-like sinc instrument response with first zero at 0.056 cm-1.',
            warnings=[],assumptions=[])
        if probe_wavenumber_cm1 is not None:
            probe=float(probe_wavenumber_cm1)
            routing=route_probe(curves,bounds,probe)
            metadata['probe_routing']=routing
            metadata['probe_wavenumber_cm1']=probe
            if routing['passed']:
                try:
                    records=fit_band(curves,bounds)
                    nearest=min(records,key=lambda r:abs(r['center_cm1']-probe))
                    fitted_pass=abs(nearest['center_cm1']-probe)<=.65 and eligible(nearest)
                except ValueError as exc:
                    records=[];nearest=None;fitted_pass=False
                    metadata['cold_fit_failure']=str(exc)
                metadata['nearest_cold_fit_eligible']=bool(fitted_pass)
                if fitted_pass:
                    sigma=predict_lines(self.line,records,probe,targets,pressures)
                    if not np.isfinite(sigma).all() or np.any(sigma<0):
                        raise ValueError('The observed-line prediction is not finite and nonnegative.')
                    metadata.update(branch='observed_line_component',
                        prediction_scope='Cross-section at the probe from retained lines observed in the three cold spectra; not a complete hot-band spectrum.',
                        detected_cold_lines=len(records),retained_cold_lines=sum(eligible(r) for r in records),
                        nearest_fitted_line_center_cm1=nearest['center_cm1'],
                        cold_fit_relative_RMS_threshold=.03,nearest_cold_fit_relative_RMS=nearest['fit_relative_rms'],
                        pressure_handling='Cold Lorentz widths scaled with target pressure and a cold-fitted temperature exponent, then corrected by frozen training model.')
                    metadata['assumptions']=[
                        'Linear-rotor-like partition population approximation with one vibrational scale inferred from the cold band.',
                        'One training-derived Doppler proxy is used for all new spectra; no target mass is inferred.',
                        'Cold nitrogen-bath widths are transferred to the specified target pressure; bath-gas-specific broadening is not inferred.',
                        'Transitions absent from all cold inputs and pressure-dependent line shifts are not generated.']
                    if targets.max()>1800:metadata['warnings'].append('Some target temperatures exceed the line-head training maximum of 1800 K.')
                    if pressures.max()>1.5:metadata['warnings'].append('Some pressures exceed the line-head training maximum of 1.5 atm while remaining inside the study pressure range.')
                    return dict(wavenumber_cm1=np.array([probe]),cross_section_cm2_molecule=sigma[:,None],metadata=metadata)
                metadata['warnings'].append('The nearest detected peak did not pass cold line-fitting checks; the probe uses the broad envelope.')
        broad=self.broad.predict([(x,y) for x,y,T in curves],ts,targets,bounds,
            preserve_native_detail=False,units='cross-section')
        metadata.update(branch='broad_band_envelope',prediction_scope='Original 512-point broad-band envelope after 2 cm-1 smoothing.',
            pressure_handling='The broad model is not pressure-conditioned; supplied pressures are recorded as experiment conditions.',
            preprocessing=broad['metadata']['preprocessing'])
        metadata['warnings'].extend(broad['metadata']['warnings'])
        metadata['assumptions']=['Band-envelope prediction using the frozen measured/HITEMP/HITRAN training model.',
            'Narrow line centers and widths are not resolved by the broad envelope.']
        grid=broad['wavenumber_cm1'];sigma=broad['cross_section_cm2_molecule']
        if probe_wavenumber_cm1 is not None:
            probe=float(probe_wavenumber_cm1)
            sigma=np.array([np.interp(probe,grid,y) for y in sigma])[:,None]
            grid=np.array([probe]);metadata['prediction_scope']='Broad-envelope cross-section interpolated at the requested probe; no line-resolution claim.'
        return dict(wavenumber_cm1=grid,cross_section_cm2_molecule=sigma,metadata=metadata)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spectrum',nargs=2,action='append',required=True,metavar=('FILE','T_K'),help='Repeat exactly three times.')
    parser.add_argument('--units',choices=['pnnl','cm2-molecule'],required=True)
    parser.add_argument('--target-temperature',type=float,nargs='+',required=True,metavar='K')
    parser.add_argument('--target-pressure',type=float,nargs='+',required=True,metavar='ATM')
    parser.add_argument('--context-band',type=float,nargs=2,required=True,metavar=('LOW_CM1','HIGH_CM1'))
    parser.add_argument('--probe',type=float,metavar='CM1',help='Optional single sensing wavenumber; enables local cold-feature routing.')
    parser.add_argument('--output',type=Path,required=True,help='CSV path; matching JSON records branch and assumptions.')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    try:
        if len(args.spectrum)!=3:raise ValueError('Provide exactly three --spectrum FILE T_K arguments.')
        if args.output.suffix.lower()!='.csv':raise ValueError('--output must be a .csv file.')
        paths=[Path(s[0]).resolve() for s in args.spectrum]
        output=args.output.resolve();metapath=output.with_suffix('.json')
        if output in paths or metapath in paths:raise ValueError('Outputs must not overwrite an input spectrum.')
        package_root=Path(__file__).resolve().parent
        manifest_path=package_root/'PACKAGE_MANIFEST.json'
        if manifest_path.exists():
            packaged_files={(package_root/name).resolve() for name in json.loads(manifest_path.read_text(encoding='utf-8'))['files']}
            packaged_files.add(manifest_path.resolve())
        else:
            packaged_files={p.resolve() for p in package_root.rglob('*.py')}
            packaged_files.update(p.resolve() for p in package_root.rglob('*.npz'))
            packaged_files.update(p.resolve() for p in package_root.rglob('*model*.json'))
        if output in packaged_files or metapath in packaged_files:raise ValueError('Outputs must not overwrite package files.')
        if not args.overwrite and (output.exists() or metapath.exists()):raise ValueError('Output already exists; choose a new path or use --overwrite.')
        result=ThreeSpectrumPredictor().predict([core.read_spectrum(p) for p in paths],
            [float(s[1]) for s in args.spectrum],args.target_temperature,args.target_pressure,
            args.context_band,units=args.units,probe_wavenumber_cm1=args.probe)
        meta=result['metadata']
        meta['input_files']=[dict(file_name=p.name,temperature_K=float(s[1]),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,s in zip(paths,args.spectrum)]
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['temperature_K','pressure_atm','wavenumber_cm1','cross_section_cm2_molecule'])
            for T,P,sigma in zip(meta['target_temperatures_K'],meta['target_pressures_atm'],result['cross_section_cm2_molecule']):
                writer.writerows(zip(np.full(len(sigma),T),np.full(len(sigma),P),result['wavenumber_cm1'],sigma))
        metapath.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
        print('Branch:',meta['branch'])
        print('Scope:',meta['prediction_scope'])
        print('Wrote',output,'and',metapath)
        for warning in meta['warnings']:print('Note:',warning)
    except (ValueError,OSError,ImportError) as exc:parser.error(str(exc))

if __name__=='__main__':main()
