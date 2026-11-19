"""thIRmo-ET: one spectral prediction function for three near-ambient inputs (version 5.0.0).

Routing follows the manuscript (Section 2.9). The fixed cold-spectrum criteria and the cold line fitter are applied
to the three inputs. When they retain qualifying isolated lines, the molecule is line-resolved: the published
envelope and the retained lines are used (spectral_model.py, frozen weights, unchanged). Otherwise the molecule is
broad-band: the near-ambient band shift and broadening are extrapolated physically with conserved band area and a
correction learned from other molecules is added (broad_band/). Requesting a probe changes only the output
coordinates; it never selects another model.
"""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from broad_model import frozen_core as core
from spectral_model import ThreeSpectrumPredictor,VERSION as LINE_RESOLVED_BRANCH_VERSION
from broad_band import broad_band as BB

VERSION='5.0.0'
TESTED_BROAD_K=974.22      # highest temperature of the experimental test spectra (broad-band molecules)
TESTED_LINE_K=1573.15      # highest temperature of the HITEMP test references (line-resolved molecules)

def predict(spectra,input_temperatures_K,target_temperatures_K,target_pressures_atm,context_window_cm1,units='cm2/molecule',
            probe_wavenumber_cm1=None,wavenumber_grid_cm1=None,output_fwhm_cm1=0.):
    """Routed thIRmo-ET prediction.

    Returns a dict with wavenumber_cm1, cross_section_cm2_molecule (one row per target temperature), interval90
    (calibrated 90% prediction interval of the broad-band branch as dict(low, high); None for line-resolved
    molecules) and metadata (branch, hashes, assumptions, warnings)."""
    line=ThreeSpectrumPredictor()
    context=line.prepare(spectra,input_temperatures_K,context_window_cm1,units)
    if context['line_records']:
        result=line.predict(spectra,input_temperatures_K,target_temperatures_K,target_pressures_atm,context_window_cm1,units=units,
            probe_wavenumber_cm1=probe_wavenumber_cm1,wavenumber_grid_cm1=wavenumber_grid_cm1,output_fwhm_cm1=output_fwhm_cm1)
        meta=result['metadata']
        meta.update(version=VERSION,branch='line_resolved',line_resolved_branch_version=LINE_RESOLVED_BRANCH_VERSION,
            routing='The cold inputs retain qualifying isolated lines: published envelope plus retained lines (frozen weights).')
        T=np.atleast_1d(np.asarray(target_temperatures_K,float))
        if T.max()>TESTED_LINE_K:meta['warnings'].append('Some output temperatures exceed the highest test temperature of line-resolved molecules (1573 K); the envelope does not change with temperature beyond its training temperatures.')
        result['interval90']=None
        return result
    # broad-band branch
    T=np.atleast_1d(np.asarray(target_temperatures_K,float));P=np.atleast_1d(np.asarray(target_pressures_atm,float))
    if T.ndim!=1 or not len(T) or not np.isfinite(T).all() or np.any(T<=max(map(float,input_temperatures_K))):raise ValueError('Target temperatures must exceed the warmest input.')
    if P.ndim!=1 or len(P) not in (1,len(T)) or not np.isfinite(P).all():raise ValueError('One pressure or one pressure per target temperature is required.')
    P=np.broadcast_to(P,T.shape)
    if np.any((P<.6)|(P>2)):raise ValueError('Target pressures must lie within 0.6-2.0 atm.')
    resolution=float(output_fwhm_cm1)
    if not np.isfinite(resolution) or resolution<0:raise ValueError('Output FWHM must be finite and nonnegative.')
    if probe_wavenumber_cm1 is not None and wavenumber_grid_cm1 is not None:raise ValueError('Supply a grid or a probe, not both.')
    g=BB.prepare(spectra,input_temperatures_K,context_window_cm1,'pnnl' if units=='pnnl' else 'cm2/molecule');model=BB.BroadBandPredictor()
    if probe_wavenumber_cm1 is not None:x=np.array([float(probe_wavenumber_cm1)])
    elif wavenumber_grid_cm1 is not None:x=np.asarray(wavenumber_grid_cm1,float)
    else:x=g['x']
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or np.any((x<g['x'][0])|(x>g['x'][-1])):raise ValueError('Requested wavenumbers must lie within the supplied context band.')
    values=[];low=[];high=[]
    for t in T:
        y,lo,hi=model.predict(g,t,with_interval=True)
        if resolution>core.FWHM:   # the output is learned at 2 cm-1; only the additional Gaussian variance is applied
            s=np.sqrt(resolution**2-core.FWHM**2)/np.sqrt(8*np.log(2))/(g['x'][1]-g['x'][0])
            y,lo,hi=[gaussian_filter1d(a,s,mode='nearest',truncate=6) for a in (y,lo,hi)]
        values.append(np.interp(x,g['x'],y));low.append(np.interp(x,g['x'],lo));high.append(np.interp(x,g['x'],hi))
    r2,nue,rho=BB.ph.shift_law(g['cp'])
    warnings=[]
    if T.max()>TESTED_BROAD_K:warnings.append('Some output temperatures exceed the highest experimental test spectrum (974 K) and are extrapolations.')
    if T.max()>BB.T_CAP:warnings.append('Above 1000 K the learned correction is not updated; the 1000 K spectrum is carried by the physical shift and broadening.')
    if resolution<core.FWHM:warnings.append('Broad-band output is the learned 2 cm-1 response; no isolated-line contrast is retained for finer responses.')
    meta=dict(tool='thIRmo-ET',version=VERSION,branch='broad_band',input_count=3,input_temperatures_K=g['Tin'].tolist(),
        target_temperatures_K=T.tolist(),target_pressures_atm=P.tolist(),context_window_cm1=[float(g['x'][0]),float(g['x'][-1])],
        routing='No qualifying isolated line is retained by the cold criteria: physical stage plus learned correction.',
        molecular_identity_input=False,molecular_mass_input=False,molecular_properties_input=False,hot_reference_input=False,
        broad_band_weights_sha256=model.forest.sha256,
        cold_shift_cm1=dict(first_interval=float(g['cp']['d12']),second_interval=float(g['cp']['d23'])),
        rate_ratio=None if rho!=rho else float(rho),effective_wavenumber_cm1=float(nue),cold_broadening_cm1=float(g['cp']['w13']),
        correction_updated_up_to_K=BB.T_CAP,interval=f'calibrated 90% prediction interval, q(s_tree + 0.02 max) with q = {BB.Q_INTERVAL:.3f}',
        output_units='cm2/molecule (natural-log convention)',input_units=units,
        output_gaussian_fwhm_cm1=float(max(resolution,core.FWHM)),envelope_training_gaussian_fwhm_cm1=core.FWHM,
        pressure_handling='Broad-band output does not depend on pressure.',
        retained_cold_line_count=0,rejected_cold_fits=context['rejected_cold_fits'],
        validated_range='experimental test spectra up to 974 K; higher temperatures are extrapolations',warnings=warnings)
    return dict(wavenumber_cm1=x,cross_section_cm2_molecule=np.array(values),interval90=dict(low=np.array(low),high=np.array(high)),metadata=meta)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spectrum',nargs=2,action='append',required=True,metavar=('FILE','T_K'),help='Repeat exactly three times.')
    parser.add_argument('--units',choices=['pnnl','cm2-molecule'],required=True)
    parser.add_argument('--target-temperature',type=float,nargs='+',required=True,metavar='K')
    parser.add_argument('--target-pressure',type=float,nargs='+',required=True,metavar='ATM')
    parser.add_argument('--context-band',type=float,nargs=2,required=True,metavar=('LOW_CM1','HIGH_CM1'))
    parser.add_argument('--probe',type=float,metavar='CM1',help='Optional single sensing wavenumber; uses the same prediction function.')
    parser.add_argument('--output-fwhm-cm1',type=float,default=0.,metavar='CM1',help='Requested Gaussian FWHM: 0 keeps fine output for retained lines; 2 returns the learned band response.')
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
        units='pnnl' if args.units=='pnnl' else 'cm2/molecule'
        result=predict([core.read_spectrum(p) for p in paths],[float(s[1]) for s in args.spectrum],args.target_temperature,args.target_pressure,
            args.context_band,units=units,probe_wavenumber_cm1=args.probe,output_fwhm_cm1=args.output_fwhm_cm1)
        meta=result['metadata']
        meta['input_files']=[dict(file_name=p.name,temperature_K=float(s[1]),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,s in zip(paths,args.spectrum)]
        output.parent.mkdir(parents=True,exist_ok=True)
        band=result['interval90']
        with output.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream)
            writer.writerow(['temperature_K','pressure_atm','wavenumber_cm1','cross_section_cm2_molecule']+(['interval90_low','interval90_high'] if band else []))
            for k,(T,P,sigma) in enumerate(zip(meta['target_temperatures_K'],meta['target_pressures_atm'],result['cross_section_cm2_molecule'])):
                cols=[np.full(len(sigma),T),np.full(len(sigma),P),result['wavenumber_cm1'],sigma]+([band['low'][k],band['high'][k]] if band else [])
                writer.writerows(zip(*cols))
        metapath.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
        print('Branch:',meta['branch'])
        if 'prediction_scope' in meta:print('Scope:',meta['prediction_scope'])
        print('Wrote',output,'and',metapath)
        for warning in meta['warnings']:print('Note:',warning)
    except (ValueError,OSError,ImportError) as exc:parser.error(str(exc))

if __name__=='__main__':main()
