"""thIRmo-ET: one spectral prediction function for three near-ambient inputs."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np
from broad_model import frozen_core as core
from spectral_model import ThreeSpectrumPredictor,VERSION

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
