"""Score the shared spectral function using acquired cold and reference files."""
from pathlib import Path
import sys,json,csv,argparse,hashlib
import numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'three_input'));sys.path.insert(0,str(REPO))
from spectral_model import ThreeSpectrumPredictor
from broad_model import frozen_core as core
from reproduction.pipeline import load_dataset,summarize

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--native-input-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    specs=json.loads(args.native_input_manifest.read_text(encoding='utf8'))
    groups=load_dataset(args.dataset)
    models={'final':ThreeSpectrumPredictor(),
        'without_hitran':ThreeSpectrumPredictor(REPO/'three_input/broad_model/models/thirmo_et_without_added_hitran.npz')}
    rows=[]
    for g in groups:
        if g['split']!='testing':continue
        if len(g['inputs'])!=3 or len(np.unique(g['input_temperatures_K']))!=3:raise ValueError('Three actual cold inputs required.')
        s=specs[g['window']];curves=[]
        for i,name in enumerate(s['files']):
            path=args.native_input_manifest.parent/name
            if s.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest()!=s['sha256'][i]:raise ValueError('Native input checksum differs.')
            curves.append(core.read_spectrum(path))
        context=models['final'].prepare(curves,g['input_temperatures_K'],[g['x'][0],g['x'][-1]],s['units'])
        T=[r['temperature_K'] for r in g['records']];P=[r['pressure_atm'] for r in g['records']]
        predictions={}
        for name,model in models.items():
            # The supervised envelope already has the reference's 2 cm-1 response.
            predictions[name]=model.evaluate(context,T,P,g['x'],output_fwhm_cm1=core.FWHM)
        for i,r in enumerate(g['records']):
            if not r.get('primary_evaluation',True):continue
            m=g['masks'][i];ref=g['references'][i];den=np.maximum(ref[m],0).sum()
            row=dict(species=g['species'],window=g['window'],record_id=r['record_id'],temperature_K=r['temperature_K'],pressure_atm=r['pressure_atm'],source_kind=r['source_kind'])
            for name in models:row[name+'_error_percent']=float(100*abs(predictions[name][i,m]-ref[m]).sum()/den)
            row['unchanged_error_percent']=float(100*abs(g['inputs'][-1,m]-ref[m]).sum()/den);rows.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'record_errors.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={key:summarize([dict(r,error_percent=r[key+'_error_percent']) for r in rows]) for key in ['final','without_hitran','unchanged']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(args.output)

if __name__=='__main__':main()
