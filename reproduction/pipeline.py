"""Self-contained study pipeline; licensed source spectra are supplied separately.

Run ``python -m reproduction.pipeline --help`` from the repository root.
The preparation, feature, weighting and scoring definitions match the study.
There are no imports from an author's private workspace and no pickle inputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve
import thirmo_et as inference

PARAMETERS = dict(n_estimators=128, max_depth=16, min_samples_leaf=16,
                  max_features=.8, random_state=20261029, n_jobs=3)
NA = 6.02214076e23
KB = 1.380649e-23


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    """Digest includes dtype, shape and C-order bytes; preserves missing values."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf8', newline='\n')


def canonical_cross_section(values, units, temperature, column_density=None):
    """Convert published ordinate units to natural-log cm2/molecule."""
    if units == 'm2/mol':
        return np.asarray(values) * 1e4 / NA
    if units == 'cm2/mol':
        return np.asarray(values) / NA
    if units == 'Pa-1 m-1':
        return np.asarray(values) * KB * float(temperature) * 1e4
    return inference.to_cross_section(values, units, column_density)


def read_curve(base, specification):
    path = Path(base) / specification['file']
    if specification.get('sha256') and sha(path) != specification['sha256']:
        raise ValueError(f'Source checksum differs: {path.name}')
    a = np.loadtxt(path, delimiter=specification.get('delimiter', ','),
                   skiprows=specification.get('skiprows', 1), ndmin=2)
    if a.shape[1] < 2:
        raise ValueError(f'Two columns required in {path.name}')
    x = a[:, specification.get('wavenumber_column', 0)]
    y = canonical_cross_section(a[:, specification.get('value_column', 1)],
        specification['units'], specification['temperature_K'],
        specification.get('column_density_molecules_cm2'))
    return x, y


def observed_segments(x, y, maximum_gap=5.):
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    good = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
    if not len(good):
        return []
    cuts = (np.diff(good) > 1) | (np.diff(x[good]) > maximum_gap)
    return [(x[j], y[j]) for j in np.split(good, np.flatnonzero(cuts)+1) if len(j)>1]


def smooth_record(x, y, grid, trim=3.):
    """Apply 2 cm-1 Gaussian within observed segments, retaining reference masks."""
    out = np.full(len(grid), np.nan); mask = np.zeros(len(grid), bool)
    for sx, sy in observed_segments(x, y):
        original_lo, original_hi = sx[0], sx[-1]
        step = float(np.median(np.diff(sx)))
        margin = 6*inference.SIGMA_CM1 + 2*step
        keep = (sx >= grid.min()-margin) & (sx <= grid.max()+margin)
        sx, sy = sx[keep], sy[keep]
        if len(sx)<2:
            continue
        dx = (sx[-1]-sx[0])/(len(sx)-1)
        uniform = np.max(np.abs(np.diff(sx)-dx)) <= max(1e-5, 1e-3*dx)
        covered = (grid>=sx[0]) & (grid<=sx[-1])
        if not np.any(covered):
            continue
        if uniform and inference.SIGMA_CM1/dx > 30:
            sigma = inference.SIGMA_CM1/dx; radius = int(6*sigma+.5)
            offsets = np.arange(-radius,radius+1,dtype=float)
            kernel = np.exp(-.5*(offsets/sigma)**2); kernel /= kernel.sum()
            filtered = fftconvolve(np.pad(sy,radius,mode='edge'),kernel,mode='valid')
            values = np.interp(grid[covered],sx,filtered)
        else:
            values, _ = inference.native_smooth(sx,sy,grid[covered])
        out[covered] = values
        mask |= covered & (grid>=original_lo+trim) & (grid<=original_hi-trim)
    return out, mask


def prepare(source_manifest, output):
    """Prepare supplied two-column source curves using the declared band windows."""
    source_manifest = Path(source_manifest); output = Path(output)
    source = json.loads(source_manifest.read_text(encoding='utf8'))
    groups = []
    for number, item in enumerate(source['groups']):
        x = np.linspace(*item['window_cm1'],inference.N)
        inputs = sorted(item['inputs'], key=lambda r:r['temperature_K'])
        ts = np.asarray([r['temperature_K'] for r in inputs],float)
        p = np.stack([smooth_record(*read_curve(source_manifest.parent,r),x,trim=0)[0] for r in inputs])
        inference.make_features(p,ts,x)  # validates full cold-input coverage
        references=[]; masks=[]; records=[]
        for target in item['targets']:
            T = float(target['temperature_K']); P = float(target['pressure_atm'])
            if T<=max(ts) or not .6<=P<=2.:
                raise ValueError('Manifest contains an out-of-scope target; do not silently filter it.')
            y, mask = smooth_record(*read_curve(source_manifest.parent,target),x)
            if mask.sum()<25 or np.maximum(y[mask],0).sum()<=0:
                raise ValueError('Insufficient valid reference coverage.')
            references.append(y); masks.append(mask)
            records.append({k:v for k,v in target.items() if k not in ['file','delimiter','skiprows','units']})
        meta=dict(species=item['species'],window=item['window'],split=item['split'],
                  input_origin=item.get('input_origin','PNNL'),records=records,
                  inputs=[{k:v for k,v in r.items() if k!='file'} for r in inputs])
        meta['file']=f'group_{number:03d}.npz'
        arrays=dict(x=x,inputs=p,input_temperatures_K=ts,references=np.stack(references),masks=np.stack(masks))
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/meta['file'],**arrays)
        meta['sha256']=sha(output/meta['file'])
        meta['array_checksums']={k:array_sha(a) for k,a in arrays.items()}
        groups.append(meta)
    manifest=dict(schema_version=1,groups=groups,source_manifest_sha256=sha(source_manifest))
    write_json(output/'dataset_manifest.json',manifest)
    return manifest


def load_dataset(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'dataset_manifest.json').read_text(encoding='utf8'))
    groups=[]
    for item in manifest['groups']:
        path=directory/item['file']
        if not path.is_file():
            raise FileNotFoundError(f'Missing externally supplied spectrum bundle: {path}. See reproduction/README.md.')
        if item.get('sha256') and sha(path)!=item['sha256']:
            raise ValueError(f'Dataset file checksum differs: {path.name}')
        with np.load(path,allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        for k,digest in item.get('array_checksums',{}).items():
            if array_sha(arrays[k])!=digest:
                raise ValueError(f'Array checksum differs: {path.name}:{k}')
        features,area,anchor,_,_=inference.make_features(arrays['inputs'],arrays['input_temperatures_K'],arrays['x'])
        group=dict(item,**arrays,features=features,area=area,anchor=anchor)
        assert len(group['records'])==len(group['references'])==len(group['masks'])
        groups.append(group)
    train={g['species'] for g in groups if g['split']!='testing'}
    test={g['species'] for g in groups if g['split']=='testing'}
    if train&test:
        raise ValueError('Molecular identity appears in both fitting and testing.')
    return groups


def training_arrays(groups, hitran_weight=.5):
    X=[];y=[];w=[];audit=[]
    for block,total in [('training',1.),('hitran',hitran_weight)]:
        selected=[g for g in groups if g['split']==block]
        if total==0:continue
        ids={g['species'] for g in selected}
        bands={s:sum(g['species']==s for g in selected) for s in ids}
        for g in selected:
            domains={r['source_kind'] for r in g['records']}
            counts={d:sum(r['source_kind']==d for r in g['records']) for d in domains}
            subtotal=0.
            for i,r in enumerate(g['records']):
                m=g['masks'][i];n=int(m.sum())
                X.append(inference.features_at(g['features'],g['input_temperatures_K'][-1],r['temperature_K'])[m])
                y.append(g['references'][i,m]*np.ptp(g['x'])/g['area']-g['anchor'][m])
                weight=total/bands[g['species']]/len(domains)/counts[r['source_kind']]/n
                w.extend([weight]*n);subtotal+=weight*n
            audit.append(dict(species=g['species'],window=g['window'],block=block,raw_weight=subtotal))
    w=np.asarray(w);w*=len(w)/w.sum()
    return np.concatenate(X),np.concatenate(y),w,audit


def export_forest(model,path,metadata):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    offsets=np.r_[0,np.cumsum([t.tree_.node_count for t in model.estimators_])]
    arrays={a:np.concatenate([getattr(t.tree_,b) for t in model.estimators_]) for a,b in
            [('left','children_left'),('right','children_right'),('feature','feature'),('threshold','threshold')]}
    arrays.update(value=np.concatenate([t.tree_.value[:,0,0] for t in model.estimators_]),offsets=offsets,
                  n_features=np.asarray(model.n_features_in_,dtype=np.int64))
    np.savez_compressed(path,**arrays)
    write_json(path.with_suffix('.json'),dict(metadata,weights_sha256=sha(path),parameters=PARAMETERS))


def train(dataset,output,hitran_weight=.5):
    from sklearn.ensemble import ExtraTreesRegressor
    groups=load_dataset(dataset)
    X,y,w,audit=training_arrays(groups,hitran_weight)
    model=ExtraTreesRegressor(**PARAMETERS)
    model.fit(X,y,sample_weight=w)
    export_forest(model,output,dict(hitran_weight=hitran_weight,fit_rows=len(y),weight_audit=audit,
        training_identities=sorted({g['species'] for g in groups if g['split']=='training' or (g['split']=='hitran' and hitran_weight>0)}),
        testing_identities=sorted({g['species'] for g in groups if g['split']=='testing'})))


def summarize(rows):
    output={}
    for label,selected in [('all',rows),('measurement',[r for r in rows if r['source_kind']=='measurement']),
                            ('HITEMP',[r for r in rows if r['source_kind']=='HITEMP'])]:
        by_species={}
        for s in sorted({r['species'] for r in selected}):
            windows={r['window'] for r in selected if r['species']==s}
            by_species[s]=float(np.mean([np.mean([r['error_percent'] for r in selected if r['species']==s and r['window']==b]) for b in windows]))
        output[label]=dict(n_molecules=len(by_species),n_records=len(selected),
                           mean_error_percent=float(np.mean(list(by_species.values()))) if by_species else None,
                           molecule_error_percent=by_species)
    return output


def evaluate(dataset,model,output):
    groups=load_dataset(dataset);forest=inference.NumericForest(model);rows=[]
    for g in groups:
        if g['split']!='testing':continue
        for i,r in enumerate(g['records']):
            if not r.get('primary_evaluation',True):continue
            X=inference.features_at(g['features'],g['input_temperatures_K'][-1],r['temperature_K'])
            y=np.maximum((g['anchor']+forest.predict(X))*g['area']/np.ptp(g['x']),0)
            m=g['masks'][i];reference=g['references'][i]
            error=100*float(np.abs(y[m]-reference[m]).sum()/np.maximum(reference[m],0).sum())
            rows.append(dict(species=g['species'],window=g['window'],record_id=r['record_id'],
                             temperature_K=r['temperature_K'],pressure_atm=r['pressure_atm'],
                             source_kind=r['source_kind'],error_percent=error))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'summary.json',summarize(rows))
    with (output/'record_errors.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--source-manifest',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train');p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);p.add_argument('--hitran-weight',type=float,default=.5)
    p=sub.add_parser('evaluate');p.add_argument('--dataset',required=True);p.add_argument('--model',default=str(inference.DEFAULT_MODEL));p.add_argument('--output',required=True)
    p=sub.add_parser('aggregate');p.add_argument('--errors',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.source_manifest,args.output)
    elif args.action=='train':train(args.dataset,args.output,args.hitran_weight)
    elif args.action=='evaluate':evaluate(args.dataset,args.model,args.output)
    else:
        with open(args.errors,newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
        for r in rows:r['error_percent']=float(r['error_percent'])
        write_json(args.output,summarize(rows))


if __name__=='__main__':
    main()
