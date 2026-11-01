"""Reproduce the three-spectrum-only line model from externally supplied arrays.

No NO hot data are loaded. Raw source spectra and line arrays are not distributed.
"""
from pathlib import Path
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'three_input'))
from line_model.cold_line_representation import fit_line,cold_band_centroid,add_cold_thermal_parameters,PNNL_COLUMN
import train_helpers as h
DATA=OUT=ARTIFACTS=PNNLROOT=None
SPECIES=['CO','HCl','HF']
WINDOWS={'CO':[1800,2300],'HCl':[2400,3150],'HF':[3500,4300]}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fit_cold(selection,mass_proxy,tag):
    records=[]
    for s in selection:
        curves=[]
        for d,T in zip(s['pnnl_inputs'],s['input_temperatures_K']):
            a=np.loadtxt(d['file']);a=a[np.argsort(a[:,0])]
            curves.append((a[:,0],a[:,1]*np.log(10)/PNNL_COLUMN,float(T)))
        vib=cold_band_centroid(curves,WINDOWS[s['species']])
        for i,line in enumerate(s['lines']):
            x,y,_=curves[-1];mask=abs(x-line['nu'])<.12
            center=float(x[mask][np.argmax(y[mask])])
            r=fit_line(curves,center,mass_proxy,'sinc_0.056')
            r=add_cold_thermal_parameters(r,vib)
            r.update(species=s['species'],line_index=i,reference_center_for_matching_only=line['nu'],
                cold_vibrational_proxy_window_cm1=WINDOWS[s['species']],
                mass_basis='Equal-species mean training Doppler proxy',
                usable_cold_representation=bool(r['fit_success'] and r['fit_relative_rms']<.03 and not r['fit_at_bound']))
            records.append(r)
        print(tag,s['species'],'fitted',len(s['lines']),'cold lines',flush=True)
    dest=ARTIFACTS/(tag+'_cold_records.json')
    dest.write_text(json.dumps(dict(records=records,mass_proxy_amu=mass_proxy,
        no_hot_references_used=True),indent=2),encoding='utf-8')
    return dest

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    selected_path=DATA/'selected_lines.json'
    selection=json.loads(selected_path.read_text())
    # Resolve every external cold input under the explicitly supplied PNNL root.
    # Original workstation directories are neither required nor accessed.
    for species in selection:
        for entry in species['pnnl_inputs']:
            source_path=Path(str(entry['file']).replace(chr(92),'/'))
            mapped=PNNLROOT/source_path.parent.name/source_path.name
            if not mapped.is_file():raise FileNotFoundError(mapped)
            expected_hash=entry.get('file_sha256') or entry.get('sha256')
            if expected_hash and sha(mapped)!=expected_hash:
                raise ValueError('PNNL input checksum mismatch: '+mapped.name)
            entry['file']=str(mapped)
    assert sorted(s['species'] for s in selection)==sorted(SPECIES)
    masses={s['species']:s['mass_main_isotopologue_u'] for s in selection}
    proxy=float(np.mean(list(masses.values())))
    protocol=dict(candidate='Three-spectra line branch with training-derived constant Doppler proxy',
        created_utc=datetime.now(timezone.utc).isoformat(),
        allowed_training_species=SPECIES,forbidden_for_training='All NO hot spectra, cross sections, concentration histories and performance scores',
        molecular_inference_inputs='Three near-ambient cross-section spectra with their measured temperatures; no target species identity, formula, mass, line list or measured hot spectrum',
        operational_inputs='Context wavenumber interval, target wavenumber grid, target temperature and pressure; input spectra pressure defaults to 1 atm',
        doppler_proxy_rule='Equal-species arithmetic mean of training isotopologue masses. Each LOSO fold uses only the two fitting species; the final model uses all three.',
        training_species_masses_amu=masses,final_doppler_proxy_amu=proxy,
        removed_feature='mass_per100amu',feature_names=h.FEATURE_NAMES,parameters=h.PARAMS,
        cold_refit='Refit every training and evaluation cold spectrum with the applicable fixed Doppler proxy; unchanged sinc first-zero0.056cm^-1, optimization, peak and quality rules',
        branch_selection='Select residual head only if equal-species LOSO mean local integrated L1 is strictly below physical baseline',
        evaluation_plan='Freeze model, configuration, code and protocol hashes; then fit NO cold inputs with frozen proxy; then evaluate existing three NO histories once with no recalibration',
        scope='Post hoc exploratory extension motivated by earlier narrow-model results; this is not a newly blinded external test. Linear-rotor population approximation remains and is not validated for every polyatomic resolved band.',
        selected_training_source_sha256=sha(selected_path))
    protocol_path=ARTIFACTS/'protocol_before_training.json'
    if protocol_path.exists():
        raise RuntimeError('A protocol already exists: do not silently rerun or alter the candidate after testing.')
    protocol_path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    h.DATA=DATA
    baseline=[];residual=[];fold_reports=[]
    for held in SPECIES:
        fold_proxy=float(np.mean([masses[s] for s in SPECIES if s!=held]))
        path=fit_cold(selection,fold_proxy,'fold_hold_'+held)
        h.COLD=path
        records,datasets,decisions=h.load_training()
        training=[r for r in records if r['species']!=held]
        testing=[r for r in records if r['species']==held]
        assert training and testing
        model=h.train(training,datasets)
        p=h.evaluate(testing,datasets);q=h.evaluate(testing,datasets,model)
        baseline+=p;residual+=q
        report=dict(held_species=held,proxy_amu=fold_proxy,physical=h.summarize(p),residual=h.summarize(q),cold_records_sha256=sha(path))
        fold_reports.append(report)
        print('LOSO',json.dumps(report),flush=True)
    p=h.summarize(baseline);q=h.summarize(residual)
    choose=q['mean_species_local_integrated_L1_percent']<p['mean_species_local_integrated_L1_percent']
    finalcold=fit_cold(selection,proxy,'final_training')
    h.COLD=finalcold
    records,datasets,decisions=h.load_training()
    final=h.train(records,datasets)
    weights=OUT/'narrow_residual_forest.npz'
    h.export_numeric(final,weights)
    config=dict(schema_version=2,
        selected_branch='physical_plus_extra_trees_residual' if choose else 'physical_baseline',
        doppler_mass_proxy_amu=proxy,doppler_mass_proxy_basis=protocol['doppler_proxy_rule'],
        molecular_mass_is_inference_input=False,species_identity_is_inference_input=False,
        training_species=SPECIES,parameters=h.PARAMS,feature_names=h.FEATURE_NAMES,features_count=len(h.FEATURE_NAMES),
        numeric_model_sha256=sha(weights),cold_records_sha256=sha(finalcold),
        line_filter_decisions=decisions,line_counts={sp:sum(r['species']==sp for r in records) for sp in SPECIES},
        validation=dict(physical_baseline=p,residual_head=q,folds=fold_reports),
        temperature_training_K=sorted({float(t) for z in datasets.values() for t in z['temperatures_K'] if t>=350}),
        pressure_training_atm=sorted({float(p) for z in datasets.values() for p in z['pressures_atm']}),
        physical_proxy='Linear rigid rotor plus one harmonic vibration; vibrational scale is warm-input band centroid; lower-energy scale is inferred from three cold line areas',
        no_hot_NO_consumed=True,selection_rule=protocol['branch_selection'],scope=protocol['scope'])
    configpath=OUT/'narrow_model_config.json'
    configpath.write_text(json.dumps(config,indent=2),encoding='utf-8')
    for name,rows in [('physical_LOSO.csv',baseline),('residual_LOSO.csv',residual)]:
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from line_model.spectra_only_line import SpectraOnlyNarrowHead,feature_rows
    frozen=SpectraOnlyNarrowHead(OUT)
    X=np.concatenate([feature_rows(r,[400,800,1500]) for r in records])
    expected=final.predict(X) if choose else np.zeros((len(X),2))
    np.testing.assert_allclose(frozen.correction(X),expected,rtol=1e-12,atol=1e-14)
    freeze=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        hot_NO_test_not_run=True,protocol_sha256=sha(protocol_path),
        weights_sha256=sha(weights),config_sha256=sha(configpath),
        inference_code_sha256=sha(REPO/'three_input/line_model/spectra_only_line.py'),
        cold_physics_code_sha256=sha(REPO/'three_input/line_model/cold_line_representation.py'),
        training_script_sha256=sha(Path(__file__)),training_helpers_sha256=sha(HERE/'train_helpers.py'),
        training_cold_records_sha256=sha(finalcold),
        input_data_sha256={sp:sha(DATA/f'{sp}_native_training.npz') for sp in SPECIES},
        no_target_fitting=True,numeric_evaluator_verified_against_sklearn=True)
    (OUT/'frozen_manifest.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
    print('FROZEN',json.dumps(freeze),flush=True)
    print('BASELINE',p['mean_species_local_integrated_L1_percent'],'RESIDUAL',q['mean_species_local_integrated_L1_percent'],'SELECTED',config['selected_branch'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True,help='External selected_lines.json, cold_gate.json, and three native training NPZ files.')
    parser.add_argument('--pnnl-root',type=Path,required=True,help='Directory containing the original CO, HCl, and HF PNNL species folders.')
    parser.add_argument('--output-dir',type=Path,required=True,help='New output directory for reproduced weights and audits.')
    args=parser.parse_args()
    DATA=args.data_dir.resolve();PNNLROOT=args.pnnl_root.resolve();OUT=args.output_dir.resolve()
    ARTIFACTS=OUT/'audit';ARTIFACTS.mkdir(parents=True,exist_ok=True)
    main()
