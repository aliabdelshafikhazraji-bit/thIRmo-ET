"""Evaluation only: replace a published calibration and score a time history.

Eq. (7): x_recalculated = x_published * sigma_reference / sigma_predicted.
Eq. (8): E = 100 integral(abs(x_recalculated-x_published),dt)
              / integral(max(x_published,0),dt).

No spectral model is trained or called. Hot reference data supplied to this
comparison script are evaluation inputs, never inputs to thIRmo-ET inference.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

def recalculate(time,published_mole_fraction,reference_cross_section,predicted_cross_section,
                cross_section_units,reported_sigma_fraction=None,reported_x_fraction=None):
    """Return recalculated fractions, integrated error, and optional source bars.

    Both cross-section arrays must use the single declared unit convention.
    Scalar cross-sections are broadcast over time. Time may use any single
    consistent unit, because that unit cancels in the normalized integral.
    Negative measured concentrations are preserved, including measurement noise.
    Optional uncertainty bounds describe the published reference alone. They
    are not model prediction intervals and are not combined with model error.
    """
    t=np.asarray(time,float);x=np.asarray(published_mole_fraction,float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<2:
        raise ValueError('Time and published mole fraction must be matching one-dimensional arrays with at least two samples.')
    if not np.isfinite(t).all() or not np.isfinite(x).all() or np.any(np.diff(t)<=0):
        raise ValueError('Time must be finite and strictly increasing; mole fractions must be finite.')
    if cross_section_units not in ('cm2/molecule','m2/mol','cm2/mol'):
        raise ValueError('Declare the common cross-section units: cm2/molecule, m2/mol, or cm2/mol.')
    def sigma_array(value):
        a=np.asarray(value,float)
        if a.ndim>1 or (a.ndim==1 and len(a) not in (1,len(t))):
            raise ValueError('Each cross-section must be scalar or aligned with every time sample.')
        a=np.broadcast_to(a,t.shape)
        if not np.isfinite(a).all() or np.any(a<=0):
            raise ValueError('Both cross-section arrays must contain finite, strictly positive values.')
        return a
    reference=sigma_array(reference_cross_section);prediction=sigma_array(predicted_cross_section)
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        multiplier=reference/prediction;new=x*multiplier
    if not np.isfinite(multiplier).all() or not np.isfinite(new).all():
        raise ValueError('The cross-section ratio or recalculated concentration is not finite.')
    denominator=float(np.trapezoid(np.maximum(x,0),t))
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('The integrated positive reference concentration must be finite and greater than zero.')
    numerator=float(np.trapezoid(np.abs(new-x),t))
    if not np.isfinite(numerator):raise ValueError('The integrated absolute concentration difference must be finite.')
    result=dict(time=t,published_mole_fraction=x,recalculated_mole_fraction=new,
        reference_cross_section=reference,predicted_cross_section=prediction,cross_section_ratio=multiplier,
        integrated_absolute_difference_percent=100*numerator/denominator,
        integral_absolute_difference=numerator,integral_positive_reference=denominator,
        cross_section_units=cross_section_units,uncertainty_kind=None)
    if reported_sigma_fraction is not None and reported_x_fraction is not None:
        raise ValueError('Specify either reported calibration uncertainty or reported concentration uncertainty, not both.')
    fraction=reported_sigma_fraction if reported_sigma_fraction is not None else reported_x_fraction
    if fraction is not None:
        u=float(fraction)
        if not np.isfinite(u) or not 0<=u<1:raise ValueError('The reported fractional uncertainty must lie between zero and one.')
        if reported_sigma_fraction is not None:
            a=x/(1+u);b=x/(1-u)
            lower=np.minimum(a,b);upper=np.maximum(a,b)
            kind='Published-reference calibration contribution: inverse cross-section propagation at fixed measured absorption'
        else:
            lower=x-u*np.abs(x);upper=x+u*np.abs(x)
            kind='Reported relative uncertainty of the published concentration'
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():raise ValueError('The reported uncertainty bounds are not finite.')
        result.update(published_uncertainty_lower=lower,published_uncertainty_upper=upper,
            reported_fraction=u,uncertainty_kind=kind)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='User-supplied CSV with aligned time, concentration and cross-sections.')
    parser.add_argument('--output',type=Path,required=True,help='Comparison CSV; an adjacent JSON stores the metric and assumptions.')
    parser.add_argument('--time-column',default='time_ms')
    parser.add_argument('--time-unit',choices=['s','ms','us'],default='ms')
    parser.add_argument('--published-column',default='published_mole_fraction')
    parser.add_argument('--reference-column',default='sigma_reference')
    parser.add_argument('--prediction-column',default='sigma_predicted')
    parser.add_argument('--cross-section-units',choices=['cm2/molecule','m2/mol','cm2/mol'],required=True,
        help='Both supplied cross-section columns must use these same units.')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--reported-sigma-fraction',type=float,help='Optional reported calibration contribution, e.g.0.06. Produces reference-only inverse-propagation bounds.')
    group.add_argument('--reported-x-fraction',type=float,help='Optional reported relative concentration uncertainty, e.g.0.059. Produces reference-only symmetric bounds.')
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.output.suffix.lower()!='.csv':raise ValueError('--output must be a CSV file.')
        out=args.output.resolve();meta_path=out.with_suffix('.json');source=args.input.resolve()
        if source in (out,meta_path):raise ValueError('Output must not overwrite the source data.')
        if not args.overwrite and (out.exists() or meta_path.exists()):raise ValueError('Output exists; choose a new path or specify --overwrite.')
        with source.open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        columns=[args.time_column,args.published_column,args.reference_column,args.prediction_column]
        if not rows or any(c not in rows[0] for c in columns):raise ValueError('Input CSV must contain all four requested columns.')
        arrays=[np.asarray([float(r[c]) for r in rows]) for c in columns]
        result=recalculate(*arrays,args.cross_section_units,args.reported_sigma_fraction,args.reported_x_fraction)
        out.parent.mkdir(parents=True,exist_ok=True)
        names=['time_'+args.time_unit,'published_mole_fraction','recalculated_mole_fraction','sigma_reference','sigma_predicted','cross_section_ratio']
        values=[result[k] for k in ['time','published_mole_fraction','recalculated_mole_fraction','reference_cross_section','predicted_cross_section','cross_section_ratio']]
        if result['uncertainty_kind']:
            names+=['published_uncertainty_lower','published_uncertainty_upper']
            values+=[result['published_uncertainty_lower'],result['published_uncertainty_upper']]
        with out.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(names);writer.writerows(zip(*values))
        metadata=dict(evaluation_only=True,model_inference_called=False,fitting_or_normalization=False,
            source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),input_columns=columns,
            time_unit=args.time_unit,mole_fraction_unit='mol/mol',cross_section_units=args.cross_section_units,
            cross_section_units_basis='User declares identical units and convention for both cross-section arrays; this script does not infer or convert them.',
            calculation='x_recalculated=x_published*sigma_reference/sigma_predicted',
            metric='100 integral(abs(x_recalculated-x_published),dt)/integral(max(x_published,0),dt)',
            integrated_absolute_difference_percent=result['integrated_absolute_difference_percent'],
            integral_absolute_difference=result['integral_absolute_difference'],integral_positive_reference=result['integral_positive_reference'],
            uncertainty_kind=result['uncertainty_kind'],reported_fraction=result.get('reported_fraction'),
            uncertainty_scope='Bounds apply to the published reference only, retain the source uncertainty interpretation, and do not estimate model uncertainty.',
            assumptions='Reference and predicted cross-sections correspond to the same species, sensing wavenumber and gas conditions. Published concentration corrections are retained.')
        meta_path.write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
        print(f'Integrated absolute concentration difference: {result["integrated_absolute_difference_percent"]:.6g}%')
        print('Wrote',out,'and',meta_path)
    except (ValueError,OSError,KeyError) as exc:parser.error(str(exc))

if __name__=='__main__':main()
