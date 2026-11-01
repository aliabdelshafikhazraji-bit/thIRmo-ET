"""Broad-band inference from exactly three measured ambient spectra.

No species name, hot reference spectrum, molecular constants, or external source
lookup is accepted. The fitted ExtraTrees model uses only genuine three-spectrum training records.
Optional native-detail transport carries forward the warmest cold contrast; it
does not infer new hot lines or their broadening.
"""
from pathlib import Path
import numpy as np
from . import frozen_core as core


class ThreeInputBroadModel:
    """Three arrays plus their temperatures, band boundaries and output T.

    Input temperatures must be distinct and within 275-350 K. This is an
    training and inference requirement. Every fitted group and every withheld
    group contains three actual cold measurements; no input is synthesized.
    """
    def __init__(self, model_path=None):
        path=Path(model_path) if model_path is not None else Path(__file__).parent/'models/thirmo_et_final.npz'
        self.forest=core.NumericForest(path)

    def predict(self, spectra, input_temperatures_K, target_temperatures_K,
                window_cm1, preserve_native_detail=False, units='cross-section',
                input_columns_molecules_cm2=None):
        """Return spectra in cmÂ²/molecule (natural-log convention).

        spectra: exactly three (wavenumber_cm1, ordinate) array pairs.
        units: cross-section [cmÂ²/molecule], pnnl, absorbance-ln, or
        absorbance-decadic. Absorbance needs known column densities.
        window_cm1: two boundaries covered by every input spectrum.
        target_temperatures_K: scalar or one-dimensional temperature array.
        No high-temperature references enter preprocessing or prediction.
        """
        ts=np.asarray(input_temperatures_K,float)
        targets=np.atleast_1d(np.asarray(target_temperatures_K,float))
        if len(spectra)!=3 or ts.shape!=(3,):
            raise ValueError('Exactly three measured spectra and three input temperatures are required.')
        if not np.all(np.isfinite(ts)) or len(np.unique(ts))!=3 or np.any((ts<275)|(ts>350)):
            raise ValueError('Three distinct measured input temperatures within 275-350 K are required.')
        if targets.ndim!=1 or not len(targets) or not np.all(np.isfinite(targets)) or np.any(targets<=ts.max()):
            raise ValueError('Every target temperature must exceed the warmest measured input.')
        bounds=np.asarray(window_cm1,float)
        if bounds.shape!=(2,) or not np.all(np.isfinite(bounds)) or not 0<bounds[0]<bounds[1]:
            raise ValueError('Supply two finite increasing positive band boundaries.')
        low,high=bounds
        if input_columns_molecules_cm2 is None:cols=[None]*3
        else:
            cols=list(np.atleast_1d(input_columns_molecules_cm2))
            if len(cols)==1:cols*=3
            if len(cols)!=3:raise ValueError('Supply one shared column density or three column densities.')
        order=np.argsort(ts);ts=ts[order]
        native=[]
        for i in order:
            x,y=core.unique_sorted(*spectra[i])
            if low<x[0] or high>x[-1]:
                raise ValueError('All three input spectra must cover the entire requested band.')
            native.append((x,core.to_cross_section(y,units,cols[i])))
        grid=np.linspace(low,high,core.N)
        prepared=[];preprocessing=[]
        for x,y in native:
            pp,detail=core.native_smooth(x,y,grid)
            prepared.append(pp);preprocessing.append(detail)
        prepared=np.asarray(prepared)
        f,area,anchor,_,_=core.make_features(prepared,ts,grid)
        broad=np.asarray([np.maximum((anchor+self.forest.predict(core.features_at(f,ts[-1],T)))*area/(high-low),0) for T in targets])
        # Work on the warmest source's native coordinates. Include exact band
        # boundaries by the same edge handling as the earlier detail experiment.
        nx,ny=native[-1];keep=(nx>=low)&(nx<=high)
        if keep.sum()<2:raise ValueError('At least two native points must fall within the requested band.')
        warm_smooth,_=core.native_smooth(nx,ny,nx[keep])
        xx=nx[keep];yy=ny[keep]
        native_grid=np.r_[low,xx[(xx>low)&(xx<high)],high]
        warm=np.interp(native_grid,xx,yy)
        warm_filtered=np.interp(native_grid,xx,warm_smooth)
        regularizer=.01*np.max(np.maximum(warm_filtered,0))
        ratio=(np.maximum(warm,0)+regularizer)/(np.maximum(warm_filtered,0)+regularizer)
        normalizers=[];output=[]
        for b in broad:
            envelope=np.interp(native_grid,grid,b)
            corrected=envelope*ratio if preserve_native_detail else envelope
            base_area=np.trapezoid(envelope,native_grid)
            candidate_area=np.trapezoid(corrected,native_grid)
            if base_area==0:normalizer=1.
            elif candidate_area<=0:raise ValueError('Nonpositive corrected band area.')
            else:normalizer=float(base_area/candidate_area)
            output.append(corrected*normalizer);normalizers.append(normalizer)
        flags=[]
        if targets.max()>1573.15:flags.append('Target exceeds the historical 1573.15 K held-out benchmark maximum.')
        if preserve_native_detail:flags.append('Experimental cold-detail transport; new hot lines and temperature-dependent fine structure are not inferred.')
        output_grid=native_grid if preserve_native_detail else grid
        output_spectra=np.asarray(output) if preserve_native_detail else broad
        return dict(wavenumber_cm1=output_grid,cross_section_cm2_molecule=output_spectra,
            broad_wavenumber_cm1=grid,broad_cross_section_cm2_molecule=broad,
            prepared_cold_cross_section_cm2_molecule=prepared,
            warm_native_cross_section_cm2_molecule=warm,
            warm_filtered_cross_section_cm2_molecule=warm_filtered,
            cold_detail_ratio=ratio,
            metadata=dict(input_temperatures_K=ts.tolist(),target_temperatures_K=targets.tolist(),
                window_cm1=bounds.tolist(),input_count=3,model_weights_sha256=self.forest.sha256,
                input_units=units,output_units='cm2/molecule (natural-log convention)',
                convolution_FWHM_cm1=core.FWHM,native_detail=bool(preserve_native_detail),
                native_detail_regularizer_fraction=.01,area_normalizers=normalizers,
                species_name_required=False,hot_reference_used=False,
                preprocessing=preprocessing,warnings=flags,
                pressure_handling='Frozen broad model is not pressure-conditioned.'))
