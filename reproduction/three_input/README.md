# Three-input extension: reproduction

The executable common interface and both frozen numeric models are in [`../../three_input/`](../../three_input/). The broad model and historical reproduction pipeline are retained unchanged. This directory describes the new mass-free-input line branch, its source requirements, training-only selection protocol, and the local cold-spectrum routing rule.

## Frozen design

[`frozen_protocol.json`](frozen_protocol.json) was saved before fitting the new candidate. CO, HCl, and HF are the only line-training species. The target molecular mass feature was removed, leaving ten features. A single Doppler mass proxy is the equal-species arithmetic mean of those three training isotopologue masses. Each leave-one-species-out validation fold computes the proxy from its two fitting species only and refits both fitting and held-out cold spectra with that proxy.

The unchanged Extra Trees settings are 128 trees, maximum depth 8, minimum leaf size 4, all features eligible at each split, and random seed 20261101. Species receive equal total training weight. The residual branch is selected only if its equal-species mean local integrated absolute spectral error is below the physical baseline in leave-one-species-out validation. The resulting means were 17.0903% and 18.2166%, respectively. The final fits retained 40 CO, 10 HCl, and 2 HF lines.

[`original_freeze_manifest.json`](original_freeze_manifest.json) records the original model, configuration, code, and input-array hashes and the freeze time. Packaging subsequently changed imports and command-line entry points; the numeric weights are unchanged. The [`PACKAGE_MANIFEST.json`](../../three_input/PACKAGE_MANIFEST.json) records the distributed files. The new NO evaluation followed this freeze once, without tuning to its errors. Earlier related NO results had already been inspected during development, so this is not presented as a newly blind external validation.

## Obtain the external source inputs

No raw PNNL spectrum, HITRAN/HITEMP line list, or licensed native spectral array is redistributed here. [`source_requirements.json`](source_requirements.json) identifies the species, source databases, PNNL filenames, temperatures, selected-line counts, and checksums. Obtain the measurements and permitted line data from [PNNL](https://nwir.pnl.gov/), [HITRAN](https://hitran.org/), and [HITEMP](https://hitran.org/hitemp/).

Exact retraining additionally requires these externally prepared files:

- `selected_lines.json`: selected CO/HCl/HF line identities, PNNL input metadata, input temperatures, and training isotopologue masses.
- `cold_gate.json`: near-ambient PNNL/reference agreement decisions and relative mismatch for each selected line.
- `CO_native_training.npz`, `HCl_native_training.npz`, and `HF_native_training.npz`: the native reference spectra, line strengths and widths, temperatures, pressures, and wavenumber grids.

The NPZ arrays are named `temperatures_K`, `pressures_atm`, `line_strength_cm_molecule`, `lorentz_hwhm_cm1`, `wavenumber_cm1`, and `spectra_native_cm2_molecule`. Their dimensions follow line index, temperature index, and pressure index where applicable; the wavenumber grid and reference spectrum include the local interval around each selected line. The published checksum inventory distinguishes the exact prepared arrays from newly generated approximations. Access to the source websites alone does not recreate these intermediate arrays; use the matching externally supplied preparation outputs for exact reproduction.

Install the inference requirements and scikit-learn, then run from the repository root:

```sh
python -m pip install -r three_input/requirements.txt scikit-learn
python reproduction/three_input/train_spectra_only.py --data-dir /path/to/prepared_native_arrays --pnnl-root /path/to/PNNL_Database --output-dir reproduced_line_model
```

The script resolves PNNL inputs beneath the explicitly supplied root using the species-directory and filename recorded in `selected_lines.json`. Original workstation paths are not accessed. Existing protocols are not silently overwritten. The output includes fold-specific cold fits, leave-one-species-out errors, final numeric weights, configuration, and a new freeze manifest. Random seed and hyperparameters are fixed; dependency versions can affect numeric hashes, so compare the array values as well as the hash when using a different environment.

## Routing and numerical verification

[`cold_router.py`](../../three_input/cold_router.py) contains the fixed width, separation, prominence, and sampling criteria. The public probe interface uses these criteria for the nearest peak within ±0.65 cm⁻¹ in each cold input, then requires an eligible line fit. It does not apply the earlier whole-window 90% prominence gate or a minimum count of three isolated peaks. The model parameters do not change with this local routing decision.

Portable synthetic tests verify strict input count, temperature range, units, full-band equivalence, broad-probe interpolation, local routing of a mixed window, and the command-line interface:

```sh
python -m unittest discover -s three_input/tests -v
```

Private parity checks using the original PNNL files verified that the two broad ether probe predictions and all three NO probe cross-section histories are exactly equal to their saved pre-packaging outputs. Those raw files and private path-bearing checks are not distributed. [`benchmark_summary.json`](../../three_input/benchmark_summary.json) contains the public NO source locator, calculation, uncertainties, conditions, and aggregate errors. The published concentration histories and calibration are used only for evaluation and do not enter either model's prediction interface.

## Recalculate a published shock-tube history

[`recalculate_histories.py`](recalculate_histories.py) is a standalone **evaluation script** for the cross-section substitution and integrated error calculations. It does not call, fit, or normalize a spectral model. Supply a CSV with aligned `time_ms`, `published_mole_fraction`, `sigma_reference`, and `sigma_predicted` columns; mole fraction is in mol/mol and the two cross-sections must use the same declared units. Custom column names can be supplied with the corresponding command-line options.

```sh
python reproduction/three_input/recalculate_histories.py --input supplied_history.csv --output comparison.csv --cross-section-units m2/mol --reported-sigma-fraction 0.06
```

The script calculates `x_recalculated = x_published * sigma_reference / sigma_predicted`. Its reported percentage is `100 * integral(abs(x_recalculated-x_published),dt) / integral(max(x_published,0),dt)`. Time must be finite and strictly increasing; both cross-sections must be finite and positive. Negative concentration samples are preserved as supplied. The output CSV contains both histories and their cross-section ratio, while an adjacent JSON defines the metric and assumptions.

`--reported-sigma-fraction 0.06` optionally propagates a reported calibration contribution to the published reference as `x_published/1.06` to `x_published/0.94`, with bounds reordered for negative noise samples. For a directly reported concentration uncertainty, use `--reported-x-fraction 0.059` instead. These alternatives describe the published-reference uncertainty only. They are not uncertainty intervals for the model and are not added to model error. Retain whether the original study reported an upper bound or a confidence interval; the script does not infer that interpretation.

Hot references are required for this evaluation calculation, while the separate `three_input/thirmo_three.py` prediction interface consumes only the three cold spectra and specified conditions. Raw published histories are not bundled. The [six organic cases](../../three_input/organic_benchmark_summary.json) and [three NO cases](../../three_input/benchmark_summary.json) provide source DOIs, figure or workbook locations, conditions, cross-section provenance, and aggregate results. Reported post-shock temperature and pressure are held constant for the six organic cases; the NO cases use published measured pressure and author-calculated temperature histories.

Test the evaluation equations, including a known constant cross-section ratio, nonuniform time spacing, negative measurement noise, and uncertainty propagation:

```sh
python -m unittest discover -s reproduction/three_input -p "test_*.py" -v
```

The original frozen protocol and line-model configuration retain their exact bytes, protected by `.gitattributes`. Other packaged text files use LF line endings; current distribution checksums are recorded separately from the original freeze checksums.
