# thIRmo-ET

Repository: https://github.com/aliabdelshafikhazraji-bit/thIRmo-ET

**thIRmo-ET predicts a molecular infrared absorption band at requested higher temperatures from one to three measured low-temperature spectra.** Its extremely randomized regression trees use spectral shape and temperature, without a molecule name or molecular structure as input. The supplied model was fitted on 31 molecular identities. An unknown molecule can therefore be submitted without retraining; accuracy on a new molecule remains conditional on the band shape, input quality, temperature, and reference-data coverage.

Training includes two- and three-input examples, and the reported held-out benchmark uses three low-temperature input spectra per molecule. Two-input operation is supported, but the reported errors do not independently validate unseen molecules supplied with only two measurements. **One room-temperature spectrum is supported as an experimental adapter, not as an equally validated replacement.** The model was not refitted on one-input examples. For one input, the observed shape occupies all three shape slots, the temperature slope is zero, and the input-count feature remains 1. The program never invents an additional measured temperature.

## Installation

Use Python 3.10 or newer in this repository directory:

```sh
python -m pip install -r requirements.txt
python thirmo_et.py --help
```

NumPy and SciPy are sufficient for prediction. Optional plots require `python -m pip install matplotlib`. A scikit-learn installation is **not** required. Model weights are compressed numeric NumPy arrays loaded with `allow_pickle=False`; their SHA-256 checksum is checked against the accompanying model metadata.

## Inputs and outputs

Each input file contains two numeric columns: wavenumber in cm⁻¹, then absorption. Comma-separated or whitespace-separated text is accepted, with an optional single header row. Values must be finite. Duplicate wavenumbers are averaged, decreasing grids are sorted, and gaps exceeding 5 cm⁻¹ are rejected within the requested band. Temperature is always in kelvin. Supply a separate `--spectrum FILE TEMPERATURE_K` option for every measured input, and explicitly select the units:

| `--units` | Second-column quantity | Required conversion information |
|---|---|---|
| `cross-section` | Natural-log absorption cross-section, cm²/molecule | None |
| `absorbance-ln` | Dimensionless optical depth, ln(I₀/I) | Absorber column density, molecules/cm² |
| `absorbance-decadic` | Dimensionless decadic absorbance, log₁₀(I₀/I) | Absorber column density, molecules/cm² |
| `pnnl` | Decadic absorbance normalized to 1 ppm·m using a **fixed 296 K** reference at 1 atm | None; this convention must match the input file |

Absorber column density is the number of absorbing molecules per beam area, **not** total-gas column density. The conversions are optical depth = cross-section × absorber column density and decadic absorbance = optical depth / ln(10). The `pnnl` conversion uses the ideal-gas column at 296 K, 101325 Pa, 1 ppm mole fraction and 1 m optical path. It does not substitute the measured spectrum temperature into that normalization. Other PNNL export conventions must first be converted to the specified cross-section convention.

Inputs should describe the same absorption band, with consistent baseline, concentration normalization and instrumental conditions. Use `--window LOW HIGH` for a single band; otherwise the common interval of the input files is used. Do not interpret a prediction made from an entire multi-band infrared spectrum as validated merely because the software accepts the file. All selected wavenumbers must be covered by every input. Include a few cm⁻¹ of measured margin around the band when possible to reduce convolution boundary effects.

The software applies the benchmark preprocessing: a Gaussian with 2 cm⁻¹ full width at half maximum on the native measurement grid, then interpolation to 512 points across the band. This is an additional smoothing operation, not deconvolution of the original instrument response. Broad-band shape is the intended prediction target; the output does not preserve individual unresolved rotational transitions. A coarse original measurement cannot acquire missing spectral information from interpolation.

The output CSV contains `wavenumber_cm-1`, `temperature_K`, and `cross_section_cm2_per_molecule`; each target temperature occupies 512 rows. Supplying `--output-column-density` adds both natural-log and decadic absorbance columns at that explicitly selected column density. An adjacent JSON file records the input temperatures, model checksum, preprocessing, and warnings. Existing output files require `--overwrite` before replacement.

## Examples

Three input spectra, two output temperatures:

```sh
python thirmo_et.py --spectrum cold.csv 278.15 --spectrum room.csv 298.15 --spectrum warm.csv 323.15 --units cross-section --window 1600 2050 --target-temperature 673.15 973.15 --output predictions/hot.csv
```

One observed room-temperature spectrum (experimental one-input mode):

```sh
python thirmo_et.py --spectrum room.csv 298.15 --units cross-section --window 1150 1350 --target-temperature 773.15 --output predictions/one_input.csv
```

Absorbance input and absorbance output, with explicitly supplied absorber columns:

```sh
python thirmo_et.py --spectrum measured.csv 298.15 --units absorbance-decadic --input-column-density 2.0e17 --window 1150 1350 --target-temperature 773.15 --output-column-density 1.5e17 --output predictions/absorbance.csv --plot
```

For several absorbance inputs, `--input-column-density` accepts one shared value or one value per input, in the same order as the `--spectrum` options. The supplied columns should reflect the measurement at each input temperature. To predict absorbance at fixed pressure and path length, calculate the desired hot-gas absorber column separately; its number density changes with temperature. The software does not silently assume constant pressure, path length or mole fraction.

Python use:

```python
from thirmo_et import read_spectrum, predict_spectra

x, cross_sections, provenance = predict_spectra(
    [read_spectrum("cold.csv"), read_spectrum("warm.csv")],
    temperatures=[278.15, 323.15],
    targets=[673.15, 973.15],
    units="cross-section",
    window=(1600, 2050),
)
```

## Algorithm and model selection

The model comprises 128 extremely randomized regression trees, with maximum depth 16, minimum leaf size 16, and a random 80% feature subset at each split. Each output-grid point is represented by 27 local spectral samples across three low-temperature shape slots, a low-temperature difference slope, its relative and absolute wavenumber, band width and center, the warmest input's integrated positive absorption, low-temperature area ratios, spectral centroid and width, number of measured inputs, coldest and warmest input temperatures, requested temperature, and requested temperature increase. This gives 42 numeric features. With two observed temperatures, the intermediate shape is the arithmetic average; it is a feature representation, not a claimed measurement.

The training target is the difference between the high-temperature spectrum and the warmest input after scaling by the warmest input's positive integrated area divided by band width. Averaging the tree predictions supplies this residual. The tool adds it to the normalized warmest spectrum, restores the original cross-section scale, and clips negative predicted cross-sections to zero. Individual line positions, vibrational states, and partition functions are not computed by this regression model.

The released fit combines measured high-temperature references, HITEMP calculations and additional HITRAN calculations. The 0.5 multiplier for HITRAN training sources was chosen by cross-validation that held out training molecular identities; the held-out test identities were not used to choose that multiplier. The 20% PNNL–line-database mismatch rule used in dataset construction is a **source-admission filter**, not a prediction-accuracy guarantee. No such accuracy guarantee is applied to arbitrary uploaded spectra.

The published evaluation must be interpreted at the molecular-identity level: multiple temperatures and spectral points from one molecule are not independent test molecules. There is no molecular-identity overlap between the fitted model and the designated test set. All four ether molecules in the assembled collection were tested as an entire family absent from training. The manuscript and accompanying evaluation report give molecule-dependent errors against measured or simulated references; a simulation comparison does not establish agreement with an independent experiment. This package does not estimate calibrated prediction intervals.

The retained evaluation comprises 70 reference spectra in 10 band windows from nine held-out molecular identities ([calculated benchmark summary](benchmark_summary.json)). This reporting subset was selected after inspection of the spectral comparisons from the original 80-record benchmark; it is not a predeclared blind choice of test bands. The fitted models and training-only cross-validation weights are unchanged, and the original numerical records are preserved in the supplementary provenance archive. For each reference spectrum, error is the sum of absolute prediction–reference differences divided by the sum of positive reference cross-sections on the measured, uniformly spaced grid. Errors are averaged across temperatures within each band, then across bands within each molecule, and finally equally across molecules. This avoids weighting a molecule more heavily merely because it has more temperatures or spectral points.

| Input mode | Measured references | HITEMP references | All held-out molecules |
|---|---:|---:|---:|
| Three low-temperature input spectra | 23.71% | 29.02% | 24.89% |
| One input nearest 298.15 K, experimental adapter | 33.22% | 43.51% | 35.51% |

These calculated percentages are normalized spectral errors on the retained benchmark, not guaranteed errors for uploaded molecules. The one-input assessment uses the same model without refitting and is worse on average; providing the additional measured low-temperature spectra is preferable. Each reference source and temperature domain should be considered separately when deciding whether a predicted spectrum is adequate for a particular measurement application.

## Scope and limitations

- The principal input temperatures are approximately 278–323 K. Predictions above the warmest input are required. Held-out temperature coverage varies by molecule; a temperature evaluated for one molecule is not validated for every other molecule.
- Broad bands are the intended use case. Sharp rotational structure, narrow features and band boundaries can be less accurate. HITRAN/HITEMP line-by-line calculations remain valuable when accurate line structure and suitable spectroscopic data are available.
- The inputs determine the spectral window and absorption scale. Baseline offsets, incompatible instrument response, clipped bands, poor signal-to-noise ratio or incorrect units can propagate into the output.
- Pressure, bath gas and chemical composition are not model inputs. An air-broadened reference and an argon shock-tube measurement can disagree because of collisional broadening. Smoothing reduces some sensitivity but does not establish bath-gas invariance. Thermal decomposition and reactions are not predicted.
- Requested temperatures beyond the benchmark coverage produce extrapolations without an established accuracy bound. Predicted spectra should be checked against independent measurements whenever available.
- A single input provides no measured temperature slope. The one-input adapter changes an input-count feature beyond the model's fitted two/three-input modes and is therefore marked experimental.

## Reproducibility and provenance

`models/thirmo_et_v1.json` records the training configuration, source-model checksum, and exported-weights checksum. `verification.json` records the numeric export checks performed against the original fitted scikit-learn model. The source dataset is not redistributed here. Obtain spectra under the terms of their original providers: [PNNL infrared spectral database](https://nwir.pnl.gov/), [HITRAN](https://hitran.org/) and [HITEMP](https://hitran.org/hitemp/), together with the measured references identified in the accompanying paper.

Run package checks with:

```sh
python -m unittest discover -s tests -v
```

`tools/export_model.py` is an optional developer utility for exporting an already fitted **trusted** scikit-learn pickle. It must never be used with an untrusted file. This utility is not invoked during ordinary prediction, and scikit-learn is needed only to run the export utility. The released weights are unchanged from the selected fitted model; there is no automatic retraining or network download.

When citing the software before a manuscript DOI is assigned, cite **thIRmo-ET, version 1.0.0**, the repository URL and the release or commit identifier. The manuscript citation can be added when a permanent bibliographic record exists.
