# thIRmo-ET

**Predict a hot infrared spectrum from exactly three measured near-ambient spectra of a molecule absent from training.** Supply the spectra, their three distinct measurement temperatures (275–350 K), a spectral interval, and the output temperature and pressure. No molecular identity, formula, mass, other molecular properties, line list, or hot spectrum is supplied to prediction.

The current entry point is `three_input/thirmo_three.py`, version **4.0.1-experimental**. Broad envelopes and isolated-line corrections are fitted separately, both using three cold measurements paired with hot reference spectra from training molecules. One function combines their outputs. Requesting a probe value changes only the output coordinates; it never selects another model.

## Run

```sh
python -m pip install -r three_input/requirements.txt
python three_input/thirmo_three.py --spectrum cold.csv 278.15 --spectrum room.csv 298.15 --spectrum warm.csv 323.15 --units pnnl --context-band 1600 2050 --target-temperature 700 1000 --target-pressure 1 --output hot_spectrum.csv
```

Add `--probe 1812.64` to evaluate the same function at one sensing wavenumber. Inputs are two-column CSV or whitespace-delimited arrays: wavenumber in cm⁻¹, then cross section. Use `--units cm2-molecule` for natural-log cm²/molecule, or `--units pnnl` for PNNL's decadic 1 ppm-m ordinate referenced to 296 K. The CSV output contains temperature, pressure, wavenumber and predicted cross section; JSON records model hashes, selected cold peaks and assumptions.

## How features are treated

The three cold spectra determine which individual peaks receive a line correction. Each peak must satisfy fixed width (≤0.40 cm⁻¹), separation (≥1.30 cm⁻¹), prominence/height (≥0.50), sampling (≤0.08 cm⁻¹) and cold-fit checks in all inputs. A band can contain both treatments. No molecular-size label is required.

The tree envelope `B` is fitted to a 2 cm⁻¹ Gaussian response. For a requested Gaussian full width at half maximum (FWHM) `r ≤ 2 cm⁻¹`, the spectral function is `max(B + sum(G_r*L − G_2*L), 0)`, where `L` is a predicted line observed in the cold inputs and `G_r` is a unit-area Gaussian of FWHM `r`. The default `r = 0` preserves the fine-spectrum and probe calculation. At `r = 2`, the line contrasts cancel and the learned envelope is returned without a second convolution. For `r > 2`, only the additional Gaussian width `sqrt(r² − 2²)` is applied to the envelope.

Use `--output-fwhm-cm1 2` for the response used in the reported band comparisons. The Python argument is `output_fwhm_cm1=2.0`. Finer output restores eligible cold-observed line contrast while the unresolved background retains the envelope response; it is not a complete deconvolution of the spectrum. The envelope uses 512 points, and default line-resolved sampling is at most 0.002 cm⁻¹. Sampling is distinct from instrumental resolution. See [output-resolution details](three_input/RESOLUTION_HANDLING.md).

Line areas and effective widths are supervised by fits to hot training spectra, not tabulated transition parameters. A common Gaussian-width coefficient is learned from those spectral shapes; molecular masses are not read. The line-population baseline uses a linear-rotor/one-vibration approximation inferred from the cold band. New transitions absent from all three inputs enter only through the learned envelope. Cold shape alone does not establish that the line-population approximation is valid for every polyatomic band.

Target pressures must be 0.6–2.0 atm. The envelope is not pressure-conditioned. The resolved Lorentz widths scale with target pressure relative to the assumed 1 atm cold input; bath-gas-specific widths and pressure shifts are not inferred. The cold line fit assumes the PNNL-like boxcar/sinc instrument response (first zero 0.056 cm⁻¹). The line training covers 400–1800 K and 0.7–1.5 atm.

## Evaluation

The envelope fit uses **33 training identities**, all with three actual cold measurements. **Eight identities are excluded from both fitted components**: 1,2-dimethoxyethane, di-n-propyl ether, diisopropyl ether, n-propyl acetate, toluene, acetone, NO and CH₄. The three non-acetal acyclic ethers are withheld together; dimethoxymethane remains an acetal training molecule.

Band errors compare the learned 2 cm⁻¹ response directly against reference spectra at that response, using positive integrated reference absorption as denominator. Errors are averaged within windows, then molecules, with molecules equally weighted. Single-wavelength retrieval uses the same fitted components at the default fine-output setting. The response setting changes no fitted parameter, training identity or cold-input selection rule.

| Reference group | Molecules / band–temperature records | Mean band error |
|---|---:|---:|
| Published measurements | 6 / 45 | 22.54% |
| HITEMP | 2 / 39 | 26.47% |
| All | 8 / 84 | 23.52% |

The line head uses CO, HCl and HF only: 50 cold-qualified lines and 1021 usable spectral-reference fits. Leave-one-training-species-out mean local spectral error is 15.63%. The shared Gaussian coefficient is estimated from the fitting species only in each fold. The fixed rules retain 32 NO fundamental lines and five CH₄ lines in 4000–4700 cm⁻¹; no peak passes in the CH₄ 1280–1340 cm⁻¹ window.

Replacing the calibration in three NO concentration histories gives 16.23%, 17.09% and 17.99% integrated concentration differences. The reported NO measurement uncertainty is ±5.9%; it is not a model uncertainty interval. The source is [Zou et al., Combustion and Flame 276 (2025) 114135](https://doi.org/10.1016/j.combustflame.2025.114135). Concentration is recalculated as published concentration times reference/predicted cross section; no kinetic history is predicted or fitted.

These are developmental transfer tests. Earlier versions' results for these molecules had already been inspected; the new calculation is not represented as newly blinded. Test identities and hot measurements never supply fit features, training labels, or calibration factors to the current predictor. A qualification audit correction and the freeze records are retained in `reproduction/three_input/`.

## Reproduce and test

See [the three-input guide](three_input/README.md) and [reproduction instructions](reproduction/three_input/README.md). Numeric weights, exact dataset manifests, source records and 84 error records are included. Licensed PNNL spectra and HITRAN/HITEMP line lists remain with [PNNL](https://nwir.pnl.gov/), [HITRAN](https://hitran.org/) and [HITEMP](https://hitran.org/hitemp/); the repository does not redistribute the raw spectra.

```sh
python -m unittest discover -s three_input/tests -v
python -m unittest discover -s reproduction/three_input -p "test_*.py" -v
```

Inference uses NumPy and SciPy, runs offline, and does not load a target reference spectrum. Refitting additionally requires scikit-learn and acquired source arrays. The root `thirmo_et.py` and its frozen weights are retained for historical reproduction; use the explicit current entry point above. Cite **thIRmo-ET 4.0.1-experimental** and this repository until a manuscript DOI is assigned.
