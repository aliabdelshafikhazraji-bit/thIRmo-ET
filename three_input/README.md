# thIRmo-ET

**Predict a hot infrared spectrum from exactly three measured near-ambient spectra of a molecule absent from training.** Supply the spectra, their three distinct measurement temperatures (275–350 K), a spectral interval, and the output temperature and pressure. No molecular identity, formula, mass, other molecular properties, line list, or hot spectrum is supplied to prediction.

The current entry point is `thirmo_three.py`, version **5.0.0**, which implements the model of the revised manuscript. The three inputs decide the treatment:

- **Broad-band molecules** (the cold inputs retain no qualifying isolated line): the band shift and Gaussian broadening fitted between the three inputs are extrapolated to the requested temperature with conserved band area, and 128 extremely randomized trees with 16 predictors that contain no absolute wavenumber add a correction learned from other molecules ([`broad_band/`](broad_band/)). Above 1000 K the correction is not updated; the 1000 K spectrum is carried to the requested temperature by the additional physical shift and broadening.
- **Line-resolved molecules** (qualifying isolated lines are retained, for example NO and CH₄): the published envelope and isolated-line ensemble with frozen weights ([`spectral_model.py`](spectral_model.py), unchanged from version 4.0.1).

Requesting a probe value changes only the output coordinates; it never selects another model.

## Run

```sh
python -m pip install -r requirements.txt
python thirmo_three.py --spectrum cold.csv 278.15 --spectrum room.csv 298.15 --spectrum warm.csv 323.15 --units pnnl --context-band 610 740 --target-temperature 700 1000 --target-pressure 1 --output hot_spectrum.csv
```

Add `--probe 673.94` to evaluate the same function at one sensing wavenumber. Inputs are two-column CSV or whitespace-delimited arrays: wavenumber in cm⁻¹, then cross section. Use `--units cm2-molecule` for natural-log cm²/molecule, or `--units pnnl` for PNNL's decadic 1 ppm-m ordinate referenced to 296 K. The CSV output contains temperature, pressure, wavenumber and predicted cross section; for broad-band molecules it also contains the calibrated 90% prediction interval (`interval90_low`, `interval90_high`). The JSON file records the branch, model hashes, the near-ambient shift and broadening, the selected cold peaks, assumptions and warnings.

Python, from this directory:

```python
import numpy as np
from thirmo_three import predict

curves = [np.loadtxt(p) for p in ["cold.txt", "room.txt", "warm.txt"]]
result = predict([(a[:, 0], a[:, 1]) for a in curves], [278.15, 298.15, 323.15],
                 target_temperatures_K=[800, 1000], target_pressures_atm=[1],
                 context_window_cm1=[610, 740], units="pnnl")
result["metadata"]["branch"], result["wavenumber_cm1"], result["cross_section_cm2_molecule"], result["interval90"]
```

## How features are treated

The three cold spectra determine which individual peaks receive a line correction. Each peak must satisfy fixed width (≤0.40 cm⁻¹), separation (≥1.30 cm⁻¹), prominence/height (≥0.50), sampling (≤0.08 cm⁻¹) and cold-fit checks in all inputs. A molecule whose inputs retain at least one such line is line-resolved; otherwise it is broad-band. No molecular-size label is required.

**Broad-band branch.** Between each pair of consecutive inputs, the band shift and an area-conserving Gaussian broadening are fitted. When the shift rate between the two warmer inputs is faster than between the two colder ones, the shift is extrapolated with the temperature dependence of a vibrational (Bose–Einstein) population at an effective wavenumber fixed by the ratio of the two rates and limited to 300 cm⁻¹; otherwise the second rate is continued. The broadening variance grows linearly with temperature (scale 2.5). The tree ensemble learns the remaining change from this physical reference. The output is the 2 cm⁻¹ Gaussian response on a 512-point grid, does not depend on pressure, and comes with a calibrated 90% interval σ̂ ± q (s + 0.02 σ̂ₘₐₓ), where s is the spread of the 128 tree predictions and q = 0.98 was set with training molecules withheld in turn.

**Line-resolved branch.** The tree envelope `B` is fitted to a 2 cm⁻¹ Gaussian response. For a requested Gaussian full width at half maximum (FWHM) `r ≤ 2 cm⁻¹`, the spectral function is `max(B + sum(G_r*L − G_2*L), 0)`, where `L` is a predicted line observed in the cold inputs and `G_r` is a unit-area Gaussian of FWHM `r`. The default `r = 0` preserves the fine-spectrum and probe calculation. At `r = 2`, the line contrasts cancel and the learned envelope is returned without a second convolution. For `r > 2`, only the additional Gaussian width `sqrt(r² − 2²)` is applied to the envelope. Finer output restores eligible cold-observed line contrast while the unresolved background retains the envelope response; it is not a complete deconvolution of the spectrum. See [output-resolution details](RESOLUTION_HANDLING.md).

Line areas and effective widths are supervised by fits to hot training spectra, not tabulated transition parameters. A common Gaussian-width coefficient is learned from those spectral shapes; molecular masses are not read. New transitions absent from all three inputs enter only through the learned envelope.

Target pressures must be 0.6–2.0 atm. Only the Lorentz widths of retained lines scale with the target pressure; bath-gas-specific widths and pressure shifts are not inferred. The cold line fit assumes the PNNL-like boxcar/sinc instrument response (first zero 0.056 cm⁻¹). The line training covers 400–1800 K and 0.7–1.5 atm.

## Evaluation (revised manuscript)

The envelopes are fitted with **33 training identities**, all with three actual cold measurements. **Eight identities are excluded from all fitted components**: 1,2-dimethoxyethane, di-n-propyl ether, diisopropyl ether, n-propyl acetate, toluene, acetone, NO and CH₄. Band errors compare the 2 cm⁻¹ response with reference spectra at that response, using positive integrated reference absorption as denominator, averaged within windows, then molecules, with molecules equally weighted.

| Reference group | Molecules / band–temperature records | thIRmo-ET 5.0.0 | Physical stage alone | Unchanged warmest input |
|---|---:|---:|---:|---:|
| Published measurements (340–974 K) | 6 / 45 | 14.24% (95% bootstrap interval 12.4–16.1%) | 13.90% | 45.30% |
| HITEMP (373–1573 K) | 2 / 39 | 26.47% | – | 65.99% |
| All | 8 / 84 | 17.30% | – | 50.47% |

Withholding each of the 21 molecules with experimental high-temperature spectra in turn gives 18.7%, and withholding each of seven chemical families gives 19.1%; these are the more representative expectations for an unseen broad-band molecule. The validated range is that of the experimental test spectra, up to 974 K for broad bands; higher temperatures are extrapolations. The predictions are an initial estimate and do not replace a measured calibration.

Replacing the calibration in three NO concentration histories gives 16.23%, 17.09% and 17.99% integrated concentration differences. The reported NO measurement uncertainty is ±5.9%; it is not a model uncertainty interval. The source is [Zou et al., Combustion and Flame 276 (2025) 114135](https://doi.org/10.1016/j.combustflame.2025.114135). Concentration is recalculated as published concentration times reference/predicted cross section; no kinetic history is predicted or fitted.

The results of version 4.0.1-experimental (22.54%, 26.47% and 23.52% for the three groups) are retained in `benchmark_summary.json`, `../study/` and `../reproduction/three_input/` for reproduction.

## Line-resolved branch API

The published predictor of the line-resolved branch remains available unchanged:

```python
import numpy as np
from thirmo_three import ThreeSpectrumPredictor

curves = [np.loadtxt(p, delimiter=",") for p in ["cold.csv", "room.csv", "warm.csv"]]
model = ThreeSpectrumPredictor()
context = model.prepare([(a[:, 0], a[:, 1]) for a in curves],
                        [278.15, 298.15, 323.15], [1600, 2050], units="pnnl")
grid = np.linspace(1600, 2050, 225001)
spectrum = model.evaluate(context, [1000], [1], grid)
probe = model.evaluate(context, [1000], [1], [1812.64])
```

Use the same context interval and gas conditions for a band/probe comparison. `evaluate` accepts arbitrary requested wavenumbers; identical coordinates give identical values regardless of the other grid points. A missing third spectrum is rejected rather than fabricated.

## Tests

```sh
python -m unittest discover -s tests -v
python broad_band/verify_release.py --dataset /path/to/prepared/dataset
```

The second command recomputes the six molecule-mean errors with experimental references from the prepared data set (expected mean 14.24%). See [the analyses of the revised manuscript](../revision_2026/README.md) for the remaining tables and figures.
