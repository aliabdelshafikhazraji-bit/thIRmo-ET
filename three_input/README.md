# thIRmo-ET: three-spectrum prediction

This experimental package predicts a hot infrared band or the absorption cross-section at one sensing wavenumber from **exactly three measured spectra at distinct temperatures between 275 and 350 K**. Molecular identity, formula, molecular mass, line lists, and measured hot spectra are not required at inference.

Install Python 3.10 or newer and run:

```text
python -m pip install -r requirements.txt
```

Each input is a two-column CSV or whitespace-delimited text file containing wavenumber in cm⁻¹ and the spectral ordinate. One header row is allowed. Use `--units pnnl` for standard PNNL decadic 1 ppm-m ordinates, or `--units cm2-molecule` for cross-sections in cm²/molecule under the natural-log Beer–Lambert convention.

## Predict a band

```text
python thirmo_three.py --spectrum cold_278.csv 278.15 --spectrum cold_298.csv 298.15 --spectrum cold_323.csv 323.15 --units pnnl --target-temperature 700 1000 --target-pressure 1 --context-band 1100 1350 --output hot_band.csv
```

The output contains the original broad-band model's 512-point spectrum at each requested temperature. The model smooths input spectra with a 2 cm⁻¹ Gaussian before forming spectral features. This output describes the band envelope.

## Predict at one sensing wavenumber

Add `--probe` to request a cross-section at a particular wavenumber:

```text
python thirmo_three.py --spectrum cold_278.csv 278.15 --spectrum cold_298.csv 298.15 --spectrum cold_323.csv 323.15 --units pnnl --target-temperature 700 1000 --target-pressure 1 --context-band 1600 2050 --probe 1812.64 --output hot_probe.csv
```

The code examines the nearest cold peak within ±0.65 cm⁻¹ of the probe in each of the three input spectra. The fixed criteria require a half-prominence width no greater than 0.40 cm⁻¹, at least 1.30 cm⁻¹ separation from the next detected peak, prominence at least 50% of peak height, and native sampling gaps no greater than 0.08 cm⁻¹. A qualifying line must also pass the cold Voigt-profile fit, including a relative fitting RMS below 3%. These are local criteria; a mixed band does not need 90% of its total peak prominence to come from isolated lines.

When the checks pass, the narrow branch returns the **contribution of retained lines observed in the cold inputs** at the probe. Otherwise, the result is interpolated from the broad-band envelope. The JSON file records which calculation was used. The complete band output remains the original broad envelope; the code does not blend an incomplete set of narrow lines into a purported complete hot spectrum.

## Conditions and interpretation

All input spectra must cover the entire selected context band. The three input temperatures are required, and every target temperature must exceed the warmest input. Supply either one target pressure or one pressure for each target temperature. The supported study pressure range is 0.6–2.0 atm.

The narrow branch assumes the input spectra were measured near 1 atm with a PNNL-like instrument response. It uses a fixed Doppler mass proxy of 27.9926073333 amu derived from the equal-species mean of CO, HCl, and HF training masses; this is not an inferred target molecular mass. Its thermal population approximation is appropriate for linear-rotor-like behavior and is not established for every polyatomic molecule with resolved lines. The method cannot create transitions absent from all three cold inputs. Cold nitrogen-bath line widths are transferred to the target pressure; the target bath gas is not inferred. The narrow residual model was trained up to 1800 K and 1.5 atm.

The broad model is not pressure-conditioned. Its recorded target pressure describes the intended experiment, and its output does not include an inferred pressure or bath-gas correction. Neither branch converts cross-sections to concentrations or absorbance automatically.

The accompanying developmental NO benchmark gave integrated concentration differences of approximately 13.5–15.0% when using the narrow branch in place of the published calibration. These figures are benchmark outcomes, not prediction uncertainty intervals. This model extension was evaluated after earlier versions had already been inspected on related cases; it is not presented as a newly blinded external validation.

## Python API

```python
from thirmo_three import ThreeSpectrumPredictor

model = ThreeSpectrumPredictor()
result = model.predict(
    spectra=[(nu1, sigma1), (nu2, sigma2), (nu3, sigma3)],
    input_temperatures_K=[278.15, 298.15, 323.15],
    target_temperatures_K=[700, 1000],
    target_pressures_atm=1.0,
    context_window_cm1=[1100, 1350],
    units="cm2/molecule",
)
```

`result["cross_section_cm2_molecule"]` has one row per target temperature. `result["metadata"]` contains the branch, model hashes, assumptions, input temperatures, target temperatures and pressures, and any extrapolation notes. Adding `probe_wavenumber_cm1=...` requests one probe value per temperature.

The historical broad model contains training examples with one, two, or three measured cold inputs. Requiring three inputs in this interface does not imply that every historical training example used three spectra.

## Check the installation

```text
python -m unittest discover -s tests -v
```

The portable tests use synthetic spectra and packaged numeric weights. Private parity checks against the original PNNL files are kept outside this distribution. No scientific data download, network connection, pickle file, or scikit-learn installation is needed for prediction.
