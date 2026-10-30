# thIRmo-ET

**thIRmo-ET predicts a high-temperature infrared absorption band for a molecule absent from its training set.** The user supplies one to three measured near-ambient spectra, the temperature of each spectrum, a wavenumber interval, and a hotter target temperature. The program returns the predicted molecular absorption cross section at 512 wavenumbers. Molecular identity and structure are not model inputs.

The default model, version 2.5.0, is an ensemble of 128 extremely randomized regression trees. It was fitted on 46 molecular identities using spectra at 0.6–2.0 atm from published experiments and qualified HITEMP and HITRAN calculations. Eight molecular identities were withheld: three acyclic ethers as a complete family, n-propyl acetate, toluene, acetone, NO, and CH₄. The six molecules with published hot measurements have a 21.61% equal-molecule mean normalized integrated absolute error across 45 band–temperature comparisons. NO and CH₄ have a 30.09% two-molecule mean against 30 HITEMP comparisons; an additional CH₄ window was checked against nine HITRAN spectra. Across the eight test molecules, the equal-molecule mean is 22.90%. [Benchmark details](benchmark_summary.json) report each molecule, source, and paired fit without added HITRAN training spectra.

## Install

Use Python 3.10 or newer:

```sh
python -m pip install -r requirements.txt
python thirmo_et.py --help
```

NumPy and SciPy are sufficient for prediction. Add Matplotlib to use `--plot`. The numeric model weights are in [`models/thirmo_et_final.npz`](models/thirmo_et_final.npz) and are checked against the SHA-256 digest in the accompanying JSON file. Ordinary inference does not require scikit-learn or the original training spectra.

## Input files and units

Each spectrum is a two-column CSV or whitespace-delimited text file: wavenumber in cm⁻¹, then absorption. Supply a `--spectrum FILE TEMPERATURE_K` option for each measured spectrum. The spectra must describe the same molecule and cover the entire chosen band. Select the second-column units with `--units`:

| Option | Second-column quantity | Additional input |
|---|---|---|
| `cross-section` | Absorption cross section, cm²/molecule | None |
| `pnnl` | PNNL decadic absorbance normalized to 1 ppm·m at the fixed 296 K reference density | None |
| `absorbance-ln` | Natural-log absorbance, ln(I₀/I) | Absorber column density, molecules/cm² |
| `absorbance-decadic` | Decadic absorbance, log₁₀(I₀/I) | Absorber column density, molecules/cm² |

The absorber column density is the number of absorbing molecules per beam area. For absorbance inputs, provide it with `--input-column-density`. To obtain absorbance as well as cross section in the output, specify `--output-column-density` for the desired gas condition. The program stores those column densities in its output metadata.

## Run a prediction

One room-temperature spectrum:

```sh
python thirmo_et.py --spectrum room.csv 298.15 --units cross-section --window 1150 1350 --target-temperature 773.15 --output predictions/hot.csv --plot
```

Three PNNL spectra and several target temperatures:

```sh
python thirmo_et.py --spectrum cold.txt 278.15 --spectrum room.txt 298.15 --spectrum warm.txt 323.15 --units pnnl --window 1600 2050 --target-temperature 673.15 973.15 --output predictions/hot.csv --plot
```

Absorbance input and output:

```sh
python thirmo_et.py --spectrum measured.csv 298.15 --units absorbance-decadic --input-column-density 2.0e17 --window 1150 1350 --target-temperature 773.15 --output-column-density 1.5e17 --output predictions/absorbance.csv
```

The output CSV contains `wavenumber_cm-1`, `temperature_K`, and `cross_section_cm2_per_molecule`; it adds absorbance columns when an output column density is supplied. The adjacent JSON file records input temperatures, band limits, model digest, preprocessing, and applicability flags. `--plot` adds a PNG preview. The selected band must have continuous input coverage; gaps greater than 5 cm⁻¹ are rejected.

Python use:

```python
from thirmo_et import read_spectrum, predict_spectra

x, cross_sections, metadata = predict_spectra(
    [read_spectrum("cold.csv"), read_spectrum("room.csv")],
    temperatures=[278.15, 298.15],
    targets=[673.15, 973.15],
    units="cross-section",
    window=(1150, 1350),
)
```

## What the model calculates

The program smooths each input with an additional 2 cm⁻¹ full-width-at-half-maximum Gaussian, samples the requested band at 512 points, and constructs local spectral-shape and temperature features. Each tree predicts a normalized change relative to the warmest supplied spectrum. Averaging the trees and restoring the input band's area produces a cross section in cm²/molecule. One measured input is repeated in the feature slots with zero measured temperature slope; two inputs are represented with their midpoint. The supplied model was fitted on one-, two-, and three-input examples.

The model provides an approximate band envelope. For n-propyl acetate, the measured broad-band maximum moved 8.4 cm⁻¹ toward lower wavenumber between 671 and 862 K; the model predicted 3.6 cm⁻¹ of movement. This peak estimate provides a starting region for a sensor wavelength scan. Narrow rotational structure and bath-gas conditions also affect local comparisons. HITRAN and HITEMP remain valuable for explicit narrow-line calculations when their transition data cover the desired molecule and temperature. The 20% near-ambient PNNL/database agreement rule was used only to decide which line-list bands entered the research dataset; it is not an error threshold for user predictions.

## Provenance and citation

The final model partition, experimental sources, data processing, predictions, and per-record errors are documented in the companion manuscript and Supplementary Archive S1. The archive accompanies the manuscript; this repository distributes the inference code and trained weights. Published spectra remain with their original providers, including [PNNL](https://nwir.pnl.gov/), [HITRAN](https://hitran.org/), and [HITEMP](https://hitran.org/hitemp/).

Until the manuscript has a DOI, cite **thIRmo-ET version 2.5.0** and this repository URL. Run the local inference checks with:

```sh
python -m unittest discover -s tests -v
```
