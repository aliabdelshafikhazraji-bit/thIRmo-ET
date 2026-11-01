# thIRmo-ET

**thIRmo-ET predicts a hot infrared band or a cross-section at one sensing wavenumber from exactly three measured near-ambient spectra.** Supply the three spectra, their distinct temperatures between 275 and 350 K, a context wavenumber interval, the target temperature and pressure, and optionally a sensing wavenumber. No target molecular identity, formula, molecular mass, line list, or measured hot spectrum is required.

The default entry point is [`three_input/thirmo_three.py`](three_input/thirmo_three.py), version **3.0.0-experimental**. It combines the unchanged broad-band model with a mass-free-input narrow-feature branch. The historical [`thirmo_et.py`](thirmo_et.py), model weights, benchmark records, and reproduction files are retained for earlier results; [historical usage](reproduction/HISTORICAL_2_5_1_USAGE.md) documents that older one-to-three-input interface.

## Install and predict

Use Python 3.10 or newer:

```sh
python -m pip install -r three_input/requirements.txt
python three_input/thirmo_three.py --help
```

Predict a complete band envelope:

```sh
python three_input/thirmo_three.py --spectrum cold.csv 278.15 --spectrum room.csv 298.15 --spectrum warm.csv 323.15 --units pnnl --context-band 1100 1350 --target-temperature 700 1000 --target-pressure 1 --output predictions/hot_band.csv
```

Predict at a single sensing wavenumber:

```sh
python three_input/thirmo_three.py --spectrum cold.csv 278.15 --spectrum room.csv 298.15 --spectrum warm.csv 323.15 --units pnnl --context-band 1600 2050 --probe 1812.64 --target-temperature 700 1000 --target-pressure 1 --output predictions/hot_probe.csv
```

Each input is a two-column CSV or whitespace-delimited text file containing wavenumber in cm⁻¹ and the spectral ordinate. Choose `pnnl` for the standard PNNL decadic 1 ppm-m ordinate at its fixed 296 K reference density, or `cm2-molecule` for absorption cross-section in cm²/molecule under the natural-log Beer–Lambert convention. Each spectrum must cover the context band. Input spectra must be actual measurements at three distinct near-ambient temperatures; the interface does not synthesize a missing third measurement.

Outputs are a CSV of cross-sections and a JSON file identifying the selected calculation, model hashes, temperatures, pressures, and assumptions. Supply either one target pressure or one pressure per target temperature. The study pressure range is 0.6–2.0 atm. The [three-input guide](three_input/README.md) documents the Python API and all options.

## What is predicted

For a complete band, the original 128-tree extremely randomized regression model returns the same 512-point envelope as before. Input spectra receive an additional 2 cm⁻¹ full-width-at-half-maximum Gaussian smoothing. The broad model uses spectral-shape and temperature features and is not pressure-conditioned.

At a specified probe wavenumber, the code checks the nearest peak within ±0.65 cm⁻¹ in all three cold spectra. Fixed cold-only width, separation, contrast, and sampling criteria are followed by a cold line-fit quality check. When these pass, the narrow branch sums retained lines observed in the cold inputs and evaluates their contribution at the probe. Otherwise, it interpolates the original broad envelope. This is a local decision: a mixed band does not have to satisfy a whole-window 90% resolved-peak criterion.

The narrow branch uses a training-derived constant Doppler mass proxy of 27.9926073333 amu, rather than an input or inferred target molecular mass. Its ten-feature residual model was fitted using CO, HCl, and HF only. Cold line areas, widths, positions, and their near-ambient temperature trends are obtained from the three input spectra. The thermal population approximation remains linear-rotor-like; cold morphology alone does not establish that this approximation applies to every polyatomic molecule. The calculated narrow contribution does not generate transitions absent from all three cold spectra. The complete-band output remains the broad envelope.

For the line branch, input pressure is assumed to be 1 atm and the cold instrument response is PNNL-like. Pressure scaling transfers fitted cold widths to target conditions, without inferring bath-gas-specific broadening. Training targets extend to 1800 K and 1.5 atm; metadata flags predictions beyond these limits. A pressure inside the wider 0.6–2.0 atm study range does not by itself establish that the result lies inside the line model's training range.

## Evaluation and provenance

The **unchanged full-band benchmark** uses 46 training molecular identities and eight withheld identities: three non-acetal acyclic ethers, n-propyl acetate, toluene, acetone, NO, and CH₄. Dimethoxymethane and cyclic acetals remain in training. The six molecules evaluated against published hot measurements have a 21.61% equal-molecule mean normalized integrated absolute error over 45 band–temperature comparisons. NO and CH₄ have a 26.17% two-molecule mean over 39 HITEMP comparisons. The eight-molecule mean is 22.75%. These values are reproduced from [the original benchmark records](benchmark_summary.json). All reported test cases use three measured cold inputs; historical broad-model training includes one-, two-, and three-input records.

The **new narrow-probe benchmark** replaces the published NO calibration in three shock-tube concentration histories from [Zou et al., Combustion and Flame 276 (2025) 114135](https://doi.org/10.1016/j.combustflame.2025.114135). The calculation retains published corrected NO concentrations, measured pressure histories, and author-calculated temperature histories, and multiplies concentration by the published-to-predicted cross-section ratio. No normalization is fitted to those histories.

| Initial reflected-shock temperature | Initial pressure | Integrated NO concentration difference |
|---|---:|---:|
| 1523.2 K | 1.434 atm | 13.51% |
| 1575.9 K | 1.538 atm | 14.21% |
| 1653.4 K | 1.533 atm | 15.04% |

The percentage is 100 times the time integral of the absolute recalculated-minus-published concentration difference, divided by the time integral of positive published concentration. All measured pressure samples lie within 0.6–2.0 atm. Some later samples in the second and third histories exceed 1800 K. The reported measurement uncertainty is ±5.9% of NO concentration; it is not a model prediction interval. [Narrow benchmark metadata](three_input/benchmark_summary.json) provides the source, temperature ranges, and temperature-separated results.

The mass-free-input model and protocol were frozen before these three histories were evaluated, with no subsequent calibration to their errors. The development followed earlier inspection of related NO cases, so this is a developmental held-out-species benchmark, not a newly blinded external validation. The results do not replace the original full-band HITEMP benchmark.

The broad branch was also used for six shock-tube cases: three di-n-propyl ether histories and three 1,2-dimethoxyethane histories. Their published cross-section correlations, source figures, pressures, temperatures, and all six results are provided in [the organic benchmark summary](three_input/organic_benchmark_summary.json). The reference and predicted cross-sections are held at the reported post-shock conditions for these cases. The [standalone history evaluator](reproduction/three_input/recalculate_histories.py) reproduces the concentration substitution and integrated error calculation from user-supplied histories. Hot reference data are evaluation inputs to that script, not inputs to spectral prediction.

## Reproduction and tests

The historical [reproduction directory](reproduction/README.md) and [study files](study/) remain unchanged. The [three-input reproduction guide](reproduction/three_input/README.md) documents training-only model selection, the frozen protocol, source-array requirements, and the portable line-model training script. The distributed package contains numeric model weights and code; licensed raw PNNL spectra and line lists remain with their providers: [PNNL](https://nwir.pnl.gov/), [HITRAN](https://hitran.org/), and [HITEMP](https://hitran.org/hitemp/).

```sh
python -m unittest discover -s three_input/tests -v
python -m unittest discover -s tests -v
python -m unittest discover -s reproduction/three_input -p "test_*.py" -v
```

Ordinary inference requires only NumPy and SciPy. It does not access private files, download reference spectra, or require scikit-learn. The packaged broad and narrow predictions were checked against the saved developmental outputs with zero numerical differences.

Until a manuscript DOI is available, cite **thIRmo-ET version 3.0.0-experimental** and this repository URL. Cite version 2.5.1 when reproducing the historical interface or its original full-band results.
