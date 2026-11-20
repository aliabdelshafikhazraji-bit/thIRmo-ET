# Analyses of the revised manuscript (thIRmo-ET 5.0.0)

This folder contains the scripts, result files, and digitized published data behind the tables and figures of the revised manuscript and its supplementary material. The prediction code itself is in [`three_input/`](../three_input/): the routed entry point `thirmo_three.py`, the broad-band branch in `three_input/broad_band/`, and the unchanged line-resolved branch (`spectral_model.py`, frozen weights).

## Contents

| Folder | Content |
|---|---|
| `analysis/` | Python scripts. Library modules: `config.py` (paths), `common.py` (data set loading and Eq. 4), `physics.py` (physical stage, Eqs. 3a–3c), `broad.py` (broad-band branch and its training), `linebranch.py` (published line-resolved envelope), `baselines.py` (comparison methods), `figstyle.py`. Analyses `a01`–`a17` and the figure scripts are listed below. |
| `results/` | JSON result files written by the analyses, as used for the manuscript, the supplementary material, and the figures. |
| `digitized/` | Data digitized from published figures (benzene history of Shakfa et al. 2021, Fig. 8; ether mole-fraction histories of Adil and Farooq) and the published diisopropyl ether correlation of Adil and Farooq (2023) with its source and stated range. |

## Analyses

| Script | Purpose |
|---|---|
| `a01_dataset.py` | Data set accounting (Table 1) and per-window source audit. |
| `a02_main.py` | Main benchmark: fixed 33/8 molecular partition, 84 test records; broad-band molecules with the physical stage and learned correction, NO and CH₄ with the published line-resolved envelope. |
| `a03_lomo_train.py` | Leave-one-molecule-out over the 15 experimental training molecules (selections and interval calibration; test molecules never used). |
| `a04_validation.py` | Molecule-disjoint validation over the 21 molecules with experimental high-temperature spectra: each molecule withheld, each family withheld, and random splits. |
| `a05_source_consistency.py` | Agreement of spectra of the same molecule from different facilities (reference uncertainty floor). |
| `a06_hitran.py` | Role of the added HITRAN training collection, weight sensitivity, paired uncertainty, and seed variability. |
| `a07_intervals.py` | Per-wavenumber prediction intervals and their coverage. |
| `a08_pointwise.py` | Wavelength-selection measures (predicted versus reference maximum, sensitivity lost, cross-section error there). |
| `a09_wakatsuki.py` | Temperature equation of Wakatsuki et al. (2005) with the three inputs and in its intended input regime. |
| `a10_pressure.py` | Collisional-width and pressure mismatch for a line-resolved band with NO-like constants. |
| `a11_doppler.py` | Common Gaussian-width coefficient versus the mass dependence of Doppler broadening. |
| `a12_shift.py` | Extrapolated versus fitted band shift and broadening; n-propyl acetate band maximum. |
| `a13_bootstrap.py` | Bootstrap intervals of the molecule-mean errors. |
| `a14_seed_probe.py` | Seed variability and intervals of single-wavenumber predictions (benzene, ether probes). |
| `a15_hot_growth.py` | Growth of absorption outside the near-ambient band (NO, CH₄) with temperature. |
| `a16_benzene_fig8.py` | Benzene history of Shakfa et al. (961 K, 1.36 bar) recalculated with thIRmo-ET, and the sensitivity at the predicted maximum. |
| `a17_dipe_probe.py` | Diisopropyl ether at 1121.82 cm⁻¹: published shock-tube correlation and scanned test spectra versus the prediction. |
| `a18_no_native.py` | Retained NO lines at native resolution against HITEMP2019 calculations at 0.7–1.9 atm: error within ±1 cm⁻¹ of each line and the cross section at the line centers. Needs the three PNNL NO spectra (`THIRMO_PNNL`) and the HITEMP NO line list (`THIRMO_HITEMP`), which are not redistributed. |
| `benzene_inputs.py` | Prepares the benzene ν4 window (610–740 cm⁻¹) from the three PNNL benzene spectra, which are not redistributed (their folder is given as argument or in `THIRMO_PNNL`); run it before `benzene.py`, `a14_seed_probe.py` and `a16_benzene_fig8.py`. |
| `benzene.py`, `ethers.py` | Benzene laser wavenumber and ether probe wavenumbers: published correlations versus thIRmo-ET, and recalculated histories. |
| `export_model.py` | Exports the broad-band forest in the portable weight format of `three_input/broad_band/models/` and checks it against the fitted forest. |
| `figures_a.py`, `figures_b.py`, `figures_s.py`, `figure_s5.py` | Manuscript and supplementary figures. |

## Running

```sh
python -m pip install -r revision_2026/requirements.txt
export THIRMO_DATASET=/path/to/prepared/dataset     # 69 group_*.npz files and dataset_manifest.json
export THIRMO_PNNL=/path/to/PNNL                    # BENZENE_5T/25T/50T.TXT (benzene analyses) and NO_5T/25T/50T.TXT (a18)
export THIRMO_HITEMP=/path/to/08_HITEMP2019.par.bz2  # HITEMP NO line list (a18 only)
cd revision_2026/analysis
python a01_dataset.py   # then a02 ... a18, and the figure scripts; benzene_inputs.py before the benzene analyses
```

The prepared data set is built with [`reproduction/pipeline.py`](../reproduction/) from the licensed source spectra, which this repository does not redistribute ([PNNL](https://nwir.pnl.gov/), [HITRAN](https://hitran.org/), [HITEMP](https://hitran.org/hitemp/), and the published measurements listed in `study/source_records.json`). Results are written to `revision_2026/out/`; the files in `results/` are those of the manuscript. The broad-band forest is fitted with scikit-learn on first use and cached; the distributed weights in `three_input/broad_band/models/` were exported from it with `export_model.py` (SHA-256 `dfd54cf1…6997`).

To check the distributed broad-band weights against the manuscript without refitting:

```sh
python three_input/broad_band/verify_release.py --dataset /path/to/prepared/dataset
```

Expected molecule-mean errors (Eq. 4): 1,2-dimethoxyethane 17.83%, di-n-propyl ether 15.64%, diisopropyl ether 15.39%, n-propyl acetate 11.08%, toluene 12.61%, acetone 12.87%; mean 14.24%.
