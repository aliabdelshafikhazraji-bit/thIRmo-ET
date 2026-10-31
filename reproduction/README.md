# Reproducing the study calculations

The inference program and fitted numeric weights can be used immediately. This directory also supplies the study's preparation, training and scoring code, with no dependency on the author's private workspace. Version 2.5.1 retains the fitted weights of 2.5.0 and updates the evaluation to use HITEMP for all methane test windows.

## What is included

- `pipeline.py`: source-unit conversion, segmented Gaussian convolution, 512-point sampling, feature construction, sample weights, fitting, numeric-model export and masked scoring.
- `../study/dataset_manifest.json`: all 90 prepared band groups, exact input and target temperatures, pressures, test/training roles, spectral windows and checksums for each prepared array. It records 46 training identities and eight test identities.
- `../study/source_records.json`: retained target identifiers, source editions or DOIs and available source-file provenance.
- `../study/protocol.json`: the existing study partition and preprocessing protocol, synchronized with the methane HITEMP reference.
- `../study/record_errors.csv`: 84 individual band-temperature errors for the released model; `all_record_errors.csv` additionally records the existing comparison predictions.
- `../study/current_summaries.json`: the current molecule/window means and temperature-bin summaries.
- `../study/CH4_outer_HITEMP_audit.json`: exact temperatures, source checksum and near-ambient source-screen results for the methane outer window.

The held-out ether group comprises **three non-acetal acyclic ethers**: 1,2-dimethoxyethane, di-n-propyl ether and diisopropyl ether. Dimethoxymethane and the cyclic acetals remain in training. The description does not claim exclusion of every compound containing ether linkages.

## Source spectra and access

Original and processed third-party absorption spectra are not redistributed in this repository. Their publication or online availability is not assumed to grant unrestricted redistribution rights. Obtain the relevant files from the source provider, retaining the provider's citation and applicable terms:

1. PNNL near-ambient spectra: [PNNL infrared database](https://nwir.pnl.gov/). The manifest identifies the local library species folder when available and the actual measured input temperatures. PNNL values are decadic absorbance normalized to 1 ppm m at the fixed 296 K reference density; the source sample temperature is retained separately.
2. Line-list references: [HITRAN](https://hitran.org/) and [HITEMP](https://hitran.org/hitemp/). HITEMP editions are 2010 for H2O/CO2, 2019 for CO/NO/NO2 and 2020 for CH4. HITRAN2024 is the lower-weight training source. The existing calculation uses 1 atm air broadening, natural isotope abundances, temperature-dependent line strengths, a Voigt profile and 25 cm-1 wing cutoff, followed by the documented 2 cm-1 Gaussian convolution. Follow the individual source edition and band temperatures in the manifest; the methane 4000-4700 cm-1 test uses nine temperatures, 373.15-773.15 K in 50 K increments.
3. Published measured spectra: follow each record's DOI in `source_records.json` and obtain the publisher supplementary workbook or author/group archive. Source-file provenance, original units, worksheet names or other retained extraction details are included where available. Records originally recovered from figures require the corresponding recovered coordinates; the original publication alone does not guarantee bitwise reproduction of those coordinates.

For exact numerical reproduction, obtain the source arrays corresponding to `study/dataset_manifest.json` and place the named `group_*.npz` files beside that manifest. The numeric bundle is a documented data interface, not a publicly downloadable dataset. All array hashes are published so an acquired copy can be verified. The package does not claim that all original calibrated/source-recovery arrays are publicly retrievable or that every source can be automatically downloaded. Where a provider does not offer those arrays, contact the corresponding author named in the manuscript to resolve data access. No supplementary archive is required by this code release.

## Install

Use Python 3.12 or newer for the pinned reproduction environment. From the repository root:

```sh
python -m pip install -r reproduction/requirements.txt
python -m reproduction.pipeline --help
```

The scientific versions used for this release check are recorded in `study/environment.json`. Numeric tree inference does not require scikit-learn. Training uses the exact tree parameters, seed and weighting equations in `pipeline.py`; differing numerical-library versions or floating-point platforms may prevent byte-for-byte equality of a refitted tree file.

## Reproduce the reported aggregate from published per-record errors

This calculation works without external spectra:

```sh
python -m reproduction.pipeline aggregate --errors study/record_errors.csv --output results/aggregated.json
```

It should give an equal-molecule mean of 22.74867520735205 percent overall, 21.606796659375785 percent for the six experimental test molecules, and 26.17431085128086 percent for NO and CH4. These values are calculated from `study/record_errors.csv` and recorded in `study/aggregation_expected.json`. The denominator for each record is positive integrated reference cross section; errors are averaged over records within each band, bands within each molecule and then molecules equally. The published aggregate accuracy uses three measured input temperatures for each test molecule.

## Prepare acquired source curves

Copy `source_manifest.example.json` outside the repository and replace its illustrative entries with the actual source files and conditions. Each CSV is a native-resolution two-column curve, with a header by default. Missing target segments must remain NaN; gaps greater than 5 cm-1 are not filled. Specify the source units explicitly. Supported units include `pnnl`, `cross-section` (cm2/molecule), `m2/mol`, `cm2/mol`, `Pa-1 m-1`, `absorbance-ln` and `absorbance-decadic`. Absorbance inputs require their actual absorber column density.

```sh
python -m reproduction.pipeline prepare --source-manifest acquired/source_curves.json --output acquired/prepared
```

The preparer applies the existing study's extra 2 cm-1 Gaussian smoothing within each observed segment, interpolates onto the declared 512-point band grid, requires full input coverage, and trims 3 cm-1 at reference segment edges. It does not choose bands, change the split or tune a model from the test errors. Match the existing study's source curves, declared windows and masks to reproduce its exact arrays; preparing a different curve export can produce a different hash.

## Recompute predictions and errors

With the acquired prepared dataset and its `dataset_manifest.json`:

```sh
python -m reproduction.pipeline evaluate --dataset acquired/prepared --model models/thirmo_et_final.npz --output results/released_model
```

The loader verifies file/array checksums and molecular separation. It predicts each held-out target with the numeric forest and scores only observed reference positions. It does not refit the model.

## Refit the declared model, when the source data are available

```sh
python -m reproduction.pipeline train --dataset acquired/prepared --hitran-weight 0.5 --output results/refitted.npz
python -m reproduction.pipeline train --dataset acquired/prepared --hitran-weight 0 --output results/no_added_HITRAN.npz
```

The first fit uses 46 effective training identities; the second uses the shared 28 experimental/HITEMP identities. The 19 added HITRAN identities include 18 additional molecular identities plus ethylene, which already has measured training data. An original-source identity has raw weight 1 and a HITRAN identity has raw weight 0.5. Weight is divided among that identity's bands, source domains, records and valid wavenumber positions. Neither fit uses held-out molecular identities.

## Verification and remaining access limits

`study/reproduction_verification.json` records checks against the existing prepared arrays and the original stored predictions. The release check does not retrain the forest or independently recreate every source measurement or line-list simulation. It establishes portable feature/weight assembly, scoring and aggregation for the documented arrays. Exact independent replication from source publications additionally depends on access to the source files and recovered coordinates listed above.
