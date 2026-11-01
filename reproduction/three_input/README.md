# Reproducing the current three-input calculation

The current inference function is `three_input/spectral_model.py`. `coherent_protocol.json`, `qualification_audit_correction.json` and `coherent_freeze_before_evaluation.json` preserve development and qualification decisions. A preceding mass-proxy protocol is retained as historical evidence and is not the current training procedure.

## Broad envelope

Acquire the source arrays identified by `study/dataset_manifest.json` (69 groups, each with exactly three genuine inputs). The dataset separates 33 training and eight test identities, and gives all array checksums. Place its `group_*.npz` arrays alongside the manifest. The trained target ordinate is a normalized change from the warmest input; names and molecular properties are not feature columns.

Use `train_broad.py --dataset acquired/prepared --output new_models` to verify the strict-input structure and refit the stated 128-tree envelope and its paired variant without the added HITRAN envelope block. The current manifest, source records and scoring tables replace the historical 90-group study.

## Isolated-line component

Acquire `CO_native_training.npz`, `HCl_native_training.npz`, `HF_native_training.npz`, the three native PNNL files per identity, and `cold_gate.json`, following their providers' terms. The NPZ source data were generated from the cited official HITEMP2019 CO and HITRAN2024 HCl/HF lists. Tabulated constants are needed to produce reference simulations, but they are not read by the learning procedure. The learner loads only wavenumber, temperature, pressure and spectral ordinate arrays.

Copy `training_inputs.example.json` to `training_inputs.json` inside the acquired data directory and set the cold-spectrum paths (relative paths resolve there). The example contains paths, temperatures and hashes only. The native arrays have `wavenumber_cm1` (lines × 1001), `temperatures_K` (three cold + eight hot), `pressures_atm` (0.7, 1.0, 1.5) and `spectra_native_cm2_molecule` (lines × temperatures × pressures × 1001). Preserve the source bundle's line order for the compatibility screen. `cold_gate.json` provides a pass decision per species/line index, based on ≤20% mismatch at all three PNNL temperatures. No failed line supplies labels or the Gaussian coefficient.

```sh
python reproduction/three_input/train_spectra_only.py --data acquired/native --output new_models/lines
```

This script first fits the hot reference spectral profiles to obtain area, effective Lorentz width and a common spectral Gaussian coefficient. It jointly fits the three measured cold spectra; tabulated mass, lower-state energy, line strength and width are not learning features or labels. The line area/width correction uses ten cold-spectral and temperature-ratio features. Leave-one-species-out selection compares the residual head with its population baseline using CO/HCl/HF only. Each fold estimates its Gaussian coefficient from its fitting species only. The final fit uses 50 qualified cold lines and 1021 reference fits. Full training audits are written beside new weights; test hot spectra are never read by this script.

## Score the common function

Acquire the prepared reference arrays and the full-resolution cold files in `study/native_input_manifest.json`. Run `evaluate_common.py --dataset acquired/prepared --native-input-manifest acquired/native_input_manifest.json --output results`. Relative native paths resolve beside that manifest. The script cold-selects and fits the resolved components, applies the same intrinsic spectral expression for every request, then applies the declared 2 cm⁻¹ Gaussian display response to band predictions. Fine-grid point predictions remain unfiltered for sensing.

`recalculate_histories.py` accepts externally supplied published concentrations and reference/predicted cross sections. It applies the inverse-cross-section concentration ratio and reports the time-integrated difference over positive integrated reference concentration. Published concentration/calibration uncertainties remain separate from model discrepancies.

Raw third-party spectra and original recovered coordinates are not distributed. Published checksums describe an exact reproduction interface, not a claim that every underlying array can be automatically downloaded. Inference can be used immediately without these source files. This development is a fixed held-out-species assessment, not a newly blinded validation after earlier results were inspected.
