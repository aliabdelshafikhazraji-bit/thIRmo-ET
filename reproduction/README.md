# Study reproduction

The current study uses the strict three-input shared spectral function, version 4.0.1-experimental. Follow [the current reproduction guide](three_input/README.md). `study/` contains the current 69-group dataset manifest, 33 training and eight test identities, source records and revised errors. Every fitting and testing group has three actual near-ambient input spectra.

`pipeline.py` retains the source-unit conversion, 2 cm⁻¹ segmented convolution, fixed 512-point feature construction, species/source weights and numeric forest export. The current `train_broad.py` enforces the three-input restriction before invoking it. The root model and older protocol records are historical reproduction resources and are not the current default predictor.

Acquire licensed source data from PNNL, HITRAN/HITEMP and the cited publications. No raw third-party spectrum is redistributed here. The current guide defines native line-training arrays, the data interface and public commands for refitting and scoring both components.
