"""Paths for the thIRmo-ET revision analyses (defaults relative to this repository).

Environment variables override the defaults:
  THIRMO_REPO       repository root (default: two levels above this file)
  THIRMO_DATASET    folder with the 69 prepared group_*.npz files and dataset_manifest.json (built with
                    reproduction/pipeline.py from the licensed source spectra; default: <repo>/data/dataset)
  THIRMO_OUT        folder for results, caches and figures (default: revision_2026/out)
  THIRMO_DIGITIZED  digitized published data and single-wavenumber correlations (default: revision_2026/digitized)
"""
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('THIRMO_REPO', HERE.parents[1]))
DATASET = Path(os.environ.get('THIRMO_DATASET', REPO / 'data' / 'dataset'))
OUT = Path(os.environ.get('THIRMO_OUT', HERE.parent / 'out'))
DIGITIZED = Path(os.environ.get('THIRMO_DIGITIZED', HERE.parent / 'digitized'))
RESULTS = OUT / 'results'; CACHE = OUT / 'cache'; FIGS = OUT / 'figures'
for p in (RESULTS, CACHE, FIGS): p.mkdir(parents=True, exist_ok=True)
