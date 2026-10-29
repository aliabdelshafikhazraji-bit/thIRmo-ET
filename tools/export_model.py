"""Developer utility: convert a TRUSTED sklearn model to safe numeric arrays.

Never run this on an untrusted pickle. End users do not need this script or sklearn.
"""
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trusted_pickle", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    with args.trusted_pickle.open("rb") as f:
        model = pickle.load(f)
    trees = [e.tree_ for e in model.estimators_]
    arrays = dict(offsets=np.r_[0, np.cumsum([t.node_count for t in trees])].astype("i8"),
                  feature=np.concatenate([t.feature for t in trees]).astype("i4"),
                  threshold=np.concatenate([t.threshold for t in trees]),
                  left=np.concatenate([t.children_left for t in trees]).astype("i4"),
                  right=np.concatenate([t.children_right for t in trees]).astype("i4"),
                  value=np.concatenate([t.value[:, 0, 0] for t in trees]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    metadata = dict(name="ThermoIR-ET", version="1.0.0", algorithm="Extremely randomized regression trees",
                    n_estimators=len(trees), n_features=42, max_depth=16, min_samples_leaf=16,
                    max_features=0.8, random_state=20261029, training_identities=31,
                    hitran_source_weight=0.5, weight_selection="Training-identity cross-validation",
                    source_model_sha256=hashlib.sha256(args.trusted_pickle.read_bytes()).hexdigest(),
                    weights_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                    input_mode_fit="Two or three low-temperature spectra",
                    single_input_mode="Experimental adapter; input_count=1; no retraining",
                    output="Residual of area-normalized spectrum from warmest input",
                    normalization="Multiply (warmest normalized input + residual) by warmest positive area / band width; clip negative output to zero")
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf8")
    print(json.dumps(dict(nodes=len(arrays["value"]), bytes=args.output.stat().st_size, **metadata), indent=2))


if __name__ == "__main__":
    main()
