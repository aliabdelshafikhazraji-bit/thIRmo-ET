"""Export the broad-band forest in the repository's portable weight format and verify that it reproduces the predictions."""
from common import *
import broad as B, hashlib
from config import OUT
mdl = B.main_model(); path = OUT / 'models' / 'thirmo_et_broad_band.npz'
tr_meas, tr_hitemp, hitran, test = split_groups()
pub.export_forest(mdl, path, dict(description='thIRmo-ET broad-band envelope: learned correction to the physical reference of Eq. (3c)', predictors=B.FEATURE_NAMES, temperature_cap_K=B.T_CAP, random_state=B.SEED,
                                  physics=dict(nu_e_max_cm1=300.0, k_shift=1.0, k_w=2.5), training_identities=sorted({g['species'] for g in tr_meas + tr_hitemp}), testing_identities=sorted({g['species'] for g in test})))
f = inference.NumericForest(path); g = next(g for g in test if g['species'] == 'acetone'); X, fr = B.predictors(g, 800.0)
print('max abs difference sklearn vs exported:', float(np.max(np.abs(tree_predict_all(f, X).mean(0) - mdl.predict(X)))))
print('sha256', hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes', path.stat().st_size)
