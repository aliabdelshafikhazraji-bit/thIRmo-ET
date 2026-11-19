"""Line-resolved branch (NO, CH4 among the test identities): the published thIRmo-ET envelope, unchanged.
The frozen forests distributed in the public repository are evaluated on the prepared dataset inputs."""
from common import *
_F = {}
def forest(name='final'):
    if name not in _F: _F[name] = inference.NumericForest(FROZEN_FINAL if name == 'final' else FROZEN_NOH)
    return _F[name]
def predict(g, T, name='final', per_tree=False):
    X = inference.features_at(g['features'], g['Tin'][-1], T); scale = g['area'] / np.ptp(g['x'])
    if per_tree: return np.maximum(g['anchor'][None, :] + tree_predict_all(forest(name), X), 0) * scale
    return np.maximum(g['anchor'] + forest(name).predict(X), 0) * scale
