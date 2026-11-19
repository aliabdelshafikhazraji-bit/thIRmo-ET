"""Shared loading and scoring helpers. The public repository's dataset loader, feature construction,
weights and forest format are used unchanged."""
import sys, json, csv
from pathlib import Path
import numpy as np
from config import REPO, DATASET, RESULTS, CACHE, FIGS
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'three_input'))
import thirmo_et as inference                      # published feature construction (line-resolved branch)
from reproduction import pipeline as pub           # published training pipeline

FROZEN_FINAL = REPO / 'three_input/broad_model/models/thirmo_et_final.npz'
FROZEN_NOH = REPO / 'three_input/broad_model/models/thirmo_et_without_added_hitran.npz'
ORDER = ['1,2-dimethoxyethane', 'di-n-propyl ether', 'diisopropyl ether', 'n-propyl acetate', 'toluene', 'acetone', 'NO', 'CH4']
MEAS = ORDER[:6]; ETHERS = ORDER[:3]; OTHERS = ORDER[3:6]; LINE_RESOLVED = ('NO', 'CH4')
FAMILY = {'propane': 'alkanes', 'n-heptane': 'alkanes',
          'ethylene': 'alkenes', 'propene': 'alkenes', '1-butene': 'alkenes', 'isobutene': 'alkenes', '1,3-butadiene': 'alkenes',
          'toluene': 'aromatics', 'ethanol': 'alcohols', 'acetaldehyde': 'aldehydes and ketones', 'acetone': 'aldehydes and ketones',
          'methyl formate': 'esters', 'methyl acetate': 'esters', 'ethyl acetate': 'esters', 'n-propyl acetate': 'esters', 'n-butyl acetate': 'esters', 'methyl propionate': 'esters',
          'dimethoxymethane': 'ethers and acetals', '1,2-dimethoxyethane': 'ethers and acetals', 'di-n-propyl ether': 'ethers and acetals', 'diisopropyl ether': 'ethers and acetals'}

_groups = None
def groups():
    """All 69 prepared windows. Each group: x(512), inputs(3,512), input temperatures, references, masks, records."""
    global _groups
    if _groups is None:
        _groups = pub.load_dataset(DATASET)
        for gi, g in enumerate(_groups):
            g['gi'] = gi; g['kind'] = g['records'][0]['source_kind']
            g['Tin'] = np.asarray(g['input_temperatures_K'], float); g['warm'] = g['inputs'][-1]
            assert np.all(np.diff(g['Tin']) > 0)
    return _groups

def split_groups(G=None):
    G = G or groups()
    tr_meas = [g for g in G if g['split'] == 'training' and g['kind'] == 'measurement']
    tr_hitemp = [g for g in G if g['split'] == 'training' and g['kind'] == 'HITEMP']
    hitran = [g for g in G if g['split'] == 'hitran']; test = [g for g in G if g['split'] == 'testing']
    return tr_meas, tr_hitemp, hitran, test

def E(pred, ref, m):
    """Eq. (4): normalized integrated absolute error in percent."""
    return 100 * float(np.abs(pred[m] - ref[m]).sum() / np.maximum(ref[m], 0).sum())

def published_rows():
    """Per-record errors of the published three-input model (study/record_errors.csv)."""
    with open(REPO / 'study/record_errors.csv', newline='', encoding='utf8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('temperature_K', 'pressure_atm', 'final_error_percent', 'without_hitran_error_percent', 'unchanged_error_percent'): r[k] = float(r[k])
    return rows

def summarize(rows, key='error_percent'):
    """records -> windows -> molecules (equal weights at each level), as in the published pipeline."""
    out = {}
    for s in sorted({r['species'] for r in rows}):
        wins = sorted({r['window'] for r in rows if r['species'] == s})
        out[s] = float(np.mean([np.mean([r[key] for r in rows if r['species'] == s and r['window'] == w]) for w in wins]))
    return out

def mean_of(d, keys): return float(np.mean([d[k] for k in keys]))

def tree_predict_all(forest, X):
    """Per-tree predictions (n_trees, n_rows) from a published NumericForest."""
    X = np.asarray(X, dtype='f4'); rows = np.arange(len(X)); out = []
    for start, end in zip(forest.offsets[:-1], forest.offsets[1:]):
        nodes = np.zeros(len(X), dtype=np.int64)
        for _ in range(64):
            feats = forest.feature[start + nodes]; active = feats >= 0
            if not np.any(active): break
            cur = start + nodes[active]
            go_left = X[rows[active], feats[active]] <= forest.threshold[cur]
            nodes[active] = np.where(go_left, forest.left[cur], forest.right[cur])
        out.append(forest.value[start + nodes])
    return np.asarray(out)

def jdump(obj, name):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    p = RESULTS / name; p.write_text(json.dumps(obj, indent=1, default=conv), encoding='utf8'); return p
def jload(name): return json.loads((RESULTS / name).read_text(encoding='utf8'))
