"""Refit the declared envelope only after enforcing three real input spectra."""
from pathlib import Path
import sys,argparse,numpy as np
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from reproduction import pipeline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    groups=pipeline.load_dataset(args.dataset)
    for g in groups:
        ts=g['input_temperatures_K']
        if len(g['inputs'])!=3 or len(ts)!=3 or len(np.unique(ts))!=3 or np.any((ts<275)|(ts>350)):
            raise ValueError('Every training and test group must contain three actual distinct-temperature measurements within 275–350 K.')
    args.output.mkdir(parents=True,exist_ok=True)
    pipeline.train(args.dataset,args.output/'broad_final.npz',.5)
    pipeline.train(args.dataset,args.output/'broad_without_hitran.npz',0.)
    print('Fitted the strict three-input envelope and paired source-ablation variant.')

if __name__=='__main__':main()
