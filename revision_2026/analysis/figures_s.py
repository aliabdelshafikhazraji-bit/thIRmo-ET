"""Supplementary figures: leave-one-molecule-out errors, extrapolated versus fitted band shifts, prediction intervals."""
from common import *
import broad as B, physics as ph
import figstyle as F
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pickle
NCOL = 1e18

def figS1():
    v = jload('validation.json'); m = v['lomo']['molecule']; lf = v['lofo']['molecule']; fams = ['alkanes', 'alkenes', 'aromatics', 'alcohols', 'aldehydes and ketones', 'esters', 'ethers and acetals']
    order = [s for f in fams for s in sorted(m['model']) if FAMILY[s] == f]; test_set = set(MEAS)
    fig, ax = plt.subplots(figsize=(8.8, 8.0)); y = np.arange(len(order))
    for i, s in enumerate(order):
        ax.plot([min(m['model'][s], m['physics'][s], m['unchanged'][s]), max(m['model'][s], m['physics'][s], m['unchanged'][s])], [i, i], color='#C8C8C8', lw=1.3, zorder=1)
        ax.plot(m['unchanged'][s], i, 'o', color='#8C8C8C', ms=6.5, zorder=2); ax.plot(m['physics'][s], i, 's', color=F.BLUE, ms=7, zorder=3); ax.plot(m['model'][s], i, 'D', color=F.ORANGE, ms=7, zorder=4)
        ax.plot(lf['model'][s], i, 'x', color='k', ms=6, mew=1.3, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([s + (' *' if s in test_set else '') for s in order], fontsize=10.5); ax.invert_yaxis(); ax.grid(axis='y', alpha=0); ax.set_xlabel('Molecule-mean integrated absolute error (%)', fontsize=12); ax.set_xlim(0, 85)
    k = 0
    for f in fams:
        n = sum(FAMILY[s] == f for s in order); ax.text(84, k + (n - 1) / 2, f, ha='right', va='center', fontsize=9.5, color='#555555', style='italic')
        if k: ax.axhline(k - 0.5, color='#DDDDDD', lw=0.8)
        k += n
    ax.legend(handles=[Line2D([], [], marker='o', ls='', color='#8C8C8C', ms=6.5, label='Unchanged warmest input'), Line2D([], [], marker='s', ls='', color=F.BLUE, ms=7, label='Physical stage, Eq. (3c)'),
                       Line2D([], [], marker='D', ls='', color=F.ORANGE, ms=7, label='thIRmo-ET, molecule left out'), Line2D([], [], marker='x', ls='', color='k', ms=6, mew=1.3, label='thIRmo-ET, family left out')], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9.5, columnspacing=1.5)
    F.save(fig, FIGS / 'Figure_S1')

def figS2():
    d = jload('shift.json')['rows']; fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(wspace=0.28, left=0.08, right=0.985, top=0.92, bottom=0.14)); cols = dict(zip(MEAS, [F.BLUE, F.GREEN, F.ORANGE, F.PINK, '#56B4E9', '#E69F00']))
    for r in d:
        axs[0].plot(-r['shift_fit'], -r['shift_pred'], 'o', color=cols[r['species']], ms=5.5, alpha=0.9); axs[1].plot(r['broad_fit'], r['broad_pred'], 'o', color=cols[r['species']], ms=5.5, alpha=0.9)
    for ax, lim, t in ((axs[0], 28, '(a) Band shift toward lower wavenumber'), (axs[1], 36, '(b) Added Gaussian broadening')):
        ax.plot([0, lim], [0, lim], color='#444444', lw=0.9); ax.set_xlim(-2.5 if lim == 28 else 0, lim); ax.set_ylim(-2.5 if lim == 28 else 0, lim); ax.set_title(t, fontsize=12.5); ax.set_aspect('equal')
        ax.set_xlabel('Fitted to the withheld measurement (cm$^{-1}$)'); ax.set_ylabel('Extrapolated from the three inputs (cm$^{-1}$)')
    axs[0].legend(handles=[Line2D([], [], marker='o', ls='', color=cols[s], ms=5.5, label=s) for s in MEAS], fontsize=9, loc='upper left')
    F.save(fig, FIGS / 'Figure_S2')

def figS3():
    recs = pickle.loads((CACHE / 'interval_test_records.pkl').read_bytes()); iv = jload('intervals.json'); q = float(iv['broad_global']['q']['None']); beta = 0.02
    want = [('di-n-propyl ether', None, 838), ('n-propyl acetate', None, 862), ('toluene', 3000, 774.15), ('acetone', None, 974.22)]
    fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2), gridspec_kw=dict(hspace=0.36, wspace=0.22, left=0.08, right=0.985, top=0.95, bottom=0.08))
    for ax, (sp, lo, T), lab in zip(axs.ravel(), want, 'abcd'):
        r = min([r for r in recs if r['species'] == sp and (lo is None or abs(r['x'][0] - lo) < 1)], key=lambda r: abs(r['T'] - T)); x = r['x']; m = r['mask']; half = q * (r['tree_sd'] + beta * r['pred'].max())
        ax.fill_between(x, np.maximum(r['pred'] - half, 0) * NCOL, (r['pred'] + half) * NCOL, color=F.ORANGE, alpha=0.25, lw=0, label='Calibrated 90% interval')
        ax.plot(x, r['pred'] * NCOL, '--', color=F.ORANGE, lw=1.5, label='thIRmo-ET'); ax.plot(x[m], r['ref'][m] * NCOL, color='k', lw=1.3, label='Measured')
        cov = np.mean((r['ref'][m] >= np.maximum(r['pred'] - half, 0)[m]) & (r['ref'][m] <= (r['pred'] + half)[m])) * 100
        ax.set_title(f'({lab}) {sp}, {T:g} K', fontsize=12.5); ax.set_xlabel(F.WN); ax.set_xlim(x[0], x[-1]); ax.text(0.03, 0.95, f'coverage {cov:.0f}%', transform=ax.transAxes, va='top', fontsize=10)
        if lab in 'ac': ax.set_ylabel('Standardized absorbance')
        if lab == 'b': ax.legend(fontsize=8.8, loc='upper right')
    F.save(fig, FIGS / 'Figure_S3')

if __name__ == '__main__':
    import sys
    for name in (sys.argv[1:] or ['figS1', 'figS2', 'figS3']): globals()[name](); print(name, 'done')
