"""Manuscript figures 2, 3, 5 and 6 (revised model)."""
from common import *
import broad as B, physics as ph
import figstyle as F
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
NCOL = 1e18   # standardized column density, molecules cm-2
tr_meas, tr_hitemp, hitran, test = split_groups(); mdl = B.main_model()
def win(species, lo=None):
    return next(g for g in test if g['species'] == species and (lo is None or abs(g['x'][0] - lo) < 1))
def tlabel(T): return f'{T:.2f}'.rstrip('0').rstrip('.') + ' K'

def fig2():
    g = win('n-propyl acetate'); x = g['x']
    fig = plt.figure(figsize=(9.6, 8.5)); gs = GridSpec(2, 2, height_ratios=[1.0, 2.25], hspace=0.20, wspace=0.12, left=0.085, right=0.985, top=0.975, bottom=0.075)
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis('off')
    ax.text(0.0, 39.5, '(a)', fontsize=14, va='top', ha='left')
    ax.text(51, 39.5, 'Both ensembles fitted on other molecules: three cold spectra paired with hot references', fontsize=11.5, va='top', ha='center')
    def box(cx, cy, w, h, text, fs=11):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0.0,rounding_size=0.9', fc='#EAF1F7', ec='#34495E', lw=1.3)); ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, linespacing=1.0)
    def arrow(p, q): ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=13, lw=1.4, color='#34495E', shrinkA=0, shrinkB=0))
    box(9.5, 17, 18, 13, 'Three cold spectra\n+ input temperatures'); box(31.0, 17, 18, 13, 'Spectral descriptors\n+ target T and P')
    box(54.5, 26.2, 21.5, 14.6, 'Cold shift and broadening\n→ physical reference\n→ envelope tree ensemble', 10.5); box(54.5, 7.8, 21.5, 14.6, 'Qualifying isolated peaks\n→ cold line fits\n→ line tree ensemble', 10.5)
    box(77.5, 17, 16.5, 13, 'Combine envelope\nand line contrast'); box(93.0, 17, 11.5, 14.6, 'Hot spectrum\n↓\nProbe value')
    arrow((18.5, 17), (22.0, 17)); arrow((40.0, 19), (43.75, 25.5)); arrow((40.0, 15), (43.75, 8.5)); arrow((65.25, 25.5), (69.25, 19.5)); arrow((65.25, 8.5), (69.25, 14.5)); arrow((85.75, 17), (87.25, 17))
    axb = fig.add_subplot(gs[1, 0]); axc = fig.add_subplot(gs[1, 1], sharey=axb); cols = [F.BLUE, F.GREEN, F.ORANGE, F.PINK]
    styles = [('--', '#444444'), ('-', '#888888'), ('-', 'k')]
    for k in range(3): axb.plot(x, g['inputs'][k] * NCOL, ls=styles[k][0], color=styles[k][1], lw=1.5, label=f"PNNL {g['Tin'][k]:.2f} K")
    for i, r in enumerate(g['records']):
        m = g['masks'][i]; axb.plot(x[m], g['references'][i][m] * NCOL, color=cols[i], lw=1.6, label=tlabel(r['temperature_K']))
        y = B.predict(mdl, g, r['temperature_K']) * NCOL; axc.plot(x, y, '--', color=cols[i], lw=1.6, marker='^', markevery=(12 + 3 * i, 55), ms=5.5, label=tlabel(r['temperature_K']))
    axb.set_title('(b) Inputs and withheld measurements', fontsize=14); axc.set_title('(c) Predicted outputs', fontsize=14)
    for a in (axb, axc): a.set_xlabel(F.WN, fontsize=13); a.set_xlim(1163, 1328); a.tick_params(labelsize=11)
    axb.set_ylabel('Standardized absorbance', fontsize=13); axb.set_ylim(-0.08, 2.12); plt.setp(axc.get_yticklabels(), visible=False)
    axb.legend(loc='upper right', fontsize=9.5, handlelength=2.2, labelspacing=0.55); axc.legend(loc='upper right', fontsize=9.5, handlelength=2.6, labelspacing=0.55)
    ins = axb.inset_axes([0.045, 0.56, 0.32, 0.385]); 
    for k in range(3): ins.plot(x, g['inputs'][k] * NCOL, ls=styles[k][0], color=styles[k][1], lw=1.5)
    ins.set_xlim(1235.5, 1247.5); ins.set_ylim(1.56, 2.04); ins.set_xticks([]); ins.set_yticks([]); ins.grid(False)
    for s in ins.spines.values(): s.set_visible(True); s.set_color('#666666'); s.set_linewidth(1.0)
    axb.indicate_inset_zoom(ins, edgecolor='#888888', alpha=1.0, lw=1.0)
    F.save(fig, FIGS / 'Figure_02')

def spectral_block(fig, sub, g, recs, labels, title, legend_cols=2, ypad=1.32):
    """Measured panel, prediction panel (with the warmest input) and a residual panel underneath."""
    inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=sub, height_ratios=[2.75, 1.0], hspace=0.42, wspace=0.10)
    a1 = fig.add_subplot(inner[0, 0]); a2 = fig.add_subplot(inner[0, 1], sharey=a1); a3 = fig.add_subplot(inner[1, :]); x = g['x']; cols = [F.BLUE, F.GREEN, F.ORANGE, F.PINK]; top = 0; lo, hi = 1e9, -1e9
    for c, i in zip(cols, recs):
        r = g['records'][i]; m = g['masks'][i]; ref = g['references'][i] * NCOL; y = B.predict(mdl, g, r['temperature_K']) * NCOL
        a1.plot(x[m], ref[m], color=c, lw=1.5, label=tlabel(r['temperature_K'])); a2.plot(x, y, '--', color=c, lw=1.5, marker='^', ms=4.5, markevery=(10 + 4 * recs.index(i), 60)); a3.plot(x[m], (ref - y)[m], color=c, lw=1.2)
        top = max(top, ref[m].max(), y.max()); lo, hi = min(lo, x[m][0]), max(hi, x[m][-1])
    a2.plot(x, g['warm'] * NCOL, color='k', lw=1.4, label=f"PNNL {g['Tin'][-1]:.1f} K"); top = max(top, (g['warm'] * NCOL).max())
    a1.set_ylim(-0.03 * top, ypad * top); a1.set_xlim(x[0], x[-1]); a2.set_xlim(x[0], x[-1]); a3.set_xlim(lo, hi); plt.setp(a2.get_yticklabels(), visible=False)
    a1.set_title(f'({labels[0]}) Measured reference', fontsize=12); a2.set_title(f'({labels[1]}) Model prediction', fontsize=12)
    a1.legend(loc='upper left', ncol=legend_cols, fontsize=8.5, columnspacing=1.2, handlelength=1.8, borderaxespad=0.3); a2.legend(loc='upper left', fontsize=8.5, handlelength=1.8, borderaxespad=0.3)
    a1.set_ylabel('Standardized\nabsorbance', fontsize=11.5); a3.set_ylabel('Residual', fontsize=11.5); a3.set_xlabel(F.WN, fontsize=12); a3.axhline(0, color='#444444', lw=0.9)
    a3.text(0.985, 0.90, f'({labels[2]})', transform=a3.transAxes, ha='right', va='top', fontsize=12); a3.locator_params(axis='y', nbins=3)
    for a in (a1, a2): a.locator_params(axis='x', nbins=4)
    bb = sub.get_position(fig); fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.043, title, ha='center', va='bottom', fontsize=14)
    return a1, a2, a3

def fig3():
    fig = plt.figure(figsize=(10.4, 9.3)); gs = GridSpec(2, 2, hspace=0.36, wspace=0.20, left=0.075, right=0.985, top=0.925, bottom=0.06)
    def idx(g, Ts): return [min(range(len(g['records'])), key=lambda i: abs(g['records'][i]['temperature_K'] - T)) for T in Ts]
    g = win('di-n-propyl ether'); spectral_block(fig, gs[0, 0], g, idx(g, [604, 758, 838]), 'abc', 'di-n-propyl ether')
    g = win('n-propyl acetate'); spectral_block(fig, gs[0, 1], g, idx(g, [671, 754, 806, 862]), 'def', 'n-propyl acetate')
    g = win('toluene', 3000); spectral_block(fig, gs[1, 0], g, idx(g, [350, 450, 774.15]), 'ghi', 'toluene')
    g = win('acetone'); spectral_block(fig, gs[1, 1], g, idx(g, [604.81, 782.23, 974.22]), 'jkl', 'acetone')
    F.save(fig, FIGS / 'Figure_03')

def fig5():
    main = jload('main.json'); rows = main['rows']
    fig, axs = plt.subplots(1, 3, figsize=(10.2, 4.3), sharey=True, gridspec_kw=dict(wspace=0.07, left=0.07, right=0.99, top=0.9, bottom=0.14))
    groups_ = [('Non-acetal acyclic ethers', ETHERS, '(a)'), ('Other measured tests', OTHERS, '(b)'), ('HITEMP tests', list(LINE_RESOLVED), '(c)')]; names = {'CH4': r'CH$_4$'}
    for ax, (title, mols, lab) in zip(axs, groups_):
        for c, s in zip([F.BLUE, F.GREEN, F.ORANGE], mols):
            Ts = sorted({r['temperature_K'] for r in rows if r['species'] == s}); ys = [np.mean([r['model'] for r in rows if r['species'] == s and r['temperature_K'] == T]) for T in Ts]
            ax.plot(Ts, ys, '-o', color=c, ms=5, lw=1.5, label=names.get(s, s))
        ax.set_title(title, fontsize=14); ax.set_xlabel('Target temperature (K)', fontsize=13); ax.text(0.03, 0.965, lab, transform=ax.transAxes, va='top', fontsize=14)
        ax.legend(loc='lower right' if lab == '(c)' else 'upper right', fontsize=9.5)
    axs[0].set_ylabel('Integrated absolute error (%)', fontsize=13); axs[0].set_ylim(0, 72)
    F.save(fig, FIGS / 'Figure_05')

def fig6():
    d = jload('shift.json')['npa_peak']; T = [p['T'] for p in d]
    fig, ax = plt.subplots(figsize=(8.2, 3.9)); ax.plot(T, [p['measured'] for p in d], '-o', color=F.BLUE, ms=8, lw=1.6, label='Measured'); ax.plot(T, [p['predicted'] for p in d], '-s', color=F.ORANGE, ms=8, lw=1.6, label='Predicted')
    ax.set_title('n-propyl acetate', fontsize=13); ax.set_xlabel('Target temperature (K)', fontsize=12.5); ax.set_ylabel(r'Broad-band peak (cm$^{-1}$)', fontsize=12.5); ax.legend(fontsize=11, loc='upper right'); ax.set_ylim(1219, 1232.5)
    F.save(fig, FIGS / 'Figure_06')

if __name__ == '__main__':
    import sys
    for name in (sys.argv[1:] or ['fig2', 'fig3', 'fig5', 'fig6']): globals()[name](); print(name, 'done')
