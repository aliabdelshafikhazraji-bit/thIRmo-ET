"""Manuscript figures 7, 8, 10 and 11 (revised model)."""
from common import *
import broad as B, physics as ph, ethers as ET, benzene as BZ
import figstyle as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
NAMES = {'CH4': r'CH$_4$'}
T_VALID = 974.22   # highest temperature of the experimental test spectra

def fig7():
    h = jload('hitran.json'); main = jload('main.json'); pubw = h['original_submission']
    without = {s: h['broad_weight_scan']['0.0']['molecule'][s] for s in MEAS}; with_ = {s: h['broad_weight_scan']['0.5']['molecule'][s] for s in MEAS}
    for s in LINE_RESOLVED: without[s] = pubw['without_hitran'][s]; with_[s] = pubw['with_hitran'][s]
    fig, ax = plt.subplots(figsize=(8.6, 5.3)); y = np.arange(len(ORDER))
    for i, s in enumerate(ORDER):
        ax.plot([without[s], with_[s]], [i, i], color='#B0B0B0', lw=1.5, zorder=1)
        used_with = s in LINE_RESOLVED
        ax.plot(without[s], i, 's', color=F.ORANGE, ms=8.5, zorder=3, mec='k' if not used_with else F.ORANGE, mew=1.4 if not used_with else 0)
        ax.plot(with_[s], i, 'D', color=F.BLUE, ms=8.5, zorder=3, mec='k' if used_with else F.BLUE, mew=1.4 if used_with else 0)
    ax.set_yticks(y); ax.set_yticklabels([NAMES.get(s, s) for s in ORDER], fontsize=12); ax.invert_yaxis(); ax.set_xlabel('Molecule-mean integrated absolute error (%)', fontsize=12.5); ax.grid(axis='y', alpha=0)
    ax.legend(handles=[Line2D([], [], marker='s', ls='', color=F.ORANGE, ms=8.5, label='Without added HITRAN envelope data'), Line2D([], [], marker='D', ls='', color=F.BLUE, ms=8.5, label='With weighted HITRAN envelope data'),
                       Line2D([], [], marker='o', ls='', mfc='none', mec='k', mew=1.4, ms=9, label='Variant used in thIRmo-ET (black outline)')], loc='upper right', fontsize=10)
    ax.set_xlim(5, 45); F.save(fig, FIGS / 'Figure_07')

def fig8():
    w = jload('wakatsuki.json'); three = w['three_input']['rows']; wide = w['wide_input']
    fig = plt.figure(figsize=(11.6, 4.1)); gl = GridSpec(1, 2, wspace=0.30, left=0.062, right=0.60, top=0.90, bottom=0.15); gr = GridSpec(1, 1, left=0.775, right=0.99, top=0.90, bottom=0.15)
    for k, (sp, lab) in enumerate([('n-propyl acetate', '(a) n-propyl acetate'), ('di-n-propyl ether', '(b) di-n-propyl ether')]):
        ax = fig.add_subplot(gl[0, k]); rr = sorted([r for r in three if r['species'] == sp], key=lambda r: r['temperature_K']); T = [r['temperature_K'] for r in rr]
        ax.plot(T, [r['thirmo'] for r in rr], '-o', color=F.BLUE, ms=5.5, label='thIRmo-ET'); ax.plot(T, [r['wakatsuki_three_input_bounded'] for r in rr], '-s', color=F.ORANGE, ms=5.5, label='Wakatsuki equation,\nthree PNNL inputs')
        ax.set_title(lab, fontsize=12.5); ax.set_xlabel('Target temperature (K)'); ax.set_ylim(0, 125 if k == 0 else 85)
        if k == 0: ax.set_ylabel('Integrated absolute error (%)')
        ax.legend(fontsize=9, loc='upper left')
    ax = fig.add_subplot(gr[0, 0]); mols = ['1,2-dimethoxyethane', 'di-n-propyl ether', 'diisopropyl ether', 'toluene', 'acetone']; mw = wide['molecule']
    nin = {f['species']: (f['n_inputs'], max(f['fit_temperatures'])) for f in wide['fits']}
    for i, s in enumerate(mols):
        ax.plot([mw['thirmo'][s], mw['wakatsuki_wide_input'][s]], [i, i], color='#B0B0B0', lw=1.5, zorder=1)
        ax.plot(mw['thirmo'][s], i, 'o', color=F.BLUE, ms=8, zorder=3, label='thIRmo-ET,\nthree PNNL inputs' if i == 0 else None); ax.plot(mw['wakatsuki_wide_input'][s], i, 's', color=F.ORANGE, ms=8, zorder=3, label='Wakatsuki equation,\nextended inputs' if i == 0 else None)
    ax.set_yticks(range(len(mols))); ax.set_yticklabels([f'{s}\n({nin[s][0]} inputs to {nin[s][1]:.0f} K)' for s in mols], fontsize=9.5, linespacing=1.0); ax.set_ylim(len(mols) - 0.45, -1.75); ax.grid(axis='y', alpha=0); ax.set_xlim(7, 21.5); ax.set_xticks([8, 10, 12, 14, 16, 18, 20])
    ax.set_title('(c) Inputs extended to about 600 K', fontsize=12.5); ax.set_xlabel('Mean error above 615 K (%)'); ax.legend(fontsize=8.8, loc='upper center', ncol=2, columnspacing=1.0, handletextpad=0.3, borderaxespad=0.4)
    F.save(fig, FIGS / 'Figure_08')

def fig10():
    e = jload('ethers.json'); fig = plt.figure(figsize=(9.2, 10.6)); gs = GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.62, left=0.085, right=0.985, top=0.965, bottom=0.055)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.22)
    for k, (sp, lab) in enumerate([('di-n-propyl ether', 'a'), ('1,2-dimethoxyethane', 'b')]):
        ax = fig.add_subplot(top[0, k]); c = e[sp]['curve']; T = np.array([r['T'] for r in c]); ax.axvspan(T_VALID, T.max() + 40, color='#EDEDED', zorder=0, lw=0)
        ax.plot(T, np.array([r['corr'] for r in c]) * 1e19, color=F.BLUE, lw=2.0, label='Published correlation'); Tm = np.arange(min(m['T'] for m in e[sp]['measured']), T.max() + 1, 10.0); ax.plot(Tm, [ET.sigma_probe(B.main_model(), sp, t) * 1e19 for t in Tm], '--', color=F.ORANGE, lw=1.8, label='thIRmo-ET')
        ms = [m for m in e[sp]['measured'] if m['covered']]; ax.plot([m['T'] for m in ms], [m['measured'] * 1e19 for m in ms], 'o', mfc='white', mec='k', ms=6, mew=1.2, label='Published spectra', zorder=5)
        sh = e[sp]['shock']; ax.plot([s['T'] for s in sh], [s['corr'] * 1e19 for s in sh], 'o', color=F.BLUE, ms=7); ax.plot([s['T'] for s in sh], [s['pred'] * 1e19 for s in sh], '^', color=F.ORANGE, ms=7.5)
        ax.set_xlim(min(T.min(), min(m['T'] for m in ms)) - 30, T.max() + 40); ax.set_xlabel('Temperature (K)'); ax.set_title(f'({lab}) {sp} · {e[sp]["probe"]:g} cm$^{{-1}}$', fontsize=12.5)
        if k == 0: ax.set_ylabel('Cross section\n(10$^{-19}$ cm$^2$ molecule$^{-1}$)')
        ax.legend(fontsize=9, loc='upper right' if k == 0 else 'lower left'); yl = ax.get_ylim(); ax.text(T_VALID + 12, yl[0] + 0.035 * (yl[1] - yl[0]), 'beyond tested\ntemperatures', fontsize=8.5, color='#555555', va='bottom', ha='left', linespacing=0.95)
    hist = e['histories']; labs = iter('cdefgh')
    for row, sp in enumerate(['di-n-propyl ether', '1,2-dimethoxyethane']):
        sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[row + 1], wspace=0.10); hs = sorted([h for h in hist if h['species'] == sp], key=lambda h: h['T5']); ymax = 1.05 if row == 0 else 1.05
        bb = gs[row + 1].get_position(fig); fig.text(0.5 * (bb.x0 + bb.x1), bb.y1 + 0.034, f'{sp} · {ET.PROBE[sp]:g} cm$^{{-1}}$', ha='center', va='bottom', fontsize=13.5)
        for k, h in enumerate(hs):
            ax = fig.add_subplot(sub[0, k]); t = np.array(h['t']); xp = np.array(h['x_pub']); xn = np.array(h['x_new'])
            ax.plot(t, xp, color=F.BLUE, lw=1.7, label='Published cross-section\nretrieval'); ax.plot(t, xn, '--', color=F.ORANGE, lw=1.7, label='thIRmo-ET retrieval')
            idx = np.linspace(0, len(t) - 1, 12).astype(int); ax.errorbar(t[idx], xp[idx], yerr=0.06 * xp[idx], fmt='none', ecolor=F.BLUE, elinewidth=1.2, capsize=2.5, label='Reported cross-section\nuncertainty component')
            ax.set_xlim(0, 1.0); ax.set_ylim(0, ymax); ax.set_xlabel('Time (ms)'); ax.set_title(f'({next(labs)}) {h["T5"]:.0f} K, {h["P5"]:.3f} atm', fontsize=12)
            if k == 0: ax.set_ylabel('Mole fraction (%)')
            else: plt.setp(ax.get_yticklabels(), visible=False)
            if row == 0 and k == 2: ax.legend(fontsize=8.5, loc='upper right', handlelength=1.8, labelspacing=0.6)
    F.save(fig, FIGS / 'Figure_10')

def fig11():
    b = jload('benzene.json'); rows = b['rows']; g = BZ.window(); mdl = B.main_model(); x = g['x']; sp = jload('seed_probe.json'); q = sp['q']; beta = sp['beta']
    fig = plt.figure(figsize=(10.2, 7.7)); gs = GridSpec(2, 2, hspace=0.38, wspace=0.26, left=0.085, right=0.985, top=0.95, bottom=0.075)
    ax = fig.add_subplot(gs[0, 0]); Ts = [600, 800, 1000, 1200, 1400]; cols = [F.BLUE, F.GREEN, F.ORANGE, F.PINK, '#56B4E9']
    for T, c in zip(Ts, cols):
        y = B.predict(mdl, g, T) * 1e18; ax.plot(x, y, color=c, lw=1.6, ls='-' if T <= B.T_CAP else '--', label=f'{T} K'); ax.plot([BZ.NU_LASER], [BZ.correlation(T) * 1e18], 'o', color=c, ms=7, mec='k', mew=0.9, zorder=5)
    ax.axvline(BZ.NU_LASER, color='#666666', lw=0.9, ls=':'); ax.set_xlim(650, 700); ax.set_ylim(0, 1.45); ax.set_xlabel(F.WN); ax.set_ylabel('Cross section\n(10$^{-18}$ cm$^2$ molecule$^{-1}$)'); ax.set_title('(a) Predicted band and laser values', fontsize=12.5)
    h, l = ax.get_legend_handles_labels(); h.append(Line2D([], [], marker='o', ls='', color='#999999', mec='k', mew=0.9, ms=7)); l.append('Laser correlation,\n673.94 cm$^{-1}$'); ax.legend(h, l, fontsize=9, loc='upper right')
    T = np.array([r['T'] for r in rows]); o = np.argsort(T); T = T[o]; get = lambda k: np.array([r[k] for r in rows])[o]
    lo, hi, glo, ghi = [], [], [], []
    for t in T:
        trees = B.predict(mdl, g, t, per_tree=True); y = B.predict(mdl, g, t); half = q * (trees.std(0) + beta * y.max()); hl = np.interp(BZ.NU_LASER, x, half); yl = np.interp(BZ.NU_LASER, x, y); lo.append(max(yl - hl, 0)); hi.append(yl + hl)
        nu, so = BZ.peak(x, y); ho = np.interp(nu, x, half); glo.append(max(so - ho, 0) / BZ.correlation(t)); ghi.append((so + ho) / BZ.correlation(t))
    lo, hi, glo, ghi = map(np.array, (lo, hi, glo, ghi))
    ax = fig.add_subplot(gs[0, 1]); ax.axvspan(T_VALID, 1500, color='#EDEDED', zorder=0, lw=0); ax.fill_between(T, lo * 1e18, hi * 1e18, color=F.ORANGE, alpha=0.16, lw=0, label='90% prediction interval')
    ax.plot(T, get('sigma_corr') * 1e18, color=F.BLUE, lw=2.0, label='Shock-tube correlation'); ax.plot(T, get('sigma_laser_pred') * 1e18, '--', color=F.ORANGE, lw=1.8, label='thIRmo-ET')
    ax.set_xlim(540, 1490); ax.set_ylim(0, 1.6); ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Cross section at 673.94 cm$^{-1}$\n(10$^{-18}$ cm$^2$ molecule$^{-1}$)'); ax.set_title('(b) Fixed laser wavenumber', fontsize=12.5); ax.legend(fontsize=9.5, loc='upper right')
    ax.text(T_VALID + 14, 1.0, 'beyond tested\ntemperatures', fontsize=8.5, color='#555555', va='bottom', ha='left', linespacing=0.95)
    ax = fig.add_subplot(gs[1, 0]); ax.axvspan(T_VALID, 1500, color='#EDEDED', zorder=0, lw=0); ax.plot(T, get('nu_opt'), color=F.ORANGE, lw=1.8, label='Predicted optimum'); ax.axhline(BZ.NU_LASER, color=F.BLUE, lw=1.8, label='Fixed wavenumber, 673.94 cm$^{-1}$')
    ax.set_xlim(540, 1490); ax.set_ylim(667.5, 675); ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Wavenumber of maximum\nabsorption (cm$^{-1}$)'); ax.set_title('(c) Temperature-specific optimum', fontsize=12.5); ax.legend(fontsize=9.5, loc='lower left')
    ax = fig.add_subplot(gs[1, 1]); ax.axvspan(T_VALID, 1500, color='#EDEDED', zorder=0, lw=0); ax.fill_between(T, glo, ghi, color=F.ORANGE, alpha=0.16, lw=0, label='90% prediction interval'); ax.plot(T, get('gain_vs_corr'), color=F.ORANGE, lw=1.8, label='Predicted optimum relative to\nthe measured value at 673.94 cm$^{-1}$')
    ax.axhline(1, color='#444444', lw=0.9); ax.set_xlim(540, 1490); ax.set_ylim(0, 6.5); ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Absorbance relative to\nthe fixed wavenumber'); ax.set_title('(d) Sensitivity at the predicted optimum', fontsize=12.5); ax.legend(fontsize=9, loc='upper left')
    F.save(fig, FIGS / 'Figure_11')

if __name__ == '__main__':
    import sys
    for name in (sys.argv[1:] or ['fig7', 'fig8', 'fig10', 'fig11']): globals()[name](); print(name, 'done')
