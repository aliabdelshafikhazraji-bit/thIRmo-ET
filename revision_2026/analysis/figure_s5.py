"""Figure S5: diisopropyl ether at 1121.82 cm-1. Published shock-tube correlation (Adil and Farooq) over its stated
500-1500 K range, the scanned test spectra of the same study at that wavenumber, and the thIRmo-ET prediction from the
three PNNL spectra with its calibrated 90% interval."""
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import figstyle as F
R = jload('dipe_probe.json'); T_VALID = 974.22
rows = R['rows']; T = np.array([r['T'] for r in rows])
fig, ax = plt.subplots(figsize=(5.4, 4.0)); plt.subplots_adjust(left=0.16, right=0.97, top=0.9, bottom=0.14)
ax.fill_between(T, [r['lo'] * 1e19 for r in rows], [r['hi'] * 1e19 for r in rows], color=F.ORANGE, alpha=0.16, lw=0, label='90% prediction interval')
ax.plot(T, [r['corr'] * 1e19 for r in rows], color=F.BLUE, lw=2.0, label='Published correlation')
ax.plot(T, [r['pred'] * 1e19 for r in rows], '--', color=F.ORANGE, lw=1.8, label='thIRmo-ET')
m = R['measured_spectra']; ax.plot([r['T'] for r in m], [r['measured'] * 1e19 for r in m], 'o', mfc='white', mec='k', ms=6, mew=1.2, label='Published spectra', zorder=5)
ax.set_xlim(T.min() - 20, T.max() + 20); ax.set_ylim(0, 6.5)
ax.axvspan(T_VALID, ax.get_xlim()[1], color='#EDEDED', zorder=0, lw=0)
ax.text(T_VALID + 12, 0.2, 'beyond tested\ntemperatures', fontsize=8.5, color='#555555', va='bottom', ha='left', linespacing=0.95)
ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Cross section\n(10$^{-19}$ cm$^2$ molecule$^{-1}$)')
ax.set_title(f"Diisopropyl ether · {R['probe']:.2f} cm$^{{-1}}$", fontsize=12); ax.legend(fontsize=8.6, loc='lower left', handlelength=1.8)
F.save(fig, FIGS / 'Figure_S5'); print('Figure_S5 written')
