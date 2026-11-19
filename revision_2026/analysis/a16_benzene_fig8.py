"""Benzene time history of Shakfa et al. (Fig. 8 of Shakfa et al.; 961 K, 1.36 bar) recalculated with thIRmo-ET, and the
noise of the same measurement projected to the predicted wavenumber of maximum absorption at that temperature.

The published mole fraction x was obtained with the cross section of the shock-tube correlation at 673.94 cm-1.
(i) Replacing it by the predicted cross section rescales the history by sigma_corr / sigma_pred (as Eq. 9).
(ii) For the same absorbance noise, probing at the predicted maximum divides the mole-fraction noise by
     G = sigma_pred(nu_opt) / sigma_corr(673.94), the predicted sensitivity gain of Section 4.3.
The noise is the scatter of the digitized trace about a smooth fit (0.25-3.9 ms) and before the shock."""
from common import *
import broad as B, benzene as BZ
from scipy.signal import savgol_filter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import figstyle as F
from config import DIGITIZED as D; T5 = 961.0; P5 = 1.36
ex = np.loadtxt(D / 'benzene_fig8_experiment.csv', delimiter=',', skiprows=5); sim = np.loadtxt(D / 'benzene_fig8_simulation.csv', delimiter=',', skiprows=5)
t, x = ex[:, 0], ex[:, 1]; g = BZ.window(); mdl = B.main_model(); X = g['x']; sp = jload('seed_probe.json'); q, beta = sp['q'], sp['beta']
y = B.predict(mdl, g, T5); trees = B.predict(mdl, g, T5, per_tree=True); half = q * (trees.std(0) + beta * y.max())
s_corr = BZ.correlation(T5); s_pred = float(np.interp(BZ.NU_LASER, X, y)); h_pred = float(np.interp(BZ.NU_LASER, X, half))
nu_opt, s_opt = BZ.peak(X, y); h_opt = float(np.interp(nu_opt, X, half)); G = s_opt / s_corr; G_lo, G_hi = max(s_opt - h_opt, 0) / s_corr, (s_opt + h_opt) / s_corr
post = (t > 0.25) & (t < 3.9); pre = t < -0.25
smooth = np.full_like(x, np.nan); smooth[post] = savgol_filter(x[post], 41, 2)
res_post = x[post] - smooth[post]; sd_post = float(res_post.std(ddof=1)); sd_pre = float(x[pre].std(ddof=1)); base = float(x[pre].mean())
x_recalc = x * s_corr / s_pred                       # (i)
x_proj = np.where(post, smooth + (x - smooth) / G, np.nan)   # (ii) same absorbance noise at the predicted maximum
n_tot = P5 * 1e5 / (1.380649e-23 * T5) * 1e-6    # molecules cm-3
res = dict(T_K=T5, P_bar=P5, sigma_corr_673_94=s_corr, sigma_pred_673_94=s_pred, sigma_pred_interval=[max(s_pred - h_pred, 0), s_pred + h_pred], ratio_pred_over_corr=s_pred / s_corr,
           recalculated_over_published=s_corr / s_pred, nu_opt_cm1=nu_opt, sigma_pred_opt=s_opt, gain=G, gain_interval=[G_lo, G_hi],
           noise_post_sd=sd_post, noise_pre_sd=sd_pre, preshock_mean=base, mean_x_post=float(x[post].mean()), snr_post=float(x[post].mean() / sd_post),
           detection_limit_3sd_fixed=3 * sd_post, detection_limit_3sd_opt=3 * sd_post / G, detection_limit_3sd_opt_interval=[3 * sd_post / G_hi, 3 * sd_post / G_lo if G_lo > 0 else None],
           absorbance_per_mole_fraction_fixed_L1cm=s_corr * n_tot, absorbance_per_mole_fraction_opt_L1cm=s_opt * n_tot,
           note='Noise is that of the digitized plotted trace; projection assumes equal absorbance noise at both wavenumbers.')
jdump(res, 'benzene_fig8.json')
print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in res.items()})
# ------------------------------------------------------------------ Figure S4
fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9)); plt.subplots_adjust(left=0.075, right=0.985, top=0.88, bottom=0.15, wspace=0.22)
ax = axs[0]
ax.plot(t, x * 100, color=F.BLUE, lw=1.3, label='Published, correlation at 673.94 cm$^{-1}$')
ax.plot(t, x_recalc * 100, color=F.ORANGE, lw=1.3, label='Recalculated, thIRmo-ET at 673.94 cm$^{-1}$')
ax.plot(sim[:, 0], sim[:, 1] * 100, color=F.GREEN, lw=1.4, ls='--', label='Simulation (Shakfa et al.)')
ax.set_title('(a) Fixed wavenumber, 673.94 cm$^{-1}$', fontsize=12); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Benzene mole fraction (%)'); ax.legend(fontsize=9, loc='upper left', frameon=False); ax.set_xlim(-1, 4); ax.set_ylim(0, 2.2)
ax = axs[1]
snr_fix = x[post] / sd_post; snr_opt = (G * smooth[post] + (x[post] - smooth[post])) / sd_post
ax.fill_between(t[post], G_lo * smooth[post] / sd_post, G_hi * smooth[post] / sd_post, color=F.PINK, alpha=0.18, lw=0, label='90% interval of the predicted gain')
ax.plot(t[post], snr_opt, color=F.PINK, lw=1.3, label=f'Predicted optimum, {nu_opt:.1f} cm$^{{-1}}$')
ax.plot(t[post], snr_fix, color=F.BLUE, lw=1.3, label='Measured, 673.94 cm$^{-1}$')
ax.set_title(f'(b) Signal-to-noise ratio at {T5:.0f} K, same absorbance noise', fontsize=12); ax.set_xlabel('Time (ms)'); ax.set_ylabel('Absorbance / noise'); ax.legend(fontsize=9, loc='upper left', frameon=False); ax.set_xlim(0, 4); ax.set_ylim(0, None)
F.save(fig, FIGS / 'Figure_S4'); print('Figure_S4 written')
