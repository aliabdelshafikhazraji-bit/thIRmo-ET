"""Benzene nu4 band (610-740 cm-1): prediction from the three PNNL spectra and comparison with the published
shock-tube laser correlation at 673.94 cm-1 (Shakfa et al., Proc. Combust. Inst. 38 (2021) 1787-1796)."""
from common import *
import broad as B, physics as ph
NU_LASER = 673.94; T_RANGE = (553.0, 1473.0)
def correlation(T):
    """Published correlation, cm2/molecule, valid 553-1473 K, 1.17-2.48 bar."""
    T = np.asarray(T, float); return 1.71e-16 * np.exp(-T / 109.34) + 1.02e-18 * np.exp(-T / 558.43) + 4.99e-20
def window():
    d = np.load(CACHE / 'benzene_inputs.npz')
    g = dict(x=d['x'], inputs=d['inputs'], Tin=d['Tin'], warm=d['inputs'][-1], area=float(d['area']), anchor=d['anchor'], species='benzene', window='benzene_610_740')
    g['cp'] = {k: v for k, v in json.load(open(CACHE / 'benzene_cold_params.json')).items() if k != 'structure'}
    return g
def peak(x, y, lo=655.0, hi=685.0):
    """Sub-grid maximum (parabolic refinement) of y within [lo, hi]."""
    m = (x >= lo) & (x <= hi); i = np.flatnonzero(m)[np.argmax(y[m])]
    if 0 < i < len(x) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]; den = a - 2 * b + c; off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(x[i] + off * (x[1] - x[0])), float(b - 0.25 * (a - c) * off)
    return float(x[i]), float(y[i])
if __name__ == '__main__':
    g = window(); mdl = B.main_model(); x = g['x']; cp = g['cp']
    print('shift law (r2, nu_e, rho):', ph.shift_law(cp))
    Ts = np.arange(553, 1474, 10.0); Ts = np.unique(np.r_[Ts, 1473.0, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]); out = []
    for T in Ts:
        y = B.predict(mdl, g, T); yp = B.predict(None, g, T); s = float(np.interp(NU_LASER, x, y)); nu_opt, s_opt = peak(x, y)
        out.append(dict(T=float(T), sigma_laser_pred=s, sigma_laser_phys=float(np.interp(NU_LASER, x, yp)), sigma_corr=float(correlation(T)), ratio=s / float(correlation(T)), nu_opt=nu_opt, sigma_opt=s_opt,
                        gain_pred=s_opt / s, gain_vs_corr=s_opt / float(correlation(T)), area_ratio=float(np.trapezoid(y, x) / np.trapezoid(g['warm'], x)), shift=float(ph.shift(cp, T)), w=float(ph.broadening(cp, T))))
    jdump(dict(rows=out, nu_laser=NU_LASER, cold_params=cp, shift_law=list(ph.shift_law(cp))), 'benzene.json')
    for r in out:
        if r['T'] in (553, 603, 653, 703, 803, 903, 1003, 1103, 1203, 1303, 1403, 1473) or r['T'] % 100 == 0:
            print(f"T {r['T']:6.0f}  pred {r['sigma_laser_pred']:.3e}  phys {r['sigma_laser_phys']:.3e}  corr {r['sigma_corr']:.3e}  ratio {r['ratio']:.2f}  nu_opt {r['nu_opt']:.2f}  gain_pred {r['gain_pred']:.3f}  gain_vs_corr {r['gain_vs_corr']:.2f}  area {r['area_ratio']:.2f}  shift {r['shift']:.2f} w {r['w']:.2f}")
