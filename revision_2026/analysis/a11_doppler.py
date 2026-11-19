"""Common Gaussian-width coefficient of Eq. (7) versus the molecular-mass dependence of Doppler broadening."""
from common import *
from scipy.special import voigt_profile
cfg = json.load(open(REPO / 'three_input/line_model/model_frozen/narrow_model_config.json')); print({k: (v if not isinstance(v, (list, dict)) or len(str(v)) < 300 else '...') for k, v in cfg.items()})
K = 3.5812e-7 / np.sqrt(2 * np.log(2))     # Doppler standard deviation coefficient: s = K nu sqrt(T/M)
d = float(cfg.get('gaussian_coefficient', cfg.get('gaussian_width_coefficient', 6.13592e-8)))
M = {'CO': 28.010, 'HCl': 36.461, 'HF': 20.006, 'NO': 30.006, 'CH4': 16.043}
out = dict(learned_coefficient=d, doppler_constant=K, species={})
for s, m in M.items():
    th = K / np.sqrt(m); out['species'][s] = dict(mass=m, doppler_coefficient=th, learned_over_doppler=d / th, percent=100 * (d / th - 1))
# effect of the coefficient mismatch on the line-centre cross section (Voigt peak) for representative collisional widths
def peak(sg, gl): return float(voigt_profile(0.0, sg, gl))
cases = []
for s, nu, g0, n in [('NO', 1812.64, 0.055, 0.70), ('CH4', 4300.0, 0.060, 0.70)]:
    for T in (700.0, 1000.0, 1500.0, 1800.0):
        for P in (1.0, 1.5):
            gl = g0 * P * (296 / T) ** n; sg_true = K / np.sqrt(M[s]) * nu * np.sqrt(T); sg_model = d * nu * np.sqrt(T)
            cases.append(dict(species=s, nu=nu, T=T, P=P, lorentz_hwhm=gl, doppler_sd=sg_true, model_sd=sg_model, peak_change_percent=100 * (peak(sg_model, gl) / peak(sg_true, gl) - 1)))
out['line_centre_effect'] = cases; jdump(out, 'doppler.json')
for s, v in out['species'].items(): print(f"{s:4s} M {v['mass']:7.3f}  Doppler coefficient {v['doppler_coefficient']:.4e}  learned/Doppler {v['learned_over_doppler']:.3f} ({v['percent']:+.1f}%)")
for c in cases: print(f"{c['species']:4s} {c['T']:6.0f} K {c['P']:.1f} atm  gamma_L {c['lorentz_hwhm']:.4f}  s_D {c['doppler_sd']:.4f}  model {c['model_sd']:.4f}  peak change {c['peak_change_percent']:+.2f}%")
