"""Numerical experiment on collisional-broadening mismatch.
A line-resolved band with NO-like constants (rigid-rotor P and R branches, Boltzmann intensities, Voigt lines) is
computed with different collisional widths. The effect is evaluated (a) on the integrated metric E of Eq. (4) at the
2 cm-1 response used for the band benchmark and (b) on the monochromatic cross section at a line centre and
between lines, which is what a narrow-linewidth laser samples."""
from common import *
from scipy.special import voigt_profile
from scipy.ndimage import gaussian_filter1d
C2 = 1.438776877; NU0 = 1876.0; BROT = 1.70; MASS = 30.0; G0 = 0.055; NEXP = 0.70   # cm-1, cm-1, g/mol, cm-1/atm HWHM at 296 K, temperature exponent
x = np.arange(1700.0, 2050.0, 0.002); FW = 2.0; SIG = FW / np.sqrt(8 * np.log(2))
def band(T, P, width_factor=1.0):
    J = np.arange(0, 80); pop = (2 * J + 1) * np.exp(-C2 * BROT * J * (J + 1) / T); pop /= pop.sum()
    cen = np.r_[NU0 + 2 * BROT * (J + 1), NU0 - 2 * BROT * J[1:]]; S = np.r_[pop * (J + 1) / (2 * J + 1), pop[1:] * J[1:] / (2 * J[1:] + 1)]
    gL = width_factor * G0 * P * (296.0 / T) ** NEXP; sD = 3.0416e-7 * cen * np.sqrt(T / MASS)     # Doppler standard deviation
    y = np.zeros_like(x)
    for c, s, sd in zip(cen, S, sD):
        if s < 1e-7: continue
        m = np.abs(x - c) < 25.0; y[m] += s * voigt_profile(x[m] - c, sd, gL)
    return y, cen, S
def smooth(y): return gaussian_filter1d(y, SIG / (x[1] - x[0]), mode='nearest')
def Eint(a, b): return 100 * float(np.abs(a - b).sum() / np.maximum(b, 0).sum())
out = dict(parameters=dict(nu0=NU0, B=BROT, mass=MASS, gamma0_cm1_atm=G0, n=NEXP, response_fwhm=FW), cases=[])
for T in (300.0, 700.0, 1000.0, 1500.0):
    ref, cen, S = band(T, 1.0); k = int(np.argmax(S[:80])); c0 = cen[k]; i0 = int(np.argmin(np.abs(x - c0))); iv = int(np.argmin(np.abs(x - (c0 + BROT)))); refs = smooth(ref)
    for label, P, wf in [('width -30%', 1.0, 0.7), ('width -15%', 1.0, 0.85), ('width +15%', 1.0, 1.15), ('width +30%', 1.0, 1.3), ('P = 1.5 atm', 1.5, 1.0), ('P = 1.9 atm', 1.9, 1.0), ('P = 0.7 atm', 0.7, 1.0), ('P = 1.9 atm, width -15%', 1.9, 0.85)]:
        y, _, _ = band(T, P, wf); ys = smooth(y)
        out['cases'].append(dict(T=T, case=label, P=P, width_factor=wf, E_2cm_percent=Eint(ys, refs), E_native_percent=Eint(y, ref), line_centre_change_percent=100 * (y[i0] / ref[i0] - 1), valley_change_percent=100 * (y[iv] / ref[iv] - 1),
                                 max_pointwise_change_2cm_percent=float(100 * np.max(np.abs(ys - refs) / refs.max())), area_change_percent=100 * float(np.trapezoid(y, x) / np.trapezoid(ref, x) - 1)))
jdump(out, 'pressure_experiment.json')
print(f"{'T':>6s} {'case':26s} {'E(2 cm-1)%':>11s} {'E(native)%':>11s} {'line centre %':>14s} {'between lines %':>16s} {'max pointwise (2cm) %':>22s}")
for c in out['cases']: print(f"{c['T']:6.0f} {c['case']:26s} {c['E_2cm_percent']:11.2f} {c['E_native_percent']:11.1f} {c['line_centre_change_percent']:14.1f} {c['valley_change_percent']:16.1f} {c['max_pointwise_change_2cm_percent']:22.2f}")
