"""Physical stage of thIRmo-ET for broad absorption bands (manuscript Section 2.4, Eqs. 3a-3c).

The band shift and Gaussian broadening fitted between the three near-ambient inputs are extrapolated to the
target temperature with conserved band area. NumPy and SciPy only."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, brentq

C2 = 1.438776877          # second radiation constant, cm K
NU_E_MAX = 300.0          # cm-1, limit of the effective wavenumber in Eq. (3a)
K_W = 2.5                 # broadening scale in Eq. (3b), selected on training molecules

def _err(pred, ref, m):
    return 100 * np.sum(np.abs(pred[m] - ref[m])) / np.sum(np.maximum(ref[m], 0))

def deform(x, y, delta, w):
    """Shift by delta (cm-1) and convolve with a unit-area Gaussian of standard deviation w (cm-1)."""
    dx = x[1] - x[0]
    yb = gaussian_filter1d(y, w / dx, mode='nearest') if w > 1e-3 else y
    return np.interp(x - delta, x, yb)

def fit_deformation(x, src, ref, mask=None):
    """Shift and broadening that map src onto ref with the smallest integrated absolute difference."""
    m = np.ones_like(x, bool) if mask is None else mask
    f = lambda p: _err(deform(x, src, p[0], abs(p[1])), ref, m)
    best = None
    for d0 in (-4, -2, -1, -0.3, 0, 0.5):
        for w0 in (0.5, 2, 4, 7):
            r = minimize(f, [d0, w0], method='Nelder-Mead', options=dict(xatol=1e-3, fatol=1e-4, maxiter=2000))
            if best is None or r.fun < best.fun: best = r
    return float(best.x[0]), float(abs(best.x[1])), float(best.fun)

def cold_parameters(x, inputs, temperatures):
    """Pairwise deformation fits between the three near-ambient inputs (sorted by temperature)."""
    T = np.asarray(temperatures, float); S = np.asarray(inputs, float)
    d12, w12, e12 = fit_deformation(x, S[0], S[1]); d23, w23, e23 = fit_deformation(x, S[1], S[2]); d13, w13, e13 = fit_deformation(x, S[0], S[2])
    m = np.ones_like(x, bool)
    return dict(T=T.tolist(), d12=d12, w12=w12, d23=d23, w23=w23, d13=d13, w13=w13, e13=e13, e13_unchanged=float(_err(S[0], S[2], m)))

def n_be(T, nu): return 1.0 / np.expm1(C2 * nu / T)
def dn_dT(T, nu):
    x = C2 * nu / T; e = np.exp(x); return (x / T) * e / (e - 1) ** 2

def shift_law(cp, nu_max=NU_E_MAX):
    """Return (r2, effective wavenumber, rate ratio)."""
    T1, T2, T3 = cp['T']; r1 = cp['d12'] / (T2 - T1); r2 = cp['d23'] / (T3 - T2); Tm1, Tm2 = (T1 + T2) / 2, (T2 + T3) / 2
    same_sign = r1 * r2 > 0 and abs(r1) > 1e-4
    rho = r2 / r1 if same_sign else float('nan')
    if not same_sign or rho <= 1.0 or nu_max <= 0: return r2, 0.0, rho
    ratio = lambda v: dn_dT(Tm2, v) / dn_dT(Tm1, v)
    if ratio(nu_max) <= rho: return r2, float(nu_max), rho
    return r2, float(brentq(lambda v: ratio(v) - rho, 1e-3, nu_max)), rho

def shift(cp, T, nu_max=NU_E_MAX, k=1.0):
    r2, nu, _ = shift_law(cp, nu_max); T1, T2, T3 = cp['T']
    if nu <= 0: return k * r2 * (T - T3)
    return k * r2 * (n_be(T, nu) - n_be(T3, nu)) / dn_dT((T2 + T3) / 2, nu)

def broadening(cp, T, k_w=K_W):
    T1, T2, T3 = cp['T']
    return float(np.sqrt(max(k_w * cp['w13'] ** 2 / (T3 - T1) * (T - T3), 0.0)))

def reference_spectrum(x, warm, cp, T, nu_max=NU_E_MAX, k=1.0, k_w=K_W):
    """Eq. (3c): deformed warmest input at target temperature T (band area conserved)."""
    return deform(x, warm, shift(cp, T, nu_max, k), broadening(cp, T, k_w))
