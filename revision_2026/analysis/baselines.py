"""Comparison methods that use only the three supplied spectra (manuscript Section 2.5)."""
import numpy as np
C2 = 1.438776877; KB = 1.380649e-23; TREF = 300.0

def unchanged(g, T): return g['warm'].copy()

def linear(g, T):
    """Least-squares line in temperature at each wavenumber, extended from the warmest input."""
    Tin = np.asarray(g['Tin'], float); A = np.column_stack([np.ones(3), Tin - Tin.mean()])
    slope = np.linalg.lstsq(A, g['inputs'], rcond=None)[0][1]
    return np.maximum(g['warm'] + slope * (T - Tin[-1]), 0)

def exponential(g, T, floor=1e-5, limit=20.0):
    """ln(sigma) fitted against 1/T at each wavenumber; the extrapolated ratio multiplies the warmest input."""
    Tin = np.asarray(g['Tin'], float); f = floor * g['warm'].max(); L = np.log(np.maximum(g['inputs'], f))
    u = 1 / Tin; A = np.column_stack([np.ones(3), u - u.mean()]); b = np.linalg.lstsq(A, L, rcond=None)[0][1]
    return np.maximum(g['warm'], 0) * np.exp(np.clip(b * (1 / T - 1 / Tin[-1]), -limit, limit))

def wakatsuki_fit(x, spectra, temperatures, floor=1e-5, bound_exponent=True):
    """Temperature equation of Wakatsuki et al. (2005) at each wavenumber:
    kappa(nu,T) = nu [1 - exp(-c2 nu/T)] exp[a0 + a1 (Tref/T - 1) + a2 ln(T/Tref)], kappa = sigma/(kB T),
    i.e. kappa = S0 T^(-n) exp(-c2 eps/T) [1 - exp(-c2 nu/T)] with n = -a2. Three coefficients per wavenumber are fitted by
    least squares on ln(kappa). bound_exponent=True applies the constraint 0 <= n <= 2, which is imposed here (Wakatsuki et al. state no bound);
    False reproduces the unconstrained fit of the original submission."""
    T = np.asarray(temperatures, float); S = np.maximum(np.asarray(spectra, float), floor * np.max(spectra))
    A = np.column_stack([np.ones(len(T)), TREF / T - 1, np.log(T / TREF)])
    logs = np.log(S * 1e-4 / (KB * T[:, None]) / (x[None, :] * (-np.expm1(-C2 * x[None, :] / T[:, None]))))
    coef = np.linalg.lstsq(A, logs, rcond=None)[0]
    if bound_exponent:
        a2 = np.clip(coef[2], -2.0, 0.0); sub = np.linalg.lstsq(A[:, :2], logs - np.outer(A[:, 2], a2), rcond=None)[0]; coef = np.vstack([sub, a2])
    return coef, float(np.linalg.cond(A))

def wakatsuki_predict(x, coef, T):
    at = np.array([1.0, TREF / T - 1, np.log(T / TREF)])
    return np.exp(np.clip(at @ coef + np.log(x) + np.log(-np.expm1(-C2 * x / T)) + np.log(KB * T * 1e4), -700, 700))
