"""Seed-to-seed variability and calibrated intervals of single-wavenumber predictions (benzene laser wavenumber and
predicted optimum; ether probe wavenumbers). Ten refits of the broad-band forest with random seeds 0-9."""
from common import *
import broad as B, benzene as BZ, ethers as ET
g = BZ.window(); x = g['x']; Tb = np.array([553, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400, 1473], float); q = float(jload('intervals.json')['broad_global']['q']['None']); BETA = 0.02
Te = {'di-n-propyl ether': [604, 690, 758, 838, 1000, 1151, 1228, 1306], '1,2-dimethoxyethane': [690, 800, 816, 1000, 1184, 1226, 1249]}
def bz(mdl):
    out = []
    for T in Tb:
        y = B.predict(mdl, g, T); nu, s = BZ.peak(x, y); out.append((float(np.interp(BZ.NU_LASER, x, y)), nu, s))
    return np.array(out)
main = B.main_model(); m0 = bz(main); seeds = [bz(B.fit(B.main_blocks(), seed=sd)) for sd in range(10)]; S = np.array(seeds); corr = BZ.correlation(Tb)
rows = []
for k, T in enumerate(Tb):
    trees = B.predict(main, g, T, per_tree=True); y = B.predict(main, g, T); sd = trees.std(0); half = q * (sd + BETA * y.max()); jl = np.interp(BZ.NU_LASER, x, half); jo = np.interp(m0[k, 1], x, half)
    rows.append(dict(T=float(T), corr=float(corr[k]), laser_pred=float(m0[k, 0]), laser_ratio=float(m0[k, 0] / corr[k]), laser_ratio_seed_min=float(S[:, k, 0].min() / corr[k]), laser_ratio_seed_max=float(S[:, k, 0].max() / corr[k]),
                     laser_ratio_seed_mean=float(S[:, k, 0].mean() / corr[k]), laser_ratio_seed_sd=float(S[:, k, 0].std(ddof=1) / corr[k]), laser_interval=[float(max(m0[k, 0] - jl, 0)), float(m0[k, 0] + jl)],
                     nu_opt=float(m0[k, 1]), nu_opt_seed_min=float(S[:, k, 1].min()), nu_opt_seed_max=float(S[:, k, 1].max()), sigma_opt=float(m0[k, 2]), sigma_opt_seed_min=float(S[:, k, 2].min()), sigma_opt_seed_max=float(S[:, k, 2].max()),
                     sigma_opt_interval=[float(max(m0[k, 2] - jo, 0)), float(m0[k, 2] + jo)], gain_vs_corr=float(m0[k, 2] / corr[k]), gain_seed_min=float(S[:, k, 2].min() / corr[k]), gain_seed_max=float(S[:, k, 2].max() / corr[k]),
                     covered=bool(max(m0[k, 0] - jl, 0) <= corr[k] <= m0[k, 0] + jl)))
eth = {}
for sp, Ts in Te.items():
    ms = [B.fit(B.main_blocks(), seed=sd) for sd in range(10)] if not eth else ms
    eth[sp] = [dict(T=float(T), corr=float(ET.correlation(sp, T)), pred=ET.sigma_probe(main, sp, T), seed_min=float(min(ET.sigma_probe(m, sp, T) for m in ms)), seed_max=float(max(ET.sigma_probe(m, sp, T) for m in ms)),
                    seed_sd_percent=float(100 * np.std([ET.sigma_probe(m, sp, T) for m in ms], ddof=1) / ET.sigma_probe(main, sp, T))) for T in Ts]
jdump(dict(benzene=rows, ethers=eth, q=q, beta=BETA), 'seed_probe.json')
print(' T    ratio  seed[min,max] mean   interval/corr      covered   nu_opt [seed range]   gain  [seed range]')
for r in rows: print(f"{r['T']:5.0f} {r['laser_ratio']:5.2f}  [{r['laser_ratio_seed_min']:.2f},{r['laser_ratio_seed_max']:.2f}] {r['laser_ratio_seed_mean']:.2f}  [{r['laser_interval'][0]/r['corr']:.2f},{r['laser_interval'][1]/r['corr']:.2f}]  {str(r['covered']):5s}  {r['nu_opt']:.2f} [{r['nu_opt_seed_min']:.2f},{r['nu_opt_seed_max']:.2f}]  {r['gain_vs_corr']:.2f} [{r['gain_seed_min']:.2f},{r['gain_seed_max']:.2f}]")
for sp, v in eth.items(): print(sp, [(int(e['T']), round(e['pred'] / e['corr'], 2), round(e['seed_min'] / e['corr'], 2), round(e['seed_max'] / e['corr'], 2), round(e['seed_sd_percent'], 1)) for e in v])
