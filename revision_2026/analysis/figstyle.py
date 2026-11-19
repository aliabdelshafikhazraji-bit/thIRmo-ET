"""Common figure style: serif Garamond typography and Okabe-Ito colours, matching the original submission figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from pathlib import Path
import os
FONT_DIR = Path(os.environ.get('THIRMO_FONTS', Path(__file__).resolve().parent / 'fonts'))
for f in FONT_DIR.rglob('*.ttf'):
    try: fm.fontManager.addfont(str(f))
    except Exception: pass
HAVE = any('EB Garamond' == f.name for f in fm.fontManager.ttflist)
FAMILY = 'EB Garamond' if HAVE else 'TeX Gyre Pagella'
BLUE, GREEN, ORANGE, PINK, GREY, BLACK = '#0072B2', '#009E73', '#D55E00', '#CC79A7', '#7F7F7F', '#000000'
CYCLE = [BLUE, GREEN, ORANGE, PINK, '#56B4E9', '#E69F00']
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': [FAMILY, 'TeX Gyre Pagella', 'DejaVu Serif'], 'mathtext.fontset': 'custom', 'mathtext.rm': FAMILY, 'mathtext.it': FAMILY + ':italic', 'mathtext.bf': FAMILY + ':bold', 'mathtext.cal': FAMILY + ':italic', 'mathtext.sf': FAMILY, 'mathtext.tt': FAMILY, 'mathtext.fallback': 'stix',
    'font.size': 11, 'axes.titlesize': 12.5, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9, 'figure.titlesize': 13,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 1.0, 'axes.grid': True, 'grid.alpha': 0.18, 'grid.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.width': 1.0, 'ytick.major.width': 1.0, 'xtick.major.size': 4, 'ytick.major.size': 4, 'lines.linewidth': 1.5, 'legend.frameon': False, 'axes.prop_cycle': matplotlib.cycler(color=CYCLE),
    'savefig.dpi': 400, 'figure.dpi': 100, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
WN = r'Wavenumber (cm$^{-1}$)'
def save(fig, path, pad=0.08):
    """Write PNG (400 dpi) and vector PDF with a uniform margin so that nothing is clipped when placed in Word."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=400, bbox_inches='tight', pad_inches=pad); fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=pad); plt.close(fig)
