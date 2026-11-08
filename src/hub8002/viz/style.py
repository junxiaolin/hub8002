"""One look for every figure (paper is printed: light surface only).

Categorical hues in fixed order (never cycled); sequential = one blue ramp; text in ink tokens.
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
GRID = '#e4e3df'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
BLUE_RAMP = ['#fcfcfb', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SEQ = LinearSegmentedColormap.from_list('hub_blue', BLUE_RAMP)
NEUTRAL = '#a8a7a2'


def setup():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'lines.linewidth': 1.6, 'lines.markersize': 4, 'legend.frameon': False,
        'pdf.fonttype': 42, 'font.family': ['DejaVu Sans'],
    })
    for name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'Source Han Sans SC'):
        from matplotlib import font_manager
        if any(name == f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            break
    plt.rcParams['axes.unicode_minus'] = False


def save(fig, path):
    fig.savefig(path, bbox_inches='tight')
    fig.savefig(str(path).replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close(fig)
