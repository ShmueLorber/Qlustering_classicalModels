"""
Figures for the Liouvillian-gap-vs-size study. Reads only
results/liouvillian_gap/summary.json (written by
`python scripts/liouvillian_gap_vs_size.py analyze`), so restyling takes ~1 s
and never touches the eigensolves.

Edit the SETTINGS block below and rerun:
    python scripts/plot_liouvillian_gap.py

PDF and SVG are written with real (editable) text, so labels, fonts and
colors can also be changed afterwards in Inkscape / Illustrator.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================ SETTINGS
SUMMARY = os.path.join('results', 'liouvillian_gap', 'summary.json')
OUT_DIR = os.path.join('results', 'liouvillian_gap')
FORMATS = ['png', 'pdf', 'svg']
DPI = 200

CONFIGS = ['k3_q5', 'k10_q3', 'k100_q100']       # which panels to draw
COMBINED = False                                 # also write one 1x3 figure with all configs
FIGSIZE = (6.4, 4.6)                             # per-config figure (inches)
FIGSIZE_COMBINED = (15, 4.6)

SHOW = {                                         # toggle layers
    'seeds': True,                               # thin line per seed
    'iqr_band': True,                            # 25-75 % band across seeds
    'median': True,                              # ensemble median markers
    'fit': True,                                 # fitted equation curve
    'uniform': True,                             # uniform h_ij = 1, exact
    'uniform_scaled': True,                      # uniform scaled by 1/sqrt(n)
    'fit_in_legend': True,                       # equation + parameters in legend
    'title': True,
}

XLIM = None                                      # e.g. (5, 1.2e4); None = auto
YLIM = None                                      # e.g. (1e-7, 0.2)
XSCALE, YSCALE = 'log', 'log'
LEGEND_LOC = 'best'
LEGEND_FONTSIZE = 8
FONT_SIZE = 10

COLORS = {
    'series': '#2a78d6',                         # ensemble (seeds, band, median, fit)
    'uniform': '#0b0b0b',
    'uniform_scaled': '#8a8984',
    'ink': '#0b0b0b', 'ink2': '#52514e', 'grid': '#e4e3df',
}
STYLE = {
    'seed_lw': 0.8, 'seed_alpha': 0.25,
    'band_alpha': 0.15,
    'median_ms': 4.5,
    'fit_lw': 1.8,
    'uniform_lw': 1.5, 'uniform_ls': '--',
    'uniform_scaled_lw': 1.5, 'uniform_scaled_ls': ':',
}

XLABEL = 'hidden-layer size $m$'
YLABEL = 'Liouvillian gap  (smallest nonzero $|\\mathrm{Re}\\,\\lambda|$)'
LABELS = {
    'median': 'Gaussian ensemble, median (IQR band; thin = single seeds)',
    'uniform': 'uniform $h_{ij}=1$: exact; closed form $\\to \\gamma q/(2m)$',
    'uniform_scaled': 'uniform, scaled $1/\\sqrt{n}$',
}
# ================================================================


def fit_label(fits):
    P = fits['params']
    if 'beta' in P:
        return (f"fit: $[(a\\,m^{{\\beta}})^{{-s}}+(C\\,m^{{-\\alpha}})^{{-s}}]^{{-1/s}}$\n"
                f"   $a$={P['a']['value']:.2g}, $\\beta$={P['beta']['value']:.2f}, "
                f"$C$={P['C']['value']:.3g}, $\\alpha$={P['alpha']['value']:.2f}, $s$={P['s']['value']:.2g}")
    return (f"fit: $A/(1+m/m_0)^{{\\alpha}}$\n"
            f"   $A$={P['A']['value']:.3g}, $m_0$={P['m0']['value']:.1f}, $\\alpha$={P['alpha']['value']:.2f}")


def draw(ax, c, meta):
    C, S = COLORS, STYLE
    cv = c['curves']
    if SHOW['seeds']:
        for pts in c['per_seed'].values():
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, color=C['series'], lw=S['seed_lw'], alpha=S['seed_alpha'], zorder=1)
    if SHOW['iqr_band']:
        ax.fill_between(c['m'], c['gap_q25'], c['gap_q75'], color=C['series'], alpha=S['band_alpha'], lw=0, zorder=1)
    if SHOW['median']:
        ax.plot(c['m'], c['gap_median'], color=C['series'], lw=0, marker='o', ms=S['median_ms'], zorder=3,
                label=LABELS['median'])
    if SHOW['fit']:
        ax.plot(cv['m'], cv['ensemble_fit'], color=C['series'], lw=S['fit_lw'], zorder=2,
                label=fit_label(c['fits']) if SHOW['fit_in_legend'] else None)
    if SHOW['uniform']:
        ax.plot(cv['m'], cv['uniform'], color=C['uniform'], lw=S['uniform_lw'], ls=S['uniform_ls'], zorder=2,
                label=LABELS['uniform'])
    if SHOW['uniform_scaled']:
        ax.plot(cv['m'], cv['uniform_scaled'], color=C['uniform_scaled'], lw=S['uniform_scaled_lw'],
                ls=S['uniform_scaled_ls'], zorder=2, label=LABELS['uniform_scaled'])

    ax.set_xscale(XSCALE); ax.set_yscale(YSCALE)
    if XLIM: ax.set_xlim(*XLIM)
    if YLIM: ax.set_ylim(*YLIM)
    ax.set_xlabel(XLABEL); ax.set_ylabel(YLABEL)
    if SHOW['title']:
        ax.set_title(f"k = {c['k']} inputs, q = {c['q']} outputs   "
                     f"($\\gamma_{{in}}={meta['gamma_in']:g}$, $\\gamma_{{out}}={meta['gamma_out']:g}$, "
                     f"$\\gamma_{{dep}}={meta['gamma_dep']:g}$)", fontsize=FONT_SIZE)
    ax.grid(True, which='major', color=C['grid'], lw=0.6)
    ax.grid(True, which='minor', color=C['grid'], lw=0.3, alpha=0.6)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE, loc=LEGEND_LOC)


def save(fig, stem):
    for ext in FORMATS:
        path = f'{stem}.{ext}'
        try:
            fig.savefig(path, dpi=DPI)
        except PermissionError:   # file open in a viewer on Windows
            path = f'{stem}_new.{ext}'
            fig.savefig(path, dpi=DPI)
            print(f'  WARNING: {stem}.{ext} is locked')
        print('wrote', path)


def main():
    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.edgecolor': COLORS['uniform_scaled'], 'axes.labelcolor': COLORS['ink'],
        'xtick.color': COLORS['ink2'], 'ytick.color': COLORS['ink2'], 'text.color': COLORS['ink'],
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # embed TrueType -> text stays editable in PDF
        'svg.fonttype': 'none',                  # SVG text as <text>, not paths
    })
    with open(SUMMARY) as f:
        summary = json.load(f)

    for key in CONFIGS:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        draw(ax, summary['configs'][key], summary)
        fig.tight_layout()
        save(fig, os.path.join(OUT_DIR, f'liouvillian_gap_{key}'))
        plt.close(fig)

    if COMBINED:
        fig, axes = plt.subplots(1, len(CONFIGS), figsize=FIGSIZE_COMBINED)
        for ax, key in zip(np.atleast_1d(axes), CONFIGS):
            draw(ax, summary['configs'][key], summary)
        fig.tight_layout()
        save(fig, os.path.join(OUT_DIR, 'liouvillian_gap_all'))
        plt.close(fig)


if __name__ == '__main__':
    main()
