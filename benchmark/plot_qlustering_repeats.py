"""
benchmark.plot_qlustering_repeats -- training-propagation plots for
qlustering_repeats.py's results_qlustering/repeats.pkl: per-run cost/RI/ARI
curves (10 runs overlaid), the iteration each search used (star marker), and
the paper's reported RI/ARI as a reference line/band where one exists.
"""

import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Manuscript-transcribed reference points (see the published "Qlustering
# Benchmark Matrix" artifact). None where the paper reports no value at all;
# a (lo, hi) tuple where the text gives only a qualitative range.
PAPER = {
    'overlap3d_w0.1': {'RI': 1.00, 'ARI': 1.00},
    'overlap3d_w0.3': {'RI': 0.85, 'ARI': 0.63},
    'ipr_gap7':        {'RI': None, 'ARI': None},
    'ipr_gap1':        {'RI': (0.7, 0.8), 'ARI': None},
}

_CMAP = plt.get_cmap('tab10')


def _draw_reference(ax, ref, label):
    if ref is None:
        return
    if isinstance(ref, tuple):
        ax.axhspan(ref[0], ref[1], color='black', alpha=0.08, label=f'paper {label}={ref[0]}-{ref[1]} (qualitative)')
    else:
        ax.axhline(ref, color='black', linestyle='--', linewidth=1, label=f'paper {label}={ref}')


def plot_dataset(name, data, out_path):
    run_results = data['run_results']
    cm = data['consensus_metrics']
    stab = data['stability']

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.5), sharex=False)
    ax_cost, ax_ri, ax_ari = axes

    for i, r in enumerate(run_results):
        color = _CMAP(i % 10)
        x = list(range(len(r['cost_curve'])))
        ax_cost.plot(x, r['cost_curve'], color=color, linewidth=1.3, alpha=0.85, label=f'run {i}')
        ax_ri.plot(x, r['ri_curve'], color=color, linewidth=1.3, alpha=0.85)
        ax_ari.plot(x, r['ari_curve'], color=color, linewidth=1.3, alpha=0.85)

        bi = r['best_iter']
        ax_ri.plot(bi, r['ri_curve'][bi], marker='*', color=color, markersize=10, markeredgecolor='black')
        ax_ari.plot(bi, r['ari_curve'][bi], marker='*', color=color, markersize=10, markeredgecolor='black')

    ax_cost.set_ylabel('cost')
    ax_cost.set_title('Cost per run')
    ax_cost.grid(True, alpha=0.3)
    ax_cost.legend(fontsize=7, ncol=2, loc='upper right')

    for ax, key in ((ax_ri, 'RI'), (ax_ari, 'ARI')):
        _draw_reference(ax, PAPER[name][key], key)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(key)
        ax.set_title(f'{key} per run (★ = best-found iteration used for that run)')
        ax.grid(True, alpha=0.3)
        if PAPER[name][key] is not None:
            ax.legend(fontsize=8, loc='lower right')

    ax_ari.set_xlabel('search iteration')

    def _fmt(mr):
        return f'{mr.value:.3f}' if mr.ok else 'n/a'

    fig.suptitle(
        f"{name}  |  consensus RI={_fmt(cm['rand_index'])} ARI={_fmt(cm['adjusted_rand_index'])} "
        f"CP={_fmt(cm['compactness'])} DVI={_fmt(cm['dunn_index'])} SIL={_fmt(cm['silhouette'])}  |  "
        f"stability={_fmt(stab)}",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    with open('results_qlustering/repeats.pkl', 'rb') as f:
        all_results = pickle.load(f)

    for name, data in all_results.items():
        out_path = f"figures/qlustering_search_{name}.png"
        plot_dataset(name, data, out_path)
        print(f'Saved {out_path}')
