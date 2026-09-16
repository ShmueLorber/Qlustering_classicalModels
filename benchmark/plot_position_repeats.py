"""
benchmark.plot_position_repeats -- training-propagation plots for
qlustering_position_repeats.py's results_qlustering/position_repeats.pkl:
per-run cost/RI/ARI curves (10 runs overlaid), the run each search stopped
at (star marker), and the consensus RI/ARI as a reference line.
"""

import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = {
    'overlap3d_w0.1': {'RI': 1.00, 'ARI': 1.00},
    'overlap3d_w0.3': {'RI': 0.85, 'ARI': 0.63},
}

_CMAP = plt.get_cmap('tab10')


def plot_dataset(name, data, out_path):
    run_results = data['run_results']
    consensus_ri, consensus_ari = data['consensus_ri'], data['consensus_ari']

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
        ax.axhline(PAPER[name][key], color='black', linestyle='--', linewidth=1, label=f"paper {key}={PAPER[name][key]}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(key)
        ax.set_title(f'{key} per run (★ = best-found iteration used for that run)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')

    ax_ari.set_xlabel('search iteration')

    mean_ri = np.mean([r['ri_curve'][r['best_iter']] for r in run_results])
    mean_ari = np.mean([r['ari_curve'][r['best_iter']] for r in run_results])
    fig.suptitle(
        f'{name}  |  mean best RI={mean_ri:.3f} ARI={mean_ari:.3f}  |  '
        f'consensus RI={consensus_ri:.3f} ARI={consensus_ari:.3f}  |  '
        f'paper RI={PAPER[name]["RI"]} ARI={PAPER[name]["ARI"]}',
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    with open('results_qlustering/position_repeats.pkl', 'rb') as f:
        all_results = pickle.load(f)

    for name, data in all_results.items():
        out_path = f"figures/qlustering_search_{name}.png"
        plot_dataset(name, data, out_path)
        print(f'Saved {out_path}')
