"""
Liouvillian gap vs hidden-layer size m, for (k, q) = (3, 5), (10, 3), (100, 100).

Gap := smallest nonzero |Re lambda| over the spectrum of the Lindbladian L
built by QTN/qtn/core/lindblad.py::build_L (gamma_in = gamma_out = 1,
gamma_dep = 0), with the H_matrix connectivity of QTN/qtn/core/hamiltonian.py
(input-input, input-output, output-output blocks zero).

Spectral shortcut (exact for gamma_dep = 0, verified against build_L in
`verify`): the jump terms only feed |0><0|, so L is block-triangular and

    spec(L) = { -i (eps_a - conj(eps_b)) },   eps = spec(H_eff),
    H_eff   = H - (i/2) (gamma_in |0><0| + gamma_out * sum_j |out_j><out_j|)

i.e. Re lambda = Im eps_a + Im eps_b. With y = -Im eps >= 0: if any dark
mode (y = 0) exists the gap is min(y > 0), otherwise it is 2 * min(y).
This needs an l x l eigensolve instead of l^2 x l^2.

Hamiltonians
------------
* uniform (reference): h_ii = 0, h_ij = 1 on every allowed bond. Permutation
  symmetry reduces it exactly to a 3x3 problem on the uniform input / hidden /
  output modes; every other mode is dark (Im = 0), or decays at gamma/2
  (antisymmetric outputs, vacuum). Shown both unscaled (h_ij = 1, the original
  spec) and scaled by 1/sqrt(n) like the ensemble.
* nested Gaussian ensemble: allowed bonds ~ N(0, SIGMA^2), on-site energies
  ~ N(0, W^2) with W = SIGMA, whole H scaled by 1/sqrt(n), n = k + m + q, so
  the spectral width of H stays O(1) for every m. "Nested": for each seed one
  master draw is made for (K_MAX inputs, M_MAX hidden, Q_MAX outputs); the H
  for (k, m, q) is its sub-block (first k inputs, first m hidden, first q
  outputs). Growing m only adds sites -- existing couplings never change --
  and all three (k, q) configs share the same hidden-hidden draw.

Usage
-----
    python scripts/liouvillian_gap_vs_size.py verify    # checks vs build_L / H_from_vec
    python scripts/liouvillian_gap_vs_size.py compute   # resumable, one JSON per task
    python scripts/liouvillian_gap_vs_size.py analyze   # fits -> results/liouvillian_gap/summary.json
    python scripts/liouvillian_gap_vs_size.py plot      # analyze + scripts/plot_liouvillian_gap.py
"""
import os
import sys
import json
import time

N_WORKERS = int(os.environ.get('GAP_WORKERS', 4))   # each worker peaks at ~5-6 GB at m = 10000
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(max(1, 12 // N_WORKERS)))   # 4 procs x 3 threads ~ 4x throughput of 1 x 14

import numpy as np

QTN_PATH = r'C:\Users\User\PycharmProjects\QTN'
OUT_DIR = os.path.join('results', 'liouvillian_gap')
RAW_DIR = os.path.join(OUT_DIR, 'raw')

CONFIGS = [(3, 5), (10, 3), (100, 100)]          # (k, q)
M_GRID = np.unique(np.round(np.logspace(np.log10(6), 4, 30)).astype(int))
SEEDS = list(range(10))
SIGMA = 1.0
W = SIGMA
GAMMA_IN = 1.0
GAMMA_OUT = 1.0
K_MAX, M_MAX, Q_MAX = 100, 10000, 100
DARK_TOL = 1e-10
FIT_M_MIN = 300


# ---------------------------------------------------------------- Hamiltonians
def nested_network_H(seed, k, m, q):
    """(n x n) real symmetric network H (no vacuum), sub-block of the seed's master draw, unscaled."""
    assert k <= K_MAX and m <= M_MAX and q <= Q_MAX
    n = k + m + q
    H = np.zeros((n, n))
    hid = slice(k, k + m)
    out = slice(k + m, n)
    # hidden-hidden: row i of the master upper triangle comes from its own stream -> nested in m
    hh = np.zeros((m, m))
    for i in range(m - 1):
        hh[i, i + 1:] = np.random.default_rng([seed, 0, i]).standard_normal(M_MAX)[i + 1:m]
    H[hid, hid] = SIGMA * (hh + hh.T)
    for a in range(k):
        H[a, hid] = SIGMA * np.random.default_rng([seed, 1, a]).standard_normal(M_MAX)[:m]
    for b in range(q):
        H[k + m + b, hid] = SIGMA * np.random.default_rng([seed, 2, b]).standard_normal(M_MAX)[:m]
    H[hid, :k] = H[:k, hid].T
    H[hid, out] = H[out, hid].T
    d = np.concatenate([
        np.random.default_rng([seed, 4]).standard_normal(K_MAX)[:k],
        np.random.default_rng([seed, 3]).standard_normal(M_MAX)[:m],
        np.random.default_rng([seed, 5]).standard_normal(Q_MAX)[:q],
    ])
    H[np.diag_indices(n)] = W * d
    return H


def uniform_network_H(k, m, q):
    n = k + m + q
    H = np.ones((n, n)) - np.eye(n)
    H[:k, :k] = 0
    H[k + m:, k + m:] = 0
    H[:k, k + m:] = 0
    H[k + m:, :k] = 0
    return H


def H_eff_from_network(Hn, k, m, q):
    """Prepend the vacuum site and add the anti-Hermitian loss part."""
    n = k + m + q
    l = n + 1
    He = np.zeros((l, l), dtype=complex)
    He[1:, 1:] = Hn
    He[0, 0] -= 0.5j * GAMMA_IN
    idx = np.arange(1 + k + m, l)
    He[idx, idx] -= 0.5j * GAMMA_OUT
    return He


# ---------------------------------------------------------------- gap
def gap_from_eps(eps):
    y = np.clip(-np.imag(eps), 0.0, None)
    n_dark = int(np.sum(y < DARK_TOL))
    y_nz = y[y >= DARK_TOL]
    gap = float(y_nz.min()) if n_dark > 0 else float(2 * y.min())
    return gap, n_dark, float(y.min())


def gap_full_L(Hn, k, m, q, tol=1e-9):
    """Brute force on the full l^2 x l^2 Lindbladian from QTN build_L."""
    sys.path.insert(0, QTN_PATH)
    from qtn.core.lindblad import build_L
    l = k + m + q + 1
    H_net = np.zeros((l, l), dtype=complex)
    H_net[1:, 1:] = Hn
    L = build_L(k, m, q, H_net, GAMMA_IN * np.ones(k), GAMMA_OUT * np.ones(q), 0.0).toarray()
    re = -np.linalg.eigvals(L).real
    return float(re[re > tol].min())


def gap_uniform_3x3(k, m, q, scale=1.0):
    """Exact uniform-H gap from the 3x3 symmetric sector (+ gamma/2 modes)."""
    A = scale * np.array([[0, np.sqrt(k * m), 0],
                          [np.sqrt(k * m), m - 1, np.sqrt(m * q)],
                          [0, np.sqrt(m * q), 0]], dtype=complex)
    A[2, 2] -= 0.5j * GAMMA_OUT
    y = -np.linalg.eigvals(A).imag
    cands = list(y) + [GAMMA_IN / 2] + ([GAMMA_OUT / 2] if q >= 2 else [])
    # dark modes always exist here for k >= 2 or m >= 2, so the gap is min single-mode rate
    assert k >= 2 or m >= 2
    return float(min(c for c in cands if c > DARK_TOL))


# ---------------------------------------------------------------- verify
def verify():
    sys.path.insert(0, QTN_PATH)
    from qtn.core.hamiltonian import H_from_vec
    worst = 0.0
    print('H construction vs H_from_vec (H_matrix masking):')
    for k, q in CONFIGS:
        for m in [6, 13]:
            for Hn in [nested_network_H(0, k, m, q), uniform_network_H(k, m, q)]:
                n = k + m + q
                junk = np.random.default_rng(99).standard_normal((n, n)) * 10  # forbidden entries must be masked
                V = Hn.copy()
                Vmask = H_from_vec(np.ones(n * n), k, m, q)[1:, 1:].real != 0
                V[~Vmask] = junk[~Vmask]
                V = np.triu(V) + np.triu(V, 1).T
                assert np.allclose(H_from_vec(V.ravel(), k, m, q)[1:, 1:].real, Hn), (k, m, q)
    print('  ok')

    print('nestedness: H(m) is the leading block of H(m\')')
    for k, q in CONFIGS:
        A, B = nested_network_H(3, k, 20, q), nested_network_H(3, k, 57, q)
        sel = np.r_[0:k + 20, k + 57:k + 57 + q]
        assert np.array_equal(B[np.ix_(sel, sel)], A)
    print('  ok')

    print('gap: H_eff shortcut vs full build_L (dense, l^2 x l^2)')
    for k, q in [(3, 5), (10, 3)]:
        for m in [6, 10, 20]:
            n = k + m + q
            for label, Hn in [('ens s0', nested_network_H(0, k, m, q) / np.sqrt(n)),
                              ('ens s1', nested_network_H(1, k, m, q) / np.sqrt(n)),
                              ('uniform', uniform_network_H(k, m, q))]:
                g_full = gap_full_L(Hn, k, m, q)
                g_eff = gap_from_eps(np.linalg.eigvals(H_eff_from_network(Hn, k, m, q)))[0]
                rel = abs(g_full - g_eff) / g_full
                worst = max(worst, rel)
                extra = ''
                if label == 'uniform':
                    g3 = gap_uniform_3x3(k, m, q)
                    extra = f'  3x3={g3:.10f}'
                    worst = max(worst, abs(g3 - g_full) / g_full)
                print(f'  k={k:3d} m={m:3d} q={q:3d} {label:8s} full={g_full:.10f} H_eff={g_eff:.10f}{extra}')

    print('uniform 3x3 vs H_eff dense, larger m (both configs incl. k=q=100)')
    for k, q in CONFIGS:
        for m in [50, 400]:
            n = k + m + q
            for s in [1.0, 1 / np.sqrt(n)]:
                g_eff = gap_from_eps(np.linalg.eigvals(H_eff_from_network(s * uniform_network_H(k, m, q), k, m, q)))[0]
                g3 = gap_uniform_3x3(k, m, q, s)
                worst = max(worst, abs(g3 - g_eff) / g_eff)
    print(f'worst relative mismatch: {worst:.2e}')
    assert worst < 1e-6
    print('ALL CHECKS PASSED')


# ---------------------------------------------------------------- compute
def task_path(k, q, seed, m):
    return os.path.join(RAW_DIR, f'k{k}_q{q}_s{seed}_m{m}.json')


def run_task(args):
    k, q, seed, m = args
    path = task_path(k, q, seed, m)
    if os.path.exists(path):
        return path
    t0 = time.time()
    n = k + m + q
    Hn = nested_network_H(seed, k, m, q) / np.sqrt(n)
    He = H_eff_from_network(Hn, k, m, q)
    del Hn
    eps = np.linalg.eigvals(He)
    gap, n_dark, y_min = gap_from_eps(eps)
    rec = dict(k=k, q=q, seed=seed, m=m, n=n, gap=gap, n_dark=n_dark, y_min=y_min,
               seconds=time.time() - t0)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    print(f'[{time.strftime("%H:%M:%S")}] k={k} q={q} seed={seed} m={m}: gap={gap:.4e} '
          f'dark={n_dark} ({rec["seconds"]:.0f}s)', flush=True)
    return path


def compute():
    from multiprocessing import Pool
    os.makedirs(RAW_DIR, exist_ok=True)
    tasks = [(k, q, s, int(m)) for m in M_GRID for (k, q) in CONFIGS for s in SEEDS]
    todo = [t for t in tasks if not os.path.exists(task_path(*t))]
    print(f'{len(todo)} / {len(tasks)} tasks to run, {N_WORKERS} workers', flush=True)
    with Pool(N_WORKERS) as pool:
        for _ in pool.imap_unordered(run_task, todo):
            pass
    print('DONE', flush=True)


# ---------------------------------------------------------------- fits
def gap_uniform_closed_form(k, m, q):
    """Eliminate the uniform hidden mode: rate = (gamma/2) * its output weight. Parameter-free, <0.2% vs 3x3."""
    m = np.asarray(m, dtype=float)
    a2, b2, E = k * m, m * q, m - 1.0
    eps = (E + np.sqrt(E ** 2 + 4 * (a2 + b2))) / 2
    return 0.5 * GAMMA_OUT * b2 / (eps ** 2 + a2 + b2)


def model_decay(m, logA, logm0, alpha):
    """g = A / (1 + m/m0)^alpha : plateau A at small m, A m0^alpha m^-alpha at large m."""
    return logA - alpha * np.log1p(m / np.exp(logm0))


def model_peak(m, loga, beta, logC, alpha, logs):
    """Smooth minimum of a rising bottleneck branch a m^beta and a falling branch C m^-alpha:
    g = [ (a m^beta)^-s + (C m^-alpha)^-s ]^(-1/s)."""
    s = np.exp(logs)
    l1 = loga + beta * np.log(m)
    l2 = logC - alpha * np.log(m)
    return -np.logaddexp(-s * l1, -s * l2) / s


def model_power(m, logC, alpha):
    return logC - alpha * np.log(m)


FIT_MODELS = {(3, 5): ('decay', model_decay, [np.log(0.1), np.log(20), 1.4]),
              (10, 3): ('decay', model_decay, [np.log(0.005), np.log(20), 1.7]),
              (100, 100): ('peak', model_peak, [np.log(1e-6), 2.0, np.log(15.0), 0.9, np.log(2.0)])}


def _median_curve(rs, seeds):
    ms = sorted({r['m'] for r in rs})
    by = {(r['seed'], r['m']): r['gap'] for r in rs}
    med = []
    for m in ms:
        g = [by[(s, m)] for s in seeds if (s, m) in by]
        med.append(np.median(g) if g else np.nan)
    return np.array(ms, float), np.array(med)


def fit_config(rs, k, q, n_boot=300):
    """Fit the full-range model and a large-m power law to log(median gap); seed bootstrap for CIs."""
    from scipy.optimize import curve_fit
    name, f, p0 = FIT_MODELS[(k, q)]

    def fit_once(seeds):
        ms, med = _median_curve(rs, seeds)
        ok = np.isfinite(med) & (med > 0)
        p, _ = curve_fit(f, ms[ok], np.log(med[ok]), p0=p0, maxfev=20000)
        big = ok & (ms >= FIT_M_MIN)
        pp, _ = curve_fit(model_power, ms[big], np.log(med[big]), p0=[0.0, 1.0])
        return p, pp

    p, pp = fit_once(SEEDS)
    rng = np.random.default_rng(12345)
    boots, boots_pp = [], []
    for _ in range(n_boot):
        try:
            b, bpp = fit_once(list(rng.choice(SEEDS, len(SEEDS), replace=True)))
            boots.append(b); boots_pp.append(bpp)
        except RuntimeError:
            pass
    boots, boots_pp = np.array(boots), np.array(boots_pp)

    ms, med = _median_curve(rs, SEEDS)
    resid = np.log(med) - f(ms, *p)
    names = {'decay': ['A', 'm0', 'alpha'], 'peak': ['a', 'beta', 'C', 'alpha', 's']}[name]
    logged = {'decay': {0, 1}, 'peak': {0, 2, 4}}[name]

    def tr(i, v):   # amplitudes / scales are fitted in log
        return np.exp(v) if i in logged else v

    params = {}
    for i, nm in enumerate(names):
        lo, hi = np.percentile([tr(i, b[i]) for b in boots], [2.5, 97.5])
        params[nm] = {'value': float(tr(i, p[i])), 'ci95': [float(lo), float(hi)]}
    a_lo, a_hi = np.percentile(boots_pp[:, 1], [2.5, 97.5])
    return {
        'model': {'decay': 'g = A / (1 + m/m0)^alpha',
                  'peak': 'g = [ (a m^beta)^-s + (C m^-alpha)^-s ]^(-1/s)'}[name],
        'params': params,
        'rms_log10_residual': float(np.sqrt(np.mean(resid ** 2)) / np.log(10)),
        'large_m_power_law': {'m_min': FIT_M_MIN, 'alpha': float(pp[1]), 'alpha_ci95': [float(a_lo), float(a_hi)],
                              'C': float(np.exp(pp[0])), 'evt_prediction_alpha': 1 + 2 / q},
        'm_max_fitted': int(ms.max()),
        '_f': f, '_p': p, '_pp': pp,
    }


# ---------------------------------------------------------------- analyze
def analyze():
    """Aggregate raw tasks, fit, and write summary.json with every curve the figures need.
    Drawing lives in scripts/plot_liouvillian_gap.py, which reads only summary.json."""
    recs = []
    for fn in os.listdir(RAW_DIR):
        if fn.endswith('.json'):
            with open(os.path.join(RAW_DIR, fn)) as f:
                recs.append(json.load(f))

    m_dense = np.unique(np.round(np.logspace(np.log10(6), 4, 400)).astype(int))
    summary = {'definition': 'smallest nonzero |Re lambda| of L', 'sigma': SIGMA, 'W': W,
               'gamma_in': GAMMA_IN, 'gamma_out': GAMMA_OUT, 'gamma_dep': 0.0,
               'scaling': 'H / sqrt(k+m+q) for ensemble', 'configs': {}}

    for k, q in CONFIGS:
        rs = [r for r in recs if r['k'] == k and r['q'] == q]
        ms = sorted({r['m'] for r in rs})
        per_seed = {s: sorted([(r['m'], r['gap']) for r in rs if r['seed'] == s]) for s in SEEDS}
        med, lo, hi, n_at = [], [], [], []
        for m in ms:
            g = np.array([r['gap'] for r in rs if r['m'] == m])
            med.append(np.median(g)); lo.append(np.percentile(g, 25)); hi.append(np.percentile(g, 75))
            n_at.append(len(g))

        uni = np.array([gap_uniform_3x3(k, m, q) for m in m_dense])
        uni_s = np.array([gap_uniform_3x3(k, m, q, 1 / np.sqrt(k + m + q)) for m in m_dense])
        uni_cf = gap_uniform_closed_form(k, m_dense, q)
        fit = fit_config(rs, k, q)
        fit['uniform_closed_form'] = {
            'formula': 'g = (gamma_out/2) m q / (eps^2 + (k+q) m),  eps = [m-1 + sqrt((m-1)^2 + 4(k+q)m)] / 2',
            'asymptote': 'g -> gamma_out q / (2 m)',
            'max_rel_err_vs_exact': float(np.max(np.abs(uni_cf / uni - 1)))}

        summary['configs'][f'k{k}_q{q}'] = {
            'k': k, 'q': q,
            'm': ms, 'gap_median': med, 'gap_q25': lo, 'gap_q75': hi, 'n_seeds': n_at,
            'max_dark_modes': int(max([r['n_dark'] for r in rs], default=0)),
            'min_y_min': float(min([r['y_min'] for r in rs], default=np.nan)),
            'per_seed': {str(s): v for s, v in per_seed.items()},
            'fits': {key: v for key, v in fit.items() if not key.startswith('_')},
            'curves': {'m': m_dense.tolist(),
                       'ensemble_fit': np.exp(fit['_f'](m_dense, *fit['_p'])).tolist(),
                       'uniform': uni.tolist(),
                       'uniform_scaled': uni_s.tolist()},
        }
        print(f'k={k} q={q}: {len(rs)} records')
        print(json.dumps(summary['configs'][f'k{k}_q{q}']['fits'], indent=1))

    with open(os.path.join(OUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=float)


def plot():
    analyze()
    import subprocess
    subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'plot_liouvillian_gap.py')], check=True)


if __name__ == '__main__':
    {'verify': verify, 'compute': compute, 'analyze': analyze, 'plot': plot}[sys.argv[1]]()
