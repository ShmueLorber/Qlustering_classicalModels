# Test plan

Living document. Updated whenever a new module/step is added or a bug
teaches us something. Rule: **every step ships with its own tests before we
move to the next one; the full suite must stay green; any bug found later
gets a named regression test, not just a silent fix.**

## Testing layers

1. **Unit tests** — one module in isolation. Wherever a metric/algorithm has
   a standard external definition, test against an oracle that is NOT our
   own code (brute-force pair-counting, sklearn, a hand calculation,
   permutation search) rather than re-deriving the same formula twice.
2. **Integration/pipeline tests** — do the pieces compose (dataset ->
   algorithm -> metrics -> aggregation -> report)? Grows every time a new
   stage joins the chain.
3. **Regression tests** — anything verified against an external source of
   truth (the paper, `Internal_metrics.mlx`, a hand calculation) gets a
   named test locking that fact in, so a future refactor can't quietly
   reintroduce a fixed bug.
4. **Risk-area tests** — bugs specific to this project's structure, not
   generic correctness. See below.

## Coverage status

### Steps 1-4 -- DONE (55 tests: test_datasets.py, test_algorithms.py, test_metrics.py, test_pipeline.py)
- Dataset generators: shapes, unit-norm, reproducibility, paper-config group
  sizes, delta_IPR boundary-sweep regression.
- Algorithms: ClusteringResult contract, recovers known ground truth,
  determinism verified bitwise (dbscan/agglomerative) vs. seeded (spectral/gmm).
- Metrics: RI vs. brute force, ARI vs. sklearn, silhouette vs. sklearn
  sqeuclidean (regression for the MATLAB-alignment fix), stability vs.
  brute-force permutation search, edge cases (single cluster, all-noise,
  DBSCAN -1 handling) all explicitly `undefined` rather than forced.
- Pipeline: full dataset x algorithm matrix, QM9 no-ground-truth path,
  DBSCAN noise-heavy path, stability integration (stochastic + deterministic).

### Step 6 (runner) -- PLANNED, not yet built
Highest risk here is **silent config/mapping errors**: a result silently
computed under the wrong dataset config (e.g. QM9 logged as q=2 but actually
run with q=4; an ipr result logged as "Delta_IPR=7" but actually using the
boundaries=(6,5) config). Decision: enforce this at the type level rather
than by convention.

- [ ] `DatasetInstance` frozen dataclass: `(name, X, y_true, q, config)`
      bundling data + label + cluster-count + human-readable config name as
      ONE object that travels through the whole pipeline together, so a
      result can never be logged under a different config than the one that
      actually produced it.
- [ ] Registry test: every planned dataset instance (overlap3d w=0.1,
      overlap3d w=0.3, ipr Delta=7, ipr Delta=1, iris full, iris reduced,
      qm9 q=2, qm9 q=4) constructs without error and has the expected q/shape.
- [ ] **No-leakage guard test**: assert no baseline algorithm's `fit`/
      `fit_predict` call ever receives `y_true` as an argument -- only as a
      post-hoc comparison target passed separately to `evaluate_clustering`.
      Structurally true today (`run_spectral(X, n_clusters, seed)` etc. have
      no `y_true` parameter at all), but add an explicit test so it stays
      true as the runner wires things together and it'd be easy to
      accidentally thread labels through "for convenience."
- [ ] Seed-per-run integrity: running `seeds = list(range(10))` through the
      runner actually calls the algorithm with 10 *different* seed values
      (not the same seed 10 times due to a loop/closure bug) -- assert the
      set of seeds actually used equals `set(range(10))`.
- [ ] No-overwrite test: confirm storing run `i`'s result never clobbers run
      `i-1`'s (a classic off-by-one/dict-key bug when logging repeated runs).
- [ ] Deterministic-algorithm run: confirm the runner still executes and
      logs a deterministic algorithm (dbscan/agglomerative) exactly once per
      "seed" (so we have 10 stored rows) but tags them so Step 9 doesn't
      compute a fake std-dev as if they were independent trials.

### Step 7 (stability measurement) -- PLANNED
Mostly covered already by `metrics.stability`'s own tests. Additional:
- [ ] Cross-algorithm test: stability computed identically (same function,
      same alignment procedure) for every algorithm including Qlustering's
      own results, once those are wired in -- no separate "Qlustering
      stability" code path that could silently diverge from the classical
      baselines' path.

### Step 8-9 (comparison + statistics) -- PLANNED
- [ ] Aggregation-with-`undefined` test: mean/std/min/max over 10 runs where
      some runs have `status='undefined'` (e.g. silhouette undefined because
      a run collapsed to k=1) must exclude those runs explicitly and report
      how many were excluded -- NOT silently coerce `None` to 0 or NaN into
      the mean.
- [ ] Determinism-aware statistics: a deterministic algorithm's "10 runs"
      must be reported as a single value with an explicit
      "deterministic, not resampled" flag, not as mean+-std=X+-0.0 (which
      looks like genuine stochastic evidence of stability but isn't).
- [ ] k-means Table I values integrate as fixed reference points (no run
      distribution) without breaking the plotting/reporting code path that
      expects mean+-std for the stochastic methods.

### Step 10 (visualization) -- PLANNED
Hard to unit-test rendering itself; test the data-shaping functions that
feed the plots (e.g. "does the stability-comparison figure's input table
have exactly one row per (dataset, algorithm) pair, no duplicates/gaps")
rather than pixel output.

## Process reminder
Before starting each new step: write/extend its test file first (or
alongside), run the full suite, only then report the step done. Any
surprising result during manual smoke-testing (like the noise-note
assertion bug found in Step 5) becomes a permanent test, not just a fix.
