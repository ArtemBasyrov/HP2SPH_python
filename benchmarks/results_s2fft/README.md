# Archived benchmark results — the pre-2026-08-09 baseline

Superseded by `benchmarks/results/`. Kept so a re-baseline can be diffed against
what the documentation recorded before the project environment changed:

    python -m benchmarks.compare_results benchmarks/results_s2fft benchmarks/results

Measured 2026-08-01 in the PyPI-wheel `s2fft` env — python 3.11.15, healpy 1.19.0,
finufft 2.5.1, numpy 2.4.4, `OMP_NUM_THREADS=1` — against the jax-era pipeline,
with `libfasttransforms` built from Homebrew against Apple Accelerate.

The current environment is `hp2sph-omp` (conda-forge; see `environment.yml`).

What the 2026-08-09 re-baseline found: every ACCURACY number reproduced exactly.
Speed moved 1.09x on the intensity forward, traced to conda-forge finufft 2.5.0
against the wheel's 2.5.1. Records here predate the `_computed_utc` stamp.
