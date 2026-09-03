# HP2SPH_python

HEALPix map ↔ spherical-harmonic coefficients (`alm`), for intensity (`I`) and spin-2 polarization (`Q`/`U` → `E`/`B`).
Implements Drake & Wright, [arXiv:1904.10514](https://arxiv.org/abs/1904.10514).

**Research code under active development.** The API is not stable and is not tuned for wall-clock speed.

## What it does

A HEALPix grid has no exact quadrature rule, so the transform is routed through a structured latitude–longitude grid in four stages:

1. **Ring FFTs** — FFT each ring onto an equiangular longitude grid.
2. **Double Fourier Sphere** — mirror across the poles to make latitude periodic.
3. **Latitude nuFFT** — ring colatitudes are non-uniform, so fit the band-limited latitude modes (finufft + CG).
4. **FSHT** — Slevinsky's `libfasttransforms` maps the result to `alm`.

The forward pass also models the polar-ring longitude alias instead of zero-padding it away.
A coarse ring measures a whole alias family, not a single mode `m`, and for spin-2 the modes near `|m| = 2` are O(1) at a pole.
That is what produces the polarization accuracy below.

Each stage is importable on its own from `hp2sph`.

## Install

conda-forge only.
The PyPI `healpy` and `finufft` wheels each vendor their own OpenMP runtime, and more than one in a process deadlocks, so they can never thread.

```bash
micromamba create -y -f environment.yml   # creates hp2sph-omp
micromamba activate hp2sph-omp
pip install -e . --no-deps
scripts/build_fasttransforms.sh --prefix "$CONDA_PREFIX"
```

`libfasttransforms` is the only dependency built from source; it has no conda-forge package.
It needs FFTW, MPFR, OpenBLAS (or Accelerate) and OpenMP — on macOS, `brew install libomp fftw mpfr gmp`.
The script links it against the environment's `llvm-openmp` and patches an upstream bug that breaks Apple Silicon builds.

Threading is on by default: `HP2SPH_OMP_THREADS` sizes the OpenMP pool (the FSHT), `HP2SPH_NUFFT_WORKERS` the Python pool (stages 1–3).
On macOS also set `MallocMediumZone=0`, worth ~1.15×; it cannot be set from inside Python.

## Usage

```python
import healpy as hp
from hp2sph import forward_alm, forward_C, backward_map

nside = 256
mp = hp.read_map("sky_map.fits", field=0)   # one intensity map; use mp[0] for an IQU stack

alm = forward_alm(mp, lmax=2 * nside)       # healpy-ordered alm
C = forward_C(mp)                           # or the raw (L+1, 2L+1) array
mp_back = backward_map(C)
```

```python
from hp2sph import forward_spin, backward_spin

Q, U = hp.read_map("sky_map.fits", field=(1, 2))
aE, aB = forward_spin(Q, U, lmax=2 * nside - 1)
Q_back, U_back = backward_spin(aE, aB, nside)
```

`backward_spin` reproduces `hp.alm2map_spin` to ~2e-13 for `lmax ≤ 2·nside − 1`.
The `l = m = 2·nside` coefficient is the longitude Nyquist and cannot be represented; `lmax > 2·nside` raises `ValueError`.

## Benchmarks

Drake & Wright's metric: absolute error of the scaled spectrum `l(l+1)C_l`, median over the band, `cosmology` scenario, median of seeds 0–3.
The band is `lmax = 2·nside` for intensity and `2·nside − 1` for polarization.
From `benchmarks/results/` (tracked in git); [`benchmarks/README.md`](benchmarks/README.md) states how each competitor is configured.
`healpy iter=3` is an iterative least squares and a near-exact inverse of `alm2map` by construction — the other columns, like HP2SPH, are single-pass quadrature.
`ducc0 adjoint` is `adjoint_synthesis` applied with no quadrature weights, and `ducc0 pseudo` is `pseudo_analysis` at `epsilon=1e-10`.
`ducc0 adjoint` is equivalent to healpy with equal weights.

**†** healpy pixel weights are exact quadrature only below `l = 1.5·nside`, so their column is summarised over **that window alone** — the regime they are designed for — while every other column covers the full band.

Polarization `C_l^EE`:

| nside | HP2SPH | healpy pixel † | healpy ring | ducc0 adjoint | healpy `iter=3` | ducc0 pseudo |
|---|---|---|---|---|---|---|
| 32 | 3.82e-7 | 2.73e-16 | 4.57e-6 | 6.03e-5 | 1.01e-7 | 7.40e-13 |
| 64 | 8.22e-8 | 2.54e-16 | 9.83e-7 | 1.34e-5 | 2.44e-8 | 1.47e-13 |
| 128 | 1.61e-8 | 2.37e-16 | 1.85e-7 | 4.04e-6 | 7.29e-9 | 4.32e-14 |
| 256 | 2.66e-9 | 2.26e-16 | 3.67e-8 | 8.42e-7 | 1.47e-9 | 5.78e-14 |
| 512 | 4.71e-10 | 4.71e-16 | 6.19e-9 | 1.85e-7 | 3.20e-10 | 9.09e-15 |
| 1024 | 7.40e-11 | 8.11e-16 | 1.40e-9 | 5.92e-8 | 1.03e-10 | 2.39e-15 |
| 2048 | 1.37e-11 | 1.22e-16 | 2.46e-10 | 9.38e-9 | 1.65e-11 | 2.09e-14 |

E→B leakage, median recovered `C_l^BB / C_l^EE` from a pure `E` input.

| nside | HP2SPH | healpy pixel † | healpy ring | ducc0 adjoint | healpy `iter=3` | ducc0 pseudo |
|---|---|---|---|---|---|---|
| 32 | 9.86e-10 | 8.79e-29 | 7.37e-8 | 5.20e-6 | 8.73e-11 | 1.91e-21 |
| 64 | 2.41e-10 | 4.69e-28 | 4.12e-8 | 9.56e-6 | 2.56e-11 | 4.63e-22 |
| 128 | 7.37e-11 | 2.23e-27 | 8.87e-9 | 3.46e-6 | 1.65e-11 | 1.08e-20 |
| 256 | 1.47e-11 | 2.97e-26 | 2.75e-9 | 8.95e-7 | 1.15e-12 | 9.03e-21 |
| 512 | 4.87e-12 | 1.76e-24 | 9.36e-10 | 7.54e-7 | 1.64e-12 | 1.84e-21 |
| 1024 | 1.12e-12 | 1.26e-23 | 2.80e-10 | 3.39e-7 | 8.20e-13 | 1.34e-21 |
| 2048 | 2.23e-13 | 3.80e-24 | 1.48e-10 | 2.72e-7 | 4.91e-13 | 3.65e-19 |

Intensity `C_l^TT`:

| nside | HP2SPH | healpy pixel † | healpy ring | ducc0 adjoint | healpy `iter=3` | ducc0 pseudo |
|---|---|---|---|---|---|---|
| 256 | 2.47e-8 | 2.97e-16 | 3.65e-8 | 4.93e-7 | 8.12e-10 | 9.74e-14 |
| 512 | 3.96e-9 | 5.31e-16 | 6.13e-9 | 1.12e-7 | 1.86e-10 | 2.53e-14 |
| 1024 | 6.40e-10 | 7.04e-16 | 1.22e-9 | 2.74e-8 | 4.74e-11 | 3.58e-14 |
| 2048 | 1.05e-10 | 2.13e-16 | 2.56e-10 | 6.90e-9 | 1.22e-11 | 2.53e-14 |

Scalar forward timings, 8 threads for every backend, min of 3 timed calls after a warm-up.

| nside | HP2SPH | healpy pixel | healpy ring | ducc0 adjoint | healpy `iter=3` | ducc0 pseudo |
|---|---|---|---|---|---|---|
| 256 | 0.069 s | 0.005 s | 0.004 s | 0.0021 s | 0.021 s | 0.037 s |
| 512 | 0.281 s | 0.024 s | 0.022 s | 0.0113 s | 0.122 s | 0.155 s |
| 1024 | 1.262 s | 0.150 s | 0.138 s | 0.072 s | 0.800 s | 0.967 s |
| 2048 | 6.257 s | 1.023 s | 0.989 s | 0.533 s | 5.921 s | 5.880 s |

## Tests

```bash
python -m pytest              # 415 tests, ~12 s
python -m pytest -m "not ft"  # skip tests needing libfasttransforms
```

## License

MIT. See [LICENSE](LICENSE).
