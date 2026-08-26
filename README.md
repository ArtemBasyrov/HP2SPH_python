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

RMS relative `C_l` error over the top quarter of the band, `cosmology` scenario, `lmax = 2·nside`, median of seeds 0–3.
From `benchmarks/results/` (tracked in git); [`benchmarks/README.md`](benchmarks/README.md) states how each competitor is configured.
`healpy iter=3` is an iterative least squares and a near-exact inverse of `alm2map` by construction — the other columns, like HP2SPH, are single-pass quadrature.

Polarization `C_l^EE`, the most accurate single-pass method at every nside measured:

| nside | HP2SPH | healpy pixel | healpy ring | healpy `iter=3` |
|---|---|---|---|---|
| 32 | **3.28e-5** | 5.71e-4 | 6.97e-4 | 5.30e-8 |
| 64 | **1.43e-5** | 2.19e-4 | 2.89e-4 | 1.47e-8 |
| 128 | **4.33e-6** | 8.20e-5 | 1.25e-4 | 5.04e-9 |
| 256 | **1.55e-6** | 2.50e-5 | 4.47e-5 | 2.37e-9 |

E→B leakage, median leaked `C_l^BB` from a pure `E` input, improving with resolution:

| nside | HP2SPH | healpy pixel | healpy ring |
|---|---|---|---|
| 64 | **3.64e-9** | 2.33e-6 | 4.23e-6 |
| 128 | **8.53e-10** | 4.09e-7 | 1.10e-6 |
| 256 | **2.94e-10** | 1.10e-7 | 4.05e-7 |

Intensity `C_l^TT`, which beats ring weights but not pixel weights in this band:

| nside | HP2SPH | healpy pixel | healpy ring | healpy `iter=3` |
|---|---|---|---|---|
| 256 | 5.05e-5 | 2.51e-5 | 5.39e-5 | 2.74e-9 |
| 512 | 1.72e-5 | 9.91e-6 | 2.24e-5 | 7.70e-10 |
| 1024 | 6.25e-6 | 3.39e-6 | 9.94e-6 | 3.00e-10 |
| 2048 | 2.51e-6 | 1.31e-6 | 2.55e-6 | 9.71e-11 |

The scalar advantage sits one band lower: over `l` from `3·lmax/4` to `7·lmax/8`, the ring-weights error divided by the HP2SPH error grows from 1.97 at nside 32 to 4.42 at nside 1024.

Scalar forward timings, Apple M4 Pro (10 performance cores), `MallocMediumZone=0`, best of 5:

| nside | serial | threaded | healpy ring | healpy `iter=3` |
|---|---|---|---|---|
| 256 | 0.148 s | 0.067 s | 0.003 s | 0.019 s |
| 512 | 0.711 s | 0.277 s | 0.017 s | 0.096 s |
| 1024 | 3.861 s | 1.169 s | 0.110 s | 0.611 s |

All four stages thread; at nside 1024 the FSHT gains 5.2×, the latitude nuFFT 2.5×, the ring FFTs 2.2×.
Threaded output matches serial to machine precision.
healpy is a mature C library and is ~10× faster in wall-clock; this is Python around finufft and `libfasttransforms`.

## Tests

```bash
python -m pytest              # 415 tests, ~12 s
python -m pytest -m "not ft"  # skip tests needing libfasttransforms
```

## License

MIT. See [LICENSE](LICENSE).
