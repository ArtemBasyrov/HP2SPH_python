# HP2SPH_python

A Python implementation of **HP2SPH** — fast, accurate conversion between
**HEALPix sky maps** and **spherical-harmonic coefficients (`alm`)**, reproducing

> K. P. Drake & G. B. Wright, *A Fast and Accurate Algorithm for Spherical
> Harmonic Analysis on HEALPix Grids with Applications to the Cosmic Microwave
> Background Radiation*, [arXiv:1904.10514](https://arxiv.org/abs/1904.10514).

The transform is routed through a structured latitude–longitude grid where fast algorithms apply.
The four stages are ring FFTs, a Double Fourier Sphere, a latitude non-uniform FFT, and Slevinsky's fast spherical-harmonic transform (`libfasttransforms`).

Both the scalar intensity (`I`) and the spin-2 polarization (`Q`/`U` → `E`/`B`) transforms are implemented.

## Installation

The Python dependencies are on PyPI.
The FSHT stage additionally needs the native **`libfasttransforms`** C library, which is **not** packaged on PyPI.

### 1. Python package

```bash
pip install -e .          # installs numpy, scipy, astropy, healpy, finufft
```

### 2. The `libfasttransforms` C library

This is the only non-Python dependency.
Build it from source ([MikaelSlevinsky/FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms)).
Its own dependencies are FFTW, MPFR, OpenBLAS (or Apple Accelerate) and OpenMP.

- **Linux:** `make` (install FFTW/MPFR/OpenBLAS/OpenMP via your package manager or conda).
- **macOS:** `brew install libomp fftw mpfr gmp`, then `make CC=clang FT_USE_APPLEBLAS=1`.
  On **Apple Silicon** the upstream `make.inc` hardcodes Intel Homebrew paths (`/usr/local/opt/...`).
  Edit it to `/opt/homebrew/opt/...` before building, or it fails with `'fftw3.h' file not found`.

The library is **located automatically** at runtime.
No environment variable is required when it is installed in any of these places, searched in order:

1. `$FASTTRANSFORMS_LIB` (explicit override; full path to the library),
2. a `lib/` directory next to the package or at the repo root (drop or symlink the built `libfasttransforms.{dylib,so}` there for a self-contained checkout),
3. the active conda/virtualenv `lib` dir or the OS loader path,
4. a prebuilt `FastTransforms.jl` artifact under `~/.julia`, if present (just a precompiled binary — no Julia runtime is used).

If none load, the FSHT stage raises an `ImportError` with a build/install hint.

## Usage

Run from the repo root.
The OpenMP guards are set automatically on import, so **no environment-variable prefix is needed** -- in particular the `OMP_NUM_THREADS=1` older notes asked for is handled for you (see [OpenMP](#openmp) below):

```bash
python main.py path/to/sky_map.fits            # forward transform
python main.py path/to/sky_map.fits --roundtrip --save
```

### Intensity (scalar)

```python
import healpy as hp
from main import forward, backward

mp = hp.read_map("sky_map.fits", field=(0, 1, 2))  # (I, Q, U); forward uses I
alm = forward(mp)          # HEALPix map -> spherical-harmonic coefficients
mp_back = backward(alm)    # and back
```

### Polarization (spin-2)

`forward_spin` takes a `(Q, U)` map and returns healpy-ordered `(aE, aB)`.
Ground truth throughout is healpy's `map2alm_spin` / `alm2map_spin` with `spin=2`.

```python
import healpy as hp
from src.spin_transform import forward_spin, backward_spin

Q, U = hp.read_map("sky_map.fits", field=(1, 2))
aE, aB = forward_spin(Q, U, lmax=2 * nside)        # the true HP2SPH route
Q_back, U_back = backward_spin(aE, aB, nside)
```

Each direction has two routes — `forward_spin(analysis=...)` and `backward_spin(synthesis=...)`:

| route | what it does | when to use it |
|---|---|---|
| `"hp2sph"` (default) | the hand-rolled DFS + latitude nuFFT, no resampling | the real method; more accurate than healpy at every nside tested |
| `"library"` | resample onto the FastTransforms equiangular grid, then its exact `spinsph_analysis` / `spinsph_synthesis` | a resampling-limited cross-check; needs `nside` well above `lmax` |

`backward_spin` on the native route is **exact**: it reproduces `hp.alm2map_spin` to ~2e-13 at every nside tested (8 to 64), for any band `lmax ≤ 2·nside - 1`.
The single `ℓ = m = lmax` coefficient at `lmax = 2·nside` is the grid's longitude Nyquist and cannot be represented; use `lmax ≤ 2·nside - 1` if it matters.
`lmax > 2·nside` raises `ValueError`.

The forward models the polar-ring longitude alias explicitly.
A HEALPix ring of `npix` pixels measures the whole alias family of a mode, not the mode itself, and for a spin-2 field the modes near `|m| = 2` are O(1) at a pole, so the zero-padding misattributed them.
An earlier fix dropped those entries instead of modelling them; it was removed once the fold proved better in every regime tested, and the "old" column below is what it measured.

RMS relative `C_ℓ^EE` error over the top quarter of the band, with `lmax = 2·nside` and a smooth band-limited `(aE, aB)` sky (`slope 1.5`, median of 4 seeds), from `benchmarks/results/accuracy_P.json`:

| nside | HP2SPH | HP2SPH (old, removed) | healpy pixel weights | healpy ring weights | healpy `map2alm_spin` |
|---|---|---|---|---|---|
| 8 | 2.31e-4 | 5.42e-3 | – | 2.24e-3 | 1.07e-2 |
| 16 | 9.08e-5 | 6.15e-3 | – | 2.53e-3 | 4.39e-3 |
| 32 | 3.37e-5 | 2.18e-3 | 5.71e-4 | 6.97e-4 | 1.66e-3 |
| 64 | 1.37e-5 | 1.85e-3 | 2.19e-4 | 2.89e-4 | 8.31e-4 |

That makes it the most accurate single-pass polarization analysis in the suite at every nside measured.
healpy's *iterative* `map2alm` still wins by about 3 orders, because it is a near-exact inverse of `alm2map` by construction rather than a quadrature.
The fold is also 1.25–2.9× faster than the route it replaced, because the system is full rank again and plain CG replaces LSMR.
Reproduce with `tests/test_alias_fold.py` and `tests/test_spin_paper_accuracy.py`; the backward accuracy is pinned in `tests/test_spin_backward.py`.

The individual pipeline stages are exposed in the `src` package (`transform_healpix_to_grid`, `DFS`, `apply_nuFFT`, `FSHT`, and their inverses).

## OpenMP

Three dependencies each vendor their own copy of the LLVM OpenMP runtime, and all three load into one process:

| library | its libomp |
|---|---|
| healpy | `site-packages/healpy/.dylibs/libomp.dylib` |
| finufft | `site-packages/finufft/.dylibs/libomp.dylib` |
| libfasttransforms | Homebrew's `/opt/homebrew/opt/libomp/lib/libomp.dylib` |

libomp keeps process-wide state, so more than one copy running a worker pool is unsupported.
The observed failures are a `OMP: Error #15` abort, a segfault inside the first FastTransforms call, and a hang inside `finufft.Plan.setpts`.
One thread per runtime avoids all three, so **single-threaded OpenMP is a correctness requirement here, not a tuning choice**.

You do not have to arrange it.
`src/_bootstrap.py` sets `OMP_NUM_THREADS=1` and `KMP_DUPLICATE_LIB_OK=TRUE` before anything links libomp, and `src/_openmp.py` pins any runtime that loaded first anyway.
`OMP_NUM_THREADS` is forced rather than defaulted, because an exported `OMP_NUM_THREADS=8` is the most common way to make the pipeline hang.

Set `HP2SPH_OMP_THREADS` to override the count.
It is refused with a `MultipleOpenMPRuntimes` error, naming the offending libraries, whenever more than one runtime is loaded -- an explanation beats a segfault.
`tests/test_openmp_guard.py` pins this behaviour.

### Running with threads

Threading needs every library to share ONE OpenMP runtime.
The PyPI wheels cannot give you that, so it takes a conda-forge environment plus a matching libfasttransforms build:

```bash
micromamba create -y -n hp2sph-omp -c conda-forge \
    python=3.11 numpy scipy astropy healpy finufft ducc0 pytest \
    llvm-openmp fftw mpfr gmp openblas

tools/build_fasttransforms.sh --prefix "$HOME/micromamba/envs/hp2sph-omp"
```

conda-forge healpy and finufft both link the env's `llvm-openmp`, and the script links libfasttransforms against the same one.
Check before opting in -- one path means you are clear:

```bash
python -c "from src import _openmp; print(_openmp.runtime_paths())"
HP2SPH_OMP_THREADS=8 python your_script.py
```

**A repo-local `lib/libfasttransforms.*` shadows the environment**, because it is searched first.
If one is present, either remove it or point `FASTTRANSFORMS_LIB` at the env build.
Getting this wrong used to segfault; it now raises with both libomp paths named.

Measured forward transform, this machine (14 cores), median of 3, band `lmax = 2*nside`:

| nside | 1 thread | 4 | 8 | 14 | speedup |
|---|---|---|---|---|---|
| 256 | 0.447 s | 0.303 s | 0.281 s | 0.274 s | 1.6x |
| 512 | 2.031 s | 1.067 s | 0.865 s | 0.840 s | 2.4x |

Only the two C stages scale: at nside 512 the latitude nuFFT goes 1.522 s -> 0.601 s and the FSHT 0.358 s -> 0.096 s.
`data_interpolation` and `DFS` are serial numpy and do not move.
Returns flatten past 8 threads.
Output is identical to the single-threaded result to machine precision.

## Tests

```bash
PYTHONPATH=. python -m pytest    # full suite (235 tests)
python -m pytest -m "not ft"     # skip tests that need libfasttransforms
```

Run from the repo root with `PYTHONPATH=.`; the top-level `__init__.py` otherwise hides `src` from `python -m pytest`.
See [`tests/README.md`](tests/README.md) for the layout.
Tests that need the C library skip cleanly when it is not installed.

## Notes

- The math requires float64. The pipeline is pure numpy, so that is the default and there is no precision flag to forget.
- `backward_spin` runs only the `spin = +2` pass: `z = Q + iU` carries the whole real `(Q, U)` pair, and the `-2` coefficients are fixed by the reality of `Q` and `U`.
