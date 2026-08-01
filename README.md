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
pip install -e .          # installs numpy, scipy, astropy, healpy, jax, jax_healpy, finufft
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
The OpenMP guard (`KMP_DUPLICATE_LIB_OK`) and JAX float64 are enabled automatically on import, so **no environment-variable prefix is needed**:

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

The forward models the polar-ring longitude alias explicitly (`forward_spin(alias="fold")`, the default).
A HEALPix ring of `npix` pixels measures the whole alias family of a mode, not the mode itself, and for a spin-2 field the modes near `|m| = 2` are O(1) at a pole, so the zero-padding misattributed them.
`alias="mask"` restores the previous route, which dropped those entries instead.

RMS relative `C_ℓ^EE` error over the top band `3·lmax/4 ≤ ℓ ≤ 7·lmax/8`, with `lmax = 2·nside` and a smooth band-limited `(aE, aB)` sky (`slope 1.5`, median of 3 seeds):

| nside | HP2SPH (fold) | HP2SPH (mask) | healpy ring weights | healpy `map2alm_spin` |
|---|---|---|---|---|
| 8 | 1.52e-4 | 5.52e-3 | 2.04e-3 | 8.48e-3 |
| 16 | 6.05e-5 | 1.06e-3 | 1.85e-3 | 3.85e-3 |
| 32 | 2.68e-5 | 3.28e-4 | 6.27e-4 | 1.28e-3 |
| 64 | 9.63e-6 | 1.49e-4 | 2.77e-4 | 1.30e-3 |
| 128 | 3.56e-6 | 5.55e-5 | 9.68e-5 | 5.68e-4 |

The fold is also about twice as fast as the mask, because the system is full rank again and plain CG replaces LSMR.
Reproduce with `tests/test_alias_fold.py` and `tests/test_spin_paper_accuracy.py`; the backward accuracy is pinned in `tests/test_spin_backward.py`.

The individual pipeline stages are exposed in the `src` package (`transform_healpix_to_grid`, `DFS`, `apply_nuFFT`, `FSHT`, and their inverses).

## Tests

```bash
PYTHONPATH=. python -m pytest    # full suite (231 tests)
python -m pytest -m "not ft"     # skip tests that need libfasttransforms
```

Run from the repo root with `PYTHONPATH=.`; the top-level `__init__.py` otherwise hides `src` from `python -m pytest`.
See [`tests/README.md`](tests/README.md) for the layout.
Tests that need the C library skip cleanly when it is not installed.

## Notes

- The math requires float64; this is handled for you (see `src/_bootstrap.py`).
- `backward_spin` runs only the `spin = +2` pass: `z = Q + iU` carries the whole real `(Q, U)` pair, and the `-2` coefficients are fixed by the reality of `Q` and `U`.
