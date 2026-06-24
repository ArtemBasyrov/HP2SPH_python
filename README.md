# HP2SPH_python

A Python implementation of **HP2SPH** — fast, accurate conversion between
**HEALPix sky maps** and **spherical-harmonic coefficients (`alm`)**, reproducing

> K. P. Drake & G. B. Wright, *A Fast and Accurate Algorithm for Spherical
> Harmonic Analysis on HEALPix Grids with Applications to the Cosmic Microwave
> Background Radiation*, [arXiv:1904.10514](https://arxiv.org/abs/1904.10514).

The transform is routed through a structured latitude–longitude grid where fast
algorithms apply: ring FFTs → Double Fourier Sphere → a latitude non-uniform FFT
→ Slevinsky's fast spherical-harmonic transform (`libfasttransforms`).

## Installation

The Python dependencies are on PyPI; the FSHT stage additionally needs the native
**`libfasttransforms`** C library, which is **not** packaged on PyPI.

### 1. Python package

```bash
pip install -e .          # installs numpy, scipy, astropy, healpy, jax, jax_healpy, finufft
```

### 2. The `libfasttransforms` C library

This is the only non-Python dependency. Build it from source
([MikaelSlevinsky/FastTransforms](https://github.com/MikaelSlevinsky/FastTransforms));
its own dependencies are FFTW, MPFR, OpenBLAS (or Apple Accelerate) and OpenMP.

- **Linux:** `make` (install FFTW/MPFR/OpenBLAS/OpenMP via your package manager or conda).
- **macOS:** `brew install libomp fftw mpfr gmp`, then `make CC=clang FT_USE_APPLEBLAS=1`.
  On **Apple Silicon** the upstream `make.inc` hardcodes Intel Homebrew paths
  (`/usr/local/opt/...`); edit it to `/opt/homebrew/opt/...` before building, or
  it fails with `'fftw3.h' file not found`.

The library is **located automatically** at runtime — no environment variable is
required when it is installed in any of these places (searched in order):

1. `$FASTTRANSFORMS_LIB` (explicit override; full path to the library),
2. a `lib/` directory next to the package or at the repo root (drop or symlink
   the built `libfasttransforms.{dylib,so}` there for a self-contained checkout),
3. the active conda/virtualenv `lib` dir or the OS loader path,
4. a prebuilt `FastTransforms.jl` artifact under `~/.julia`, if present (just a
   precompiled binary — no Julia runtime is used).

If none load, the FSHT stage raises an `ImportError` with a build/install hint.

## Usage

Run from the repo root. The OpenMP guard (`KMP_DUPLICATE_LIB_OK`) and JAX float64
are enabled automatically on import, so **no environment-variable prefix is
needed**:

```bash
python main.py path/to/sky_map.fits            # forward transform
python main.py path/to/sky_map.fits --roundtrip --save
```

Or from Python:

```python
import healpy as hp
from main import forward, backward

mp = hp.read_map("sky_map.fits", field=(0, 1, 2))  # (I, Q, U); only I is used
alm = forward(mp)          # HEALPix map -> spherical-harmonic coefficients
mp_back = backward(alm)    # and back
```

The individual pipeline stages are exposed in the `src` package
(`transform_healpix_to_grid`, `DFS`, `apply_nuFFT`, `FSHT`, and their inverses).

## Tests

```bash
python -m pytest                # full suite
python -m pytest -m "not ft"    # skip tests that need libfasttransforms
```

See [`tests/README.md`](tests/README.md) for the layout. Tests that need the C
library skip cleanly when it is not installed.

## Notes

- The pipeline currently transforms only the intensity (**I**) component; `Q`/`U`
  are accepted but ignored.
- The math requires float64; this is handled for you (see `src/_bootstrap.py`).
