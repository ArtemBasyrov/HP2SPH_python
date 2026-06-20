# HP2SPH test suite

Per-stage and end-to-end tests for the HEALPix <-> alm pipeline. The tests are
designed to **encode correct behaviour and surface failure modes**, not to be
made green by loosening tolerances — several currently fail on purpose because
the pipeline is not yet fully correct (see "Known failures" below).

## Running

From the repo root, with the `s2fft` micromamba env:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
  /Users/basyrov/micromamba/envs/s2fft/bin/python3 -m pytest
```

- `OMP_NUM_THREADS=1` is required or the finufft + scipy-CG step deadlocks.
- `KMP_DUPLICATE_LIB_OK=TRUE` avoids the libomp double-load abort.
- jax x64 and these env vars are also set defensively in `conftest.py`.

Skip the (slow) Julia / FastTransforms.jl tests:

```bash
... -m "not julia"
```

## Layout

| file | stage | what it checks |
|------|-------|----------------|
| `test_data_interpolation.py` | 1 | ring geometry, grid shape, HEALPix<->grid round trip (exact) |
| `test_double_fourier_sphere.py` | 2 | DFS shapes, DFS round trip (exact), ring-area weights |
| `test_nuFFT.py` | 3 | nuFFT shapes, forward/backward round trip, Voronoi weights |
| `test_FSHT.py` | 4 | `preparation`<->`convert` round trip, fourier2sph<->sph2fourier (julia) |
| `test_pipeline.py` | all | full map round trip (exact), **forward alm vs input / vs map2alm** |

Fixtures (`conftest.py`) parametrise over `nside in {4, 8, 16}` and provide a
random band-limited `random_alm`, the synthesised `healpix_map`, and a `relerr`
helper. `pipeline_helpers.py` wires the four stages into `forward_C` /
`forward_alm` / `backward_map` without the FITS I/O that `main.py` does.

## Known failures (real defects, not test bugs)

- `test_forward_alm_matches_*` (~0.37, constant with nside): the
  FastTransforms<->healpy **convention conversion** (monopole leakage + m>0
  longitude phase). This is the next fix; it is *not* a quadrature error.
- `test_preparation_convert_roundtrip` / `test_fsht_inverse_roundtrip` (~1e-3):
  `preparation` deliberately zeros odd-m at the Nyquist latitude band.

The *self-consistency* tests (all the round trips) pass — the pipeline is a
correct invertible operator; the open issue is absolute alm correctness.
