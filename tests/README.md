# HP2SPH test suite

Per-stage and end-to-end tests for the HEALPix <-> alm pipeline. The tests are
designed to **encode correct behaviour and surface failure modes**, not to be
made green by loosening tolerances — several currently fail on purpose because
the pipeline is not yet fully correct (see "Known failures" below).

## Running

From the repo root, with a Python env that has the pipeline deps (substitute its
interpreter for `python` below):

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m pytest
```

- `OMP_NUM_THREADS=1` is required or the finufft + scipy-CG step deadlocks.
- `KMP_DUPLICATE_LIB_OK=TRUE` avoids the libomp double-load abort.
- jax x64 and these env vars are also set defensively in `conftest.py`.

Skip the (slow) Julia / FastTransforms.jl tests:

```bash
... -m "not julia"
```

**FSHT backend.** The FSHT-stage tests (and the pipeline) use the in-process
`libfasttransforms` backend when it is available, falling back to the Julia
subprocess otherwise (see the FSHT backend notes in the top-level `CLAUDE.md`).
Set `FASTTRANSFORMS_LIB` to the built library's path to use the fast path:

```bash
FASTTRANSFORMS_LIB=/path/to/libfasttransforms.dylib \
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -m pytest
```

`test_ft_sphere.py` **skips** entirely if the library can't be loaded.

## Layout

| file | stage | what it checks |
|------|-------|----------------|
| `test_data_interpolation.py` | 1 | ring geometry, grid shape, HEALPix<->grid round trip (exact) |
| `test_double_fourier_sphere.py` | 2 | DFS shapes, DFS round trip (exact), ring-area weights |
| `test_nuFFT.py` | 3 | nuFFT shapes, forward/backward round trip, Voronoi weights |
| `test_FSHT.py` | 4 | `preparation`<->`convert` round trip, fourier2sph<->sph2fourier (julia) |
| `test_ft_sphere.py` | 4 | in-process `libfasttransforms` backend == Julia, bit-for-bit (skips w/o the C library) |
| `test_pipeline.py` | all | full map round trip (exact), **forward alm vs input / vs map2alm** |
| `test_paper_accuracy.py` | all | paper-style known-alm per-`l` error + convergence vs `nside`, compared to healpy |

Fixtures (`conftest.py`) parametrise over `nside in {4, 8, 16}` and provide a
random band-limited `random_alm`, the synthesised `healpix_map`, and a `relerr`
helper. `pipeline_helpers.py` wires the four stages into `forward_C` /
`forward_alm` / `backward_map` without the FITS I/O that `main.py` does.

## Status

All tests pass for `nside in {4, 8, 16}`. The forward transform agrees with the
input alm and with `hp.map2alm` to ~1e-2 below the longitude Nyquist band
(`l <= lmax-1`), improving with nside, and the map round trips to machine
precision. See `test_forward_alm_converges_with_nside` for the convergence guard.

What the tolerances encode (kept tight on purpose, not loosened to pass):

- `test_forward_alm_matches_*` assert agreement **below `l = lmax = 2*nside`**.
  The top band is the longitude Nyquist edge (m up to 2*nside has a single stored
  coefficient), so no transform on this grid can resolve it — it is excluded, not
  hidden. A convention/normalization bug would show up as an O(0.1–1) error here.
- `test_preparation_convert_are_consistent_inverses` checks the **projection**
  invariant (`convert . prep` is idempotent), the correct property for a lossy
  projection — not losslessness, which would be a false invariant.

Known limitation (not covered by the suite): at higher `nside` the map round trip
degrades. This is **not** a `maxiter` problem — the CG always converges in tens of
iterations. Up to `nside = 32` it is the scipy-CG `rtol=1e-6` floor (the relative
tolerance on the normal-equations residual does not translate to sample-space
accuracy on this ill-conditioned system; tightening `rtol` to ~1e-9 fixes it). At
`nside >= 64` there is an intrinsic floor (~2.5e-3) that neither `rtol` nor the
NUFFT precision removes — genuine numerical rank-deficiency of the square latitude
interpolation at the clustered HEALPix colatitudes. It is a scalability/conditioning
issue, separate from the sub-band forward-transform correctness the suite asserts.
