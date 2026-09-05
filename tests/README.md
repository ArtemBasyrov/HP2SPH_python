# HP2SPH test suite

Per-stage and end-to-end tests for the HEALPix <-> alm pipeline, scalar and spin-2.
The tests encode correct behaviour and surface failure modes.
They are not to be made green by loosening tolerances.
The whole suite passes: 408 tests in ~14 s, of which 117 need the C library.

## Running

From the repo root, with a Python env that has the pipeline deps (substitute its interpreter for `python` below):

```bash
python -m pytest              # 408 tests
python -m pytest -m "not ft"  # 291 tests; skips everything needing libfasttransforms
```

`pytest.ini` sets `testpaths = tests`, so the bare command finds them from the repo root.
`ft` is the only marker.

**The OpenMP environment is set for you, and setting it yourself breaks things.**
`hp2sph/_bootstrap.py` forces `OMP_NUM_THREADS=1` at import, because libomp reads its thread count when the image loads.
`conftest.py` repeats that, because it imports healpy before any `hp2sph` code runs; it is duplicated rather than imported so that `-m "not ft"` still works without the C library.
Do not put an `OMP_NUM_THREADS=...` prefix on the command.

On a stack where several dependencies each vendor their own libomp, more than one of them threaded makes the process crash or hang.
`hp2sph/_openmp.py` records the failure modes and `tests/test_openmp_guard.py` pins the guards against them in subprocesses.
Two behaviours that guard pins are worth knowing before you debug a threading problem:
the DEFAULT thread count degrades silently to 1 when more than one runtime is loaded,
while an EXPLICIT `HP2SPH_OMP_THREADS` raises `MultipleOpenMPRuntimes` instead, because someone asked for it.

The tests run single-threaded.
`conftest.py` defaults `HP2SPH_OMP_THREADS` to `1`, where the package itself defaults to `auto`.
Set `HP2SPH_OMP_THREADS=auto` to run the suite the way the library runs.

**FSHT backend.**
The FSHT stage runs in-process through the `libfasttransforms` C library.
`hp2sph/ft_sphere.py` locates it, most specific first: `$FASTTRANSFORMS_LIB`, a `lib/` dir in the checkout, the active conda/virtualenv `lib`, the OS loader path, a prebuilt `FastTransforms.jl` artifact under `~/.julia`, then the bare name.
See the top-level `README.md` for how to build and install it.
Tests that need it carry `@pytest.mark.ft`, and the end-to-end ones `importorskip` it so they skip cleanly when it cannot be loaded.

## Layout

Scalar, per stage and end to end:

| file | stage | what it checks |
|------|-------|----------------|
| `test_data_interpolation.py` | 1 | ring colatitudes and their symmetry, ring pixel counts, grid shape, HEALPix<->grid round trip (exact) |
| `test_double_fourier_sphere.py` | 2 | DFS shapes, DFS round trip (exact, `belt_split=False`), the polynomial pole fill beating the linear one, ring-area weights partitioning the sphere |
| `test_nuFFT.py` | 3 | shapes; the square round trip is exact and the default one is a bounded projection; Voronoi weights (sum, mirror symmetry, a WHOLE cell at the pole, both orientations, non-monotonic input refused) |
| `test_FSHT.py` | 4 | `convert . preparation` is idempotent (projection, not losslessness); `from_healpy_alm` <-> `to_healpy_alm`; the conjugate column; fourier2sph<->sph2fourier (`ft`) |
| `test_ft_sphere.py` | 4 | the C backend: round trip, the overwrite path bit-identical to the allocating one, input left untouched, non-contiguous input |
| `test_pipeline.py` | all | forward alm vs input alm and vs `map2alm` below Nyquist; map round trip in both bands; backward vs `alm2map`; the half-domain route bit-identical |
| `test_paper_accuracy.py` | all | paper-style known-alm per-`l` error and convergence with `nside`, against healpy |
| `test_conditioning.py` | 3 | the latitude Vandermonde condition number per band, and the dense SVD solving the square band where CG floors |

Spin-2 (polarization):

| file | stage | what it checks |
|------|-------|----------------|
| `test_spin_ft_sphere.py` | 4 | the spin backend round trip, the plan cache, and `spin=0` reproducing the scalar routine (`ft`) |
| `test_spin_plumbing.py` | 1, 3 | complex `Q+iU` carried through interp + nuFFT with the real `I` path unchanged; `sample_mask` inert when all-true; LSMR agreeing with CG; the answer insensitive to the CG tolerance |
| `test_spin_dfs.py` | 2 | spin mirror parity `(-1)^(m+s)`, the glide reflection being a `phi` SHIFT rather than a flip, the south-pole row, the mask matching scalar for even spin |
| `test_spin_FSHT.py` | 4 | the `F`-array decode, the E/B algebra, the conjugate readout standing in for the `s=-2` pass, and the decode validated against healpy through the library route |
| `test_spin_paper_accuracy.py` | all | `forward_spin` vs healpy `map2alm_spin`; single-harmonic gains; pure E stays E; the single `s=+2` pass reproducing the two-pass forward |
| `test_spin_backward.py` | all | `backward_spin` vs healpy `alm2map_spin` (**exact**); the signed-`m` phase; polar rings aliased rather than truncated; the band-edge corner as the only inexact mode |

The solver, the alias fold, and the machinery that exists only for speed:

| file | subject | what it checks |
|------|---------|----------------|
| `test_cg.py` | `hp2sph/cg.py` | the iteration against SciPy and against closed-form answers on small dense systems; `rho` really being the data residual; and each stopping rule (`rtol`, `maxiter`, `level`, `eta`) firing in the documented order with the documented `info` |
| `test_stopping_rule.py` | 3 | the same rules on the real spin latitude operator: the stagnation stop cuts the iteration count without moving the spectrum or the E->B leakage, is inert on the well-posed scalar solve, and `nyquist_discrepancy` predicts the floor it stops at |
| `test_alias_fold.py` | 1, 3 | the alias model against a measured ring spectrum; the fold operator being a true adjoint; the `l*theta` envelope bounding the true profile; the fold removing a single harmonic's leakage and beating ring weights at the band edge |
| `test_half_domain.py` | 2, 3 | the mirror plan and the sparse fold plan reproduce the full-domain solve; equality against a dense reference, not a tolerance; and the guards that refuse a wrong-height array instead of solving the wrong problem |
| `test_threaded_stages.py` | 1, 2 | the row-block thread split is the same arithmetic in the same order (equality); the `_shifted_into` Nyquist widening; and the worker-count policy (`usable_cores`, the env override, small batches not split) |
| `test_openmp_guard.py` | env | the OpenMP guards, in subprocesses, because the environment has to be wrong before import |

Fixtures (`conftest.py`) parametrise over `nside in {4, 8, 16}` and provide a seeded `rng`, `lmax = 2*nside`, a random band-limited `random_alm` (real-map symmetry built in), the synthesised `healpix_map`, an `iqu_map`, and a `relerr` helper.
The pipeline composition itself lives in `hp2sph/pipeline.py`; `pipeline_helpers.py` re-exports it and adds the one test-only helper, `calibrate_scale`.

## What the numbers are

Measured by the snippet below on `hp2sph-omp`, seed 20260620, `lmax = 2*nside`, default `eps`, single-threaded.
These are one seed, not medians; they are here to say what the assertions leave room for, not to be cited.

| nside | sub-band alm vs input | map round trip, default band | map round trip, square band |
|---|---|---|---|
| 4 | 1.23e-2 | 2.51e-2 | 1.35e-13 |
| 8 | 7.79e-3 | 2.53e-2 | 5.25e-13 |
| 16 | 4.46e-3 | 9.35e-3 | 9.39e-13 |

**Only the square band round trips to machine precision.**
It fits one latitude mode per DFS sample, so it interpolates exactly.
The default compact band is a projection: it drops the above-band polar content, so its round trip is accurate to a few percent at these nside and shrinks with resolution.
Do not read "round trips exactly" as a property of the default path.

The sub-band forward error falls with nside, which is what `test_forward_alm_converges_with_nside` guards.
A convention bug would leave a constant floor instead.

## What the tolerances encode

Kept tight on purpose, not loosened to pass.

- `test_forward_alm_matches_input` and `..._matches_healpy` assert agreement **below `l = lmax = 2*nside`**, at `< 3.5e-2`.
  The top band is the grid's longitude Nyquist edge; `_sub_band` excludes it, and the exclusion is stated rather than hidden.
  A convention or normalization bug would show up as an O(0.1-1) error below the edge, far outside that bound.
- `test_forward_backward_map_roundtrip_exact_mode` asserts `< 1e-5` where the measured value is ~1e-13.
  The gap is deliberate: the test is asserting "interpolates", not pinning a digit count.
- `test_preparation_convert_are_consistent_inverses` checks the **projection** invariant, that `convert . prep` is idempotent.
  That is the correct property for a lossy projection.
  Losslessness would be a false invariant.
- The half-domain, threading and sparse-plan tests assert **equality**, not closeness.
  All three are restructurings that must not move a digit, so a tolerance there would hide exactly the bug they exist to catch.

## What the suite does not cover

**Resolution.**
Everything here runs at `nside in {4, 8, 16}`, below the regime the method is claimed for.
The high-`l` advantage over healpy needs `nside >= 512` to become large, and no test measures it.
That is the benchmark suite's job (`benchmarks/`), and its results are the ones to quote.

**The band trade, measured only at three points.**
`test_conditioning.py` asserts the compact band `|k| <= 2*nside` has condition number `< 2.0` at every `nside` in the fixture, and that the square band `|k| <= 4*nside` grows monotonically past `1e8` by `nside 64`.
The `1/eps` wall at `nside 128` is documented in that file's module docstring but is not asserted anywhere.

**The belt Nyquist split has no positive test.**
`DFS(belt_split=...)` defaults to on, and the end-to-end tests do run through it.
But every test that NAMES the flag passes `belt_split=False` to scope it out, because with the split on `DFS` is deliberately not invertible on the two `|m| = 2*nside` columns.
Nothing in `tests/` asserts that the split recovers the `l = m = lmax` coefficient it exists to recover, and the fixture skies carry little power there, so a regression would move the assertions above by very little.
That gain is verified by single-harmonic probes outside the suite.
