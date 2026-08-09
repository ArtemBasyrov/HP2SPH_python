# Benchmarks

Three benchmarks comparing this repository's HP2SPH implementation against **healpy** and **ducc0**.

Speed, forward accuracy against known coefficients, and round-trip fidelity.
The accuracy and round-trip benchmarks each run in two channels: intensity (`I`) and polarization (`P`).

Every script writes JSON to `results/` and prints a markdown table.
`plots.py` reads only that JSON, so figures can be regenerated without re-running any transform.

## Running

The suite needs the environment that has all the pipeline dependencies plus the native `libfasttransforms`.
On this machine that is the `s2fft` micromamba environment.

```bash
P=/Users/basyrov/micromamba/envs/s2fft/bin/python

$P -m benchmarks.bench_speed      --channel I
$P -m benchmarks.bench_speed      --channel P
$P -m benchmarks.bench_accuracy   --channel I
$P -m benchmarks.bench_accuracy   --channel P
$P -m benchmarks.bench_roundtrip  --channel I --include-nyquist-corner
$P -m benchmarks.bench_roundtrip  --channel P --include-nyquist-corner
$P -m benchmarks.plots
```

The default ladders take about half an hour on one core.
Every script accepts `--nside`, `--backends`, `--seeds` and `--out`.

Results are written after each cell, so an interrupted run keeps what it finished.
Re-running skips completed cells; pass `--no-resume` to recompute them.

**`--no-resume` merges, it does not replace.**
It recomputes only the cells in the *current* `--nside` ladder and leaves every other row untouched.
So a run whose ladder omits an nside keeps that nside's old rows, and a run without the flag skips cells that already exist.
Re-baselining the whole file means passing the full ladder explicitly, e.g. `--nside 8 16 32 64 128 256 512 1024 --no-resume`.

Each record carries `_computed_utc`, so a stale cell is detectable after the fact.
`python -m benchmarks.compare_results [OLD_DIR] [NEW_DIR]` recomputes every number the documentation quotes, diffs two result sets, and reports the computation dates per file — mixed dates mean part of the file was not refreshed.
Do not use bit-identical numbers as a staleness signal: a deterministic backend reproduces exactly, so identical output is the normal case, not evidence of a skip.

**Speed rows are not comparable across days.**
A single HP2SPH timing varies ~6% run to run (healpy and ducc0 ~1%), and machine state drifts.
Compare speed by running both configurations alternately in one sitting and taking medians; the accuracy rows are deterministic and can be compared against a stored baseline safely.

Do not set `OMP_NUM_THREADS` or `KMP_DUPLICATE_LIB_OK` by hand.
`benchmarks/common.py` imports `src._bootstrap` before any numerical library loads, which sets both.
Importing it late lets several OpenMP runtimes come up multithreaded, which crashes or hangs the process rather than merely slowing it; see `src/_openmp.py`.

## Backends

| key | what it runs |
|---|---|
| `hp2sph` | this repo, default compact latitude band |
| `hp2sph-square` | `solver="svd"`, `solve_modes=8*nside+1`; intensity only, capped at nside 64 |
| `healpy-plain` | `map2alm(iter=0)` with no weights; `map2alm_spin` for polarization |
| `healpy-ring` | `map2alm(iter=0, use_weights=True)` |
| `healpy-pixel` | `map2alm(iter=0, use_pixel_weights=True)` |
| `healpy-iter3` | `map2alm(iter=3, use_weights=True)`, healpy's default |
| `ducc0-adjoint` | `adjoint_synthesis` with a uniform `4π/npix` ring factor |
| `ducc0-pseudo` | `pseudo_analysis`, ducc0's iterative least-squares analysis |

`healpy-plain` and `ducc0-adjoint` are the same mathematical operation.
They agree to 6e-15, so their overlap in the accuracy plots is a check that the harness drives both correctly.
Their speed differs, because the implementations differ.

This healpy build (1.19.0) uses its own bundled libsharp and does not link ducc0.
The two are therefore independent engines, not one wrapping the other.

### Backends that are skipped, and why

`healpy-pixel` is unavailable below nside 32.
healpy-data ships no pixel weights there, and `use_pixel_weights=True` silently falls back to no weights at all after the download fails.
Reporting that fallback under a "pixel weights" label would be a fabricated row, so `backends.pixel_weights_available` probes each nside and the cell is skipped instead.

`hp2sph-square` is capped at nside 64.
Its latitude Vandermonde has condition number about 6e15 by nside 128, which is `1/eps`.

`hp2sph-square` runs on the intensity channel only.
`forward_spin` fixes its own solver, the masked LSMR the rank-deficient spin fit requires, and ignores the nuFFT options, so a polarization entry would silently duplicate plain `hp2sph`.

## Benchmark 1: speed

`bench_speed.py` times forward and backward transforms against nside.
Every backend gets the same map, the same band `lmax = 2*nside`, one thread, and float64.

One warm-up call is discarded, then `--repeats` calls are timed and the minimum is reported.
The warm-up excludes one-off setup for every backend: FFTW plans, the FastTransforms rotation precompute, JAX tracing, and first-touch page faults.
This matters most for `hp2sph-square`, whose one-off cost is a cached O(nside³) truncated SVD.
Its timings are therefore the amortised cost of many transforms at one nside, not the cost of a single cold transform.

Single-threaded is the headline on purpose.
healpy and ducc0 both scale across cores, while the HP2SPH latitude solve is largely serial.
A multi-core comparison would measure OpenMP maturity rather than the algorithm.

The plot draws `O(N^1.5)` and `O(N log² N)` guide slopes so the measured scaling can be checked rather than assumed.
HP2SPH's advertised `O(N log² N)` comes entirely from the FastTransforms butterfly algorithm.
`nm -gU` on this build's `libfasttransforms` shows no butterfly symbols, so it runs the plain O(n³) Givens rotations.
Expect HP2SPH to sit in the same `O(N^1.5)` class as healpy here.

A second figure breaks the HP2SPH scalar forward into its four stages.
An end-to-end number cannot say which stage costs what, and the stage split is the only part of the speed story that is actionable.

## Benchmark 2: forward accuracy

`bench_accuracy.py` follows Drake & Wright (arXiv:1904.10514, Sec. 4).
A forward is not tested against its own inverse, which only proves self-consistency.
Instead the coefficients are known, the map is synthesised from them, and the analysis is compared against those known coefficients.

Three signal regimes are run.
They differ on two independent axes: whether the sky is band-limited, and how smooth it is.

**`cosmology` is the primary regime and the one to draw conclusions from.**
It uses `signal_lmax = lmax = 2*nside` with amplitude spectrum `sqrt(C_l) ∝ (1+l)^-1.5`: smooth and band-limited.
That is the regime cosmology actually works in.
It is also the exact configuration of the repository's own paper reproduction, `tests/test_paper_accuracy.py::test_compact_band_reproduces_paper_high_ell`.

**`flat`** uses `signal_lmax = 2*nside` with a flat spectrum.
Band-limited like `cosmology`, but carrying full power at the band edge.
It is the worst case within the primary regime rather than a different regime, so it bounds the error instead of describing a sky.

**`aliased`** uses `signal_lmax = 4*nside` with the same slope as `cosmology`.
The above-band tail aliases during analysis, which exposes each method's latitude quadrature.
It is diagnostic only.
The resulting error is dominated by grid aliasing common to every method, so it separates the methods poorly, and no real analysis runs there.
It is the regime in which the spin-2 defect below shows up.

In every regime `m` is capped at `2*nside-1` so `alm2map` stays an exact sampler of the function and the truth stays well defined.

Polarization adds a third measurement, `leakage`.
A pure-E sky goes in and the recovered `C_l^BB / C_l^EE` comes out, then the same with a pure-B sky.
For CMB work that ratio, not the diagonal accuracy, decides whether a transform is usable.
Neither the paper nor healpy reports it directly.

Two metrics are recorded per multipole.
The relative `C_l` error is the paper's, and is blind to coefficient phase.
The relative `a_lm` L2 error is phase sensitive, so it catches convention bugs a `C_l` comparison cannot see.
Both are summarised over four `l` bands, because one number across the whole spectrum averages the interesting top of the band away against the bulk, where every method is near-exact.

Per-`l` curves in the JSON are the median across `--seeds` realisations.

### Findings that refine the repository's documentation

Both were measured by this suite.
They are recorded here so the next person does not have to rediscover them.

**The scalar high-`l` advantage is real and grows with nside, reaching 4.4x at nside 1024.**
Configuration: scenario `cosmology`, median of 4 seeds, `l` from `3*lmax/4` to `7*lmax/8`, which is the band `CLAUDE.md` quotes.
Measured ratio of the healpy ring-weights error to the HP2SPH error:

| nside | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| ring / HP2SPH | 1.97 | 2.01 | 2.16 | 2.21 | 3.41 | 4.42 |
| pixel / HP2SPH | 1.84 | 1.62 | 1.36 | 1.07 | 1.47 | 1.64 |

The advantage needs high nside to appear, exactly as `CLAUDE.md` says.
At nside 1024 it reaches the bottom of the recorded 4x to 12x range.
The upper end of that range is not reproduced under 4-seed median averaging.
At nside 1024 over `l` 1536-1792 this suite measures HP2SPH 1.43e-6 against ring weights 6.35e-6.
`CLAUDE.md` records 2.9e-6 against 3.5e-5 for the same band.
The HP2SPH number here is better than the recorded one and the ring-weights number is 5.5x better, so the discrepancy sits entirely in the baseline rather than in HP2SPH.
Seed averaging is the most likely cause and has not been confirmed.

**The spin-2 forward is not usable at the top of the band when the sky is not band-limited.**
Configuration: scenario `aliased`, channel `P`, `signal_lmax = 4*nside`.
The relative `C_l^EE` error at the band edge reaches about 20 at nside 8 and about 0.5 at nside 64, against roughly 0.02 for healpy and ducc0 at the same points.
It converges with nside, so this is a convergence-rate problem rather than an outright failure.
In the primary `cosmology` regime the spin forward is competitive, so this does not affect normal use.
It matters because every spin validation in `tests/` uses a band-limited input, so nothing covered this case.

Tightening the tolerance of the masked LSMR solve makes it worse, not better.
At nside 8, `lmax = 16`, `signal_lmax = 32`, the top four multipoles give relative `C_l` errors of `[4.7, 4.6, 14.9, 27.3]` at the default `rtol=1e-6`.
At `rtol=1e-10` the same four give `[10.3, 89.6, 57.5, 521]`.
At `rtol=1e-10` with `maxiter=20000` they give `[143, 1208, 3076, 16263]`.
The rank-deficient solve is being held together by early stopping.
`CLAUDE.md` states that the physical output is stable under tolerance even though the raw coefficient vector is not.
That holds for a band-limited sky and does not hold here.

## Benchmark 3: round trip

`bench_roundtrip.py` runs three compositions.

**`harmonic`** is `alm → map → alm`, and is the primary measurement.
The starting coefficients are inside the analysis band, so the identity really is the right answer.

**`pixel`** is `map → alm → map`, on a band-limited map.
On a general map the composition is a projection rather than the identity, so the error would be dominated by above-band content every method discards.
That would measure aliasing, not the round trip.

**`native`** is `map → C → map` for the HP2SPH backends only, staying in the pipeline's own `(L+1, 2L+1)` coefficient array.
The cross-backend round trip has to pass through healpy-ordered `alm`, because that is the only representation all three libraries share.
`to_healpy_alm` reads only the triangular part of `C` and discards the quadrature residue the forward leaves in the tail cells.
That discard is what stops `hp2sph-square` from showing its bit-exact interpolation property.
Measured at nside 8, 16 and 32: 1.5e-13, 1.8e-13 and 1.5e-13 natively, against 2.2e-3 through the `alm` conversion.

All three run at `lmax = 2*nside - 1` rather than the full `2*nside`.
At exactly `2*nside` the single `l = m = lmax` coefficient cannot be represented.
`m = +2*nside` and `m = -2*nside` are the same mode on a `4*nside`-point longitude grid, and the per-ring `phi0` offsets give them different phases, so no column can carry both.
That corner is a property of the HEALPix grid rather than of any implementation.
Letting one coefficient dominate every curve would hide what the rest of the band does.
`--include-nyquist-corner` measures it explicitly instead: HP2SPH returns gain 0.5 there and healpy and ducc0 return 1.0, at nside 8 and 16.

## `src.FSHT.from_healpy_alm`

This benchmark suite is why `from_healpy_alm` exists.
It is the inverse of `to_healpy_alm` and the repository did not have it.
The scalar inverse pipeline consumes the raw FastTransforms `C` array, not a healpy `alm`, so a coefficients-in round trip needs this direction.
It now lives in `src/FSHT.py` next to its forward counterpart and is covered by `tests/test_FSHT.py` and `tests/test_pipeline.py`.

It relies on `to_healpy_alm` reading only the odd column of each real-spherical-harmonic pair because the even one is its conjugate.
That was verified to hold to 2.8e-16 on real pipeline output before the function was relied on.

Measured: `backward_map(from_healpy_alm(alm), nside)` reproduces `hp.alm2map(alm)` to 2.8e-13, 2.9e-13 and 3.4e-13 at nside 8, 16 and 32.
That is the same machine-precision agreement the native spin backward reaches, and it establishes that the scalar synthesis is exact in healpy's convention, not merely convergent.

## Figures

Colour encodes the library and linestyle plus marker encode the configuration.
Eight distinct hues would not survive a colour-vision check.
Three do, with a worst all-pairs CVD ΔE of 9.2 and a worst normal-vision ΔE of 24.0 against the light surface.

One palette slot sits below 3:1 contrast on that surface.
The required relief is that every chart carries a legend and every benchmark also prints and stores the same numbers as a table.
