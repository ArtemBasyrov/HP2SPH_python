"""The Python side of the FSHT stage holds ~4.7x the g array. Can it hold 1.5x?

`ft_sphere._apply` allocates, live simultaneously: the input `A`, an `out` array,
a real Fortran buffer per part, and the `np.conj` result. This prototypes a
low-water version and checks it is BIT-IDENTICAL to the shipped one.

Usage: python -u tools/fsht_diagnostics/f2_wrapper_copies.py <nside> [--mem]
"""

import ctypes, gc, os, resource, subprocess, sys, time
import numpy as np

sys.path.insert(0, ".")
from src import ft_sphere as fts  # noqa: E402

GB = float(1 << 30)
peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GB


def live():
    o = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout.strip()
    return int(o) * 1024 / GB


def apply_lowmem(symbol, A, overwrite=False):
    """Same result as ft_sphere._apply, with two of the four full-size arrays gone.

    Shipped: A + out + buf + conj(out)          -> ~4.7x the g array, measured.
    Here:    A + out + buf   (overwrite: A + buf) -> 2.5x (1.5x).

    The two savings are independent. `np.conjugate(..., out=)` writes the
    conjugation in place instead of allocating a fourth array, and one real
    Fortran scratch is reused across the real and imaginary passes instead of
    being reallocated per part.
    """
    A = np.asarray(A, dtype=np.complex128)
    n, m = A.shape
    fn = getattr(fts._lib, symbol)
    fts._pin_threads()
    p = fts._plan(n)
    out = A if overwrite else np.empty_like(A)
    buf = np.empty((n, m), dtype=np.float64, order="F")
    ptr = buf.ctypes.data_as(ctypes.c_void_p)
    for part in ("real", "imag"):
        np.copyto(buf, getattr(A, part))
        fn(fts._TRANS_N, p, ptr, n, m)
        setattr(out, part, buf)
    np.conjugate(out, out=out)
    return out


def main(nside, mem=False):
    L = 2 * nside
    n, m = L + 1, 2 * L + 1
    g_gb = n * m * 16 / GB
    rng = np.random.default_rng(0)
    g = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    print(f"nside {nside}  n {n}  m {m}  g {g_gb:.3f} GB")

    if not mem:
        a = fts.fourier2sph(g)
        b = apply_lowmem("ft_execute_fourier2sph", g.copy())
        c = apply_lowmem("ft_execute_fourier2sph", g.copy(), overwrite=True)
        print(f"  low-mem  == shipped : {np.array_equal(a, b)}")
        print(f"  overwrite== shipped : {np.array_equal(a, c)}")
        a2 = fts.sph2fourier(g)
        b2 = apply_lowmem("ft_execute_sph2fourier", g.copy())
        print(f"  sph2fourier         : {np.array_equal(a2, b2)}")
        return

    for label, fn in (
        ("shipped  ", lambda x: fts.fourier2sph(x)),
        ("low-mem  ", lambda x: apply_lowmem("ft_execute_fourier2sph", x)),
        (
            "overwrite",
            lambda x: apply_lowmem("ft_execute_fourier2sph", x, overwrite=True),
        ),
    ):
        pid = os.fork()
        if pid == 0:
            gc.collect()
            x = g.copy()
            p0 = peak()
            t = time.perf_counter()
            r = fn(x)
            dt = time.perf_counter() - t
            print(
                f"  {label}  {dt:7.3f} s   peak +{peak() - p0:7.3f} GB"
                f"   = {(peak() - p0) / g_gb:5.2f} x g   (checksum {r[0, 0].real:.6e})"
            )
            os._exit(0)
        os.waitpid(pid, 0)


if __name__ == "__main__":
    main(int(sys.argv[1]), "--mem" in sys.argv)
