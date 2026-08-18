"""Where the FSHT stage's time and memory go, split by sub-step.

One nside per process: peak RSS is a process-lifetime high-water mark, so several
resolutions in one process would report the running maximum of the largest so far.

For n = L+1 with L = 2*nside this measures, separately:

  plan      ft_plan_sph2fourier(n)   -- four dense n x n matrices, the Givens
                                        rotation tables, and an n x (2n-1) scratch
  rot       ft_execute_sph_lo2hi     -- step 1, the dense Givens sweep
  full      ft_execute_fourier2sph   -- step 1 + step 2 (four dense dtrmm)
  wrapper   src.ft_sphere.fourier2sph -- the Python side, including its copies

Usage: python -u tools/fsht_diagnostics/f1_fsht_budget.py <nside> [reps]
"""

import ctypes, gc, os, resource, subprocess, sys, time
import numpy as np

sys.path.insert(0, ".")
from src import ft_sphere as fts  # noqa: E402

GB = float(1 << 30)


def peak():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r if sys.platform == "darwin" else r * 1024) / GB


def live():
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    ).stdout.strip()
    return int(out) * 1024 / GB


class RotPlan(ctypes.Structure):
    _fields_ = [("s", ctypes.c_void_p), ("c", ctypes.c_void_p), ("n", ctypes.c_int)]


class HarmPlan(ctypes.Structure):
    _fields_ = [
        ("RP", ctypes.POINTER(ctypes.POINTER(RotPlan))),
        ("MP", ctypes.c_void_p),
        ("B", ctypes.c_void_p),
        ("P", ctypes.POINTER(ctypes.c_void_p)),
        ("Pinv", ctypes.POINTER(ctypes.c_void_p)),
        ("alpha", ctypes.c_double),
        ("beta", ctypes.c_double),
        ("gamma", ctypes.c_double),
        ("delta", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("NRP", ctypes.c_int),
        ("NMP", ctypes.c_int),
        ("NP", ctypes.c_int),
    ]


def main(nside, reps=3):
    L = 2 * nside
    n, m = L + 1, 2 * L + 1
    g_gb = n * m * 16 / GB
    rot_gb = n * (n + 1) * 8 / GB  # s and c, n(n+1)/2 doubles each
    mat_gb = 4 * n * n * 8 / GB  # P[0], P[1], Pinv[0], Pinv[1]
    scratch_gb = n * m * 8 / GB  # P->B
    print(f"nside {nside}   L {L}   n {n}   m {m}")
    print("  PREDICTED, from the C source:")
    print(f"    g array (complex128)         {g_gb:8.3f} GB")
    print(f"    rotation tables s,c          {rot_gb:8.3f} GB   (closed form!)")
    print(f"    4 dense n x n connection     {mat_gb:8.3f} GB   (upper triangular)")
    print(f"    plan scratch B, n x (2n-1)   {scratch_gb:8.3f} GB")
    print(f"    plan total                   {rot_gb + mat_gb + scratch_gb:8.3f} GB")

    lib = fts._lib
    lib.ft_plan_sph2fourier.restype = ctypes.POINTER(HarmPlan)
    lib.ft_execute_sph_lo2hi.restype = None
    lib.ft_execute_sph_lo2hi.argtypes = [
        ctypes.POINTER(RotPlan),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    fts._pin_threads()

    print("\n  MEASURED:")
    gc.collect()
    p0, l0 = peak(), live()
    t = time.perf_counter()
    P = lib.ft_plan_sph2fourier(n)
    t_plan = time.perf_counter() - t
    print(
        f"    plan build      {t_plan:9.3f} s   retained {live() - l0:7.3f} GB"
        f"   transient peak {peak() - p0:7.3f} GB"
    )

    rng = np.random.default_rng(0)
    A = np.asfortranarray(rng.standard_normal((n, m)))
    ptr = A.ctypes.data_as(ctypes.c_void_p)
    RP0 = P.contents.RP[0]

    def timeit(call, label):
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            call()
            ts.append(time.perf_counter() - t)
        print(f"    {label:14s} {min(ts):9.3f} s")
        return min(ts)

    t_rot = timeit(
        lambda: lib.ft_execute_sph_lo2hi(RP0, ptr, P.contents.B, m), "rotations"
    )
    t_full = timeit(
        lambda: lib.ft_execute_fourier2sph(fts._TRANS_N, P, ptr, n, m), "fourier2sph"
    )
    print(
        f"    step 2 (by diff) {t_full - t_rot:8.3f} s"
        f"   = {100 * (t_full - t_rot) / t_full:4.1f}% of fourier2sph"
    )

    del A, ptr
    gc.collect()
    p1, l1 = peak(), live()
    g = rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))
    l2 = live()
    t = time.perf_counter()
    out = fts.fourier2sph(g)
    t_wrap = time.perf_counter() - t
    print(
        f"\n    wrapper         {t_wrap:9.3f} s"
        f"   (raw x2 for re/im = {2 * t_full:.3f} s,"
        f" overhead {t_wrap - 2 * t_full:+.3f} s)"
    )
    print(f"    g array held    {l2 - l1:9.3f} GB")
    print(
        f"    wrapper peak   +{peak() - p1:9.3f} GB over pre-g"
        f"   = {(peak() - p1) / g_gb:5.2f} x the g array"
    )
    print(f"    TOTAL peak RSS  {peak():9.3f} GB")
    assert out.shape == (n, m)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 3)
