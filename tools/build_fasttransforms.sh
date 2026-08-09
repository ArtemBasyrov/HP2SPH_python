#!/usr/bin/env bash
#
# Build libfasttransforms for HP2SPH.
#
# The FSHT stage needs Slevinsky's libfasttransforms, which is not on PyPI or
# conda-forge, so it has to be built. Two modes, and the difference is which
# OpenMP runtime the result links:
#
#   --prefix <env>   link the OpenMP/FFTW/MPFR in <env>, and install into
#                    <env>/lib. Use this with a conda-forge environment whose
#                    healpy and finufft also link that env's llvm-openmp. That
#                    is the ONLY configuration in which the pipeline can use
#                    threads -- see the OpenMP section of README.md.
#
#                    BLAS comes from Apple Accelerate by default on macOS, NOT
#                    from the environment. ft_execute_fourier2sph is four
#                    cblas_dtrmm calls, and Accelerate beats conda-forge
#                    OpenBLAS by 1.26x on that stage (FSHT 0.286 s vs 0.358 s
#                    at nside 512, median of 3). Accelerate threads through GCD
#                    rather than OpenMP, so it does NOT reintroduce a second
#                    OpenMP runtime. Pass --blas openblas to override.
#
#   --homebrew       link Homebrew's libomp/fftw/mpfr and Apple Accelerate, and
#                    install into the repo's lib/. Works with PyPI healpy and
#                    finufft wheels, but each of those vendors its own libomp,
#                    so the process is stuck at one thread.
#
# Usage:
#   tools/build_fasttransforms.sh --homebrew
#   tools/build_fasttransforms.sh --prefix "$HOME/micromamba/envs/hp2sph-omp"
#   tools/build_fasttransforms.sh --prefix <env> --source /path/to/FastTransforms
#
# With no --source the script clones FastTransforms into build/ beside the repo.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="https://github.com/MikaelSlevinsky/FastTransforms"

MODE=""
PREFIX=""
SOURCE=""
BLAS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --homebrew) MODE="homebrew"; shift ;;
        --prefix)   MODE="prefix"; PREFIX="${2:?--prefix needs a path}"; shift 2 ;;
        --source)   SOURCE="${2:?--source needs a path}"; shift 2 ;;
        --blas)     BLAS="${2:?--blas needs accelerate|openblas}"; shift 2 ;;
        -h|--help)  sed -n '2,42p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$MODE" ]] || { echo "pass --homebrew or --prefix <env>" >&2; exit 2; }

if [[ -z "$SOURCE" ]]; then
    SOURCE="$REPO_ROOT/build/FastTransforms"
    if [[ ! -d "$SOURCE" ]]; then
        echo ">> cloning $UPSTREAM into $SOURCE"
        mkdir -p "$(dirname "$SOURCE")"
        git clone --depth 1 "$UPSTREAM" "$SOURCE"
    fi
fi
[[ -f "$SOURCE/Makefile" ]] || { echo "no Makefile in $SOURCE" >&2; exit 1; }

if [[ "$MODE" == "homebrew" ]]; then
    BREW="$(brew --prefix 2>/dev/null || echo /usr/local)"
    echo ">> homebrew prefix: $BREW"
    for pkg in libomp fftw mpfr gmp; do
        [[ -d "$BREW/opt/$pkg" ]] || { echo "missing: brew install $pkg" >&2; exit 1; }
    done
    # Upstream Make.inc hardcodes the INTEL homebrew prefix (/usr/local/opt). On
    # Apple Silicon brew lives at /opt/homebrew, and the build dies with
    # "'fftw3.h' file not found". Patch the paths in place; it is idempotent.
    if [[ "$BREW" != "/usr/local" ]]; then
        echo ">> patching Make.inc: /usr/local/opt -> $BREW/opt"
        perl -pi -e "s{/usr/local/opt}{$BREW/opt}g" "$SOURCE/Make.inc"
    fi
    BUILD_ARGS=(CC=clang FT_USE_APPLEBLAS=1)
    DEST="$REPO_ROOT/lib"
else
    [[ -d "$PREFIX/lib" ]] || { echo "no such prefix: $PREFIX" >&2; exit 1; }
    [[ -e "$PREFIX/lib/libomp.dylib" || -e "$PREFIX/lib/libomp.so" ]] || {
        echo "no libomp in $PREFIX/lib -- install llvm-openmp in that env" >&2
        exit 1
    }
    echo ">> prefix: $PREFIX"
    if [[ -z "$BLAS" ]]; then
        if [[ "$(uname -s)" == "Darwin" ]]; then BLAS=accelerate; else BLAS=openblas; fi
    fi
    echo ">> blas: $BLAS"
    # FT_PREFIX is Make.inc's own supported hook for a non-system dependency tree.
    # The rpath is what lets the built library find that env's libomp at runtime.
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib"
    if [[ "$BLAS" == "accelerate" ]]; then
        SDK="$(xcrun --show-sdk-path)"
        VECLIB="$SDK/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A"
        [[ -d "$VECLIB" ]] || { echo "no vecLib under $SDK" >&2; exit 1; }
        # vecLib's lib dir and headers must come BEFORE the prefix, or -lblas
        # picks up the environment's libblas.dylib (a symlink to OpenBLAS).
        # CFLAGS has to be given whole: Make.inc guards it with ifndef, so it
        # cannot be appended to, and -mcpu=native must be reinstated by hand.
        export LDFLAGS="-L$VECLIB $LDFLAGS"
        BUILD_ARGS=(CC=clang FT_PREFIX="$PREFIX" FT_BLAS=blas
                    CFLAGS="-O3 -mcpu=native -std=gnu11 -I./src -I$VECLIB/Headers -I$PREFIX/include")
    else
        BUILD_ARGS=(CC=clang FT_PREFIX="$PREFIX" FT_BLAS=openblas)
    fi
    DEST="$PREFIX/lib"
fi

echo ">> building in $SOURCE"
make -C "$SOURCE" assembly lib "${BUILD_ARGS[@]}"

LIB="$SOURCE/libfasttransforms.dylib"
[[ -f "$LIB" ]] || LIB="$SOURCE/libfasttransforms.so"
[[ -f "$LIB" ]] || { echo "build produced no library" >&2; exit 1; }

mkdir -p "$DEST"
cp "$LIB" "$DEST/$(basename "$LIB")"
echo ">> installed $DEST/$(basename "$LIB")"

echo ">> OpenMP runtime and BLAS it links:"
if command -v otool >/dev/null; then
    otool -L "$DEST/$(basename "$LIB")" | grep -iE "omp|blas" || echo "   (none found)"
else
    ldd "$DEST/$(basename "$LIB")" | grep -iE "omp|blas" || echo "   (none found)"
fi

if [[ "$MODE" == "prefix" ]]; then
    cat <<EOF

Built against $PREFIX and installed into its lib/, where src/ft_sphere.py finds
it automatically -- no FASTTRANSFORMS_LIB needed, PROVIDED $REPO_ROOT/lib holds
no other copy. That directory is searched first and will shadow this one.

To use threads, confirm one runtime and then opt in:

    python -c "from src import _openmp; print(_openmp.runtime_paths())"
    HP2SPH_OMP_THREADS=8 python your_script.py

If more than one path prints, threading will raise rather than crash.
EOF
fi
