#!/bin/sh
# Triton autotuner compiles cuda_utils.so via `gcc -lcuda -L/usr/local/nvidia/lib64`
# at import time. Vertex mounts /usr/local/nvidia as READ-ONLY, so we can't symlink
# there. Instead create a writable shim dir + point gcc/dlopen at it.
set -e

SHIM_DIR=/tmp/cudalib
mkdir -p "$SHIM_DIR"

LIBCUDA=""
for p in \
    /usr/local/nvidia/lib64/libcuda.so.1 \
    /usr/local/nvidia/lib64/libcuda.so \
    /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    /usr/lib/x86_64-linux-gnu/libcuda.so \
    /usr/lib64/libcuda.so.1 \
    /usr/local/cuda/lib64/libcuda.so.1 \
    /usr/local/cuda/lib64/libcuda.so \
    /usr/local/cuda/compat/libcuda.so.1; do
    if [ -e "$p" ]; then
        LIBCUDA="$p"
        break
    fi
done

if [ -n "$LIBCUDA" ]; then
    ln -sf "$LIBCUDA" "$SHIM_DIR/libcuda.so"
    ln -sf "$LIBCUDA" "$SHIM_DIR/libcuda.so.1"
    echo "[entrypoint] libcuda.so source: $LIBCUDA"
else
    echo "[entrypoint] WARN: libcuda.so not found anywhere — triton compile will fail"
    find / -name 'libcuda*' 2>/dev/null | head -20
fi

# LIBRARY_PATH: gcc reads at link time (Triton's gcc compile).
# LD_LIBRARY_PATH: dynamic linker / dlopen at runtime.
export LIBRARY_PATH="$SHIM_DIR:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SHIM_DIR:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

exec python /workspace/train.py "$@"
