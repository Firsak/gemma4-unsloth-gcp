#!/bin/sh
# Eval entrypoint for Vertex AI. Same libcuda shim as training (transformers →
# bitsandbytes.triton triggers a runtime gcc compile that needs libcuda + headers).
set -e

SHIM_DIR=/tmp/cudalib
mkdir -p "$SHIM_DIR"
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
        ln -sf "$p" "$SHIM_DIR/libcuda.so"
        ln -sf "$p" "$SHIM_DIR/libcuda.so.1"
        echo "[eval-entrypoint] libcuda.so source: $p"
        break
    fi
done
export LIBRARY_PATH="$SHIM_DIR:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$SHIM_DIR:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Inputs (env, with defaults)
# GCS_MODEL is either a gs:// path to a merged_16bit dir (fine-tuned run) OR a
# bare HuggingFace model id (e.g. unsloth/gemma-4-e2b-it) for a vanilla baseline.
GCS_MODEL="${GCS_MODEL:?set GCS_MODEL=gs://.../merged_16bit OR an HF id}"
GCS_EVAL="${GCS_EVAL:-gs://your-training-bucket/dataset/v1/eval.messages.jsonl}"
GCS_REPORT="${GCS_REPORT:?set GCS_REPORT=gs://.../eval_report.json}"

mkdir -p /workspace/model /workspace/data

# Always fetch the eval set from GCS.
python - <<'PY'
import os
from google.cloud import storage
def parse(uri):
    b, _, p = uri[5:].partition("/"); return b, p
client = storage.Client()
eb, ep = parse(os.environ["GCS_EVAL"])
client.bucket(eb).blob(ep).download_to_filename("/workspace/data/eval.messages.jsonl")
print("  got eval set", flush=True)
PY

case "$GCS_MODEL" in
  gs://*)
    echo "[eval-entrypoint] downloading fine-tuned model from $GCS_MODEL"
    python - <<'PY'
import os
from google.cloud import storage
def parse(uri):
    b, _, p = uri[5:].partition("/"); return b, p
client = storage.Client()
mb, mp = parse(os.environ["GCS_MODEL"])
for blob in client.bucket(mb).list_blobs(prefix=mp.rstrip("/") + "/"):
    rel = blob.name[len(mp.rstrip("/")) + 1:]
    if not rel or rel.endswith("/"):
        continue
    dst = os.path.join("/workspace/model", rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    blob.download_to_filename(dst)
    print("  got", rel, flush=True)
PY
    MODEL_ARG=/workspace/model
    ;;
  *)
    echo "[eval-entrypoint] vanilla HF model id: $GCS_MODEL (transformers will fetch)"
    MODEL_ARG="$GCS_MODEL"
    ;;
esac

echo "[eval-entrypoint] running eval (model=$MODEL_ARG)"
python /workspace/eval.py \
    --model-path "$MODEL_ARG" \
    --eval-file /workspace/data/eval.messages.jsonl \
    --out /workspace/eval_report.json

echo "[eval-entrypoint] uploading report to $GCS_REPORT"
python - <<'PY'
import os
from google.cloud import storage
def parse(uri):
    b, _, p = uri[5:].partition("/"); return b, p
client = storage.Client()
rb, rp = parse(os.environ["GCS_REPORT"])
client.bucket(rb).blob(rp).upload_from_filename("/workspace/eval_report.json")
print("uploaded", flush=True)
PY
echo "[eval-entrypoint] DONE"
