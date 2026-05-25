#!/bin/sh
# Conversion entrypoint for Vertex AI (CPU machine).
# Download merged_16bit from GCS -> run litert-torch export_hf -> upload .litertlm.
set -e

# Inputs (env, with defaults)
GCS_MODEL="${GCS_MODEL:?set GCS_MODEL=gs://.../merged_16bit}"
GCS_OUT_LITERTLM="${GCS_OUT_LITERTLM:?set GCS_OUT_LITERTLM=gs://.../litertlm/<name>.litertlm}"
QUANT="${QUANT:-dynamic_wi4_afp32}"

export MODEL_DIR=/workspace/model
export OUTPUT_DIR=/workspace/out
export QUANT

mkdir -p "$MODEL_DIR" "$OUTPUT_DIR"

echo "[convert-entrypoint] downloading model from $GCS_MODEL"
python - <<'PY'
import os
from google.cloud import storage
def parse(uri):
    b, _, p = uri[5:].partition("/"); return b, p
client = storage.Client()
mb, mp = parse(os.environ["GCS_MODEL"])
prefix = mp.rstrip("/") + "/"
got = 0
for blob in client.bucket(mb).list_blobs(prefix=prefix):
    rel = blob.name[len(prefix):]
    if not rel or rel.endswith("/"):
        continue
    # skip the HF .cache dir — not needed for conversion
    if rel.startswith(".cache/"):
        continue
    dst = os.path.join(os.environ["MODEL_DIR"], rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    blob.download_to_filename(dst)
    got += 1
    print("  got", rel, flush=True)
print(f"  downloaded {got} files", flush=True)
PY

echo "[convert-entrypoint] running conversion (quant=$QUANT)"
python /workspace/convert.py

LITERTLM_PATH=$(cat "$OUTPUT_DIR/.litertlm_path")
echo "[convert-entrypoint] uploading $LITERTLM_PATH -> $GCS_OUT_LITERTLM"
LITERTLM_PATH="$LITERTLM_PATH" python - <<'PY'
import os
from google.cloud import storage
def parse(uri):
    b, _, p = uri[5:].partition("/"); return b, p
client = storage.Client()
rb, rp = parse(os.environ["GCS_OUT_LITERTLM"])
src = os.environ["LITERTLM_PATH"]
size = os.path.getsize(src)
print(f"  uploading {src} ({size/1e9:.2f} GB)", flush=True)
client.bucket(rb).blob(rp).upload_from_filename(src)
print("  uploaded", flush=True)
PY

echo "[convert-entrypoint] DONE"
