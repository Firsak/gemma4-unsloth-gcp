#!/usr/bin/env bash
# Submit a Vertex AI job: convert a LoRA-merged Gemma 4 E2B HF checkpoint -> .litertlm.
# litert-torch export_hf is CPU-only, so we use a CPU machine (n1-highmem-32) —
# avoids the A100 spot queue entirely. ~$1.50/hr on-demand CPU; conversion ~20-40min.
#
# Usage: RUN_ID=v2-20260520-174526 ./submit_convert_job.sh
set -euo pipefail

PROJECT="${PROJECT:-your-gcp-project}"
REGION="${REGION:-us-central1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/${PROJECT}/your-ar-repo/trainer-convert:latest}"
RUN_ID="${RUN_ID:?set RUN_ID=<training run id, e.g. v2-20260520-174526>}"
GCS_MODEL="${GCS_MODEL:-gs://your-training-bucket/runs/${RUN_ID}/merged_16bit}"
GCS_OUT_LITERTLM="${GCS_OUT_LITERTLM:-gs://your-training-bucket/runs/${RUN_ID}/litertlm/gemma-4-E2B-it.litertlm}"
QUANT="${QUANT:-dynamic_wi4_afp32}"
CONVERT_RUN_ID="convert-${RUN_ID}-$(date +%H%M%S)"

CONFIG_FILE=$(mktemp)
cat > "$CONFIG_FILE" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: n1-highmem-32
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${IMAGE}
      env:
        - name: GCS_MODEL
          value: ${GCS_MODEL}
        - name: GCS_OUT_LITERTLM
          value: ${GCS_OUT_LITERTLM}
        - name: QUANT
          value: ${QUANT}
EOF

echo "CONVERT_RUN_ID=${CONVERT_RUN_ID}"
echo "model=${GCS_MODEL}"
echo "out=${GCS_OUT_LITERTLM}"
echo "quant=${QUANT}"
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="convert-${CONVERT_RUN_ID}" \
  --config="${CONFIG_FILE}" \
  --project="${PROJECT}"
rm -f "$CONFIG_FILE"
