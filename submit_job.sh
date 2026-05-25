#!/usr/bin/env bash
# Submit Vertex AI Custom Job for Patchnote LoRA fine-tune.
# L4 spot in us-central1. ~$0.21/hr. ~1.5-2hr run = ~$0.40.

set -euo pipefail

PROJECT="${PROJECT:-your-gcp-project}"
REGION="${REGION:-us-central1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/${PROJECT}/your-ar-repo/trainer:latest}"
RUN_ID="${RUN_ID:-v2-$(date +%Y%m%d-%H%M%S)}"
GCS_OUTPUT="gs://your-training-bucket/runs/${RUN_ID}"
GCS_DATASET="${GCS_DATASET:-gs://your-training-bucket/dataset/v2}"
BASE_MODEL="${BASE_MODEL:-unsloth/gemma-4-e2b-it-unsloth-bnb-4bit}"

CONFIG_FILE=$(mktemp)
cat > "$CONFIG_FILE" <<EOF
workerPoolSpecs:
  - machineSpec:
      machineType: a2-highgpu-1g
      acceleratorType: NVIDIA_TESLA_A100
      acceleratorCount: 1
    replicaCount: 1
    diskSpec:
      bootDiskType: pd-ssd
      bootDiskSizeGb: 200
    containerSpec:
      imageUri: ${IMAGE}
      env:
        - name: GCS_OUTPUT
          value: ${GCS_OUTPUT}
        - name: GCS_DATASET
          value: ${GCS_DATASET}
        - name: BASE_MODEL
          value: ${BASE_MODEL}
        - name: EPOCHS
          value: "3"
        - name: BATCH
          value: "2"
        - name: GRAD_ACCUM
          value: "4"
        - name: LORA_R
          value: "16"
        - name: LORA_ALPHA
          value: "32"
scheduling:
  strategy: SPOT
  restartJobOnWorkerRestart: false
EOF

echo "RUN_ID=${RUN_ID}"
echo "GCS_OUTPUT=${GCS_OUTPUT}"
echo "Submitting Vertex AI custom job..."

gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="finetune-${RUN_ID}" \
  --config="${CONFIG_FILE}" \
  --project="${PROJECT}"

rm -f "$CONFIG_FILE"
