#!/usr/bin/env bash
# Submit a Vertex AI eval job: run the held-out eval set through a trained model,
# grade apply_edits JSON, write report to GCS. Reuses the trainer image with the
# eval_entrypoint.sh override. A100 spot.
#
# Usage: RUN_ID=v1-20260520-095423 ./submit_eval_job.sh
set -euo pipefail

PROJECT="${PROJECT:-your-gcp-project}"
REGION="${REGION:-us-central1}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/${PROJECT}/your-ar-repo/trainer:latest}"
RUN_ID="${RUN_ID:?set RUN_ID=<training run id, e.g. v1-20260520-095423>}"
GCS_MODEL="${GCS_MODEL:-gs://your-training-bucket/runs/${RUN_ID}/merged_16bit}"
GCS_REPORT="${GCS_REPORT:-gs://your-training-bucket/runs/${RUN_ID}/eval_report.json}"
EVAL_RUN_ID="eval-${RUN_ID}-$(date +%H%M%S)"

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
      command: ["/workspace/eval_entrypoint.sh"]
      env:
        - name: GCS_MODEL
          value: ${GCS_MODEL}
        - name: GCS_EVAL
          value: ${GCS_EVAL:-gs://your-training-bucket/dataset/v2/eval.messages.jsonl}
        - name: GCS_REPORT
          value: ${GCS_REPORT}
scheduling:
  strategy: SPOT
  restartJobOnWorkerRestart: false
EOF

echo "EVAL_RUN_ID=${EVAL_RUN_ID}"
echo "model=${GCS_MODEL}"
echo "report→${GCS_REPORT}"
gcloud ai custom-jobs create \
  --region="${REGION}" \
  --display-name="eval-${EVAL_RUN_ID}" \
  --config="${CONFIG_FILE}" \
  --project="${PROJECT}"
rm -f "$CONFIG_FILE"
