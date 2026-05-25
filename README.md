# Gemma 4 QLoRA fine-tuning on GCP Vertex AI

Working Docker image + Vertex AI Custom Training scripts for fine-tuning **Gemma 4 (E2B / E4B)** with Unsloth QLoRA on an A100 spot in us-central1. ~37 minutes, ~$0.75 per run.

This repo is the residue of **16 dependency-archaeology fixes**. The default Colab notebook didn't work on Gemma 4. The default Unsloth `install.sh` doesn't work on Vertex A100 spot drivers. The published `cu118-ampere-torch270` extras can't be resolved. The Dockerfile here is the stack that does work.

See [`ITERATION-LOG.md`](./ITERATION-LOG.md) for the full failure-by-failure story.

---

## What's in here

| File | Purpose |
|---|---|
| `Dockerfile` | Trainer image — torch 2.7.1+cu118, transformers 5.x (git head), unsloth 2026.5.x, cut-cross-entropy, libcuda shim. |
| `Dockerfile.convert` | Sibling image for converting HF safetensors → LiteRT `.litertlm` via `ai-edge-torch` / `litert-torch-nightly`. CPU only. |
| `train.py` | Unsloth QLoRA r=16/α=32, 3 epochs, cosine, bf16. Reads OpenAI-style messages JSONL from GCS. |
| `eval.py` | HF greedy generation + tolerant brace-walk JSON parser + exact-match grader by op category. Works on fine-tuned or vanilla HF model ids. |
| `convert.py` | `litert-torch` export — Gemma 4 → INT4 `.litertlm` bundle for on-device deploy (LiteRT-LM). |
| `entrypoint.sh` | Runtime shim that symlinks the host driver's `libcuda.so` into a writable dir (Vertex mounts `/usr/local/nvidia` read-only). |
| `eval_entrypoint.sh` / `convert_entrypoint.sh` | Same shim + GCS download/upload for the eval and convert jobs. |
| `submit_job.sh` / `submit_eval_job.sh` / `submit_convert_job.sh` | Vertex AI Custom Job submission via `gcloud ai custom-jobs create`. |
| `cloudbuild.yaml` | Optional Cloud Build config. **Use local Docker for iteration** — it's ~10× faster (layer cache works). |
| `ITERATION-LOG.md` | The full 16-fix debugging chronicle. |

---

## Prerequisites

- GCP project with billing enabled
- APIs enabled: `compute`, `aiplatform`, `artifactregistry`, `cloudbuild` (last one optional)
- One GCS bucket for dataset + run artifacts; one for Cloud Build logs (if using Cloud Build)
- One Artifact Registry Docker repo
- Compute SA granted `cloudbuild.builds.builder`, `artifactregistry.writer`, `logging.logWriter`, `storage.admin`
- A100 spot quota ≥ 1 in your region. In `us-central1` the default is **8** — no quota request needed.
- `gcloud` CLI authenticated locally
- Docker (with `docker login` to your AR repo: `gcloud auth configure-docker us-central1-docker.pkg.dev`)

---

## Configure

The scripts read env vars with sensible defaults; set these to your project (or `sed` the placeholders in the `.sh` files):

```bash
export PROJECT=your-gcp-project
export REGION=us-central1
export GCS_BUCKET=gs://your-training-bucket
export IMAGE=us-central1-docker.pkg.dev/${PROJECT}/your-ar-repo/trainer:latest
export GCS_DATASET=${GCS_BUCKET}/dataset/v1
export RUN_ID=$(date +%Y%m%d-%H%M%S)
```

Dataset format: two files, OpenAI-style messages JSONL:

```jsonl
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Upload to `${GCS_DATASET}/train.messages.jsonl` and `${GCS_DATASET}/eval.messages.jsonl`.

---

## Build the trainer image

```bash
# Local — fastest iteration (~30s for re-pip steps thanks to layer cache)
docker build -t $IMAGE -f Dockerfile .
docker push $IMAGE
```

The Dockerfile asserts torch CUDA version at the end so a wrong pin fails the BUILD (~seconds), not a queued Vertex job (~minutes).

---

## Train

```bash
./submit_job.sh
```

Submits a Vertex Custom Job (`a2-highgpu-1g`, 1× A100-40GB, spot). Poll with:

```bash
gcloud ai custom-jobs describe <JOB_ID> --region=$REGION --format='value(state)'
```

Artifacts land in `${GCS_BUCKET}/runs/${RUN_ID}/`:
- `adapter/adapter_model.safetensors` — LoRA weights (~100 MB)
- `merged_16bit/model.safetensors` — full merged fp16 (eval / conversion input)
- `checkpoints/checkpoint-N`
- `metrics.json`

---

## Eval

```bash
GCS_MODEL=${GCS_BUCKET}/runs/${RUN_ID}/merged_16bit \
GCS_REPORT=${GCS_BUCKET}/runs/${RUN_ID}/eval_report.json \
./submit_eval_job.sh
```

Eval also accepts a vanilla HF id (no GCS download) for baseline comparison:

```bash
GCS_MODEL=google/gemma-4-2b-it ./submit_eval_job.sh
```

The grader is exact-match (multiset of normalized ops) — read the comments in `eval.py` for caveats; tolerant device-faithful grading is a separate exercise.

---

## Convert to LiteRT (on-device deploy)

```bash
docker build -t ${IMAGE_CONVERT:?set IMAGE_CONVERT} -f Dockerfile.convert .
docker push $IMAGE_CONVERT

GCS_MODEL=${GCS_BUCKET}/runs/${RUN_ID}/merged_16bit \
GCS_OUT_LITERTLM=${GCS_BUCKET}/runs/${RUN_ID}/litertlm/model.litertlm \
./submit_convert_job.sh
```

Runs on a CPU VM (`n1-highmem-32`) — conversion is RAM-bound, not GPU. Produces an INT4 `.litertlm` bundle that loads directly into the LiteRT-LM Android runtime.

---

## Why these pins

Short version of `ITERATION-LOG.md`:

- **torch 2.7.1+cu118** — Vertex A100 spot driver caps at CUDA 12.0; cu121+ fails with "driver too old". torch ≥ 2.7 because torchao (transitive via transformers 5.x) needs `torch.int1` (added 2.6).
- **transformers from git head (5.x)** — Gemma 4 `model_type` landed in 4.56; current head recognizes it cleanly.
- **peft ≥ 0.19** — older peft imports `HybridCache` which transformers 5.x removed.
- **torchao removed entirely** — peft 0.19 hard-requires torchao ≥ 0.16, but 0.16+ needs torch 2.8 (no cu118 wheel). Removing torchao + `sed`-patching transformers' quantizer import makes peft skip its check. We use bnb-4bit, never torchao.
- **`cut-cross-entropy` explicitly installed** — `unsloth_zoo.loss_utils.fused_linear_cross_entropy` calls `linear_cross_entropy()` from this optional package without checking the guard. Without it: `NameError` at training step 0.
- **triton 3.3.0** — torch 2.7's inductor needs the older `triton_key` API; triton 3.6 dropped it.
- **`--no-deps` for unsloth** — Unsloth's `cu118-ampere-torch270` extras pull `xformers` and `flash-attn` which have no wheel for this combo. Handpick runtime deps in the preceding RUN.
- **`entrypoint.sh` libcuda shim** — Triton autotuner compiles a native shim at import via `gcc -lcuda`. Vertex mounts `/usr/local/nvidia` read-only so we can't symlink there. Entrypoint symlinks the host driver's `libcuda.so` into `/tmp/cudalib` and exports `LIBRARY_PATH` (link time) + `LD_LIBRARY_PATH` (runtime dlopen).

---

## Tips

- **Iterate locally, not via Cloud Build.** Cloud Build ≈ 10 min/cycle. Local Docker + layer cache ≈ 30s for re-pip steps + 2-3 min push. Reserve Cloud Build for the canonical final image only.
- **Build-time asserts catch wrong pins fast.** Every pin you care about gets a `python -c "assert ..."` step. Wrong CUDA fails the BUILD, not a queued job.
- **`restartJobOnWorkerRestart: false` on spot jobs** so a preemption mid-training doesn't double-bill you. Re-submit instead — 30 min is short enough.
- **Polling cadence.** Vertex jobs queue ~3-5 min on spot, run ~30 min. Polling every 5 min is plenty.

---

## License

Apache 2.0 (see `LICENSE`).

Built on Unsloth (Apache 2.0), Hugging Face transformers / peft / trl / accelerate (Apache 2.0), and Google's Gemma 4 weights (Gemma Terms of Use — see <https://ai.google.dev/gemma/terms>).
