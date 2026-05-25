FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev python3-pip git curl ca-certificates \
    gcc build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python -m pip install --upgrade pip wheel setuptools uv

# Vertex A100 spot VM driver supports up to CUDA 12.0 → torch+cu118.
# Unsloth's auto-detect would pick cu124 from a 12.0 driver which would fail
# at runtime ("driver too old"), so install torch explicitly first to lock cu118.
RUN uv pip install --system \
    torch==2.7.1+cu118 \
    torchvision==0.22.1+cu118 \
    torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Unsloth runtime deps first (handpicked from unsloth's pyproject.toml minus xformers/flash-attn
# which require torch-ABI-matched wheels not available for torch 2.7.1+cu118).
RUN uv pip install --system \
    "transformers>=4.56.0" \
    "peft>=0.18.0" \
    "trl>=0.18.2,<=0.24.0,!=0.19.0" \
    "datasets>=3.4.1,!=4.0.0,!=4.0.1,!=4.1.0,<4.4.0" \
    "accelerate>=0.34.1" \
    "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0" \
    "triton==3.3.0" \
    sentencepiece protobuf pillow hf_transfer google-cloud-storage \
    msgspec tyro

# Unsloth + unsloth-zoo with --no-deps to skip xformers + flash-attn (no cu118+torch271 wheels).
RUN uv pip install --system --no-deps \
    "unsloth-zoo>=2026.5.2" "unsloth>=2026.5.3"

# cut-cross-entropy is an OPTIONAL unsloth_zoo dep that's not force-resolved. unsloth_zoo's
# loss_utils.py imports linear_cross_entropy from it under try/except but calls it unguarded
# → NameError at training step 0 when absent. Install it explicitly. (Bug #16, see ITERATION-LOG.md)
RUN uv pip install --system cut-cross-entropy

# Patch out transformers torchao quantizer import — torchao would be pulled by transformers,
# triggers triton autotuner at import → gcc -lcuda fails on Vertex VM (libcuda path differs).
RUN pip uninstall -y torchao && \
    sed -i '/from \.quantizer_torchao/d' /usr/local/lib/python3.10/dist-packages/transformers/quantizers/auto.py && \
    sed -i '/TorchAoHfQuantizer/d' /usr/local/lib/python3.10/dist-packages/transformers/quantizers/auto.py

# Force-reinstall torch cu118 — earlier pip steps pulled cu126 wheel via uv default index.
RUN uv pip install --system --reinstall --no-deps \
    torch==2.7.1+cu118 \
    torchvision==0.22.1+cu118 \
    torchaudio==2.7.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Sanity: confirm cu118 still pinned + key modules importable on CPU build host
RUN python -c "import torch; assert torch.version.cuda == '11.8', torch.version.cuda; import transformers; import peft; import trl; print(f'TORCH={torch.__version__} TX={transformers.__version__} PEFT={peft.__version__} TRL={trl.__version__}')"

WORKDIR /workspace
COPY train.py /workspace/train.py
COPY eval.py /workspace/eval.py
COPY entrypoint.sh /workspace/entrypoint.sh
COPY eval_entrypoint.sh /workspace/eval_entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh /workspace/eval_entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
