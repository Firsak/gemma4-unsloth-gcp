# Iteration log — Gemma 4 QLoRA on Vertex AI

Chronological record of every failure and fix while getting Unsloth QLoRA fine-tuning of **Gemma 4 E2B** running on Google Cloud's Vertex AI Custom Training service (A100 spot, us-central1).

Dates: 2026-05-18 → 2026-05-20.

---

## Goal

Fine-tune `unsloth/gemma-4-e2b-it-unsloth-bnb-4bit` with LoRA (r=16, α=32, dropout 0.05, 7 target modules) on a chat-format JSONL dataset → LoRA adapter + merged fp16 weights → downstream conversion to LiteRT for on-device deploy.

## Why Vertex AI (not Colab)

Tried Colab first. The Studio "Merged 16-bit" export crashed; the free-tier GPU quota ran out. Pivoted to Vertex AI Custom Jobs for repeatable, monitorable, pay-per-second training that I could trigger and forget about.

---

## Infrastructure setup (one-time, ~30 min)

| Item | Value |
|---|---|
| GCP project | billing enabled |
| APIs enabled | compute, aiplatform, artifactregistry, cloudbuild |
| GCS bucket (dataset + artifacts) | one bucket |
| GCS bucket (build logs) | optional, for Cloud Build |
| Artifact Registry | one Docker repo |
| Compute SA roles | `cloudbuild.builds.builder`, `artifactregistry.writer`, `logging.logWriter`, `storage.admin` |
| Dataset | `train.messages.jsonl` + `eval.messages.jsonl` (OpenAI-style messages) in GCS |

### Quota discovery (us-central1, Vertex AI training GPUs)

| GPU | Spot quota | Decision |
|---|---|---|
| L4 | **0** | blocked |
| T4 | 1 | fallback (slow) |
| **A100** | **8** | **chosen** — no quota request needed, ~$1.20/hr, ~30 min run ≈ $0.60 |

New GCP projects have **no default Cloud Build SA** — you must grant the Compute SA `cloudbuild.builds.builder` + supply `serviceAccount:` + `logsBucket:` + `logging: GCS_ONLY` in `cloudbuild.yaml`.

---

## The failure chain (15 build/runtime fixes + 1 upstream bug)

Each row = one job submission or build that failed, the root cause, and the fix. Each fix permanently resolved that layer and surfaced the next.

### 1. `gsutil` missing in container
- **Symptom:** `FileNotFoundError: [Errno 2] No such file or directory: 'gsutil'`
- **Cause:** `train.py` used `subprocess.run(["gsutil", ...])` to pull the dataset. Base image `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` has no gcloud SDK.
- **Fix:** Rewrote dataset I/O to use the `google-cloud-storage` Python client (`storage.Client().bucket(...).blob(...).download_to_filename(...)`).
- **Lesson:** Never depend on the gcloud CLI in a custom training container — the Python client is one pip package and well-isolated.

### 2. `unsloth_zoo` missing
- **Symptom:** `ImportError: Please install unsloth_zoo via 'pip install unsloth_zoo'`
- **Cause:** Installing only `unsloth` from git head pulled headers but not the sibling `unsloth_zoo` package.
- **Fix:** Added explicit `pip install unsloth_zoo`.
- **Lesson:** Unsloth splits zoo into a separate package; install both.

### 3. `torch._inductor.config` missing
- **Symptom:** `AttributeError: module 'torch._inductor' has no attribute 'config'`
- **Cause:** unsloth_zoo head needs `torch._inductor.config` (torch ≥ 2.5); we had torch 2.4.1.
- **Fix:** Bumped torch 2.4.1 → 2.5.1.
- **Lesson:** Unsloth pulls bleeding-edge torch internals; match its minimum.

### 4. CUDA driver too old (`found version 12020`)
- **Symptom:** `CUDA initialization: The NVIDIA driver on your system is too old (found version 12020)`
- **Cause:** Vertex A100 spot VM driver supports max CUDA toolkit **12.0** (12020 = 12.0.20). torch 2.5.1+**cu121** needs driver ≥ 12.1.
- **Fix:** Switched to torch+**cu118** wheels (CUDA 11.8 ≤ 12.0, runs on the older driver).
- **Lesson:** Vertex prebuilt VM drivers lag torch's default CUDA target by 1–2 versions. The number in torch's error IS the CUDA toolkit cap. Always pick CUDA ≤ what the VM driver supports.

### 5. cu118 pin didn't stick
- **Symptom:** Same "driver too old" error despite asking for cu118.
- **Cause:** The `unsloth[cu121-ampere-torch241] @ git+...` extras spec re-installed cu121 torch from PyPI default index, overriding the earlier `--index-url cu118` install.
- **Fix:** Bare `unsloth` (no extras) + pin `torch==2.7.1+cu118` with explicit `+cu118` suffix + `--force-reinstall --no-deps` torch at the END of the Dockerfile + a build-time assert `python -c "import torch; assert torch.version.cuda == '11.8'"`.
- **Lesson:** pip silently overrides an earlier explicit install when a later package constrains the same dep. Force-reinstall critical pins last; assert them at build time so wrong CUDA fails the BUILD (~30s), not the JOB (~5-10 min queue + provision).

### 6. `PIL` missing
- **Symptom:** `ModuleNotFoundError: No module named 'PIL'`
- **Cause:** `--no-deps` on unsloth (used to stop torch override) also stripped Pillow, which `unsloth_zoo.vision_utils` imports.
- **Fix:** Dropped `--no-deps`; installed pillow explicitly; force-reinstalled torch at the end instead.
- **Lesson:** `--no-deps` is a hammer that cuts legit runtime deps too. Surgical alternative: install with deps, then force-reinstall the one package you want pinned.

### 7. `torch.int1` missing
- **Symptom:** `AttributeError: module 'torch' has no attribute 'int1'`
- **Cause:** `torchao` (pulled transitively by `transformers`) references `torch.int1` dtype, which exists in torch ≥ 2.6; we had 2.5.1.
- **Fix:** Bumped torch 2.5.1 → **2.7.1+cu118** AND pinned `torchao==0.15.0` (last version supporting torch 2.7; 0.16+ needs torch 2.8).
- **Lesson:** torchao is now a transitive dep of recent transformers; pin it to a torch-compatible version BEFORE installing transformers.

### 8. `flash-attn` build fails
- **Symptom:** `ModuleNotFoundError: No module named 'torch'` while building flash-attn wheel.
- **Cause:** Unsloth's `cu118-ampere-torch270` extras include flash-attn, which ships only an sdist (no prebuilt wheel for this combo). pip's PEP 517 isolated build env doesn't see the host torch.
- **Fix:** Dropped the ampere extras; installed bare unsloth. flash-attn is acceleration, not a requirement (~30-40% slower training without it).
- **Lesson:** PEP 517 build isolation hides the host torch from sdist builds. `--no-build-isolation` is the alternative; dropping the extra was simpler.

### 9. unsloth_zoo refuses CPU import at BUILD time
- **Symptom:** `NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.` during `docker build`.
- **Cause:** `import unsloth_zoo` runs GPU detection at module import; the build host has no GPU.
- **Fix:** Relaxed the build-time assert to import only `torch + transformers + torchao + bitsandbytes` (all CPU-importable); deferred unsloth_zoo import to runtime.
- **Lesson:** Build host has no GPU — don't import GPU-required libs in build asserts.

### 10. `gemma4` model_type unknown to transformers
- **Symptom:** `KeyError: 'gemma4'` / `ValueError: ... not supported yet in transformers==4.51.3`
- **Cause:** Gemma 4 architecture lands in transformers 4.56+; we had 4.51.3.
- **Fix progression:** tried `>=4.58` (not released) → `>=4.56,<4.57` (still didn't recognise gemma4 per Unsloth's check) → **transformers from git head** (resolved to 5.8.1, has gemma4).
- **Lesson:** Unsloth's PyPI exclusion list documents real conflict versions; for a brand-new model arch, git-head transformers may be the only option.

### 11. Unsloth `install.sh` canonical install + wheel auto-detect trap
- **Discovery:** `curl https://unsloth.ai/install.sh` shows the canonical path: `uv pip install unsloth-zoo unsloth --torch-backend=auto`. Its `get_torch_index_url()` maps a CUDA-12.0 driver to **cu124** wheels (`elif major>=12 → cu124`) — which would fail on Vertex's 12.0 driver.
- **Fix:** Pre-install torch+cu118 explicitly BEFORE unsloth so `--torch-backend=auto` preserves it instead of pulling cu124.
- **Lesson:** Unsloth's auto-detector assumes the driver supports its detected CUDA version; on capped drivers, pin the wheel yourself.

### 12. xformers / flash-attn no wheel for torch 2.7.1+cu118
- **Symptom:** uv resolver: `unsloth>=2026.5.3 cannot be used` because it depends on `xformers>=0.0.27.post2` (no matching cu118+torch2.7 wheel).
- **Fix:** `uv pip install --no-deps "unsloth>=2026.5.3" "unsloth-zoo>=2026.5.2"` + handpick all runtime deps explicitly.
- **Lesson:** When an optional accelerator (xformers) blocks resolution and you don't need it, `--no-deps` + a manual dep list is the escape.

### 13. peft < 0.18 imports removed `HybridCache`
- **Symptom:** `ImportError: cannot import name 'HybridCache' from 'transformers'`
- **Cause:** transformers 5.x removed `HybridCache`; peft < 0.18 still imports it.
- **Fix:** Use peft ≥ 0.18 (resolved to 0.19.1).
- **Lesson:** Pinning an old peft to dodge one problem creates an import-incompat with new transformers. Move forward, not back.

### 14. peft 0.19 rejects torchao 0.15
- **Symptom:** `ImportError: Found an incompatible version of torchao. Found version 0.15.0, but only versions above 0.16.0 are supported`
- **Cause:** peft 0.19's `is_torchao_available()` hard-requires torchao ≥ 0.16; but torchao 0.16+ needs torch 2.8 (no cu118 wheel exists). Deadlock.
- **Fix:** **Uninstall torchao entirely** + `sed`-patch transformers' `quantizers/auto.py` to drop its `from .quantizer_torchao import TorchAoHfQuantizer` line. With torchao absent, peft's check returns False and skips. Our LoRA uses bnb-4bit, never torchao.
- **Lesson:** When two libs disagree on an optional dep's version and neither version fits your constraints, removing the optional dep (so both skip it) beats satisfying either.

### 15. Triton autotuner compile fails at import — read-only nvidia mount
- **Symptom:** `subprocess.CalledProcessError: gcc ... -lcuda -L/usr/local/nvidia/lib64 ...` then `ln: failed to create symbolic link '/usr/local/nvidia/lib64/libcuda.so': Read-only file system`.
- **Cause:** `import bitsandbytes.triton.*` (pulled via unsloth) triggers Triton's autotuner at IMPORT time → it JIT-compiles `cuda_utils.so` via `gcc -lcuda -L/usr/local/nvidia/lib64`. That dir doesn't have `libcuda.so` and Vertex mounts `/usr/local/nvidia` **read-only** so we can't symlink there.
- **Fix:** Added `entrypoint.sh` that, at container start, symlinks the host driver's `libcuda.so` into a **writable** `/tmp/cudalib` and exports `LIBRARY_PATH` (gcc link-time) + `LD_LIBRARY_PATH` (runtime dlopen).
- **Lesson:** Triton compiles native CUDA shims at import; needs gcc + libcuda + headers all aligned. Vertex mounts the driver read-only — shim into a writable dir + point the linker env at it.

### 15a. Wrong libcuda picked (cu118 compat shim)
- **Symptom:** `Error 803: system has unsupported display driver / cuda driver combination`; entrypoint logged `libcuda.so source: /usr/local/cuda/compat/libcuda.so.1`.
- **Cause:** Entrypoint found the container's CUDA 11.8 **forward-compat** libcuda first, which conflicts with the Vertex host driver.
- **Fix:** Reordered the search to prefer the host driver `/usr/local/nvidia/lib64/libcuda.so.1` over the compat shim.
- **Lesson:** Prefer the host driver's libcuda over any in-container compat shim.

### 15b. `Python.h` missing for Triton compile
- **Symptom:** `fatal error: Python.h: No such file or directory`
- **Cause:** Triton's gcc compile needs Python dev headers; base image had `python3.10` but not `python3.10-dev`.
- **Fix:** Added `python3.10-dev build-essential gcc` to the apt install.
- **Lesson:** Any runtime JIT-compile (Triton) needs the dev headers + a full toolchain in the image.

### 16. UPSTREAM BUG — `linear_cross_entropy` undefined
- **Symptom:** Training reached step 0, ran ~33s, then `NameError: name 'linear_cross_entropy' is not defined. Did you mean: 'fast_linear_cross_entropy'?` at `unsloth_zoo/loss_utils.py:199` in `fused_linear_cross_entropy`, via the Gemma 4 path (`gemma4_moe.py` → `unsloth_compiled_module_gemma4.py`).
- **Cause:** `loss_utils.py` imports `linear_cross_entropy` from the **optional** `cut_cross_entropy` package inside a guarded `try/except` (sets `HAS_CUT_CROSS_ENTROPY`). But `fused_linear_cross_entropy` calls `linear_cross_entropy(...)` **without** checking that flag. When `cut_cross_entropy` isn't installed (it's an optional dep with only a Python ≥ 3.10 marker, not force-resolved), the name is undefined → NameError at first loss computation. Bug present on unsloth_zoo `main` as of 2026.5.3 — **no released version fixes it**.
- **Fix:** `pip install cut-cross-entropy` so the guarded import succeeds and `linear_cross_entropy` is actually defined. All its prereqs (Python 3.10, torch 2.4+, Triton 3.0+, Ampere A100) are met. Alternative: set `UNSLOTH_COMPILE_DISABLE=1` before `import unsloth` to bypass the compiled forward that calls the fused loss.
- **Lesson:** A library's "optional" dep is only truly optional if every call site checks the guard flag. When the import is guarded but a caller forgets to gate, the package becomes mandatory in practice. Read upstream source when an error names something you didn't think was load-bearing.

---

## Final working dependency stack

```
Base:          nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
Python:        3.10 (+ python3.10-dev, build-essential, gcc)
torch:         2.7.1+cu118 (+ torchvision 0.22.1, torchaudio 2.7.1, all cu118)
transformers:  5.8.1 (git head — for gemma4 model_type)
peft:          0.19.1
trl:           0.24.0
unsloth:       2026.5.5
unsloth_zoo:   2026.5.2  (+ cut-cross-entropy required, see #16)
triton:        3.3.0     (NOT 3.6 — torch 2.7 inductor needs the older triton_key API)
bitsandbytes:  0.45.5+
torchao:       REMOVED   (uninstalled + transformers quantizer import sed-patched out)
Extras:        sentencepiece, protobuf, pillow, hf_transfer, google-cloud-storage, msgspec, tyro
Entrypoint:    entrypoint.sh shims libcuda.so → /tmp/cudalib + sets LIBRARY_PATH / LD_LIBRARY_PATH
```

Vertex job spec: `a2-highgpu-1g` (1× A100-SXM4-40GB), `NVIDIA_TESLA_A100`, `pd-ssd` 200 GB boot, `scheduling: { strategy: SPOT, restartJobOnWorkerRestart: false }`.

---

## Key process wins

1. **Local Docker > Cloud Build for iteration.** Cloud Build cycle was ~9-11 min each. Local build with layer cache is ~30-60s for re-pip steps; tag + push to AR ~2-3 min. Reserve Cloud Build for the final canonical image only.
2. **Build-time asserts catch wrong pins in ~30s** instead of after a ~5-10 min Vertex queue + provision. Every pin you care about gets a `python -c "assert ..."` step.
3. **A research escape hatch.** After 5+ trial-and-error builds, I delegated a focused research task with the exact failure chain + constraints to an assistant. It returned a definitive dep matrix (cu118 wheel availability, torchao/torch coupling, transformers exclusion list) in ~5 min — far faster than continued guessing. Used again to diagnose bug #16 against unsloth-zoo source.
4. **Poll, don't watch.** Set an auto-ping on job state every 5 min (PENDING) / 10 min (RUNNING) instead of staring at the console. Caught each failure without manual checking.

---

## Job submission history

| Job ID (tail) | Outcome | Died on |
|---|---|---|
| ...140679168 | FAILED | gsutil missing (#1) |
| ...530714112 | FAILED | unsloth_zoo missing (#2) |
| ...918784000 | FAILED | torch._inductor (#3) |
| ...093203968 | FAILED | CUDA driver / cu118 stick (#4, #5) |
| ...043226624 | FAILED | PIL (#6) |
| ...340279808 | FAILED | torch.int1 (#7) |
| ...714112 (rebuilds) | FAILED | flash-attn, gemma4, etc. (#8-#10) |
| ...205114142720 | FAILED | torchao / triton import (#14, #15) |
| ...284087234560 | FAILED | read-only nvidia mount (#15) |
| ...617026105344 | FAILED | wrong libcuda + Python.h (#15a, #15b) |
| ...798436515840 | FAILED | **upstream bug #16** — trained to step 0, then `linear_cross_entropy` NameError |
| **...5710214390750380032** | **SUCCEEDED** | added `cut-cross-entropy` (#16) |

---

## ✅ SUCCESS

After 16 fixes and 11 failed jobs, the run that worked:

- 3 epochs on the training set
- ~37 minutes on A100-40GB spot, ~$0.75
- **train_loss 0.067** (final step loss ~0.02), clean cosine descent 0.51 → 0.02
- Output: LoRA adapter (~114 MB) + merged fp16 weights + checkpoints + metrics.json

The whole chain was dependency / environment archaeology. The actual training was textbook once the stack aligned.
