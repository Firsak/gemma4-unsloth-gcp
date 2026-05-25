#!/usr/bin/env python3
"""Convert a LoRA-merged Gemma 4 E2B HF safetensors checkpoint to .litertlm.

Wraps the `litert-torch export_hf` CLI (Generative API, CPU-only). The CLI
auto-bundles the produced .tflite + tokenizer + metadata into a single
`.litertlm` when --bundle_litert_lm is set.

Driven by env (set by convert_entrypoint.sh):
  MODEL_DIR   local dir with config.json + model.safetensors + tokenizer + chat_template.jinja
  OUTPUT_DIR  local dir for export_hf output
  QUANT       quantization recipe (default dynamic_wi4_afp32 = int4)
"""
import os
import subprocess
import sys
import glob


def run(cmd: list[str]) -> int:
    print("[convert] $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def main() -> int:
    model_dir = os.environ.get("MODEL_DIR", "/workspace/model")
    output_dir = os.environ.get("OUTPUT_DIR", "/workspace/out")
    quant = os.environ.get("QUANT", "dynamic_wi4_afp32")
    os.makedirs(output_dir, exist_ok=True)

    # litert-torch export_hf:
    #  --model                 local fine-tuned safetensors dir
    #  --quantization_recipe   dynamic_wi4_afp32 == dynamic weight-only int4
    #  --externalize_embedder  REQUIRED for Gemma 4 (embedder split out)
    #  --task text_generation  text-only -> skips vision encoder (no GPU path)
    #  --use_jinja_template    parse chat_template.jinja from the checkpoint
    #  --bundle_litert_lm      emit a .litertlm directly (tflite+tokenizer+meta)
    cmd = [
        "litert-torch", "export_hf",
        "--model", model_dir,
        "--output_dir", output_dir,
        "--quantization_recipe", quant,
        "--externalize_embedder", "true",
        "--task", "text_generation",
        "--use_jinja_template", "true",
        "--bundle_litert_lm", "true",
    ]
    rc = run(cmd)
    if rc != 0:
        print(f"[convert] export_hf failed rc={rc}", flush=True)
        return rc

    produced = sorted(glob.glob(os.path.join(output_dir, "**", "*.litertlm"), recursive=True))
    print(f"[convert] .litertlm produced: {produced}", flush=True)
    if not produced:
        # Surface what we DID get so failures are diagnosable.
        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                p = os.path.join(root, f)
                print(f"[convert]   output file: {p} ({os.path.getsize(p)} bytes)", flush=True)
        print("[convert] ERROR: no .litertlm produced", flush=True)
        return 2

    biggest = max(produced, key=os.path.getsize)
    size = os.path.getsize(biggest)
    print(f"[convert] selected {biggest} ({size/1e9:.2f} GB)", flush=True)
    # Hand the chosen path back to the entrypoint via a sentinel file.
    with open(os.path.join(output_dir, ".litertlm_path"), "w") as fh:
        fh.write(biggest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
