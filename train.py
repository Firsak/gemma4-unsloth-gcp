"""Patchnote LoRA fine-tune on Vertex AI Custom Job.

Reads dataset from GCS, trains, writes adapter to GCS.
Mirrors Unsloth Studio Colab config that produced loss 0.02.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from google.cloud import storage

BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/gemma-4-e2b-it-unsloth-bnb-4bit")
GCS_DATASET = os.environ.get("GCS_DATASET", "gs://your-training-bucket/dataset/v1")
GCS_OUTPUT = os.environ.get("GCS_OUTPUT", "gs://your-training-bucket/runs/latest")
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "4096"))
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "2e-4"))
BATCH = int(os.environ.get("BATCH", "2"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
SEED = int(os.environ.get("SEED", "42"))

LOCAL_DATA = Path("/workspace/data")
LOCAL_OUT = Path("/workspace/out")


def _parse_gcs(uri: str) -> tuple[str, str]:
    assert uri.startswith("gs://"), uri
    rest = uri[5:]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix


def _download(client: storage.Client, gcs_uri: str, local: Path) -> None:
    bucket_name, blob_name = _parse_gcs(gcs_uri)
    print(f"GET {gcs_uri} -> {local}", flush=True)
    client.bucket(bucket_name).blob(blob_name).download_to_filename(str(local))


def _upload_dir(client: storage.Client, local_dir: Path, gcs_prefix: str) -> None:
    bucket_name, base_prefix = _parse_gcs(gcs_prefix)
    bucket = client.bucket(bucket_name)
    for path in local_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_dir).as_posix()
        blob_name = f"{base_prefix}/{rel}" if base_prefix else rel
        print(f"PUT {path} -> gs://{bucket_name}/{blob_name}", flush=True)
        bucket.blob(blob_name).upload_from_filename(str(path))


def pull_dataset() -> tuple[Path, Path]:
    LOCAL_DATA.mkdir(parents=True, exist_ok=True)
    client = storage.Client()
    train_local = LOCAL_DATA / "train.messages.jsonl"
    eval_local = LOCAL_DATA / "eval.messages.jsonl"
    _download(client, f"{GCS_DATASET}/train.messages.jsonl", train_local)
    _download(client, f"{GCS_DATASET}/eval.messages.jsonl", eval_local)
    return train_local, eval_local


def push_output() -> None:
    client = storage.Client()
    _upload_dir(client, LOCAL_OUT, GCS_OUTPUT)


def main() -> None:
    print(f"BASE_MODEL={BASE_MODEL}", flush=True)
    print(f"GCS_DATASET={GCS_DATASET}", flush=True)
    print(f"GCS_OUTPUT={GCS_OUTPUT}", flush=True)

    train_path, eval_path = pull_dataset()
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import torch
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
    )

    def fmt(row: dict) -> dict:
        msgs = row["messages"]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    ds_train = load_dataset("json", data_files=str(train_path), split="train").map(fmt)
    ds_eval = load_dataset("json", data_files=str(eval_path), split="train").map(fmt)
    print(f"train={len(ds_train)} eval={len(ds_eval)}", flush=True)

    cfg = SFTConfig(
        output_dir=str(LOCAL_OUT / "checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=SEED,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        args=cfg,
    )

    trainer.train()

    adapter_dir = LOCAL_OUT / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    merged_dir = LOCAL_OUT / "merged_16bit"
    model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")

    train_loss = next(
        (h["train_loss"] for h in reversed(trainer.state.log_history) if "train_loss" in h),
        next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), -1.0),
    )
    metrics = {
        "final_loss": float(train_loss),
        "train_rows": len(ds_train),
        "eval_rows": len(ds_eval),
        "epochs": EPOCHS,
        "lr": LR,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "base_model": BASE_MODEL,
    }
    (LOCAL_OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    push_output()
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
