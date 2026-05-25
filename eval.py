"""Eval harness for the Patchnote apply_edits fine-tune.

Runs the held-out eval set through a model, parses the apply_edits JSON the same
way the production Android ApplyEditsParser does, grades against gold, and reports
per-category accuracy + the Step 25 eval gate verdict.

Runnable two ways:
  - Local CPU:   python eval.py --model-path ./models/.../merged_16bit --eval-file ../../dataset/v1/eval.messages.jsonl
  - Vertex job:  via submit_eval_job.sh (GPU, reads/writes GCS)

Grading = exact op-list match (parse both sides, normalize whitespace, compare as
multisets ignoring order). Also reports softer op-type / find-match rates.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


# ---------- apply_edits JSON extraction (mirrors :core:llm ApplyEditsParser) ----------

def extract_first_json_block(text: str) -> str | None:
    """Return the first balanced {...} block, respecting double-quoted strings."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_edits(text: str) -> list[dict] | None:
    """Parse model output → list of op dicts, or None if unparseable."""
    block = extract_first_json_block(text)
    if block is None:
        return None
    # tolerant: strip trailing commas before } or ]
    block = re.sub(r",\s*([}\]])", r"\1", block)
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "edits" not in obj:
        return None
    edits = obj["edits"]
    if not isinstance(edits, list):
        return None
    return [e for e in edits if isinstance(e, dict)]


# ---------- op normalization + grading ----------

def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_op(op: dict) -> tuple:
    """Order-independent comparable key for one edit op. Drops opId-style noise."""
    kind = _norm(op.get("op", ""))
    fields = {k: _norm(v) for k, v in op.items() if k not in ("op", "id", "opId", "source")}
    return (kind, tuple(sorted(fields.items())))


def ops_match(pred: list[dict], gold: list[dict]) -> bool:
    return Counter(normalize_op(o) for o in pred) == Counter(normalize_op(o) for o in gold)


# ---------- heuristic categorization (dataset has no category tags) ----------

def categorize(gold: list[dict], instruction: str) -> list[str]:
    cats = []
    if len(gold) == 0:
        cats.append("off_script")
        return cats
    if any(_norm(o.get("op")) == "insert_at_beginning" for o in gold):
        cats.append("prepend")
    out_lines = [str(o.get("line", "")) + str(o.get("with", "")) for o in gold]
    if any("- [ ]" in l or "- [x]" in l for l in out_lines):
        cats.append("checkbox")
    if any(l.startswith("  ") or "\n  " in l for l in out_lines):
        cats.append("nested")
    inserts = [o for o in gold if _norm(o.get("op")).startswith("insert")]
    if len(inserts) >= 2:
        cats.append("multi_item")
    if not cats:
        cats.append("general")
    return cats


# ---------- model inference ----------

def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    # Decoder-only batched generation requires LEFT padding so the last token of
    # every (right-aligned) prompt sits at the same position for the first step.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    has_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
    )
    model.eval()
    return model, tok


def generate_batch(model, tok, batch_messages: list[list[dict]], max_new_tokens: int) -> list[str]:
    """Batched greedy generation. Returns one decoded completion per input."""
    import torch

    prompts = [
        tok.apply_chat_template(
            [m for m in msgs if m["role"] in ("system", "user")],
            tokenize=False,
            add_generation_prompt=True,
        )
        for msgs in batch_messages
    ]
    # add_special_tokens=False: apply_chat_template already inserted BOS into the
    # string. Re-tokenizing with the default would double-add BOS, which Gemma is
    # very sensitive to → degraded generation (model emits prose instead of JSON).
    enc = tok(
        prompts, return_tensors="pt", padding=True, truncation=True,
        max_length=4096, add_special_tokens=False,
    )
    device = next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy = deterministic eval
            pad_token_id=tok.pad_token_id,
        )
    in_len = enc["input_ids"].shape[1]
    gens = out[:, in_len:]
    return [tok.decode(g, skip_special_tokens=True) for g in gens]


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="local dir or HF id of merged model")
    ap.add_argument("--eval-file", required=True, help="path to eval.messages.jsonl")
    ap.add_argument("--out", default="eval_report.json", help="report output path")
    ap.add_argument("--limit", type=int, default=0, help="only eval first N rows (0=all)")
    ap.add_argument("--batch-size", type=int, default=8, help="gen batch (8 keeps A100 KV cache in budget at 2048 out)")
    ap.add_argument("--max-new-tokens", type=int, default=2048, help="covers multi-minute dictation echoed as JSON")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.eval_file).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Loaded {len(rows)} eval rows", flush=True)

    model, tok = load_model(args.model_path)
    print("Model loaded", flush=True)

    cat_total: Counter = Counter()
    cat_exact: Counter = Counter()
    n_parse_ok = 0
    n_exact = 0
    failures = []

    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        raws = generate_batch(
            model, tok, [r["messages"] for r in chunk], args.max_new_tokens
        )
        for j, (row, raw) in enumerate(zip(chunk, raws)):
            i = start + j
            messages = row["messages"]
            gold_text = next(m["content"] for m in messages if m["role"] == "assistant")
            instruction = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            ).split("INSTRUCTION:")[-1].strip()

            gold = parse_edits(gold_text) or []
            cats = categorize(gold, instruction)
            pred = parse_edits(raw)

            parse_ok = pred is not None
            exact = parse_ok and ops_match(pred, gold)

            n_parse_ok += int(parse_ok)
            n_exact += int(exact)
            for c in cats:
                cat_total[c] += 1
                cat_exact[c] += int(exact)

            if not exact and len(failures) < 30:
                failures.append({
                    "idx": i,
                    "cats": cats,
                    "parse_ok": parse_ok,
                    "gold": gold,
                    "pred": pred,
                    "raw_head": raw[:200],
                })

        done = min(start + args.batch_size, len(rows))
        print(f"  {done}/{len(rows)}  exact={n_exact}  parse_ok={n_parse_ok}", flush=True)

    n = len(rows)
    report = {
        "n": n,
        "parse_ok_rate": round(n_parse_ok / n, 4) if n else 0,
        "exact_match_rate": round(n_exact / n, 4) if n else 0,
        "by_category": {
            c: {
                "n": cat_total[c],
                "exact": cat_exact[c],
                "rate": round(cat_exact[c] / cat_total[c], 4) if cat_total[c] else 0,
            }
            for c in sorted(cat_total)
        },
        "failures_sample": failures,
    }

    # Step 25 eval gate
    def rate(c: str) -> float:
        return report["by_category"].get(c, {}).get("rate", 0.0)

    gate = {
        "multi_item>=0.85": rate("multi_item") >= 0.85,
        "nested>=0.75": rate("nested") >= 0.75,
        "checkbox>=0.75": rate("checkbox") >= 0.75,
        "prepend>=0.75": rate("prepend") >= 0.75,
        "off_script>=0.95": rate("off_script") >= 0.95,
    }
    report["gate"] = gate
    report["gate_pass"] = all(gate.values())

    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n=== EVAL REPORT ===", flush=True)
    print(f"parse_ok_rate    {report['parse_ok_rate']}", flush=True)
    print(f"exact_match_rate {report['exact_match_rate']}", flush=True)
    for c, v in report["by_category"].items():
        print(f"  {c:12s} {v['exact']}/{v['n']} = {v['rate']}", flush=True)
    print(f"GATE: {'PASS' if report['gate_pass'] else 'FAIL'}  {gate}", flush=True)

    # Optional GCS upload if --out is a gs:// path handled by caller; here local only.
    return 0


if __name__ == "__main__":
    sys.exit(main())
