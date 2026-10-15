"""Day 17 solution: profile prefill and cached decoding as separate phases."""

from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
RESULTS_DIR = Path(__file__).parent / "results"
PROMPT_TOKENS, DECODE_STEPS, WARMUP = 256, 16, 3
SORT_KEY = "self_cuda_time_total" if DEVICE == "cuda" else "self_cpu_time_total"


def sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def make_prompt(tokenizer) -> torch.Tensor:
    ids = tokenizer("Profiling separates prefill from decoding. " * 64)["input_ids"][:PROMPT_TOKENS]
    return torch.tensor([ids], device=DEVICE)


def run_prefill(model, prompt_ids: torch.Tensor):
    out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids), use_cache=True, logits_to_keep=1)
    return out.logits[:, -1].argmax(dim=-1, keepdim=True), out.past_key_values


def run_decode(model, first_token: torch.Tensor, cache, steps: int) -> None:
    token = first_token
    for _ in range(steps):
        mask = torch.ones(1, cache.get_seq_length() + 1, dtype=torch.long, device=DEVICE)
        out = model(input_ids=token, attention_mask=mask, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)


def profile_phase(label: str, fn):
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE == "cuda" else [])
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function(label):
            fn()
            sync()
    return prof


def top_table(prof, rows: int = 15) -> str:
    return prof.key_averages(group_by_input_shape=True).table(sort_by=SORT_KEY, row_limit=rows)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE).eval()
    prompt_ids = make_prompt(tokenizer)
    RESULTS_DIR.mkdir(exist_ok=True)

    with torch.inference_mode():
        for _ in range(WARMUP):
            token, cache = run_prefill(model, prompt_ids)
            run_decode(model, token, cache, DECODE_STEPS)
        sync()

        prefill_prof = profile_phase("prefill", lambda: run_prefill(model, prompt_ids))
        token, cache = run_prefill(model, prompt_ids)
        decode_prof = profile_phase("decode", lambda: run_decode(model, token, cache, DECODE_STEPS))

    for label, prof in (("prefill", prefill_prof), ("decode", decode_prof)):
        table = top_table(prof)
        print(f"\n===== {label} ({DEVICE}, {DTYPE}) =====\n{table}")
        (RESULTS_DIR / f"{label}_top_ops.txt").write_text(table)
        prof.export_chrome_trace(str(RESULTS_DIR / f"{label}.pt.trace.json"))
    print(f"\nTraces and tables saved in {RESULTS_DIR}. Open the traces at https://ui.perfetto.dev")
    if DEVICE != "cuda":
        print("CPU only: no GPU activity was profiled.")


if __name__ == "__main__":
    main()
