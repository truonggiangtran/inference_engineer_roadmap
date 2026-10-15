"""Day 9 solution: RoPE, grouped-query attention, and KV cache size."""

import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

MODEL_ID = "Qwen/Qwen3-0.6B"
RESULTS_DIR = Path(__file__).parent / "results"
B, HQ, HKV, T, D = 1, 4, 2, 6, 8


def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), inv_freq)  # [T, D/2]
    emb = torch.cat([angles, angles], dim=-1)  # [T, D]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, kv_heads, seq, head_dim = kv.shape
    return kv[:, :, None].expand(batch, kv_heads, n_rep, seq, head_dim).reshape(batch, kv_heads * n_rep, seq, head_dim)


def kv_cache_elements(layers: int, batch: int, tokens: int, kv_heads: int, head_dim: int) -> int:
    return 2 * layers * batch * tokens * kv_heads * head_dim


def main() -> None:
    torch.manual_seed(0)
    cos, sin = build_rope_cache(T, D)
    assert cos.shape == sin.shape == (T, D)

    x = torch.randn(B, HQ, T, D)
    rotated = apply_rope(x, cos, sin)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(rotated[..., 0, :], x[..., 0, :], msg="position 0 must be unchanged")

    q_vec, k_vec = torch.randn(D), torch.randn(D)
    long_cos, long_sin = build_rope_cache(32, D)
    def score(m: int, n: int) -> torch.Tensor:
        return apply_rope(q_vec, long_cos[m], long_sin[m]) @ apply_rope(k_vec, long_cos[n], long_sin[n])
    torch.testing.assert_close(score(5, 2), score(25, 22))
    print(f"score(5,2)={score(5, 2):.5f} == score(25,22)={score(25, 22):.5f}: depends only on m - n")

    k = torch.randn(B, HKV, T, D)
    hf_q, hf_k = apply_rotary_pos_emb(x, k, cos[None], sin[None])
    torch.testing.assert_close(rotated, hf_q)
    torch.testing.assert_close(apply_rope(k, cos, sin), hf_k)
    print("matches transformers' Qwen3 apply_rotary_pos_emb")

    n_rep = HQ // HKV
    expanded = repeat_kv(k, n_rep)
    assert expanded.shape == (B, HQ, T, D)
    for h in range(HQ):
        torch.testing.assert_close(expanded[:, h], k[:, h // n_rep])
    print(f"GQA: {HQ} query heads share {HKV} KV heads ({n_rep} query heads per KV head)")

    config = AutoConfig.from_pretrained(MODEL_ID)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    tokens = 128
    correct = kv_cache_elements(config.num_hidden_layers, 1, tokens, config.num_key_value_heads, head_dim)
    wrong = kv_cache_elements(config.num_hidden_layers, 1, tokens, config.num_attention_heads, head_dim)
    assert correct == 2 * config.num_hidden_layers * tokens * config.num_key_value_heads * head_dim
    report = {
        "tokens": tokens,
        "layers": config.num_hidden_layers,
        "kv_heads": config.num_key_value_heads,
        "query_heads": config.num_attention_heads,
        "head_dim": head_dim,
        "rope_theta": getattr(config, "rope_theta", None),
        "kv_elements": correct,
        "kv_bytes_bf16": correct * 2,
        "overestimate_with_query_heads": wrong / correct,
    }
    print(json.dumps(report, indent=2))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "rope_gqa.json").write_text(json.dumps(report, indent=2))
    print("All Day 9 checks passed.")


if __name__ == "__main__":
    main()
