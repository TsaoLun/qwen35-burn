# qwen35-burn

Qwen3.5-0.8B inference in Rust using the [Burn](https://github.com/tracel-ai/burn) deep learning
framework with a WGPU/Metal backend.  Supports the full hybrid architecture
(full softmax attention + GatedDeltaNet linear-attention layers), streaming
token output, and optional chain-of-thought (thinking) mode.

---

## Requirements

- Rust (stable, 1.75+)
- macOS with a WGPU/Metal-capable GPU (tested on Apple M3)
- Python 3.9+ (for model download and benchmarks)

---

## Quick start

### 1 — Download the model

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-0.8B', local_dir='./models/Qwen3.5-0.8B',
    allow_patterns=['*.safetensors', 'config.json', 'tokenizer.json'])
"
```

### 2 — Chat (streaming, non-thinking mode)

```bash
cargo run --release --features wgpu --example chat -- \
  --model-dir ./models/Qwen3.5-0.8B \
  --prompt "Explain quicksort in one sentence"
```

### 3 — Chat with thinking mode

Appends `<think>\n` to the prompt so the model emits its chain-of-thought
before the final answer.

```bash
cargo run --release --features wgpu --example chat -- \
  --model-dir ./models/Qwen3.5-0.8B \
  --thinking \
  --prompt "Explain quicksort in one sentence"
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | *(required)* | Path containing `config.json` + `.safetensors` |
| `--tokenizer` | `<model-dir>/tokenizer.json` | Custom tokenizer path |
| `--max-seq-len` | `4096` | RoPE / KV-cache capacity |
| `--max-new-tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature (0 = greedy) |
| `--top-p` | `0.9` | Nucleus sampling p |
| `--system` | `"You are a helpful assistant."` | System prompt |
| `--prompt` | — | Single prompt (non-interactive) |
| `--thinking` | off | Enable chain-of-thought mode |
| `--backend` | `wgpu` | `wgpu` or `ndarray` |

---


## TODO
- recurrent forward parallel optimization
- multi-turn dialogue with KV-cache management
- attn_output_gate, mlp_only_layers never used/implemented
- extract hardcoded kernel_size=4 in conv1d_prefill as a param
- ineffective KV cache eviction strategy
- multimodal support

## Benchmarks

The `bench/` directory compares throughput between **burn-wgpu** (this project)
and **mlx-lm** on Apple Silicon.

### Metrics

| Metric | Definition |
|--------|-----------|
| **TTFT** | Time to first output token (prefill + first decode step) |
| **Prefill tok/s** | Prompt tokens processed per second (≈ prompt\_tokens / TTFT) |
| **Decode tok/s** | Autoregressive decode throughput after the first token |

### Quick run

```bash
# 1. Install mlx-lm
pip install mlx-lm

# 2. Run both benchmarks and print table
bash bench/run_all.sh
```

### Individual scripts

```bash
# Burn (WGPU/Metal, fusion + autotune)
bash bench/run_burn.sh

# MLX
python3 bench/run_mlx.py

# Print table from existing results
python3 bench/compare.py
python3 bench/compare.py --csv    # CSV output
```

Environment variables accepted by the scripts:

```
MODEL_DIR       path to model dir    (default: ./models/Qwen3.5-0.8B)
DECODE_TOKENS   tokens per run       (default: 100)
WARMUP          warmup runs          (default: 5)
RUNS            measured runs        (default: 5)
```

### Sample results (Apple M3, fp32, Qwen3.5-0.8B, 66-token prompt → 100 decode tokens)

> Run `bash bench/run_all.sh` to reproduce on your machine.

| Framework | Model | Backend | Prefill (tok/s) | Decode (tok/s) | TTFT (ms) |
|-----------|-------|---------|----------------:|---------------:|----------:|
| **burn-wgpu** (this project) | Qwen3.5-0.8B | WGPU/Metal (fusion+autotune) | 82.1 | 12.3 | 804 |
| mlx-lm | Qwen3.5-0.8B | MLX/Metal | 522.3 | 33.3 | 131 |

The decode gap vs MLX is mainly caused by the serial GatedDeltaNet
`recurrent_forward` loop (per-token kernel dispatch, no parallel scan yet).

---

## Architecture notes

- **Hybrid layers**: alternates between full softmax attention and GatedDeltaNet
  (delta-rule linear attention), as specified by `layer_types` in `config.json`.
- **Partial RoPE**: rotary embeddings applied to the first 25 % of the head
  dimension (`partial_rotary_factor = 0.25`).
- **Streaming output**: each token is decoded and printed as it is generated.
- **Think filter**: in non-thinking mode the `<think>…</think>` block is
  stripped from the stream in real time before printing.

---

## License

Apache-2.0 OR MIT