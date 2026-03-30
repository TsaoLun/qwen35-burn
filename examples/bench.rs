/// Inference benchmark for Qwen3.5-0.8B.
///
/// Measures:
///   - TTFT   (time-to-first-token, ms)  = prefill + first decode step
///   - Prefill throughput  (tok/s)  = prompt_tokens / TTFT
///   - Decode throughput   (tok/s)  = remaining_tokens / remaining_time
///
/// Results are written as JSON so `bench/compare.py` can aggregate them.
use std::time::Instant;

use clap::Parser;
use serde_json::json;

#[cfg(feature = "ndarray")]
use burn::backend::NdArray;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;

use qwen35_burn::{Qwen35, Qwen35Tokenizer, Sampler};

// A prompt long enough to give a stable prefill measurement (~100 tokens).
const DEFAULT_PROMPT: &str = concat!(
    "Explain the following algorithms and data structures in detail, covering ",
    "time complexity, space complexity, and practical use cases: quicksort, ",
    "mergesort, binary search trees, hash tables, dynamic programming, BFS/DFS ",
    "graph traversal, and heap sort. For each, give a concrete code example.",
);

#[derive(Parser, Debug)]
#[command(name = "bench", about = "Qwen3.5 inference throughput benchmark")]
struct Args {
    /// Path to model directory (config.json + .safetensors files)
    #[arg(long)]
    model_dir: String,

    /// Path to tokenizer.json (defaults to <model_dir>/tokenizer.json)
    #[arg(long)]
    tokenizer: Option<String>,

    /// Maximum context length
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Number of tokens to decode per run
    #[arg(long, default_value_t = 100)]
    decode_tokens: usize,

    /// Warmup runs (discarded — lets Metal/WGPU compile shaders)
    #[arg(long, default_value_t = 1)]
    warmup: usize,

    /// Measured runs (averaged)
    #[arg(long, default_value_t = 3)]
    runs: usize,

    /// Backend: "wgpu" or "ndarray"
    #[arg(long, default_value = "wgpu")]
    backend: String,

    /// Write JSON results to this path (prints to stdout if omitted)
    #[arg(long)]
    output: Option<String>,

    /// Override the benchmark prompt
    #[arg(long)]
    prompt: Option<String>,

    /// Sampling temperature (0 = greedy).  Default 0.7 matches the chat example
    /// and exercises the full top-p path — including the GPU→CPU logit transfer
    /// that occurs on every decode step when temperature > 0.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,
}

fn run<B: burn::prelude::Backend>(args: &Args) {
    let tok_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| format!("{}/tokenizer.json", args.model_dir));
    let tokenizer = Qwen35Tokenizer::new(&tok_path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {}", e));

    let device = B::Device::default();
    eprintln!("Loading model from {} ...", args.model_dir);
    let mut model = Qwen35::<B>::from_pretrained(&args.model_dir, args.max_seq_len, &device)
        .unwrap_or_else(|e| panic!("Failed to load model: {}", e));

    let prompt_text = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);
    // Use the same ChatML template as the chat example for consistency.
    let formatted = format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt_text
    );
    let prompt_tokens = tokenizer.encode(&formatted).len();
    eprintln!("Prompt length: {} tokens", prompt_tokens);

    let mut sampler = Sampler::new_top_p(0.9, 42);

    // ── Warmup ────────────────────────────────────────────────────────────────
    // Warmup uses greedy (temp=0): the GPU shaders are identical regardless of
    // temperature — the sampling path difference is purely CPU-side.
    eprintln!(
        "Warming up ({} run(s) — compiling Metal/WGPU shaders)...",
        args.warmup
    );
    for i in 0..args.warmup {
        let _ = model.generate(
            tokenizer.inner(),
            &formatted,
            32,
            0.0, // greedy — shader compilation only
            &mut sampler,
            &mut |_| {},
        );
        eprintln!("  warmup {}/{} done", i + 1, args.warmup);
    }

    // ── Benchmark ─────────────────────────────────────────────────────────────
    let mut ttft_ms_all: Vec<f64> = Vec::new();
    let mut prefill_tps_all: Vec<f64> = Vec::new();
    let mut decode_tps_all: Vec<f64> = Vec::new();

    for run_i in 0..args.runs {
        let mut first_token_elapsed: Option<f64> = None;
        let t0 = Instant::now();

        let out = model
            .generate(
                tokenizer.inner(),
                &formatted,
                args.decode_tokens,
                args.temperature,
                &mut sampler,
                &mut |_piece| {
                    if first_token_elapsed.is_none() {
                        first_token_elapsed = Some(t0.elapsed().as_secs_f64());
                    }
                },
            )
            .unwrap_or_else(|e| panic!("Generation failed: {}", e));

        let total_s = t0.elapsed().as_secs_f64();
        let ttft_s = first_token_elapsed.unwrap_or(total_s);

        // decode_tokens - 1 because token #1 is included in TTFT
        let n_decode = out.tokens.saturating_sub(1);
        let decode_time_s = (total_s - ttft_s).max(1e-9);
        let decode_tps = n_decode as f64 / decode_time_s;

        // prefill throughput approximation: prompt_tokens / TTFT
        let prefill_tps = prompt_tokens as f64 / ttft_s.max(1e-9);

        eprintln!(
            "  run {}/{}: TTFT={:.0}ms  prefill={:.1} tok/s  decode={:.1} tok/s  ({} tokens)",
            run_i + 1,
            args.runs,
            ttft_s * 1000.0,
            prefill_tps,
            decode_tps,
            out.tokens,
        );

        ttft_ms_all.push(ttft_s * 1000.0);
        prefill_tps_all.push(prefill_tps);
        decode_tps_all.push(decode_tps);
    }

    let avg = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;

    let result = json!({
        "framework": format!("burn-{}", args.backend),
        "model": "Qwen3.5-0.8B",
        "backend": args.backend,
        "device": "Apple M3",
        "prompt_tokens": prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "temperature": args.temperature,
        "warmup_runs": args.warmup,
        "bench_runs": args.runs,
        "ttft_ms_avg":      avg(&ttft_ms_all),
        "prefill_tps_avg":  avg(&prefill_tps_all),
        "decode_tps_avg":   avg(&decode_tps_all),
        "ttft_ms_all":      ttft_ms_all,
        "prefill_tps_all":  prefill_tps_all,
        "decode_tps_all":   decode_tps_all,
    });

    let json_str = serde_json::to_string_pretty(&result).unwrap();
    if let Some(path) = &args.output {
        std::fs::write(path, &json_str)
            .unwrap_or_else(|e| panic!("Failed to write {}: {}", path, e));
        eprintln!("Results written to {}", path);
    } else {
        println!("{}", json_str);
    }
}

fn main() {
    let args = Args::parse();
    match args.backend.as_str() {
        #[cfg(feature = "wgpu")]
        "wgpu" => run::<Wgpu>(&args),
        #[cfg(feature = "ndarray")]
        "ndarray" => run::<NdArray>(&args),
        other => {
            eprintln!("Unknown backend '{}'. Available: wgpu, ndarray", other);
            std::process::exit(1);
        }
    }
}
