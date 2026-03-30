use std::io::{self, Write};

use clap::Parser;

#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
#[cfg(feature = "ndarray")]
use burn::backend::NdArray;

use qwen35_burn::{Qwen35, Qwen35Tokenizer, Sampler};
use qwen35_burn::model::GenerationOutput;

// ─────────────────────────────────────────────────────────────
// Think-block filter for non-thinking mode
// ─────────────────────────────────────────────────────────────

enum FilterState {
    Normal,
    InThink,
}

/// Filters `<think>...</think>` blocks out of a streaming token sequence.
/// When `enabled` is false (thinking mode) the filter is a no-op pass-through.
struct ThinkFilter {
    enabled: bool,
    state: FilterState,
    buf: String,
}

impl ThinkFilter {
    fn new(enabled: bool) -> Self {
        Self { enabled, state: FilterState::Normal, buf: String::new() }
    }

    /// Process an incoming text piece; returns the portion safe to display.
    fn process(&mut self, text: &str) -> String {
        if !self.enabled {
            return text.to_string();
        }
        self.buf.push_str(text);
        let mut output = String::new();
        loop {
            match self.state {
                FilterState::Normal => {
                    if let Some(pos) = self.buf.find("<think>") {
                        output.push_str(&self.buf[..pos]);
                        self.buf = self.buf[pos + 7..].to_string(); // 7 = len("<think>")
                        self.state = FilterState::InThink;
                    } else {
                        // Keep last 6 chars buffered for a potential partial "<think" prefix
                        let safe = self.buf.len().saturating_sub(6);
                        output.push_str(&self.buf[..safe]);
                        self.buf = self.buf[safe..].to_string();
                        break;
                    }
                }
                FilterState::InThink => {
                    if let Some(pos) = self.buf.find("</think>") {
                        let after = pos + 8; // 8 = len("</think>")
                        let rest = self.buf[after..].to_string();
                        // Strip one leading newline so the answer starts cleanly
                        self.buf = if rest.starts_with('\n') {
                            rest[1..].to_string()
                        } else {
                            rest
                        };
                        self.state = FilterState::Normal;
                    } else {
                        // Discard all but potential partial "</think" prefix (7 chars)
                        let safe = self.buf.len().saturating_sub(7);
                        self.buf = self.buf[safe..].to_string();
                        break;
                    }
                }
            }
        }
        output
    }

    /// Flush any remaining buffered text after generation ends.
    fn flush(&mut self) -> String {
        match self.state {
            FilterState::Normal => std::mem::take(&mut self.buf),
            FilterState::InThink => {
                self.buf.clear();
                String::new()
            }
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "chat", about = "Qwen3.5 chat inference")]
struct Args {
    /// Path to model directory (containing config.json + .safetensors files)
    #[arg(long)]
    model_dir: String,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: Option<String>,

    /// Maximum context length (RoPE / KV cache capacity)
    #[arg(long, default_value_t = 4096)]
    max_seq_len: usize,

    /// Maximum new tokens per response
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus-sampling top-p
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// System prompt
    #[arg(long, default_value = "You are a helpful assistant.")]
    system: String,

    /// Single prompt (non-interactive mode)
    #[arg(long)]
    prompt: Option<String>,

    /// Backend: "wgpu" or "ndarray"
    #[arg(long, default_value = "wgpu")]
    backend: String,

    /// Enable thinking mode: show the full <think>...</think> chain-of-thought.
    /// In the default (non-thinking) mode the think block is suppressed from output.
    #[arg(long)]
    thinking: bool,
}

/// Build the assistant-turn prompt, optionally pre-seeding `<think>\n` for thinking mode.
fn build_prompt(tokenizer: &Qwen35Tokenizer, system: &str, user: &str, thinking: bool) -> String {
    let mut p = tokenizer.apply_chat_template(system, user);
    if thinking {
        p.push_str("<think>\n");
    }
    p
}

fn run<B: burn::prelude::Backend>(args: &Args) {
    // Tokenizer
    let tok_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| format!("{}/tokenizer.json", args.model_dir));
    let tokenizer = Qwen35Tokenizer::new(&tok_path)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {}", e));

    // Model
    let device = B::Device::default();
    eprintln!("Loading model from {} ...", args.model_dir);
    let mut model = Qwen35::<B>::from_pretrained(&args.model_dir, args.max_seq_len, &device)
        .unwrap_or_else(|e| panic!("Failed to load model: {}", e));

    let mut sampler = Sampler::new_top_p(args.top_p, 42);

    // ── Single-prompt mode ──────────────────────────────────────────────────
    if let Some(prompt) = &args.prompt {
        let formatted = build_prompt(&tokenizer, &args.system, prompt, args.thinking);

        let mut filter = ThinkFilter::new(!args.thinking);
        let out = model
            .generate(
                tokenizer.inner(),
                &formatted,
                args.max_new_tokens,
                args.temperature,
                &mut sampler,
                &mut |piece: &str| {
                    let printable = filter.process(piece);
                    if !printable.is_empty() {
                        print!("{}", printable);
                        io::stdout().flush().ok();
                    }
                },
            )
            .unwrap_or_else(|e| panic!("Generation failed: {}", e));

        let tail = filter.flush();
        if !tail.is_empty() {
            print!("{}", tail);
        }
        println!();
        eprintln!(
            "[{} tokens in {:.1}s — {:.1} tok/s]",
            out.tokens,
            out.time,
            out.tokens as f64 / out.time.max(1e-9),
        );
        return;
    }

    // ── Interactive chat loop ───────────────────────────────────────────────
    println!("Qwen3.5 chat — type your message and press Enter. Ctrl-C to quit.\n");
    let stdin = io::stdin();

    loop {
        print!("You: ");
        io::stdout().flush().ok();

        let mut user_input = String::new();
        if stdin.read_line(&mut user_input).is_err() || user_input.trim().is_empty() {
            break;
        }
        let user_input = user_input.trim();

        let formatted = build_prompt(&tokenizer, &args.system, user_input, args.thinking);

        print!("\nAssistant: ");
        io::stdout().flush().ok();

        let mut filter = ThinkFilter::new(!args.thinking);
        let out = model
            .generate(
                tokenizer.inner(),
                &formatted,
                args.max_new_tokens,
                args.temperature,
                &mut sampler,
                &mut |piece: &str| {
                    let printable = filter.process(piece);
                    if !printable.is_empty() {
                        print!("{}", printable);
                        io::stdout().flush().ok();
                    }
                },
            )
            .unwrap_or_else(|e| {
                eprintln!("Generation error: {}", e);
                GenerationOutput { text: String::new(), tokens: 0, time: 0.0 }
            });

        let tail = filter.flush();
        if !tail.is_empty() {
            print!("{}", tail);
        }
        println!();
        eprintln!(
            "[{} tokens in {:.1}s — {:.1} tok/s]\n",
            out.tokens,
            out.time,
            out.tokens as f64 / out.time.max(1e-9),
        );
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
