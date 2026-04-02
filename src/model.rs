use std::any::TypeId;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::TensorData;

use crate::cache::{AttentionCache, DeltaNetCache, LayerCache};
use crate::config::Qwen35TextConfig;
use crate::sampling::Sampler;
use crate::transformer::{build_causal_mask, PartialRotaryEmbedding, Transformer};

// ─────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────

/// Outcome of a generation call.
pub struct GenerationOutput {
    /// Full decoded text (may not include the EOS token text).
    pub text: String,
    /// Number of new tokens produced.
    pub tokens: usize,
    /// Wall-clock time in seconds for the complete call.
    pub time: f64,
}

// ─────────────────────────────────────────────────────────────
// Internal weight-loading types
// ─────────────────────────────────────────────────────────────

/// Raw tensor bytes stored in the backend's preferred float type to minimise
/// host-to-device bandwidth: f16 when the backend is f16, f32 otherwise.
enum WeightData {
    F32(Vec<f32>),
    F16(Vec<half::f16>),
}

impl WeightData {
    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::F16(v) => v.len(),
        }
    }

    /// Convert to a `TensorData` in the backend's native element type.
    fn into_tensor_data<B: Backend>(self, shape: Vec<usize>) -> TensorData {
        let is_f16 = TypeId::of::<B::FloatElem>() == TypeId::of::<half::f16>();
        if is_f16 {
            match self {
                Self::F16(v) => TensorData::new(v, shape),
                Self::F32(v) => TensorData::new(
                    v.into_iter().map(half::f16::from_f32).collect::<Vec<_>>(),
                    shape,
                ),
            }
        } else {
            match self {
                Self::F32(v) => TensorData::new(v, shape),
                Self::F16(v) => {
                    TensorData::new(v.into_iter().map(|x| x.to_f32()).collect::<Vec<_>>(), shape)
                }
            }
        }
    }
}

/// `(data, shape)` map keyed by SafeTensors tensor name.
type TensorMap = HashMap<String, (WeightData, Vec<usize>)>;

// ─────────────────────────────────────────────────────────────
// TensorMap helpers
// ─────────────────────────────────────────────────────────────

fn take_tensor_1d<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
    let (data, _shape) = map
        .remove(name)
        .ok_or_else(|| format!("Tensor '{}' not found in SafeTensors", name))?;
    let len = data.len();
    Ok(Tensor::from_data(
        data.into_tensor_data::<B>(vec![len]),
        device,
    ))
}

fn take_tensor_2d<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    let (data, shape) = map
        .remove(name)
        .ok_or_else(|| format!("Tensor '{}' not found in SafeTensors", name))?;
    assert_eq!(
        shape.len(),
        2,
        "Expected 2D tensor for {}, got {:?}",
        name,
        shape
    );
    Ok(Tensor::from_data(data.into_tensor_data::<B>(shape), device))
}

/// Take a `[out, in]` weight and transpose to `[in, out]` (Burn convention).
fn take_linear_weight<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    Ok(take_tensor_2d(map, name, device)?.transpose())
}

/// Take a conv1d weight stored as `[channels, 1, kernel_size]` and reshape to `[channels, kernel_size]`.
fn take_conv1d_weight<B: Backend>(
    map: &mut TensorMap,
    name: &str,
    device: &Device<B>,
) -> Result<Tensor<B, 2>, Box<dyn std::error::Error>> {
    let (data, shape) = map
        .remove(name)
        .ok_or_else(|| format!("Tensor '{}' not found in SafeTensors", name))?;
    // Shape is [channels, 1, kernel_size] from PyTorch conv1d; flatten to [channels, kernel_size]
    let channels = shape[0];
    let kernel_size = *shape.last().unwrap();
    Ok(Tensor::from_data(
        data.into_tensor_data::<B>(vec![channels, kernel_size]),
        device,
    ))
}

// ─────────────────────────────────────────────────────────────
// Layer-readiness checks
// ─────────────────────────────────────────────────────────────

fn layer_tensors_ready_full(prefix: &str, map: &TensorMap) -> bool {
    let has = |suffix: &str| map.contains_key(&format!("{}.{}", prefix, suffix));
    has("self_attn.q_proj.weight")
        && has("self_attn.k_proj.weight")
        && has("self_attn.v_proj.weight")
        && has("self_attn.o_proj.weight")
        && has("self_attn.q_norm.weight")
        && has("self_attn.k_norm.weight")
        && has("mlp.gate_proj.weight")
        && has("mlp.up_proj.weight")
        && has("mlp.down_proj.weight")
        && has("input_layernorm.weight")
        && has("post_attention_layernorm.weight")
}

fn layer_tensors_ready_linear(prefix: &str, map: &TensorMap) -> bool {
    let has = |suffix: &str| map.contains_key(&format!("{}.{}", prefix, suffix));
    has("linear_attn.in_proj_qkv.weight")
        && has("linear_attn.in_proj_z.weight")
        && has("linear_attn.in_proj_b.weight")
        && has("linear_attn.in_proj_a.weight")
        && has("linear_attn.out_proj.weight")
        && has("linear_attn.conv1d.weight")
        && has("linear_attn.A_log")
        && has("linear_attn.dt_bias")
        && has("linear_attn.norm.weight")
        && has("mlp.gate_proj.weight")
        && has("mlp.up_proj.weight")
        && has("mlp.down_proj.weight")
        && has("input_layernorm.weight")
        && has("post_attention_layernorm.weight")
}

fn layer_tensors_ready(idx: usize, layer_prefix: &str, is_full: bool, map: &TensorMap) -> bool {
    let prefix = format!("{}.{}", layer_prefix, idx);
    if is_full {
        layer_tensors_ready_full(&prefix, map)
    } else {
        layer_tensors_ready_linear(&prefix, map)
    }
}

// ─────────────────────────────────────────────────────────────
// Per-layer weight loading
// ─────────────────────────────────────────────────────────────

fn load_full_attn_layer<B: Backend>(
    map: &mut TensorMap,
    transformer: Transformer<B>,
    idx: usize,
    layer_prefix: &str,
    device: &Device<B>,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    let p = format!("{}.{}", layer_prefix, idx);

    let q_proj_w = take_linear_weight(map, &format!("{}.self_attn.q_proj.weight", p), device)?;
    let k_proj_w = take_linear_weight(map, &format!("{}.self_attn.k_proj.weight", p), device)?;
    let v_proj_w = take_linear_weight(map, &format!("{}.self_attn.v_proj.weight", p), device)?;
    let o_proj_w = take_linear_weight(map, &format!("{}.self_attn.o_proj.weight", p), device)?;
    let q_norm_w = take_tensor_1d(map, &format!("{}.self_attn.q_norm.weight", p), device)?;
    let k_norm_w = take_tensor_1d(map, &format!("{}.self_attn.k_norm.weight", p), device)?;

    let gate_proj_w = take_linear_weight(map, &format!("{}.mlp.gate_proj.weight", p), device)?;
    let up_proj_w = take_linear_weight(map, &format!("{}.mlp.up_proj.weight", p), device)?;
    let gate_up_w = Tensor::cat(vec![gate_proj_w, up_proj_w], 1);
    let down_w = take_linear_weight(map, &format!("{}.mlp.down_proj.weight", p), device)?;

    let input_ln_w = take_tensor_1d(map, &format!("{}.input_layernorm.weight", p), device)?;
    let post_ln_w = take_tensor_1d(
        map,
        &format!("{}.post_attention_layernorm.weight", p),
        device,
    )?;

    Ok(transformer.load_full_attn_layer(
        idx, q_proj_w, k_proj_w, v_proj_w, o_proj_w, q_norm_w, k_norm_w, gate_up_w, down_w,
        input_ln_w, post_ln_w,
    ))
}

fn load_linear_attn_layer<B: Backend>(
    map: &mut TensorMap,
    transformer: Transformer<B>,
    idx: usize,
    layer_prefix: &str,
    device: &Device<B>,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    let p = format!("{}.{}", layer_prefix, idx);
    let la = format!("{}.linear_attn", p);

    let in_proj_qkv_w = take_linear_weight(map, &format!("{}.in_proj_qkv.weight", la), device)?;
    let in_proj_z_w = take_linear_weight(map, &format!("{}.in_proj_z.weight", la), device)?;
    let in_proj_b_w = take_linear_weight(map, &format!("{}.in_proj_b.weight", la), device)?;
    let in_proj_a_w = take_linear_weight(map, &format!("{}.in_proj_a.weight", la), device)?;
    let out_proj_w = take_linear_weight(map, &format!("{}.out_proj.weight", la), device)?;
    let conv1d_w = take_conv1d_weight(map, &format!("{}.conv1d.weight", la), device)?;
    let a_log = take_tensor_1d(map, &format!("{}.A_log", la), device)?;
    let dt_bias = take_tensor_1d(map, &format!("{}.dt_bias", la), device)?;
    let norm_w = take_tensor_1d(map, &format!("{}.norm.weight", la), device)?;

    let gate_proj_w = take_linear_weight(map, &format!("{}.mlp.gate_proj.weight", p), device)?;
    let up_proj_w = take_linear_weight(map, &format!("{}.mlp.up_proj.weight", p), device)?;
    let gate_up_w = Tensor::cat(vec![gate_proj_w, up_proj_w], 1);
    let down_w = take_linear_weight(map, &format!("{}.mlp.down_proj.weight", p), device)?;

    let input_ln_w = take_tensor_1d(map, &format!("{}.input_layernorm.weight", p), device)?;
    let post_ln_w = take_tensor_1d(
        map,
        &format!("{}.post_attention_layernorm.weight", p),
        device,
    )?;

    Ok(transformer.load_linear_attn_layer(
        idx,
        in_proj_qkv_w,
        in_proj_z_w,
        in_proj_b_w,
        in_proj_a_w,
        out_proj_w,
        conv1d_w,
        a_log,
        dt_bias,
        norm_w,
        gate_up_w,
        down_w,
        input_ln_w,
        post_ln_w,
    ))
}

// ─────────────────────────────────────────────────────────────
// SafeTensors loading
// ─────────────────────────────────────────────────────────────

/// Detect the model-weight prefix used in this SafeTensors archive.
///
/// Text-only `Qwen35ForCausalLM` uses `"model."`.
/// Multi-modal `Qwen35ForConditionalGeneration` wraps it as `"model.language_model."`.
fn detect_model_prefix(files: &[std::path::PathBuf]) -> Result<String, Box<dyn std::error::Error>> {
    for path in files {
        let bytes = std::fs::read(path)?;
        let st = safetensors::SafeTensors::deserialize(&bytes)?;
        for (name, _) in st.tensors() {
            if name.starts_with("model.language_model.embed_tokens") {
                return Ok("model.language_model".to_owned());
            }
            if name.starts_with("model.embed_tokens") {
                return Ok("model".to_owned());
            }
        }
    }
    Err(
        "Could not detect model weight prefix (neither 'model.' nor 'model.language_model.' found)"
            .into(),
    )
}

/// Stream safetensors shards into the model, loading and freeing weights layer-by-layer.
fn load_safetensors<B: Backend>(
    mut transformer: Transformer<B>,
    model_dir: &Path,
    config: &Qwen35TextConfig,
    device: &Device<B>,
) -> Result<Transformer<B>, Box<dyn std::error::Error>> {
    // Collect shard paths
    let mut st_files: Vec<std::path::PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension().is_some_and(|ext| ext == "safetensors") {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    if st_files.is_empty() {
        return Err("No .safetensors files found in model directory".into());
    }
    st_files.sort();

    let prefix = detect_model_prefix(&st_files)?;
    let layer_prefix = format!("{}.layers", prefix);
    eprintln!("Detected weight prefix: \"{}\"", prefix);

    let use_f16 = TypeId::of::<B::FloatElem>() == TypeId::of::<half::f16>();
    let num_layers = config.num_hidden_layers;

    let mut tensor_map: TensorMap = HashMap::new();
    let mut next_layer: usize = 0;
    let mut embed_loaded = false;
    let mut lm_head_weight: Option<Tensor<B, 2>> = None;

    for (shard_idx, path) in st_files.iter().enumerate() {
        eprintln!("Reading shard {}/{}...", shard_idx + 1, st_files.len());

        {
            let file_bytes = std::fs::read(path)?;
            let tensors = safetensors::SafeTensors::deserialize(&file_bytes)?;

            for (name, view) in tensors.tensors() {
                // Skip irrelevant (out-of-range) layer tensors early
                if let Some(rest) = name.strip_prefix(&format!("{}.", layer_prefix)) {
                    if let Some(dot) = rest.find('.') {
                        if let Ok(li) = rest[..dot].parse::<usize>() {
                            if li >= num_layers {
                                continue;
                            }
                        }
                    }
                }
                let shape: Vec<usize> = view.shape().to_vec();
                let raw = view.data();

                let weight_data = if use_f16 {
                    let f16_data: Vec<half::f16> = match view.dtype() {
                        safetensors::Dtype::F16 => raw
                            .chunks_exact(2)
                            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                            .collect(),
                        safetensors::Dtype::BF16 => raw
                            .chunks_exact(2)
                            .map(|c| {
                                half::f16::from_f32(
                                    half::bf16::from_le_bytes([c[0], c[1]]).to_f32(),
                                )
                            })
                            .collect(),
                        safetensors::Dtype::F32 => raw
                            .chunks_exact(4)
                            .map(|c| {
                                half::f16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            })
                            .collect(),
                        dt => return Err(format!("Unsupported dtype {:?} for {}", dt, name).into()),
                    };
                    WeightData::F16(f16_data)
                } else {
                    let f32_data: Vec<f32> = match view.dtype() {
                        safetensors::Dtype::F32 => raw
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        safetensors::Dtype::F16 => raw
                            .chunks_exact(2)
                            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                            .collect(),
                        safetensors::Dtype::BF16 => raw
                            .chunks_exact(2)
                            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                            .collect(),
                        dt => return Err(format!("Unsupported dtype {:?} for {}", dt, name).into()),
                    };
                    WeightData::F32(f32_data)
                };

                tensor_map.insert(name.to_string(), (weight_data, shape));
            }
        } // file_bytes freed here

        // Load embedding as soon as it's available
        if !embed_loaded {
            let embed_key = format!("{}.embed_tokens.weight", prefix);
            if tensor_map.contains_key(&embed_key) {
                let embed_w = take_tensor_2d(&mut tensor_map, &embed_key, device)?;
                if config.tie_word_embeddings {
                    lm_head_weight = Some(embed_w.clone().transpose());
                }
                transformer = transformer.load_embed_tokens(embed_w);
                embed_loaded = true;
                eprintln!("Embedding loaded.");
            }
        }

        // Load any fully-buffered layers
        while next_layer < num_layers {
            let is_full = config.is_full_attention(next_layer);
            if !layer_tensors_ready(next_layer, &layer_prefix, is_full, &tensor_map) {
                break;
            }
            if next_layer % 10 == 0 {
                eprintln!("Loading layer {}/{}...", next_layer, num_layers);
            }
            transformer = if is_full {
                load_full_attn_layer(
                    &mut tensor_map,
                    transformer,
                    next_layer,
                    &layer_prefix,
                    device,
                )?
            } else {
                load_linear_attn_layer(
                    &mut tensor_map,
                    transformer,
                    next_layer,
                    &layer_prefix,
                    device,
                )?
            };
            next_layer += 1;
        }
    }

    // Final norm and lm_head
    eprintln!("Loading final norm and lm_head...");
    let norm_w = take_tensor_1d(&mut tensor_map, &format!("{}.norm.weight", prefix), device)?;
    transformer = transformer.load_norm(norm_w);

    if let Some(w) = lm_head_weight {
        transformer = transformer.load_lm_head(w);
    } else {
        let lm_head_key = "lm_head.weight".to_owned();
        let w = take_linear_weight(&mut tensor_map, &lm_head_key, device)?;
        transformer = transformer.load_lm_head(w);
    }

    eprintln!("Model loaded successfully.");
    Ok(transformer)
}

// ─────────────────────────────────────────────────────────────
// Model struct
// ─────────────────────────────────────────────────────────────

/// Qwen3.5 text model (dense variants: 0.6B / 1.7B / 4B / 9B / 27B).
pub struct Qwen35<B: Backend> {
    transformer: Transformer<B>,
    rope: PartialRotaryEmbedding<B>,
    caches: Vec<LayerCache<B>>,
    config: Qwen35TextConfig,
    device: Device<B>,
}

impl<B: Backend> Qwen35<B> {
    /// Load model from a directory containing `config.json` and `.safetensors` weight files.
    pub fn from_pretrained(
        model_dir: impl AsRef<Path>,
        max_seq_len: usize,
        device: &Device<B>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let model_dir = model_dir.as_ref();

        let config = Qwen35TextConfig::from_file(model_dir.join("config.json"))?;
        eprintln!(
            "Config: hidden={} layers={} heads={} kv_heads={} vocab={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.vocab_size,
        );

        let transformer = Transformer::new(&config, device);
        let transformer = load_safetensors(transformer, model_dir, &config, device)?;

        let rotary_dim = config.rotary_dim();
        let rope = PartialRotaryEmbedding::new(
            config.head_dim,
            rotary_dim,
            max_seq_len,
            config.rope_theta(),
            device,
        );

        let caches = build_caches(&config, max_seq_len, device);

        Ok(Self {
            transformer,
            rope,
            caches,
            config,
            device: device.clone(),
        })
    }

    /// Reset all caches for a fresh generation.
    pub fn reset_caches(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    /// Autoregressive text generation.
    ///
    /// Resets caches before each call.  Returns all newly generated tokens as text.
    /// `on_token` is called with each new decoded text piece as it is generated (streaming).
    pub fn generate(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        sampler: &mut Sampler,
        on_token: &mut impl FnMut(&str),
    ) -> Result<GenerationOutput, String> {
        self.reset_caches();

        let input_ids = tokenizer
            .encode(prompt, false)
            .map_err(|e| format!("Tokenization failed: {}", e))?
            .get_ids()
            .to_vec();
        let prompt_len = input_ids.len();

        if prompt_len == 0 {
            return Err("Prompt encoded to zero tokens".into());
        }

        let vocab = self.config.vocab_size;
        let eos = self.config.eos_token_id;

        let start = Instant::now();

        // ── Prefill ───────────────────────────────────────────────────────────

        let token_ids_i32: Vec<i32> = input_ids.iter().map(|&t| t as i32).collect();
        let td = TensorData::new(token_ids_i32, vec![prompt_len]);
        let token_tensor = Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>(); // [1, S]

        let mask = build_causal_mask::<B>(prompt_len, prompt_len, &self.device);
        let prefill_logits =
            self.transformer
                .forward(token_tensor, &self.rope, &mut self.caches, 0, Some(mask)); // [1, S, vocab]

        // Sample first decode token from the last prefill position
        let last_logits = prefill_logits
            .slice([0..1, (prompt_len - 1)..prompt_len, 0..vocab])
            .reshape([1, vocab]); // [1, vocab]
        let mut next_token = sample_token(&last_logits, temperature, sampler);

        let _ = B::sync(&self.device);

        // ── Decode ────────────────────────────────────────────────────────────

        let mut pos = prompt_len;
        let mut generated: Vec<u32> = Vec::new();

        loop {
            if next_token == eos {
                break;
            }
            generated.push(next_token);
            // Stream this token piece immediately
            let piece = tokenizer
                .decode(&[next_token], true)
                .map_err(|e| format!("Decode failed: {}", e))?;
            on_token(&piece);
            if generated.len() >= max_new_tokens {
                break;
            }

            let td = TensorData::new(vec![next_token as i32], vec![1]);
            let tok = Tensor::<B, 1, Int>::from_data(td, &self.device).unsqueeze::<2>();
            let logits = self
                .transformer
                .forward(tok, &self.rope, &mut self.caches, pos, None);
            let logits = logits.reshape([1, vocab]);
            next_token = sample_token(&logits, temperature, sampler);
            pos += 1;
        }

        let elapsed = start.elapsed().as_secs_f64();

        let decoded = tokenizer
            .decode(&generated, true)
            .map_err(|e| format!("Decode failed: {}", e))?;

        Ok(GenerationOutput {
            text: decoded,
            tokens: generated.len(),
            time: elapsed,
        })
    }
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

/// Build one `LayerCache` per layer.
fn build_caches<B: Backend>(
    config: &Qwen35TextConfig,
    max_seq_len: usize,
    device: &Device<B>,
) -> Vec<LayerCache<B>> {
    (0..config.num_hidden_layers)
        .map(|i| {
            if config.is_full_attention(i) {
                LayerCache::Attn(AttentionCache::new(
                    1,
                    config.num_key_value_heads,
                    max_seq_len,
                    config.head_dim,
                    device,
                ))
            } else {
                let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
                let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
                let conv_dim = 2 * key_dim + value_dim;
                LayerCache::Linear(DeltaNetCache::new(
                    1,
                    conv_dim,
                    config.linear_conv_kernel_dim,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    device,
                ))
            }
        })
        .collect()
}

/// Extract logits, apply temperature, and sample.
///
/// For greedy decoding (`temperature <= 0`), argmax is computed on-device and only
/// the resulting index (~4 bytes) is transferred to CPU, avoiding the ~1 MB
/// GPU→CPU readback of the full vocabulary every decode step.
fn sample_token<B: Backend>(logits: &Tensor<B, 2>, temperature: f64, sampler: &mut Sampler) -> u32 {
    if temperature <= 0.0 {
        // argmax stays on-device; only the winning index is copied to CPU
        let idx = logits.clone().argmax(1).into_data();
        return idx.iter::<i64>().next().unwrap_or(0) as u32;
    }

    let logits_f64: Vec<f64> = logits.to_data().iter::<f64>().collect();
    let max = logits_f64.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits_f64
        .iter()
        .map(|&x| ((x - max) / temperature).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    let probs: Vec<f64> = exp.iter().map(|&x| x / sum).collect();
    sampler.sample_probs(&probs)
}
