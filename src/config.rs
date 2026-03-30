use std::path::Path;

use serde::Deserialize;

/// RoPE parameters nested inside text_config.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub rope_type: Option<String>,
}

fn default_rope_theta() -> f64 {
    10_000_000.0
}
fn default_partial_rotary_factor() -> f64 {
    0.25
}

/// The text-model configuration extracted from Qwen3.5's `config.json`.
///
/// The HuggingFace config nests text parameters under `text_config`; this struct
/// can be deserialized from either the nested form (via [`Qwen35Config`]) or
/// standalone.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,	

    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    /// Full-attention head dimension (Qwen3.5 uses 256).
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,

    /// Whether to apply a sigmoid gate to the attention output.
    #[serde(default = "default_attn_output_gate")]
    pub attn_output_gate: bool,

    /// Per-layer type ("full_attention" or "linear_attention").
    /// Length must equal `num_hidden_layers`.
    #[serde(default)]
    pub layer_types: Vec<String>,

    // Linear attention (GatedDeltaNet) parameters
    #[serde(default = "default_linear_conv_kernel_dim")]
    pub linear_conv_kernel_dim: usize,
    #[serde(default = "default_linear_key_head_dim")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_linear_value_head_dim")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_linear_num_key_heads")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_linear_num_value_heads")]
    pub linear_num_value_heads: usize,

    /// RoPE configuration (theta, partial_rotary_factor).
    #[serde(default = "default_rope_parameters")]
    pub rope_parameters: RopeParameters,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,

    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
}

fn default_max_position_embeddings() -> usize { 262144 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_head_dim() -> usize { 256 }
fn default_attn_output_gate() -> bool { true }
fn default_linear_conv_kernel_dim() -> usize { 4 }
fn default_linear_key_head_dim() -> usize { 128 }
fn default_linear_value_head_dim() -> usize { 128 }
fn default_linear_num_key_heads() -> usize { 16 }
fn default_linear_num_value_heads() -> usize { 16 }
fn default_eos_token_id() -> u32 { 248044 }
fn default_bos_token_id() -> u32 { 151643 }
fn default_rope_parameters() -> RopeParameters {
    RopeParameters {
        rope_theta: 10_000_000.0,
        partial_rotary_factor: 0.25,
        rope_type: None,
    }
}

impl Qwen35TextConfig {
    /// Compute the actual rotary embedding dimension from head_dim × partial_rotary_factor.
    pub fn rotary_dim(&self) -> usize {
        let d = (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor).round() as usize;
        // Must be even
        d & !1
    }

    /// rope_theta from rope_parameters.
    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    /// Returns true if layer `i` is a full (softmax) attention layer.
    pub fn is_full_attention(&self, layer_idx: usize) -> bool {
        if self.layer_types.is_empty() {
            // Fallback: default to every 4th layer being full attention
            return (layer_idx + 1) % 4 == 0;
        }
        self.layer_types
            .get(layer_idx)
            .map(|t| t == "full_attention")
            .unwrap_or(false)
    }
}

/// Top-level Qwen3.5 config.json wrapper.
///
/// HuggingFace stores the multimodal `Qwen35ForConditionalGeneration` config with
/// text parameters nested under `text_config`.  This struct handles that nesting.
#[derive(Debug, Deserialize)]
struct Qwen35Outer {
    #[serde(default)]
    text_config: Option<Qwen35TextConfig>,

    // Flat form (used by text-only checkpoints or when text_config is absent)
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    num_key_value_heads: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    vocab_size: Option<usize>,

    // Top-level tie_word_embeddings used for HF multimodal models
    #[serde(default)]
    tie_word_embeddings: Option<bool>,
}

impl Qwen35TextConfig {
    /// Load from a HuggingFace `config.json`, handling both the nested
    /// `text_config` format and any flat fallback.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let outer: Qwen35Outer = serde_json::from_str(&contents)?;

        if let Some(mut cfg) = outer.text_config {
            // Prefer the outermost tie_word_embeddings if the nested one is false
            // (HF sometimes sets it only at the outer level)
            if let Some(outer_tie) = outer.tie_word_embeddings {
                cfg.tie_word_embeddings = outer_tie;
            }
            return Ok(cfg);
        }

        // Attempt to deserialize the whole JSON as a flat text config
        let cfg: Qwen35TextConfig = serde_json::from_str(&contents)
            .map_err(|e| format!("Could not parse config.json as text config: {}", e))?;
        Ok(cfg)
    }
}
