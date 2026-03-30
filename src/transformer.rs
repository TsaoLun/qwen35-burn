use burn::module::Param;
use burn::nn;
use burn::nn::Embedding;
use burn::prelude::*;
use burn::tensor::activation;

use crate::cache::{AttentionCache, DeltaNetCache, LayerCache};
use crate::config::Qwen35TextConfig;

// ─────────────────────────────────────────────────────────────
// RMS normalisation variants
// ─────────────────────────────────────────────────────────────

/// Standard RMSNorm for Qwen3.5.
///
/// Weight is **initialised to 0**, and the formula is `(1 + w) * x / rms(x)`.
/// A per-channel coefficient of (1 + 0) = 1 at init.  The f16-safe pre-scaling
/// trick prevents squared overflow: `(x/s) / rms(x/s) = x / rms(x)`.
#[derive(Module, Debug)]
pub struct Qwen35RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> Qwen35RmsNorm<B> {
    pub fn new(size: usize, eps: f64, device: &Device<B>) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::zeros([size], device)),
            eps,
        }
    }

    /// Forward for `[batch, seq, hidden]` input.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, hidden] = x.dims();
        let variance = x.clone().powf_scalar(2.0).mean_dim(2);
        let normed = x * (variance + self.eps).sqrt().recip();
        let w = (self.weight.val().unsqueeze::<3>() + 1.0).expand([batch, seq, hidden]);
        normed * w
    }

    /// Forward accepting 3-D input already shaped as `[N, 1, D]`
    /// (e.g. for per-head QK norm).  Returns the same shape.
    pub fn forward_3d(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward(x)
    }
}

/// Gated RMSNorm used inside GatedDeltaNet.
///
/// Weight **initialised to 1**.  Formula: `w * (x / rms(x)) * silu(gate)`.
#[derive(Module, Debug)]
pub struct GatedRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    eps: f64,
}

impl<B: Backend> GatedRmsNorm<B> {
    pub fn new(size: usize, eps: f64, device: &Device<B>) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::ones([size], device)),
            eps,
        }
    }

    /// Forward for 2-D `[N, D]` inputs (batch × time × heads flattened).
    pub fn forward(&self, x: Tensor<B, 2>, gate: Tensor<B, 2>) -> Tensor<B, 2> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(1);
        let normed = x * (variance + self.eps).sqrt().recip();
        let w = self.weight.val().unsqueeze::<2>();
        activation::silu(gate) * (normed * w)
    }
}

// ─────────────────────────────────────────────────────────────
// Partial Rotary Position Embedding
// ─────────────────────────────────────────────────────────────

/// Rotary embeddings that apply rotation only to the first `rotary_dim`
/// elements of each head (partial_rotary_factor = rotary_dim / head_dim).
///
/// Cos/sin tables are precomputed on CPU in f64 to retain precision for
/// small frequencies, then stored on device in the backend's float type.
pub struct PartialRotaryEmbedding<B: Backend> {
    /// `[max_seq, rotary_dim]`
    cos: Tensor<B, 2>,
    /// `[max_seq, rotary_dim]`
    sin: Tensor<B, 2>,
    pub rotary_dim: usize,
    pub head_dim: usize,
}

impl<B: Backend> PartialRotaryEmbedding<B> {
    pub fn new(
        head_dim: usize,
        rotary_dim: usize,
        max_seq_len: usize,
        theta: f64,
        device: &Device<B>,
    ) -> Self {
        let half = rotary_dim / 2;
        let len = max_seq_len * rotary_dim;
        let mut cos_data = Vec::with_capacity(len);
        let mut sin_data = Vec::with_capacity(len);

        for pos in 0..max_seq_len {
            for i in 0..half {
                let freq = 1.0 / theta.powf(i as f64 * 2.0 / rotary_dim as f64);
                let angle = pos as f64 * freq;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            // Duplicate so that positions [half..rotary_dim] have the same freqs
            // (required by rotate_half: [-x2, x1] where x1 and x2 use matching freqs)
            let start = cos_data.len() - half;
            cos_data.extend_from_within(start..);
            sin_data.extend_from_within(start..);
        }

        let cos = Tensor::<B, 1>::from_floats(cos_data.as_slice(), device)
            .reshape([max_seq_len, rotary_dim]);
        let sin = Tensor::<B, 1>::from_floats(sin_data.as_slice(), device)
            .reshape([max_seq_len, rotary_dim]);

        Self {
            cos,
            sin,
            rotary_dim,
            head_dim,
        }
    }

    /// Apply partial RoPE to query and key tensors.
    ///
    /// - `q`, `k`: `[batch, heads, seq_len, head_dim]`
    /// - `start_pos`: position offset for cached generation
    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        start_pos: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, heads_q, seq_len, _] = q.dims();
        let heads_k = k.dims()[1];
        let rd = self.rotary_dim;
        let hd = self.head_dim;

        // Slice cos/sin for current positions: [seq_len, rotary_dim]
        // Then unsqueeze to [1, 1, seq_len, rotary_dim] for broadcasting
        let cos = self
            .cos
            .clone()
            .slice([start_pos..start_pos + seq_len])
            .unsqueeze::<3>()
            .unsqueeze::<4>();
        let sin = self
            .sin
            .clone()
            .slice([start_pos..start_pos + seq_len])
            .unsqueeze::<3>()
            .unsqueeze::<4>();

        // Split rotated / pass-through portions
        let q_rot = q.clone().slice([0..batch, 0..heads_q, 0..seq_len, 0..rd]);
        let q_pass = q.slice([0..batch, 0..heads_q, 0..seq_len, rd..hd]);
        let k_rot = k.clone().slice([0..batch, 0..heads_k, 0..seq_len, 0..rd]);
        let k_pass = k.slice([0..batch, 0..heads_k, 0..seq_len, rd..hd]);

        let q_embed = q_rot.clone() * cos.clone()
            + rotate_half_partial(q_rot, rd, batch, heads_q, seq_len) * sin.clone();
        let k_embed =
            k_rot.clone() * cos + rotate_half_partial(k_rot, rd, batch, heads_k, seq_len) * sin;

        let q_out = Tensor::cat(vec![q_embed, q_pass], 3);
        let k_out = Tensor::cat(vec![k_embed, k_pass], 3);
        (q_out, k_out)
    }
}

/// Compute `[-x2, x1]` for a `[b, h, s, rotary_dim]` tensor.
fn rotate_half_partial<B: Backend>(
    x: Tensor<B, 4>,
    rotary_dim: usize,
    batch: usize,
    heads: usize,
    seq: usize,
) -> Tensor<B, 4> {
    let half = rotary_dim / 2;
    let x1 = x.clone().slice([0..batch, 0..heads, 0..seq, 0..half]);
    let x2 = x.slice([0..batch, 0..heads, 0..seq, half..rotary_dim]);
    Tensor::cat(vec![x2.neg(), x1], 3)
}

// ─────────────────────────────────────────────────────────────
// Full (softmax) Attention layer
// ─────────────────────────────────────────────────────────────

/// Full multi-head attention with:
/// - Separate Q/K/V projections (no fusion — different output dims)
/// - q_proj outputs 2 × head_dim per head: `[query | gate]`; gate applied after attn
/// - QK-Norm (Qwen3.5RMSNorm on each head)
/// - Partial RoPE
/// - Grouped-Query Attention (GQA)
#[derive(Module, Debug)]
pub struct FullAttention<B: Backend> {
    /// [hidden, num_heads * head_dim * 2]  (query + gate)
    q_proj: nn::Linear<B>,
    /// [hidden, num_kv_heads * head_dim]
    k_proj: nn::Linear<B>,
    /// [hidden, num_kv_heads * head_dim]
    v_proj: nn::Linear<B>,
    /// [num_heads * head_dim, hidden]
    o_proj: nn::Linear<B>,
    q_norm: Qwen35RmsNorm<B>,
    k_norm: Qwen35RmsNorm<B>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> FullAttention<B> {
    pub fn new(
        hidden: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_eps: f64,
        device: &Device<B>,
    ) -> Self {
        let q_out_dim = num_heads * head_dim * 2; // query + gate
        let kv_out_dim = num_kv_heads * head_dim;
        let attn_out_dim = num_heads * head_dim;

        Self {
            q_proj: nn::LinearConfig::new(hidden, q_out_dim)
                .with_bias(false)
                .init(device),
            k_proj: nn::LinearConfig::new(hidden, kv_out_dim)
                .with_bias(false)
                .init(device),
            v_proj: nn::LinearConfig::new(hidden, kv_out_dim)
                .with_bias(false)
                .init(device),
            o_proj: nn::LinearConfig::new(attn_out_dim, hidden)
                .with_bias(false)
                .init(device),
            q_norm: Qwen35RmsNorm::new(head_dim, rms_eps, device),
            k_norm: Qwen35RmsNorm::new(head_dim, rms_eps, device),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Forward pass.
    ///
    /// - `x`: `[batch, seq_len, hidden]`
    /// - `rope`: Partial RoPE
    /// - `mask`: optional causal mask `[1, 1, q_seq, kv_seq]`
    /// - `cache`: KV cache
    /// - `start_pos`: position offset for RoPE and KV cache
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &PartialRotaryEmbedding<B>,
        mask: Option<Tensor<B, 4>>,
        cache: &mut AttentionCache<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let q_total_dim = self.num_heads * self.head_dim;

        // Projections
        let q_raw = self.q_proj.forward(x.clone()); // [B, S, num_heads * head_dim * 2]
        let k_raw = self.k_proj.forward(x.clone()); // [B, S, num_kv_heads * head_dim]
        let v_raw = self.v_proj.forward(x); // [B, S, num_kv_heads * head_dim]

        // Split Q into query and gate halves
        // q_raw: [B, S, num_heads * head_dim * 2] → view [B, S, num_heads, head_dim * 2]
        let q_raw = q_raw.reshape([batch, seq_len, self.num_heads, self.head_dim * 2]);
        let query =
            q_raw
                .clone()
                .slice([0..batch, 0..seq_len, 0..self.num_heads, 0..self.head_dim]);
        // gate: [B, S, num_heads, head_dim] → reshape to [B, S, num_heads * head_dim]
        let gate = q_raw
            .slice([
                0..batch,
                0..seq_len,
                0..self.num_heads,
                self.head_dim..self.head_dim * 2,
            ])
            .reshape([batch, seq_len, q_total_dim]);

        // Reshape K, V
        let k = k_raw.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);
        let v = v_raw.reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // QK-Norm (per-head, normalise head_dim dimension)
        let q = self
            .q_norm
            .forward(query.reshape([batch * seq_len * self.num_heads, 1, self.head_dim]))
            .reshape([batch, seq_len, self.num_heads, self.head_dim]);
        let k = self
            .k_norm
            .forward(k.reshape([batch * seq_len * self.num_kv_heads, 1, self.head_dim]))
            .reshape([batch, seq_len, self.num_kv_heads, self.head_dim]);

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        // Partial RoPE
        let (q, k) = rope.apply(q, k, start_pos);

        // KV cache update
        let k = cache.k_cache.forward(k);
        let v = cache.v_cache.forward(v);

        // GQA: expand K, V to match Q heads
        let n_groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_groups);
        let v = repeat_kv(v, n_groups);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let attn_weights = q.matmul(k.transpose()) * scale;
        let attn_weights = match mask {
            Some(m) => attn_weights + m,
            None => attn_weights,
        };
        let attn_weights = activation::softmax(attn_weights, 3);
        let attn_out = attn_weights.matmul(v); // [B, num_heads, S, head_dim]

        // Reshape → [B, S, num_heads * head_dim]
        let attn_out = attn_out
            .swap_dims(1, 2)
            .reshape([batch, seq_len, q_total_dim]);

        // Output gate (sigmoid)
        let attn_out = attn_out * activation::sigmoid(gate);

        // Output projection
        self.o_proj.forward(attn_out)
    }
}

/// Expand KV heads for GQA: `[B, kv_heads, S, D]` → `[B, kv_heads*n, S, D]`.
fn repeat_kv<B: Backend>(x: Tensor<B, 4>, n: usize) -> Tensor<B, 4> {
    if n == 1 {
        return x;
    }
    let [b, kv, s, d] = x.dims();
    x.unsqueeze_dim::<5>(2)
        .expand([b, kv, n, s, d])
        .reshape([b, kv * n, s, d])
}

// ─────────────────────────────────────────────────────────────
// GatedDeltaNet (linear attention / SSM layer)
// ─────────────────────────────────────────────────────────────

/// Gated Delta-Rule linear-attention layer (Qwen3.5 "linear_attention" blocks).
///
/// Architecture:
/// 1. Project `hidden` → `[q | k | v]` via `in_proj_qkv` (dims: 2·key_dim + value_dim)
/// 2. Apply causal depthwise conv1d to the projected Q/K/V stream
/// 3. Project `hidden` → gate `z`, per-head beta `b`, per-head decay input `a`
/// 4. Run the gated delta-rule recurrence
/// 5. Apply gated RMSNorm and output projection
#[derive(Module, Debug)]
pub struct GatedDeltaNet<B: Backend> {
    in_proj_qkv: nn::Linear<B>,
    in_proj_z: nn::Linear<B>,
    in_proj_b: nn::Linear<B>,
    in_proj_a: nn::Linear<B>,
    out_proj: nn::Linear<B>,
    /// `[conv_dim, conv_kernel_size]` — depthwise conv1d weights (no bias).
    conv1d_weight: Param<Tensor<B, 2>>,
    /// Log-magnitude of per-head decay: `[num_v_heads]`.
    a_log: Param<Tensor<B, 1>>,
    /// Per-head dt bias: `[num_v_heads]`.
    dt_bias: Param<Tensor<B, 1>>,
    norm: GatedRmsNorm<B>,
    // --- shape info ---
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    conv_dim: usize,
    conv_kernel: usize,
}

impl<B: Backend> GatedDeltaNet<B> {
    pub fn new(
        hidden: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        conv_kernel: usize,
        rms_eps: f64,
        device: &Device<B>,
    ) -> Self {
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;

        Self {
            in_proj_qkv: nn::LinearConfig::new(hidden, conv_dim)
                .with_bias(false)
                .init(device),
            in_proj_z: nn::LinearConfig::new(hidden, value_dim)
                .with_bias(false)
                .init(device),
            in_proj_b: nn::LinearConfig::new(hidden, num_v_heads)
                .with_bias(false)
                .init(device),
            in_proj_a: nn::LinearConfig::new(hidden, num_v_heads)
                .with_bias(false)
                .init(device),
            out_proj: nn::LinearConfig::new(value_dim, hidden)
                .with_bias(false)
                .init(device),
            conv1d_weight: Param::from_tensor(Tensor::zeros([conv_dim, conv_kernel], device)),
            a_log: Param::from_tensor(Tensor::zeros([num_v_heads], device)),
            dt_bias: Param::from_tensor(Tensor::ones([num_v_heads], device)),
            norm: GatedRmsNorm::new(head_v_dim, rms_eps, device),
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_kernel,
        }
    }

    /// Causal depthwise conv1d for the **prefill** case (seq_len > 1).
    ///
    /// Returns `[batch, conv_dim, seq_len]` and updates `cache.conv_state`.
    fn conv1d_prefill(
        &self,
        x: Tensor<B, 3>, // [B, conv_dim, S]
        cache: &mut DeltaNetCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, conv_dim, seq_len] = x.dims();
        let k = self.conv_kernel;
        let w = self.conv1d_weight.val(); // [conv_dim, K]

        // Left-pad with K−1 zeros for causal alignment
        let pad = Tensor::zeros([batch, conv_dim, k - 1], &x.device());
        let padded = Tensor::cat(vec![pad, x.clone()], 2); // [B, conv_dim, S+K-1]

        // Unrolled for fixed kernel size K (K=4 in practice).
        // out[:,:,t] = Σ_j padded[:,:,t+j] * w[:,j]
        let out = {
            let w0 = w
                .clone()
                .slice([0..conv_dim, 0..1])
                .reshape([1, conv_dim, 1]);
            let w1 = w
                .clone()
                .slice([0..conv_dim, 1..2])
                .reshape([1, conv_dim, 1]);
            let w2 = w
                .clone()
                .slice([0..conv_dim, 2..3])
                .reshape([1, conv_dim, 1]);
            let w3 = w.slice([0..conv_dim, 3..4]).reshape([1, conv_dim, 1]);

            let s0 = padded.clone().slice([0..batch, 0..conv_dim, 0..seq_len]);
            let s1 = padded
                .clone()
                .slice([0..batch, 0..conv_dim, 1..seq_len + 1]);
            let s2 = padded
                .clone()
                .slice([0..batch, 0..conv_dim, 2..seq_len + 2]);
            let s3 = padded
                .clone()
                .slice([0..batch, 0..conv_dim, 3..seq_len + 3]);

            s0 * w0 + s1 * w1 + s2 * w2 + s3 * w3
        };

        // Update conv state: last K elements of x (zero-padded from left if seq_len < K)
        let state_start = if seq_len >= k { seq_len - k } else { 0 };
        let zero_pad = (k as isize - seq_len as isize).max(0) as usize;
        cache.conv_state = if zero_pad > 0 {
            let part = x.clone().slice([0..batch, 0..conv_dim, 0..seq_len]);
            let zeros = Tensor::zeros([batch, conv_dim, zero_pad], &x.device());
            Tensor::cat(vec![zeros, part], 2)
        } else {
            x.slice([0..batch, 0..conv_dim, state_start..seq_len])
        };

        activation::silu(out)
    }

    /// Causal depthwise conv1d for the **decode** case (seq_len == 1).
    ///
    /// Returns `[batch, conv_dim, 1]` and updates `cache.conv_state`.
    fn conv1d_decode(
        &self,
        x: Tensor<B, 3>, // [B, conv_dim, 1]
        cache: &mut DeltaNetCache<B>,
    ) -> Tensor<B, 3> {
        let [batch, conv_dim, _] = x.dims();
        let k = self.conv_kernel;
        let w = self.conv1d_weight.val(); // [conv_dim, K]

        // Concatenate state (K) with new token (1) → [B, conv_dim, K+1]
        let combined = Tensor::cat(vec![cache.conv_state.clone(), x], 2);

        // Slice [1..K+1] once: this is both the conv window and the new conv state
        let tail = combined.slice([0..batch, 0..conv_dim, 1..k + 1]); // [B, conv_dim, K]
        let w_broad = w.reshape([1, conv_dim, k]);
        let out = (tail.clone() * w_broad).sum_dim(2); // [B, conv_dim, 1]

        // New conv state = same tail (the last K inputs)
        cache.conv_state = tail;

        activation::silu(out)
    }

    /// L2-normalise the last dimension of a 4-D tensor.
    fn l2_norm(x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, s, h, d] = x.dims();
        // sum_dim keeps the reduced dim as size-1, so result is [B, S, H, 1] (still 4D)
        let sum_sq = x.clone().powf_scalar(2.0).sum_dim(3); // [B, S, H, 1]
        let inv = (sum_sq + 1e-6f32).sqrt().recip(); // [B, S, H, 1] — no unsqueeze needed
        x * inv.expand([b, s, h, d])
    }

    /// Numerically-stable softplus: `max(0, x) + log(1 + exp(−|x|))`.
    fn softplus(x: Tensor<B, 3>) -> Tensor<B, 3> {
        x.clone().clamp_min(0.0) + ((-x.abs()).exp() + 1.0).log()
    }

    /// Run the gated delta-rule recurrence over all `seq_len` positions.
    ///
    /// All inputs in `[batch, seq_len, num_v_heads, dim]` form.
    /// Returns `[batch, seq_len, num_v_heads, head_v_dim]`.
    fn recurrent_forward(
        q: Tensor<B, 4>,      // [B, S, nv, head_k_dim]  — already l2-normed & scaled
        k: Tensor<B, 4>,      // [B, S, nv, head_k_dim]  — l2-normed
        v: Tensor<B, 4>,      // [B, S, nv, head_v_dim]
        g: Tensor<B, 3>,      // [B, S, nv]  (decay, ≤ 0)
        beta: Tensor<B, 3>,   // [B, S, nv]
        h: &mut Tensor<B, 4>, // [B, nv, head_k_dim, head_v_dim]  — mutated in-place
    ) -> Tensor<B, 4> {
        let [batch, seq_len, n_v_heads, head_v_dim] = v.dims();
        let head_k_dim = q.dims()[3];

        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Slice step t → [B, nv, head_k_dim / head_v_dim]
            let q_t = q
                .clone()
                .slice([0..batch, t..t + 1, 0..n_v_heads, 0..head_k_dim])
                .reshape([batch, n_v_heads, head_k_dim]);
            let k_t = k
                .clone()
                .slice([0..batch, t..t + 1, 0..n_v_heads, 0..head_k_dim])
                .reshape([batch, n_v_heads, head_k_dim]);
            let v_t = v
                .clone()
                .slice([0..batch, t..t + 1, 0..n_v_heads, 0..head_v_dim])
                .reshape([batch, n_v_heads, head_v_dim]);
            let beta_t = beta
                .clone()
                .slice([0..batch, t..t + 1, 0..n_v_heads])
                .reshape([batch, n_v_heads]);
            let g_t = g
                .clone()
                .slice([0..batch, t..t + 1, 0..n_v_heads])
                .reshape([batch, n_v_heads]);

            // decay: exp(g_t) → [B, nv, 1, 1]
            let decay = g_t.exp().unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3); // [B, nv, 1, 1]

            // Decay state
            let h_new = h.clone() * decay.expand([batch, n_v_heads, head_k_dim, head_v_dim]);

            // Memory lookup: kv_mem = (h * k_t[..., None]).sum(-2)
            // k_t: [B, nv, head_k_dim] → [B, nv, head_k_dim, 1]
            let k_exp = k_t.clone().unsqueeze_dim::<4>(3); // [B, nv, hk, 1]
            let kv_mem = (h_new.clone()
                * k_exp
                    .clone()
                    .expand([batch, n_v_heads, head_k_dim, head_v_dim]))
            .sum_dim(2)
            .reshape([batch, n_v_heads, head_v_dim]); // [B, nv, hv]

            // Delta correction: delta = (v_t - kv_mem) * beta_t
            let delta = (v_t - kv_mem) * beta_t.unsqueeze_dim::<3>(2); // [B, nv, hv]

            // State update: h += k_t[..., None] * delta[..., None, :]
            let delta_exp = delta.unsqueeze_dim::<4>(2); // [B, nv, 1, hv]
            let h_new = h_new
                + k_exp.expand([batch, n_v_heads, head_k_dim, head_v_dim])
                    * delta_exp.expand([batch, n_v_heads, head_k_dim, head_v_dim]);

            // Output: out_t = (h * q_t[..., None]).sum(-2)
            let q_exp = q_t.unsqueeze_dim::<4>(3); // [B, nv, hk, 1]
            let out_t = (h_new.clone() * q_exp.expand([batch, n_v_heads, head_k_dim, head_v_dim]))
                .sum_dim(2)
                .reshape([batch, n_v_heads, head_v_dim]); // [B, nv, hv]

            *h = h_new;
            outputs.push(out_t.unsqueeze_dim::<4>(1)); // [B, 1, nv, hv]
        }

        Tensor::cat(outputs, 1) // [B, S, nv, hv]
    }

    /// Forward pass.
    ///
    /// - `x`: `[batch, seq_len, hidden]`
    /// - `cache`: mutable DeltaNetCache for this layer
    pub fn forward(&self, x: Tensor<B, 3>, cache: &mut DeltaNetCache<B>) -> Tensor<B, 3> {
        let [batch, seq_len, _hidden] = x.dims();
        let key_dim = self.num_k_heads * self.head_k_dim;
        let value_dim = self.num_v_heads * self.head_v_dim;
        let nv = self.num_v_heads;
        let hk = self.head_k_dim;
        let hv = self.head_v_dim;

        // Projections
        let qkv_raw = self.in_proj_qkv.forward(x.clone()); // [B, S, conv_dim]
        let z = self.in_proj_z.forward(x.clone()); // [B, S, value_dim]
        let b = self.in_proj_b.forward(x.clone()); // [B, S, num_v_heads]
        let a = self.in_proj_a.forward(x); // [B, S, num_v_heads]

        // Causal conv1d on Q/K/V stream (transpose to [B, conv_dim, S])
        let qkv_t = qkv_raw.swap_dims(1, 2); // [B, conv_dim, S]
        let mixed_t = if seq_len == 1 && cache.tokens_seen > 0 {
            self.conv1d_decode(qkv_t, cache)
        } else {
            self.conv1d_prefill(qkv_t, cache)
        };
        // Back to [B, S, conv_dim]
        let mixed = mixed_t.swap_dims(1, 2);

        // Split Q, K, V
        let q = mixed.clone().slice([0..batch, 0..seq_len, 0..key_dim]);
        let k = mixed
            .clone()
            .slice([0..batch, 0..seq_len, key_dim..2 * key_dim]);
        let v = mixed.slice([0..batch, 0..seq_len, 2 * key_dim..2 * key_dim + value_dim]);

        // Reshape to [B, S, num_k/v_heads, head_dim]
        let q = q.reshape([batch, seq_len, self.num_k_heads, hk]);
        let k = k.reshape([batch, seq_len, self.num_k_heads, hk]);
        let v = v.reshape([batch, seq_len, nv, hv]);

        // L2 normalise Q and K
        let q = Self::l2_norm(q);
        let k = Self::l2_norm(k);

        // GQA expansion (if num_v_heads > num_k_heads)
        let n_rep = nv / self.num_k_heads;
        let (q, k) = if n_rep > 1 {
            (
                repeat_heads(q, n_rep, batch, seq_len, self.num_k_heads, hk),
                repeat_heads(k, n_rep, batch, seq_len, self.num_k_heads, hk),
            )
        } else {
            (q, k)
        };
        // Now q, k: [B, S, nv, hk]

        // beta = sigmoid(b), shape [B, S, nv]
        let beta = activation::sigmoid(b);

        // g = -A_log.exp() * softplus(a + dt_bias)  — all values ≤ 0
        let dt_bias = self
            .dt_bias
            .val()
            .reshape([1, 1, nv])
            .expand([batch, seq_len, nv]);
        let a_log_exp = self
            .a_log
            .val()
            .exp()
            .reshape([1, 1, nv])
            .expand([batch, seq_len, nv]);
        let g = -(a_log_exp * Self::softplus(a + dt_bias)); // [B, S, nv]

        // Scale Q by 1/sqrt(head_k_dim)
        let scale = (hk as f32).sqrt().recip();
        let q = q * scale;

        // Run recurrent delta rule
        let mut h = cache.recurrent_state.clone();
        let core_out = Self::recurrent_forward(q, k, v, g, beta, &mut h);
        cache.recurrent_state = h;
        cache.tokens_seen += seq_len;

        // core_out: [B, S, nv, hv] → reshape to [B*S*nv, hv]
        let core_flat = core_out.reshape([batch * seq_len * nv, hv]);
        let z_flat = z.reshape([batch * seq_len * nv, hv]);

        // Gated RMSNorm
        let normed = self.norm.forward(core_flat, z_flat); // [B*S*nv, hv]

        // Reshape back and project
        let out = normed.reshape([batch, seq_len, value_dim]);
        self.out_proj.forward(out)
    }
}

/// Expand heads along dim 2: [B, S, nk, D] → [B, S, nk*n, D].
fn repeat_heads<B: Backend>(
    x: Tensor<B, 4>,
    n: usize,
    batch: usize,
    seq: usize,
    n_k: usize,
    dim: usize,
) -> Tensor<B, 4> {
    if n == 1 {
        return x;
    }
    x.unsqueeze_dim::<5>(3) // [B, S, nk, 1, D]
        .expand([batch, seq, n_k, n, dim])
        .reshape([batch, seq, n_k * n, dim])
}

// ─────────────────────────────────────────────────────────────
// SwiGLU Feed-Forward Network (identical to Qwen3)
// ─────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    /// Fused [gate | up] weight: `[hidden, 2*intermediate]`
    gate_up_proj: nn::Linear<B>,
    down_proj: nn::Linear<B>,
    intermediate_size: usize,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(hidden: usize, intermediate_size: usize, device: &Device<B>) -> Self {
        Self {
            gate_up_proj: nn::LinearConfig::new(hidden, 2 * intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: nn::LinearConfig::new(intermediate_size, hidden)
                .with_bias(false)
                .init(device),
            intermediate_size,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gu = self.gate_up_proj.forward(x);
        let [b, s, _] = gu.dims();
        let i = self.intermediate_size;
        let gate = activation::silu(gu.clone().slice([0..b, 0..s, 0..i]));
        let up = gu.slice([0..b, 0..s, i..2 * i]);
        self.down_proj.forward(gate * up)
    }
}

// ─────────────────────────────────────────────────────────────
// Hybrid decoder block
// ─────────────────────────────────────────────────────────────

/// Mixer: either full attention or GatedDeltaNet.
#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum LayerMixer<B: Backend> {
    Full(FullAttention<B>),
    Linear(GatedDeltaNet<B>),
}

/// One hybrid decoder block (prenorm residual stream).
#[derive(Module, Debug)]
pub struct HybridBlock<B: Backend> {
    pub input_ln: Qwen35RmsNorm<B>,
    pub mixer: LayerMixer<B>,
    pub post_ln: Qwen35RmsNorm<B>,
    pub mlp: FeedForward<B>,
}

impl<B: Backend> HybridBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        rope: &PartialRotaryEmbedding<B>,
        mask: Option<Tensor<B, 4>>,
        layer_cache: &mut LayerCache<B>,
        start_pos: usize,
    ) -> Tensor<B, 3> {
        // Token mixer (attention or SSM) with pre-norm + residual
        let x = {
            let residual = x.clone();
            let h = self.input_ln.forward(x);
            let h = match (&self.mixer, layer_cache) {
                (LayerMixer::Full(attn), LayerCache::Attn(cache)) => {
                    attn.forward(h, rope, mask, cache, start_pos)
                }
                (LayerMixer::Linear(dn), LayerCache::Linear(cache)) => dn.forward(h, cache),
                _ => panic!("Mismatched LayerMixer and LayerCache variant"),
            };
            h + residual
        };

        // FFN with pre-norm + residual
        let residual = x.clone();
        let h = self.post_ln.forward(x);
        let h = self.mlp.forward(h);
        h + residual
    }
}

// ─────────────────────────────────────────────────────────────
// Full Transformer
// ─────────────────────────────────────────────────────────────

/// Qwen3.5 hybrid transformer.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<HybridBlock<B>>,
    pub norm: Qwen35RmsNorm<B>,
    pub lm_head: nn::Linear<B>,
    vocab_size: usize,
}

impl<B: Backend> Transformer<B> {
    /// Build a zero-initialised transformer from a config.
    pub fn new(cfg: &Qwen35TextConfig, device: &Device<B>) -> Self {
        let hid = cfg.hidden_size;
        let eps = cfg.rms_norm_eps;

        let embed_tokens = nn::EmbeddingConfig::new(cfg.vocab_size, hid).init(device);

        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let input_ln = Qwen35RmsNorm::new(hid, eps, device);
                let post_ln = Qwen35RmsNorm::new(hid, eps, device);
                let mlp = FeedForward::new(hid, cfg.intermediate_size, device);

                let mixer = if cfg.is_full_attention(i) {
                    LayerMixer::Full(FullAttention::new(
                        hid,
                        cfg.num_attention_heads,
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                        eps,
                        device,
                    ))
                } else {
                    LayerMixer::Linear(GatedDeltaNet::new(
                        hid,
                        cfg.linear_num_key_heads,
                        cfg.linear_num_value_heads,
                        cfg.linear_key_head_dim,
                        cfg.linear_value_head_dim,
                        cfg.linear_conv_kernel_dim,
                        eps,
                        device,
                    ))
                };
                HybridBlock {
                    input_ln,
                    mixer,
                    post_ln,
                    mlp,
                }
            })
            .collect();

        let norm = Qwen35RmsNorm::new(hid, eps, device);

        let lm_head = nn::LinearConfig::new(hid, cfg.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            vocab_size: cfg.vocab_size,
        }
    }

    /// Forward pass → logits `[batch, seq_len, vocab_size]`.
    ///
    /// - `tokens`: `[batch, seq_len]`
    /// - `rope`: Partial RoPE tables
    /// - `mask`: optional 2D causal mask (unsqueezed inside)
    /// - `caches`: one `LayerCache` per layer
    /// - `start_pos`: position offset
    pub fn forward(
        &self,
        tokens: Tensor<B, 2, Int>,
        rope: &PartialRotaryEmbedding<B>,
        mask: Option<Tensor<B, 2>>,
        caches: &mut [LayerCache<B>],
        start_pos: usize,
    ) -> Tensor<B, 3> {
        let mut x = self.embed_tokens.forward(tokens);

        // Pre-unsqueeze mask: [q, kv] → [1, 1, q, kv]
        let mask4 = mask.map(|m| m.unsqueeze::<3>().unsqueeze::<4>());

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x, rope, mask4.clone(), &mut caches[i], start_pos);
        }

        x = self.norm.forward(x);
        self.lm_head.forward(x)
    }

    // ─── Weight loading helpers (called from model.rs) ────────────────────────

    pub fn load_embed_tokens(mut self, w: Tensor<B, 2>) -> Self {
        self.embed_tokens.weight = Param::from_tensor(w);
        self
    }

    pub fn load_norm(mut self, w: Tensor<B, 1>) -> Self {
        self.norm.weight = Param::from_tensor(w);
        self
    }

    pub fn load_lm_head(mut self, w: Tensor<B, 2>) -> Self {
        self.lm_head.weight = Param::from_tensor(w);
        self
    }

    /// Load weights for a **full-attention** layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_full_attn_layer(
        mut self,
        idx: usize,
        q_proj_w: Tensor<B, 2>,
        k_proj_w: Tensor<B, 2>,
        v_proj_w: Tensor<B, 2>,
        o_proj_w: Tensor<B, 2>,
        q_norm_w: Tensor<B, 1>,
        k_norm_w: Tensor<B, 1>,
        gate_up_w: Tensor<B, 2>,
        down_w: Tensor<B, 2>,
        input_ln_w: Tensor<B, 1>,
        post_ln_w: Tensor<B, 1>,
    ) -> Self {
        let layer = &mut self.layers[idx];
        if let LayerMixer::Full(ref mut attn) = layer.mixer {
            attn.q_proj.weight = Param::from_tensor(q_proj_w);
            attn.k_proj.weight = Param::from_tensor(k_proj_w);
            attn.v_proj.weight = Param::from_tensor(v_proj_w);
            attn.o_proj.weight = Param::from_tensor(o_proj_w);
            attn.q_norm.weight = Param::from_tensor(q_norm_w);
            attn.k_norm.weight = Param::from_tensor(k_norm_w);
        } else {
            panic!(
                "load_full_attn_layer called on non-full-attention layer {}",
                idx
            );
        }
        layer.mlp.gate_up_proj.weight = Param::from_tensor(gate_up_w);
        layer.mlp.down_proj.weight = Param::from_tensor(down_w);
        layer.input_ln.weight = Param::from_tensor(input_ln_w);
        layer.post_ln.weight = Param::from_tensor(post_ln_w);
        self
    }

    /// Load weights for a **GatedDeltaNet** (linear attention) layer.
    #[allow(clippy::too_many_arguments)]
    pub fn load_linear_attn_layer(
        mut self,
        idx: usize,
        in_proj_qkv_w: Tensor<B, 2>,
        in_proj_z_w: Tensor<B, 2>,
        in_proj_b_w: Tensor<B, 2>,
        in_proj_a_w: Tensor<B, 2>,
        out_proj_w: Tensor<B, 2>,
        conv1d_w: Tensor<B, 2>, // [conv_dim, K]
        a_log: Tensor<B, 1>,
        dt_bias: Tensor<B, 1>,
        norm_w: Tensor<B, 1>,
        gate_up_w: Tensor<B, 2>,
        down_w: Tensor<B, 2>,
        input_ln_w: Tensor<B, 1>,
        post_ln_w: Tensor<B, 1>,
    ) -> Self {
        let layer = &mut self.layers[idx];
        if let LayerMixer::Linear(ref mut dn) = layer.mixer {
            dn.in_proj_qkv.weight = Param::from_tensor(in_proj_qkv_w);
            dn.in_proj_z.weight = Param::from_tensor(in_proj_z_w);
            dn.in_proj_b.weight = Param::from_tensor(in_proj_b_w);
            dn.in_proj_a.weight = Param::from_tensor(in_proj_a_w);
            dn.out_proj.weight = Param::from_tensor(out_proj_w);
            dn.conv1d_weight = Param::from_tensor(conv1d_w);
            dn.a_log = Param::from_tensor(a_log);
            dn.dt_bias = Param::from_tensor(dt_bias);
            dn.norm.weight = Param::from_tensor(norm_w);
        } else {
            panic!(
                "load_linear_attn_layer called on non-linear-attention layer {}",
                idx
            );
        }
        layer.mlp.gate_up_proj.weight = Param::from_tensor(gate_up_w);
        layer.mlp.down_proj.weight = Param::from_tensor(down_w);
        layer.input_ln.weight = Param::from_tensor(input_ln_w);
        layer.post_ln.weight = Param::from_tensor(post_ln_w);
        self
    }
}

// ─────────────────────────────────────────────────────────────
// Causal mask builder
// ─────────────────────────────────────────────────────────────

/// Build a `[seq_len, total_seq_len]` causal mask using on-device integer ops.
///
/// Returns 0.0 at positions the query can attend to and `−∞` elsewhere.
pub fn build_causal_mask<B: Backend>(
    seq_len: usize,
    total_seq_len: usize,
    device: &Device<B>,
) -> Tensor<B, 2> {
    let offset = (total_seq_len - seq_len) as i64;
    let rows = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).reshape([seq_len, 1]);
    let cols =
        Tensor::<B, 1, Int>::arange(0..total_seq_len as i64, device).reshape([1, total_seq_len]);
    let mask_bool = cols.greater(rows + offset);
    Tensor::<B, 2>::zeros([seq_len, total_seq_len], device).mask_fill(mask_bool, f32::NEG_INFINITY)
}
