use burn::tensor::{backend::Backend, Device, Tensor};

/// Key-value cache for autoregressive generation.
///
/// Stores cached key or value tensors with shape `[batch_size, num_heads, seq_len, head_dim]`.
pub struct KvCache<B: Backend> {
    cache: Tensor<B, 4>,
    max_seq_len: usize,
    cur_seq_len: usize,
}

impl<B: Backend> KvCache<B> {
    /// Creates a new empty cache.
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            cache: Tensor::empty([batch_size, num_heads, max_seq_len, head_dim], device),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        self.cache = Tensor::empty(self.cache.shape(), &self.cache.device());
        self.cur_seq_len = 0;
    }

    /// Append new key/value tensor to the cache and return the accumulated result.
    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, head_dim] = tensor.dims();

        if seq_len >= self.max_seq_len {
            let start = seq_len - self.max_seq_len;
            let truncated =
                tensor.slice([0..batch_size, 0..num_heads, start..seq_len, 0..head_dim]);
            // Use take+replace to keep refcount=1, allowing potential in-place write
            let old = std::mem::replace(
                &mut self.cache,
                Tensor::empty([batch_size, num_heads, self.max_seq_len, head_dim], &truncated.device()),
            );
            let updated = old.slice_assign(
                [0..batch_size, 0..num_heads, 0..self.max_seq_len, 0..head_dim],
                truncated,
            );
            self.cur_seq_len = self.max_seq_len;
            let valid = updated.clone().slice([
                0..batch_size, 0..num_heads, 0..self.max_seq_len, 0..head_dim,
            ]);
            self.cache = updated;
            return valid;
        }

        let mut new_seq_len = self.cur_seq_len + seq_len;

        if new_seq_len > self.max_seq_len {
            // Shift cache left by seq_len positions
            let keep = self.max_seq_len - seq_len;
            let prev_slice = self.cache.clone().slice([
                0..batch_size, 0..num_heads, seq_len..self.max_seq_len, 0..head_dim,
            ]);
            let old = std::mem::replace(
                &mut self.cache,
                Tensor::empty([batch_size, num_heads, self.max_seq_len, head_dim], &prev_slice.device()),
            );
            let shifted = old.slice_assign(
                [0..batch_size, 0..num_heads, 0..keep, 0..head_dim],
                prev_slice,
            );
            self.cache = shifted;
            self.cur_seq_len = keep;
            new_seq_len = self.max_seq_len;
        }

        let write_start = self.cur_seq_len;
        let old = std::mem::replace(
            &mut self.cache,
            Tensor::empty([batch_size, num_heads, self.max_seq_len, head_dim], &tensor.device()),
        );
        let updated = old.slice_assign(
            [0..batch_size, 0..num_heads, write_start..new_seq_len, 0..head_dim],
            tensor,
        );
        self.cur_seq_len += seq_len;
        let valid = updated.clone().slice([
            0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..head_dim,
        ]);
        self.cache = updated;
        valid
    }

    /// Returns the current cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }

    pub fn is_empty(&self) -> bool {
        self.cur_seq_len == 0
    }
}

/// KV-cache pair (key + value) for a full-attention layer.
pub struct AttentionCache<B: Backend> {
    pub k_cache: KvCache<B>,
    pub v_cache: KvCache<B>,
}

impl<B: Backend> AttentionCache<B> {
    pub fn new(
        batch_size: usize,
        num_kv_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            k_cache: KvCache::new(batch_size, num_kv_heads, max_seq_len, head_dim, device),
            v_cache: KvCache::new(batch_size, num_kv_heads, max_seq_len, head_dim, device),
        }
    }

    pub fn reset(&mut self) {
        self.k_cache.reset();
        self.v_cache.reset();
    }

    pub fn seq_len(&self) -> usize {
        self.k_cache.len()
    }
}

/// State cache for a GatedDeltaNet (linear attention) layer.
///
/// Stores:
/// - `conv_state`: the last `conv_kernel_size` inputs to the causal conv1d,
///   shape `[batch, conv_channels, conv_kernel_size]`.
/// - `recurrent_state`: the gated delta-rule recurrent state,
///   shape `[batch, num_v_heads, head_k_dim, head_v_dim]`.
pub struct DeltaNetCache<B: Backend> {
    /// [batch, conv_dim, conv_kernel_size]
    pub conv_state: Tensor<B, 3>,
    /// [batch, num_v_heads, head_k_dim, head_v_dim]
    pub recurrent_state: Tensor<B, 4>,
    pub batch: usize,
    pub conv_dim: usize,
    pub conv_kernel: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    /// Number of tokens processed so far (used for detecting prefill vs decode).
    pub tokens_seen: usize,
}

impl<B: Backend> DeltaNetCache<B> {
    pub fn new(
        batch: usize,
        conv_dim: usize,
        conv_kernel: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            conv_state: Tensor::zeros([batch, conv_dim, conv_kernel], device),
            recurrent_state: Tensor::zeros([batch, num_v_heads, head_k_dim, head_v_dim], device),
            batch,
            conv_dim,
            conv_kernel,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tokens_seen: 0,
        }
    }

    pub fn reset(&mut self) {
        let device = self.conv_state.device();
        self.conv_state =
            Tensor::zeros([self.batch, self.conv_dim, self.conv_kernel], &device);
        self.recurrent_state = Tensor::zeros(
            [self.batch, self.num_v_heads, self.head_k_dim, self.head_v_dim],
            &device,
        );
        self.tokens_seen = 0;
    }
}

/// Per-layer cache — either a full-attention KV cache or a GatedDeltaNet state.
pub enum LayerCache<B: Backend> {
    Attn(AttentionCache<B>),
    Linear(DeltaNetCache<B>),
}

impl<B: Backend> LayerCache<B> {
    pub fn reset(&mut self) {
        match self {
            LayerCache::Attn(c) => c.reset(),
            LayerCache::Linear(c) => c.reset(),
        }
    }
}
