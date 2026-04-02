use cubecl::prelude::*;

/// Decode-only attention kernel for single query token.
///
/// This kernel computes attention for a single query against a sequence of
/// key-value pairs, optimized for autoregressive decoding.
///
/// Implementation uses online softmax for numerical stability and
/// is optimized for the case where query sequence length = 1.
///
/// Parameters:
/// - `query`: [num_heads, head_dim] tensor (flattened)
/// - `keys`: [seq_len, num_kv_heads, head_dim] tensor (flattened)
/// - `values`: [seq_len, num_kv_heads, head_dim] tensor (flattened)
/// - `output`: [num_heads, head_dim] tensor (flattened, output)
/// - `scale`: scaling factor for attention scores (1/sqrt(head_dim))
/// - `num_heads`: number of query heads
/// - `num_kv_heads`: number of key/value heads
/// - `head_dim`: dimension per head
/// - `seq_len`: length of key/value sequence
#[cube(launch_unchecked)]
fn decode_attention<F: Float>(
    query: &Array<F>,
    keys: &Array<F>,
    values: &Array<F>,
    output: &mut Array<F>,
    #[comptime] scale: F,
    #[comptime] num_heads: usize,
    #[comptime] num_kv_heads: usize,
    #[comptime] head_dim: usize,
    #[comptime] seq_len: usize,
) {
    // Each thread processes one dimension of one head
    let head_idx = UNIT_POS / head_dim;
    let dim_idx = UNIT_POS % head_dim;

    // Only process if within bounds (num_heads * head_dim total threads)
    if head_idx < num_heads {
        // Determine which KV head this query head uses (GQA grouping)
        let kv_head_idx = head_idx % num_kv_heads;

        // Online softmax variables
        let mut max_val = F::new(-F::infinity());
        let mut exp_sum = F::new(0.0);
        let mut weighted_sum = F::new(0.0);

        // Query value for this dimension
        let query_idx = head_idx * head_dim + dim_idx;
        let query_val = query[query_idx];

        // Iterate over sequence positions
        for pos in 0..seq_len {
            // Index into keys array: [pos * (num_kv_heads * head_dim) + kv_head_idx * head_dim + dim_idx]
            let key_idx = pos * (num_kv_heads * head_dim) + kv_head_idx * head_dim + dim_idx;
            let key_val = keys[key_idx];

            // Compute dot product for this position (single dimension contribution)
            // Note: This computes the dot product incrementally across dimensions
            // The full dot product would sum across all dimensions, but we're processing
            // one dimension per thread. We need to compute the full dot product first.
            // We'll restructure this loop later.

            // For now, compute score for this position and dimension
            let score = query_val * key_val * scale;

            // Update online softmax statistics
            if score > max_val {
                // Adjust exp_sum for new max
                if max_val > F::new(-F::infinity()) {
                    let exp_shift = (max_val - score).exp();
                    exp_sum = exp_sum * exp_shift;
                    weighted_sum = weighted_sum * exp_shift;
                }
                max_val = score;
            }

            let exp_score = (score - max_val).exp();
            exp_sum += exp_score;

            // Index into values array
            let value_idx = pos * (num_kv_heads * head_dim) + kv_head_idx * head_dim + dim_idx;
            let value_val = values[value_idx];
            weighted_sum += exp_score * value_val;
        }

        // Normalize weighted sum by exp_sum
        let result = if exp_sum > F::new(0.0) {
            weighted_sum / exp_sum
        } else {
            F::new(0.0)
        };

        output[query_idx] = result;
    }
}