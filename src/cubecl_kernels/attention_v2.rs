use cubecl::prelude::*;

/// Launch function for testing the decode attention kernel.
pub fn launch_test<R: Runtime>(device: &R::Device) {
    let client = R::client(device);

    // Test with small parameters
    let num_heads = 4;
    let num_kv_heads = 2;
    let head_dim = 256;
    let seq_len = 10;
    let vector_size = 4; // Assuming Line<f32>::size() == 4

    let vectors_per_head = head_dim / vector_size;
    let total_query_size = num_heads * vectors_per_head;
    let total_kv_size = seq_len * num_kv_heads * vectors_per_head;

    // Create dummy data
    let query_data: Vec<f32> = (0..num_heads * head_dim)
        .map(|i| (i % 5) as f32 * 0.1)
        .collect();
    let keys_data: Vec<f32> = (0..seq_len * num_kv_heads * head_dim)
        .map(|i| (i % 7) as f32 * 0.05)
        .collect();
    let values_data: Vec<f32> = (0..seq_len * num_kv_heads * head_dim)
        .map(|i| (i % 9) as f32 * 0.03)
        .collect();

    // Note: This is a placeholder launch function
    // Actual implementation would need to handle Array<Line<F>> data layout
    println!("Decode attention kernel test placeholder");
    println!("Parameters: num_heads={}, num_kv_heads={}, head_dim={}, seq_len={}",
             num_heads, num_kv_heads, head_dim, seq_len);
}

/// Vectorized decode-only attention kernel using Line types for SIMD optimization.
///
/// This kernel computes attention for a single query token using vectorized
/// operations for better memory throughput and compute efficiency.
///
/// Assumes head_dim is divisible by vector_size (typically 4, 8, or 16).
///
/// Parameters:
/// - `query`: [num_heads, head_dim] as Array<Line<F>>
/// - `keys`: [seq_len, num_kv_heads, head_dim] as Array<Line<F>>
/// - `values`: [seq_len, num_kv_heads, head_dim] as Array<Line<F>>
/// - `output`: [num_heads, head_dim] as Array<Line<F>> (output)
/// - `scale`: scaling factor (1/sqrt(head_dim))
/// - `num_heads`: number of query heads
/// - `num_kv_heads`: number of key/value heads
/// - `head_dim`: dimension per head
/// - `seq_len`: length of key/value sequence
/// - `vector_size`: size of Line vectors (compiler-determined)
#[cube(launch_unchecked)]
fn decode_attention_vectorized<F: Float>(
    query: &Array<Line<F>>,
    keys: &Array<Line<F>>,
    values: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] scale: F,
    #[comptime] num_heads: usize,
    #[comptime] num_kv_heads: usize,
    #[comptime] head_dim: usize,
    #[comptime] seq_len: usize,
) {
    // Each thread processes one vector of one head
    // head_dim is divided into vectors of size Line<F>::size()
    let vectors_per_head = head_dim / Line::<F>::size();
    let total_vectors = num_heads * vectors_per_head;

    let vector_idx = UNIT_POS as usize;

    if vector_idx < total_vectors {
        let head_idx = vector_idx / vectors_per_head;
        let vector_in_head = vector_idx % vectors_per_head;

        // Determine which KV head this query head uses (GQA grouping)
        let kv_head_idx = head_idx % num_kv_heads;

        // Online softmax variables
        let mut max_val = F::new(-F::infinity());
        let mut exp_sum = F::new(0.0);
        let mut weighted_sum = Line::<F>::new(F::new(0.0));

        // Get query vector for this position
        let query_vector = query[vector_idx];

        // Iterate over sequence positions
        for pos in 0..seq_len {
            // Calculate index for key vector at this position
            // Layout: [pos][kv_head_idx][vector_in_head]
            let key_idx = pos * (num_kv_heads * vectors_per_head) +
                          kv_head_idx * vectors_per_head +
                          vector_in_head;

            let key_vector = keys[key_idx];

            // Compute dot product between query_vector and key_vector
            // This is a vector dot product (sum of element-wise products)
            let mut dot_product = F::new(0.0);

            // Manually unroll the dot product computation
            // Line<F> doesn't have a dot product method, so we compute it
            for i in 0..Line::<F>::size() {
                let q_elem = query_vector.extract(i);
                let k_elem = key_vector.extract(i);
                dot_product = dot_product + q_elem * k_elem;
            }

            // Apply scaling
            let score = dot_product * scale;

            // Update online softmax statistics
            if score > max_val {
                // Adjust exp_sum for new max
                if max_val > F::new(-F::infinity()) {
                    let exp_shift = (max_val - score).exp();
                    exp_sum = exp_sum * exp_shift;
                    weighted_sum = weighted_sum * Line::new(exp_shift);
                }
                max_val = score;
            }

            let exp_score = (score - max_val).exp();
            exp_sum += exp_score;

            // Get value vector at this position
            let value_idx = pos * (num_kv_heads * vectors_per_head) +
                            kv_head_idx * vectors_per_head +
                            vector_in_head;
            let value_vector = values[value_idx];

            weighted_sum = weighted_sum + value_vector * Line::new(exp_score);
        }

        // Normalize weighted sum by exp_sum
        let result_vector = if exp_sum > F::new(0.0) {
            weighted_sum / Line::new(exp_sum)
        } else {
            Line::new(F::new(0.0))
        };

        output[vector_idx] = result_vector;
    }
}