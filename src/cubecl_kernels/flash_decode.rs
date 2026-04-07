//! Flash-decode attention kernel for single-query (seq_len=1) GQA attention.
//!
//! Fuses q·K^T scaled dot-product + softmax + V accumulation into a single GPU kernel,
//! avoiding the expensive generic matmul path and the `repeat_kv` memory expansion.
//!
//! Thread mapping:
//! - One workgroup (cube) per query head: `CUBE_POS_X` = head index
//! - Within each workgroup, `head_dim` threads (e.g. 256): each thread owns one element
//!   of the head dimension
//!
//! Algorithm per workgroup (one query head):
//! 1. Each thread loads its element of q[head, dim]
//! 2. For each KV position t in 0..seq_len:
//!    a. Each thread computes q[dim] * k[t, kv_head, dim] (partial dot product)
//!    b. Tree reduction in shared memory to get full dot product score
//!    c. Thread 0 computes online softmax update (running max, exp_sum)
//!    d. Broadcast exp_weight to all threads
//!    e. Each thread accumulates: acc[dim] += exp_weight * v[t, kv_head, dim]
//! 3. Normalize: output[dim] = acc[dim] / exp_sum
//!
//! GQA: kv_head_idx = head_idx / group_size

use cubecl::prelude::*;

/// Flash-decode attention kernel.
///
/// Inputs (all flattened `Array<F>`):
/// - `query`:  `[num_heads * head_dim]` — single query token
/// - `keys`:   `[num_kv_heads * seq_len * head_dim]` — cached keys (kv_heads-major)
/// - `values`: `[num_kv_heads * seq_len * head_dim]` — cached values (kv_heads-major)
/// - `output`: `[num_heads * head_dim]` — output (written by kernel)
///
/// Runtime scalars:
/// - `seq_len`: current KV cache sequence length (runtime to avoid recompilation)
///
/// Comptime parameters:
/// - `num_heads`, `num_kv_heads`, `head_dim`: model constants (usize for cubecl compatibility)
#[cube(launch)]
fn flash_decode_attention<F: Float>(
    query: &Array<F>,
    keys: &Array<F>,
    values: &Array<F>,
    output: &mut Array<F>,
    seq_len: u32,
    #[comptime] num_heads: usize,
    #[comptime] num_kv_heads: usize,
    #[comptime] head_dim: usize,
) {
    // Which query head this workgroup processes
    let head_idx = CUBE_POS_X;
    // Which dimension element this thread owns
    let dim_idx = UNIT_POS_X;

    // GQA: map query head → KV head
    let group_size = comptime!(num_heads / num_kv_heads) as u32;
    let kv_head_idx = head_idx / group_size;

    // Scale factor: 1/sqrt(head_dim)
    let scale_val = comptime!(1.0f32 / (head_dim as f32).sqrt());
    let scale = F::new(scale_val);

    // Comptime head_dim as u32 for arithmetic with CUBE_POS_X
    let hd = comptime!(head_dim) as u32;

    // Load this thread's query element
    let q_val = query[(head_idx * hd + dim_idx) as usize];

    // Shared memory for tree reduction of dot products
    let mut smem = SharedMemory::<F>::new(head_dim);
    // Shared memory for broadcasting softmax state from thread 0
    let mut softmax_state = SharedMemory::<F>::new(3usize); // [rescale, exp_weight, final_sum]

    // Online softmax accumulators (per-thread)
    let mut acc = F::new(0.0); // weighted value accumulator for this dim
    let mut running_max = F::new(-1e30);
    let mut running_sum = F::new(0.0);

    // Iterate over all KV positions
    for t in 0..seq_len {
        // --- Phase 1: Compute q·k dot product via tree reduction ---

        // Each thread computes its partial product
        // KV layout: [kv_heads, seq, dim] → idx = kv_head * (seq_len * hd) + t * hd + dim
        let k_idx = (kv_head_idx * seq_len * hd + t * hd + dim_idx) as usize;
        let partial = q_val * keys[k_idx];
        smem[dim_idx as usize] = partial;
        sync_cube();

        // Tree reduction: sum up all head_dim elements
        // After this, smem[0] contains the full dot product
        // Use CUBE_DIM_X (runtime) so stride is a runtime variable that can be reassigned.
        let mut stride = CUBE_DIM_X / 2u32;
        while stride > 0u32 {
            if dim_idx < stride {
                smem[dim_idx as usize] =
                    smem[dim_idx as usize] + smem[(dim_idx + stride) as usize];
            }
            sync_cube();
            stride = stride / 2u32;
        }

        // --- Phase 2: Online softmax update (thread 0 computes, broadcasts) ---
        if dim_idx == 0 {
            let score = smem[0usize] * scale;

            if score > running_max {
                // New max: rescale previous accumulator
                let shift = F::exp(running_max - score);
                running_sum = running_sum * shift;
                // Store rescale factor for other threads
                softmax_state[0usize] = shift; // rescale_factor for acc
                running_max = score;
            } else {
                softmax_state[0usize] = F::new(1.0); // no rescale needed
            }

            let exp_score = F::exp(score - running_max);
            running_sum = running_sum + exp_score;

            // Broadcast to all threads
            softmax_state[1usize] = exp_score; // weight for this position's value
        }
        sync_cube();

        // --- Phase 3: Update value accumulator ---
        let rescale = softmax_state[0usize];
        let exp_weight = softmax_state[1usize];

        // Rescale previous accumulator if max changed
        acc = acc * rescale;

        // Accumulate weighted value
        let v_idx = (kv_head_idx * seq_len * hd + t * hd + dim_idx) as usize;
        acc = acc + exp_weight * values[v_idx];
    }

    // --- Phase 4: Normalize and write output ---
    // Broadcast final running_sum from thread 0
    if dim_idx == 0 {
        softmax_state[2usize] = running_sum;
    }
    sync_cube();

    let final_sum = softmax_state[2usize];
    let out_idx = (head_idx * hd + dim_idx) as usize;

    if final_sum > F::new(0.0) {
        output[out_idx] = acc / final_sum;
    } else {
        output[out_idx] = F::new(0.0);
    }
}

/// Launch the flash-decode attention kernel.
///
/// # Safety
/// Caller must ensure handles point to valid GPU buffers with correct sizes.
///
/// # Arguments
/// - `client`: CubeCL compute client
/// - `q_handle`: query buffer `[num_heads * head_dim]` floats
/// - `k_handle`: keys buffer `[seq_len * num_kv_heads * head_dim]` floats
/// - `v_handle`: values buffer `[seq_len * num_kv_heads * head_dim]` floats
/// - `out_handle`: output buffer `[num_heads * head_dim]` floats
/// - `num_heads`, `num_kv_heads`, `head_dim`: model shape constants
/// - `seq_len`: current KV cache sequence length (runtime)
pub unsafe fn launch_flash_decode<R: Runtime>(
    client: &ComputeClient<R>,
    q_handle: cubecl_runtime::server::Handle,
    k_handle: cubecl_runtime::server::Handle,
    v_handle: cubecl_runtime::server::Handle,
    out_handle: cubecl_runtime::server::Handle,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
) {
    let q_len = (num_heads * head_dim) as usize;
    let kv_len = (seq_len * num_kv_heads * head_dim) as usize;

    flash_decode_attention::launch::<f32, R>(
        client,
        CubeCount::Static(num_heads, 1, 1),
        CubeDim::new_1d(head_dim),
        ArrayArg::from_raw_parts(q_handle, q_len),
        ArrayArg::from_raw_parts(k_handle, kv_len),
        ArrayArg::from_raw_parts(v_handle, kv_len),
        ArrayArg::from_raw_parts(out_handle, q_len),
        seq_len,
        num_heads as usize,
        num_kv_heads as usize,
        head_dim as usize,
    );
}
