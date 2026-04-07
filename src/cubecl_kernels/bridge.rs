//! Bridge between burn fusion tensors and the custom CubeCL flash-decode kernel.

use burn::backend::wgpu::WgpuRuntime;
use burn::backend::Wgpu;
use burn::prelude::Backend;
use burn::tensor::{Element, Shape, Tensor, TensorPrimitive};
use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor, CubeBackend};
use burn_fusion::{stream::{Operation, OperationStreams}, FusionBackend, FusionRuntime};
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};

use super::flash_decode::launch_flash_decode;

type BaseWgpu = CubeBackend<WgpuRuntime, f32, i32, u32>;
type FusionWgpuRuntime = <BaseWgpu as FusionBackend>::FusionRuntime;

#[derive(Clone, Debug)]
struct FlashDecodeOp {
    desc: CustomOpIr,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
}

impl FlashDecodeOp {
    fn new(
        desc: CustomOpIr,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Self {
        Self {
            desc,
            num_heads,
            num_kv_heads,
            head_dim,
            seq_len,
        }
    }
}

impl Operation<FusionWgpuRuntime> for FlashDecodeOp {
    fn execute(
        &self,
        handles: &mut HandleContainer<<FusionWgpuRuntime as FusionRuntime>::FusionHandle>,
    ) {
        let ([query, keys, values], [output]) = self.desc.as_fixed();
        let query = handles.get_float_tensor::<BaseWgpu>(query);
        let keys = handles.get_float_tensor::<BaseWgpu>(keys);
        let values = handles.get_float_tensor::<BaseWgpu>(values);

        let output_tensor = flash_decode_primitive(
            query,
            keys,
            values,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.seq_len,
        );

        handles.register_float_tensor::<BaseWgpu>(&output.id, output_tensor);
    }
}

fn flash_decode_primitive(
    query: CubeTensor<WgpuRuntime>,
    keys: CubeTensor<WgpuRuntime>,
    values: CubeTensor<WgpuRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> CubeTensor<WgpuRuntime> {
    let query = into_contiguous(query);
    let keys = into_contiguous(keys);
    let values = into_contiguous(values);

    let out_shape = Shape::new([1, num_heads, 1, head_dim]);
    let out_handle = query
        .client
        .empty(out_shape.num_elements() * core::mem::size_of::<f32>());

    unsafe {
        launch_flash_decode::<WgpuRuntime>(
            &query.client,
            query.handle.clone(),
            keys.handle.clone(),
            values.handle.clone(),
            out_handle.clone(),
            num_heads as u32,
            num_kv_heads as u32,
            head_dim as u32,
            seq_len as u32,
        );
    }

    CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        out_shape,
        out_handle,
        <BaseWgpu as Backend>::FloatElem::dtype(),
    )
}

/// Run flash-decode attention on the Wgpu backend.
///
/// Replaces the generic matmul attention path for decode (seq_len=1).
///
/// # Arguments
/// - `q`: query tensor `[batch, num_heads, 1, head_dim]`
/// - `k`: cached keys `[batch, num_kv_heads, seq_len, head_dim]`
/// - `v`: cached values `[batch, num_kv_heads, seq_len, head_dim]`
/// - `num_heads`, `num_kv_heads`, `head_dim`: model constants
///
/// # Returns
/// `[batch, num_heads, 1, head_dim]` — attention output
///
/// # Panics
/// - If batch != 1 (current kernel only supports single-batch decode)
/// - If q seq dimension != 1
pub fn flash_decode_wgpu(
    q: Tensor<Wgpu, 4>,
    k: Tensor<Wgpu, 4>,
    v: Tensor<Wgpu, 4>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Tensor<Wgpu, 4> {
    let [batch, _h, q_seq, _d] = q.dims();
    let [_, _, kv_seq, _] = k.dims();
    assert_eq!(batch, 1, "flash_decode only supports batch=1");
    assert_eq!(q_seq, 1, "flash_decode only supports seq_len=1 query");

    let q_flat = q.reshape([num_heads * head_dim]).into_primitive().tensor();
    let k_flat = k
        .reshape([num_kv_heads * kv_seq * head_dim])
        .into_primitive()
        .tensor();
    let v_flat = v
        .reshape([num_kv_heads * kv_seq * head_dim])
        .into_primitive()
        .tensor();

    let client = q_flat.client.clone();
    let streams = OperationStreams::with_inputs([&q_flat, &k_flat, &v_flat]);
    let out = TensorIr::uninit(
        client.create_empty_handle(),
        Shape::new([1, num_heads, 1, head_dim]),
        <BaseWgpu as Backend>::FloatElem::dtype(),
    );

    let desc = CustomOpIr::new(
        "flash_decode_attention",
        &[q_flat.into_ir(), k_flat.into_ir(), v_flat.into_ir()],
        &[out],
    );

    let out = client
        .register(
            streams,
            OperationIr::Custom(desc.clone()),
            FlashDecodeOp::new(desc, num_heads, num_kv_heads, head_dim, kv_seq),
        )
        .output();

    Tensor::<Wgpu, 4>::from_primitive(TensorPrimitive::Float(out))
}
