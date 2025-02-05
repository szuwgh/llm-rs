mod common;
mod tokenizer;
mod unicode;
use crate::common::BinarySerialize;
use crate::common::GGufRead;
use crate::common::LLMError;
use crate::common::LLMResult;
use crate::meta::GGufContext;
use crate::meta::GGufHeader;
use crate::meta::GGufMetadataValue;
use crate::meta::GGufStr;
use crate::meta::GGufVersion;
use crate::meta::LLamaVocab;
use crate::meta::LlamaBatch;
use crate::meta::LlamaHparams;
use crate::meta::LlamaKvCache;
use crate::tokenizer::replace_all;
use crate::tokenizer::LLMtokenizerSpm;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use galois::cuda::CudaDevice;
use galois::kernels::init_cuda_function;
use galois::op::RopeCustomOption;
use galois::op::UnaryOp;
use galois::Device;
use galois::TensorProto;
use galois::TensorView;
use memmap2::Mmap;
use std::collections::HashMap;
use tokenizer::tokenize;
mod meta;
use std::f32::INFINITY;
use std::io::BufReader;

use std::io::Read;
use std::vec;
type Endian = LittleEndian;
use galois::shape::MAX_DIM;
use galois::GGmlType;
use galois::Shape;
use galois::Tensor;
use galois::GS_BLCK_SIZE;
use galois::GS_TYPE_SIZE;
use lazy_static::lazy_static;
use std::fs::File;
use std::fs::OpenOptions;

fn get_blck_size(t: GGmlType) -> usize {
    return GS_BLCK_SIZE[t as usize];
}

fn get_type_size(t: GGmlType) -> usize {
    return GS_TYPE_SIZE[t as usize];
}

fn get_type_sizef(t: GGmlType) -> f32 {
    return (GS_TYPE_SIZE[t as usize]) as f32 / GS_BLCK_SIZE[t as usize] as f32;
}

lazy_static! {
    static ref LLAMA_N_PARTS: HashMap<i32, usize> = {
        let mut map = HashMap::new();
        map.insert(4096, 1);
        map.insert(5120, 2);
        map.insert(6656, 4);
        map.insert(8192, 8);
        map
    };
}

macro_rules! function {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);
        name.strip_suffix("::f").unwrap()
    }};
}

struct TensorContext {
    offset: usize,
    size: usize,
    n_objects: usize,
    // tensor_infos: HashMap<GGufStr, GGufDiskTensorInfo>,
}

impl TensorContext {
    fn new() -> TensorContext {
        TensorContext {
            offset: 0,
            size: 0,
            n_objects: 0,
            //tensor_infos: tensor_infos,
        }
    }

    // fn slice(&self, offset: usize, size: usize) -> &[u8] {
    //     &self.buf[offset..offset + size]
    // }
}

fn create_tensor<'a, const N: usize, T: TensorProto>(
    buf: &'a [u8],
    tensor_infos: &HashMap<GGufStr, GGufDiskTensorInfo>,
    name: &str,
    ne: [usize; N],
    dev: &Device,
) -> LLMResult<T> {
    let tensor_info = tensor_infos.get(name).unwrap();
    let dtype = tensor_info.typ;
    let cur_offset = tensor_info.offset as usize;
    let mut size_needed: usize = get_type_size(dtype) * (ne[0] / get_blck_size(dtype));
    let n_dims = ne.len();
    for i in 1..n_dims {
        size_needed *= ne[i];
    }
    size_needed = ggml_pad(size_needed, GGUF_DEFAULT_ALIGNMENT);
    assert!(ne.len() == tensor_info.n_dims as usize);
    for i in 0..ne.len() {
        assert!(ne[i] == tensor_info.dimensions[i]);
    }
    assert!(size_needed == tensor_info.size as usize);
    println!(
        "name:{},cur_offset:{},size_needed:{},get_type_sizef(dtype):{}",
        name,
        cur_offset,
        size_needed,
        get_type_sizef(dtype) as usize
    );
    if cur_offset + size_needed > buf.len() {
        return Err(LLMError::NotEnoughSpace);
    }
    let t = unsafe {
        T::from_bytes(
            &buf[cur_offset..cur_offset + size_needed],
            n_dims,
            Shape::from_array(ne),
            dtype,
            dev,
        )
    }?;
    Ok(t)
}

fn new_tensor_1d<'a>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    dtype: GGmlType,
    ne0: usize,
    dev: &Device,
) -> LLMResult<TensorView<'a>> {
    let dim = [ne0];
    new_tensor(ctx, buf, 1, dtype, Shape::from_array(dim), dev)
}

fn new_tensor_2d<'a>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    dtype: GGmlType,
    ne0: usize,
    ne1: usize,
    dev: &Device,
) -> LLMResult<TensorView<'a>> {
    let dim = [ne0, ne1];
    new_tensor(ctx, buf, 2, dtype, Shape::from_array(dim), dev)
}

fn new_tensor_3d<'a>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    dtype: GGmlType,
    ne0: usize,
    ne1: usize,
    ne2: usize,
    dev: &Device,
) -> LLMResult<TensorView<'a>> {
    let dim = [ne0, ne1, ne2];
    new_tensor(ctx, buf, 3, dtype, Shape::from_array(dim), dev)
}

fn new_tensor_4d<'a, T: TensorProto>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    dtype: GGmlType,
    ne0: usize,
    ne1: usize,
    ne3: usize,
    ne4: usize,
    dev: &Device,
) -> LLMResult<T> {
    let dim = [ne0, ne1, ne3, ne4];
    new_tensor(ctx, buf, 2, dtype, Shape::from_array(dim), dev)
}

fn new_f32_tensor<'a>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    value: f32,
    dev: &Device,
) -> LLMResult<TensorView<'a>> {
    let mut result = new_tensor_1d(ctx, buf, GGmlType::F32, 1, dev)?;
    result.set_value(value);
    Ok(result)
}

fn new_tensor<'a, T: TensorProto>(
    ctx: &mut TensorContext,
    buf: &[u8],
    n_dims: usize,
    dtype: GGmlType,
    shape: Shape,
    dev: &Device,
) -> LLMResult<T> {
    let cur_offset = ctx.offset;
    let cur_size = ctx.size;
    let ne = shape.layout();
    let mut size_needed: usize = get_type_size(dtype) * (ne[0]);
    for i in 1..n_dims {
        size_needed *= ne[i];
    }
    //  size_needed = ggml_pad(size_needed, GGUF_DEFAULT_ALIGNMENT);
    // println!(
    //     "size_needed:{},get_type_size(dtype):{}",
    //     size_needed,
    //     get_type_sizef(dtype) as usize
    // );
    if cur_offset + size_needed > buf.len() {
        return Err(LLMError::NotEnoughSpace);
    }
    let t = unsafe {
        T::from_bytes(
            &buf[cur_offset..cur_offset + size_needed],
            n_dims,
            shape,
            dtype,
            dev,
        )
    }?;
    ctx.offset = cur_offset + size_needed;
    ctx.size = size_needed;
    ctx.n_objects += 1;
    Ok(t)
}

fn view_tensor<T: TensorProto>(
    //buf: &[u8],
    a: &T,
    offset: usize,
    n_dims: usize,
    dtype: GGmlType,
    shape: Shape,
    //dev: &Device,
) -> LLMResult<TensorView<'_>> {
    // Ok(unsafe { TensorView::from_bytes(buf, n_dims, shape, dtype, dev)? })
    let v = a.view_tensor(offset, n_dims, dtype, shape);
    Ok(v)
}

fn dup_tensor<'a, T: TensorProto, R: TensorProto>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    a: &T,
) -> LLMResult<R> {
    let dtype = a.dtype();
    let shape = Shape::from_slice(a.dim().shape());
    new_tensor(ctx, buf, a.n_dims(), dtype, shape, a.device())
}

fn view_1d<'a, T: TensorProto>(a: &'a T, ne0: usize, offset: usize) -> LLMResult<TensorView<'a>> {
    let dtype = a.dtype();
    // let buf = a.as_bytes();
    let shape = Shape::from_array([ne0]);
    view_tensor(a, offset, 1, dtype, shape)
}

fn view_2d<'a, T: TensorProto>(
    a: &'a T,
    ne0: usize,
    ne1: usize,
    nb1: usize,
    offset: usize,
) -> LLMResult<TensorView<'a>> {
    let dtype = a.dtype();
    // let buf = a.as_bytes();
    let shape = Shape::from_array([ne0, ne1]);
    let mut t = view_tensor(a, offset, 2, dtype, shape)?;
    let nb0 = t.dim().stride_1d();
    let nb = [nb0, nb1, nb1 * ne1, nb1 * ne1];
    t.ret_stride(nb);
    Ok(t)
}

fn view_3d<'a, T: TensorProto>(
    a: &'a T,
    ne0: usize,
    ne1: usize,
    ne2: usize,
    nb1: usize,
    nb2: usize,
    offset: usize,
) -> LLMResult<TensorView<'a>> {
    let dtype = a.dtype();
    //let buf = a.as_bytes();
    let shape = Shape::from_array([ne0, ne1, ne2]);
    let mut t = view_tensor(a, offset, 3, dtype, shape)?;
    let nb = t.stride_layout_mut();
    nb[1] = nb1;
    nb[2] = nb2;
    nb[3] = nb[2] * ne2;
    Ok(t)
}

fn transpose<'a>(a: &'a mut TensorView<'a>) -> LLMResult<TensorView<'a>> {
    Ok(a.transpose(0, 1)?)
}

fn permute<'a, T: TensorProto>(
    a: T,
    axis0: usize,
    axis1: usize,
    axis2: usize,
    axis3: usize,
) -> LLMResult<T> {
    assert!(axis0 < MAX_DIM);
    assert!(axis1 < MAX_DIM);
    assert!(axis2 < MAX_DIM);
    assert!(axis3 < MAX_DIM);

    assert!(axis0 != axis1);
    assert!(axis0 != axis2);
    assert!(axis0 != axis3);
    assert!(axis1 != axis2);
    assert!(axis1 != axis3);
    assert!(axis2 != axis3);
    let mut dst = a;

    // let mut dst = view_tensor(
    //     a,
    //     0,
    //     a.n_dims(),
    //     a.dtype(),
    //     Shape::from_slice(a.shape_layout()),
    // )?;

    let mut ne = [0usize; MAX_DIM];
    let mut nb = [0usize; MAX_DIM];

    let (ne0, ne1, ne2, ne3) = dst.dim4();
    let (nb0, nb1, nb2, nb3) = dst.stride4();

    ne[axis0] = ne0;
    ne[axis1] = ne1;
    ne[axis2] = ne2;
    ne[axis3] = ne3;

    nb[axis0] = nb0;
    nb[axis1] = nb1;
    nb[axis2] = nb2;
    nb[axis3] = nb3;

    {
        let shape = dst.shape_layout_mut();
        shape[0] = ne[0];
        shape[1] = ne[1];
        shape[2] = ne[2];
        shape[3] = ne[3];
    }

    {
        let stride = dst.stride_layout_mut();
        stride[0] = nb[0];
        stride[1] = nb[1];
        stride[2] = nb[2];
        stride[3] = nb[3];
    }

    Ok(dst)
}

fn cpy<'a, X: TensorProto, Y: TensorProto>(src: &X, dst: &mut Y) -> LLMResult<()> {
    // let mut dst = cur.view();
    galois::op::galois_cpy(src, dst)?;
    Ok(())
}

fn cont_4d<'a, 'b: 'a, T: TensorProto>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    a: &T,
    ne0: usize,
    ne1: usize,
    ne2: usize,
    ne3: usize,
) -> LLMResult<T> {
    let mut dst = new_tensor_4d(ctx, buf, a.dtype(), ne0, ne1, ne2, ne3, a.device())?;
    galois::op::galois_cont(a, &mut dst)?;
    Ok(dst)
}

fn cont_2d<'a, 'b: 'a, T: TensorProto>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    a: &T,
    ne0: usize,
    ne1: usize,
) -> LLMResult<T> {
    cont_4d(ctx, buf, a, ne0, ne1, 1, 1)
}

fn unary<'a, T: TensorProto>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    a: &T,
    op: UnaryOp,
) -> LLMResult<T> {
    let mut dst = dup_tensor(ctx, buf, a)?;
    galois::op::galois_unary(a, &mut dst, op)?;
    Ok(dst)
}

fn silu<'a, T: TensorProto>(ctx: &mut TensorContext, buf: &'a [u8], a: &T) -> LLMResult<T> {
    unary(ctx, buf, a, UnaryOp::Silu)
}

fn reshape_2d<'a, T: TensorProto>(a: &'a T, ne0: usize, ne1: usize) -> LLMResult<TensorView<'a>> {
    assert!(a.ggml_is_contiguous());
    assert!(a.elem_count() == ne0 * ne1);
    let ne: [usize; 2] = [ne0, ne1];
    let result = view_tensor(a, 0, 2, a.dtype(), Shape::from_array(ne))?;
    Ok(result)
}

fn reshape_3d<'a, T: TensorProto>(
    a: &'a T,
    ne0: usize,
    ne1: usize,
    ne2: usize,
) -> LLMResult<TensorView<'a>> {
    assert!(a.ggml_is_contiguous());
    assert!(a.elem_count() == ne0 * ne1 * ne2);
    let ne: [usize; 3] = [ne0, ne1, ne2];
    let result = view_tensor(a, 0, 3, a.dtype(), Shape::from_array(ne))?;
    Ok(result)
}

fn get_rows<'a>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    a: &TensorView<'a>,
    b: &TensorView<'a>,
) -> LLMResult<TensorView<'a>> {
    let mut dst = new_tensor_2d(ctx, buf, GGmlType::F32, a.dim1(), b.dim1(), a.device())?;
    galois::op::galois_get_rows(a, b, &mut dst)?;
    Ok(dst)
}

fn set_f32<'a>(a: &mut TensorView<'a>, value: f32) -> LLMResult<()> {
    let n = a.dim().nrows();
    let nc = a.shape_layout()[0];
    let n1 = a.stride_layout()[1];

    let t = a.dtype();
    match t {
        GGmlType::F32 => {
            let data = unsafe { a.as_slice_mut::<f32>() };
            for i in 0..n {
                data[i * n1..i * n1 + nc].fill(value);
            }
        }
        _ => {
            todo!()
        }
    }

    Ok(())
}

fn add<'a, X: TensorProto, Y: TensorProto>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    a: &X,
    b: &Y,
) -> LLMResult<X> {
    let mut dst = dup_tensor(ctx, buf, a)?;
    galois::op::galois_add(a, b, &mut dst)?;
    Ok(dst)
}

fn rms_norm<T: TensorProto>(ctx: &mut TensorContext, buf: &[u8], a: &T, eps: f32) -> LLMResult<T> {
    let mut dst = dup_tensor::<T, T>(ctx, buf, a)?;
    galois::op::galois_rms_norm(a, &mut dst, eps)?;
    Ok(dst)
}

fn soft_max<'a, T: TensorProto>(ctx: &mut TensorContext, buf: &'a [u8], a: &T) -> LLMResult<T> {
    let mut dst = dup_tensor(ctx, buf, a)?;
    galois::op::galois_soft_max(a, &mut dst)?;
    Ok(dst)
}

fn repeat<'a>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    src: &'a TensorView<'a>,
    cur: &'a TensorView<'a>,
) -> LLMResult<TensorView<'a>> {
    let mut dst = new_tensor(
        ctx,
        buf,
        cur.n_dims(),
        src.dtype(),
        Shape::from_slice(cur.shape()),
        src.device(),
    )?;
    galois::op::galois_repeat(src, &mut dst)?;
    Ok(dst)
}

fn mul<X: TensorProto, Y: TensorProto>(
    ctx: &mut TensorContext,
    buf: &[u8],
    a: &X,
    b: &Y,
) -> LLMResult<X> {
    let mut dst = dup_tensor(ctx, buf, a)?;
    galois::op::galois_mul(a, b, &mut dst)?;
    Ok(dst)
}

fn matmul<X: TensorProto, Y: TensorProto>(
    ctx: &mut TensorContext,
    buf: &[u8],
    a: &X,
    b: &Y,
) -> LLMResult<Y> {
    let ne = [
        a.ggml_shape()[1],
        b.ggml_shape()[1],
        b.ggml_shape()[2],
        b.ggml_shape()[3],
    ];
    let mut dst = new_tensor(
        ctx,
        buf,
        std::cmp::max(a.n_dims(), b.n_dims()),
        GGmlType::F32,
        Shape::from_array(ne),
        a.device(),
    )?;
    galois::op::galois_matmul(a, b, &mut dst)?;
    Ok(dst)
}

fn scale_inplace<'a>(a: &'a TensorView<'a>, b: &'a TensorView<'a>) -> LLMResult<TensorView<'a>> {
    let mut dst = a.view();
    galois::op::galois_scale(a, b, &mut dst)?;
    Ok(dst)
}

fn scale<'a, X: TensorProto, Y: TensorProto>(
    ctx: &mut TensorContext,
    buf: &'a [u8],
    a: &X,
    b: &Y,
) -> LLMResult<X> {
    let mut dst = dup_tensor(ctx, buf, a)?;
    galois::op::galois_scale(a, b, &mut dst)?;
    Ok(dst)
}

fn rope_custom<X: TensorProto, Y: TensorProto>(
    ctx: &mut TensorContext,
    buf: &[u8],
    a: &X,
    b: &Y,
    n_dims: usize,
    mode: i32,
    n_ctx: i32,
    freq_base: f32,
    freq_scale: f32,
    xpos_base: f32,
    xpos_down: bool,
) -> LLMResult<Y> {
    assert!(b.is_vector());
    assert!(b.dtype() == GGmlType::I32);
    assert!(a.dim_2() == b.dim_0());
    let mut dst = dup_tensor(ctx, buf, a)?;

    let op = RopeCustomOption {
        n_dims: n_dims,
        mode: mode,
        n_ctx: n_ctx,
        freq_base: freq_base,
        freq_scale: freq_scale,
        xpos_base: xpos_base,
        xpos_down: xpos_down,
    };
    galois::op::galois_rope_custom(op, a, b, &mut dst)?;
    Ok(dst)
}

// struct GptVocab {
//     n_vocab: i32,
//     token_to_id: HashMap<Token, LlamaToken>,
//     id_to_token: HashMap<LlamaToken, Token>,
// }

// impl GptVocab {
//     // fn load<T: Read + BufRead>(r: &mut T, n_vocab: i32) -> LLMResult<GptVocab> {
//     //     let mut token_to_id: HashMap<Token, GptVocabId> = HashMap::new();
//     //     let mut id_to_token: HashMap<GptVocabId, Token> = HashMap::new();
//     //     for i in 0..n_vocab {
//     //         let len: u32 = r.read_u32::<Endian>()?;
//     //         let mut tmp = vec![0; len as usize];
//     //         r.read_exact(&mut tmp)?;
//     //         let word = String::from_utf8_lossy(&tmp).to_string();
//     //         // if i == 1111 {
//     //         //     println!("{}: vocab[{}] =       = {}\n", function!(), i, word);
//     //         // }
//     //         token_to_id.insert(word.clone(), i);
//     //         id_to_token.insert(i, word);
//     //     }

//     //     Ok(GptVocab {
//     //         n_vocab,
//     //         token_to_id,
//     //         id_to_token,
//     //     })
//     // }
// }

enum LLMArch {
    LLM_ARCH_LLAMA,
}

struct GGUFFile(Mmap);

impl GGUFFile {
    fn buf(&self) -> &[u8] {
        &self.0
    }
}

struct LlamaModel<'a> {
    hparams: LlamaHparams,

    tok_embeddings: TensorView<'a>,

    output_norm: TensorView<'a>,

    output: TensorView<'a>,

    gpu_layers: Vec<CudaLayer>,

    cpu_layers: Vec<CpuLayer<'a>>,
    // key + value memory
    // memory_k: Tensor,
    // memory_v: Tensor,
    //
    // struct ggml_context * ctx;
    // std::map<std::string, struct ggml_tensor *> tensors;
}

struct CudaLayer {
    // normalization
    attention_norm: Tensor,

    // attention
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,

    // normalization
    ffn_norm: Tensor,

    // ff
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
}

struct CpuLayer<'a> {
    // normalization
    attention_norm: TensorView<'a>,

    // attention
    wq: TensorView<'a>,
    wk: TensorView<'a>,
    wv: TensorView<'a>,
    wo: TensorView<'a>,

    // normalization
    ffn_norm: TensorView<'a>,

    // ff
    w1: TensorView<'a>,
    w2: TensorView<'a>,
    w3: TensorView<'a>,
}

// impl Default for LlamaHparams {
//     fn default() -> Self {
//         Self {
//             n_vocab: 32000,
//             n_ctx: 512,
//             n_embd: 4096,
//             n_mult: 256,
//             n_head: 32,
//             n_layer: 32,
//             n_rot: 64,
//             f16: 1,
//         }
//     }
// }

struct MmapReader {
    mmap: Mmap,
    offset: usize,
    file_size: usize,
}

impl MmapReader {
    fn new(mmap: Mmap, file_size: usize) -> MmapReader {
        Self {
            mmap: mmap,
            offset: 0,
            file_size: file_size,
        }
    }

    fn read_bytes(&mut self, n: usize) -> LLMResult<&[u8]> {
        if self.offset + n > self.file_size {
            return Err(LLMError::UnexpectedEof);
        }
        let v = &self.mmap[self.offset..self.offset + n];
        self.offset += n;
        Ok(v)
    }

    fn cursor(&self) -> &[u8] {
        &self.mmap[self.offset..]
    }

    fn offset(&self) -> usize {
        self.offset
    }
}

impl Read for MmapReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let buf_len = buf.len();
        if self.offset + buf_len > self.file_size {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        buf.copy_from_slice(&self.mmap[self.offset..self.offset + buf_len]);
        self.offset += buf_len;
        Ok(buf_len)
    }
}

struct GGufMmapReader {
    r: MmapReader,
    version: GGufVersion,
}

impl GGufMmapReader {
    fn new(r: MmapReader, version: GGufVersion) -> GGufMmapReader {
        GGufMmapReader { r, version }
    }

    fn version(&self) -> &GGufVersion {
        &self.version
    }

    fn mmap_reader(&mut self) -> &mut MmapReader {
        &mut self.r
    }
}

impl GGufRead for GGufMmapReader {
    fn read_bytes(&mut self, n: usize) -> LLMResult<&[u8]> {
        self.r.read_bytes(n)
    }

    fn offset(&self) -> usize {
        self.r.offset()
    }

    fn cursor(&self) -> &[u8] {
        self.r.cursor()
    }

    fn read_len(&mut self) -> LLMResult<usize> {
        let v = match self.version() {
            GGufVersion::V1 => self.r.read_u32::<Endian>()? as usize,
            GGufVersion::V2 => self.r.read_u64::<Endian>()? as usize,
            GGufVersion::V3 => self.r.read_u64::<Endian>()? as usize,
        };
        Ok(v)
    }
}

impl Read for GGufMmapReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.r.read(buf)
    }
}

impl<'a> LlamaModel<'a> {
    // fn eval(llama_batch: Vec<LLamaToken>) {}

    fn load<T: Read + GGufRead>(
        gguf_reader: &'a mut T,
        hparams: LlamaHparams,
        tensor_infos: HashMap<GGufStr, GGufDiskTensorInfo>,
        gpu_dev: &Device,
        n_gpu_layer: usize,
    ) -> LLMResult<LlamaModel<'a>> {
        let n_embd = hparams.n_embd as usize;
        let n_embd_gqa = hparams.n_embd_gqa() as usize;
        let n_layer = hparams.n_layer as usize;
        let n_vocab = hparams.n_vocab as usize;
        let n_ff = hparams.n_ff as usize;
        assert!(n_gpu_layer < n_layer);
        let position = gguf_reader.offset();
        // let alignment = header.alignment() as usize;
        let next_position = position - (position % GGUF_DEFAULT_ALIGNMENT) + GGUF_DEFAULT_ALIGNMENT;
        // 跳过 内存对齐
        let _ = gguf_reader.read_bytes(next_position - position)?;
        let tensor_data = gguf_reader.cursor();

        //let tensor_ctx: TensorContext<'a> = TensorContext::new(tensor_data);
        let cpu_dev = Device::Cpu;
        let tok_embeddings: TensorView<'a> = create_tensor(
            &tensor_data,
            &tensor_infos,
            "token_embd.weight",
            [n_embd, n_vocab],
            &cpu_dev,
        )?;
        let output_norm: TensorView<'a> = create_tensor(
            &tensor_data,
            &tensor_infos,
            "output_norm.weight",
            [n_embd],
            &cpu_dev,
        )?;

        let output: TensorView<'a> = create_tensor(
            &tensor_data,
            &tensor_infos,
            "output.weight",
            [n_embd, n_vocab],
            &cpu_dev,
        )?;
        // let output = create_tensor(&mut tensor_ctx, "output.weight", [n_embd, n_vocab])?;
        // let output_norm = create_tensor(&mut tensor_ctx, "output_norm.weight", [n_embd]);

        let mut gpu_layers = Vec::new();

        for i in 0..n_gpu_layer {
            let attention_norm: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_norm.weight", i),
                [n_embd],
                gpu_dev,
            )?;

            let wq: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_q.weight", i),
                [n_embd, n_embd],
                gpu_dev,
            )?;
            let wk: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_k.weight", i),
                [n_embd, n_embd_gqa],
                gpu_dev,
            )?;
            let wv: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_v.weight", i),
                [n_embd, n_embd_gqa],
                gpu_dev,
            )?;
            let wo: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_output.weight", i),
                [n_embd, n_embd],
                gpu_dev,
            )?;

            let ffn_norm: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_norm.weight", i),
                [n_embd],
                gpu_dev,
            )?;

            let w1: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_gate.weight", i),
                [n_embd, n_ff],
                gpu_dev,
            )?;
            let w2: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_down.weight", i),
                [n_ff, n_embd],
                gpu_dev,
            )?;
            let w3: Tensor = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_up.weight", i),
                [n_embd, n_ff],
                gpu_dev,
            )?;

            gpu_layers.push(CudaLayer {
                attention_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w1,
                w2,
                w3,
            });
        }

        let mut cpu_layers = Vec::new();
        for i in 0..n_layer - n_gpu_layer {
            let attention_norm: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_norm.weight", i),
                [n_embd],
                &cpu_dev,
            )?;

            let wq: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_q.weight", i),
                [n_embd, n_embd],
                &cpu_dev,
            )?;
            let wk: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_k.weight", i),
                [n_embd, n_embd_gqa],
                &cpu_dev,
            )?;
            let wv: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_v.weight", i),
                [n_embd, n_embd_gqa],
                &cpu_dev,
            )?;
            let wo: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.attn_output.weight", i),
                [n_embd, n_embd],
                &cpu_dev,
            )?;

            let ffn_norm: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_norm.weight", i),
                [n_embd],
                &cpu_dev,
            )?;

            let w1: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_gate.weight", i),
                [n_embd, n_ff],
                &cpu_dev,
            )?;
            let w2: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_down.weight", i),
                [n_ff, n_embd],
                &cpu_dev,
            )?;
            let w3: TensorView<'a> = create_tensor(
                &tensor_data,
                &tensor_infos,
                &format!("blk.{}.ffn_up.weight", i),
                [n_embd, n_ff],
                &cpu_dev,
            )?;

            cpu_layers.push(CpuLayer {
                attention_norm,
                wq,
                wk,
                wv,
                wo,
                ffn_norm,
                w1,
                w2,
                w3,
            });
        }
        // let x = unsafe { tok_embeddings.as_slice::<BlockQ4_0>() };
        // let mut sum: f64 = 0.0;
        // for i in 0..x.tok_embeddings() {
        //     sum += x[i].abs() as f64;
        // }
        Ok(LlamaModel {
            hparams: hparams,

            tok_embeddings: tok_embeddings,

            output_norm: output_norm,
            output: output,
            gpu_layers: gpu_layers,
            cpu_layers: cpu_layers,
            // key + value memory
            // memory_k: Tensor,
            // memory_v: Tensor,
        })
    }

    // fn load<T: Read + BufRead>(
    //     r: &mut T,
    //     hparams: LlamaHparams,
    //     buf_model: &mut [u8],
    // ) -> LLMResult<LlamaModel> {
    //     let wtype = match hparams.f16 {
    //         0 => GGmlType::F32,
    //         1 => GGmlType::F16,
    //         2 => GGmlType::Q4_0,
    //         _ => {
    //             todo!()
    //         }
    //     };
    //     let n_ff = (((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult)
    //         * hparams.n_mult) as f32;
    //     // let hparams = LlamaHparams::load(r)?;
    //     // GptVocab::load(r, hparams.n_vocab)?;
    //     let n_parts = 1;
    //     let mut tensors: HashMap<String, *mut Tensor> = HashMap::new();

    //     let model = {
    //         let mut tensor_ctx = TensorContext::new(buf_model);
    //         let n_embd = hparams.n_embd as usize;
    //         let n_layer = hparams.n_layer as usize;
    //         let n_ctx = hparams.n_ctx as usize;
    //         let n_vocab = hparams.n_vocab as usize;

    //         let mut layers: Vec<LlamaLayer> = Vec::with_capacity(n_layer);

    //         let mut tok_embeddings = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_vocab)?;
    //         let mut norm = new_tensor_1d(&mut tensor_ctx, GGmlType::F32, n_embd)?;
    //         let mut output = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_vocab)?;

    //         tensors.insert(
    //             "tok_embeddings.weight".to_string(),
    //             &mut tok_embeddings as *mut Tensor,
    //         );
    //         tensors.insert("norm.weight".to_string(), &mut norm as *mut Tensor);
    //         tensors.insert("output.weight".to_string(), &mut output as *mut Tensor);

    //         for i in 0..n_layer {
    //             let attention_norm = new_tensor_1d(&mut tensor_ctx, GGmlType::F32, n_embd)?;

    //             let wq = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;
    //             let wk = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;
    //             let wv = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;
    //             let wo = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;

    //             let ffn_norm = new_tensor_1d(&mut tensor_ctx, GGmlType::F32, n_embd)?;

    //             let w1 = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_ff as usize)?;
    //             let w2 = new_tensor_2d(&mut tensor_ctx, wtype, n_ff as usize, n_embd)?;
    //             let w3 = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_ff as usize)?;

    //             layers.push(LlamaLayer {
    //                 attention_norm,
    //                 wq,
    //                 wk,
    //                 wv,
    //                 wo,
    //                 ffn_norm,
    //                 w1,
    //                 w2,
    //                 w3,
    //             });

    //             let layer = layers.last_mut().unwrap();

    //             tensors.insert(
    //                 format!("layers.{}.attention_norm.weight", i),
    //                 &mut layer.attention_norm as *mut Tensor,
    //             );

    //             tensors.insert(
    //                 format!("layers.{}.attention.wq.weight", i),
    //                 &mut layer.wq as *mut Tensor,
    //             );
    //             tensors.insert(
    //                 format!("layers.{}.attention.wk.weight", i),
    //                 &mut layer.wk as *mut Tensor,
    //             );
    //             tensors.insert(
    //                 format!("layers.{}.attention.wv.weight", i),
    //                 &mut layer.wv as *mut Tensor,
    //             );
    //             tensors.insert(
    //                 format!("layers.{}.attention.wo.weight", i),
    //                 &mut layer.wo as *mut Tensor,
    //             );

    //             tensors.insert(
    //                 format!("layers.{}.ffn_norm.weight", i),
    //                 &mut layer.ffn_norm as *mut Tensor,
    //             );

    //             tensors.insert(
    //                 format!("layers.{}.feed_forward.w1.weight", i),
    //                 &mut layer.w1 as *mut Tensor,
    //             );
    //             tensors.insert(
    //                 format!("layers.{}.feed_forward.w2.weight", i),
    //                 &mut layer.w2 as *mut Tensor,
    //             );
    //             tensors.insert(
    //                 format!("layers.{}.feed_forward.w3.weight", i),
    //                 &mut layer.w3 as *mut Tensor,
    //             );
    //         }

    //         let n_embd = hparams.n_embd as usize;
    //         let n_layer = hparams.n_layer as usize;
    //         let n_ctx = hparams.n_ctx as usize;
    //         let n_vocab = hparams.n_vocab as usize;

    //         let n_mem = n_layer * n_ctx;
    //         let n_elements = n_embd * n_mem;

    //         let memory_k = new_tensor_1d(&mut tensor_ctx, GGmlType::F32, n_elements)?;
    //         let memory_v = new_tensor_1d(&mut tensor_ctx, GGmlType::F32, n_elements)?;
    //         let memory_size = memory_k.nbytes() + memory_v.nbytes();
    //         println!(
    //             "{}: memory_size = {:8.2} MB, n_mem = {},offset={}",
    //             function!(),
    //             memory_size as f32 / 1024.0 / 1024.0,
    //             n_mem,
    //             tensor_ctx.offset,
    //         );
    //         LlamaModel {
    //             hparams,
    //             tok_embeddings,
    //             norm,
    //             output,
    //             layers,
    //             // key + value memory
    //             memory_k,
    //             memory_v,
    //         }
    //     };
    //     {
    //         let mut total_size: usize = 0;
    //         let j: usize = 0;
    //         loop {
    //             let n_dims = r.read_i32::<Endian>()?;
    //             let length = r.read_i32::<Endian>()?;
    //             let ftype = r.read_i32::<Endian>()?;

    //             let mut nelements: usize = 1;
    //             let mut ne: [usize; 2] = [1, 1];
    //             // let n_dims = 3; // Assume this value is set appropriately
    //             print!(".");
    //             for i in 0..n_dims as usize {
    //                 ne[i] = r.read_i32::<Endian>()? as usize;
    //                 nelements *= ne[i];
    //             }
    //             //  println!("nelements:{}", nelements);
    //             let mut buffer = vec![0; length as usize];
    //             r.read_exact(&mut buffer)?;
    //             let name = String::from_utf8_lossy(&buffer).to_string();
    //             // println!("name:{}", name);
    //             let ref_tensor = tensors
    //                 .get_mut(name.as_str())
    //                 .ok_or(LLMError::UnknownTensor(name.clone()))?;

    //             if let Some(tensor) = unsafe { (*ref_tensor).as_mut() } {
    //                 let split_type = if name.contains("tok_embeddings") {
    //                     0
    //                 } else if name.contains("layers") {
    //                     if name.contains("attention.wo.weight") {
    //                         0
    //                     } else if name.contains("feed_forward.w2.weight") {
    //                         0
    //                     } else {
    //                         1
    //                     }
    //                 } else if name.contains("output") {
    //                     1
    //                 } else {
    //                     // Define a default split_type if needed
    //                     // For instance, if none of the conditions match
    //                     // you can decide what to return
    //                     0
    //                 };

    //                 if n_dims == 1 {
    //                     if tensor.elem_count() != nelements {
    //                         return Err(LLMError::WrongSizeTensor(
    //                             name,
    //                             tensor.elem_count(),
    //                             nelements,
    //                         ));
    //                     }
    //                 } else {
    //                     if tensor.elem_count() / n_parts != nelements {
    //                         return Err(LLMError::WrongSizeTensor(
    //                             name,
    //                             tensor.elem_count(),
    //                             nelements,
    //                         ));
    //                     }
    //                 }
    //                 let (ne0, ne1) = tensor.dim2();
    //                 if n_dims == 1 {
    //                     if ne0 != ne[0] || ne1 != ne[1] {
    //                         return Err(LLMError::WrongShapeTensor(
    //                             name,
    //                             vec![ne0, ne1],
    //                             ne.to_vec(),
    //                         ));
    //                     }
    //                 } else {
    //                     if split_type == 0 {
    //                         if ne0 / n_parts != ne[0] || ne1 != ne[1] {
    //                             return Err(LLMError::WrongShapeTensor(
    //                                 name,
    //                                 vec![ne0 / n_parts, ne1],
    //                                 ne.to_vec(),
    //                             ));
    //                         }
    //                     } else {
    //                         if ne0 != ne[0] || (ne1 / n_parts) != ne[1] {
    //                             return Err(LLMError::WrongShapeTensor(
    //                                 name,
    //                                 vec![ne0, ne1 / n_parts],
    //                                 ne.to_vec(),
    //                             ));
    //                         }
    //                     }
    //                 }

    //                 let bpe = match ftype {
    //                     0 => get_type_size(GGmlType::F32),
    //                     1 => get_type_size(GGmlType::F16),
    //                     2 => {
    //                         assert!(ne[0] % 64 == 0);
    //                         get_type_size(GGmlType::Q4_0)
    //                     }
    //                     _ => {
    //                         return Err(LLMError::UnknownFtypeGTensor(ftype));
    //                     }
    //                 };

    //                 if (nelements * bpe) / get_blck_size(tensor.dtype()) != tensor.nbytes() {
    //                     return Err(LLMError::WrongBytesTensor(
    //                         name,
    //                         (nelements * bpe) / get_blck_size(tensor.dtype()),
    //                         tensor.nbytes(),
    //                     ));
    //                 }
    //                 r.read_exact(tensor.as_bytes_mut())?;
    //                 // println!("name:{},nbytes:{}", name, tensor.as_bytes_mut().len());
    //                 // if name == "tok_embeddings.weight".to_string() {
    //                 //     let x: &[BlockQ4_0] = unsafe { tensor.as_slice::<BlockQ4_0>() };
    //                 //     let mut sum: f64 = 0.0;
    //                 //     for i in 0..tensor.elem_count() {
    //                 //         sum += x[i].d().abs() as f64;
    //                 //     }
    //                 //     println!(
    //                 //         "tok_embeddings,sum:{:?},sha
    //                 //         pe:{:?},stride:{:?}",
    //                 //         sum,
    //                 //         tensor.ggml_shape(),
    //                 //         tensor.dim().stride_4d()
    //                 //     );
    //                 //     //exit(1);
    //                 // }
    //                 total_size += tensor.nbytes();
    //                 // whisper_mode.n_loaded += 1;
    //                 match r.fill_buf() {
    //                     Ok(r) => {
    //                         if r.len() < 12 {
    //                             break;
    //                         }
    //                     }
    //                     Err(e) => match e.kind() {
    //                         std::io::ErrorKind::UnexpectedEof => break,
    //                         _ => return Err(LLMError::UnexpectIO(e)),
    //                     },
    //                 }
    //             } else {
    //                 println!("break");
    //                 return Err(LLMError::BadRefTensor(name));
    //             }
    //         }
    //     }
    //     println!("success");
    //     Ok(model)
    // }
}

struct LLMContext<'a> {
    t_load_us: i64,
    t_mel_us: i64,
    t_sample_us: i64,
    t_encode_us: i64,
    t_decode_us: i64,
    t_start_us: i64,
    magic: u32,
    model: LlamaModel<'a>,
}

// impl LLMContext {
//     fn new(fname: &str) -> LLMResult<LLMContext> {
//         let mut fin = open_file_stream(fname)?;
//         let magic = fin.read_u32::<Endian>()?;
//         if magic != GGML_MAGIC {
//             return Err(LLMError::BadMagic(fname.to_string()));
//         }

//         // let version = file.read_u32::<Endian>()?;

//         todo!()
//     }
// }

// 打开文件流
fn open_file_stream(fname: &str) -> LLMResult<BufReader<File>> {
    let file = File::open(fname)?;
    let buf_reader = BufReader::new(file);
    Ok(buf_reader)
}

const MAX_TOKEN_LEN: usize = 18;

// fn llama_tokenize(vocab: &GptVocab, text: &str, bos: bool) -> Vec<LlamaToken> {
//     let mut res: Vec<LlamaToken> = Vec::new();
//     let mut score: Vec<usize> = Vec::new();
//     let mut prev: Vec<LlamaToken> = Vec::new();
//     let len = text.len();

//     score.resize(len + 1, 0);
//     prev.resize(len + 1, 0);

//     for i in 0..len {
//         let max_len = std::cmp::min(len - i, MAX_TOKEN_LEN);
//         for sub_len in 1..=max_len {
//             let sub = &text[i..i + sub_len];
//             if let Some(&token_id) = vocab.token_to_id.get(sub) {
//                 let token_score = sub.len() * sub.len();
//                 let local_score = score[i] + token_score;
//                 let next = i + sub_len;
//                 if score[next] < local_score {
//                     score[next] = local_score;
//                     prev[next] = token_id;
//                 }
//             }
//         }
//     }

//     let mut i = len;
//     while i > 0 {
//         let token_id = prev[i];
//         if token_id == 0 {
//             // TODO: Return error or something more meaningful
//             eprintln!("failed to tokenize string!");
//             break;
//         }
//         res.push(token_id);

//         if let Some(token) = vocab.id_to_token.get(&token_id) {
//             i -= token.len();
//         } else {
//             // Handle the case where token_id is not found in the vocabulary
//             eprintln!("token_id not found in vocabulary!");
//             break;
//         }
//     }

//     if bos {
//         res.push(1); // TODO: replace with vocab.bos
//     }

//     // Pieces are in reverse order so correct that
//     res.reverse();

//     return res;
// }

fn llama_eval<'a>(
    model: &LlamaModel,
    batch: &LlamaBatch,
    kv_self: &LlamaKvCache,
    mem_per_token: usize,
    gpu_dev: &Device,
    n_gpu_layer: usize,
) -> LLMResult<()> {
    let embd_inp = batch.embd_inp();
    let n_tokens = embd_inp.len();
    let hparams = &model.hparams;

    let n_embd = hparams.n_embd as usize;
    let n_layer = hparams.n_layer as usize;
    let n_head = hparams.n_head as usize;
    let n_head_kv = hparams.n_head_kv as usize;
    let n_vocab = hparams.n_vocab as usize;
    let n_rot = (hparams.n_embd / hparams.n_head) as usize;
    let d_key = (n_embd / n_head) as usize;
    let n_embd_head = hparams.n_embd_head() as usize;
    let n_embd_gqa = hparams.n_embd_gqa() as usize;
    let norm_rms_eps = hparams.f_norm_rms_eps;
    assert!(n_embd_head == hparams.n_rot as usize);
    let freq_base = 10000.0;
    let freq_scale = 1.0;

    let n_embd = hparams.n_embd_gqa() as usize;
    let n_layer = hparams.n_layer as usize;
    let n_ctx = 512usize;
    let n_mem = n_layer * n_ctx;
    let n_elements = (n_embd * n_mem) as usize;
    let kv_head = 0; // head;
    let n_kv = 32;

    let buf_size = 512 * 1024 * 1024; //30MB
    let buf: Vec<u8> = vec![0u8; buf_size];
    let mut tensor_ctx = TensorContext::new();

    // let mut buf_kv: Vec<u8> = vec![0u8; 10 * 1024 * 1024];
    // let mut kv_buf_ctx = TensorContext::new(&mut buf_kv);

    let memory_k = new_tensor_1d(
        &mut tensor_ctx,
        &buf,
        GGmlType::F16,
        n_elements,
        &Device::Cpu,
    )?;
    let memory_v = new_tensor_1d(
        &mut tensor_ctx,
        &buf,
        GGmlType::F16,
        n_elements,
        &Device::Cpu,
    )?;

    let cuda_k = memory_k.to_cuda_tensor(gpu_dev)?;
    let cuda_v = memory_v.to_cuda_tensor(gpu_dev)?;

    let mut embd = new_tensor_1d(&mut tensor_ctx, &buf, GGmlType::I32, n_tokens, &Device::Cpu)?;

    unsafe {
        embd.as_slice_mut::<i32>().copy_from_slice(embd_inp);
    }

    let mut inp_l = get_rows(&mut tensor_ctx, &buf, &model.tok_embeddings, &embd)?;

    let mut KQ_scale = new_tensor_1d(&mut tensor_ctx, &buf, GGmlType::F32, 1, &Device::Cpu)?;

    set_f32(&mut KQ_scale, 1.0f32 / (n_embd_head as f32).sqrt())?;

    let mut KQ_mask = new_tensor_3d(
        &mut tensor_ctx,
        &buf,
        GGmlType::F32,
        n_kv,
        n_tokens,
        1,
        &Device::Cpu,
    )?;

    unsafe {
        let data = KQ_mask.as_slice_mut::<f32>();
        for h in 0..1 {
            for j in 0..n_tokens {
                let pos = batch.pos[j];
                let seq_id = batch.seq_id[j];
                for i in 0..n_kv {
                    if !kv_self.cells[i].has_seq_id(&seq_id) || kv_self.cells[i].pos > pos {
                        data[h * (n_kv * n_tokens) + j * n_kv + i] = -INFINITY;
                    }
                }
            }
        }
    }

    let cuda_KQ_mask = KQ_mask.to_cuda_tensor(gpu_dev)?;

    let mut KQ_pos = new_tensor_1d(&mut tensor_ctx, &buf, GGmlType::I32, n_tokens, &Device::Cpu)?;

    unsafe {
        let s = KQ_pos.as_slice_mut::<i32>();
        for i in 0..n_tokens {
            s[i] = batch.pos[i]
        }
    }

    let cuda_KQ_pos = KQ_pos.to_cuda_tensor(gpu_dev)?;

    let mut cuda_inp_l = inp_l.to_cuda_tensor(gpu_dev)?;

    for il in 0..n_gpu_layer as usize {
        let mut cur = rms_norm(&mut tensor_ctx, &buf, &cuda_inp_l, norm_rms_eps)?;

        cur = mul(
            &mut tensor_ctx,
            &buf,
            &cur,
            &model.gpu_layers[il].attention_norm,
        )?;

        let tmpk = matmul(&mut tensor_ctx, &buf, &model.gpu_layers[il].wk, &cur)?;

        let cpu_v = tmpk.to_cpu_tensor()?;

        let tmpk_reshape_3d = reshape_3d(&tmpk, n_embd_head, n_head_kv, n_tokens)?;

        let Kcur = rope_custom(
            &mut tensor_ctx,
            &buf,
            &tmpk_reshape_3d,
            &cuda_KQ_pos,
            n_embd_head,
            0,
            0,
            freq_base,
            freq_scale,
            0.0f32,
            false,
        )?;

        let tmpq = matmul(&mut tensor_ctx, &buf, &model.gpu_layers[il].wq, &cur)?;

        let tmpq_reshape_3d = reshape_3d(&tmpq, n_embd_head, n_head, n_tokens)?;

        let Qcur = rope_custom(
            &mut tensor_ctx,
            &buf,
            &tmpq_reshape_3d,
            &cuda_KQ_pos,
            n_embd_head,
            0,
            0,
            freq_base,
            freq_scale,
            0.0f32,
            false,
        )?;

        {
            let tmpv = matmul(&mut tensor_ctx, &buf, &model.gpu_layers[il].wv, &cur)?;

            let mut tmpv_reshape_2d = reshape_2d(&tmpv, n_embd_gqa, n_tokens)?;
            let Vcur = transpose(&mut tmpv_reshape_2d)?;

            let mut k = view_1d(
                &cuda_k,
                n_tokens * n_embd_gqa,
                (n_embd_gqa) * (il * n_ctx + kv_head), //memory_k.elem_size()
            )?;

            let mut v = view_2d(
                &cuda_v,
                n_tokens,
                n_embd_gqa,
                n_ctx,
                (il * n_ctx) * n_embd_gqa + kv_head, //memory_v.elem_size()
            )?;

            // important: storing RoPE-ed version of K in the KV cache!
            cpy(&Kcur, &mut k)?;
            cpy(&Vcur, &mut v)?;
        }

        let Q = permute(Qcur, 0, 2, 1, 3)?;

        let K = view_3d(
            &cuda_k,
            n_embd_head,
            n_kv,
            n_head_kv,
            n_embd_gqa,
            n_embd_head,
            n_embd_gqa * n_ctx * il, //memory_k.elem_size()
        )?;

        let KQ: Tensor = matmul(&mut tensor_ctx, &buf, &K, &Q)?;

        let KQ_scaled = scale(&mut tensor_ctx, &buf, &KQ, &KQ_scale)?;

        let KQ_masked = add(&mut tensor_ctx, &buf, &KQ_scaled, &cuda_KQ_mask)?;

        let KQ_soft_max = soft_max(&mut tensor_ctx, &buf, &KQ_masked)?;

        let V = view_3d(
            &cuda_v,
            n_kv,
            n_embd_head,
            n_head_kv,
            n_ctx,
            n_ctx * n_embd_head,
            n_ctx * n_embd_gqa * il, //memory_v.elem_size()
        )?;
        let KQV = matmul(&mut tensor_ctx, &buf, &V, &KQ_soft_max)?;

        let KQV_merged = permute(KQV, 0, 2, 1, 3)?;

        cur = cont_2d(&mut tensor_ctx, &buf, &KQV_merged, n_embd, n_tokens)?;

        cur = matmul(&mut tensor_ctx, &buf, &model.gpu_layers[il].wo, &cur)?;
        let inpFF = add(&mut tensor_ctx, &buf, &cur, &cuda_inp_l)?;

        // feed-forward network
        {
            {
                cur = rms_norm(&mut tensor_ctx, &buf, &inpFF, norm_rms_eps)?;

                cur = mul(&mut tensor_ctx, &buf, &cur, &model.gpu_layers[il].ffn_norm)?;
            }

            let tmp = matmul(&mut tensor_ctx, &buf, &model.gpu_layers[il].w3, &cur)?;
            cur = matmul(&mut tensor_ctx, &buf, &model.gpu_layers[il].w1, &cur)?;

            // SILU activation
            cur = silu(&mut tensor_ctx, &buf, &cur)?;

            cur = mul(&mut tensor_ctx, &buf, &cur, &tmp)?;

            cur = matmul(&mut tensor_ctx, &buf, &model.gpu_layers[il].w2, &cur)?;
        }

        cur = add(&mut tensor_ctx, &buf, &cur, &inpFF)?;

        // input for next layer
        cuda_inp_l = cur;
    }

    let mut cur = cuda_inp_l;

    {
        cur = rms_norm(&mut tensor_ctx, &buf, &cur, norm_rms_eps)?;

        cur = mul(&mut tensor_ctx, &buf, &cur, &model.output_norm)?;
    }

    // // lm_head
    // cur = matmul(&mut tensor_ctx, &buf, &model.output, &cur)?;

    //let mut Qcur: Option<TensorView<'_>> = None;

    for il in 0..1 as usize {
        //let inpSA = &inp_l;
        let mut cur = rms_norm(&mut tensor_ctx, &buf, &inp_l, norm_rms_eps)?;

        cur = mul(
            &mut tensor_ctx,
            &buf,
            &cur,
            &model.cpu_layers[il].attention_norm,
        )?;

        let tmpk = matmul(&mut tensor_ctx, &buf, &model.cpu_layers[il].wk, &cur)?;

        let tmpk_reshape_3d = reshape_3d(&tmpk, n_embd_head, n_head_kv, n_tokens)?;

        let Kcur = rope_custom(
            &mut tensor_ctx,
            &buf,
            &tmpk_reshape_3d,
            &KQ_pos,
            n_embd_head,
            0,
            0,
            freq_base,
            freq_scale,
            0.0f32,
            false,
        )?;

        let tmpq = matmul(&mut tensor_ctx, &buf, &model.cpu_layers[il].wq, &cur)?;

        let tmpq_reshape_3d = reshape_3d(&tmpq, n_embd_head, n_head, n_tokens)?;

        let Qcur = rope_custom(
            &mut tensor_ctx,
            &buf,
            &tmpq_reshape_3d,
            &KQ_pos,
            n_embd_head,
            0,
            0,
            freq_base,
            freq_scale,
            0.0f32,
            false,
        )?;
        println!("{} ,stride:{:?}", "Qcur", Qcur.dim().stride_4d());
        // store key and value to memory
        {
            let tmpv = matmul(&mut tensor_ctx, &buf, &model.cpu_layers[il].wv, &cur)?;

            let mut tmpv_reshape_2d = reshape_2d(&tmpv, n_embd_gqa, n_tokens)?;
            let Vcur = transpose(&mut tmpv_reshape_2d)?;

            let mut k = view_1d(
                &memory_k,
                n_tokens * n_embd_gqa,
                (n_embd_gqa) * (il * n_ctx + kv_head), //memory_k.elem_size()
            )?;

            let mut v = view_2d(
                &memory_v,
                n_tokens,
                n_embd_gqa,
                n_ctx,
                (il * n_ctx) * n_embd_gqa + kv_head, //memory_v.elem_size()
            )?;

            // important: storing RoPE-ed version of K in the KV cache!
            cpy(&Kcur, &mut k)?;
            cpy(&Vcur, &mut v)?;
        }
        {
            let Q = permute(Qcur, 0, 2, 1, 3)?;

            let K = view_3d(
                &memory_k,
                n_embd_head,
                n_kv,
                n_head_kv,
                n_embd_gqa,
                n_embd_head,
                n_embd_gqa * n_ctx * il, //memory_k.elem_size()
            )?;
            println!(
                "cpu K,shape:{:?} ,stride:{:?}",
                K.dim().shape_layout(),
                K.dim().stride_4d()
            );
            println!(
                "cpu Q,shape:{:?} ,stride:{:?}",
                Q.dim().shape_layout(),
                Q.dim().stride_4d()
            );

            let KQ = matmul(&mut tensor_ctx, &buf, &K, &Q)?;

            let KQ_scaled = scale(&mut tensor_ctx, &buf, &KQ, &KQ_scale)?;

            let KQ_masked: TensorView = add(&mut tensor_ctx, &buf, &KQ_scaled, &KQ_mask)?;

            let KQ_soft_max = soft_max(&mut tensor_ctx, &buf, &KQ_masked)?;

            let V = view_3d(
                &memory_v,
                n_kv,
                n_embd_head,
                n_head_kv,
                n_ctx,
                n_ctx * n_embd_head,
                n_ctx * n_embd_gqa * il, //memory_v.elem_size()
            )?;
            let KQV = matmul(&mut tensor_ctx, &buf, &V, &KQ_soft_max)?;

            let KQV_merged = permute(KQV, 0, 2, 1, 3)?;

            cur = cont_2d(&mut tensor_ctx, &buf, &KQV_merged, n_embd, n_tokens)?;

            cur = matmul(&mut tensor_ctx, &buf, &model.cpu_layers[il].wo, &cur)?;

            let inpFF = add(&mut tensor_ctx, &buf, &cur, &inp_l)?;

            // feed-forward network
            {
                {
                    cur = rms_norm(&mut tensor_ctx, &buf, &inpFF, norm_rms_eps)?;

                    cur = mul(&mut tensor_ctx, &buf, &cur, &model.cpu_layers[il].ffn_norm)?;
                }

                let tmp = matmul(&mut tensor_ctx, &buf, &model.cpu_layers[il].w3, &cur)?;
                cur = matmul(&mut tensor_ctx, &buf, &model.cpu_layers[il].w1, &cur)?;

                // SILU activation
                cur = silu(&mut tensor_ctx, &buf, &cur)?;

                cur = mul(&mut tensor_ctx, &buf, &cur, &tmp)?;

                cur = matmul(&mut tensor_ctx, &buf, &model.cpu_layers[il].w2, &cur)?;
            }

            cur = add(&mut tensor_ctx, &buf, &cur, &inpFF)?;
        }

        let x: &[f32] = unsafe { cur.as_slice::<f32>() };
        let mut sum: f32 = 0.0;
        for i in 0..cur.elem_count() {
            sum += x[i];
        }
        println!(
            "{} cpu cur,sum:{:?},shape:{:?},stride:{:?}",
            "KQ",
            sum,
            cur.shape_layout(),
            cur.dim().stride_4d()
        );
        return Ok(());

        inp_l = cur;
        // input for next layer
    }

    let mut cur = inp_l;

    {
        cur = rms_norm(&mut tensor_ctx, &buf, &cur, norm_rms_eps)?;

        cur = mul(&mut tensor_ctx, &buf, &cur, &model.output_norm)?;
    }

    // lm_head
    cur = matmul(&mut tensor_ctx, &buf, &model.output, &cur)?;

    let x: &[f32] = unsafe { cur.as_slice::<f32>() };
    let mut sum: f32 = 0.0;
    for i in 0..cur.elem_count() {
        sum += x[i];
    }
    println!(
        "cur,sum:{:?},shape:{:?},stride:{:?}",
        sum,
        cur.shape_layout(),
        cur.dim().stride_4d()
    );
    return Ok(());

    Ok(())
}

impl BinarySerialize for GGmlType {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let ggml_type = u32::deserialize(r)?;
        let v = match ggml_type {
            0 => GGmlType::F32,
            1 => GGmlType::F16,
            2 => GGmlType::Q4_0,
            3 => GGmlType::Q4_1,
            6 => GGmlType::Q5_0,
            7 => GGmlType::Q5_1,
            8 => GGmlType::Q8_0,
            9 => GGmlType::Q8_1,
            10 => GGmlType::Q2K,
            11 => GGmlType::Q3K,
            12 => GGmlType::Q4K,
            13 => GGmlType::Q5K,
            14 => GGmlType::Q6K,
            15 => GGmlType::Q8K,
            16 => GGmlType::I8,
            17 => GGmlType::I16,
            18 => GGmlType::I32,
            _ => return Err(LLMError::UnknownGGmlType(ggml_type)),
        };
        Ok(v)
    }
}

const GGUF_DEFAULT_ALIGNMENT: usize = 32;

fn ggml_pad(x: usize, n: usize) -> usize {
    (x + n - 1) & !(n - 1)
}

struct GGufDiskTensorInfo {
    //name: GGufStr,
    n_dims: u32,
    dimensions: [usize; MAX_DIM],
    typ: GGmlType,
    offset: u64,
    size: usize,
}

impl GGufDiskTensorInfo {
    fn new(
        n_dims: u32,
        dimensions: [usize; MAX_DIM],
        typ: GGmlType,
        offset: u64,
    ) -> GGufDiskTensorInfo {
        let mut size_needed: usize = get_type_size(typ) * (dimensions[0] / get_blck_size(typ));
        for i in 1..n_dims as usize {
            size_needed *= dimensions[i];
        }
        size_needed = ggml_pad(size_needed, GGUF_DEFAULT_ALIGNMENT);
        Self {
            n_dims,
            dimensions,
            typ,
            offset,
            size: size_needed,
        }
    }
}

fn gguf_read() -> LLMResult<()> {
    let model_path = "/opt/cproject/llama.cpp-gguf-fix-publish/models/llama-2-7b.Q4_0.gguf";
    let file = OpenOptions::new().read(true).open(model_path)?;
    let file_size = file.metadata()?.len();
    let mmap: Mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| format!("mmap failed: {}", e))?
    };
    let mut mmap_reader = MmapReader::new(mmap, file_size as usize);
    let header = GGufHeader::load(&mut mmap_reader)?;
    let mut gguf_reader = GGufMmapReader::new(mmap_reader, header.version().clone());

    let n_kv = header.n_kv();
    let mut gg_kvs: HashMap<GGufStr, GGufMetadataValue> = HashMap::new();
    for i in 0..n_kv {
        let key = GGufStr::deserialize(&mut gguf_reader)?;
        let value = GGufMetadataValue::deserialize(&mut gguf_reader)?;
        gg_kvs.insert(key, value);
    }
    let mut gg_tensor_infos: HashMap<GGufStr, GGufDiskTensorInfo> = HashMap::new();
    let n_tensors = header.n_tensors();
    for i in 0..n_tensors {
        let name = GGufStr::deserialize(&mut gguf_reader)?;
        let n_dims = u32::deserialize(&mut gguf_reader)?;
        let mut dimensions = [1usize; 4];
        assert!(n_dims <= 4);
        for j in 0..n_dims as usize {
            dimensions[j] = usize::deserialize(&mut gguf_reader)?;
        }
        let typ = GGmlType::deserialize(&mut gguf_reader)?;
        let offset = u64::deserialize(&mut gguf_reader)?;
        gg_tensor_infos.insert(
            name,
            GGufDiskTensorInfo::new(n_dims, dimensions, typ, offset),
        );
    }
    let gguf_ctx = GGufContext {
        header: header,
        metas: gg_kvs,
    };
    let h = LlamaHparams::load(&gguf_ctx)?;
    let vocab = LLamaVocab::load(&gguf_ctx)?;

    let output = tokenize(&vocab, "what is computer english".to_string(), true)?;
    println!("{:?}", output);
    // let mut output = Vec::new();
    //let mut t = LLMtokenizerSpm::new(&vocab);
    // let mut s = "who am i".to_string();
    // s.insert(0, ' ');
    // replace_all(&mut s, " ", "▁");
    // t.tokenize(&s, &mut output)?;
    // println!("out:{:?}", output);
    return Ok(());

    let gpu = CudaDevice::new(0)?;
    init_cuda_function(&gpu)?;
    let gpu_dev = Device::Gpu(gpu);
    let n_gpu_layer = 0;
    let model = LlamaModel::load(&mut gguf_reader, h, gg_tensor_infos, &gpu_dev, n_gpu_layer)?;

    //解析token

    let tokens = vec![1i32, 2];

    let n_batch = 0;
    let n_ctx = 512usize;
    let all_pos_0 = n_batch;
    let all_pos_1 = 1;
    let all_seq_id = 0;

    let mut pos = vec![0i32; tokens.len()];
    for i in 0..tokens.len() {
        pos[i] = all_pos_0 + i as i32 * all_pos_1;
    }

    let mut seq_id = vec![0i32; tokens.len()];
    for i in 0..tokens.len() {
        seq_id[i] = all_seq_id;
    }

    let batch = LlamaBatch {
        token: tokens,
        pos: pos,
        seq_id: seq_id,
    };

    let mut kv_cache = LlamaKvCache::new(n_ctx);
    if !kv_cache.llama_kv_cache_find_slot(&batch) {
        return Ok(());
    }

    llama_eval(&model, &batch, &kv_cache, 0, &gpu_dev, n_gpu_layer).unwrap();
    Ok(())
}

fn main() -> LLMResult<()> {
    gguf_read()?;
    Ok(())
}
