use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use galois::error::GError;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Error as IOError;
use std::io::Read;
use thiserror::Error;
type Endian = LittleEndian;
use galois::shape::MAX_DIM;
use galois::GGmlType;
use galois::Shape;
use galois::Tensor;
use galois::GS_BLCK_SIZE;
use galois::GS_TYPE_SIZE;
use lazy_static::lazy_static;
use std::borrow::Borrow;
use std::fs::File;
use std::fs::OpenOptions;
use std::hash::Hash;
use std::ptr::NonNull;

const LLM_KV_GENERAL_ARCHITECTURE: &'static str = "general.architecture";
const LLM_KV_TOKENIZER_LIST: &'static str = "tokenizer.ggml.tokens";
const LLM_KV_CONTEXT_LENGTH: &'static str = ".context_length";
const LLM_KV_EMBEDDING_LENGTH: &'static str = ".embedding_length";
const LLM_KV_FEED_FORWARD_LENGTH: &'static str = ".feed_forward_length";
const LLM_KV_ATTENTION_HEAD_COUNT: &'static str = ".attention.head_count";
const LLM_KV_BLOCK_COUNT: &'static str = ".block_count";
const LLM_KV_ATTENTION_HEAD_COUNT_KV: &'static str = ".attention.head_count_kv";
const LLM_KV_ROPE_FREQ_BASE: &'static str = ".rope.freq_base";
const LLM_KV_ROPE_SCALE_LINEAR: &'static str = ".rope.scale_linear";
const LLM_KV_ROPE_DIMENSION_COUNT: &'static str = ".rope.dimension_count";

const LLM_KV_TOKENIZER_MODEL: &'static str = "tokenizer.ggml.model";
const LLM_KV_TOKENIZER_TOKEN_TYPE: &'static str = "tokenizer.ggml.token_type";
const LLM_KV_TOKENIZER_SCORES: &'static str = "tokenizer.ggml.scores";
const LLM_KV_TOKENIZER_MERGES: &'static str = "tokenizer.ggml.merges";
const LLM_KV_TOKENIZER_BOS_ID: &'static str = "tokenizer.ggml.bos_token_id";
const LLM_KV_TOKENIZER_EOS_ID: &'static str = "tokenizer.ggml.eos_token_id";
const LLM_KV_TOKENIZER_UNK_ID: &'static str = "tokenizer.ggml.unknown_token_id";
const LLM_KV_TOKENIZER_SEP_ID: &'static str = "tokenizer.ggml.seperator_token_id";
const LLM_KV_TOKENIZER_PAD_ID: &'static str = "tokenizer.ggml.padding_token_id";
const LLM_KV_TOKENIZER_HF_JSON: &'static str = "tokenizer.huggingface.json";
const LLM_KV_TOKENIZER_RWKV: &'static str = "tokenizer.rwkv.world";

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

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGML_MAGIC: u32 = 0x67676d6c; // "GGML"

#[derive(Error, Debug)]
pub enum LLMError {
    #[error("UnexpectedEof")]
    UnexpectedEof,
    #[error("Unexpected: {0}")]
    Unexpected(String),
    #[error("Unexpected IO: {0}")]
    UnexpectIO(IOError),
    #[error("invalid model file '{0}' (bad magic)\n")]
    BadMagic(u32),
    #[error("unknown version '{0}' \n")]
    UnknownVersion(u32),
    #[error("unknown model architecture '{0}' \n")]
    UnknownModelArchitecture(String),
    #[error("unknown meta type '{0}' \n")]
    UnknownMetaType(u32),
    #[error("unknown array meta type '{0:?}' \n")]
    UnknownArrayMetaType(GGufMetadataValueType),
    #[error("unknown ggml type '{0:?}' \n")]
    UnknownGGmlType(u32),
    #[error("not enough space in the context's memory pool\n")]
    NotEnoughSpace,
    #[error("unknown tensor '{0}' in model file\n")]
    UnknownTensor(String),
    #[error("invalid ref tensor '{0}'\n")]
    BadRefTensor(String),
    #[error("tensor {0} has wrong size in model file, got:{1}, expected:{2}\n")]
    WrongSizeTensor(String, usize, usize),
    #[error("tensor {0} has wrong shape in model file, got:{1:?}, expected:{2:?}\n")]
    WrongShapeTensor(String, Vec<usize>, Vec<usize>),
    #[error("tensor {0} has wrong bytes in model file, got:{1:?}, expected:{2:?}\n")]
    WrongBytesTensor(String, usize, usize),
    #[error("galois tensor:'{0}'")]
    WrongGTensor(GError),
    #[error(" unknown ftype '{0}' in model file")]
    UnknownFtypeGTensor(i32),
}

impl From<IOError> for LLMError {
    fn from(e: IOError) -> Self {
        LLMError::UnexpectIO(e)
    }
}

impl From<&str> for LLMError {
    fn from(e: &str) -> Self {
        LLMError::Unexpected(e.to_string())
    }
}

impl From<String> for LLMError {
    fn from(e: String) -> Self {
        LLMError::Unexpected(e)
    }
}

impl From<GError> for LLMError {
    fn from(e: GError) -> Self {
        LLMError::WrongGTensor(e)
    }
}

pub type LLMResult<T> = Result<T, LLMError>;

#[derive(Hash, Eq, PartialEq, Debug, Default)]
enum EModel {
    #[default]
    MODEL_UNKNOWN,
    MODEL_1B,
    MODEL_3B,
    MODEL_7B,
    MODEL_8B,
    MODEL_13B,
    MODEL_15B,
    MODEL_30B,
    MODEL_34B,
    MODEL_40B,
    MODEL_65B,
    MODEL_70B,
}

trait BinarySerialize: Sized {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self>;
}

#[derive(Debug, Clone)]
enum GGufVersion {
    V1,
    V2,
    V3,
}

impl GGufVersion {
    fn decode(version: u32) -> LLMResult<GGufVersion> {
        match version {
            1 => Ok(GGufVersion::V1),
            2 => Ok(GGufVersion::V2),
            3 => Ok(GGufVersion::V3),
            _ => Err(LLMError::UnknownVersion(version)),
        }
    }
}

#[derive(Clone)]
struct GGufStr {
    ptr: NonNull<u8>,
    len: usize,
}

impl PartialEq for GGufStr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str().eq(other.as_str())
    }
}

impl Eq for GGufStr {}

impl Hash for GGufStr {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl BinarySerialize for GGufStr {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let len = usize::deserialize(r)?;
        let buf = r.read_bytes(len)?;
        Ok(GGufStr {
            ptr: unsafe { NonNull::new_unchecked(buf.as_ptr() as *mut u8) },
            len: len,
        })
    }
}

impl GGufStr {
    fn as_str(&self) -> &str {
        let slice = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) };
        std::str::from_utf8(slice).unwrap()
    }
}

impl Drop for GGufStr {
    fn drop(&mut self) {
        // do nothing
    }
}

impl Debug for GGufStr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self.as_str()))
    }
}

impl Borrow<str> for GGufStr {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

#[derive(Clone)]
struct GGufSliceData<P> {
    ptr: NonNull<P>,
    len: usize,
}

impl<P> GGufSliceData<P> {
    fn as_slice(&self) -> &[P] {
        let slice = unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) };
        slice
    }

    fn get_len(&self) -> usize {
        self.len
    }
}

impl<P> Drop for GGufSliceData<P> {
    fn drop(&mut self) {
        // do nothing
    }
}

impl<P: Debug> Debug for GGufSliceData<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("slice len:{:?}", self.len))
    }
}

impl<P: BinarySerialize> BinarySerialize for GGufSliceData<P> {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let len = usize::deserialize(r)?;
        let mem_size = len * std::mem::size_of::<P>();
        let buf = r.read_bytes(mem_size)?;
        Ok(GGufSliceData {
            ptr: unsafe { NonNull::new_unchecked(buf.as_ptr() as *mut P) },
            len: len,
        })
    }
}

impl<P: BinarySerialize> BinarySerialize for Vec<P> {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let len = usize::deserialize(r)?;
        let mut v: Vec<P> = Vec::with_capacity(len);
        for _ in 0..len {
            v.push(P::deserialize(r)?);
        }
        Ok(v)
    }
}

#[derive(Debug, Clone)]
enum GGufArr {
    U8Array(GGufSliceData<u8>),
    I8Array(GGufSliceData<i8>),
    U16Array(GGufSliceData<u16>),
    I16Array(GGufSliceData<i16>),
    U32Array(GGufSliceData<u32>),
    I32Array(GGufSliceData<i32>),
    U64Array(GGufSliceData<u64>),
    I64Array(GGufSliceData<i64>),
    F32Array(GGufSliceData<f32>),
    F64Array(GGufSliceData<f64>),
    BoolArray(GGufSliceData<u8>),
    StringArray(Vec<GGufStr>),
    NestedArray(GGufSliceData<GGufArr>),
}

impl GGufArr {
    fn get_len(&self) -> usize {
        match self {
            GGufArr::U8Array(u) => u.get_len(),
            GGufArr::I8Array(u) => u.get_len(),
            GGufArr::U16Array(u) => u.get_len(),
            GGufArr::I16Array(u) => u.get_len(),
            GGufArr::U32Array(u) => u.get_len(),
            GGufArr::I32Array(u) => u.get_len(),
            GGufArr::U64Array(u) => u.get_len(),
            GGufArr::I64Array(u) => u.get_len(),
            GGufArr::F32Array(u) => u.get_len(),
            GGufArr::F64Array(u) => u.get_len(),
            GGufArr::BoolArray(u) => u.get_len(),
            GGufArr::StringArray(u) => u.len(),
            _ => 0,
        }
    }
}

impl BinarySerialize for GGufArr {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let typ: GGufMetadataValueType = GGufMetadataValueType::deserialize(r)?;
        let v = match typ {
            GGufMetadataValueType::U8 => GGufArr::U8Array(GGufSliceData::<u8>::deserialize(r)?),
            GGufMetadataValueType::I8 => GGufArr::I8Array(GGufSliceData::<i8>::deserialize(r)?),
            GGufMetadataValueType::U16 => GGufArr::U16Array(GGufSliceData::<u16>::deserialize(r)?),
            GGufMetadataValueType::I16 => GGufArr::I16Array(GGufSliceData::<i16>::deserialize(r)?),
            GGufMetadataValueType::U32 => GGufArr::U32Array(GGufSliceData::<u32>::deserialize(r)?),
            GGufMetadataValueType::I32 => GGufArr::I32Array(GGufSliceData::<i32>::deserialize(r)?),
            GGufMetadataValueType::F32 => GGufArr::F32Array(GGufSliceData::<f32>::deserialize(r)?),
            GGufMetadataValueType::F64 => GGufArr::F64Array(GGufSliceData::<f64>::deserialize(r)?),
            GGufMetadataValueType::U64 => GGufArr::U64Array(GGufSliceData::<u64>::deserialize(r)?),
            GGufMetadataValueType::I64 => GGufArr::I64Array(GGufSliceData::<i64>::deserialize(r)?),
            GGufMetadataValueType::Bool => GGufArr::BoolArray(GGufSliceData::<u8>::deserialize(r)?),
            GGufMetadataValueType::String => GGufArr::StringArray(Vec::<GGufStr>::deserialize(r)?),
            _ => return Err(LLMError::UnknownArrayMetaType(typ)),
        };
        Ok(v)
    }
}

enum GGufTypeValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(u8),
    String(GGufStr), //String(&'a str),
}

#[derive(Debug, Clone)]
enum GGufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(u8),
    String(GGufStr),
    Array(GGufArr),
}

impl GGufMetadataValue {
    fn get_u32(&self) -> Option<u32> {
        match self {
            GGufMetadataValue::U32(e) => Some(*e),
            _ => None,
        }
    }

    fn get_str(&self) -> Option<&str> {
        match self {
            GGufMetadataValue::String(e) => Some(e.as_str()),
            _ => None,
        }
    }

    fn get_f32(&self) -> Option<f32> {
        match self {
            GGufMetadataValue::F32(e) => Some(*e),
            _ => None,
        }
    }

    fn get_arr_n(&self) -> Option<usize> {
        match self {
            GGufMetadataValue::Array(e) => Some(e.get_len()),
            _ => None,
        }
    }

    fn get_str_arr(&self) -> Option<&[GGufStr]> {
        match self {
            GGufMetadataValue::Array(e) => match e {
                GGufArr::StringArray(arr) => Some(arr),
                _ => None,
            },
            _ => None,
        }
    }

    fn get_f32_arr(&self) -> Option<&[f32]> {
        match self {
            GGufMetadataValue::Array(e) => match e {
                GGufArr::F32Array(arr) => Some(arr.as_slice()),
                _ => None,
            },
            _ => None,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GGufMetadataValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl BinarySerialize for GGufMetadataValueType {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let t = u32::deserialize(r)?;
        let v = match t {
            0 => GGufMetadataValueType::U8,
            1 => GGufMetadataValueType::I8,
            2 => GGufMetadataValueType::U16,
            3 => GGufMetadataValueType::I16,
            4 => GGufMetadataValueType::U32,
            5 => GGufMetadataValueType::I32,
            6 => GGufMetadataValueType::F32,
            7 => GGufMetadataValueType::Bool,
            8 => GGufMetadataValueType::String,
            9 => GGufMetadataValueType::Array,
            10 => GGufMetadataValueType::U64,
            11 => GGufMetadataValueType::I64,
            12 => GGufMetadataValueType::F64,
            _ => return Err(LLMError::UnknownMetaType(t)),
        };
        Ok(v)
    }
}

impl BinarySerialize for u8 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_u8()?;
        Ok(u)
    }
}

impl BinarySerialize for i8 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_i8()?;
        Ok(u)
    }
}

impl BinarySerialize for u16 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_u16::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for i16 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_i16::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for i32 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_i32::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for u32 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_u32::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for f32 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_f32::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for f64 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_f64::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for u64 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_u64::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for i64 {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_i64::<Endian>()?;
        Ok(u)
    }
}

impl BinarySerialize for usize {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let u = r.read_len()?;
        Ok(u)
    }
}

impl BinarySerialize for GGufMetadataValue {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self> {
        let t = GGufMetadataValueType::deserialize(r)?;
        let v = match t {
            GGufMetadataValueType::U8 => GGufMetadataValue::U8(u8::deserialize(r)?),
            GGufMetadataValueType::I8 => GGufMetadataValue::I8(i8::deserialize(r)?),
            GGufMetadataValueType::U16 => GGufMetadataValue::U16(u16::deserialize(r)?),
            GGufMetadataValueType::I16 => GGufMetadataValue::I16(i16::deserialize(r)?),
            GGufMetadataValueType::U32 => GGufMetadataValue::U32(u32::deserialize(r)?),
            GGufMetadataValueType::I32 => GGufMetadataValue::I32(i32::deserialize(r)?),
            GGufMetadataValueType::F32 => GGufMetadataValue::F32(f32::deserialize(r)?),
            GGufMetadataValueType::F64 => GGufMetadataValue::F64(f64::deserialize(r)?),
            GGufMetadataValueType::U64 => GGufMetadataValue::U64(u64::deserialize(r)?),
            GGufMetadataValueType::I64 => GGufMetadataValue::I64(i64::deserialize(r)?),
            GGufMetadataValueType::String => GGufMetadataValue::String(GGufStr::deserialize(r)?),
            GGufMetadataValueType::Bool => GGufMetadataValue::Bool(u8::deserialize(r)?),
            GGufMetadataValueType::Array => GGufMetadataValue::Array(GGufArr::deserialize(r)?),
            // _=>return Err(LLMError::UnknownVersion(()))
            //
        };
        Ok(v)
    }
}

struct TensorContext<'a> {
    offset: usize,
    size: usize,
    n_objects: usize,
    buf: &'a [u8],
}

impl<'a> TensorContext<'a> {
    fn new(buf: &'a [u8]) -> TensorContext<'a> {
        TensorContext {
            offset: 0,
            size: 0,
            n_objects: 0,
            buf: buf,
        }
    }
}

fn create_tensor() {}

fn new_tensor_1d(ctx: &mut TensorContext, dtype: GGmlType, ne0: usize) -> LLMResult<Tensor> {
    let dim = [ne0];
    new_tensor(ctx, 1, dtype, Shape::from_array(dim))
}

fn new_tensor_2d(
    ctx: &mut TensorContext,
    dtype: GGmlType,
    ne0: usize,
    ne1: usize,
) -> LLMResult<Tensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, 2, dtype, Shape::from_array(dim))
}

fn new_tensor_3d(
    ctx: &mut TensorContext,
    dtype: GGmlType,
    ne0: usize,
    ne1: usize,
    ne2: usize,
) -> LLMResult<Tensor> {
    let dim = [ne0, ne1, ne2];
    new_tensor(ctx, 3, dtype, Shape::from_array(dim))
}

fn new_tensor_4d(
    ctx: &mut TensorContext,
    dtype: GGmlType,
    ne0: usize,
    ne1: usize,
    ne3: usize,
    ne4: usize,
) -> LLMResult<Tensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, 2, dtype, Shape::from_array(dim))
}

fn new_f32_tensor(ctx: &mut TensorContext, value: f32) -> LLMResult<Tensor> {
    let mut result = new_tensor_1d(ctx, GGmlType::F32, 1)?;
    result.set_value(value);
    Ok(result)
}

fn new_tensor(
    ctx: &mut TensorContext,
    n_dims: usize,
    dtype: GGmlType,
    shape: Shape,
) -> LLMResult<Tensor> {
    let cur_offset = ctx.offset;
    let cur_size = ctx.size;
    let ne = shape.layout();
    let mut size_needed: usize = get_type_size(dtype) * (ne[0] / get_blck_size(dtype));
    for i in 1..n_dims {
        size_needed *= ne[i];
    }
    size_needed = ggml_pad(size_needed, GGUF_DEFAULT_ALIGNMENT);
    // println!(
    //     "size_needed:{},get_type_sizef(dtype):{}",
    //     size_needed,
    //     get_type_sizef(dtype) as usize
    // );
    if cur_offset + size_needed > ctx.buf.len() {
        return Err(LLMError::NotEnoughSpace);
    }
    let t = unsafe {
        Tensor::from_bytes(
            &ctx.buf[cur_offset..cur_offset + size_needed],
            n_dims,
            shape,
            dtype,
        )
    };
    ctx.offset = cur_offset + size_needed;
    ctx.size = size_needed;
    ctx.n_objects += 1;
    Ok(t)
}

fn view_tensor(buf: &[u8], n_dims: usize, dtype: GGmlType, shape: Shape) -> LLMResult<Tensor> {
    Ok(unsafe { Tensor::from_bytes(buf, n_dims, shape, dtype) })
}

fn dup_tensor(ctx: &mut TensorContext, a: &Tensor) -> LLMResult<Tensor> {
    let dtype = a.dtype();
    let shape = Shape::from_slice(a.dim().shape());
    new_tensor(ctx, a.n_dims(), dtype, shape)
}

fn view_1d(a: &Tensor, ne0: usize, offset: usize) -> LLMResult<Tensor> {
    let dtype = a.dtype();
    let buf = a.as_bytes();
    let shape = Shape::from_array([ne0]);
    view_tensor(&buf[offset..], 1, dtype, shape)
}

fn view_2d(a: &Tensor, ne0: usize, ne1: usize, nb1: usize, offset: usize) -> LLMResult<Tensor> {
    let dtype = a.dtype();
    let buf = a.as_bytes();
    let shape = Shape::from_array([ne0, ne1]);
    let mut t = view_tensor(&buf[offset..], 2, dtype, shape)?;
    let nb0 = t.dim().stride_1d();
    let nb = [nb0, nb1, nb1 * ne1, nb1 * ne1];
    t.ret_stride(nb);
    Ok(t)
}

fn reshape_3d(a: &Tensor, ne0: usize, ne1: usize, ne2: usize) -> LLMResult<Tensor> {
    assert!(a.ggml_is_contiguous());
    assert!(a.elem_count() == ne0 * ne1 * ne2);
    let ne: [usize; 3] = [ne0, ne1, ne2];
    let result = view_tensor(a.as_bytes(), a.n_dims(), a.dtype(), Shape::from_array(ne))?;
    Ok(result)
}

fn get_rows(ctx: &mut TensorContext, a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let mut dst = new_tensor_2d(ctx, GGmlType::F32, a.dim1(), b.dim1())?;
    galois::op::galois_get_rows(a, b, &mut dst)?;
    Ok(dst)
}

fn rms_norm(ctx: &mut TensorContext, a: &Tensor) -> LLMResult<Tensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_rms_norm(a, &mut dst)?;
    Ok(dst)
}

type GptVocabId = usize;
type Token = String;

struct GptVocab {
    n_vocab: i32,
    token_to_id: HashMap<Token, GptVocabId>,
    id_to_token: HashMap<GptVocabId, Token>,
}

impl GptVocab {
    // fn load<T: Read + BufRead>(r: &mut T, n_vocab: i32) -> LLMResult<GptVocab> {
    //     let mut token_to_id: HashMap<Token, GptVocabId> = HashMap::new();
    //     let mut id_to_token: HashMap<GptVocabId, Token> = HashMap::new();
    //     for i in 0..n_vocab {
    //         let len: u32 = r.read_u32::<Endian>()?;
    //         let mut tmp = vec![0; len as usize];
    //         r.read_exact(&mut tmp)?;
    //         let word = String::from_utf8_lossy(&tmp).to_string();
    //         // if i == 1111 {
    //         //     println!("{}: vocab[{}] =       = {}\n", function!(), i, word);
    //         // }
    //         token_to_id.insert(word.clone(), i);
    //         id_to_token.insert(i, word);
    //     }

    //     Ok(GptVocab {
    //         n_vocab,
    //         token_to_id,
    //         id_to_token,
    //     })
    // }
}

struct LLamaVocab {
    tokens: Vec<String>,
    token_to_id: HashMap<Token, GptVocabId>,
    token_scores: HashMap<GptVocabId, f32>,
}

impl LLamaVocab {
    fn load(ctx: &GGufContext) -> LLMResult<LLamaVocab> {
        let vocab = ctx
            .metas_data()
            .get(LLM_KV_TOKENIZER_LIST)
            .unwrap()
            .get_str_arr()
            .unwrap()
            .iter()
            .map(|s| s.as_str().to_string())
            .collect::<Vec<_>>();
        let eos_token = ctx
            .metas_data()
            .get(LLM_KV_TOKENIZER_EOS_ID)
            .unwrap()
            .get_u32()
            .unwrap() as usize;
        let bos_token = ctx
            .metas_data()
            .get(LLM_KV_TOKENIZER_BOS_ID)
            .unwrap()
            .get_u32()
            .unwrap() as usize;
        let tokenizer_kind = ctx
            .metas_data()
            .get(LLM_KV_TOKENIZER_MODEL)
            .unwrap()
            .get_str()
            .unwrap();
        match tokenizer_kind {
            "llama" => {
                let vocab_scores = ctx
                    .metas_data()
                    .get(LLM_KV_TOKENIZER_SCORES)
                    .unwrap()
                    .get_f32_arr()
                    .unwrap()
                    .iter()
                    .map(|s| *s)
                    .collect::<Vec<_>>();
                let token_ids = vocab
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v.clone(), i))
                    .collect::<HashMap<_, _>>();
                let token_scores = vocab_scores
                    .into_iter()
                    .enumerate()
                    .collect::<HashMap<_, _>>();
                Ok(Self {
                    tokens: vocab,
                    token_to_id: token_ids,
                    token_scores: token_scores,
                })
            }
            _ => {
                return Err(LLMError::UnknownModelArchitecture(
                    tokenizer_kind.to_string(),
                ))
            }
        }
    }
}

enum LLMArch {
    LLM_ARCH_LLAMA,
}

struct GGUFFile(Mmap);

impl GGUFFile {
    fn buf(&self) -> &[u8] {
        &self.0
    }
}

struct LlamaModel {
    //mtype: EModel,
    hparams: LlamaHparams,

    tok_embeddings: Tensor,

    norm: Tensor,
    output: Tensor,

    layers: Vec<LlamaLayer>,

    // key + value memory
    memory_k: Tensor,
    memory_v: Tensor,
    //
    // struct ggml_context * ctx;
    // std::map<std::string, struct ggml_tensor *> tensors;
}

struct LlamaLayer {
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

pub enum ModelArchitecture {
    Llama,
    Gemma,
    Qwen2,
}

impl ModelArchitecture {
    fn decode(arch: &str) -> LLMResult<ModelArchitecture> {
        let v = match arch {
            "llama" => ModelArchitecture::Llama,
            "gemma" => ModelArchitecture::Gemma,
            "qwen2" => ModelArchitecture::Qwen2,
            _ => return Err(LLMError::UnknownModelArchitecture(arch.to_string())),
        };
        Ok(v)
    }
}

struct LlamaHparams {
    architecture: ModelArchitecture,
    model_name: String,
    vocab_only: bool,
    n_vocab: u32,
    n_ctx_train: u32, // context size the model was trained on
    n_head: u32,
    n_embd: u32,
    n_head_kv: u32,
    n_layer: u32,
    n_rot: u32,
    n_ff: u32,

    f_norm_eps: f32,
    f_norm_rms_eps: f32,

    rope_freq_base_train: f32,
    rope_freq_scale_train: f32,
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

impl LlamaHparams {
    fn load(r: &GGufContext) -> LLMResult<LlamaHparams> {
        let model_name = r
            .metas_data()
            .get(LLM_KV_GENERAL_ARCHITECTURE)
            .unwrap()
            .get_str()
            .unwrap();
        let n_vocab = r
            .metas_data()
            .get(LLM_KV_TOKENIZER_LIST)
            .unwrap()
            .get_arr_n()
            .unwrap();
        let n_ctx_train = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_CONTEXT_LENGTH).as_str())
            .unwrap()
            .get_u32()
            .unwrap();
        let n_embd = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_EMBEDDING_LENGTH).as_str())
            .unwrap()
            .get_u32()
            .unwrap();
        let n_ff = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_FEED_FORWARD_LENGTH).as_str())
            .unwrap()
            .get_u32()
            .unwrap();
        let n_head = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_ATTENTION_HEAD_COUNT).as_str())
            .unwrap()
            .get_u32()
            .unwrap();
        let n_layer = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_BLOCK_COUNT).as_str())
            .unwrap()
            .get_u32()
            .unwrap();
        let n_head_kv = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_ATTENTION_HEAD_COUNT_KV).as_str())
            .unwrap()
            .get_u32()
            .unwrap();

        let rope_freq_base_train = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_ROPE_FREQ_BASE).as_str())
            .unwrap_or_else(|| &GGufMetadataValue::F32(10000.0f32))
            .get_f32()
            .unwrap();

        let ropescale = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_ROPE_SCALE_LINEAR).as_str())
            .unwrap_or(&GGufMetadataValue::F32(1.0f32))
            .get_f32()
            .unwrap();

        let rope_freq_scale_train = 1.0f32 / ropescale;

        let n_rot = r
            .metas_data()
            .get(format!("{}{}", model_name, LLM_KV_ROPE_DIMENSION_COUNT).as_str())
            .unwrap_or(&GGufMetadataValue::U32(n_embd / n_head))
            .get_u32()
            .unwrap();

        println!("arch:{}", model_name);
        println!("n_vocab:{}", n_vocab);
        println!("n_ctx_train:{}", n_ctx_train);
        println!("n_embd:{}", n_embd);
        println!("n_ff:{}", n_ff);
        println!("n_head:{}", n_head);
        println!("n_layer:{}", n_layer);
        println!("n_head_kv:{}", n_head_kv);
        println!("rope_freq_base_train:{}", rope_freq_base_train);
        println!("ropescale:{}", ropescale);
        println!("n_rot:{}", n_rot);

        Ok(Self {
            architecture: ModelArchitecture::decode(model_name)?,
            model_name: model_name.to_string(),
            vocab_only: false,
            n_vocab: n_vocab as u32,
            n_ctx_train: n_ctx_train, // context size the model was trained on
            n_head: n_head,
            n_embd: n_embd,
            n_head_kv: n_head_kv,
            n_layer: n_layer,
            n_rot: n_rot,
            n_ff: n_ff,
            f_norm_eps: 0.0,
            f_norm_rms_eps: 0.0,
            rope_freq_base_train: rope_freq_base_train,
            rope_freq_scale_train: rope_freq_scale_train,
        })
    }
    // fn load<T: Read + BufRead>(r: &mut T) -> LLMResult<LlamaHparams> {
    //     let n_vocab: i32 = r.read_i32::<Endian>()?;
    //     let n_embd: i32 = r.read_i32::<Endian>()?;
    //     let n_mult: i32 = r.read_i32::<Endian>()?;
    //     let n_head: i32 = r.read_i32::<Endian>()?;
    //     let n_layer: i32 = r.read_i32::<Endian>()?;
    //     let n_rot: i32 = r.read_i32::<Endian>()?;
    //     let f16: i32 = r.read_i32::<Endian>()?;
    //     println!("{}: n_vocab  = {}", function!(), n_vocab);
    //     println!("{}: n_ctx    = {}", function!(), 512);
    //     println!("{}: n_embd   = {}", function!(), n_embd);
    //     println!("{}: n_mult   = {}", function!(), n_mult);
    //     println!("{}: n_head   = {}", function!(), n_head);
    //     println!("{}: n_layer  = {}", function!(), n_layer);
    //     println!("{}: n_rot    = {}", function!(), n_rot);
    //     println!("{}: f16      = {}", function!(), f16);
    //     Ok(LlamaHparams {
    //         n_vocab: n_vocab,
    //         n_ctx: 512,
    //         n_embd: n_embd,
    //         n_mult: n_mult,
    //         n_head: n_head,
    //         n_layer: n_layer,
    //         n_rot: n_rot,
    //         f16: f16,
    //     })
    // }
}

pub trait GGufRead {
    fn read_bytes(&mut self, n: usize) -> LLMResult<&[u8]>;

    fn read_len(&mut self) -> LLMResult<usize>;
    // 返回当前 offset
    fn offset(&self) -> usize;

    fn cursor(&self) -> &[u8];
}

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

impl LlamaModel {
    fn load<T: Read + GGufRead>(gguf_reader: &mut T) -> LLMResult<LlamaModel> {
        // let mmap_reader = gguf_reader.mmap_reader();
        let position = gguf_reader.offset();
        // let alignment = header.alignment() as usize;
        let next_position = position - (position % GGUF_DEFAULT_ALIGNMENT) + GGUF_DEFAULT_ALIGNMENT;
        let _ = gguf_reader.read_bytes(next_position - position)?;
        let tensor_data = gguf_reader.cursor();

        let mut tensor_ctx = TensorContext::new(tensor_data);

        todo!()
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

struct LLMContext {
    t_load_us: i64,
    t_mel_us: i64,
    t_sample_us: i64,
    t_encode_us: i64,
    t_decode_us: i64,
    t_start_us: i64,
    magic: u32,
    model: LlamaModel,
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

fn llama_tokenize(vocab: &GptVocab, text: &str, bos: bool) -> Vec<GptVocabId> {
    let mut res: Vec<GptVocabId> = Vec::new();
    let mut score: Vec<usize> = Vec::new();
    let mut prev: Vec<GptVocabId> = Vec::new();
    let len = text.len();

    score.resize(len + 1, 0);
    prev.resize(len + 1, 0);

    for i in 0..len {
        let max_len = std::cmp::min(len - i, MAX_TOKEN_LEN);
        for sub_len in 1..=max_len {
            let sub = &text[i..i + sub_len];
            if let Some(&token_id) = vocab.token_to_id.get(sub) {
                let token_score = sub.len() * sub.len();
                let local_score = score[i] + token_score;
                let next = i + sub_len;
                if score[next] < local_score {
                    score[next] = local_score;
                    prev[next] = token_id;
                }
            }
        }
    }

    let mut i = len;
    while i > 0 {
        let token_id = prev[i];
        if token_id == 0 {
            // TODO: Return error or something more meaningful
            eprintln!("failed to tokenize string!");
            break;
        }
        res.push(token_id);

        if let Some(token) = vocab.id_to_token.get(&token_id) {
            i -= token.len();
        } else {
            // Handle the case where token_id is not found in the vocabulary
            eprintln!("token_id not found in vocabulary!");
            break;
        }
    }

    if bos {
        res.push(1); // TODO: replace with vocab.bos
    }

    // Pieces are in reverse order so correct that
    res.reverse();

    return res;
}

// fn llama_eval(model: &LlamaModel, embd_inp: &[GptVocabId], mem_per_token: usize) -> LLMResult<()> {
//     let N = embd_inp.len();
//     let hparams = &model.hparams;

//     let n_embd = hparams.n_embd;
//     let n_layer = hparams.n_layer;
//     let n_ctx = hparams.n_ctx;
//     let n_head = hparams.n_head;
//     let n_vocab = hparams.n_vocab;
//     let n_rot = hparams.n_embd / hparams.n_head;
//     let d_key = n_embd / n_head;

//     let buf_size = 512 * 1024 * 1024;
//     let mut buf: Vec<u8> = vec![0u8; buf_size];
//     let mut tensor_ctx = TensorContext::new(&mut buf);

//     let mut embd = new_tensor_1d(&mut tensor_ctx, GGmlType::I32, N)?;

//     unsafe {
//         embd.as_slice_mut::<i32>().copy_from_slice(embd_inp);
//     }

//     let x: &[i32] = unsafe { embd.as_slice::<i32>() };
//     let mut sum: f64 = 0.0;
//     for i in 0..embd.elem_count() {
//         sum += x[i].abs() as f64;
//     }
//     println!(
//         "embd,sum:{:?},sha
//         pe:{:?},stride:{:?}",
//         sum,
//         embd.ggml_shape(),
//         embd.dim().stride_4d()
//     );

//     let inp_l = get_rows(&mut tensor_ctx, &model.tok_embeddings, &embd)?;

//     let x: &[f32] = unsafe { inp_l.as_slice::<f32>() };
//     let mut sum: f64 = 0.0;
//     for i in 0..inp_l.elem_count() {
//         sum += x[i].abs() as f64;
//     }
//     println!(
//         "get_rows,sum:{:?},sha
//         pe:{:?},stride:{:?}",
//         sum,
//         inp_l.ggml_shape(),
//         inp_l.dim().stride_4d()
//     );

//     for il in 0..n_layer {
//         let cur = rms_norm(&mut tensor_ctx, &inp_l)?;
//         let x: &[f32] = unsafe { cur.as_slice::<f32>() };
//         let mut sum: f64 = 0.0;
//         for i in 0..cur.elem_count() {
//             sum += x[i].abs() as f64;
//         }
//         println!(
//             "rms_norm,sum:{:?},sha
//             pe:{:?},stride:{:?}",
//             sum,
//             cur.ggml_shape(),
//             cur.dim().stride_4d()
//         );
//         return Ok(());
//     }

//     Ok(())
// }

// fn compute_ctx_size(hparams: &LlamaHparams) -> usize {
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
//     let n_parts = *LLAMA_N_PARTS.get(&hparams.n_embd).unwrap();
//     println!("{}: n_ff      = {}", function!(), n_ff);
//     println!("{}: n_parts   = {}", function!(), n_parts);
//     let wtype2 = GGmlType::F32;
//     let mut ctx_size = 0.0f32;
//     {
//         let n_embd = hparams.n_embd as f32;
//         let n_layer = hparams.n_layer as f32;
//         let n_ctx = hparams.n_ctx as f32;
//         let n_vocab = hparams.n_vocab as f32;

//         ctx_size += n_embd * n_vocab * get_type_sizef(wtype); // tok_embeddings

//         ctx_size += n_embd * get_type_sizef(GGmlType::F32); // norm

//         ctx_size += n_embd * n_vocab * get_type_sizef(wtype); // output

//         ctx_size += n_layer * (n_embd * get_type_sizef(GGmlType::F32)); // attention_norm

//         ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wq
//         ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wk
//         ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wv
//         ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wo

//         ctx_size += n_layer * (n_embd * get_type_sizef(GGmlType::F32)); // ffn_norm

//         ctx_size += n_layer * (n_ff * n_embd * get_type_sizef(wtype)); // w1
//         ctx_size += n_layer * (n_ff * n_embd * get_type_sizef(wtype)); // w2
//         ctx_size += n_layer * (n_ff * n_embd * get_type_sizef(wtype)); // w3

//         ctx_size += n_ctx * n_layer * n_embd * get_type_sizef(GGmlType::F32); // memory_k
//         ctx_size += n_ctx * n_layer * n_embd * get_type_sizef(GGmlType::F32); // memory_v

//         ctx_size += (5.0 + 10.0 * n_layer) * 256.0; // object overhead

//         println!(
//             "{}: ctx size = {:6.2} MB\n",
//             function!(),
//             ctx_size / (1024.0 * 1024.0),
//         );
//     }
//     return ctx_size as usize;
// }

// #[repr(u32)]
// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
// pub enum GGmlType {
//     F32 = 0,
//     F16 = 1,
//     Q4_0 = 2,
//     Q4_1 = 3,
//     // GGML_TYPE_Q4_2 = 4, support has been removed
//     // GGML_TYPE_Q4_3 (5) support has been removed
//     Q5_0 = 6,
//     Q5_1 = 7,
//     Q8_0 = 8,
//     Q8_1 = 9,
//     // k-quantizations
//     Q2K = 10,
//     Q3K = 11,
//     Q4K = 12,
//     Q5K = 13,
//     Q6K = 14,
//     Q8K = 15,
//     I8 = 16,
//     I16 = 17,
//     I32 = 18,
//     COUNT = 19,
// }

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

struct GGufContext {
    header: GGufHeader,
    metas: HashMap<GGufStr, GGufMetadataValue>,
    tensor_infos: HashMap<GGufStr, GGufDiskTensorInfo>,
}

impl GGufContext {
    fn metas_data(&self) -> &HashMap<GGufStr, GGufMetadataValue> {
        &self.metas
    }
}

struct GGufHeader {
    magic: u32,
    version: GGufVersion,
    n_tensors: u64,
    n_kv: u64,
}

struct GGufKV {
    key: GGufStr,
    value: GGufMetadataValue,
}

impl GGufKV {
    fn new(key: GGufStr, value: GGufMetadataValue) -> GGufKV {
        Self { key, value }
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

impl GGufHeader {
    fn load<T: Read>(r: &mut T) -> LLMResult<GGufHeader> {
        let magic = r.read_u32::<Endian>()?;
        if magic != GGUF_MAGIC {
            return Err(LLMError::BadMagic(magic));
        }
        let version = GGufVersion::decode(r.read_u32::<Endian>()?)?;
        let n_tensors = r.read_u64::<Endian>()?;
        let n_kv = r.read_u64::<Endian>()?;
        println!(
            "magic:{:x}, version:{:?},n_tensors:{},n_kv:{}",
            magic, version, n_tensors, n_kv,
        );
        Ok(GGufHeader {
            magic: magic,
            version: version,
            n_tensors: n_tensors,
            n_kv: n_kv,
        })
    }

    fn magic(&self) -> u32 {
        self.magic
    }

    fn n_tensors(&self) -> usize {
        self.n_tensors as usize
    }

    fn n_kv(&self) -> usize {
        self.n_kv as usize
    }
}

// struct LLamaGGufModel {
//     gguf_header: GGufHeader,
// }

// fn ggml_read() -> LLMResult<()> {
//     let model_path = "E:\\cproject\\models\\ggml-model-Q4.bin";
//     let mut fin = open_file_stream(model_path)?;
//     let magic = fin.read_u32::<Endian>()?;
//     if magic != GGML_MAGIC {
//         return Err(LLMError::BadMagic(model_path.to_string()));
//     }
//     // LLMContext::new(&model_path).unwrap();
//     let hparams = LlamaHparams::load(&mut fin)?;
//     let vocab = GptVocab::load(&mut fin, hparams.n_vocab)?;
//     let mut prompt = String::from("Building a website can be done in 10 simple steps:");
//     prompt.insert(0, ' ');
//     let embd_inp = llama_tokenize(&vocab, &prompt, true);
//     println!("{}: prompt: '{}'", function!(), prompt);
//     println!(
//         "{}: number of tokens in prompt = {}",
//         function!(),
//         embd_inp.len(),
//     );
//     for inp in embd_inp.iter() {
//         println!("{} -> '{}'\n", inp, vocab.id_to_token.get(inp).unwrap());
//     }
//     let ctx_size = compute_ctx_size(&hparams);
//     let mut buf_model = vec![0u8; ctx_size as usize];
//     let model = LlamaModel::load(&mut fin, hparams, &mut buf_model)?;
//     llama_eval(&model, &[0, 1, 2, 3], 0)?;
//     drop(buf_model);
//     drop(model);
//     Ok(())
// }

fn gguf_read() -> LLMResult<()> {
    let model_path = "E:\\cproject\\llama.cpp-gguf-fix-publish\\models\\llama-2-7b.Q4_0.gguf";
    let file = OpenOptions::new().read(true).open(model_path)?;
    let file_size = file.metadata()?.len();
    let mmap: Mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(&file)
            .map_err(|e| format!("mmap failed: {}", e))?
    };
    let mut mmap_reader = MmapReader::new(mmap, file_size as usize);
    let header = GGufHeader::load(&mut mmap_reader)?;
    let mut gguf_reader = GGufMmapReader::new(mmap_reader, header.version.clone());

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
        println!("str:{}", name.as_str());
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
        tensor_infos: gg_tensor_infos,
    };
    let h = LlamaHparams::load(&gguf_ctx)?;
    let vocab = LLamaVocab::load(&gguf_ctx)?;

    LlamaModel::load(&mut gguf_reader)?;
    Ok(())
}

fn main() -> LLMResult<()> {
    gguf_read()?;
    Ok(())
}
