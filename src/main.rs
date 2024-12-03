use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use galois::error::GError;
use galois::op::RopeCustomOption;
use galois::op::UnaryOp;
use galois::F16;
use memmap2::Mmap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::f32::INFINITY;
use std::fmt::Debug;
use std::io::BufReader;
use std::io::Error as IOError;
use std::io::Read;
use std::vec;
use thiserror::Error;
type Endian = LittleEndian;
use galois::ggml_quants::BlockQ4_0;
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

    fn to_vec(&self) -> Vec<P> {
        let mut vec = Vec::with_capacity(self.len);
        let mut current_ptr = self.ptr.as_ptr() as *const u8; // Treat as a byte pointer
        for _ in 0..self.len {
            // Copy element by element
            unsafe {
                // Read the data at the current pointer
                let value: P = std::ptr::read_unaligned(current_ptr as *const P);
                vec.push(value);
                // Move the pointer to the next element
                current_ptr = current_ptr.add(std::mem::size_of::<P>());
            }
        }
        vec
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

    fn get_f32_arr(&self) -> Option<Vec<f32>> {
        match self {
            GGufMetadataValue::Array(e) => match e {
                GGufArr::F32Array(arr) => Some(arr.to_vec()),
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
    // tensor_infos: HashMap<GGufStr, GGufDiskTensorInfo>,
}

impl<'a> TensorContext<'a> {
    fn new(buf: &'a [u8]) -> TensorContext<'a> {
        TensorContext {
            offset: 0,
            size: 0,
            n_objects: 0,
            buf: buf,
            //tensor_infos: tensor_infos,
        }
    }

    fn slice(&self, offset: usize, size: usize) -> &[u8] {
        &self.buf[offset..offset + size]
    }
}

fn create_tensor<const N: usize>(
    ctx: &mut TensorContext,
    tensor_infos: &HashMap<GGufStr, GGufDiskTensorInfo>,
    name: &str,
    ne: [usize; N],
) -> LLMResult<Tensor> {
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
    if cur_offset + size_needed > ctx.buf.len() {
        return Err(LLMError::NotEnoughSpace);
    }
    let t = unsafe {
        Tensor::from_bytes(
            &ctx.buf[cur_offset..cur_offset + size_needed],
            n_dims,
            Shape::from_array(ne),
            dtype,
        )
    };
    Ok(t)
}

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
    let dim = [ne0, ne1, ne3, ne4];
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

fn view_3d(
    a: &Tensor,
    ne0: usize,
    ne1: usize,
    ne2: usize,
    nb1: usize,
    nb2: usize,
    offset: usize,
) -> LLMResult<Tensor> {
    let dtype = a.dtype();
    let buf = a.as_bytes();
    let shape = Shape::from_array([ne0, ne1, ne2]);
    let mut t = view_tensor(&buf[offset..], 3, dtype, shape)?;
    let nb = t.stride_layout_mut();
    nb[1] = nb1;
    nb[2] = nb2;
    nb[3] = nb[2] * ne2;
    Ok(t)
}

fn transpose(a: &mut Tensor) -> LLMResult<Tensor> {
    Ok(a.transpose(0, 1)?)
}

fn permute(
    a: &Tensor,
    axis0: usize,
    axis1: usize,
    axis2: usize,
    axis3: usize,
) -> LLMResult<Tensor> {
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

    let mut dst = a.view();

    let mut ne = [0usize; MAX_DIM];
    let mut nb = [0usize; MAX_DIM];

    let (ne0, ne1, ne2, ne3) = a.dim4();
    let (nb0, nb1, nb2, nb3) = a.stride4();

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

fn cpy(src: &Tensor, dst: &mut Tensor) -> LLMResult<()> {
    // let mut dst = cur.view();
    galois::op::galois_cpy(src, dst)?;
    Ok(())
}

fn cont_4d(
    ctx: &mut TensorContext,
    a: &Tensor,
    ne0: usize,
    ne1: usize,
    ne2: usize,
    ne3: usize,
) -> LLMResult<Tensor> {
    let mut dst = new_tensor_4d(ctx, a.dtype(), ne0, ne1, ne2, ne3)?;
    galois::op::galois_cont(a, &mut dst)?;
    Ok(dst)
}

fn cont_2d(ctx: &mut TensorContext, a: &Tensor, ne0: usize, ne1: usize) -> LLMResult<Tensor> {
    cont_4d(ctx, a, ne0, ne1, 1, 1)
}

fn unary(ctx: &mut TensorContext, a: &Tensor, op: UnaryOp) -> LLMResult<Tensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_unary(a, &mut dst, op)?;
    Ok(dst)
}

fn silu(ctx: &mut TensorContext, a: &Tensor) -> LLMResult<Tensor> {
    unary(ctx, a, UnaryOp::Silu)
}

fn reshape_2d(a: &Tensor, ne0: usize, ne1: usize) -> LLMResult<Tensor> {
    assert!(a.ggml_is_contiguous());
    assert!(a.elem_count() == ne0 * ne1);
    let ne: [usize; 2] = [ne0, ne1];
    let result = view_tensor(a.as_bytes(), 2, a.dtype(), Shape::from_array(ne))?;
    Ok(result)
}

fn reshape_3d(a: &Tensor, ne0: usize, ne1: usize, ne2: usize) -> LLMResult<Tensor> {
    assert!(a.ggml_is_contiguous());
    assert!(a.elem_count() == ne0 * ne1 * ne2);
    let ne: [usize; 3] = [ne0, ne1, ne2];
    let result = view_tensor(a.as_bytes(), 3, a.dtype(), Shape::from_array(ne))?;
    Ok(result)
}

fn get_rows(ctx: &mut TensorContext, a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let mut dst = new_tensor_2d(ctx, GGmlType::F32, a.dim1(), b.dim1())?;
    galois::op::galois_get_rows(a, b, &mut dst)?;
    Ok(dst)
}

fn set_f32(a: &mut Tensor, value: f32) -> LLMResult<()> {
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

fn add(ctx: &mut TensorContext, a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_add(a, b, &mut dst)?;
    Ok(dst)
}

fn rms_norm(ctx: &mut TensorContext, a: &Tensor, eps: f32) -> LLMResult<Tensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_rms_norm(a, &mut dst, eps)?;
    Ok(dst)
}

fn soft_max(ctx: &mut TensorContext, a: &Tensor) -> LLMResult<Tensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_soft_max(a, &mut dst)?;
    Ok(dst)
}

fn repeat(ctx: &mut TensorContext, src: &Tensor, cur: &Tensor) -> LLMResult<Tensor> {
    let mut dst = new_tensor(
        ctx,
        cur.n_dims(),
        src.dtype(),
        Shape::from_slice(cur.shape()),
    )?;
    galois::op::galois_repeat(src, &mut dst)?;
    Ok(dst)
}

fn mul(ctx: &mut TensorContext, a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_mul(a, b, &mut dst)?;
    Ok(dst)
}

fn matmul(ctx: &mut TensorContext, a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let ne = [
        a.ggml_shape()[1],
        b.ggml_shape()[1],
        b.ggml_shape()[2],
        b.ggml_shape()[3],
    ];
    let mut dst = new_tensor(
        ctx,
        std::cmp::max(a.n_dims(), b.n_dims()),
        GGmlType::F32,
        Shape::from_array(ne),
    )?;
    galois::op::galois_matmul(a, b, &mut dst)?;
    Ok(dst)
}

fn scale_inplace(a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let mut dst = a.view();
    galois::op::galois_scale(a, b, &mut dst)?;
    Ok(dst)
}

fn scale(ctx: &mut TensorContext, a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let mut dst = dup_tensor(ctx, a)?;
    galois::op::galois_scale(a, b, &mut dst)?;
    Ok(dst)
}

fn rope_custom(
    ctx: &mut TensorContext,
    a: &Tensor,
    b: &Tensor,
    n_dims: usize,
    mode: i32,
    n_ctx: i32,
    freq_base: f32,
    freq_scale: f32,
    xpos_base: f32,
    xpos_down: bool,
) -> LLMResult<Tensor> {
    assert!(b.is_vector());
    assert!(b.dtype() == GGmlType::I32);
    assert!(a.dim_2() == b.dim_0());
    let mut dst = dup_tensor(ctx, a)?;
    println!("dst:shape:{:?}", dst.shape());

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

type LlamaToken = i32;
type LlamaPos = i32;
type LlamaSeqId = i32;

type Token = String;

struct GptVocab {
    n_vocab: i32,
    token_to_id: HashMap<Token, LlamaToken>,
    id_to_token: HashMap<LlamaToken, Token>,
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
    token_to_id: HashMap<Token, LlamaToken>,
    token_scores: HashMap<LlamaToken, f32>,
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
                    .unwrap();
                // .iter()
                // .map(|s| *s)
                // .collect::<Vec<_>>();
                let token_ids = vocab
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v.clone(), i as i32))
                    .collect::<HashMap<_, _>>();
                let token_scores = vocab_scores
                    .into_iter()
                    .enumerate()
                    .map(|(i, v)| (i as i32, v))
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

    output_norm: Tensor,

    output: Tensor,

    layers: Vec<LlamaLayer>,
    // key + value memory
    // memory_k: Tensor,
    // memory_v: Tensor,
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
            f_norm_eps: 0.0e+00,
            f_norm_rms_eps: 1.0e-05,
            rope_freq_base_train: rope_freq_base_train,
            rope_freq_scale_train: rope_freq_scale_train,
        })
    }

    fn n_gqa(&self) -> u32 {
        return self.n_head / self.n_head_kv;
    }

    fn n_embd_gqa(&self) -> u32 {
        self.n_embd / self.n_gqa()
    }

    fn n_embd_head(&self) -> u32 {
        self.n_embd / self.n_head
    }
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

type LLamaToken = u32;

impl LlamaModel {
    fn eval(llama_batch: Vec<LLamaToken>) {}

    fn load<T: Read + GGufRead>(
        gguf_reader: &mut T,
        hparams: LlamaHparams,
        tensor_infos: HashMap<GGufStr, GGufDiskTensorInfo>,
    ) -> LLMResult<LlamaModel> {
        let n_embd = hparams.n_embd as usize;
        let n_embd_gqa = hparams.n_embd_gqa() as usize;
        let n_layer = hparams.n_layer as usize;
        let n_vocab = hparams.n_vocab as usize;
        let n_ff = hparams.n_ff as usize;

        let position = gguf_reader.offset();
        // let alignment = header.alignment() as usize;
        let next_position = position - (position % GGUF_DEFAULT_ALIGNMENT) + GGUF_DEFAULT_ALIGNMENT;
        // 跳过 内存对齐
        let _ = gguf_reader.read_bytes(next_position - position)?;
        let tensor_data = gguf_reader.cursor();

        let mut tensor_ctx = TensorContext::new(tensor_data);

        let tok_embeddings = create_tensor(
            &mut tensor_ctx,
            &tensor_infos,
            "token_embd.weight",
            [n_embd, n_vocab],
        )?;
        let output_norm = create_tensor(
            &mut tensor_ctx,
            &tensor_infos,
            "output_norm.weight",
            [n_embd],
        )?;

        let output = create_tensor(
            &mut tensor_ctx,
            &tensor_infos,
            "output.weight",
            [n_embd, n_vocab],
        )?;
        // let output = create_tensor(&mut tensor_ctx, "output.weight", [n_embd, n_vocab])?;
        // let output_norm = create_tensor(&mut tensor_ctx, "output_norm.weight", [n_embd]);
        let mut layers = Vec::new();
        for i in 0..n_layer {
            let attention_norm = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.attn_norm.weight", i),
                [n_embd],
            )?;

            let wq = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.attn_q.weight", i),
                [n_embd, n_embd],
            )?;
            let wk = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.attn_k.weight", i),
                [n_embd, n_embd_gqa],
            )?;
            let wv = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.attn_v.weight", i),
                [n_embd, n_embd_gqa],
            )?;
            let wo = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.attn_output.weight", i),
                [n_embd, n_embd],
            )?;

            let ffn_norm = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.ffn_norm.weight", i),
                [n_embd],
            )?;

            let w1 = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.ffn_gate.weight", i),
                [n_embd, n_ff],
            )?;
            let w2 = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.ffn_down.weight", i),
                [n_ff, n_embd],
            )?;
            let w3 = create_tensor(
                &mut tensor_ctx,
                &tensor_infos,
                &format!("blk.{}.ffn_up.weight", i),
                [n_embd, n_ff],
            )?;

            layers.push(LlamaLayer {
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
            layers: layers,
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

fn llama_tokenize(vocab: &GptVocab, text: &str, bos: bool) -> Vec<LlamaToken> {
    let mut res: Vec<LlamaToken> = Vec::new();
    let mut score: Vec<usize> = Vec::new();
    let mut prev: Vec<LlamaToken> = Vec::new();
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

#[derive(Clone)]
struct LlamaKvCell {
    pos: LlamaPos,
    delta: LlamaPos,
    seq_id: HashSet<LlamaSeqId>,
}

impl Default for LlamaKvCell {
    fn default() -> Self {
        Self {
            pos: -1,
            delta: 0,
            seq_id: HashSet::new(),
        }
    }
}

impl LlamaKvCell {
    fn new() -> Self {
        LlamaKvCell {
            pos: -1,
            delta: 0,
            seq_id: HashSet::new(),
        }
    }

    fn has_seq_id(&self, id: &LlamaSeqId) -> bool {
        self.seq_id.contains(id)
    }
}

struct LlamaKvCache {
    head: usize,
    size: usize,
    n: usize,
    cells: Vec<LlamaKvCell>,
}

impl LlamaKvCache {
    fn new(n_ctx: usize) -> LlamaKvCache {
        Self {
            head: 0,
            size: n_ctx,
            n: 0,
            cells: vec![LlamaKvCell::default(); n_ctx],
        }
    }

    fn llama_kv_cache_find_slot(&mut self, batch: &LlamaBatch) -> bool {
        let n_ctx = self.size;
        let n_tokens = batch.n_token();

        if n_tokens > n_ctx {
            eprintln!("Error: n_tokens={} > n_ctx={}", n_tokens, n_ctx);
            return false;
        }

        let mut n_tested = 0;

        loop {
            if (self.head + n_tokens) > n_ctx {
                n_tested += n_ctx - self.head;
                self.head = 0;
                continue;
            }

            let mut found = true;
            for i in 0..n_tokens {
                if self.cells[self.head + i].pos >= 0 {
                    found = false;
                    self.head += i + 1;
                    n_tested += i + 1;
                    break;
                }
            }

            if found {
                break;
            }

            if n_tested >= n_ctx {
                eprintln!("Error: n_tested={} > n_ctx={}", n_tested, n_ctx);
                // Optionally log an error here if needed
                return false;
            }
        }

        // Final assignment of positions and seq_id for the batch tokens
        for i in 0..n_tokens {
            let index = self.head + i;
            self.cells[index].pos = batch.pos[i];
            self.cells[index].seq_id.insert(batch.seq_id[i]);
        }

        true
    }
}

fn llama_eval(
    model: &LlamaModel,
    batch: &LlamaBatch,
    kv_self: &LlamaKvCache,
    mem_per_token: usize,
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
    let mut buf: Vec<u8> = vec![0u8; buf_size];
    let mut tensor_ctx = TensorContext::new(&mut buf);

    // let mut buf_kv: Vec<u8> = vec![0u8; 10 * 1024 * 1024];
    //let mut kv_buf_ctx = TensorContext::new(&mut buf_kv);

    let memory_k = new_tensor_1d(&mut tensor_ctx, GGmlType::F16, n_elements)?;
    let memory_v = new_tensor_1d(&mut tensor_ctx, GGmlType::F16, n_elements)?;
    let mut embd = new_tensor_1d(&mut tensor_ctx, GGmlType::I32, n_tokens)?;

    unsafe {
        embd.as_slice_mut::<i32>().copy_from_slice(embd_inp);
    }

    let mut inp_l = get_rows(&mut tensor_ctx, &model.tok_embeddings, &embd)?;

    let mut KQ_scale = new_tensor_1d(&mut tensor_ctx, GGmlType::F32, 1)?;
    set_f32(&mut KQ_scale, 1.0f32 / (n_embd_head as f32).sqrt())?;

    let mut KQ_mask = new_tensor_3d(&mut tensor_ctx, GGmlType::F32, n_kv, n_tokens, 1)?;

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

    let mut KQ_pos = new_tensor_1d(&mut tensor_ctx, GGmlType::I32, n_tokens)?;

    unsafe {
        let s = KQ_pos.as_slice_mut::<i32>();
        for i in 0..n_tokens {
            s[i] = batch.pos[i]
        }
    }

    for il in 0..1 as usize {
        //let x = unsafe { model.layers[il].wk.as_slice::<BlockQ4_0>() };
        // let mut sum: i32 = 0;
        // for i in 0..x.len() {
        //     sum += x[i].qs().iter().map(|e| *e as i32).sum::<i32>();
        // }
        // println!(
        //     "wk,sum:{:?},shape:{:?},stride:{:?},elem_count:{:?}",
        //     sum,
        //     model.layers[il].wk.ggml_shape(),
        //     model.layers[il].wk.dim().stride_4d(),
        //     model.layers[il].wk.elem_count(),
        // );
        let inpSA = &inp_l;
        let mut cur = rms_norm(&mut tensor_ctx, &inp_l, norm_rms_eps)?;

        cur = mul(&mut tensor_ctx, &cur, &model.layers[il].attention_norm)?;

        {
            let tmpk = matmul(&mut tensor_ctx, &model.layers[il].wk, &cur)?;

            let x: &[f32] = unsafe { tmpk.as_slice::<f32>() };
            let mut sum: f32 = 0.0;
            for i in 0..tmpk.elem_count() {
                sum += x[i];
            }

            let tmpq = matmul(&mut tensor_ctx, &model.layers[il].wq, &cur)?;

            let Kcur = rope_custom(
                &mut tensor_ctx,
                &reshape_3d(&tmpk, n_embd_head, n_head_kv, n_tokens)?,
                &KQ_pos,
                n_embd_head,
                0,
                0,
                freq_base,
                freq_scale,
                0.0f32,
                false,
            )?;

            let Qcur = rope_custom(
                &mut tensor_ctx,
                &reshape_3d(&tmpq, n_embd_head, n_head, n_tokens)?,
                &KQ_pos,
                n_embd_head,
                0,
                0,
                freq_base,
                freq_scale,
                0.0f32,
                false,
            )?;

            // store key and value to memory
            {
                let tmpv = matmul(&mut tensor_ctx, &model.layers[il].wv, &cur)?;
                let Vcur = transpose(&mut reshape_2d(&tmpv, n_embd_gqa, n_tokens)?)?;

                let mut k = view_1d(
                    &memory_k,
                    n_tokens * n_embd_gqa,
                    (memory_k.elem_size() * n_embd_gqa) * (il * n_ctx + kv_head),
                )?;

                let mut v = view_2d(
                    &memory_v,
                    n_tokens,
                    n_embd_gqa,
                    n_ctx,
                    (il * n_ctx) * memory_v.elem_size() * n_embd_gqa
                        + kv_head * memory_v.elem_size(),
                )?;

                // important: storing RoPE-ed version of K in the KV cache!
                cpy(&Kcur, &mut k)?;
                cpy(&Vcur, &mut v)?;
            }

            let Q = permute(&Qcur, 0, 2, 1, 3)?;

            let K = view_3d(
                &memory_k,
                n_embd_head,
                n_kv,
                n_head_kv,
                n_embd_gqa,
                n_embd_head,
                memory_k.elem_size() * n_embd_gqa * n_ctx * il,
            )?;

            let KQ = matmul(&mut tensor_ctx, &K, &Q)?;

            let KQ_scaled = scale(&mut tensor_ctx, &KQ, &KQ_scale)?;

            let KQ_masked = add(&mut tensor_ctx, &KQ_scaled, &KQ_mask)?;

            let KQ_soft_max = soft_max(&mut tensor_ctx, &KQ_masked)?;

            let V = view_3d(
                &memory_v,
                n_kv,
                n_embd_head,
                n_head_kv,
                n_ctx,
                n_ctx * n_embd_head,
                memory_v.elem_size() * n_ctx * n_embd_gqa * il,
            )?;
            let KQV = matmul(&mut tensor_ctx, &V, &KQ_soft_max)?;

            let KQV_merged = permute(&KQV, 0, 2, 1, 3)?;

            cur = cont_2d(&mut tensor_ctx, &KQV_merged, n_embd, n_tokens)?;

            cur = matmul(&mut tensor_ctx, &model.layers[il].wo, &cur)?;
        }

        let inpFF = add(&mut tensor_ctx, &cur, inpSA)?;

        // feed-forward network
        {
            {
                cur = rms_norm(&mut tensor_ctx, &inpFF, norm_rms_eps)?;

                cur = mul(&mut tensor_ctx, &cur, &model.layers[il].ffn_norm)?;

                // let x: &[f32] = unsafe { cur.as_slice::<f32>() };
                // let mut sum: f32 = 0.0;
                // for i in 0..cur.elem_count() {
                //     sum += x[i];
                // }
                // println!(
                //     "cur,sum:{:?},shape:{:?},stride:{:?}",
                //     sum,
                //     cur.shape_layout(),
                //     cur.dim().stride_4d()
                // );
                // return Ok(());
            }

            let tmp = matmul(&mut tensor_ctx, &model.layers[il].w3, &cur)?;
            cur = matmul(&mut tensor_ctx, &model.layers[il].w1, &cur)?;

            // SILU activation
            cur = silu(&mut tensor_ctx, &cur)?;

            cur = mul(&mut tensor_ctx, &cur, &tmp)?;

            cur = matmul(&mut tensor_ctx, &model.layers[il].w2, &cur)?;
        }

        cur = add(&mut tensor_ctx, &cur, &inpFF)?;

        // input for next layer
        inp_l = cur;
    }

    let mut cur = inp_l;

    {
        cur = rms_norm(&mut tensor_ctx, &cur, norm_rms_eps)?;

        cur = mul(&mut tensor_ctx, &cur, &model.output_norm)?;
    }

    // lm_head
    cur = matmul(&mut tensor_ctx, &model.output, &cur)?;

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

struct GGufContext {
    header: GGufHeader,
    metas: HashMap<GGufStr, GGufMetadataValue>,
    //   tensor_infos: HashMap<GGufStr, GGufDiskTensorInfo>,
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

struct LlamaBatch {
    token: Vec<LlamaToken>,
    pos: Vec<LlamaPos>,
    seq_id: Vec<LlamaSeqId>,
}

impl LlamaBatch {
    fn n_token(&self) -> usize {
        self.token.len()
    }

    fn embd_inp(&self) -> &[LlamaToken] {
        &self.token
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
        // tensor_infos: gg_tensor_infos,
    };
    let h = LlamaHparams::load(&gguf_ctx)?;
    let vocab = LLamaVocab::load(&gguf_ctx)?;

    let model = LlamaModel::load(&mut gguf_reader, h, gg_tensor_infos)?;

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

    llama_eval(&model, &batch, &kv_cache, 0).unwrap();
    Ok(())
}

fn main() -> LLMResult<()> {
    gguf_read()?;
    Ok(())
}
