use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use galois::error::GError;
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Error as IOError;
use std::io::Read;
use thiserror::Error;
type Endian = LittleEndian;
use galois::DType;
use galois::Shape;
use galois::Tensor;
use galois::GS_BLCK_SIZE;
use galois::GS_TYPE_SIZE;
use lazy_static::lazy_static;
use std::fs::File;

fn get_type_size(t: DType) -> usize {
    return GS_TYPE_SIZE[t as usize];
}

fn get_type_sizef(t: DType) -> f32 {
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
    #[error("Unexpected: {0}")]
    Unexpected(String),
    #[error("Unexpected IO: {0}")]
    UnexpectIO(IOError),
    #[error("invalid model file '{0}' (bad magic)\n")]
    BadMagic(String),
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

fn new_tensor_1d(ctx: &mut TensorContext, dtype: DType, ne0: usize) -> LLMResult<Tensor> {
    let dim = [ne0];
    new_tensor(ctx, 1, dtype, Shape::from_array(dim))
}

fn new_tensor_2d(
    ctx: &mut TensorContext,
    dtype: DType,
    ne0: usize,
    ne1: usize,
) -> LLMResult<Tensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, 2, dtype, Shape::from_array(dim))
}

fn new_tensor_3d(
    ctx: &mut TensorContext,
    dtype: DType,
    ne0: usize,
    ne1: usize,
    ne2: usize,
) -> LLMResult<Tensor> {
    let dim = [ne0, ne1, ne2];
    new_tensor(ctx, 3, dtype, Shape::from_array(dim))
}

fn new_tensor_4d(
    ctx: &mut TensorContext,
    dtype: DType,
    ne0: usize,
    ne1: usize,
    ne3: usize,
    ne4: usize,
) -> LLMResult<Tensor> {
    let dim = [ne0, ne1];
    new_tensor(ctx, 2, dtype, Shape::from_array(dim))
}

fn new_f32_tensor(ctx: &mut TensorContext, value: f32) -> LLMResult<Tensor> {
    let mut result = new_tensor_1d(ctx, DType::F32, 1)?;
    result.set_value(value);
    Ok(result)
}

fn new_tensor(
    ctx: &mut TensorContext,
    n_dims: usize,
    dtype: DType,
    shape: Shape,
) -> LLMResult<Tensor> {
    let cur_offset = ctx.offset;
    let cur_size = ctx.size;
    let size_needed: usize = get_type_size(dtype) * shape.size();
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

fn view_tensor(buf: &[u8], n_dims: usize, dtype: DType, shape: Shape) -> LLMResult<Tensor> {
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

type Id = i32;
type Token = String;

struct GptVocab {
    n_vocab: i32,
    token_to_id: HashMap<Token, Id>,
    id_to_token: HashMap<Id, Token>,
}

impl GptVocab {
    fn load<T: Read + BufRead>(r: &mut T, n_vocab: i32) -> LLMResult<GptVocab> {
        let mut token_to_id: HashMap<Token, Id> = HashMap::new();
        let mut id_to_token: HashMap<Id, Token> = HashMap::new();
        for i in 0..n_vocab {
            let len: u32 = r.read_u32::<Endian>()?;
            let mut tmp = vec![0; len as usize];
            r.read_exact(&mut tmp)?;
            let word = String::from_utf8_lossy(&tmp).to_string();
            if i == 50256 {
                println!("{}: vocab[{}] =       = {}\n", function!(), i, word);
            }
            token_to_id.insert(word.clone(), i);
            id_to_token.insert(i, word);
        }

        Ok(GptVocab {
            n_vocab,
            token_to_id,
            id_to_token,
        })
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

struct LlamaHparams {
    n_vocab: i32,
    n_ctx: i32,
    n_embd: i32,
    n_mult: i32,
    n_head: i32,
    n_layer: i32,
    n_rot: i32,
    f16: i32,
}

impl Default for LlamaHparams {
    fn default() -> Self {
        Self {
            n_vocab: 32000,
            n_ctx: 512,
            n_embd: 4096,
            n_mult: 256,
            n_head: 32,
            n_layer: 32,
            n_rot: 64,
            f16: 1,
        }
    }
}

struct LlamaModel {
    mtype: EModel,
    hparams: LlamaHparams,
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

impl LlamaHparams {
    fn load<T: Read + BufRead>(r: &mut T) -> LLMResult<LlamaHparams> {
        let n_vocab: i32 = r.read_i32::<Endian>()?;
        let n_embd: i32 = r.read_i32::<Endian>()?;
        let n_mult: i32 = r.read_i32::<Endian>()?;
        let n_head: i32 = r.read_i32::<Endian>()?;
        let n_layer: i32 = r.read_i32::<Endian>()?;
        let n_rot: i32 = r.read_i32::<Endian>()?;
        let f16: i32 = r.read_i32::<Endian>()?;
        println!("{}: n_vocab  = {}", function!(), n_vocab);
        println!("{}: n_ctx    = {}", function!(), 512);
        println!("{}: n_embd   = {}", function!(), n_embd);
        println!("{}: n_mult   = {}", function!(), n_mult);
        println!("{}: n_head   = {}", function!(), n_head);
        println!("{}: n_layer  = {}", function!(), n_layer);
        println!("{}: n_rot    = {}", function!(), n_rot);
        println!("{}: f16      = {}", function!(), f16);
        Ok(LlamaHparams {
            n_vocab: n_vocab,
            n_ctx: 512,
            n_embd: n_embd,
            n_mult: n_mult,
            n_head: n_head,
            n_layer: n_layer,
            n_rot: n_rot,
            f16: f16,
        })
    }
}

impl LlamaModel {
    fn load<T: Read + BufRead>(r: &mut T) -> LLMResult<LlamaModel> {
        let hparams = LlamaHparams::load(r)?;
        GptVocab::load(r, hparams.n_vocab)?;
        let wtype = match hparams.f16 {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::Q4_0,
            _ => {
                todo!()
            }
        };
        let n_ff = (((2 * (4 * hparams.n_embd) / 3 + hparams.n_mult - 1) / hparams.n_mult)
            * hparams.n_mult) as f32;
        let n_parts = LLAMA_N_PARTS.get(&hparams.n_embd).unwrap();
        println!("{}: n_ff      = {}", function!(), n_ff);
        println!("{}: n_parts   = {}", function!(), n_parts);
        let wtype2 = DType::F32;
        let mut ctx_size = 0.0f32;
        {
            let n_embd = hparams.n_embd as f32;
            let n_layer = hparams.n_layer as f32;
            let n_ctx = hparams.n_ctx as f32;
            let n_vocab = hparams.n_vocab as f32;

            ctx_size += n_embd * n_vocab * get_type_sizef(wtype); // tok_embeddings

            ctx_size += n_embd * get_type_sizef(DType::F32); // norm

            ctx_size += n_embd * n_vocab * get_type_sizef(wtype); // output

            ctx_size += n_layer * (n_embd * get_type_sizef(DType::F32)); // attention_norm

            ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wq
            ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wk
            ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wv
            ctx_size += n_layer * (n_embd * n_embd * get_type_sizef(wtype)); // wo

            ctx_size += n_layer * (n_embd * get_type_sizef(DType::F32)); // ffn_norm

            ctx_size += n_layer * (n_ff * n_embd * get_type_sizef(wtype)); // w1
            ctx_size += n_layer * (n_ff * n_embd * get_type_sizef(wtype)); // w2
            ctx_size += n_layer * (n_ff * n_embd * get_type_sizef(wtype)); // w3

            ctx_size += n_ctx * n_layer * n_embd * get_type_sizef(DType::F32); // memory_k
            ctx_size += n_ctx * n_layer * n_embd * get_type_sizef(DType::F32); // memory_v

            ctx_size += (5.0 + 10.0 * n_layer) * 256.0; // object overhead

            println!(
                "{}: ctx size = {:6.2} MB\n",
                function!(),
                ctx_size / (1024.0 * 1024.0),
            );
        }

        let mut buf_model = vec![0u8; ctx_size as usize];
        {
            let mut tensors: HashMap<String, *mut Tensor> = HashMap::new();
            let mut tensor_ctx = TensorContext::new(&mut buf_model);
            let n_embd = hparams.n_embd;
            let n_layer = hparams.n_layer;
            let n_ctx = hparams.n_ctx;
            let n_vocab = hparams.n_vocab;
        }

        // let wtype = if hparams.f16 == 1 {
        //     DType::F16
        // } else {
        //     DType::F32
        // };
        todo!()
    }
}

struct LLMContext {
    t_load_us: i64,
    t_mel_us: i64,
    t_sample_us: i64,
    t_encode_us: i64,
    t_decode_us: i64,
    t_start_us: i64,
    magic: u32,
    gguf: GGUFFile,
    model: LlamaModel,
}

impl LLMContext {
    fn new(fname: &str) -> LLMResult<LLMContext> {
        let mut fin = open_file_stream(fname)?;
        let magic = fin.read_u32::<Endian>()?;
        if magic != GGML_MAGIC {
            return Err(LLMError::BadMagic(fname.to_string()));
        }
        LlamaModel::load(&mut fin)?;
        // let version = file.read_u32::<Endian>()?;

        todo!()
    }
}

// 打开文件流
fn open_file_stream(fname: &str) -> LLMResult<BufReader<File>> {
    let file = File::open(fname)?;
    let buf_reader = BufReader::new(file);
    Ok(buf_reader)
}

fn main() {
    let model_path = "/opt/cproject/models/ggml-model-Q4.bin";
    LLMContext::new(&model_path).unwrap();
}
