use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use galois::error::GError;
use memmap2::Mmap;
use std::collections::HashMap;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Error as IOError;
use std::io::Read;
use std::process::exit;
use thiserror::Error;
type Endian = LittleEndian;
use galois::ggml_quants::BlockQ4_0;
use galois::DType;
use galois::Shape;
use galois::Tensor;
use galois::GS_BLCK_SIZE;
use galois::GS_TYPE_SIZE;
use lazy_static::lazy_static;
use std::fs::File;

fn get_blck_size(t: DType) -> usize {
    return GS_BLCK_SIZE[t as usize];
}

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

const GGML_MEM_ALIGN: usize = 16;

fn new_tensor(
    ctx: &mut TensorContext,
    n_dims: usize,
    dtype: DType,
    shape: Shape,
) -> LLMResult<Tensor> {
    let cur_offset = ctx.offset;
    let cur_size = ctx.size;
    let ne = shape.layout();
    let mut size_needed: usize = get_type_size(dtype) * (ne[0] / get_blck_size(dtype));
    for i in 1..n_dims {
        size_needed *= ne[i];
    }
    size_needed = ((size_needed + GGML_MEM_ALIGN - 1) / GGML_MEM_ALIGN) * GGML_MEM_ALIGN;
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

fn get_rows(ctx: &mut TensorContext, a: &Tensor, b: &Tensor) -> LLMResult<Tensor> {
    let mut dst = new_tensor_2d(ctx, DType::F32, a.dim1(), b.dim1())?;
    galois::op::galois_get_rows(a, b, &mut dst)?;
    Ok(dst)
}

type GptVocabId = i32;
type Token = String;

struct GptVocab {
    n_vocab: i32,
    token_to_id: HashMap<Token, GptVocabId>,
    id_to_token: HashMap<GptVocabId, Token>,
}

impl GptVocab {
    fn load<T: Read + BufRead>(r: &mut T, n_vocab: i32) -> LLMResult<GptVocab> {
        let mut token_to_id: HashMap<Token, GptVocabId> = HashMap::new();
        let mut id_to_token: HashMap<GptVocabId, Token> = HashMap::new();
        for i in 0..n_vocab {
            let len: u32 = r.read_u32::<Endian>()?;
            let mut tmp = vec![0; len as usize];
            r.read_exact(&mut tmp)?;
            let word = String::from_utf8_lossy(&tmp).to_string();
            // if i == 1111 {
            //     println!("{}: vocab[{}] =       = {}\n", function!(), i, word);
            // }
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
    fn load<T: Read + BufRead>(
        r: &mut T,
        hparams: LlamaHparams,
        buf_model: &mut [u8],
    ) -> LLMResult<LlamaModel> {
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
        // let hparams = LlamaHparams::load(r)?;
        // GptVocab::load(r, hparams.n_vocab)?;
        let n_parts = 1;
        let mut tensors: HashMap<String, *mut Tensor> = HashMap::new();

        let model = {
            let mut tensor_ctx = TensorContext::new(buf_model);
            let n_embd = hparams.n_embd as usize;
            let n_layer = hparams.n_layer as usize;
            let n_ctx = hparams.n_ctx as usize;
            let n_vocab = hparams.n_vocab as usize;

            let mut layers: Vec<LlamaLayer> = Vec::with_capacity(n_layer);

            let mut tok_embeddings = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_vocab)?;
            let mut norm = new_tensor_1d(&mut tensor_ctx, DType::F32, n_embd)?;
            let mut output = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_vocab)?;

            tensors.insert(
                "tok_embeddings.weight".to_string(),
                &mut tok_embeddings as *mut Tensor,
            );
            tensors.insert("norm.weight".to_string(), &mut norm as *mut Tensor);
            tensors.insert("output.weight".to_string(), &mut output as *mut Tensor);

            for i in 0..n_layer {
                let attention_norm = new_tensor_1d(&mut tensor_ctx, DType::F32, n_embd)?;

                let wq = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;
                let wk = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;
                let wv = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;
                let wo = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_embd)?;

                let ffn_norm = new_tensor_1d(&mut tensor_ctx, DType::F32, n_embd)?;

                let w1 = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_ff as usize)?;
                let w2 = new_tensor_2d(&mut tensor_ctx, wtype, n_ff as usize, n_embd)?;
                let w3 = new_tensor_2d(&mut tensor_ctx, wtype, n_embd, n_ff as usize)?;

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

                let layer = layers.last_mut().unwrap();

                tensors.insert(
                    format!("layers.{}.attention_norm.weight", i),
                    &mut layer.attention_norm as *mut Tensor,
                );

                tensors.insert(
                    format!("layers.{}.attention.wq.weight", i),
                    &mut layer.wq as *mut Tensor,
                );
                tensors.insert(
                    format!("layers.{}.attention.wk.weight", i),
                    &mut layer.wk as *mut Tensor,
                );
                tensors.insert(
                    format!("layers.{}.attention.wv.weight", i),
                    &mut layer.wv as *mut Tensor,
                );
                tensors.insert(
                    format!("layers.{}.attention.wo.weight", i),
                    &mut layer.wo as *mut Tensor,
                );

                tensors.insert(
                    format!("layers.{}.ffn_norm.weight", i),
                    &mut layer.ffn_norm as *mut Tensor,
                );

                tensors.insert(
                    format!("layers.{}.feed_forward.w1.weight", i),
                    &mut layer.w1 as *mut Tensor,
                );
                tensors.insert(
                    format!("layers.{}.feed_forward.w2.weight", i),
                    &mut layer.w2 as *mut Tensor,
                );
                tensors.insert(
                    format!("layers.{}.feed_forward.w3.weight", i),
                    &mut layer.w3 as *mut Tensor,
                );
            }

            let n_embd = hparams.n_embd as usize;
            let n_layer = hparams.n_layer as usize;
            let n_ctx = hparams.n_ctx as usize;
            let n_vocab = hparams.n_vocab as usize;

            let n_mem = n_layer * n_ctx;
            let n_elements = n_embd * n_mem;

            let memory_k = new_tensor_1d(&mut tensor_ctx, DType::F32, n_elements)?;
            let memory_v = new_tensor_1d(&mut tensor_ctx, DType::F32, n_elements)?;
            let memory_size = memory_k.nbytes() + memory_v.nbytes();
            println!(
                "{}: memory_size = {:8.2} MB, n_mem = {},offset={}",
                function!(),
                memory_size as f32 / 1024.0 / 1024.0,
                n_mem,
                tensor_ctx.offset,
            );
            LlamaModel {
                hparams,
                tok_embeddings,
                norm,
                output,
                layers,
                // key + value memory
                memory_k,
                memory_v,
            }
        };
        {
            let mut total_size: usize = 0;
            let j: usize = 0;
            loop {
                let n_dims = r.read_i32::<Endian>()?;
                let length = r.read_i32::<Endian>()?;
                let ftype = r.read_i32::<Endian>()?;

                let mut nelements: usize = 1;
                let mut ne: [usize; 2] = [1, 1];
                // let n_dims = 3; // Assume this value is set appropriately
                print!(".");
                for i in 0..n_dims as usize {
                    ne[i] = r.read_i32::<Endian>()? as usize;
                    nelements *= ne[i];
                }
                //  println!("nelements:{}", nelements);
                let mut buffer = vec![0; length as usize];
                r.read_exact(&mut buffer)?;
                let name = String::from_utf8_lossy(&buffer).to_string();
                // println!("name:{}", name);
                let ref_tensor = tensors
                    .get_mut(name.as_str())
                    .ok_or(LLMError::UnknownTensor(name.clone()))?;

                if let Some(tensor) = unsafe { (*ref_tensor).as_mut() } {
                    let split_type = if name.contains("tok_embeddings") {
                        0
                    } else if name.contains("layers") {
                        if name.contains("attention.wo.weight") {
                            0
                        } else if name.contains("feed_forward.w2.weight") {
                            0
                        } else {
                            1
                        }
                    } else if name.contains("output") {
                        1
                    } else {
                        // Define a default split_type if needed
                        // For instance, if none of the conditions match
                        // you can decide what to return
                        0
                    };

                    if n_dims == 1 {
                        if tensor.elem_count() != nelements {
                            return Err(LLMError::WrongSizeTensor(
                                name,
                                tensor.elem_count(),
                                nelements,
                            ));
                        }
                    } else {
                        if tensor.elem_count() / n_parts != nelements {
                            return Err(LLMError::WrongSizeTensor(
                                name,
                                tensor.elem_count(),
                                nelements,
                            ));
                        }
                    }
                    let (ne0, ne1) = tensor.dim2();
                    if n_dims == 1 {
                        if ne0 != ne[0] || ne1 != ne[1] {
                            return Err(LLMError::WrongShapeTensor(
                                name,
                                vec![ne0, ne1],
                                ne.to_vec(),
                            ));
                        }
                    } else {
                        if split_type == 0 {
                            if ne0 / n_parts != ne[0] || ne1 != ne[1] {
                                return Err(LLMError::WrongShapeTensor(
                                    name,
                                    vec![ne0 / n_parts, ne1],
                                    ne.to_vec(),
                                ));
                            }
                        } else {
                            if ne0 != ne[0] || (ne1 / n_parts) != ne[1] {
                                return Err(LLMError::WrongShapeTensor(
                                    name,
                                    vec![ne0, ne1 / n_parts],
                                    ne.to_vec(),
                                ));
                            }
                        }
                    }

                    let bpe = match ftype {
                        0 => get_type_size(DType::F32),
                        1 => get_type_size(DType::F16),
                        2 => {
                            assert!(ne[0] % 64 == 0);
                            get_type_size(DType::Q4_0)
                        }
                        _ => {
                            return Err(LLMError::UnknownFtypeGTensor(ftype));
                        }
                    };

                    if (nelements * bpe) / get_blck_size(tensor.dtype()) != tensor.nbytes() {
                        return Err(LLMError::WrongBytesTensor(
                            name,
                            (nelements * bpe) / get_blck_size(tensor.dtype()),
                            tensor.nbytes(),
                        ));
                    }
                    r.read_exact(tensor.as_bytes_mut())?;
                    // println!("name:{},nbytes:{}", name, tensor.as_bytes_mut().len());
                    // if name == "tok_embeddings.weight".to_string() {
                    //     let x: &[BlockQ4_0] = unsafe { tensor.as_slice::<BlockQ4_0>() };
                    //     let mut sum: f64 = 0.0;
                    //     for i in 0..tensor.elem_count() {
                    //         sum += x[i].d().abs() as f64;
                    //     }
                    //     println!(
                    //         "tok_embeddings,sum:{:?},sha
                    //         pe:{:?},stride:{:?}",
                    //         sum,
                    //         tensor.ggml_shape(),
                    //         tensor.dim().stride_4d()
                    //     );
                    //     //exit(1);
                    // }

                    total_size += tensor.nbytes();
                    // whisper_mode.n_loaded += 1;
                    match r.fill_buf() {
                        Ok(r) => {
                            if r.len() < 12 {
                                break;
                            }
                        }
                        Err(e) => match e.kind() {
                            std::io::ErrorKind::UnexpectedEof => break,
                            _ => return Err(LLMError::UnexpectIO(e)),
                        },
                    }
                } else {
                    println!("break");
                    return Err(LLMError::BadRefTensor(name));
                }
            }
        }
        println!("success");
        Ok(model)
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

fn llama_eval(model: &LlamaModel, embd_inp: &[GptVocabId], mem_per_token: usize) -> LLMResult<()> {
    let N = embd_inp.len();
    let hparams = &model.hparams;

    let n_embd = hparams.n_embd;
    let n_layer = hparams.n_layer;
    let n_ctx = hparams.n_ctx;
    let n_head = hparams.n_head;
    let n_vocab = hparams.n_vocab;
    let n_rot = hparams.n_embd / hparams.n_head;
    let d_key = n_embd / n_head;

    let buf_size = 512 * 1024 * 1024;
    let mut buf: Vec<u8> = vec![0u8; buf_size];
    let mut tensor_ctx = TensorContext::new(&mut buf);

    let mut embd = new_tensor_1d(&mut tensor_ctx, DType::I32, N)?;

    unsafe {
        embd.as_slice_mut::<i32>().copy_from_slice(embd_inp);
    }

    let x: &[i32] = unsafe { embd.as_slice::<i32>() };
    let mut sum: f64 = 0.0;
    for i in 0..embd.elem_count() {
        sum += x[i].abs() as f64;
    }
    println!(
        "embd,sum:{:?},sha
        pe:{:?},stride:{:?}",
        sum,
        embd.ggml_shape(),
        embd.dim().stride_4d()
    );

    let inp = get_rows(&mut tensor_ctx, &model.tok_embeddings, &embd)?;

    // let x: &[BlockQ4_0] = unsafe { model.tok_embeddings.as_slice::<BlockQ4_0>() };
    let mut sum: f64 = 0.0;
    // for i in 0..model.tok_embeddings.elem_count() {
    //     if x[i].d().is_nan() {
    //         sum += x[i].d().abs() as f64;
    //     }
    // }
    println!(
        "tok_embeddings,sum:{:?},shape:{:?},stride:{:?}",
        sum,
        model.tok_embeddings.ggml_shape(),
        model.tok_embeddings.dim().stride_4d()
    );

    let x: &[f32] = unsafe { inp.as_slice::<f32>() };
    let mut sum: f64 = 0.0;
    for i in 0..inp.elem_count() {
        sum += x[i].abs() as f64;
    }
    println!(
        "k1,sum:{:?},sha
        pe:{:?},stride:{:?}",
        sum,
        inp.ggml_shape(),
        inp.dim().stride_4d()
    );
    Ok(())
}

fn compute_ctx_size(hparams: &LlamaHparams) -> usize {
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
    let n_parts = *LLAMA_N_PARTS.get(&hparams.n_embd).unwrap();
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
    return ctx_size as usize;
}

fn main() -> LLMResult<()> {
    let model_path = "E:\\cproject\\models\\ggml-model-Q4.bin";
    let mut fin = open_file_stream(model_path)?;
    let magic = fin.read_u32::<Endian>()?;
    if magic != GGML_MAGIC {
        return Err(LLMError::BadMagic(model_path.to_string()));
    }
    // LLMContext::new(&model_path).unwrap();
    let hparams = LlamaHparams::load(&mut fin)?;
    let vocab = GptVocab::load(&mut fin, hparams.n_vocab)?;
    let mut prompt = String::from("Building a website can be done in 10 simple steps:");
    prompt.insert(0, ' ');
    let embd_inp = llama_tokenize(&vocab, &prompt, true);
    println!("{}: prompt: '{}'", function!(), prompt);
    println!(
        "{}: number of tokens in prompt = {}",
        function!(),
        embd_inp.len(),
    );
    for inp in embd_inp.iter() {
        println!("{} -> '{}'\n", inp, vocab.id_to_token.get(inp).unwrap());
    }
    let ctx_size = compute_ctx_size(&hparams);
    let mut buf_model = vec![0u8; ctx_size as usize];
    let model = LlamaModel::load(&mut fin, hparams, &mut buf_model)?;
    llama_eval(&model, &[0, 1, 2, 3], 0)?;
    drop(buf_model);
    drop(model);
    Ok(())
}
