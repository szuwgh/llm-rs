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
use std::fs::File;

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
    fn load<T: Read + BufRead>(r: &mut T, mtype: EModel) -> LLMResult<LlamaModel> {
        let hparams = LlamaHparams::load(r)?;
        let wtype = match hparams.f16 {
            0 => DType::F32,
            1 => DType::F16,
            2 => DType::Q4_0,
            _ => {
                todo!()
            }
        };
        let wtype2 = DType::F32;
        let mut ctx_mem_size = 0usize;
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
    let model_path = "/opt/cproject/llama.cpp-master-d3f202d/models/ggml-model-Q4.bin";
    LLMContext::new(&model_path).unwrap();
}
