use galois::error::GError;
use std::io::Error as IOError;
use thiserror::Error;

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

impl From<GError> for LLMError {
    fn from(e: GError) -> Self {
        LLMError::WrongGTensor(e)
    }
}

pub type LLMResult<T> = Result<T, LLMError>;

struct GGUFFileLoader {}

struct LlamaModel {}

struct LLMContext {
    t_load_us: i64,
    t_mel_us: i64,
    t_sample_us: i64,
    t_encode_us: i64,
    t_decode_us: i64,
    t_start_us: i64,
}

fn main() {
    println!("Hello, LLM-RS!");
}
