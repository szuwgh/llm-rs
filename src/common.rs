use crate::meta::GGufMetadataValueType;
use galois::error::GError;
use std::io::Error as IOError;
use std::io::Read;
use std::str::Utf8Error;
use thiserror::Error;

pub type LLMResult<T> = Result<T, LLMError>;

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
    #[error(" unknown Utf8Error '{0}' ")]
    UnknownUtf8Error(Utf8Error),
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

impl From<Utf8Error> for LLMError {
    fn from(e: Utf8Error) -> Self {
        LLMError::UnknownUtf8Error(e)
    }
}

pub trait GGufRead {
    fn read_bytes(&mut self, n: usize) -> LLMResult<&[u8]>;

    fn read_len(&mut self) -> LLMResult<usize>;
    // 返回当前 offset
    fn offset(&self) -> usize;

    fn cursor(&self) -> &[u8];
}

pub(crate) trait BinarySerialize: Sized {
    fn deserialize<R: Read + GGufRead>(r: &mut R) -> LLMResult<Self>;
}
