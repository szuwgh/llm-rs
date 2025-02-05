use crate::common::*;
use crate::BinarySerialize;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::io::Read;
use std::ptr::NonNull;

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGML_MAGIC: u32 = 0x67676d6c; // "GGML"

type Endian = LittleEndian;

pub(crate) const LLM_KV_GENERAL_ARCHITECTURE: &'static str = "general.architecture";
pub(crate) const LLM_KV_TOKENIZER_LIST: &'static str = "tokenizer.ggml.tokens";
pub(crate) const LLM_KV_CONTEXT_LENGTH: &'static str = ".context_length";
pub(crate) const LLM_KV_EMBEDDING_LENGTH: &'static str = ".embedding_length";
pub(crate) const LLM_KV_FEED_FORWARD_LENGTH: &'static str = ".feed_forward_length";
pub(crate) const LLM_KV_ATTENTION_HEAD_COUNT: &'static str = ".attention.head_count";
pub(crate) const LLM_KV_BLOCK_COUNT: &'static str = ".block_count";
pub(crate) const LLM_KV_ATTENTION_HEAD_COUNT_KV: &'static str = ".attention.head_count_kv";
pub(crate) const LLM_KV_ROPE_FREQ_BASE: &'static str = ".rope.freq_base";
pub(crate) const LLM_KV_ROPE_SCALE_LINEAR: &'static str = ".rope.scale_linear";
pub(crate) const LLM_KV_ROPE_DIMENSION_COUNT: &'static str = ".rope.dimension_count";

pub(crate) const LLM_KV_TOKENIZER_MODEL: &'static str = "tokenizer.ggml.model";
pub(crate) const LLM_KV_TOKENIZER_TOKEN_TYPE: &'static str = "tokenizer.ggml.token_type";
pub(crate) const LLM_KV_TOKENIZER_SCORES: &'static str = "tokenizer.ggml.scores";
pub(crate) const LLM_KV_TOKENIZER_MERGES: &'static str = "tokenizer.ggml.merges";
pub(crate) const LLM_KV_TOKENIZER_BOS_ID: &'static str = "tokenizer.ggml.bos_token_id";
pub(crate) const LLM_KV_TOKENIZER_EOS_ID: &'static str = "tokenizer.ggml.eos_token_id";
pub(crate) const LLM_KV_TOKENIZER_UNK_ID: &'static str = "tokenizer.ggml.unknown_token_id";
pub(crate) const LLM_KV_TOKENIZER_SEP_ID: &'static str = "tokenizer.ggml.seperator_token_id";
pub(crate) const LLM_KV_TOKENIZER_PAD_ID: &'static str = "tokenizer.ggml.padding_token_id";
pub(crate) const LLM_KV_TOKENIZER_HF_JSON: &'static str = "tokenizer.huggingface.json";
pub(crate) const LLM_KV_TOKENIZER_RWKV: &'static str = "tokenizer.rwkv.world";

//pub(crate) type LLamaToken = u32;

pub(crate) type LlamaToken = i32;
pub(crate) type LlamaPos = i32;
pub(crate) type LlamaSeqId = i32;
pub(crate) type ID = i32;

enum TokenType {
    LLAMA_TOKEN_TYPE_UNDEFINED = 0,
    LLAMA_TOKEN_TYPE_NORMAL = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN = 2,
    LLAMA_TOKEN_TYPE_CONTROL = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED = 5,
    LLAMA_TOKEN_TYPE_BYTE = 6,
}

impl TryFrom<i32> for TokenType {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(TokenType::LLAMA_TOKEN_TYPE_UNDEFINED),
            1 => Ok(TokenType::LLAMA_TOKEN_TYPE_NORMAL),
            2 => Ok(TokenType::LLAMA_TOKEN_TYPE_UNKNOWN),
            3 => Ok(TokenType::LLAMA_TOKEN_TYPE_CONTROL),
            4 => Ok(TokenType::LLAMA_TOKEN_TYPE_USER_DEFINED),
            5 => Ok(TokenType::LLAMA_TOKEN_TYPE_UNUSED),
            6 => Ok(TokenType::LLAMA_TOKEN_TYPE_BYTE),
            _ => Err("Invalid TokenType value"),
        }
    }
}

type Token = String;

pub(crate) struct TokenData {
    pub(crate) text: Token,
    pub(crate) score: f32,
    pub(crate) ty: TokenType,
}

#[derive(Default)]
pub(crate) enum LLMVocabType {
    #[default]
    SPM = 0, // SentencePiece
    BPE = 1, // Byte Pair Encoding
}

#[derive(Default)]
pub(crate) struct LLamaVocab {
    pub(crate) id_to_token: Vec<TokenData>,
    pub(crate) token_to_id: HashMap<Token, LlamaToken>,
    //token_scores: HashMap<LlamaToken, f32>,
    vocab_type: LLMVocabType,
    pub(crate) special_bos_id: ID,
    special_eos_id: ID,
    special_unk_id: ID,
    special_sep_id: ID,
    special_pad_id: ID,
}

impl LLamaVocab {
    pub(crate) fn load(ctx: &GGufContext) -> LLMResult<LLamaVocab> {
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
                let tok_types = ctx
                    .metas_data()
                    .get(LLM_KV_TOKENIZER_TOKEN_TYPE)
                    .unwrap()
                    .get_i32_arr()
                    .unwrap();
                let token_ids = vocab
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (v.clone(), i as i32))
                    .collect::<HashMap<_, _>>();
                // let token_scores = vocab_scores
                //     .iter()
                //     .enumerate()
                //     .map(|(i, v)| (i as i32, v))
                //     .collect::<HashMap<_, _>>();
                let n_vocab = ctx
                    .metas_data()
                    .get(LLM_KV_TOKENIZER_LIST)
                    .unwrap()
                    .get_arr_n()
                    .unwrap();
                let mut id_to_token = Vec::new();
                for i in 0..n_vocab {
                    id_to_token.push(TokenData {
                        text: vocab[i].clone(),
                        score: vocab_scores[i],
                        ty: TokenType::try_from(tok_types[i])?,
                    })
                }
                Ok(Self {
                    id_to_token: id_to_token,
                    token_to_id: token_ids,
                    vocab_type: LLMVocabType::SPM,
                    special_bos_id: 1,
                    special_eos_id: 2,
                    special_unk_id: 0,
                    special_sep_id: -1,
                    special_pad_id: -1,
                })
            }
            _ => {
                return Err(LLMError::UnknownModelArchitecture(
                    tokenizer_kind.to_string(),
                ))
            }
        }
    }

    pub(crate) fn get_vocab_type(&self) -> &LLMVocabType {
        &self.vocab_type
    }
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

pub(crate) struct LlamaHparams {
    pub(crate) architecture: ModelArchitecture,
    pub(crate) model_name: String,
    pub(crate) vocab_only: bool,
    pub(crate) n_vocab: u32,
    pub(crate) n_ctx_train: u32, // context size the model was trained on
    pub(crate) n_head: u32,
    pub(crate) n_embd: u32,
    pub(crate) n_head_kv: u32,
    pub(crate) n_layer: u32,
    pub(crate) n_rot: u32,
    pub(crate) n_ff: u32,

    pub(crate) f_norm_eps: f32,
    pub(crate) f_norm_rms_eps: f32,

    pub(crate) rope_freq_base_train: f32,
    pub(crate) rope_freq_scale_train: f32,
}

impl LlamaHparams {
    pub(crate) fn load(r: &GGufContext) -> LLMResult<LlamaHparams> {
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

    pub(crate) fn n_gqa(&self) -> u32 {
        return self.n_head / self.n_head_kv;
    }

    pub(crate) fn n_embd_gqa(&self) -> u32 {
        self.n_embd / self.n_gqa()
    }

    pub(crate) fn n_embd_head(&self) -> u32 {
        self.n_embd / self.n_head
    }
}

#[derive(Clone)]
pub(crate) struct LlamaKvCell {
    pub(crate) pos: LlamaPos,
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
    pub(crate) fn new() -> Self {
        LlamaKvCell {
            pos: -1,
            delta: 0,
            seq_id: HashSet::new(),
        }
    }

    pub(crate) fn has_seq_id(&self, id: &LlamaSeqId) -> bool {
        self.seq_id.contains(id)
    }
}

pub(crate) struct LlamaBatch {
    pub(crate) token: Vec<LlamaToken>,
    pub(crate) pos: Vec<LlamaPos>,
    pub(crate) seq_id: Vec<LlamaSeqId>,
}

impl LlamaBatch {
    pub(crate) fn n_token(&self) -> usize {
        self.token.len()
    }

    pub(crate) fn embd_inp(&self) -> &[LlamaToken] {
        &self.token
    }
}

pub(crate) struct LlamaKvCache {
    pub(crate) head: usize,
    pub(crate) size: usize,
    pub(crate) n: usize,
    pub(crate) cells: Vec<LlamaKvCell>,
}

impl LlamaKvCache {
    pub(crate) fn new(n_ctx: usize) -> LlamaKvCache {
        Self {
            head: 0,
            size: n_ctx,
            n: 0,
            cells: vec![LlamaKvCell::default(); n_ctx],
        }
    }

    pub(crate) fn llama_kv_cache_find_slot(&mut self, batch: &LlamaBatch) -> bool {
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

#[derive(Hash, Eq, PartialEq, Debug, Default)]
pub(crate) enum EModel {
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

#[derive(Debug, Clone)]
pub(crate) enum GGufVersion {
    V1,
    V2,
    V3,
}

impl GGufVersion {
    pub(crate) fn decode(version: u32) -> LLMResult<GGufVersion> {
        match version {
            1 => Ok(GGufVersion::V1),
            2 => Ok(GGufVersion::V2),
            3 => Ok(GGufVersion::V3),
            _ => Err(LLMError::UnknownVersion(version)),
        }
    }
}

#[derive(Clone)]
pub(crate) struct GGufStr {
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

pub(crate) enum GGufTypeValue {
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
pub(crate) enum GGufMetadataValue {
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

    fn get_i32_arr(&self) -> Option<&[i32]> {
        match self {
            GGufMetadataValue::Array(e) => match e {
                GGufArr::I32Array(arr) => Some(arr.as_slice()),
                _ => None,
            },
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

pub(crate) struct GGufContext {
    pub(crate) header: GGufHeader,
    pub(crate) metas: HashMap<GGufStr, GGufMetadataValue>,
}

impl GGufContext {
    fn metas_data(&self) -> &HashMap<GGufStr, GGufMetadataValue> {
        &self.metas
    }
}

pub(crate) struct GGufHeader {
    magic: u32,
    version: GGufVersion,
    n_tensors: u64,
    n_kv: u64,
}

pub(crate) struct GGufKV {
    key: GGufStr,
    value: GGufMetadataValue,
}

impl GGufKV {
    fn new(key: GGufStr, value: GGufMetadataValue) -> GGufKV {
        Self { key, value }
    }
}

impl GGufHeader {
    pub(crate) fn load<T: Read>(r: &mut T) -> LLMResult<GGufHeader> {
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

    pub(crate) fn version(&self) -> &GGufVersion {
        &self.version
    }

    pub(crate) fn magic(&self) -> u32 {
        self.magic
    }

    pub(crate) fn n_tensors(&self) -> usize {
        self.n_tensors as usize
    }

    pub(crate) fn n_kv(&self) -> usize {
        self.n_kv as usize
    }
}
