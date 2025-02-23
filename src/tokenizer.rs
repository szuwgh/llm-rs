use crate::common::LLMError;
use crate::meta::LLMVocabType;
use crate::meta::LlamaToken;
use crate::meta::LlamaVocabPreType;
use crate::unicode;
use crate::unicode::unicode_regex_split;
use crate::LLMResult;
use crate::LLamaVocab;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::env::var;
use std::io::Cursor;
use std::io::Write;
use std::marker::PhantomData;

pub(crate) fn tokenize(
    vocab: &LLamaVocab,
    mut raw_text: String,
    add_bos: bool,
) -> LLMResult<Vec<LlamaToken>> {
    // 在 llama.cpp 中，add_bos 代表 “add beginning-of-sequence”，
    // 即 是否在输入文本的开头添加序列起始标记（BOS，Beginning Of Sequence）。

    // 具体作用：
    // 在一些 LLM（如 LLaMA 系列模型）中，文本通常需要特定的标记来
    // 正确解析输入。例如：

    // BOS（<s>，序列起始标记），用于标明句子的开始。
    // EOS（</s>，序列结束标记），用于标明句子的结束。
    // 当 add_bos = true 时，llama.cpp 会在输入文本的 token
    // 序列前自动加上 BOS token，使模型能更准确地理解它是一个新句子或新段落。
    let mut output = Vec::new();
    if add_bos && vocab.special_bos_id != -1 {
        output.push(vocab.special_bos_id);
    }

    match vocab.get_vocab_type() {
        LLMVocabType::SPM => {
            let mut llm_tokenizer_spm = LLMtokenizerSpm::new(vocab);
            raw_text.insert(0, ' ');
            replace_all(&mut raw_text, " ", "▁");
            llm_tokenizer_spm.tokenize(&raw_text, &mut output)?;
        }
        LLMVocabType::BPE => {
            todo!()
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct LLMSymbol<'a> {
    prev: i32,
    next: i32,
    text: &'a str,
    n: usize,
}

pub(crate) fn replace_all(s: &mut String, search: &str, replace: &str) {
    *s = s.replace(search, replace);
}

#[derive(PartialEq)]
struct LlmBigramSpm {
    left: i32,
    right: i32,
    score: f32,
    size: usize,
}

// 实现 Ord 以自定义排序逻辑（最小堆）
impl Eq for LlmBigramSpm {}

impl PartialOrd for LlmBigramSpm {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LlmBigramSpm {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.score > other.score {
            Ordering::Greater // 让 BinaryHeap 变成 **最大堆**
        } else if self.score < other.score {
            Ordering::Less
        } else {
            self.left.cmp(&other.left) // 左索引小的优先
        }
    }
}
pub struct LLMtokenizerSpm<'a> {
    vocab: &'a LLamaVocab,
    symbols: Vec<LLMSymbol<'a>>,
    work_queue: BinaryHeap<LlmBigramSpm>,
    rev_merge: HashMap<String, (i32, i32)>,
}

impl<'a> LLMtokenizerSpm<'a> {
    pub(crate) fn new(vocab: &'a LLamaVocab) -> LLMtokenizerSpm<'a> {
        LLMtokenizerSpm {
            vocab: vocab,
            symbols: Vec::new(),
            work_queue: BinaryHeap::new(),
            rev_merge: HashMap::new(),
        }
    }

    pub(crate) fn tokenize(
        &'a mut self,
        text: &'a str,
        output: &mut Vec<LlamaToken>,
    ) -> LLMResult<()> {
        let mut index: i32 = 0;
        let mut offs = 0;
        for (i, ch) in text.char_indices() {
            let n = ch.len_utf8();
            offs += n;
            let prev = index - 1;
            let next = if offs == text.len() { -1 } else { index + 1 };
            index += 1;
            self.symbols.push(LLMSymbol {
                prev: prev,
                next: next,
                text: &text[i..],
                n: n,
            });
        }
        for i in self.symbols.iter() {
            println!(
                "Prev: {}, Next: {}, Text: {}, Length: {}",
                i.prev, i.next, i.text, i.n
            )
        }
        // todo!();

        for i in 1..self.symbols.len() as i32 {
            self.try_add_bigram(i - 1, i);
        }

        while let Some(bigram) = self.work_queue.pop() {
            println!(
                "pop bigram:{},{},{},{:?}",
                bigram.left, bigram.right, bigram.score, bigram.size
            );

            let (left_sym_prev, left_sym_next, right_sym_next) = {
                // if bigram.left as usize >= self.symbols.len()
                //     || bigram.right as usize >= self.symbols.len()
                // {
                //     return Ok(());
                // }

                let (left_part, right_part) = self.symbols.split_at_mut(bigram.right as usize);

                // let left_sym = &mut self.symbols[bigram.left as usize];
                // let right_sym = &mut self.symbols[bigram.right as usize];

                let left_sym = &mut left_part[bigram.left as usize];
                let right_sym = &mut right_part[0]; // 因为 right 是 right_part 的第一个元素

                if left_sym.n == 0 || right_sym.n == 0 || (left_sym.n + right_sym.n != bigram.size)
                {
                    continue;
                }

                // 合并符号
                left_sym.n += right_sym.n;
                right_sym.n = 0;
                println!("left = {:?}' size = {}\n", &left_sym.text, bigram.size);
                left_sym.next = right_sym.next;

                (left_sym.prev, left_sym.next, right_sym.next)
            };

            if right_sym_next >= 0 {
                self.symbols[right_sym_next as usize].prev = bigram.left;
            }

            // 查找更多替换
            self.try_add_bigram(left_sym_prev, bigram.left);
            self.try_add_bigram(bigram.left, left_sym_next);
        }

        let mut i = 0i32;
        while i != -1 {
            self.resegment(&self.symbols[i as usize], output)?;
            i = self.symbols[i as usize].next;
        }
        return Ok(());
    }

    fn resegment(&self, symbol: &LLMSymbol, output: &mut Vec<LlamaToken>) -> LLMResult<()> {
        let text = &symbol.text[..symbol.n];
        if let Some(token) = self.vocab.token_to_id.get(text) {
            println!("text:{},token1:{}", text, token);
            output.push(*token);
        } else {
            if let Some(p) = self.rev_merge.get(text) {
                self.resegment(&self.symbols[p.0 as usize], output)?;
                self.resegment(&self.symbols[p.1 as usize], output)?;
            } else {
                for j in 0..symbol.n {
                    let token_id = llama_byte_to_token(self.vocab, symbol.text.as_bytes()[j])?;
                    println!("token2:{}", token_id);
                    output.push(token_id);
                }
            }
        }
        Ok(())
    }

    fn try_add_bigram(&mut self, left: i32, right: i32) {
        if left == -1 || right == -1 {
            return;
        }

        let l = self.symbols[left as usize].n + self.symbols[right as usize].n;
        // 拼接左、右符号的文本
        let text = &self.symbols[left as usize].text[..l];
        let token_id = self.vocab.token_to_id.get(text);
        if token_id.is_none() {
            return;
        }
        let token_id = token_id.unwrap();

        if let Some(tok_data) = self.vocab.id_to_token.get(*token_id as usize) {
            let bigram = LlmBigramSpm {
                left: left,
                right: right,
                score: tok_data.score,
                size: text.len(),
            };
            println!("bigram:{},{},{},{:?}", left, right, tok_data.score, text);

            self.work_queue.push(bigram);
            self.rev_merge.insert(text.to_string(), (left, right));
        }
    }
}

fn llama_byte_to_token(vocab: &LLamaVocab, ch: u8) -> LLMResult<LlamaToken> {
    match vocab.get_vocab_type() {
        LLMVocabType::SPM => {
            let mut buf = [0u8; 7]; // 固定大小的字节数组，存储 "<0xXX>" + '\0'
                                    // 通过 `write!` 格式化字符串到 buf
            let mut cursor = Cursor::new(&mut buf[..]); // 创建 Cursor 以安全地写入 `buf`
            write!(&mut cursor, "<0x{:02X}>", ch)?;
            let pos = cursor.position() as usize;
            let id = vocab
                .token_to_id
                .get(std::str::from_utf8(&buf[..pos])?)
                .or_else(|| Some(&0))
                .unwrap();
            Ok(*id)
        }
        LLMVocabType::BPE => {
            let key = unicode::bytes_to_unicode_bpe(ch);
            let id = *vocab.token_to_id.get(&key).or_else(|| Some(&0)).unwrap();
            Ok(id)
        }
    }
}

struct LLMtokenizerBpe<'a> {
    vocab: &'a LLamaVocab,
    symbols: Vec<LLMSymbol<'a>>,
    symbols_final: Vec<LLMSymbol<'a>>,
    regex_exprs: Vec<&'static str>,
}

impl<'a> LLMtokenizerBpe<'a> {
    pub(crate) fn new(vocab: &'a LLamaVocab) -> Self {
        //   assert!(vocab.get_vocab_type() == LLAMA_VOCAB_TYPE_BPE);
        let regex_exprs = match vocab.get_pre_type() {
            LlamaVocabPreType::LLAMA_VOCAB_PRE_TYPE_LLAMA3 => {
                [
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

                    // adapted: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2080233989
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    ].to_vec()
            }
            _ => {
                todo!()
            }
        };
        LLMtokenizerBpe {
            vocab: vocab,
            symbols: Vec::new(),
            symbols_final: Vec::new(),
            regex_exprs: regex_exprs,
        }
    }

    pub(crate) fn tokenize(
        &'a mut self,
        text: &'a str,
        output: &mut Vec<LlamaToken>,
    ) -> LLMResult<()> {
        let final_prev_index = -1;
        let word_collection = unicode_regex_split(text, &self.regex_exprs);

        self.symbols_final.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{replace_all, LLMtokenizerSpm};
    use crate::LLamaVocab;
    #[test]
    fn test_tokenize() {
        let mut v = Vec::new();
        let vocab = LLamaVocab::default();
        let mut t = LLMtokenizerSpm::new(&vocab);
        let mut s = "who am i".to_string();
        s.insert(0, ' ');
        replace_all(&mut s, " ", "▁");
        t.tokenize(&s, &mut v);
    }
}
