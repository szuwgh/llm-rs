use crate::common::LLMResult;
use crate::unicode_data::UNICODE_MAP_UPPERCASE;
use crate::unicode_data::UNICODE_RANGES_NFD;
use crate::unicode_data::UNICODE_WHITESPACE_SET;
use crate::unicode_data::{UNICODE_MAP_LOWERCASE, UNICODE_RANGES_FLAGS};
use fancy_regex::Regex;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::OnceLock;

pub(crate) fn bytes_to_unicode_bpe(byte: u8) -> String {
    static MAP: OnceLock<HashMap<u8, String>> = OnceLock::new();
    let map = MAP.get_or_init(bytes_to_unicode_map_bpe);
    map.get(&byte).cloned().expect("Byte not found in map")
}

fn codepoint_to_utf8(cp: u32) -> String {
    let mut result = String::new();
    if cp <= 0x7f {
        result.push(cp as u8 as char);
    } else if cp <= 0x7ff {
        result.push(((0xc0 | ((cp >> 6) & 0x1f)) as u8) as char);
        result.push(((0x80 | (cp & 0x3f)) as u8) as char);
    } else if cp <= 0xffff {
        result.push(((0xe0 | ((cp >> 12) & 0x0f)) as u8) as char);
        result.push(((0x80 | ((cp >> 6) & 0x3f)) as u8) as char);
        result.push(((0x80 | (cp & 0x3f)) as u8) as char);
    } else if cp <= 0x10ffff {
        result.push(((0xf0 | ((cp >> 18) & 0x07)) as u8) as char);
        result.push(((0x80 | ((cp >> 12) & 0x3f)) as u8) as char);
        result.push(((0x80 | ((cp >> 6) & 0x3f)) as u8) as char);
        result.push(((0x80 | (cp & 0x3f)) as u8) as char);
    } else {
        panic!("invalid codepoint");
    }
    result
}

fn bytes_to_unicode_map_bpe() -> HashMap<u8, String> {
    let mut map = HashMap::new();
    for ch in ('!' as u8)..=('~' as u8) {
        map.insert(ch, codepoint_to_utf8(ch as u32));
    }
    for ch in ('¡' as u8)..=('¬' as u8) {
        map.insert(ch, codepoint_to_utf8(ch as u32));
    }
    for ch in ('®' as u8)..=('ÿ' as u8) {
        map.insert(ch, codepoint_to_utf8(ch as u32));
    }
    let mut n = 0;
    for ch in 0..=255 {
        if !map.contains_key(&(ch as u8)) {
            map.insert(ch as u8, codepoint_to_utf8(256 + n));
            n += 1;
        }
    }
    map
}

pub(crate) fn unicode_regex_split<'a>(text: &'a str, regex_expr: &[&str]) -> Vec<&'a str> {
    let mut result = vec![text];

    for expr in regex_expr {
        let re = Regex::new(expr).expect("Invalid regex");
        result = result
            .iter()
            .flat_map(|s| {
                re.find_iter(*s) // 获取匹配的 token
                    .filter_map(Result::ok) // 处理 Result
                    .map(|m| m.as_str())
            })
            .filter(|s| !s.is_empty())
            .collect();
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tunicode_regex_split() {
        let text = "I'm learning Rust! 42 days left.";
        let res = unicode_regex_split(text, &["(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"]);
        println!("{:?}", res);
    }

    #[test]
    fn text_re() {
        let text = "I'm learning Rust! 42 days left. Rust 是一门很棒的语言! 你学了 42 天了吗？";

        // 定义用于分词的正则表达式
        let pattern = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

        // 编译正则表达式
        let re = Regex::new(pattern).expect("Invalid regex");

        // 使用 split() 方法进行分词，并正确处理 Result
        let tokens: Vec<String> = re
            .find_iter(text) // 获取匹配的 token
            .filter_map(Result::ok) // 处理 Result
            .map(|m| m.as_str().to_string()) // 转换为 String
            .filter(|s| !s.trim().is_empty()) // 过滤空格和空字符串
            .collect();

        // 打印结果
        println!("{:?}", tokens);
    }
}

// fn unicode_tolower(cpt: u32) -> u32 {
//     use std::cmp::Ordering;
//     // 二分查找
//     if let Ok(index) = UNICODE_MAP_LOWERCASE.binary_search_by(|&(key, _)| {
//         if key < cpt {
//             Ordering::Less
//         } else if key > cpt {
//             Ordering::Greater
//         } else {
//             Ordering::Equal
//         }
//     }) {
//         UNICODE_MAP_LOWERCASE[index].1
//     } else {
//         cpt // 如果没有找到匹配项，返回原始字符
//     }
// }

//  const  MAX_CODEPOINTS :usize= 0x110000;

//  #[derive(Clone, Copy, Debug, Default)]
// struct UnicodeCptFlags {
//     // Codepoint type flags
//     is_undefined: bool,
//     is_number: bool,  // regex: \p{N}
//     is_letter: bool,  // regex: \p{L}
//     is_separator: bool,  // regex: \p{Z}
//     is_accent_mark: bool,  // regex: \p{M}
//     is_punctuation: bool,  // regex: \p{P}
//     is_symbol: bool,  // regex: \p{S}
//     is_control: bool,  // regex: \p{C}

//     // Helper flags
//     is_whitespace: bool,  // regex: \s
//     is_lowercase: bool,
//     is_uppercase: bool,
//     is_nfd: bool,
// }

// impl UnicodeCptFlags {
//     const UNDEFINED: u16 = 0x0001;
//     const NUMBER: u16 = 0x0002;  // regex: \p{N}
//     const LETTER: u16 = 0x0004;  // regex: \p{L}
//     const SEPARATOR: u16 = 0x0008;  // regex: \p{Z}
//     const ACCENT_MARK: u16 = 0x0010;  // regex: \p{M}
//     const PUNCTUATION: u16 = 0x0020;  // regex: \p{P}
//     const SYMBOL: u16 = 0x0040;  // regex: \p{S}
//     const CONTROL: u16 = 0x0080;  // regex: \p{C}
//     const MASK_CATEGORIES: u16 = 0x00FF;

//     fn from_u16(flags: u16) -> Self {
//         Self {
//             is_undefined: (flags & Self::UNDEFINED) != 0,
//             is_number: (flags & Self::NUMBER) != 0,
//             is_letter: (flags & Self::LETTER) != 0,
//             is_separator: (flags & Self::SEPARATOR) != 0,
//             is_accent_mark: (flags & Self::ACCENT_MARK) != 0,
//             is_punctuation: (flags & Self::PUNCTUATION) != 0,
//             is_symbol: (flags & Self::SYMBOL) != 0,
//             is_control: (flags & Self::CONTROL) != 0,
//             is_whitespace: false,
//             is_lowercase: false,
//             is_uppercase: false,
//             is_nfd: false,
//         }
//     }

//     fn as_u16(&self) -> u16 {
//         let mut flags = 0;
//         if self.is_undefined { flags |= Self::UNDEFINED; }
//         if self.is_number { flags |= Self::NUMBER; }
//         if self.is_letter { flags |= Self::LETTER; }
//         if self.is_separator { flags |= Self::SEPARATOR; }
//         if self.is_accent_mark { flags |= Self::ACCENT_MARK; }
//         if self.is_punctuation { flags |= Self::PUNCTUATION; }
//         if self.is_symbol { flags |= Self::SYMBOL; }
//         if self.is_control { flags |= Self::CONTROL; }
//         flags
//     }

//     fn category_flag(&self) -> u16 {
//         self.as_u16() & Self::MASK_CATEGORIES
//     }
// }

// fn unicode_cpt_flags_array() -> Vec<UnicodeCptFlags> {
//     let mut cpt_flags = vec![UnicodeCptFlags::default(); MAX_CODEPOINTS];

//     assert!(UNICODE_RANGES_FLAGS[0].0 == 0);
//     assert!(UNICODE_RANGES_FLAGS[UNICODE_RANGES_FLAGS.len() - 1].0 == MAX_CODEPOINTS as u32);

//     for i in 1..UNICODE_RANGES_FLAGS.len() {
//         let range_ini = UNICODE_RANGES_FLAGS[i - 1]; // (codepoint_ini, flags)
//         let range_end = UNICODE_RANGES_FLAGS[i];     // (codepoint_end, flags)
//         for cpt in range_ini.0..range_end.0 {
//             cpt_flags[cpt as usize] =UnicodeCptFlags::from_u16(range_ini.1) ;
//         }
//     }

//     for &cpt in UNICODE_WHITESPACE_SET.iter() {
//         cpt_flags[cpt as usize].is_whitespace = true;
//     }

//     for &(_, lower) in UNICODE_MAP_LOWERCASE.iter() {
//         cpt_flags[lower as usize].is_lowercase = true;
//     }

//     for &(_, upper) in UNICODE_MAP_UPPERCASE.iter() {
//         cpt_flags[upper as usize].is_uppercase = true;
//     }

//     for range in UNICODE_RANGES_NFD.iter() { // (start, last, nfd)
//         cpt_flags[range.2 as usize].is_nfd = true;
//     }

//     cpt_flags
// }

// static UNDEF: LazyLock<UnicodeCptFlags> = LazyLock::new(|| UnicodeCptFlags::from_u16(UnicodeCptFlags::UNDEFINED));
// static CPT_FLAGS: LazyLock<Vec<UnicodeCptFlags>> = LazyLock::new(unicode_cpt_flags_array);

// fn unicode_cpt_flags_from_cpt(cpt: u32) -> UnicodeCptFlags {
//     CPT_FLAGS.get(cpt as usize).copied().unwrap_or(*UNDEF)
// }

// fn unicode_regex_split_custom_gpt2(text: &str, offsets: &[usize]) -> Vec<usize> {
//     let mut bpe_offsets = Vec::with_capacity(offsets.len()); // 预分配空间
//     let cpts: Vec<u32> = unicode_cpts_from_utf8(text); // 假设存在该函数

//     let mut start = 0;
//     for &offset in offsets {
//         let offset_ini = start;
//         let offset_end = start + offset;
//         assert!(offset_end <= cpts.len());
//         start = offset_end;

//         const OUT_OF_RANGE: u32 = 0xFFFFFFFF;
//         let get_cpt = |pos: usize| -> u32 {
//             if (offset_ini..offset_end).contains(&pos) {
//                 cpts[pos]
//             } else {
//                 OUT_OF_RANGE
//             }
//         };

//         let get_flags = |pos: usize| -> UnicodeCptFlags {
//             if (offset_ini..offset_end).contains(&pos) {
//                 unicode_cpt_flags_from_cpt(cpts[pos])
//             } else {
//                 UnicodeCptFlags::default()
//             }
//         };

//         let mut prev_end = offset_ini;
//         let mut add_token = |end: usize| -> usize {
//             assert!(prev_end <= end && end <= offset_end);
//             let len = end - prev_end;
//             if len > 0 {
//                 bpe_offsets.push(len);
//             }
//             prev_end = end;
//             len
//         };

//         let mut pos = offset_ini;
//         while pos < offset_end {
//             let cpt = get_cpt(pos);
//             let flags = get_flags(pos);

//             // 处理 's | 't | 're | 've | 'm | 'll | 'd
//             if cpt == '\'' as u32 && pos + 1 < offset_end {
//                 let cpt_next = get_cpt(pos + 1);
//                 if matches!(cpt_next as u8 as char, 's' | 't' | 'm' | 'd') {
//                     pos += add_token(pos + 2);
//                     continue;
//                 }
//                 if pos + 2 < offset_end {
//                     let cpt_next_next = get_cpt(pos + 2);
//                     if (cpt_next as u8 as char == 'r' && cpt_next_next as u8 as char == 'e')
//                         || (cpt_next as u8 as char == 'v' && cpt_next_next as u8 as char == 'e')
//                         || (cpt_next as u8 as char == 'l' && cpt_next_next as u8 as char == 'l')
//                     {
//                         pos += add_token(pos + 3);
//                         continue;
//                     }
//                 }
//             }

//             let mut flags2 = if cpt == ' ' as u32 {
//                 get_flags(pos + 1)
//             } else {
//                 flags
//             };

//             // 匹配 <space>?\p{L}+
//             if flags2.is_letter {
//                 pos += (cpt == ' ' as u32) as usize;
//                 while flags2.is_letter {
//                     flags2 = get_flags(pos + 1);
//                     pos += 1;
//                 }
//                 add_token(pos);
//                 continue;
//             }

//             // 匹配 <space>?\p{N}+
//             if flags2.is_number {
//                 pos += (cpt == ' ' as u32) as usize;
//                 while flags2.is_number {
//                     flags2 = get_flags(pos + 1);
//                     pos += 1;
//                 }
//                 add_token(pos);
//                 continue;
//             }

//             // 匹配 <space>?[^\s\p{L}\p{N}]+
//             if !(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_u16() > 0 {
//                 pos += (cpt == ' ' as u32) as usize;
//                 while !(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_u16() > 0 {
//                     flags2 = get_flags(pos + 1);
//                     pos += 1;
//                 }
//                 add_token(pos);
//                 continue;
//             }

//             // 处理多个连续空格 \s+(?!\S)
//             let mut num_whitespaces = 0;
//             while get_flags(pos + num_whitespaces).is_whitespace {
//                 num_whitespaces += 1;
//             }
//             if num_whitespaces > 1 && get_cpt(pos + num_whitespaces) != OUT_OF_RANGE {
//                 pos += num_whitespaces - 1;
//                 add_token(pos);
//                 continue;
//             }

//             // 处理单个空格 \s+
//             if num_whitespaces > 0 {
//                 pos += num_whitespaces;
//                 add_token(pos);
//                 continue;
//             }

//             // 没有匹配的情况
//             add_token(pos + 1);
//         }
//     }

//     bpe_offsets
// }

// fn unicode_regex_split_custom(text: &str, regex_expr: &str, offsets: &[usize]) -> Vec<usize> {
//     let mut bpe_offsets = Vec::new();

//     if regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)" {
//         bpe_offsets = unicode_regex_split_custom_gpt2(text, offsets);
//     } else if regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
//         || regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" {

//         bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
//     }

//     bpe_offsets
// }

// fn unicode_cpt_from_utf8(utf8: &str, offset: &mut usize) -> LLMResult<u32> {
//     let bytes = utf8.as_bytes();
//     if *offset >= bytes.len() {
//         return Err("offset out of bounds".into());
//     }

//     let first = bytes[*offset];

//     if first & 0x80 == 0 {
//         *offset += 1;
//         return Ok(first as u32);
//     }
//     if first & 0x40 == 0 {
//         return Err("invalid character".into());
//     }
//     if first & 0x20 == 0 {
//         if *offset + 1 >= bytes.len() || bytes[*offset + 1] & 0xc0 != 0x80 {
//             return Err("invalid character".into());
//         }
//         let result = ((first & 0x1f) as u32) << 6 | (bytes[*offset + 1] & 0x3f) as u32;
//         *offset += 2;
//         return Ok(result);
//     }
//     if first & 0x10 == 0 {
//         if *offset + 2 >= bytes.len()
//             || bytes[*offset + 1] & 0xc0 != 0x80
//             || bytes[*offset + 2] & 0xc0 != 0x80
//         {
//             return Err("invalid character".into());
//         }
//         let result = ((first & 0x0f) as u32) << 12
//             | ((bytes[*offset + 1] & 0x3f) as u32) << 6
//             | (bytes[*offset + 2] & 0x3f) as u32;
//         *offset += 3;
//         return Ok(result);
//     }
//     if first & 0x08 == 0 {
//         if *offset + 3 >= bytes.len()
//             || bytes[*offset + 1] & 0xc0 != 0x80
//             || bytes[*offset + 2] & 0xc0 != 0x80
//             || bytes[*offset + 3] & 0xc0 != 0x80
//         {
//             return Err("invalid character".into());
//         }
//         let result = ((first & 0x07) as u32) << 18
//             | ((bytes[*offset + 1] & 0x3f) as u32) << 12
//             | ((bytes[*offset + 2] & 0x3f) as u32) << 6
//             | (bytes[*offset + 3] & 0x3f) as u32;
//         *offset += 4;
//         return Ok(result);
//     }
//     Err("failed to convert utf8 to codepoint".into())
// }

// fn unicode_cpts_from_utf8(utf8: &str) -> Vec<u32> {
//     let mut result = Vec::with_capacity(utf8.len());
//     let mut offset = 0;

//     while offset < utf8.len() {
//         match unicode_cpt_from_utf8(utf8, &mut offset) {
//             Ok(codepoint) => result.push(codepoint),
//             Err(_) => {
//                 // Silently ignore invalid UTF-8 input to avoid leaking the exception beyond llama_tokenize
//                 offset += 1;
//                 result.push(0xFFFD); // replacement character
//             }
//         }
//     }
//     result
// }

// fn unicode_regex_split_custom_llama3(text: &str, offsets: &[usize]) -> Vec<usize> {
//     let mut bpe_offsets = Vec::with_capacity(offsets.len());
//     let cpts: Vec<u32> = unicode_cpts_from_utf8(text);
//     let mut start = 0;

//     for &offset in offsets {
//         let offset_ini = start;
//         let offset_end = start + offset;
//         assert!(offset_end <= cpts.len());
//         start = offset_end;

//         const OUT_OF_RANGE: u32 = 0xFFFFFFFF;

//         let _get_cpt = |pos: usize| -> u32 {
//             if offset_ini <= pos && pos < offset_end {
//                 cpts[pos]
//             } else {
//                 OUT_OF_RANGE
//             }
//         };

//         let _get_flags = |pos: usize| -> UnicodeCptFlags {
//             if offset_ini <= pos && pos < offset_end {
//                 unicode_cpt_flags_from_cpt(cpts[pos])
//             } else {
//                 UnicodeCptFlags::default()
//             }
//         };

//         let mut _prev_end = offset_ini;
//         let mut _add_token = |end: usize| -> usize {
//             assert!(_prev_end <= end && end <= offset_end);
//             let len = end - _prev_end;
//             if len > 0 {
//                 bpe_offsets.push(len);
//             }
//             _prev_end = end;
//             len
//         };

//         let mut pos = offset_ini;
//         while pos < offset_end {
//             let cpt = char::from_u32( _get_cpt(pos) ).unwrap();
//             let flags = _get_flags(pos);

//             if cpt == '\''  && pos + 1 < offset_end {
//                 let cpt_next = unicode_tolower(_get_cpt(pos + 1));
//                 if [b's', b't', b'm', b'd'].contains(&(cpt_next as u8)) {
//                     pos += _add_token(pos + 2);
//                     continue;
//                 }
//                 if pos + 2 < offset_end {
//                     let cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
//                     if [(b'r', b'e'), (b'v', b'e'), (b'l', b'l')]
//                         .contains(&(cpt_next as u8, cpt_next_next as u8))
//                     {
//                         pos += _add_token(pos + 3);
//                         continue;
//                     }
//                 }
//             }

//             if !(cpt == '\r' || cpt == '\n' || flags.is_number) {
//                 if flags.is_letter || _get_flags(pos + 1).is_letter {
//                     pos += 1;
//                     while _get_flags(pos).is_letter {
//                         pos += 1;
//                     }
//                     _add_token(pos);
//                     continue;
//                 }
//             }

//             if flags.is_number {
//                 let mut ini = pos;
//                 while _get_flags(pos).is_number {
//                     if pos - ini >= 3 {
//                         _add_token(pos);
//                         ini = pos;
//                     }
//                     pos += 1;
//                 }
//                 _add_token(pos);
//                 continue;
//             }

//             let mut num_whitespaces = 0;
//             let mut last_end_r_or_n = 0;
//             while _get_flags(pos + num_whitespaces).is_whitespace {
//                 let cpt2 =char::from_u32(  _get_cpt(pos + num_whitespaces)).unwrap();
//                 if cpt2 == '\r' || cpt2 == '\n' {
//                     last_end_r_or_n = pos + num_whitespaces + 1;
//                 }
//                 num_whitespaces += 1;
//             }

//             if last_end_r_or_n > 0 {
//                 pos = last_end_r_or_n;
//                 _add_token(pos);
//                 continue;
//             }

//             if num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE {
//                 pos += num_whitespaces - 1;
//                 _add_token(pos);
//                 continue;
//             }

//             if num_whitespaces > 0 {
//                 pos += num_whitespaces;
//                 _add_token(pos);
//                 continue;
//             }

//             _add_token(pos + 1);
//         }
//     }

//     bpe_offsets
// }
