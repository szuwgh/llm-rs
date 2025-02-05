use std::collections::HashMap;
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
