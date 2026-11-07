//! Qwen's byte-level BPE tokenizer from the checkpoint's tokenizer.json: NFC,
//! the Split regex (ICU), the GPT-2 byte alphabet and ranked merges; the added
//! tokens by their ids. Ported from the user's mlx-lm project
//! (src/tokenizer/mod.rs, MIT). Only that pipeline is accepted.
//!
//! Text from messages is encoded without the added vocabulary
//! ([`Tokenizer::encode`]): a message that spells `<|im_start|>` stays text.
//! The template's markers are added by their ids ([`Tokenizer::special`]).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fs;
use std::path::Path;

use serde_json::Value;

use super::icu;

pub struct Tokenizer {
    split_pattern: String,
    normalize_nfc: bool,
    byte_ids: [u32; 256],
    bytes: Vec<Option<Vec<u8>>>,
    merges: HashMap<(u32, u32), (usize, u32)>,
    added: HashMap<String, u32>,
    special: Vec<bool>,
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_slice(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// GPT-2's alphabet: printable bytes stand for themselves, the others for
/// the code points from 256 on.
fn byte_alphabet() -> [char; 256] {
    let mut chars = ['\0'; 256];
    let mut extra = 256;
    for (byte, c) in chars.iter_mut().enumerate() {
        *c = if (33..=126).contains(&byte) || (161..=172).contains(&byte) || (174..=255).contains(&byte) {
            char::from_u32(byte as u32).expect("a byte is a scalar")
        } else {
            let c = char::from_u32(extra).expect("the alphabet's extension is a scalar");
            extra += 1;
            c
        };
    }
    chars
}

/// The Split regex of a pipeline this tokenizer implements, and whether it
/// normalizes to NFC.
fn pipeline(json: &Value) -> Result<(String, bool), String> {
    let fail = |what: &str| Err(format!("tokenizer.json: unsupported {what}"));
    if json["version"] != "1.0" {
        return fail("version");
    }
    if !(json["normalizer"].is_null() || json["normalizer"] == serde_json::json!({"type": "NFC"})) {
        return fail("normalizer (only NFC or none)");
    }
    if !(json["truncation"].is_null() && json["padding"].is_null()) {
        return fail("truncation or padding");
    }
    let byte_level = serde_json::json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false});
    let Some(pattern) = json["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"].as_str() else {
        return fail("pre-tokenizer (a Split regex, then ByteLevel)");
    };
    let split = serde_json::json!({"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false});
    if json["pre_tokenizer"] != serde_json::json!({"type": "Sequence", "pretokenizers": [split, byte_level.clone()]}) {
        return fail("pre-tokenizer sequence");
    }
    if json["decoder"] != byte_level || json["post_processor"] != byte_level {
        return fail("decoder or post-processor");
    }
    let mut options = json["model"].as_object().cloned().unwrap_or_default();
    options.remove("vocab");
    options.remove("merges");
    let bpe = serde_json::json!({
        "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
        "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false
    });
    if Value::Object(options) != bpe {
        return fail("BPE settings");
    }
    icu::validate_regex(pattern)?;
    Ok((pattern.into(), !json["normalizer"].is_null()))
}

type Candidate = (usize, usize, usize, usize, usize, u32); // rank, left, right, revisions, merged id

struct Symbol {
    id: u32,
    previous: Option<usize>,
    next: Option<usize>,
    alive: bool,
    revision: usize,
}

impl Tokenizer {
    pub fn load(dir: &Path) -> Result<Tokenizer, String> {
        let json = read_json(&dir.join("tokenizer.json"))?;
        let (split_pattern, normalize_nfc) = pipeline(&json)?;
        let vocab: HashMap<String, u32> = serde_json::from_value(json["model"]["vocab"].clone())
            .map_err(|e| format!("tokenizer.json: vocab: {e}"))?;
        let alphabet = byte_alphabet();
        let reverse: HashMap<char, u8> = alphabet.iter().enumerate().map(|(i, &c)| (c, i as u8)).collect();
        let mut byte_ids = [0; 256];
        for (i, c) in alphabet.iter().enumerate() {
            byte_ids[i] = *vocab.get(&c.to_string()).ok_or(format!("tokenizer.json: no token for byte {i}"))?;
        }
        let added_list = json["added_tokens"].as_array().ok_or("tokenizer.json: no added_tokens")?;
        let mut most = vocab.values().copied().max().unwrap_or(0);
        let mut added = HashMap::new();
        let mut special_ids = Vec::new();
        for t in added_list {
            let (Some(id), Some(content)) = (t["id"].as_u64(), t["content"].as_str()) else {
                return Err("tokenizer.json: an added token without id or content".into());
            };
            if t["single_word"] == true || t["lstrip"] == true || t["rstrip"] == true || t["normalized"] == true {
                return Err(format!("tokenizer.json: unsupported behavior of added token {content:?}"));
            }
            added.insert(content.to_string(), id as u32);
            if t["special"] == true {
                special_ids.push(id as u32);
            }
            most = most.max(id as u32);
        }
        let mut bytes = vec![None; most as usize + 1];
        for (piece, &id) in &vocab {
            let decoded: Option<Vec<u8>> = piece.chars().map(|c| reverse.get(&c).copied()).collect();
            let decoded = decoded.ok_or(format!("tokenizer.json: token {id} is outside the byte alphabet"))?;
            bytes[id as usize] = Some(decoded);
        }
        for (content, &id) in &added {
            bytes[id as usize] = Some(content.as_bytes().to_vec());
        }
        let mut special = vec![false; bytes.len()];
        for id in special_ids {
            special[id as usize] = true;
        }
        let list = json["model"]["merges"].as_array().ok_or("tokenizer.json: no merges")?;
        let mut merges = HashMap::with_capacity(list.len());
        for (rank, m) in list.iter().enumerate() {
            let (left, right) = match m {
                Value::String(pair) => pair.split_once(' ').ok_or("tokenizer.json: a merge without a space")?,
                Value::Array(pair) if pair.len() == 2 => (
                    pair[0].as_str().ok_or("tokenizer.json: a merge's left part")?,
                    pair[1].as_str().ok_or("tokenizer.json: a merge's right part")?,
                ),
                _ => return Err("tokenizer.json: a merge of an unknown form".into()),
            };
            let id = |s: &str| vocab.get(s).copied().ok_or(format!("tokenizer.json: merge part {s:?} is not a token"));
            merges.insert((id(left)?, id(right)?), (rank, id(&format!("{left}{right}"))?));
        }
        Ok(Tokenizer { split_pattern, normalize_nfc, byte_ids, bytes, merges, added, special })
    }

    /// An added token's id by its content (`<|im_start|>`, `<think>`, ...).
    pub fn special(&self, content: &str) -> Option<u32> {
        self.added.get(content).copied()
    }

    /// Text as tokens, without the added vocabulary.
    pub fn encode(&self, text: &str, out: &mut Vec<u32>) -> Result<(), String> {
        if text.is_empty() {
            return Ok(());
        }
        let normalized;
        let text = if self.normalize_nfc {
            normalized = icu::normalize_nfc(text)?;
            &normalized
        } else {
            text
        };
        for piece in icu::split(text, &self.split_pattern)? {
            self.encode_piece(piece.as_bytes(), out);
        }
        Ok(())
    }

    /// Text as tokens with the added tokens it spells taken as such
    /// (leftmost, longest first), as the reference tokenizer does (the
    /// tests' comparison with it; prompts never take message text so).
    #[cfg(test)]
    pub fn encode_with_added(&self, text: &str, out: &mut Vec<u32>) -> Result<(), String> {
        let mut rest = text;
        while !rest.is_empty() {
            let next = self
                .added
                .iter()
                .filter_map(|(content, &id)| rest.find(content.as_str()).map(|at| (at, Reverse(content.len()), id)))
                .min();
            match next {
                Some((at, Reverse(len), id)) => {
                    self.encode(&rest[..at], out)?;
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => return self.encode(rest, out),
            }
        }
        Ok(())
    }

    fn encode_piece(&self, input: &[u8], out: &mut Vec<u32>) {
        if input.is_empty() {
            return;
        }
        // Linked symbols and a rank heap avoid quadratic work on long lines.
        let mut symbols: Vec<Symbol> = input
            .iter()
            .enumerate()
            .map(|(i, &b)| Symbol {
                id: self.byte_ids[b as usize],
                previous: i.checked_sub(1),
                next: if i + 1 < input.len() { Some(i + 1) } else { None },
                alive: true,
                revision: 0,
            })
            .collect();
        let mut heap = BinaryHeap::new();
        for i in 0..symbols.len().saturating_sub(1) {
            self.queue(i, &symbols, &mut heap);
        }
        while let Some(Reverse((_, left, right, lr, rr, merged))) = heap.pop() {
            if !symbols[left].alive
                || !symbols[right].alive
                || symbols[left].next != Some(right)
                || symbols[left].revision != lr
                || symbols[right].revision != rr
            {
                continue;
            }
            symbols[left].id = merged;
            symbols[left].revision += 1;
            symbols[left].next = symbols[right].next;
            symbols[right].alive = false;
            if let Some(after) = symbols[right].next {
                symbols[after].previous = Some(left);
            }
            if let Some(before) = symbols[left].previous {
                self.queue(before, &symbols, &mut heap);
            }
            self.queue(left, &symbols, &mut heap);
        }
        out.extend(symbols.into_iter().filter(|s| s.alive).map(|s| s.id));
    }

    fn queue(&self, left: usize, symbols: &[Symbol], heap: &mut BinaryHeap<Reverse<Candidate>>) {
        let Some(right) = symbols[left].next else { return };
        if let Some(&(rank, merged)) = self.merges.get(&(symbols[left].id, symbols[right].id)) {
            heap.push(Reverse((rank, left, right, symbols[left].revision, symbols[right].revision, merged)));
        }
    }

    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).and_then(Option::as_deref).unwrap_or(&[])
    }

    /// A special added token (one the reply's text leaves out, such as a
    /// vision marker; the template's markers are not special).
    pub fn is_special(&self, id: u32) -> bool {
        self.special.get(id as usize).copied().unwrap_or(false)
    }
}

/// Bytes to text as tokens arrive: an incomplete UTF-8 sequence waits for
/// its next token.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).expect("a valid prefix"));
                    self.pending.drain(..valid);
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..n);
                        }
                        None => break,
                    }
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> String {
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        rest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_stream_waits_for_whole_characters() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(&[b'a', 0xf0]), "a");
        assert_eq!(s.push(&[0x9f, 0x99]), "");
        assert_eq!(s.push(&[0x82, b'b']), "🙂b");
        assert_eq!(s.finish(), "");
        assert_eq!(s.push(&[0xff, 0xe2]), "\u{FFFD}");
        assert_eq!(s.finish(), "\u{FFFD}");
    }

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let a = byte_alphabet();
        assert_eq!(a.iter().collect::<std::collections::HashSet<_>>().len(), 256);
        assert_eq!(a[b' ' as usize], 'Ġ');
        assert_eq!(a[b'\n' as usize], 'Ċ');
    }
}
