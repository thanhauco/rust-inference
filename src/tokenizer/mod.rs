//! Byte-level BPE tokenizer compatible with HuggingFace `tokenizer.json`
//! files for Qwen2/Qwen3 models.
//!
//! Pipeline (mirrors the HF one): split out added/special tokens, NFC
//! normalise, pre-tokenize with the Qwen regex, map bytes to the GPT-2
//! printable alphabet, then apply BPE merges by rank.

mod pretokenize;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::path::Path;

use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::error::{Error, Result};

pub use pretokenize::split as pre_tokenize;

#[derive(Deserialize)]
struct TokenizerJson {
    model: ModelJson,
    #[serde(default)]
    added_tokens: Vec<AddedTokenJson>,
    #[serde(default)]
    normalizer: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ModelJson {
    vocab: HashMap<String, u32>,
    merges: Vec<MergeJson>,
    #[serde(default)]
    ignore_merges: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MergeJson {
    Joined(String),
    Pair([String; 2]),
}

#[derive(Deserialize)]
struct AddedTokenJson {
    id: u32,
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Debug, Clone)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

/// GPT-2 byte <-> printable unicode mapping. Printable latin bytes map to
/// themselves; the rest are shifted to U+0100.. in byte order.
fn byte_alphabet() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut assigned = [false; 256];
    for b in (33..=126).chain(161..=172).chain(174..=255) {
        table[b] = char::from_u32(b as u32).unwrap();
        assigned[b] = true;
    }
    let mut n = 0;
    for b in 0..256 {
        if !assigned[b] {
            table[b] = char::from_u32(0x100 + n).unwrap();
            n += 1;
        }
    }
    table
}

pub struct BpeTokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// (left id, right id) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    ignore_merges: bool,
    nfc: bool,
    /// id of the token for each raw byte
    byte_ids: [u32; 256],
    /// unicode char of the byte alphabet -> raw byte
    char_to_byte: HashMap<char, u8>,
    added: Vec<AddedToken>,
    added_by_id: HashMap<u32, usize>,

    eos_id: Option<u32>,
    bos_id: Option<u32>,
    pad_id: Option<u32>,
    im_start_id: Option<u32>,
    im_end_id: Option<u32>,
}

impl BpeTokenizer {
    /// Load `tokenizer.json` (and `tokenizer_config.json` for bos/eos/pad
    /// overrides, if present) from a HuggingFace checkpoint directory.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path).map_err(|e| Error::io(&path, e))?;
        let tj: TokenizerJson = serde_json::from_str(&text).map_err(|e| Error::json(&path, e))?;
        let mut tok = Self::from_parts(tj)?;

        let cfg_path = dir.join("tokenizer_config.json");
        if let Ok(text) = std::fs::read_to_string(&cfg_path) {
            let cfg: serde_json::Value = serde_json::from_str(&text).map_err(|e| Error::json(&cfg_path, e))?;
            let special = |key: &str| -> Option<u32> {
                let v = cfg.get(key)?;
                let s = v.as_str().or_else(|| v.get("content").and_then(|c| c.as_str()))?;
                tok.token_to_id.get(s).copied()
            };
            if let Some(id) = special("bos_token") {
                tok.bos_id = Some(id);
            }
            if let Some(id) = special("eos_token") {
                tok.eos_id = Some(id);
            }
            if let Some(id) = special("pad_token") {
                tok.pad_id = Some(id);
            }
        }
        Ok(tok)
    }

    fn from_parts(tj: TokenizerJson) -> Result<Self> {
        let max_id = tj.model.vocab.values().chain(tj.added_tokens.iter().map(|a| &a.id)).copied().max().unwrap_or(0);
        let mut id_to_token = vec![String::new(); max_id as usize + 1];
        let mut token_to_id = HashMap::with_capacity(tj.model.vocab.len() + tj.added_tokens.len());
        for (s, &id) in &tj.model.vocab {
            id_to_token[id as usize] = s.clone();
            token_to_id.insert(s.clone(), id);
        }

        let mut added = Vec::with_capacity(tj.added_tokens.len());
        for a in tj.added_tokens {
            id_to_token[a.id as usize] = a.content.clone();
            token_to_id.insert(a.content.clone(), a.id);
            added.push(AddedToken { id: a.id, content: a.content, special: a.special });
        }
        // longest first so the splitter prefers e.g. "<|im_start|>" over a prefix
        added.sort_by_key(|a| Reverse(a.content.len()));
        let added_by_id = added.iter().enumerate().map(|(i, a)| (a.id, i)).collect();

        let alphabet = byte_alphabet();
        let mut byte_ids = [0u32; 256];
        let mut char_to_byte = HashMap::with_capacity(256);
        for (b, &c) in alphabet.iter().enumerate() {
            char_to_byte.insert(c, b as u8);
            byte_ids[b] = *token_to_id
                .get(&c.to_string())
                .ok_or_else(|| Error::Config(format!("vocab is missing the byte token for {b:#04x}")))?;
        }

        let mut merges = HashMap::with_capacity(tj.model.merges.len());
        for (rank, m) in tj.model.merges.into_iter().enumerate() {
            let (l, r) = match m {
                MergeJson::Pair([l, r]) => (l, r),
                MergeJson::Joined(s) => match s.split_once(' ') {
                    Some((l, r)) => (l.to_string(), r.to_string()),
                    None => return Err(Error::Config(format!("bad merge {s:?}"))),
                },
            };
            let (Some(&li), Some(&ri)) = (token_to_id.get(&l), token_to_id.get(&r)) else {
                return Err(Error::Config(format!("merge {l:?} {r:?} uses unknown tokens")));
            };
            let merged = format!("{l}{r}");
            let Some(&mi) = token_to_id.get(&merged) else {
                return Err(Error::Config(format!("merge result {merged:?} not in vocab")));
            };
            merges.entry((li, ri)).or_insert((rank as u32, mi));
        }

        let nfc = tj.normalizer.as_ref().map(|n| n.to_string().contains("\"NFC\"")).unwrap_or(false);

        let find = |s: &str| token_to_id.get(s).copied();
        Ok(BpeTokenizer {
            eos_id: find("<|endoftext|>"),
            bos_id: find("<|im_start|>"),
            pad_id: find("<|endoftext|>"),
            im_start_id: find("<|im_start|>"),
            im_end_id: find("<|im_end|>"),
            token_to_id,
            id_to_token,
            merges,
            ignore_merges: tj.model.ignore_merges,
            nfc,
            byte_ids,
            char_to_byte,
            added,
            added_by_id,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }
    pub fn im_start_id(&self) -> Option<u32> {
        self.im_start_id
    }
    pub fn im_end_id(&self) -> Option<u32> {
        self.im_end_id
    }

    /// Whether generation should stop at `id`.
    pub fn is_stop_token(&self, id: u32) -> bool {
        Some(id) == self.eos_id || Some(id) == self.im_end_id
    }

    /// Encode UTF-8 text to token ids. Added tokens (e.g. `<|im_start|>`)
    /// appearing in the text are recognised as single tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.len() / 3 + 1);
        let mut rest = text;
        while !rest.is_empty() {
            let (plain_len, hit) = self.find_added(rest);
            self.encode_plain(&rest[..plain_len], &mut ids);
            match hit {
                Some(a) => {
                    ids.push(a.id);
                    rest = &rest[plain_len + a.content.len()..];
                }
                None => break,
            }
        }
        ids
    }

    /// Byte length of the text before the first added token, and that token.
    fn find_added(&self, text: &str) -> (usize, Option<&AddedToken>) {
        if self.added.is_empty() {
            return (text.len(), None);
        }
        for (pos, _) in text.char_indices() {
            let tail = &text[pos..];
            if let Some(a) = self.added.iter().find(|a| tail.starts_with(a.content.as_str())) {
                return (pos, Some(a));
            }
        }
        (text.len(), None)
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let normalized;
        let text = if self.nfc {
            normalized = text.nfc().collect::<String>();
            normalized.as_str()
        } else {
            text
        };
        for (a, b) in pre_tokenize(text) {
            self.bpe_word(&text[a..b], ids);
        }
    }

    /// Apply merges to one pre-token. Uses a min-heap of candidate pairs over
    /// a doubly linked symbol list: lowest rank first, leftmost on ties.
    fn bpe_word(&self, word: &str, ids: &mut Vec<u32>) {
        let bytes = word.as_bytes();
        if self.ignore_merges {
            let mapped: String = bytes.iter().map(|&b| self.byte_char(b)).collect();
            if let Some(&id) = self.token_to_id.get(&mapped) {
                ids.push(id);
                return;
            }
        }
        if bytes.len() == 1 {
            ids.push(self.byte_ids[bytes[0] as usize]);
            return;
        }

        struct Sym {
            id: u32,
            prev: usize,
            next: usize,
            alive: bool,
        }
        const NONE: usize = usize::MAX;
        let n = bytes.len();
        let mut syms: Vec<Sym> = bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| Sym {
                id: self.byte_ids[b as usize],
                prev: if i == 0 { NONE } else { i - 1 },
                next: if i + 1 == n { NONE } else { i + 1 },
                alive: true,
            })
            .collect();

        // (rank, left position, left id, right id, merged id)
        let mut heap = BinaryHeap::new();
        let push = |heap: &mut BinaryHeap<_>, syms: &[Sym], left: usize| {
            let right = syms[left].next;
            if right == NONE {
                return;
            }
            if let Some(&(rank, merged)) = self.merges.get(&(syms[left].id, syms[right].id)) {
                heap.push(Reverse((rank, left, syms[left].id, syms[right].id, merged)));
            }
        };
        for i in 0..n - 1 {
            push(&mut heap, &syms, i);
        }

        while let Some(Reverse((_, left, lid, rid, merged))) = heap.pop() {
            let right = syms[left].next;
            // stale entry: either side already changed
            if !syms[left].alive || right == NONE || syms[left].id != lid || syms[right].id != rid {
                continue;
            }
            syms[left].id = merged;
            let after = syms[right].next;
            syms[left].next = after;
            syms[right].alive = false;
            if after != NONE {
                syms[after].prev = left;
            }
            let before = syms[left].prev;
            if before != NONE {
                push(&mut heap, &syms, before);
            }
            push(&mut heap, &syms, left);
        }

        let mut i = 0;
        while i != NONE {
            ids.push(syms[i].id);
            i = syms[i].next;
        }
    }

    fn byte_char(&self, b: u8) -> char {
        // byte_ids always resolve, so the id_to_token entry is a single char
        self.id_to_token[self.byte_ids[b as usize] as usize].chars().next().unwrap()
    }

    /// Raw bytes for one token. Special tokens produce nothing unless
    /// `keep_special`; non-special added tokens are emitted literally.
    fn token_bytes(&self, id: u32, keep_special: bool, out: &mut Vec<u8>) {
        if let Some(&i) = self.added_by_id.get(&id) {
            let a = &self.added[i];
            if !a.special || keep_special {
                out.extend_from_slice(a.content.as_bytes());
            }
            return;
        }
        let Some(tok) = self.id_to_token.get(id as usize) else {
            return;
        };
        for c in tok.chars() {
            match self.char_to_byte.get(&c) {
                Some(&b) => out.push(b),
                // not byte-level (should not happen for BPE vocab entries)
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }

    /// Decode ids to text, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.token_bytes(id, false, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The vocabulary string for one id (byte-decoded when possible).
    pub fn decode_token(&self, id: u32) -> String {
        let mut bytes = Vec::new();
        self.token_bytes(id, true, &mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Qwen chat-format prompt for a single user turn.
    pub fn chat_prompt(user: &str, enable_thinking: bool) -> String {
        let mut s = format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
        if !enable_thinking {
            s.push_str("<think>\n\n</think>\n\n");
        }
        s
    }
}

/// Incremental decoder for streaming generation: buffers bytes until they
/// form complete UTF-8 so multi-byte characters split across tokens print
/// correctly.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one token; returns the text that became printable.
    pub fn push(&mut self, tok: &BpeTokenizer, id: u32) -> String {
        tok.token_bytes(id, false, &mut self.pending);
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let out = s.to_string();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // an invalid (not merely incomplete) sequence: flush lossily
                if e.error_len().is_some() {
                    let out = String::from_utf8_lossy(&self.pending).into_owned();
                    self.pending.clear();
                    return out;
                }
                let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                out
            }
        }
    }

    /// Whatever is left at the end of generation.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A toy vocabulary: the 256 byte tokens plus a few merges.
    fn toy() -> BpeTokenizer {
        let alphabet = byte_alphabet();
        let mut vocab: HashMap<String, u32> = HashMap::new();
        for (i, c) in alphabet.iter().enumerate() {
            vocab.insert(c.to_string(), i as u32);
        }
        let merges = [("h", "e"), ("l", "l"), ("he", "ll"), ("hell", "o"), ("Ġ", "w"), ("Ġw", "o")];
        let mut list = Vec::new();
        for (l, r) in merges {
            let id = vocab.len() as u32;
            vocab.insert(format!("{l}{r}"), id);
            list.push(MergeJson::Pair([l.to_string(), r.to_string()]));
        }
        let n = vocab.len() as u32;
        let tj = TokenizerJson {
            model: ModelJson { vocab, merges: list, ignore_merges: false },
            added_tokens: vec![
                AddedTokenJson { id: n, content: "<|im_start|>".into(), special: true },
                AddedTokenJson { id: n + 1, content: "<|im_end|>".into(), special: true },
                AddedTokenJson { id: n + 2, content: "<think>".into(), special: false },
            ],
            normalizer: None,
        };
        BpeTokenizer::from_parts(tj).unwrap()
    }

    #[test]
    fn merges_apply_by_rank() {
        let t = toy();
        let ids = t.encode("hello wo");
        let toks: Vec<String> = ids.iter().map(|&i| t.id_to_token[i as usize].clone()).collect();
        assert_eq!(toks, vec!["hello", "Ġwo"]);
        assert_eq!(t.decode(&ids), "hello wo");
    }

    #[test]
    fn special_tokens_split_and_skip() {
        let t = toy();
        let ids = t.encode("<|im_start|>hello<|im_end|><think>");
        assert_eq!(ids[0], t.im_start_id().unwrap());
        assert_eq!(ids[ids.len() - 2], t.im_end_id().unwrap());
        assert_eq!(t.decode(&ids), "hello<think>");
        assert_eq!(t.decode_token(t.im_end_id().unwrap()), "<|im_end|>");
    }

    #[test]
    fn utf8_roundtrip_and_streaming() {
        let t = toy();
        let text = "naïve 你好 🙂";
        let ids = t.encode(text);
        assert_eq!(t.decode(&ids), text);
        let mut sd = StreamDecoder::new();
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&sd.push(&t, id));
        }
        streamed.push_str(&sd.finish());
        assert_eq!(streamed, text);
    }
}
