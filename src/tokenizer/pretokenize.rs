//! Pre-tokenizer for the Qwen2/Qwen3 (and GPT-4 style) split regex:
//!
//! ```text
//! (?i:'s|'t|'re|'ve|'m|'ll|'d)
//! | [^\r\n\p{L}\p{N}]?\p{L}+
//! | \p{N}
//! |  ?[^\s\p{L}\p{N}]+[\r\n]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//! ```
//!
//! The `regex` crate cannot express the `(?!\S)` lookahead, so the
//! alternation is matched by hand with leftmost-first semantics (each
//! alternative tried in order, including the backtracking the regex engine
//! would do). Character classes use the real Unicode general categories.

use unicode_general_category::{GeneralCategory, get_general_category};

#[inline]
fn is_letter(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic();
    }
    matches!(
        get_general_category(c),
        GeneralCategory::UppercaseLetter
            | GeneralCategory::LowercaseLetter
            | GeneralCategory::TitlecaseLetter
            | GeneralCategory::ModifierLetter
            | GeneralCategory::OtherLetter
    )
}

#[inline]
fn is_number(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    // Nd | Nl | No
    c.is_numeric()
}

#[inline]
fn is_space(c: char) -> bool {
    c.is_whitespace()
}

#[inline]
fn is_newline(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// `[^\s\p{L}\p{N}]`
#[inline]
fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

/// Split `text` into pre-tokens; returns byte ranges into `text`.
pub fn split(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { text.len() };

    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let end = match_at(&chars, i).max(i + 1);
        out.push((byte_at(i), byte_at(end)));
        i = end;
    }
    out
}

/// Returns the end (exclusive, in chars) of the match starting at `i`.
fn match_at(chars: &[(usize, char)], i: usize) -> usize {
    let n = chars.len();
    let c = |k: usize| chars[k].1;

    // 1. contractions, case-insensitive
    if c(i) == '\'' && i + 1 < n {
        let c1 = c(i + 1).to_ascii_lowercase();
        if matches!(c1, 's' | 't' | 'm' | 'd') {
            return i + 2;
        }
        if i + 2 < n {
            let c2 = c(i + 2).to_ascii_lowercase();
            if (c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l') {
                return i + 3;
            }
        }
    }

    // 2. [^\r\n\p{L}\p{N}]?\p{L}+
    {
        let mut j = i;
        if !is_letter(c(j)) && !is_newline(c(j)) && !is_number(c(j)) {
            j += 1;
        }
        if j < n && is_letter(c(j)) {
            while j < n && is_letter(c(j)) {
                j += 1;
            }
            return j;
        }
    }

    // 3. \p{N}
    if is_number(c(i)) {
        return i + 1;
    }

    // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
    {
        let mut j = i;
        if c(j) == ' ' {
            j += 1;
        }
        if j < n && is_other(c(j)) {
            while j < n && is_other(c(j)) {
                j += 1;
            }
            while j < n && is_newline(c(j)) {
                j += 1;
            }
            return j;
        }
    }

    // whitespace run [i, ws_end)
    let mut ws_end = i;
    while ws_end < n && is_space(c(ws_end)) {
        ws_end += 1;
    }
    if ws_end > i {
        // 5. \s*[\r\n]+ : backtracking \s* stops at the last newline of the run
        if let Some(last_nl) = (i..ws_end).rev().find(|&k| is_newline(c(k))) {
            return last_nl + 1;
        }
        // 6. \s+(?!\S)
        if ws_end == n {
            return ws_end;
        }
        if ws_end - i >= 2 {
            return ws_end - 1;
        }
        // 7. \s+
        return ws_end;
    }

    // Unreachable for well-formed input (every char falls in some class),
    // but never stall.
    i + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<&str> {
        split(text).into_iter().map(|(a, b)| &text[a..b]).collect()
    }

    #[test]
    fn words_and_spaces() {
        assert_eq!(pieces("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pieces("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(pieces("trailing  "), vec!["trailing", "  "]);
    }

    #[test]
    fn digits_are_single() {
        assert_eq!(pieces("abc 1234"), vec!["abc", " ", "1", "2", "3", "4"]);
        assert_eq!(pieces("x=42;"), vec!["x", "=", "4", "2", ";"]);
    }

    #[test]
    fn contractions_and_punctuation() {
        assert_eq!(pieces("I'm can't WE'LL"), vec!["I", "'m", " can", "'t", " WE", "'LL"]);
        assert_eq!(pieces("hi!!! ok?\n\nyes"), vec!["hi", "!!!", " ok", "?\n\n", "yes"]);
        assert_eq!(pieces("(x)"), vec!["(x", ")"]);
    }

    #[test]
    fn newlines() {
        assert_eq!(pieces("a\n\n  b"), vec!["a", "\n\n", " ", " b"]);
        assert_eq!(pieces("a \n b"), vec!["a", " \n", " b"]);
        assert_eq!(pieces("\r\n"), vec!["\r\n"]);
    }

    #[test]
    fn unicode_classes() {
        assert_eq!(pieces("héllo wörld"), vec!["héllo", " wörld"]);
        assert_eq!(pieces("你好世界"), vec!["你好世界"]);
        // a combining mark is not \p{L}: it ends the letter run
        assert_eq!(pieces("e\u{0301}x"), vec!["e", "\u{0301}x"]);
        // full-width digits are \p{N}
        assert_eq!(pieces("１２"), vec!["１", "２"]);
    }
}
