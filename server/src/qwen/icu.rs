//! The tokenizer's Unicode operations from the macOS ICU library
//! (libicucore): NFC normalization and the pre-tokenizer's regex split.
//! Handles and buffers belong to each call. Ported from the user's mlx-lm
//! project (src/tokenizer/icu.rs, MIT).

use std::{ffi::c_void, ptr};

#[link(name = "icucore")]
unsafe extern "C" {
    fn uregex_open(pattern: *const u16, length: i32, flags: u32, parse_error: *mut c_void, status: *mut i32)
    -> *mut c_void;
    fn uregex_close(regex: *mut c_void);
    fn uregex_setText(regex: *mut c_void, text: *const u16, length: i32, status: *mut i32);
    fn uregex_findNext(regex: *mut c_void, status: *mut i32) -> i8;
    fn uregex_start(regex: *mut c_void, group: i32, status: *mut i32) -> i32;
    fn uregex_end(regex: *mut c_void, group: i32, status: *mut i32) -> i32;
    fn unorm2_getNFCInstance(status: *mut i32) -> *const c_void;
    fn unorm2_normalize(
        normalizer: *const c_void,
        source: *const u16,
        length: i32,
        destination: *mut u16,
        capacity: i32,
        status: *mut i32,
    ) -> i32;
}

fn checked(status: i32, operation: &str) -> Result<(), String> {
    // ICU's negative status values are successes with warnings.
    if status > 0 { Err(format!("ICU {operation} failed (error {status})")) } else { Ok(()) }
}

fn length(units: &[u16]) -> Result<i32, String> {
    i32::try_from(units.len()).map_err(|_| "the text exceeds ICU's index range".to_string())
}

pub fn normalize_nfc(text: &str) -> Result<String, String> {
    if text.is_empty() {
        return Ok(String::new());
    }
    let source: Vec<u16> = text.encode_utf16().collect();
    let n = length(&source)?;
    let mut status = 0;
    // SAFETY: ICU returns an immutable singleton it owns.
    let normalizer = unsafe { unorm2_getNFCInstance(&mut status) };
    checked(status, "NFC initialization")?;
    if normalizer.is_null() {
        return Err("ICU returned no NFC normalizer".into());
    }
    // SAFETY: source is live and n bounds it; a null output asks for the length.
    let needed = unsafe { unorm2_normalize(normalizer, source.as_ptr(), n, ptr::null_mut(), 0, &mut status) };
    if status != 15 {
        // U_BUFFER_OVERFLOW_ERROR (15) is the expected preflight result.
        checked(status, "NFC length")?;
    }
    if needed < 0 {
        return Err("ICU returned a negative normalized length".into());
    }
    let mut destination = vec![0_u16; needed as usize];
    status = 0;
    // SAFETY: both buffers are live and have their stated capacities.
    let written =
        unsafe { unorm2_normalize(normalizer, source.as_ptr(), n, destination.as_mut_ptr(), needed, &mut status) };
    checked(status, "NFC normalization")?;
    if written != needed {
        return Err("ICU returned an inconsistent normalized length".into());
    }
    String::from_utf16(&destination).map_err(|_| "ICU produced invalid UTF-16".into())
}

struct Regex(*mut c_void);

impl Regex {
    fn new(pattern: &str) -> Result<Self, String> {
        if pattern.is_empty() {
            return Err("the tokenizer's split regex is empty".into());
        }
        let expression: Vec<u16> = pattern.encode_utf16().collect();
        let mut status = 0;
        // SAFETY: the expression is valid UTF-16 of the given length; ICU copies it.
        let raw = unsafe { uregex_open(expression.as_ptr(), length(&expression)?, 0, ptr::null_mut(), &mut status) };
        checked(status, "regex compilation")?;
        if raw.is_null() {
            return Err("ICU returned no regex".into());
        }
        Ok(Self(raw))
    }
}

impl Drop for Regex {
    fn drop(&mut self) {
        // SAFETY: this value owns the handle and closes it once.
        unsafe { uregex_close(self.0) };
    }
}

pub fn validate_regex(pattern: &str) -> Result<(), String> {
    Regex::new(pattern).map(|_| ())
}

/// The pieces of `text`: each match of `pattern`, and the text between
/// matches (the Split pre-tokenizer's "Isolated" behavior).
pub fn split(text: &str, pattern: &str) -> Result<Vec<String>, String> {
    // The input outlives the regex: ICU borrows it until the regex closes.
    let input: Vec<u16> = text.encode_utf16().collect();
    let mut status = 0;
    let regex = Regex::new(pattern)?;
    // SAFETY: input stays alive until after regex is dropped.
    unsafe { uregex_setText(regex.0, input.as_ptr(), length(&input)?, &mut status) };
    checked(status, "regex input")?;
    let utf16 = |units: &[u16]| String::from_utf16(units).map_err(|_| "a regex piece is invalid UTF-16".to_string());
    let mut parts = Vec::new();
    let mut consumed = 0;
    loop {
        // SAFETY: regex is live and used on this stack only.
        let found = unsafe { uregex_findNext(regex.0, &mut status) };
        checked(status, "regex matching")?;
        if found == 0 {
            break;
        }
        // SAFETY: group zero exists for every match.
        let (start, end) = unsafe { (uregex_start(regex.0, 0, &mut status), uregex_end(regex.0, 0, &mut status)) };
        checked(status, "regex match bounds")?;
        if start < 0 || end < start || (start as usize) < consumed || end as usize > input.len() {
            return Err("the tokenizer's regex returned invalid bounds".into());
        }
        let (start, end) = (start as usize, end as usize);
        if start > consumed {
            parts.push(utf16(&input[consumed..start])?);
        }
        if end > start {
            parts.push(utf16(&input[start..end])?);
        }
        consumed = end;
    }
    if consumed < input.len() {
        parts.push(utf16(&input[consumed..])?);
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfc_composes_and_reorders_unicode() {
        assert_eq!(normalize_nfc("e\u{301} Å 한글").unwrap(), "é Å 한글");
        assert_eq!(normalize_nfc("").unwrap(), "");
    }

    #[test]
    fn regex_splits_contractions_unicode_and_line_endings() {
        let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        let result = split("I'm 中🙂 123\r\n \t", pattern).unwrap();
        assert_eq!(result, ["I", "'m", " 中", "🙂", " ", "1", "2", "3", "\r\n", " \t"]);
    }

    #[test]
    fn isolated_split_keeps_the_text_between_matches() {
        assert_eq!(split("one 12! three 456?", r"\d+").unwrap(), ["one ", "12", "! three ", "456", "?"]);
        assert_eq!(split("not a match", r"\d+").unwrap(), ["not a match"]);
        assert_eq!(split("ab", "(?=b)").unwrap(), ["a", "b"]);
        assert!(validate_regex("").is_err());
        assert!(validate_regex("[").is_err());
    }
}
