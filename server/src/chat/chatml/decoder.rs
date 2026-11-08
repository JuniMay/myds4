//! The reply's framing as Qwen3.8 writes it: reasoning up to `</think>`,
//! then text, and tool calls as standalone `<tool_call>` blocks of
//! `<function=NAME>` and `<parameter=NAME>` XML.
//!
//! Text streams as it arrives. A `<tool_call>` line at the top level (not in
//! a fenced code block, not inside a line) starts the calls, which are parsed
//! at the reply's end: a clean stop gives the calls, and calls that do not
//! parse, or a reply cut short, give their text back as text.

use serde_json::{Map, Value};

const THINK_END: &str = "</think>";
const CALL_OPEN: &str = "<tool_call>";
const CALL_CLOSE: &str = "</tool_call>";

#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Text(String),
    Reasoning(String),
    Call {
        name: String,
        arguments: Map<String, Value>,
    },
}

pub struct Decoder {
    tools: Vec<Value>,
    thinking: bool,
    pending: String, // reasoning not yet known to precede </think>
    line: String,
    emitted: usize,
    fence: Option<(char, usize)>,
    calls: Option<String>,
    lead: bool, // right after </think>: the template's newlines before the text
}

/// The length of the longest proper prefix of `marker` that `text` ends with.
fn partial(text: &str, marker: &str) -> usize {
    marker
        .char_indices()
        .map(|(i, _)| i)
        .filter(|&i| i > 0)
        .rev()
        .find(|&n| text.ends_with(&marker[..n]))
        .unwrap_or(0)
}

impl Decoder {
    /// `thinking`: the prompt opened the reasoning (it ends with `<think>\n`).
    /// `tools`: the request's tools (OpenAI form), for the parameters' types.
    pub fn new(thinking: bool, tools: Vec<Value>) -> Decoder {
        Decoder {
            tools,
            thinking,
            pending: String::new(),
            line: String::new(),
            emitted: 0,
            fence: None,
            calls: None,
            lead: false,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.thinking {
            self.pending.push_str(text);
            if let Some(end) = self.pending.find(THINK_END) {
                reasoning(&mut out, &self.pending[..end]);
                let rest = self.pending[end + THINK_END.len()..].to_owned();
                self.pending.clear();
                self.thinking = false;
                self.lead = true;
                self.text(&rest, &mut out);
            } else {
                let keep = partial(&self.pending, THINK_END);
                let split = self.pending.len() - keep;
                reasoning(&mut out, &self.pending[..split]);
                self.pending.drain(..split);
            }
        } else {
            self.text(text, &mut out);
        }
        out
    }

    fn text(&mut self, text: &str, out: &mut Vec<Piece>) {
        let text = if self.lead {
            text.trim_start_matches(['\n', '\r'])
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        self.lead = false;
        if let Some(calls) = &mut self.calls {
            calls.push_str(text);
            return;
        }
        for part in text.split_inclusive('\n') {
            if let Some(calls) = &mut self.calls {
                calls.push_str(part);
                continue;
            }
            self.line.push_str(part);
            let complete = self.line.ends_with('\n');
            let body = self.line.trim_end_matches(['\r', '\n']);
            // A call opens on a standalone top-level line only.
            let candidate =
                self.fence.is_none() && self.emitted == 0 && CALL_OPEN.starts_with(body);
            if candidate && complete && body == CALL_OPEN {
                self.calls = Some(std::mem::take(&mut self.line));
                self.emitted = 0;
                continue;
            }
            if !candidate || complete {
                plain(out, &self.line[self.emitted..]);
                self.emitted = self.line.len();
            }
            if complete {
                self.fence_line();
                self.line.clear();
                self.emitted = 0;
            }
        }
    }

    fn fence_line(&mut self) {
        let line = self.line.trim_end_matches(['\r', '\n']);
        let indent = line.bytes().take_while(|&b| b == b' ').count();
        if indent > 3 {
            return;
        }
        let body = &line[indent..];
        let Some(marker) = body.chars().next().filter(|c| matches!(c, '`' | '~')) else {
            return;
        };
        let count = body.chars().take_while(|&c| c == marker).count();
        if count < 3 {
            return;
        }
        match self.fence {
            Some((open, n)) => {
                if marker == open && count >= n && body[count..].trim().is_empty() {
                    self.fence = None;
                }
            }
            None => {
                if marker != '`' || !body[count..].contains('`') {
                    self.fence = Some((marker, count));
                }
            }
        }
    }

    /// The reply's end; `clean`: it stopped at an end token.
    pub fn finish(&mut self, clean: bool) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.thinking {
            reasoning(&mut out, &std::mem::take(&mut self.pending));
        }
        if self.calls.is_none()
            && self.fence.is_none()
            && self.emitted == 0
            && self.line == CALL_OPEN
        {
            self.calls = Some(std::mem::take(&mut self.line));
        }
        match self.calls.take() {
            Some(calls) => match clean.then(|| parse(&calls, &self.tools)).flatten() {
                Some(parsed) => out.extend(parsed),
                None => plain(&mut out, &calls),
            },
            None => {
                plain(&mut out, &self.line[self.emitted..]);
                self.line.clear();
                self.emitted = 0;
            }
        }
        out
    }
}

fn plain(out: &mut Vec<Piece>, text: &str) {
    if !text.is_empty() {
        out.push(Piece::Text(text.into()));
    }
}

fn reasoning(out: &mut Vec<Piece>, text: &str) {
    if !text.is_empty() {
        out.push(Piece::Reasoning(text.into()));
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
}

fn tag<'a>(input: &mut &'a str, prefix: &str) -> Option<&'a str> {
    let after = input.trim_start().strip_prefix(prefix)?;
    let end = after.find('>')?;
    let name = &after[..end];
    *input = &after[end + 1..];
    valid_name(name).then_some(name)
}

fn literal(input: &mut &str, s: &str) -> Option<()> {
    *input = input.trim_start().strip_prefix(s)?;
    Some(())
}

/// Whether a property's schema takes a string as it is.
fn takes_string(schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(t)) => t == "string",
        Some(Value::Array(ts)) => ts.iter().any(|t| t == "string"),
        _ => ["anyOf", "oneOf"].iter().any(|k| {
            schema
                .get(*k)
                .and_then(Value::as_array)
                .is_some_and(|c| c.iter().any(takes_string))
        }),
    }
}

/// The calls of a reply's call text; None if it is not calls throughout.
fn parse(mut input: &str, tools: &[Value]) -> Option<Vec<Piece>> {
    let mut out = Vec::new();
    while !input.trim().is_empty() {
        literal(&mut input, CALL_OPEN)?;
        let name = tag(&mut input, "<function=")?.to_owned();
        let schema = tools
            .iter()
            .map(|t| t.get("function").unwrap_or(t))
            .find(|f| f.get("name").and_then(Value::as_str) == Some(&name))
            .and_then(|f| f.get("parameters"));
        let mut arguments = Map::new();
        while input.trim_start().starts_with("<parameter=") {
            let key = tag(&mut input, "<parameter=")?.to_owned();
            let end = input.find("</parameter>")?;
            let raw = &input[..end];
            let raw = raw
                .strip_prefix("\r\n")
                .or_else(|| raw.strip_prefix('\n'))
                .unwrap_or(raw);
            let raw = raw
                .strip_suffix("\r\n")
                .or_else(|| raw.strip_suffix('\n'))
                .unwrap_or(raw);
            let property = schema
                .and_then(|s| s.get("properties"))
                .and_then(|p| p.get(&key));
            let value = if property.is_some_and(takes_string) {
                Value::String(raw.to_owned())
            } else {
                serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
            };
            if arguments.insert(key, value).is_some() {
                return None;
            }
            input = &input[end + "</parameter>".len()..];
        }
        literal(&mut input, "</function>")?;
        literal(&mut input, CALL_CLOSE)?;
        out.push(Piece::Call { name, arguments });
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Value> {
        vec![
            serde_json::json!({"type": "function", "function": {"name": "read", "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}}}),
        ]
    }

    fn decode(text: &str, thinking: bool, cuts: &[usize], clean: bool) -> Vec<Piece> {
        let mut d = Decoder::new(thinking, tools());
        let mut out = Vec::new();
        let mut at = 0;
        for &n in cuts {
            out.extend(d.push(&text[at..at + n]));
            at += n;
        }
        out.extend(d.push(&text[at..]));
        out.extend(d.finish(clean));
        out
    }

    fn collect(pieces: &[Piece]) -> (String, String, Vec<(String, Map<String, Value>)>) {
        let mut r = (String::new(), String::new(), Vec::new());
        for p in pieces {
            match p {
                Piece::Text(t) => r.0.push_str(t),
                Piece::Reasoning(t) => r.1.push_str(t),
                Piece::Call { name, arguments } => r.2.push((name.clone(), arguments.clone())),
            }
        }
        r
    }

    #[test]
    fn calls_and_reasoning_survive_every_chunk_boundary() {
        let source = "Read carefully; <tool_call> is only a thought.\n</think>\n\nI will inspect it.\n<tool_call>\n<function=read>\n<parameter=path>\na b.txt\n</parameter>\n<parameter=lines>\n3\n</parameter>\n</function>\n</tool_call>";
        let expected = collect(&decode(source, true, &[], true));
        assert_eq!(expected.0, "I will inspect it.\n");
        assert_eq!(
            expected.1,
            "Read carefully; <tool_call> is only a thought.\n"
        );
        assert_eq!(
            Value::Object(expected.2[0].1.clone()),
            serde_json::json!({"path": "a b.txt", "lines": 3})
        );
        for cut in 0..source.len() {
            assert_eq!(
                collect(&decode(source, true, &[cut], true)),
                expected,
                "cut at {cut}"
            );
        }
        assert_eq!(
            collect(&decode(source, true, &vec![1; source.len()], true)),
            expected
        );
    }

    #[test]
    fn prose_and_markdown_examples_stay_text() {
        for text in [
            "Use <tool_call> to invoke a function.",
            "`<tool_call>`\n",
            "> <tool_call>\n",
            "    <tool_call>\n",
            "```xml\n<tool_call>\n<function=read>\n</function>\n</tool_call>\n```\n",
            "~~~\n<tool_call>\n~~~\n",
        ] {
            let got = collect(&decode(text, false, &vec![1; text.len()], true));
            assert_eq!(got.0, text);
            assert!(got.2.is_empty());
        }
    }

    #[test]
    fn calls_that_do_not_parse_or_are_cut_short_come_back_as_text() {
        let valid = "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(collect(&decode(valid, false, &[], true)).2.len(), 1);
        let cut = collect(&decode(valid, false, &[], false));
        assert!(cut.2.is_empty());
        assert_eq!(cut.0, valid);
        for text in [
            format!("{valid}\nExtra suffix"),
            format!("{valid}\n<tool_call>\n<function=read>"),
            "<tool_call>".into(),
        ] {
            let got = collect(&decode(&text, false, &[], true));
            assert!(got.2.is_empty(), "{text}");
            assert_eq!(got.0, text);
        }
    }

    #[test]
    fn text_streams_before_its_line_ends() {
        let mut d = Decoder::new(false, tools());
        assert_eq!(d.push("Hello"), vec![Piece::Text("Hello".into())]);
        assert_eq!(d.push(" there"), vec![Piece::Text(" there".into())]);
    }
}
