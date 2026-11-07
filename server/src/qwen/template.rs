//! The Qwen3.8 chat template (the checkpoint's chat_template.jinja, which
//! both local Qwen3.8 checkpoints share) as code: OpenAI messages and tools
//! in, the prompt's tokens out. Ported from the user's mlx-lm project
//! (src/tokenizer/chat.rs, MIT), with the tools rendered as the request gives
//! them (the template's `tool | tojson`).
//!
//! The template's markers become their token ids, and the text between two
//! markers is encoded as one piece, as the reference tokenizer does; text
//! from messages is never searched for added tokens.

use serde::Serialize;
use serde_json::Value;

use super::tokenizer::Tokenizer;

/// The template's reasoning instructions by effort (`medium` has none).
fn instructions(effort: &str) -> &'static str {
    match effort {
        "xhigh" => {
            "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
        }
        "low" => {
            "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."
        }
        _ => "",
    }
}

const TOOL_FORMAT: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The markers the template writes; those the tokenizer has as added tokens
/// become their ids.
const MARKERS: [&str; 8] =
    ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"];

/// A message as the template takes it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub reasoning: String,
    pub calls: Vec<Call>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub name: String,
    /// The arguments object, in the request's key order.
    pub arguments: serde_json::Map<String, Value>,
}

/// Jinja's `trim`: Python's str.strip, which also strips U+001C..U+001F.
pub fn trim(text: &str) -> &str {
    text.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// Hugging Face's `tojson`: Python's json.dumps with ", " and ": " and the
/// characters as they are.
pub fn tojson(value: &impl Serialize) -> String {
    struct Spaced;
    impl serde_json::ser::Formatter for Spaced {
        fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
            if first { Ok(()) } else { w.write_all(b", ") }
        }
        fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
            w.write_all(b": ")
        }
    }
    let mut out = Vec::new();
    value.serialize(&mut serde_json::Serializer::with_formatter(&mut out, Spaced)).expect("JSON values serialize");
    String::from_utf8(out).expect("JSON is UTF-8")
}

/// The rendered prompt as text and markers.
#[derive(Default)]
struct Prompt {
    text: String, // the text since the last marker
    pieces: Vec<(String, Option<&'static str>)>,
}

impl Prompt {
    fn push(&mut self, s: &str) {
        let mut rest = s;
        while !rest.is_empty() {
            // the template's own text: its markers are markers
            match MARKERS.iter().filter_map(|m| rest.find(m).map(|at| (at, *m))).min() {
                Some((at, m)) => {
                    self.text.push_str(&rest[..at]);
                    self.mark(m);
                    rest = &rest[at + m.len()..];
                }
                None => {
                    self.text.push_str(rest);
                    rest = "";
                }
            }
        }
    }
    /// Text from a message: never a marker.
    fn content(&mut self, s: &str) {
        self.text.push_str(s);
    }
    fn mark(&mut self, m: &'static str) {
        self.pieces.push((std::mem::take(&mut self.text), Some(m)));
    }
    fn tokens(mut self, t: &Tokenizer) -> Result<Vec<u32>, String> {
        let last = std::mem::take(&mut self.text);
        self.pieces.push((last, None));
        let mut out = Vec::new();
        let mut run = String::new();
        for (text, marker) in self.pieces {
            run.push_str(&text);
            if let Some(m) = marker {
                match t.special(m) {
                    Some(id) => {
                        t.encode(&run, &mut out)?;
                        run.clear();
                        out.push(id);
                    }
                    None => run.push_str(m),
                }
            }
        }
        t.encode(&run, &mut out)?;
        Ok(out)
    }
}

pub struct Options<'a> {
    pub tools: &'a [Value],
    pub thinking: bool,
    /// xhigh, medium or low.
    pub effort: &'a str,
}

/// The prompt's tokens, ending with the assistant's turn opened (and, with
/// thinking off, its empty reasoning closed).
pub fn render(messages: &[Message], o: &Options, t: &Tokenizer) -> Result<Vec<u32>, String> {
    build(messages, o)?.tokens(t)
}

fn build(messages: &[Message], o: &Options) -> Result<Prompt, String> {
    if messages.is_empty() {
        return Err("no messages".into());
    }
    if !messages.iter().any(|m| {
        let c = trim(&m.content);
        m.role == "user" && !(c.starts_with("<tool_response>") && c.ends_with("</tool_response>"))
    }) {
        return Err("no user message".into());
    }
    let reasoning = if o.thinking { instructions(o.effort) } else { "" };
    let system = if messages[0].role == "system" { trim(&messages[0].content) } else { "" };
    let mut p = Prompt::default();
    if !o.tools.is_empty() {
        p.push("<|im_start|>system\n");
        if !reasoning.is_empty() {
            p.push(reasoning);
            p.push("\n\n");
        }
        p.push("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in o.tools {
            p.push("\n");
            p.content(&tojson(tool));
        }
        p.push("\n</tools>");
        p.push(TOOL_FORMAT);
        if !system.is_empty() {
            p.push("\n\n");
            p.content(system);
        }
        p.push("<|im_end|>\n");
    } else if !system.is_empty() {
        p.push("<|im_start|>system\n");
        if !reasoning.is_empty() {
            p.push(reasoning);
            p.push("\n\n");
        }
        p.content(system);
        p.push("<|im_end|>\n");
    } else if !reasoning.is_empty() {
        p.push("<|im_start|>system\n");
        p.push(reasoning);
        p.push("<|im_end|>\n");
    }
    for (i, m) in messages.iter().enumerate() {
        let content = trim(&m.content);
        match m.role.as_str() {
            "system" if i == 0 => {}
            "system" => return Err("a system message after the first message".into()),
            "user" => {
                p.push("<|im_start|>user\n");
                p.content(content);
                p.push("<|im_end|>\n");
            }
            "assistant" => {
                // the template keeps every turn's reasoning (preserve_thinking unset)
                p.push("<|im_start|>assistant\n<think>\n");
                p.content(trim(&m.reasoning));
                p.push("\n</think>\n\n");
                p.content(content);
                for (j, call) in m.calls.iter().enumerate() {
                    p.push(if j > 0 { "\n<tool_call>\n<function=" } else if !content.is_empty() { "\n\n<tool_call>\n<function=" } else { "<tool_call>\n<function=" });
                    p.content(&call.name);
                    p.push(">\n");
                    for (name, value) in &call.arguments {
                        p.push("<parameter=");
                        p.content(name);
                        p.push(">\n");
                        match value {
                            Value::String(s) => p.content(s),
                            v => p.content(&tojson(v)),
                        }
                        p.push("\n</parameter>\n");
                    }
                    p.push("</function>\n</tool_call>");
                }
                p.push("<|im_end|>\n");
            }
            "tool" => {
                if i > 0 && messages[i - 1].role != "tool" {
                    p.push("<|im_start|>user");
                }
                p.push("\n<tool_response>\n");
                p.content(content);
                p.push("\n</tool_response>");
                if messages.get(i + 1).is_none_or(|next| next.role != "tool") {
                    p.push("<|im_end|>\n");
                }
            }
            r => return Err(format!("unsupported message role {r:?}")),
        }
    }
    p.push("<|im_start|>assistant\n<think>\n");
    if !o.thinking {
        p.push("\n</think>\n\n");
    }
    Ok(p)
}

/// The prompt as text, markers spelled out: what the reference template
/// renders.
#[cfg(test)]
pub fn render_text(messages: &[Message], o: &Options) -> Result<String, String> {
    let mut p = build(messages, o)?;
    let mut out = String::new();
    for (text, marker) in std::mem::take(&mut p.pieces) {
        out.push_str(&text);
        out.push_str(marker.unwrap_or(""));
    }
    out.push_str(&p.text);
    Ok(out)
}
