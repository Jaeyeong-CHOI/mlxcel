// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Laya rows: ports of upstream's `_render_criterion`, `_options` and
//! `Laya._sequence` (`mlx_vlm/models/laya/laya.py`).
//!
//! One row per question: `[CLS]`, the question head, `[SEP]`, one
//! `[MASK]`-led run per option (the marker), `[SEP]`, the state, `[SEP]`.
//! Unlike Clef, options keep the request order and non-string values are
//! JSON-encoded with Python's default separators and key order. Everything
//! here runs on the tokenizer alone, so the sequence tests need no weights.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;

use super::decider2_prompt::python_float;
use super::{DecisionError, Question, QuestionKind};

/// Most tokens of one option after its marker (upstream's `[:48]`).
pub(crate) const OPTION_TOKENS: usize = 48;

/// Upstream's refusal of a row whose markers do not fit `max_len`.
pub(crate) const BUDGET_ERROR: &str = "Options do not fit the Laya question budget";

/// One row: its ids and the marker position of each option.
pub(crate) type Row = (Vec<u32>, Vec<usize>);

const BOOL_FALSE: &str = "no, the statement does not hold";
const BOOL_TRUE: &str = "yes, the statement holds";

fn invalid(message: impl Into<String>) -> DecisionError {
    DecisionError::InvalidInput(message.into())
}

/// Upstream's question type token (`bool` runs as `noul`).
pub(crate) fn type_token(kind: QuestionKind) -> &'static str {
    match kind {
        QuestionKind::Score => "score",
        QuestionKind::Bool => "noul",
        _ => "choice",
    }
}

/// `QUESTION_TYPES`: `choice` 0, `score` 1, `noul` 2.
pub(crate) fn qtype(kind: QuestionKind) -> i32 {
    match kind {
        QuestionKind::Score => 1,
        QuestionKind::Bool => 2,
        _ => 0,
    }
}

/// Python's `json.dumps(value, ensure_ascii=...)` with the default
/// separators (`", "`, `": "`) and the keys in request order.
pub(crate) fn python_dumps(value: &Value, ensure_ascii: bool) -> String {
    let mut out = String::new();
    write_python(value, ensure_ascii, &mut out);
    out
}

fn write_python(value: &Value, ensure_ascii: bool, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) if number.is_i64() || number.is_u64() => {
            out.push_str(&number.to_string());
        }
        Value::Number(number) => out.push_str(&python_float(number.as_f64().unwrap_or(0.0))),
        Value::String(text) => write_python_string(text, ensure_ascii, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python(item, ensure_ascii, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (index, (key, item)) in map.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                write_python_string(key, ensure_ascii, out);
                out.push_str(": ");
                write_python(item, ensure_ascii, out);
            }
            out.push('}');
        }
    }
}

fn write_python_string(text: &str, ensure_ascii: bool, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ensure_ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `_render_criterion`: a string as is, anything else as
/// `json.dumps(value, ensure_ascii=False)`.
pub(crate) fn render_criterion(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => python_dumps(other, false),
    }
}

/// `value is None or value == ""`.
fn is_blank(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.is_empty(),
        Some(_) => false,
    }
}

/// `_options`: the rendered text of every option, in request order (bool:
/// `false` then `true`).
pub(crate) fn option_texts(question: &Question) -> Vec<String> {
    let criteria = question
        .raw
        .get("criteria")
        .filter(|value| !value.is_null());
    match question.kind {
        QuestionKind::Score => match criteria {
            Some(Value::Array(levels)) => levels
                .iter()
                .enumerate()
                .map(|(index, level)| format!("level {index}: {}", render_criterion(level)))
                .collect(),
            _ => Vec::new(),
        },
        QuestionKind::Bool => {
            let given = criteria.and_then(Value::as_object);
            [("false", BOOL_FALSE), ("true", BOOL_TRUE)]
                .iter()
                .map(|&(label, default)| {
                    let value = given.and_then(|map| map.get(label));
                    match value {
                        Some(value) if !is_blank(Some(value)) => {
                            format!("{label}: {}", render_criterion(value))
                        }
                        _ => format!("{label}: {default}"),
                    }
                })
                .collect()
        }
        _ => match criteria {
            Some(Value::Array(items)) => items.iter().map(render_criterion).collect(),
            Some(Value::Object(map)) => map
                .iter()
                .map(|(key, value)| {
                    if is_blank(Some(value)) {
                        key.clone()
                    } else {
                        format!("{key}: {}", render_criterion(value))
                    }
                })
                .collect(),
            _ => Vec::new(),
        },
    }
}

/// The tokenizer and the special tokens a row is framed with.
pub(crate) struct LayaTokenizer {
    tokenizer: tokenizers::Tokenizer,
    /// The mask token's text, replaced by a space in caller text.
    pub(crate) mask_token: String,
    pub(crate) mask_id: u32,
    pub(crate) cls_id: u32,
    pub(crate) sep_id: u32,
    pub(crate) pad_id: u32,
}

impl LayaTokenizer {
    /// Load `tokenizer.json`, reading the special tokens' text from
    /// `tokenizer_config.json` (ModernBERT's `[MASK]`, `[CLS]`, `[SEP]` and
    /// `[PAD]` when it does not name them).
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("tokenizer.json");
        let mut tokenizer = tokenizers::Tokenizer::from_file(&path)
            .map_err(|error| anyhow!("{error}"))
            .with_context(|| format!("loading {}", path.display()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow!("{error}"))?;
        tokenizer.with_padding(None);
        let config_path = model_dir.join("tokenizer_config.json");
        let config: Value = if config_path.is_file() {
            let text = std::fs::read_to_string(&config_path)
                .with_context(|| format!("reading {}", config_path.display()))?;
            serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", config_path.display()))?
        } else {
            Value::Null
        };
        let token = |key: &str, default: &str| {
            config
                .get(key)
                .and_then(|value| value.as_str().or_else(|| value.get("content")?.as_str()))
                .unwrap_or(default)
                .to_string()
        };
        let id = |text: &str| {
            tokenizer
                .token_to_id(text)
                .with_context(|| format!("the laya tokenizer has no `{text}` token"))
        };
        let mask_token = token("mask_token", "[MASK]");
        Ok(Self {
            mask_id: id(&mask_token)?,
            cls_id: id(&token("cls_token", "[CLS]"))?,
            sep_id: id(&token("sep_token", "[SEP]"))?,
            pad_id: id(&token("pad_token", "[PAD]"))?,
            mask_token,
            tokenizer,
        })
    }

    /// `tok(text.replace(mask, " "), add_special_tokens=False)`.
    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        let text = text.replace(&self.mask_token, " ");
        self.tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| DecisionError::Internal(format!("laya tokenizer: {error}")))
    }
}

/// `decision_config` lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Budget {
    /// Longest row (`max_len`).
    pub(crate) max_len: usize,
    /// Tokens the question head and its options share (`head_max_len`).
    pub(crate) head_max_len: usize,
}

/// The instruction text upstream renders: a string as is, anything else as
/// `json.dumps(instruction)` (ASCII-escaped).
pub(crate) fn instruction_text(question: &Question) -> Result<String, DecisionError> {
    match question.raw.get("instructions") {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(other) => Ok(python_dumps(other, true)),
        None => Err(invalid("Laya questions need instructions")),
    }
}

/// The state text upstream encodes: a string as is, anything else as
/// `json.dumps(state, ensure_ascii=False)`.
pub(crate) fn state_text(state: Option<&Value>) -> String {
    match state {
        Some(Value::String(text)) => text.clone(),
        Some(other) => python_dumps(other, false),
        None => "null".to_string(),
    }
}

/// `Laya._sequence`: one row's ids and the marker position of each option.
/// `state` is the already-encoded state (see [`state_text`]).
pub(crate) fn sequence(
    tok: &LayaTokenizer,
    budget: Budget,
    state: &[u32],
    question: &Question,
) -> Result<(Vec<u32>, Vec<usize>), DecisionError> {
    let instruction = instruction_text(question)?;
    let head = tok.encode(&format!(
        "{} question: {instruction}",
        type_token(question.kind)
    ))?;
    let texts = option_texts(question);
    // Every option takes at least its marker after `[CLS]` and `[SEP]`, so
    // `max_len` options or more always end in the budget refusal below:
    // refuse before tokenizing them (the request body alone bounds the
    // label count).
    if texts.len() >= budget.max_len {
        return Err(invalid(BUDGET_ERROR));
    }
    let mut options = Vec::with_capacity(texts.len());
    for text in &texts {
        let mut ids = vec![tok.mask_id];
        let encoded = tok.encode(&format!(" {text}"))?;
        ids.extend(encoded.iter().take(OPTION_TOKENS));
        options.push(ids);
    }
    let head_max = budget.head_max_len as i64;
    let total = |options: &[Vec<u32>]| options.iter().map(Vec::len).sum::<usize>() as i64;
    let mut room = head_max - total(&options);
    if room < 16 && !options.is_empty() {
        let per = (head_max - 16).div_euclid(options.len() as i64).max(4) as usize;
        options.iter_mut().for_each(|ids| ids.truncate(per));
        room = head_max - total(&options);
    }
    let mut ids = vec![tok.cls_id];
    ids.extend(head.iter().take(room.max(8) as usize));
    ids.push(tok.sep_id);
    let mut markers = Vec::with_capacity(options.len());
    for option in &options {
        markers.push(ids.len());
        ids.extend(option);
    }
    ids.push(tok.sep_id);
    let state_room = budget.max_len.saturating_sub(ids.len() + 1);
    ids.extend(state.iter().take(state_room));
    ids.push(tok.sep_id);
    if markers.iter().any(|&position| position >= budget.max_len) {
        return Err(invalid(BUDGET_ERROR));
    }
    ids.truncate(budget.max_len);
    Ok((ids, markers))
}

/// Every question's [`sequence`], refusing a request whose rows need more
/// than `max_tokens` together.
///
/// Each row is at most `max_len` tokens, but nothing upstream of `predict`
/// bounds how many questions a request holds, and every row repeats the
/// state. Building stops at the first row past the limit, so a refused
/// request never tokenizes or holds more than the budget. The refusal is
/// [`DecisionError::InvalidInput`] (a 400).
pub(crate) fn request_rows(
    tok: &LayaTokenizer,
    budget: Budget,
    state: &[u32],
    questions: &[(String, Question)],
    max_tokens: usize,
) -> Result<Vec<Row>, DecisionError> {
    let mut rows = Vec::with_capacity(questions.len().min(max_tokens));
    let mut total = 0usize;
    for (_, question) in questions {
        let row = sequence(tok, budget, state, question)?;
        total = total.saturating_add(row.0.len());
        if total > max_tokens {
            return Err(invalid(format!(
                "The questions need more than {max_tokens} input tokens in total; \
                 send fewer questions or a shorter state"
            )));
        }
        rows.push(row);
    }
    Ok(rows)
}
