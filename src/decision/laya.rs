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

//! Laya (`model_type: laya`, `nativ-community/laya`).
//!
//! A ModernBERT encoder with a two-layer decision head that scores one
//! marker per option, the questions of a request in right-padded batches
//! (one unless the request exceeds [`MAX_BATCH_TOKENS`]). Port of
//! mlx-vlm's `mlx_vlm/models/laya/laya.py`:
//!
//! - `encoder.*` loads through [`ModernBertEncoder`], with the
//!   `rope_parameters` thetas mapped onto `global_rope_theta` /
//!   `local_rope_theta` here (upstream `config.py`) and the MLP GELU set to
//!   the tanh form mlx-vlm's ModernBERT uses (`approx="precise"`).
//! - The head is the shared marker head ([`super::marker_head`]): the type
//!   embedding, two pre-norm layers with a key padding mask, the scorer at
//!   the markers. The action head reads `h[:, 0]` and four features of the
//!   option distribution; its first probability is reported as
//!   `metadata.rl_agent.act_probability`.
//! - Rows are upstream's `_sequence` ([`super::laya_sequence`]); logits are
//!   calibrated per type and option-count bucket.

use std::ops::Range;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};
use serde_json::{Map, Value, json};

use super::decider2_rows::{calibrated_softmax, round_to};
use super::head_layers::{Linear, Weights, gelu};
use super::laya_sequence::{
    Budget, LayaTokenizer, Row, qtype, request_rows, state_text, type_token,
};
use super::marker_head::{
    DecisionHead, MarkerBatch, MarkerScorer, add_type_embedding, clamp_temperature,
    entropy_confidence, option_bucket, sanitize_marker_head,
};
use super::{
    Answer, DecisionError, DecisionMedia, DecisionModel, DecisionResult, DecisionUsage, MediaKind,
    Question, QuestionKind,
};
use crate::embeddings::loader::quantization_params;
use crate::models::modernbert::{
    MlpActivation, ModernBertArgs, ModernBertEncoder, sanitize_modernbert_weights,
};

/// `config.json` `model_type` the loader detects.
pub(crate) const LAYA_MODEL_TYPE: &str = "laya";

const LAYA_TYPES: &[QuestionKind] = &[
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::Bool,
];
const LAYA_MEDIA: &[MediaKind] = &[];

/// Most row tokens one request may score, summed over its questions:
/// Decider-2b's bound (2^20, 2048 full 512-token rows).
///
/// Rows are at most `max_len` tokens each, but the shared validation does
/// not bound the question count, so a large body of short questions would
/// otherwise become one `[questions, max_len]` batch on the decision worker.
pub(crate) const MAX_REQUEST_TOKENS: usize = super::decider2::MAX_REQUEST_TOKENS;

/// Most padded tokens (`rows * longest row`) one encoder forward holds:
/// 128 full 512-token rows. A request over it runs as consecutive batches,
/// which bounds peak activation memory. Padding is masked out of the
/// encoder, the head and the scorer, so a row's result moves only by f16
/// kernel rounding at the other batch shape (`laya_parity_tests`).
pub(crate) const MAX_BATCH_TOKENS: usize = 1 << 16;

/// Consecutive row ranges whose padded size (`rows * longest row`) stays
/// within `max_tokens`; a row longer than that still runs, alone.
pub(crate) fn batch_ranges(rows: &[Row], max_tokens: usize) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    let mut longest = 0;
    for (index, (ids, _)) in rows.iter().enumerate() {
        let widened = longest.max(ids.len());
        if index > start && (index - start + 1).saturating_mul(widened) > max_tokens {
            ranges.push(start..index);
            start = index;
            longest = ids.len();
        } else {
            longest = widened;
        }
    }
    if start < rows.len() {
        ranges.push(start..rows.len());
    }
    ranges
}

fn internal(error: impl std::fmt::Display) -> DecisionError {
    DecisionError::Internal(format!("laya: {error}"))
}

/// `encoder_config` with upstream's `__post_init__` applied: the
/// `rope_parameters` thetas copied onto `global_rope_theta` (full attention)
/// and `local_rope_theta` (sliding attention), falling back to the existing
/// keys and then to 160000 / 10000, also when `rope_parameters` is absent.
pub(crate) fn encoder_config(config: &Value) -> Result<Value> {
    let mut encoder = config
        .get("encoder_config")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| anyhow!("laya config.json has no encoder_config object"))?;
    // Upstream reads `encoder_config.get("rope_parameters", {})`: an absent
    // key still applies the fallbacks, a non-object value skips the mapping.
    let rope = match encoder.get("rope_parameters") {
        None => Some(Map::new()),
        Some(Value::Object(rope)) => Some(rope.clone()),
        Some(_) => None,
    };
    if let Some(rope) = rope {
        for (kind, key, default) in [
            ("full_attention", "global_rope_theta", 160_000.0),
            ("sliding_attention", "local_rope_theta", 10_000.0),
        ] {
            let theta = rope
                .get(kind)
                .and_then(|section| section.get("rope_theta"))
                .and_then(Value::as_f64)
                .or_else(|| encoder.get(key).and_then(Value::as_f64))
                .unwrap_or(default);
            encoder.insert(key.to_string(), json!(theta));
        }
    }
    Ok(Value::Object(encoder))
}

/// The parsed `config.json`.
#[derive(Debug, Clone)]
pub(crate) struct LayaConfig {
    pub(crate) encoder: ModernBertArgs,
    pub(crate) head_layers: usize,
    pub(crate) budget: Budget,
    /// `decision_config.temperature`, indexed by [`qtype`].
    pub(crate) temperature: Vec<Value>,
    /// `decision_config.temperature_by_options`, keyed `"{type}:{bucket}"`.
    pub(crate) temperature_by_options: Map<String, Value>,
}

impl LayaConfig {
    pub(crate) fn from_config(config: &Value) -> Result<Self> {
        let encoder =
            ModernBertArgs::from_config(&encoder_config(config)?).map_err(|e| anyhow!(e))?;
        let decision = config
            .get("decision_config")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow!("laya config.json has no decision_config object"))?;
        let length = |key: &str| {
            decision
                .get(key)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value > 0)
                .ok_or_else(|| anyhow!("laya decision_config needs a positive integer `{key}`"))
        };
        Ok(Self {
            encoder,
            head_layers: config
                .get("head_layers")
                .and_then(Value::as_u64)
                .map_or(2, |value| value as usize),
            budget: Budget {
                max_len: length("max_len")?,
                head_max_len: length("head_max_len")?,
            },
            temperature: decision
                .get("temperature")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            temperature_by_options: decision
                .get("temperature_by_options")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default(),
        })
    }

    /// The clamped temperature of a `kind` question with `count` options.
    pub(crate) fn temperature(&self, kind: QuestionKind, count: usize) -> f32 {
        let key = format!("{}:{}", type_token(kind), option_bucket(count));
        let fallback = self.temperature.get(qtype(kind) as usize);
        clamp_temperature(self.temperature_by_options.get(&key).or(fallback))
    }
}

/// Split the checkpoint into the encoder (prefix stripped, then
/// ModernBERT's sanitize) and the head tensors (upstream `sanitize`).
/// Splitting first matters: ModernBERT's sanitize drops `head.*`.
pub(crate) fn split_weights(raw: WeightMap) -> Result<(WeightMap, WeightMap)> {
    let mut encoder = WeightMap::new();
    let mut head = WeightMap::new();
    for (key, value) in raw {
        match key.strip_prefix("encoder.") {
            Some(rest) => {
                encoder.insert(rest.to_string(), value);
            }
            None => {
                head.insert(key, value);
            }
        }
    }
    Ok((
        sanitize_modernbert_weights(encoder, false),
        sanitize_marker_head(head, &["scorer", "act_head"], "laya")?,
    ))
}

/// The loaded Laya scorer.
pub(crate) struct LayaModel {
    config: LayaConfig,
    encoder: ModernBertEncoder,
    type_emb: UniquePtr<MlxArray>,
    head: DecisionHead,
    scorer: MarkerScorer,
    act_head: [Linear; 2],
    tokenizer: LayaTokenizer,
    /// Padded tokens per forward, [`MAX_BATCH_TOKENS`] outside tests.
    max_batch_tokens: usize,
}

// SAFETY: the decision worker and `mlxcel decide` both build the scorer and
// call `predict` on one thread (the worker runs its loader closure on its own
// thread), so the MLX handles inside never move between threads while in
// use. `DecisionModel` requires `Send` only so the boxed scorer can be
// returned from that loader.
unsafe impl Send for LayaModel {}

impl LayaModel {
    /// Load the checkpoint at `model_dir`.
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let config = crate::embeddings::loader::read_embedding_config(model_dir)?;
        let laya = LayaConfig::from_config(&config)?;
        let raw = mlxcel_core::weights::load_weights_from_dir(model_dir)
            .map_err(|error| anyhow!("loading the Laya weights: {error}"))?;
        let (encoder_weights, head_weights) = split_weights(raw)?;
        let encoder = ModernBertEncoder::from_weights(
            &encoder_weights,
            &laya.encoder,
            quantization_params(&config),
        )
        .map_err(|error| anyhow!("loading the Laya encoder: {error}"))?
        // mlx-vlm's ModernBERT (the encoder upstream Laya builds) runs its
        // MLP with `nn.GELU(approx="precise")`, the tanh form, where HF and
        // mlxcel's embedding and rerank paths use erf. Laya follows upstream:
        // with erf the mixed parity request drifts by 1.2e-2.
        .with_mlp_activation(MlpActivation::MlxGeluApprox);
        drop(encoder_weights);
        let width = encoder.hidden_size();
        let w = Weights {
            map: &head_weights,
            family: "laya",
        };
        let type_emb = w.tensor("type_emb.weight")?;
        if mlxcel_core::array_shape(&type_emb) != [3, width as i32] {
            bail!(
                "laya type_emb.weight has shape {:?}, expected [3, {width}]",
                mlxcel_core::array_shape(&type_emb)
            );
        }
        let head = DecisionHead::from_weights(&w, "head.layers", laya.head_layers, width)?;
        let scorer = MarkerScorer::from_weights(&w, "scorer.layers", width)?;
        let (act_hidden, act_width) = w.linear_shaped("act_head.layers.0", width + 4, None)?;
        let (act_out, _) = w.linear_shaped("act_head.layers.2", act_width, Some(2))?;
        let act_head = [act_hidden, act_out];
        let tokenizer = LayaTokenizer::load(model_dir)
            .with_context(|| format!("loading the Laya tokenizer from {}", model_dir.display()))?;
        Ok(Self {
            config: laya,
            encoder,
            type_emb,
            head,
            scorer,
            act_head,
            tokenizer,
            max_batch_tokens: MAX_BATCH_TOKENS,
        })
    }

    /// Split requests into forwards of at most `tokens` padded tokens.
    #[cfg(test)]
    pub(crate) fn with_max_batch_tokens(mut self, tokens: usize) -> Self {
        self.max_batch_tokens = tokens;
        self
    }

    /// `act_head(concat(h[:, 0], features))`: the four features are the top
    /// probability, its margin over the second, the entropy over
    /// `ln(count)` and `count / 255`, all f32.
    fn act(&self, hidden: &MlxArray, logits: &MlxArray, counts: &[f32]) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden);
        let rows = shape[0];
        let probabilities = mlxcel_core::softmax(logits, -1);
        let count = mlxcel_core::from_slice_f32(counts, &[rows]);
        let floor = mlxcel_core::from_slice_f32(&[1e-9], &[]);
        let plogp = mlxcel_core::multiply(
            &probabilities,
            &mlxcel_core::log(&mlxcel_core::maximum(&probabilities, &floor)),
        );
        let entropy = mlxcel_core::divide(
            &mlxcel_core::negative(&mlxcel_core::sum_axis(&plogp, -1, false)),
            &mlxcel_core::log(&count),
        );
        let sorted = mlxcel_core::sort(&probabilities, -1);
        let options = mlxcel_core::array_shape(&sorted)[1];
        let column = |index: i32| {
            mlxcel_core::squeeze_axis(
                &mlxcel_core::slice(&sorted, &[0, index], &[rows, index + 1]),
                1,
            )
        };
        let top = column(options - 1);
        let second = column(options - 2);
        let features = mlxcel_core::stack_owned(
            &[
                mlxcel_core::copy(&top),
                mlxcel_core::subtract(&top, &second),
                entropy,
                mlxcel_core::divide(&count, &mlxcel_core::from_slice_f32(&[255.0], &[])),
            ],
            -1,
        );
        let pooled = mlxcel_core::squeeze_axis(
            &mlxcel_core::slice(hidden, &[0, 0, 0], &[rows, 1, shape[2]]),
            1,
        );
        let pooled = mlxcel_core::astype(&pooled, dtype::FLOAT32);
        let input = mlxcel_core::concatenate_many(&[&pooled, &features], -1);
        let act = self.act_head[1].forward(&gelu(&self.act_head[0].forward(&input)));
        mlxcel_core::softmax(&mlxcel_core::astype(&act, dtype::FLOAT32), -1)
    }

    /// Logits `[B * M]` and action probabilities `[B * 2]`, row-major.
    fn forward(
        &self,
        batch: &MarkerBatch,
        qtypes: &[i32],
    ) -> Result<(Vec<f32>, Vec<f32>), DecisionError> {
        // `act` reads the top two option probabilities; validation gives
        // every question at least two options, so this guards the slice.
        if batch.options < 2 {
            return Err(internal("a batch needs at least two options per row"));
        }
        let (ids, attention) = batch.arrays();
        let hidden = self.encoder.encode(&ids, &attention).map_err(internal)?;
        let hidden = add_type_embedding(&hidden, &self.type_emb, qtypes);
        let mask = batch.key_padding_mask(mlxcel_core::array_dtype(&hidden));
        let hidden = self.head.forward(&hidden, &mask);
        let logits = self
            .scorer
            .forward(&batch.gather_markers(&hidden), &batch.valid);
        let counts: Vec<f32> = batch
            .valid
            .chunks(batch.options.max(1))
            .map(|row| row.iter().filter(|&&flag| flag).count().max(2) as f32)
            .collect();
        let act = self.act(&hidden, &logits, &counts);
        // `act` depends on `logits`, so evaluating it first materializes
        // both in one pass; the second call only checks for an error.
        for array in [&act, &logits] {
            mlxcel_core::try_eval(array).map_err(|error| internal(format!("forward: {error}")))?;
        }
        Ok((
            mlxcel_core::utils::array_to_vec_f32(&logits),
            mlxcel_core::utils::array_to_vec_f32(&act),
        ))
    }
}

/// Upstream's answer for one question from its calibrated option
/// probabilities `p` (f32 values) and its action probability.
pub(crate) fn laya_answer(
    question: &Question,
    p: &[f64],
    act_probability: f32,
) -> Result<Answer, DecisionError> {
    let round4 = |value: f64| round_to(value, 4);
    let mut metadata = Map::new();
    metadata.insert(
        "rl_agent".to_string(),
        json!({ "act_probability": f64::from(act_probability) }),
    );
    if question.kind == QuestionKind::Bool {
        let probability = p
            .get(1)
            .copied()
            .ok_or_else(|| internal("bool needs two options"))?;
        return Ok(Answer::bool(round4(probability), metadata));
    }
    metadata.insert(
        "confidence".to_string(),
        json!(round4(entropy_confidence(p))),
    );
    let rounded: Vec<f64> = p.iter().copied().map(round4).collect();
    match question.kind {
        QuestionKind::Score => {
            if let Some(Value::Array(levels)) = question.raw.get("criteria") {
                let legend: Map<String, Value> = levels
                    .iter()
                    .enumerate()
                    .map(|(index, level)| (index.to_string(), level.clone()))
                    .collect();
                metadata.insert("legend".to_string(), Value::Object(legend));
            }
            let expected: f32 = p
                .iter()
                .enumerate()
                .map(|(index, &value)| index as f32 * value as f32)
                .sum();
            let mut answer = Answer::score(question, rounded, metadata)?;
            if let Answer::Score { value, .. } = &mut answer {
                *value = round4(f64::from(expected));
            }
            Ok(answer)
        }
        _ => {
            // `argmax` keeps the first maximum of the unrounded values.
            let best = p.iter().enumerate().fold(
                0,
                |best, (index, value)| if *value > p[best] { index } else { best },
            );
            let mut answer = Answer::choice(question, rounded, metadata)?;
            if let Answer::Choice { value, .. } = &mut answer {
                value.clone_from(&question.labels()[best].name);
            }
            Ok(answer)
        }
    }
}

impl DecisionModel for LayaModel {
    fn decision_types(&self) -> &[QuestionKind] {
        LAYA_TYPES
    }

    fn decision_media(&self) -> &[MediaKind] {
        LAYA_MEDIA
    }

    fn predict(
        &mut self,
        state: Option<&Value>,
        questions: &[(String, Question)],
        _media: DecisionMedia,
    ) -> Result<DecisionResult, DecisionError> {
        let state = self.tokenizer.encode(&state_text(state))?;
        let rows = request_rows(
            &self.tokenizer,
            self.config.budget,
            &state,
            questions,
            MAX_REQUEST_TOKENS,
        )?;
        let qtypes: Vec<i32> = questions.iter().map(|(_, q)| qtype(q.kind)).collect();
        let mut answers = Vec::with_capacity(questions.len());
        let mut input_tokens = 0;
        for range in batch_ranges(&rows, self.max_batch_tokens) {
            let chunk = &rows[range.clone()];
            let batch = MarkerBatch::pad(chunk, self.tokenizer.pad_id).map_err(internal)?;
            let (logits, act) = self.forward(&batch, &qtypes[range.clone()])?;
            for (row, ((name, question), (_, markers))) in
                questions[range].iter().zip(chunk).enumerate()
            {
                let count = markers.len();
                let start = row * batch.options;
                let values = logits
                    .get(start..start + count)
                    .ok_or_else(|| internal("the head returned too few logits"))?;
                let p = calibrated_softmax(values, self.config.temperature(question.kind, count));
                let act_probability = act
                    .get(row * 2)
                    .copied()
                    .ok_or_else(|| internal("the action head returned too few values"))?;
                answers.push((name.clone(), laya_answer(question, &p, act_probability)?));
            }
            input_tokens += batch.lengths.iter().sum::<usize>();
        }
        Ok(DecisionResult {
            answers,
            usage: DecisionUsage { input_tokens },
        })
    }
}
