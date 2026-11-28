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

//! Laya parity against mlx-vlm on the published `nativ-community/laya`
//! checkpoint (f16).
//!
//! Loads `models/laya` through the decision loader and answers the two
//! requests of `tests/fixtures/decision/laya_parity.json`: the README
//! request (the department choice and the refund bool) and a mixed request
//! over a JSON state (a described choice, a score with a non-string level, a
//! two-option choice with the mask token in its text, a twelve-option
//! choice in the `11+` bucket and a described bool), five rows of different
//! lengths and option counts in one padded batch. The usage must be exact and
//! every choice label identical. Every probability, value and metadata number
//! (`act_probability` included) must be within 1e-3 of upstream's on both
//! (measured: 4e-4 on the README request, 5e-4 on the mixed one).
//!
//! The mixed request is the sensitive one: long rows, and the `11+` bucket's
//! temperature (0.1, clamped to 0.5) doubles every logit gap. With the erf
//! GELU in the encoder MLP (mlxcel's ModernBERT default) instead of mlx-vlm's
//! tanh `nn.gelu_approx` it drifted by 1.2e-2 on `line`, and with the tanh
//! cube rounded twice in f16 (`x * x * x` instead of `x**3`) by 3.3e-3. On an
//! f32 copy of the checkpoint the two implementations agree to four places on
//! both requests when mlx-vlm runs with `MLX_ENABLE_TF32=0` (the test
//! binaries pin it; the f32 copy is not committed). Skips when the
//! checkpoint is absent (fails under `MLXCEL_REQUIRE_MODELS=1`).
//! Run it alone under the GPU lock:
//!
//! ```sh
//! cargo test --profile test-fast --features cuda --lib decision::laya_parity_tests
//! ```

use serde_json::Value;

use super::laya::LayaModel;
use super::laya_tests::{checkpoint, fixture};
use super::{DecisionMedia, load_decision_model, predict};

/// The acceptance bound on every number of both requests.
const TOLERANCE: f64 = 1e-3;

/// The bound between a request run in one batch and run row by row.
const SPLIT_TOLERANCE: f64 = 5e-3;

/// Assert `actual` matches `expected`; returns the largest number gap.
fn close(actual: &Value, expected: &Value, path: &str, tolerance: f64) -> f64 {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (
                a.as_f64().unwrap_or(f64::NAN),
                e.as_f64().unwrap_or(f64::NAN),
            );
            assert!((a - e).abs() <= tolerance, "{path}: {a} vs upstream {e}");
            (a - e).abs()
        }
        (Value::Object(a), Value::Object(e)) => {
            let keys =
                |map: &serde_json::Map<String, Value>| map.keys().cloned().collect::<Vec<_>>();
            assert_eq!(keys(a), keys(e), "{path}: keys");
            e.iter()
                .map(|(key, value)| close(&a[key], value, &format!("{path}.{key}"), tolerance))
                .fold(0.0, f64::max)
        }
        _ => {
            assert_eq!(actual, expected, "{path}");
            0.0
        }
    }
}

#[test]
fn laya_matches_mlx_vlm_on_the_published_checkpoint() {
    let Some(dir) = checkpoint(true) else {
        return;
    };
    let mut loaded = load_decision_model(&dir).expect("laya loads");
    assert_eq!(loaded.model_type, "laya");
    for request in fixture()["requests"].as_array().expect("requests") {
        let name = request["name"].as_str().unwrap_or("request");
        let result = predict(
            loaded.model.as_mut(),
            Some(&request["state"]),
            &request["questions"],
            DecisionMedia::default(),
        )
        .unwrap_or_else(|error| panic!("{name}: {error}"));
        let response = result.to_response_json("models/laya");
        let expected = &request["expected"];
        assert_eq!(response["usage"], expected["usage"], "{name}: usage");
        eprintln!("{name}: {}", response["answers"]);
        let gap = close(&response["answers"], &expected["answers"], name, TOLERANCE);
        eprintln!("{name}: largest gap to mlx-vlm {gap:.6} (tolerance {TOLERANCE})");
    }
}

/// A request split into one forward per row (the path a request over
/// `MAX_BATCH_TOKENS` takes) keeps the usage and every label of the single
/// padded batch, and every number within [`SPLIT_TOLERANCE`] of it. Padding
/// is masked out of every layer, so the rows are the same computation; only
/// f16 kernel rounding at the other batch shapes moves the numbers (measured
/// 1e-3 on the README request, 2.7e-3 on the mixed score's expected value,
/// which sums level index times probability).
#[test]
fn laya_split_batches_answer_like_one_batch() {
    let Some(dir) = checkpoint(true) else {
        return;
    };
    let mut whole = LayaModel::load(&dir).expect("laya loads");
    let mut split = LayaModel::load(&dir)
        .expect("laya loads")
        .with_max_batch_tokens(1);
    for request in fixture()["requests"].as_array().expect("requests") {
        let name = request["name"].as_str().unwrap_or("request");
        let run = |model: &mut LayaModel| {
            predict(
                model,
                Some(&request["state"]),
                &request["questions"],
                DecisionMedia::default(),
            )
            .unwrap_or_else(|error| panic!("{name}: {error}"))
            .to_response_json("models/laya")
        };
        let (one, rows) = (run(&mut whole), run(&mut split));
        assert_eq!(rows["usage"], one["usage"], "{name}: usage");
        let gap = close(&rows["answers"], &one["answers"], name, SPLIT_TOLERANCE);
        eprintln!("{name}: split vs one batch {gap:.6} (tolerance {SPLIT_TOLERANCE})");
    }
}
