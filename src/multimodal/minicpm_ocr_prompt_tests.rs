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

use super::*;

// Symbolic tokenizer with distinct id, boundary and newline tokens.
fn encode(s: &str, _: bool) -> Vec<i32> {
    let mut out = Vec::new();
    let mut text = s;
    while !text.is_empty() {
        let (token, id) = [
            ("<|image_pad|>", 9),
            ("<image>", 1),
            ("</image>", 2),
            ("<slice>", 3),
            ("</slice>", 4),
            ("<image_id>0</image_id>", 5),
            ("<image_id>1</image_id>", 10),
            ("<|im_start|>", 6),
            ("\n", 7),
        ]
        .into_iter()
        .find(|(t, _)| text.starts_with(t))
        .unwrap_or((&text[..1], 8));
        text = &text[token.len()..];
        out.push(id);
    }
    out
}

#[test]
fn minicpm_ocr_distinct_crop_counts_and_newlines() {
    let page = PageLayout {
        grid: Some((2, 3)),
        counts: vec![1, 2, 3, 4, 5, 6, 7],
    };
    let p = prepare("<|im_start|><|image_pad|>", &[page], true, &mut encode).unwrap();
    assert_eq!(
        p.tokens,
        vec![
            6, 5, 1, 9, 2, 3, 9, 9, 4, 3, 9, 9, 9, 4, 7, 3, 9, 9, 9, 9, 4, 3, 9, 9, 9, 9, 9, 4, 7,
            3, 9, 9, 9, 9, 9, 9, 4, 3, 9, 9, 9, 9, 9, 9, 9, 4
        ]
    );
    for (n, (start, end)) in p.image_bounds.into_iter().enumerate() {
        assert_eq!(end - start, n + 1);
        assert!(p.tokens[start..end].iter().all(|id| *id == 9));
    }
}

#[test]
fn minicpm_ocr_crop_mismatch_and_text_only() {
    let result = prepare(
        "read",
        &[PageLayout {
            grid: Some((2, 3)),
            counts: vec![1],
        }],
        false,
        &mut encode,
    );
    assert!(
        result
            .err()
            .unwrap()
            .contains("image 0 has 1 crops, expected 7")
    );
    assert_eq!(
        prepare("text", &[], true, &mut encode).unwrap().tokens,
        encode("text", true)
    );
}

#[test]
fn minicpm_ocr_two_images_keep_their_own_spans() {
    let pages = [
        PageLayout {
            grid: None,
            counts: vec![2],
        },
        PageLayout {
            grid: Some((1, 2)),
            counts: vec![3, 1, 2],
        },
    ];
    let p = prepare(
        "<|im_start|><|image_pad|><|image_pad|>",
        &pages,
        true,
        &mut encode,
    )
    .unwrap();
    assert_eq!(p.image_slots, 2);
    assert_eq!(
        p.tokens,
        vec![6, 5, 1, 9, 9, 2, 10, 1, 9, 9, 9, 2, 3, 9, 4, 7, 3, 9, 9, 4]
    );
}
