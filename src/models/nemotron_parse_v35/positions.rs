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

//! Content-normalized RADIO patch positions, derived from NVIDIA's
//! mlx-vlm nemotron_parse_v3_5/vision.py (OpenMDW-1.1).
use mlxcel_core::{MlxArray, UniquePtr};

/// Compose the antialiased resize with the border-clamped sample at each
/// patch's content midpoint. Both maps are separable in the page axes.
fn axis(source: usize, count: usize, content: usize, area: bool) -> Vec<f32> {
    let target = content.div_ceil(16);
    let scale = source as f32 / target as f32;
    let mut resize = vec![0.; target * source];
    for (i, row) in resize.chunks_exact_mut(source).enumerate() {
        if area {
            let start = i * source / target;
            let end = ((i + 1) * source).div_ceil(target);
            row[start..end].fill(1. / (end - start) as f32);
        } else {
            let center = (i as f32 + 0.5) * scale - 0.5;
            for (j, w) in row.iter_mut().enumerate() {
                *w = (1. - (center - j as f32).abs() / scale.max(1.)).max(0.);
            }
            let sum: f32 = row.iter().sum();
            for w in row {
                *w /= sum;
            }
        }
    }
    let mut out = vec![0.; count * source];
    for (i, row) in out.chunks_exact_mut(source).enumerate() {
        let start = (i * 16).min(content);
        let extent = (content - start).min(16);
        let coord = (((start as f32 + 0.5 * extent as f32) / content as f32) * target as f32 - 0.5)
            .clamp(0., (target - 1) as f32);
        let lo = coord.floor() as usize;
        let hi = (lo + 1).min(target - 1);
        for (j, w) in row.iter_mut().enumerate() {
            let a = resize[lo * source + j];
            *w = a + (resize[hi * source + j] - a) * coord.fract();
        }
    }
    out
}

pub(crate) fn content_normalized_pos_embed(
    grid: &MlxArray,
    rows: i32,
    cols: i32,
    ch: i32,
    cw: i32,
) -> Result<UniquePtr<MlxArray>, String> {
    if !(1..=128).contains(&rows)
        || !(1..=104).contains(&cols)
        || !(1..=rows * 16).contains(&ch)
        || !(1..=cols * 16).contains(&cw)
    {
        return Err(format!(
            "Invalid packed page geometry: grid {rows}x{cols}, content {ch}x{cw}"
        ));
    }
    let s = mlxcel_core::array_shape(grid);
    let &[g, w, d] = s.as_slice() else {
        return Err(format!("Invalid RADIO position grid shape {s:?}"));
    };
    if g != w || !(1..=128).contains(&g) || d <= 0 {
        return Err(format!("Invalid RADIO position grid shape {s:?}"));
    }
    let width = g
        .checked_mul(d)
        .ok_or("RADIO position grid width overflow")?;
    let area = ch <= 16 || cw <= 16;
    let matrix = |n: i32, c: i32| {
        mlxcel_core::from_slice_f32(&axis(g as usize, n as usize, c as usize, area), &[n, g])
    };
    let flat = mlxcel_core::reshape(
        &mlxcel_core::astype(grid, mlxcel_core::dtype::FLOAT32),
        &[g, width],
    );
    let y = mlxcel_core::reshape(
        &mlxcel_core::matmul(&matrix(rows, ch), &flat),
        &[rows, g, d],
    );
    let xy = mlxcel_core::matmul(&matrix(cols, cw), &y);
    Ok(mlxcel_core::reshape(&xy, &[rows * cols, d]))
}
