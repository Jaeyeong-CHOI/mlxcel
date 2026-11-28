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
// Derived from mlx-vlm's OXR implementation (MIT).

use super::*;

#[test]
fn bicubic_matches_torch_basis_planes() {
    // torch.nn.functional.interpolate(eye(32).double().view(1,32,1,32),
    // size=(1,n),mode="bicubic",align_corners=False).float()
    let mut rows = include_str!("oxr_bicubic.txt").lines();
    for n in [20, 46] {
        for row in bicubic_matrix(32, n).chunks_exact(32) {
            let mut parts = rows.next().unwrap().split_whitespace();
            let first: usize = parts.next().unwrap().parse().unwrap();
            let mut expected = [0.0; 32];
            for (x, part) in expected[first..].iter_mut().zip(parts) {
                *x = part.parse().unwrap();
            }
            for (a, b) in row.iter().zip(expected) {
                assert!((a - b).abs() <= 1e-6);
            }
        }
    }
    assert!(rows.next().is_none());
}

#[test]
fn rotary_uses_block_coordinates_for_both_axes() {
    let angles = rotary_angles(4, 6, 8);
    let rows = [
        0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3,
    ];
    let cols = [
        0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5,
    ];
    // Rotating [1,1,1,1,0,0,0,0] yields [cos, sin] for the four frequencies.
    let basis = mlxcel_core::from_slice_f32(&[1., 1., 1., 1., 0., 0., 0., 0.], &[1, 1, 8]);
    let basis = mlxcel_core::tile(&basis, &[24, 1, 1]);
    let angles = mlxcel_core::from_slice_f32(&angles, &[24, 4]);
    let rotated = super::super::qwen2_vl::apply_rotary_pos_emb_vision(&basis, &angles);
    let values = mlxcel_core::utils::array_to_vec_f32(&rotated);
    for (values, (r, c)) in values.chunks_exact(8).zip(rows.into_iter().zip(cols)) {
        for (i, angle) in [r as f32, r as f32 * 0.01, c as f32, c as f32 * 0.01]
            .into_iter()
            .enumerate()
        {
            assert!((values[i] - angle.cos()).abs() < 1e-6);
            assert!((values[4 + i] - angle.sin()).abs() < 1e-6);
        }
    }
    // The third patch starts row 1 of the first block; row-major still has row 0.
    assert!((values[2 * 8] - 0.0f32.cos()).abs() > 0.4);
}
