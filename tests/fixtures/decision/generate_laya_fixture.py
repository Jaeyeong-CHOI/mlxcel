#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the Laya sequence and parity fixture from mlx-vlm (#2363).

Out-of-band tooling: nothing in mlxcel's request path runs Python. Run it
under the GPU lock from the repository root, in an environment with
``mlx-vlm`` (0.8.1 or newer) and an MLX backend:

    gpu-lock run --tag laya-fixture -- python \
        tests/fixtures/decision/generate_laya_fixture.py \
        models/laya tests/fixtures/decision/laya_parity.json

The output pins upstream's ``Laya._sequence`` (token ids and marker
positions) for a described choice, a score with a non-string level, a bool
with and without criteria, the option-capping branch, the mask token inside
caller text, a state longer than ``max_len``, non-string instructions and
state, and the budget failure, plus the full ``predict`` answers of the
README request and a mixed request. ``src/decision/laya_tests.rs`` and
``src/decision/laya_parity_tests.rs`` read it.
"""

import json
import sys

import mlx.core as mx

from mlx_vlm.decision import predict
from mlx_vlm.models.laya.laya import Laya, _options
from mlx_vlm.utils import load

model_dir, out_path = sys.argv[1], sys.argv[2]
model, processor = load(model_dir)
laya = Laya(model, processor)

TICKET = "Please refund my duplicate charge"
department = {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {"billing": None, "technical": None, "sales": None},
}
refund = {"type": "bool", "instructions": "Does the customer request a refund?"}
described = {
    "type": "choice",
    "instructions": "Which team owns the follow-up?",
    "criteria": {
        "billing": "Charges, refunds and invoices",
        "technical": "",
        "sales": {"scope": "new plans", "priority": 2},
    },
}
urgency = {
    "type": "score",
    "instructions": "How urgent is the ticket?",
    "criteria": ["not urgent", "somewhat urgent", {"level": "critical", "weight": 1.5}],
}
described_bool = {
    "type": "bool",
    "instructions": "Is the customer angry?",
    "criteria": {"true": "the customer is upset", "false": ""},
}
many = {
    "type": "choice",
    "instructions": "Which product line is involved?",
    "criteria": {
        f"line_{i:02}": f"Product line {i} covering hardware, software, subscriptions, "
        f"warranties, accessories, upgrades and repairs for region {i}"
        for i in range(12)
    },
}
pair = {
    "type": "choice",
    "instructions": "Is this a [MASK] question about money?",
    "criteria": ["yes [MASK]", "no"],
}
json_instruction = {
    "type": "choice",
    "instructions": {"ask": "Which team?", "note": "für Kunden"},
    "criteria": ["billing", "technical"],
}
too_many = {
    "type": "choice",
    "instructions": "Pick one.",
    "criteria": [f"option {i}" for i in range(200)],
}
LONG_STATE = " ".join(["The customer was charged twice for the same order."] * 80)
JSON_STATE = {"ticket": TICKET, "customer": "Jürgen", "amount": 12.5, "items": [1, 2]}


def sequence_case(name, state, question):
    question = {**question, "type": "noul" if question["type"] == "bool" else question["type"]}
    _, option_text = _options(question)
    case = {"name": name, "state": state, "question": question}
    try:
        ids, markers = laya._sequence(state, question, option_text)
        case["ids"] = ids
        case["markers"] = markers
    except ValueError as error:
        case["error"] = str(error)
    return case


sequences = [
    sequence_case("choice_described", TICKET, described),
    sequence_case("score_levels", TICKET, urgency),
    sequence_case("bool_default", TICKET, refund),
    sequence_case("bool_described", TICKET, described_bool),
    sequence_case("choice_capped", TICKET, many),
    sequence_case("mask_in_text", "My [MASK] card was charged twice", pair),
    sequence_case("long_state", LONG_STATE, department),
    sequence_case("json_instruction_state", JSON_STATE, json_instruction),
    sequence_case("budget_failure", TICKET, too_many),
]

requests = [
    {
        "name": "readme",
        "state": TICKET,
        "questions": {"department": department, "refund": refund},
    },
    {
        "name": "mixed",
        "state": JSON_STATE,
        "questions": {
            "team": described,
            "urgency": urgency,
            "money": pair,
            "line": many,
            "angry": described_bool,
        },
    },
]
for request in requests:
    request["expected"] = predict(model, processor, request["state"], request["questions"])

fixture = {
    "generator": "mlx-vlm 0.8.1 mlx_vlm.models.laya (Laya._sequence and mlx_vlm.decision.predict)",
    "backend": str(mx.default_device()),
    "checkpoint": "nativ-community/laya",
    "sequences": sequences,
    "requests": requests,
}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(fixture, f, ensure_ascii=False, indent=1)
    f.write("\n")
print(json.dumps([r["expected"] for r in requests], ensure_ascii=False, indent=1))
