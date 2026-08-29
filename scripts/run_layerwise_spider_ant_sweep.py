#!/usr/bin/env python3
"""Run a calibrated layerwise spider-to-ant transcoder intervention sweep.

The script reuses frozen Neuronpedia candidate manifests, selects only naturally
active features, and edits Gemma's transcoder reconstruction at positions 16 and
21.  It screens one layer at a time and only tries greedy multi-layer
combinations if no single layer makes ``Six`` the top next token.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ID = "google/gemma-3-4b-it"
TRANSCODER_ROOT = (
    "https://huggingface.co/mwhanna/gemma-scope-2-4b-it/resolve/main/"
    "transcoder_all/width_262k_l0_small_affine"
)
SPIDER_PROMPT = (
    "<bos><start_of_turn>user\n"
    "Answer in one word: How many legs does a web-spinning animal have?"
    "<end_of_turn>\n<start_of_turn>model\nAnswer:"
)
IMPLICIT_ANT_QUESTION = "How many legs does a colony-building insect have?"
EXPLICIT_ANT_QUESTION = "How many legs does an ant have?"
PROMPT_TEMPLATE = (
    "<bos><start_of_turn>user\nAnswer in one word: {question}"
    "<end_of_turn>\n<start_of_turn>model\nAnswer:"
)
TARGET_POSITIONS = (16, 21)
OUTPUT_JSON = "layerwise_spider_ant_sweep.json"
OUTPUT_CSV = "layerwise_spider_ant_sweep.csv"
FEATURE_MANIFEST = "layerwise_feature_manifest.json"
OVERVIEW_FIGURE = "layerwise_sweep_overview.png"
DETAIL_FIGURE = "layerwise_sweep_best_detail.png"


@dataclass(frozen=True)
class FeatureKey:
    layer: int
    feature: int


def comma_floats(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected comma-separated numbers: {value}") from error
    if not result:
        raise argparse.ArgumentTypeError("The strength grid cannot be empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrated layerwise spider-to-ant transcoder intervention sweep"
    )
    parser.add_argument(
        "--spider-manifest",
        type=Path,
        default=Path("results/spider_layerwise_search/spider_layerwise_search.json"),
    )
    parser.add_argument(
        "--ant-manifest",
        type=Path,
        default=Path("results/ant_layerwise_search/ant_layerwise_search.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/layerwise_spider_ant_sweep"),
    )
    parser.add_argument("--cutoff", type=float, default=0.4)
    parser.add_argument(
        "--spider-suppressions",
        type=comma_floats,
        default=comma_floats("0,0.2,0.4,0.6,0.8,1.0"),
    )
    parser.add_argument(
        "--ant-factors",
        type=comma_floats,
        default=comma_floats("0,0.25,0.5,0.75,1.0,1.25,1.5"),
    )
    parser.add_argument(
        "--overdrive-factors",
        type=comma_floats,
        default=comma_floats("2.0,3.0"),
    )
    parser.add_argument(
        "--disable-overdrive",
        action="store_true",
        help="Do not extend the ant grid if the standard grid has no success.",
    )
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--implicit-ant-question", default=IMPLICIT_ANT_QUESTION)
    parser.add_argument("--explicit-ant-question", default=EXPLICIT_ANT_QUESTION)
    parser.add_argument("--min-ant-p-six", type=float, default=0.9)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def request_with_retries(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 120,
    attempts: int = 5,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def fetch_range(url: str, start: int, end: int) -> bytes:
    response = request_with_retries(
        url, headers={"Range": f"bytes={start}-{end}"}, timeout=180
    )
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"Server returned {len(response.content)} bytes for a {expected}-byte range"
        )
    return response.content


def tensor_dtype(metadata: dict[str, Any]) -> np.dtype:
    if metadata["dtype"] != "F32":
        raise RuntimeError(f"Expected F32 transcoder tensor, found {metadata['dtype']}")
    return np.dtype("<f4")


def load_layer_rows(
    layer: int, feature_ids: Iterable[int]
) -> dict[FeatureKey, dict[str, Any]]:
    """Range-load encoder/decoder rows plus biases and learned thresholds."""
    feature_ids = sorted(set(feature_ids))
    url = f"{TRANSCODER_ROOT}/layer_{layer}.safetensors?download=true"
    header_size = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    header = json.loads(fetch_range(url, 8, 8 + header_size - 1))
    data_base = 8 + header_size

    def read_tensor(name: str) -> np.ndarray:
        metadata = header[name]
        dtype = tensor_dtype(metadata)
        start, end = metadata["data_offsets"]
        raw = fetch_range(url, data_base + start, data_base + end - 1)
        return np.frombuffer(raw, dtype=dtype).reshape(metadata["shape"]).copy()

    b_enc = read_tensor("b_enc")
    thresholds = read_tensor("activation_function.threshold")
    enc_meta = header["W_enc"]
    dec_meta = header["W_dec"]
    enc_dtype = tensor_dtype(enc_meta)
    dec_dtype = tensor_dtype(dec_meta)
    feature_count, width = enc_meta["shape"]
    if dec_meta["shape"] != [feature_count, width]:
        raise RuntimeError(
            f"Unexpected W_dec shape at L{layer}: {dec_meta['shape']}"
        )
    enc_base = data_base + enc_meta["data_offsets"][0]
    dec_base = data_base + dec_meta["data_offsets"][0]
    enc_row_bytes = width * enc_dtype.itemsize
    dec_row_bytes = width * dec_dtype.itemsize

    rows: dict[FeatureKey, dict[str, Any]] = {}
    for feature in feature_ids:
        if not 0 <= feature < feature_count:
            raise RuntimeError(f"Feature {feature} is outside L{layer}'s feature range")
        enc_start = enc_base + feature * enc_row_bytes
        dec_start = dec_base + feature * dec_row_bytes
        enc_raw = fetch_range(url, enc_start, enc_start + enc_row_bytes - 1)
        dec_raw = fetch_range(url, dec_start, dec_start + dec_row_bytes - 1)
        rows[FeatureKey(layer, feature)] = {
            "W_enc": np.frombuffer(enc_raw, dtype=enc_dtype).copy(),
            "W_dec": np.frombuffer(dec_raw, dtype=dec_dtype).copy(),
            "b_enc": float(b_enc[feature]),
            "threshold": float(thresholds[feature]),
        }
    return rows


def load_all_rows(
    features_by_layer: dict[int, set[int]], workers: int
) -> dict[FeatureKey, dict[str, Any]]:
    rows: dict[FeatureKey, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(load_layer_rows, layer, feature_ids): layer
            for layer, feature_ids in features_by_layer.items()
        }
        for future in as_completed(futures):
            layer = futures[future]
            layer_rows = future.result()
            rows.update(layer_rows)
            print(f"Loaded {len(layer_rows)} encoder/decoder row pairs from L{layer}")
    return rows


def find_text_layers(model: torch.nn.Module):
    candidates = [model, getattr(model, "model", None)]
    for candidate in list(candidates):
        if candidate is not None:
            candidates.append(getattr(candidate, "language_model", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise RuntimeError("Could not locate Gemma text decoder layers")


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing feature manifest: {path}")
    return json.loads(path.read_text())


def select_spider_instances(
    manifest: dict[str, Any], cutoff: float
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for feature in manifest["features"]:
        max_act = feature.get("max_act_approx")
        if max_act is None or max_act <= 0:
            continue
        for position in TARGET_POSITIONS:
            activation = float(feature["activations"][position])
            normalized = activation / float(max_act)
            if activation > 0 and normalized >= cutoff:
                selected.append(
                    {
                        "role": "spider",
                        "layer": int(feature["layer"]),
                        "feature": int(feature["feature"]),
                        "position": position,
                        "token": manifest["tokens"][position]["token"],
                        "description": feature["description"],
                        "clean_activation": activation,
                        "max_act_approx": float(max_act),
                        "normalized_activation": normalized,
                        "learned_threshold": float(feature["threshold"]),
                    }
                )
    if not selected:
        raise RuntimeError(f"No spider instances passed the normalized cutoff {cutoff}")
    return sorted(selected, key=lambda item: (item["layer"], item["feature"], item["position"]))


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise RuntimeError(f"Expected {text!r} to be one token, found IDs {ids}")
    return int(ids[0])


def token_rows(tokenizer: Any, input_ids: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {
            "position": position,
            "token_id": int(token_id),
            "token": tokenizer.decode([int(token_id)]),
        }
        for position, token_id in enumerate(input_ids[0])
    ]


def display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t") or "<empty>"


def probability_record(
    logits: torch.Tensor,
    tokenizer: Any,
    target_ids: dict[str, int],
    top_k: int,
) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=top_k)
    top_tokens = [
        {
            "token": tokenizer.decode([int(token_id)]),
            "token_id": int(token_id),
            "probability": float(probability),
        }
        for probability, token_id in zip(
            top_probabilities.detach().cpu(), top_ids.detach().cpu(), strict=True
        )
    ]
    p_six = float(probabilities[target_ids["Six"]])
    p_eight = float(probabilities[target_ids["Eight"]])
    p_four = float(probabilities[target_ids["Four"]])
    return {
        "p_six": p_six,
        "p_eight": p_eight,
        "p_four": p_four,
        "log_p_six_minus_log_p_eight": math.log(max(p_six, 1e-45))
        - math.log(max(p_eight, 1e-45)),
        "top_token": top_tokens[0]["token"],
        "top_token_id": top_tokens[0]["token_id"],
        "top_probability": top_tokens[0]["probability"],
        "top_tokens": top_tokens,
    }


def capture_hidden_and_logits(
    model: torch.nn.Module,
    layers: Any,
    input_ids: torch.Tensor,
    capture_layers: Iterable[int],
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in sorted(set(capture_layers)):
        def capture(_module, _args, output, layer_index=layer_index):
            captured[layer_index] = output.detach()

        handles.append(
            layers[layer_index].pre_feedforward_layernorm.register_forward_hook(capture)
        )
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(capture_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Failed to capture pre-feedforward activations at layers {missing}")
    return captured, logits


def feature_activation(
    hidden: torch.Tensor,
    position: int,
    row: dict[str, Any],
    device: torch.device,
) -> tuple[float, float]:
    encoder = torch.from_numpy(row["W_enc"]).to(device=device, dtype=torch.float32)
    preactivation = float(
        torch.dot(hidden[0, position].float(), encoder).item() + row["b_enc"]
    )
    activation = preactivation if preactivation > row["threshold"] else 0.0
    return preactivation, activation


def find_explicit_ant_position(rows: list[dict[str, Any]]) -> int:
    exact = [
        row["position"]
        for row in rows
        if row["token"].strip().lower() == "ant"
    ]
    if len(exact) != 1:
        raise RuntimeError(f"Expected exactly one explicit ant token, found {exact}")
    return exact[0]


def find_turn_boundary_position(rows: list[dict[str, Any]]) -> int:
    """Return the newline immediately following the end-of-turn control token."""
    for index, row in enumerate(rows[:-1]):
        if "end_of_turn" in row["token"] and rows[index + 1]["token"] == "\n":
            return index + 1
    newline_positions = [
        row["position"] for row in rows if row["token"] == "\n"
    ]
    if len(newline_positions) < 2:
        raise RuntimeError("Could not locate the answer turn-boundary newline")
    return newline_positions[-2]


def calibrate_ant_reference(
    *,
    model: torch.nn.Module,
    layers: Any,
    tokenizer: Any,
    device: torch.device,
    ant_candidates: list[dict[str, Any]],
    rows: dict[FeatureKey, dict[str, Any]],
    target_ids: dict[str, int],
    implicit_question: str,
    explicit_question: str,
    minimum_p_six: float,
    cutoff: float,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_layers = sorted({int(item["layer"]) for item in ant_candidates})
    attempts: list[dict[str, Any]] = []

    for kind, question in [
        ("implicit", implicit_question),
        ("explicit_fallback", explicit_question),
    ]:
        prompt = PROMPT_TEMPLATE.format(question=question)
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        reference_tokens = token_rows(tokenizer, ids)
        if kind == "implicit":
            if len(reference_tokens) <= max(TARGET_POSITIONS):
                alignment_error = (
                    f"Implicit reference has only {len(reference_tokens)} tokens; "
                    "positions 16 and 21 are unavailable"
                )
                attempts.append(
                    {"kind": kind, "prompt": prompt, "tokens": reference_tokens, "error": alignment_error}
                )
                continue
            source_positions = {16: 16, 21: 21}
        else:
            try:
                source_positions = {
                    16: find_explicit_ant_position(reference_tokens),
                    21: find_turn_boundary_position(reference_tokens),
                }
            except RuntimeError as error:
                attempts.append(
                    {"kind": kind, "prompt": prompt, "tokens": reference_tokens, "error": str(error)}
                )
                continue

        hidden, logits = capture_hidden_and_logits(
            model, layers, ids.to(device), candidate_layers
        )
        metrics = probability_record(logits, tokenizer, target_ids, top_k)
        attempt = {
            "kind": kind,
            "prompt": prompt,
            "tokens": reference_tokens,
            "source_to_spider_position": {
                str(source_position): target_position
                for target_position, source_position in source_positions.items()
            },
            "mapped_positions": {
                str(target_position): {
                    "reference_position": source_position,
                    "reference_token": reference_tokens[source_position]["token"],
                    "spider_position": target_position,
                }
                for target_position, source_position in source_positions.items()
            },
            "metrics": metrics,
        }
        attempts.append(attempt)
        valid_answer = (
            metrics["top_token"].strip().lower() == "six"
            and metrics["p_six"] >= minimum_p_six
        )
        if not valid_answer:
            continue

        eligible: list[dict[str, Any]] = []
        for candidate in ant_candidates:
            key = FeatureKey(int(candidate["layer"]), int(candidate["feature"]))
            max_act = candidate.get("max_act_approx")
            if max_act is None or max_act <= 0:
                continue
            for target_position, source_position in source_positions.items():
                preactivation, activation = feature_activation(
                    hidden[key.layer], source_position, rows[key], device
                )
                normalized = activation / float(max_act)
                if activation > 0 and normalized >= cutoff:
                    eligible.append(
                        {
                            "role": "ant",
                            "layer": key.layer,
                            "feature": key.feature,
                            "position": target_position,
                            "reference_position": source_position,
                            "reference_token": reference_tokens[source_position]["token"],
                            "description": candidate["description"],
                            "forced": bool(candidate.get("forced", False)),
                            "reference_preactivation": preactivation,
                            "reference_activation": activation,
                            "max_act_approx": float(max_act),
                            "normalized_reference_activation": normalized,
                            "learned_threshold": float(rows[key]["threshold"]),
                        }
                    )
        attempt["eligible_ant_instance_count"] = len(eligible)
        if eligible:
            return {"selected": attempt, "attempts": attempts}, sorted(
                eligible,
                key=lambda item: (item["layer"], item["feature"], item["position"]),
            )
        # A correct Six answer is necessary but not sufficient calibration: the
        # reference also has to supply at least one naturally active, above-cutoff
        # feature target.  Otherwise the explicit ant-token fallback is the only
        # plan-compliant source of a calibrated injection magnitude.
        attempt["error"] = (
            "Reference answered Six but no saved same-layer ant feature passed "
            f"the normalized activation cutoff {cutoff:g} at the mapped positions"
        )

    raise RuntimeError(
        "Ant calibration failed: neither reference prompt both answered Six with "
        f"probability at least {minimum_p_six:.1%} and supplied an above-cutoff "
        "saved ant feature. Attempts: "
        + json.dumps(
            [
                {
                    "kind": attempt["kind"],
                    "error": attempt.get("error"),
                    "top_token": attempt.get("metrics", {}).get("top_token"),
                    "p_six": attempt.get("metrics", {}).get("p_six"),
                }
                for attempt in attempts
            ]
        )
    )


def group_by_layer(instances: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[int(instance["layer"])].append(instance)
    return dict(grouped)


def measure_current_activations(
    *,
    spider_hidden: dict[int, torch.Tensor],
    instances: list[dict[str, Any]],
    rows: dict[FeatureKey, dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    measured: list[dict[str, Any]] = []
    for instance in instances:
        key = FeatureKey(instance["layer"], instance["feature"])
        preactivation, activation = feature_activation(
            spider_hidden[key.layer], instance["position"], rows[key], device
        )
        measured.append(
            {
                **instance,
                "clean_spider_prompt_preactivation": preactivation,
                "clean_spider_prompt_activation": activation,
            }
        )
    return measured


def run_intervention(
    *,
    model: torch.nn.Module,
    layers: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    target_ids: dict[str, int],
    rows: dict[FeatureKey, dict[str, Any]],
    spider_by_layer: dict[int, list[dict[str, Any]]],
    ant_by_layer: dict[int, list[dict[str, Any]]],
    tested_layers: tuple[int, ...],
    spider_suppression: float,
    ant_factor: float,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    state: dict[int, torch.Tensor] = {}
    operations: list[dict[str, Any]] = []
    handles = []

    for layer_index in tested_layers:
        def capture(_module, _args, output, layer_index=layer_index):
            state[layer_index] = output

        def alter(_module, _args, output, layer_index=layer_index):
            if layer_index not in state:
                raise RuntimeError(f"L{layer_index} output hook ran before its input hook")
            changed = output.clone()
            layer_delta_by_position: dict[int, torch.Tensor] = {}

            for instance in spider_by_layer.get(layer_index, []):
                key = FeatureKey(layer_index, instance["feature"])
                row = rows[key]
                current_preactivation, current_activation = feature_activation(
                    state[layer_index], instance["position"], row, device
                )
                target_activation = instance["clean_activation"] * (1.0 - spider_suppression)
                target_activation = max(0.0, target_activation)
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target_activation - current_activation) * decoder
                position = instance["position"]
                layer_delta_by_position[position] = (
                    layer_delta_by_position.get(position, torch.zeros_like(delta)) + delta
                )
                operations.append(
                    {
                        "role": "spider",
                        "layer": layer_index,
                        "feature": instance["feature"],
                        "position": position,
                        "current_preactivation": current_preactivation,
                        "current_activation": current_activation,
                        "reference_activation": instance["clean_activation"],
                        "target_activation": target_activation,
                        "delta_norm": float(delta.norm().item()),
                    }
                )

            for instance in ant_by_layer.get(layer_index, []):
                key = FeatureKey(layer_index, instance["feature"])
                row = rows[key]
                current_preactivation, current_activation = feature_activation(
                    state[layer_index], instance["position"], row, device
                )
                target_activation = (
                    current_activation + ant_factor * instance["reference_activation"]
                )
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target_activation - current_activation) * decoder
                position = instance["position"]
                layer_delta_by_position[position] = (
                    layer_delta_by_position.get(position, torch.zeros_like(delta)) + delta
                )
                operations.append(
                    {
                        "role": "ant",
                        "layer": layer_index,
                        "feature": instance["feature"],
                        "position": position,
                        "current_preactivation": current_preactivation,
                        "current_activation": current_activation,
                        "reference_activation": instance["reference_activation"],
                        "target_activation": target_activation,
                        "delta_norm": float(delta.norm().item()),
                    }
                )

            for position, delta in layer_delta_by_position.items():
                changed[0, position] = changed[0, position] + delta.to(output.dtype)
            return changed

        handles.append(
            layers[layer_index].pre_feedforward_layernorm.register_forward_hook(capture)
        )
        handles.append(
            layers[layer_index].post_feedforward_layernorm.register_forward_hook(alter)
        )

    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()

    metrics = probability_record(logits, tokenizer, target_ids, top_k)
    success = (
        metrics["top_token"].strip().lower() == "six"
        and metrics["p_six"] > metrics["p_eight"]
    )
    total_delta_norm = math.sqrt(
        sum(operation["delta_norm"] ** 2 for operation in operations)
    )
    return {
        "tested_layers": list(tested_layers),
        "scope": "+".join(f"L{layer}" for layer in tested_layers),
        "spider_suppression": spider_suppression,
        "ant_factor": ant_factor,
        "overdrive": ant_factor > 1.5,
        "control": (
            "unmodified"
            if spider_suppression == 0 and ant_factor == 0
            else "spider_only"
            if spider_suppression > 0 and ant_factor == 0
            else "ant_only"
            if spider_suppression == 0 and ant_factor > 0
            else "joint"
        ),
        "success": success,
        "total_delta_norm": total_delta_norm,
        "operations": operations,
        **metrics,
    }


def run_grid(
    *,
    tested_layers: tuple[int, ...],
    spider_suppressions: list[float],
    ant_factors: list[float],
    common: dict[str, Any],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for suppression in spider_suppressions:
        if not 0 <= suppression <= 1:
            raise RuntimeError(f"Spider suppression must be in [0, 1], got {suppression}")
        for ant_factor in ant_factors:
            if ant_factor < 0:
                raise RuntimeError(f"Ant factor cannot be negative, got {ant_factor}")
            result = run_intervention(
                tested_layers=tested_layers,
                spider_suppression=suppression,
                ant_factor=ant_factor,
                **common,
            )
            if suppression == 0 and ant_factor == 0:
                for key in ["p_six", "p_eight", "p_four"]:
                    if not math.isclose(result[key], baseline[key], rel_tol=1e-6, abs_tol=1e-9):
                        raise RuntimeError(
                            f"{result['scope']} (0,0) differs from baseline for {key}: "
                            f"{result[key]} vs {baseline[key]}"
                        )
            results.append(result)
            print(
                f"{result['scope']:>12} spider={suppression:>4.2f} ant={ant_factor:>4.2f} "
                f"P(Six)={result['p_six']:.4%} P(Eight)={result['p_eight']:.4%} "
                f"top={display_token(result['top_token'])!r}"
            )
    return results


def best_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [run for run in runs if run["success"]]
    if successes:
        return min(
            successes,
            key=lambda run: (
                len(run["tested_layers"]),
                math.hypot(run["spider_suppression"], run["ant_factor"]),
                run["total_delta_norm"],
            ),
        )
    return max(runs, key=lambda run: run["log_p_six_minus_log_p_eight"])


def make_probability_grid(
    runs: list[dict[str, Any]],
    suppressions: list[float],
    factors: list[float],
    metric: str,
) -> np.ndarray:
    lookup = {
        (run["spider_suppression"], run["ant_factor"]): run[metric]
        for run in runs
    }
    return np.asarray(
        [[100 * lookup[suppression, factor] for factor in factors] for suppression in suppressions],
        dtype=np.float64,
    )


def annotate_heatmap(axis: Any, values: np.ndarray) -> None:
    cutoff = (float(np.nanmin(values)) + float(np.nanmax(values))) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = f"{value:.2g}" if value < 1 else f"{value:.1f}"
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color="white" if value > cutoff else "black",
            )


def plot_layer_heatmaps(
    *,
    runs: list[dict[str, Any]],
    suppressions: list[float],
    factors: list[float],
    scope: str,
    output_path: Path,
) -> None:
    p_six = make_probability_grid(runs, suppressions, factors, "p_six")
    p_eight = make_probability_grid(runs, suppressions, factors, "p_eight")
    figure, axes = plt.subplots(1, 2, figsize=(15, 6.7), constrained_layout=True)
    for axis, values, title, cmap in [
        (axes[0], p_six, "P(Six), %", "viridis"),
        (axes[1], p_eight, "P(Eight), %", "magma"),
    ]:
        image = axis.imshow(values, origin="lower", aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(factors)), [f"{value:g}" for value in factors])
        axis.set_yticks(
            range(len(suppressions)), [f"{value:g}" for value in suppressions]
        )
        axis.set_xlabel("ant reference factor")
        axis.set_ylabel("spider suppression fraction")
        axis.set_title(title, loc="left")
        annotate_heatmap(axis, values)
        figure.colorbar(image, ax=axis, label="probability (%)", shrink=0.86)
    figure.suptitle(f"{scope} isolated spider-to-ant intervention", fontsize=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_overview(
    *,
    layer_runs: dict[int, list[dict[str, Any]]],
    baseline: dict[str, Any],
    output_path: Path,
) -> None:
    layers = sorted(layer_runs)
    best = [best_run(layer_runs[layer]) for layer in layers]
    y = np.arange(len(layers))
    baseline_log_odds = baseline["log_p_six_minus_log_p_eight"]
    improvements = [
        run["log_p_six_minus_log_p_eight"] - baseline_log_odds for run in best
    ]
    p_six_percent = [100 * run["p_six"] for run in best]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(6, len(layers) * 0.72)),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.25]},
    )

    shift_axis = axes[0]
    shift_axis.barh(y, improvements, color="#2563eb")
    shift_axis.set_yticks(y, [f"L{layer}" for layer in layers])
    shift_axis.invert_yaxis()
    shift_axis.axvline(0, color="#111827", linewidth=0.8)
    shift_axis.set_xlabel("improvement in log P(Six) - log P(Eight)")
    shift_axis.set_title("Best log-odds shift", loc="left")
    shift_axis.grid(axis="x", alpha=0.25)
    for index, improvement in enumerate(improvements):
        shift_axis.text(
            improvement + 0.08,
            index,
            f"{improvement:+.2f}",
            va="center",
            fontsize=8,
        )

    probability_axis = axes[1]
    probability_axis.barh(y, p_six_percent, color="#0f766e")
    probability_axis.set_yticks(y, [f"L{layer}" for layer in layers])
    probability_axis.invert_yaxis()
    probability_axis.set_xscale("log")
    probability_axis.set_xlabel("best P(Six), % (log scale)")
    probability_axis.set_title("Best isolated P(Six), with paired P(Eight)", loc="left")
    probability_axis.grid(axis="x", which="both", alpha=0.25)
    probability_axis.axvline(
        100 * baseline["p_six"], color="#0f766e", linestyle="--", alpha=0.7
    )
    for index, run in enumerate(best):
        probability_axis.text(
            100 * run["p_six"] * 1.16,
            index,
            f"P6={run['p_six']:.4%}; P8={run['p_eight']:.4%}; "
            f"s={run['spider_suppression']:g}, a={run['ant_factor']:g}",
            va="center",
            fontsize=8,
        )
    figure.suptitle(
        "Best probability shift from each isolated eligible layer",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_best_detail(
    *,
    runs: list[dict[str, Any]],
    suppressions: list[float],
    factors: list[float],
    selected: dict[str, Any],
    output_path: Path,
) -> None:
    p_six = make_probability_grid(runs, suppressions, factors, "p_six")
    p_eight = make_probability_grid(runs, suppressions, factors, "p_eight")
    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1.08, 0.92])
    axes = [figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])]
    for axis, values, title, cmap in [
        (axes[0], p_six, "P(Six), %", "viridis"),
        (axes[1], p_eight, "P(Eight), %", "magma"),
    ]:
        image = axis.imshow(values, origin="lower", aspect="auto", cmap=cmap)
        axis.set_xticks(range(len(factors)), [f"{value:g}" for value in factors])
        axis.set_yticks(range(len(suppressions)), [f"{value:g}" for value in suppressions])
        axis.set_xlabel("ant reference factor")
        axis.set_ylabel("spider suppression fraction")
        axis.set_title(title, loc="left")
        annotate_heatmap(axis, values)
        figure.colorbar(image, ax=axis, label="probability (%)", shrink=0.83)

    curve_axis = figure.add_subplot(grid[1, :])
    selected_suppression = selected["spider_suppression"]
    curve_runs = sorted(
        [run for run in runs if run["spider_suppression"] == selected_suppression],
        key=lambda run: run["ant_factor"],
    )
    curve_factors = [run["ant_factor"] for run in curve_runs]
    curve_axis.plot(curve_factors, [100 * run["p_six"] for run in curve_runs], marker="o", linewidth=2.2, label="P(Six)", color="#0f766e")
    curve_axis.plot(curve_factors, [100 * run["p_eight"] for run in curve_runs], marker="o", linewidth=2.2, label="P(Eight)", color="#b45309")
    curve_axis.plot(curve_factors, [100 * run["p_four"] for run in curve_runs], marker="o", linewidth=1.5, label="P(Four)", color="#4f46e5")
    crossover = next((run for run in curve_runs if run["p_six"] > run["p_eight"]), None)
    if crossover is not None:
        curve_axis.axvline(crossover["ant_factor"], color="#111827", linestyle="--", alpha=0.7)
        curve_axis.annotate(
            f"first sampled Six/Eight crossover\na={crossover['ant_factor']:g}",
            xy=(crossover["ant_factor"], 100 * crossover["p_six"]),
            xytext=(10, 24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->"},
        )
    curve_axis.set_xlabel("ant reference factor")
    curve_axis.set_ylabel("next-token probability (%)")
    curve_axis.set_title(
        f"Detailed curve at spider suppression {selected_suppression:g}", loc="left"
    )
    curve_axis.grid(alpha=0.25)
    curve_axis.legend()
    figure.suptitle(
        f"Best tested scope: {selected['scope']} — "
        f"P(Six)={selected['p_six']:.3%}, P(Eight)={selected['p_eight']:.3%}",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_tidy_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scope",
        "tested_layers",
        "spider_feature_ids",
        "spider_positions",
        "ant_feature_ids",
        "ant_positions",
        "spider_suppression",
        "ant_factor",
        "overdrive",
        "control",
        "success",
        "spider_targets",
        "ant_targets",
        "operation_delta_norms",
        "total_delta_norm",
        "top_token",
        "top_probability",
        "p_six",
        "p_eight",
        "p_four",
        "log_p_six_minus_log_p_eight",
        "other_top_tokens",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            spider_ops = [op for op in run["operations"] if op["role"] == "spider"]
            ant_ops = [op for op in run["operations"] if op["role"] == "ant"]
            writer.writerow(
                {
                    "scope": run["scope"],
                    "tested_layers": json.dumps(run["tested_layers"]),
                    "spider_feature_ids": json.dumps([op["feature"] for op in spider_ops]),
                    "spider_positions": json.dumps([op["position"] for op in spider_ops]),
                    "ant_feature_ids": json.dumps([op["feature"] for op in ant_ops]),
                    "ant_positions": json.dumps([op["position"] for op in ant_ops]),
                    "spider_suppression": run["spider_suppression"],
                    "ant_factor": run["ant_factor"],
                    "overdrive": run["overdrive"],
                    "control": run["control"],
                    "success": run["success"],
                    "spider_targets": json.dumps([op["target_activation"] for op in spider_ops]),
                    "ant_targets": json.dumps([op["target_activation"] for op in ant_ops]),
                    "operation_delta_norms": json.dumps([op["delta_norm"] for op in run["operations"]]),
                    "total_delta_norm": run["total_delta_norm"],
                    "top_token": display_token(run["top_token"]),
                    "top_probability": run["top_probability"],
                    "p_six": run["p_six"],
                    "p_eight": run["p_eight"],
                    "p_four": run["p_four"],
                    "log_p_six_minus_log_p_eight": run["log_p_six_minus_log_p_eight"],
                    "other_top_tokens": json.dumps(run["top_tokens"], ensure_ascii=False),
                }
            )


def main() -> None:
    args = parse_args()
    if not 0 <= args.cutoff:
        raise RuntimeError(f"Cutoff must be non-negative, got {args.cutoff}")
    if not 0 < args.min_ant_p_six <= 1:
        raise RuntimeError("--min-ant-p-six must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_layer_dir = args.output_dir / "per_layer"
    per_layer_dir.mkdir(parents=True, exist_ok=True)

    spider_manifest = read_manifest(args.spider_manifest)
    ant_manifest = read_manifest(args.ant_manifest)
    if spider_manifest.get("model") != MODEL_ID or ant_manifest.get("model") != MODEL_ID:
        raise RuntimeError("The frozen feature manifests do not match the requested model")
    spider_instances = select_spider_instances(spider_manifest, args.cutoff)
    selected_layers = sorted({item["layer"] for item in spider_instances})
    ant_candidates = [
        feature for feature in ant_manifest["features"]
        if int(feature["layer"]) in selected_layers
    ]
    if not ant_candidates:
        raise RuntimeError("No saved ant candidates share a layer with selected spider instances")

    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for instance in spider_instances:
        features_by_layer[instance["layer"]].add(instance["feature"])
    for candidate in ant_candidates:
        features_by_layer[int(candidate["layer"])].add(int(candidate["feature"]))
    print(
        f"Selected {len(spider_instances)} spider feature-position instances across "
        f"{len(selected_layers)} layers at normalized activation >= {args.cutoff:g}."
    )
    print("Fetching the corresponding transcoder rows...")
    rows = load_all_rows(dict(features_by_layer), args.download_workers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to download gated Gemma weights")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    # Gemma emits answer words after ``Answer:`` as leading-space tokens.  The
    # no-leading-space spellings are distinct vocabulary entries and would make
    # a correct 99.998% `` Eight`` baseline look almost zero.
    target_ids = {
        name: single_token_id(tokenizer, f" {name}")
        for name in ["Six", "Eight", "Four"]
    }
    spider_ids = tokenizer(
        args.spider_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    spider_tokens = token_rows(tokenizer, spider_ids)
    if len(spider_tokens) != 27:
        raise RuntimeError(f"Expected 27 spider-prompt tokens, found {len(spider_tokens)}")
    expected_tokens = {16: "spinning", 21: "\n"}
    for position, expected in expected_tokens.items():
        if spider_tokens[position]["token"].strip() != expected.strip():
            raise RuntimeError(
                f"Spider token alignment failed at position {position}: "
                f"expected {expected!r}, found {spider_tokens[position]['token']!r}"
            )

    print("Loading Gemma-3-4B-IT in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    layers = find_text_layers(model)
    device = next(model.parameters()).device
    if max(features_by_layer) >= len(layers):
        raise RuntimeError("A selected feature layer exceeds the model depth")

    spider_ids = spider_ids.to(device)
    print("Running the clean spider prompt and calibrating the ant reference...")
    spider_hidden, spider_logits = capture_hidden_and_logits(
        model, layers, spider_ids, selected_layers
    )
    baseline = probability_record(
        spider_logits, tokenizer, target_ids, args.top_k
    )
    if baseline["top_token"].strip().lower() != "eight" or baseline["p_eight"] < 0.99:
        raise RuntimeError(f"Spider baseline did not reproduce near-certain Eight: {baseline}")

    calibration, eligible_ant_instances = calibrate_ant_reference(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        device=device,
        ant_candidates=ant_candidates,
        rows=rows,
        target_ids=target_ids,
        implicit_question=args.implicit_ant_question,
        explicit_question=args.explicit_ant_question,
        minimum_p_six=args.min_ant_p_six,
        cutoff=args.cutoff,
        top_k=args.top_k,
    )
    eligible_ant_instances = measure_current_activations(
        spider_hidden=spider_hidden,
        instances=eligible_ant_instances,
        rows=rows,
        device=device,
    )

    spider_by_layer_all = group_by_layer(spider_instances)
    ant_by_layer_all = group_by_layer(eligible_ant_instances)
    eligible_layers = sorted(set(spider_by_layer_all) & set(ant_by_layer_all))
    skipped_layers = [
        {
            "layer": layer,
            "reason": "no calibrated same-layer ant feature passed the reference cutoff",
            "spider_instance_count": len(spider_by_layer_all[layer]),
        }
        for layer in sorted(spider_by_layer_all)
        if layer not in ant_by_layer_all
    ]
    if not eligible_layers:
        raise RuntimeError(
            "Ant calibration succeeded, but no same-layer ant feature passed the cutoff "
            "in any selected spider layer"
        )
    spider_by_layer = {layer: spider_by_layer_all[layer] for layer in eligible_layers}
    ant_by_layer = {layer: ant_by_layer_all[layer] for layer in eligible_layers}
    print(
        f"Ant calibration used {calibration['selected']['kind']} prompt; "
        f"{len(eligible_ant_instances)} ant instances make {len(eligible_layers)} layers eligible."
    )

    frozen_manifest = {
        "model": MODEL_ID,
        "created_at_unix": time.time(),
        "selection_cutoff": args.cutoff,
        "selection_definition": "activation / Neuronpedia maxActApprox >= cutoff and learned-threshold active",
        "target_positions": list(TARGET_POSITIONS),
        "spider_prompt": args.spider_prompt,
        "spider_tokens": spider_tokens,
        "spider_instances": spider_instances,
        "ant_calibration": calibration,
        "eligible_ant_instances": eligible_ant_instances,
        "eligible_layers": eligible_layers,
        "skipped_layers": skipped_layers,
        "source_manifests": {
            "spider": str(args.spider_manifest),
            "ant": str(args.ant_manifest),
        },
    }
    manifest_path = args.output_dir / FEATURE_MANIFEST
    manifest_path.write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2))

    common = {
        "model": model,
        "layers": layers,
        "input_ids": spider_ids,
        "tokenizer": tokenizer,
        "target_ids": target_ids,
        "rows": rows,
        "spider_by_layer": spider_by_layer,
        "ant_by_layer": ant_by_layer,
        "top_k": args.top_k,
        "device": device,
    }
    all_runs: list[dict[str, Any]] = []
    layer_runs: dict[int, list[dict[str, Any]]] = {}
    tested_factors = list(dict.fromkeys(args.ant_factors))
    print("Starting isolated standard-grid layer screens...")
    for layer in eligible_layers:
        runs = run_grid(
            tested_layers=(layer,),
            spider_suppressions=args.spider_suppressions,
            ant_factors=tested_factors,
            common=common,
            baseline=baseline,
        )
        layer_runs[layer] = runs
        all_runs.extend(runs)

    any_single_success = any(run["success"] for run in all_runs)
    overdrive_used = False
    if not any_single_success and not args.disable_overdrive:
        overdrive_factors = [
            factor for factor in args.overdrive_factors if factor not in tested_factors
        ]
        if overdrive_factors:
            overdrive_used = True
            print("No standard-grid success. Extending isolated screens with labelled overdrive factors.")
            for layer in eligible_layers:
                extra = run_grid(
                    tested_layers=(layer,),
                    spider_suppressions=args.spider_suppressions,
                    ant_factors=overdrive_factors,
                    common=common,
                    baseline=baseline,
                )
                layer_runs[layer].extend(extra)
                all_runs.extend(extra)
            tested_factors.extend(overdrive_factors)
        any_single_success = any(run["success"] for run in all_runs)

    combination_runs: dict[str, list[dict[str, Any]]] = {}
    if not any_single_success and len(eligible_layers) >= 2:
        ranked_layers = sorted(
            eligible_layers,
            key=lambda layer: best_run(layer_runs[layer])["log_p_six_minus_log_p_eight"],
            reverse=True,
        )
        pair = tuple(ranked_layers[:2])
        print(f"No isolated success. Testing greedy pair {pair}.")
        pair_runs = run_grid(
            tested_layers=pair,
            spider_suppressions=args.spider_suppressions,
            ant_factors=tested_factors,
            common=common,
            baseline=baseline,
        )
        combination_runs["+".join(f"L{layer}" for layer in pair)] = pair_runs
        all_runs.extend(pair_runs)
        if not any(run["success"] for run in pair_runs) and len(ranked_layers) >= 3:
            triple = tuple(ranked_layers[:3])
            print(f"No pair success. Testing greedy triple {triple}.")
            triple_runs = run_grid(
                tested_layers=triple,
                spider_suppressions=args.spider_suppressions,
                ant_factors=tested_factors,
                common=common,
                baseline=baseline,
            )
            combination_runs["+".join(f"L{layer}" for layer in triple)] = triple_runs
            all_runs.extend(triple_runs)

    selected = best_run(all_runs)
    for layer, runs in layer_runs.items():
        plot_layer_heatmaps(
            runs=runs,
            suppressions=args.spider_suppressions,
            factors=tested_factors,
            scope=f"L{layer}",
            output_path=per_layer_dir / f"L{layer:02d}_probability_heatmaps.png",
        )
    overview_path = args.output_dir / OVERVIEW_FIGURE
    plot_overview(layer_runs=layer_runs, baseline=baseline, output_path=overview_path)

    selected_scope = selected["scope"]
    if len(selected["tested_layers"]) == 1:
        detail_runs = layer_runs[selected["tested_layers"][0]]
    else:
        detail_runs = combination_runs[selected_scope]
    detail_path = args.output_dir / DETAIL_FIGURE
    plot_best_detail(
        runs=detail_runs,
        suppressions=args.spider_suppressions,
        factors=tested_factors,
        selected=selected,
        output_path=detail_path,
    )

    result = {
        "model": MODEL_ID,
        "spider_prompt": args.spider_prompt,
        "token_ids": target_ids,
        "baseline": baseline,
        "selection_cutoff": args.cutoff,
        "spider_suppressions": args.spider_suppressions,
        "standard_ant_factors": args.ant_factors,
        "tested_ant_factors": tested_factors,
        "overdrive_used": overdrive_used,
        "eligible_layers": eligible_layers,
        "skipped_layers": skipped_layers,
        "ant_reference_kind": calibration["selected"]["kind"],
        "ant_reference_metrics": calibration["selected"]["metrics"],
        "single_layer_summaries": [
            {
                "layer": layer,
                "spider_instance_count": len(spider_by_layer[layer]),
                "ant_instance_count": len(ant_by_layer[layer]),
                "best_run": best_run(layer_runs[layer]),
                "success_count": sum(run["success"] for run in layer_runs[layer]),
            }
            for layer in eligible_layers
        ],
        "combination_scopes": list(combination_runs),
        "best_run": selected,
        "success": selected["success"],
        "runs": all_runs,
        "artifacts": {
            "feature_manifest": str(manifest_path),
            "tidy_csv": str(args.output_dir / OUTPUT_CSV),
            "overview_figure": str(overview_path),
            "detail_figure": str(detail_path),
            "per_layer_directory": str(per_layer_dir),
        },
    }
    json_path = args.output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    csv_path = args.output_dir / OUTPUT_CSV
    write_tidy_csv(csv_path, all_runs)

    print(
        json.dumps(
            {
                "baseline": {
                    "p_six": baseline["p_six"],
                    "p_eight": baseline["p_eight"],
                    "p_four": baseline["p_four"],
                },
                "ant_reference_kind": calibration["selected"]["kind"],
                "eligible_layers": eligible_layers,
                "best_run": {
                    key: selected[key]
                    for key in [
                        "scope",
                        "spider_suppression",
                        "ant_factor",
                        "overdrive",
                        "success",
                        "top_token",
                        "p_six",
                        "p_eight",
                        "p_four",
                    ]
                },
                "json": str(json_path),
                "csv": str(csv_path),
                "feature_manifest": str(manifest_path),
                "overview": str(overview_path),
                "detail": str(detail_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
