#!/usr/bin/env python3
"""Apply the frozen L7+L22 spider-to-ant edit across a prompt suffix.

The established intervention edits a small set of Gemma Scope transcoder
features at selected token positions.  This runner keeps the same features and
strengths but repeats each unique feature edit at every token position from a
configurable start through the final prompt token.  It compares the repeated
suffix clamp with clean, zero-strength, position-16-only, spider-only,
ant-only, and established sparse controls.

The model's original MLP output is never reconstructed from the transcoder.
Only selected decoder deltas are added, preserving the unmodelled remainder.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_layerwise_spider_ant_sweep import (
    MODEL_ID,
    SPIDER_PROMPT,
    FeatureKey,
    capture_hidden_and_logits,
    feature_activation,
    find_text_layers,
    load_all_rows,
    probability_record,
    read_manifest,
    single_token_id,
    token_rows,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
DEFAULT_LAYERS = (7, 22)
DEFAULT_START_POSITION = 16
DEFAULT_ANT_FACTOR = 4.0
DEFAULT_MANIFEST = Path(
    "results/spider_ant_layer_specific_optimization/"
    "spider_ant_layer_specific_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("results/spider_ant_all_positions")

OUTPUT_JSON = "spider_ant_all_positions.json"
OUTPUT_CSV = "spider_ant_all_positions.csv"
FEATURE_MANIFEST = "spider_ant_all_positions_manifest.json"
FIGURE = "spider_ant_all_positions_probabilities.png"


def comma_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers: {value}"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("The layer list cannot be empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the frozen L7+L22 spider-to-ant edit at every suffix token"
    )
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layers", type=comma_ints, default=list(DEFAULT_LAYERS))
    parser.add_argument("--start-position", type=int, default=DEFAULT_START_POSITION)
    parser.add_argument(
        "--end-position",
        type=int,
        default=-1,
        help="Inclusive suffix end; -1 means the final prompt token.",
    )
    parser.add_argument("--ant-factor", type=float, default=DEFAULT_ANT_FACTOR)
    parser.add_argument("--spider-suppression", type=float, default=1.0)
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def unique_features(
    instances: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one metadata record per layer/feature, preserving manifest order."""
    result: list[dict[str, Any]] = []
    seen: set[FeatureKey] = set()
    for instance in instances:
        key = FeatureKey(int(instance["layer"]), int(instance["feature"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(instance))
    return result


def replicate_at_positions(
    instances: Iterable[dict[str, Any]], positions: Iterable[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for instance in unique_features(instances):
        for position in positions:
            result.append({**instance, "position": int(position)})
    return result


def group_by_layer(
    instances: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[int(instance["layer"])].append(instance)
    return dict(grouped)


def run_intervention(
    *,
    model: torch.nn.Module,
    layers: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    target_ids: dict[str, int],
    rows: dict[FeatureKey, dict[str, Any]],
    spider_instances: list[dict[str, Any]],
    ant_instances: list[dict[str, Any]],
    tested_layers: tuple[int, ...],
    spider_suppression: float,
    ant_factor: float,
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    """Apply decoder-only feature deltas in normal forward-pass order."""
    if not 0.0 <= spider_suppression <= 1.0:
        raise RuntimeError("Spider suppression must be in [0, 1]")
    if ant_factor < 0:
        raise RuntimeError("Ant factor cannot be negative")

    spider_by_layer = group_by_layer(spider_instances)
    ant_by_layer = group_by_layer(ant_instances)
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
            delta_by_position: dict[int, torch.Tensor] = {}

            for role, instances in (
                ("spider", spider_by_layer.get(layer_index, [])),
                ("ant", ant_by_layer.get(layer_index, [])),
            ):
                for instance in instances:
                    key = FeatureKey(layer_index, int(instance["feature"]))
                    row = rows[key]
                    position = int(instance["position"])
                    current_preactivation, current_activation = feature_activation(
                        state[layer_index], position, row, device
                    )
                    if role == "spider":
                        target_activation = current_activation * (
                            1.0 - spider_suppression
                        )
                        reference_activation = current_activation
                        factor = -spider_suppression
                    else:
                        reference_activation = float(instance["reference_activation"])
                        target_activation = (
                            current_activation + ant_factor * reference_activation
                        )
                        factor = ant_factor

                    decoder = torch.from_numpy(row["W_dec"]).to(
                        device=device, dtype=torch.float32
                    )
                    delta = (target_activation - current_activation) * decoder
                    delta_by_position[position] = (
                        delta_by_position.get(position, torch.zeros_like(delta)) + delta
                    )
                    operations.append(
                        {
                            "role": role,
                            "layer": layer_index,
                            "feature": key.feature,
                            "position": position,
                            "current_preactivation": current_preactivation,
                            "current_activation": current_activation,
                            "reference_activation": reference_activation,
                            "target_activation": target_activation,
                            "factor": factor,
                            "delta_norm": float(delta.norm().item()),
                            "active_before_edit": current_activation > 0.0,
                        }
                    )

            for position, delta in delta_by_position.items():
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
    role_layer_norms: dict[str, float] = defaultdict(float)
    active_counts: dict[str, int] = defaultdict(int)
    for operation in operations:
        key = f"{operation['role']}_L{operation['layer']}"
        role_layer_norms[key] += operation["delta_norm"] ** 2
        if operation["active_before_edit"]:
            active_counts[key] += 1
    return {
        **metrics,
        "operations": operations,
        "role_layer_delta_l2": {
            key: math.sqrt(value) for key, value in role_layer_norms.items()
        },
        "active_feature_position_counts": dict(active_counts),
    }


def numeric_close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-9)


def compact(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "operations"}


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "condition",
        "scope",
        "positions",
        "spider_suppression",
        "ant_factor",
        "top_token",
        "top_probability",
        "p_six",
        "p_eight",
        "p_four",
        "log_p_six_minus_log_p_eight",
        "operation_count",
        "delta_norms",
        "active_feature_position_counts",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "condition": run["condition"],
                    "scope": run["scope"],
                    "positions": json.dumps(run["positions"]),
                    "spider_suppression": run["spider_suppression"],
                    "ant_factor": run["ant_factor"],
                    "top_token": run["top_token"],
                    "top_probability": run["top_probability"],
                    "p_six": run["p_six"],
                    "p_eight": run["p_eight"],
                    "p_four": run["p_four"],
                    "log_p_six_minus_log_p_eight": run[
                        "log_p_six_minus_log_p_eight"
                    ],
                    "operation_count": len(run["operations"]),
                    "delta_norms": json.dumps(run["role_layer_delta_l2"], sort_keys=True),
                    "active_feature_position_counts": json.dumps(
                        run["active_feature_position_counts"], sort_keys=True
                    ),
                }
            )


def plot_probabilities(path: Path, runs: list[dict[str, Any]]) -> None:
    labels = [run["label"] for run in runs]
    x = np.arange(len(runs))
    width = 0.24
    figure, axis = plt.subplots(figsize=(12.5, 6.8))
    for offset, key, label, color in (
        (-width, "p_six", "P(Six)", "#2a9d8f"),
        (0.0, "p_eight", "P(Eight)", "#e76f51"),
        (width, "p_four", "P(Four)", "#457b9d"),
    ):
        values = [100.0 * float(run[key]) for run in runs]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        for bar, value in zip(bars, values):
            if value >= 0.01:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(100.0, value + 1.2),
                    f"{value:.2f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90 if value > 92 else 0,
                )
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=24, ha="right")
    axis.set_ylim(0, 108)
    axis.set_ylabel("next-token probability (%)")
    axis.set_title(
        "L7 + L22 spider-to-ant transcoder edit repeated from position 16 to prompt end",
        loc="left",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    requested_layers = tuple(sorted(set(int(layer) for layer in args.layers)))
    if requested_layers != DEFAULT_LAYERS:
        raise RuntimeError("This experiment is frozen to L7+L22")
    if not 0.0 <= args.spider_suppression <= 1.0:
        raise RuntimeError("--spider-suppression must be in [0, 1]")
    if args.ant_factor < 0:
        raise RuntimeError("--ant-factor cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source = read_manifest(args.feature_manifest)
    if source.get("model") != MODEL_ID:
        raise RuntimeError("The feature manifest does not match Gemma-3-4B-IT")
    base_spider = [
        dict(instance)
        for instance in source["spider_instances"]
        if int(instance["layer"]) in requested_layers
    ]
    base_ant = [
        dict(instance)
        for instance in source["ant_instances"]
        if int(instance["layer"]) in requested_layers
    ]
    if not base_spider or not base_ant:
        raise RuntimeError("The frozen manifest lacks L7+L22 spider or ant features")

    feature_ids: dict[int, set[int]] = defaultdict(set)
    for instance in base_spider + base_ant:
        feature_ids[int(instance["layer"])].add(int(instance["feature"]))
    rows = load_all_rows(dict(feature_ids), args.download_workers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to download gated Gemma weights")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=hf_token, revision=args.model_revision
    )
    target_ids = {
        name: single_token_id(tokenizer, f" {name}")
        for name in ["Six", "Eight", "Four"]
    }
    input_ids = tokenizer(
        args.spider_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    prompt_tokens = token_rows(tokenizer, input_ids)
    end_position = (
        len(prompt_tokens) - 1 if args.end_position < 0 else args.end_position
    )
    if len(prompt_tokens) != 27:
        raise RuntimeError(f"Expected 27 prompt tokens, found {len(prompt_tokens)}")
    if not 0 <= args.start_position <= end_position < len(prompt_tokens):
        raise RuntimeError("The requested suffix positions are outside the prompt")
    if prompt_tokens[args.start_position]["token"].strip() != "spinning":
        raise RuntimeError("The suffix does not start at the expected 'spinning' token")
    suffix_positions = list(range(args.start_position, end_position + 1))

    print("Loading pinned Gemma-3-4B-IT in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    layers = find_text_layers(model)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    clean_hidden, clean_logits = capture_hidden_and_logits(
        model, layers, input_ids, requested_layers
    )
    baseline = probability_record(clean_logits, tokenizer, target_ids, args.top_k)
    if baseline["top_token"].strip().lower() != "eight" or baseline["p_eight"] < 0.99:
        raise RuntimeError(f"Clean baseline did not reproduce Eight: {baseline}")

    for instance in base_spider:
        key = FeatureKey(int(instance["layer"]), int(instance["feature"]))
        _, activation = feature_activation(
            clean_hidden[key.layer], int(instance["position"]), rows[key], device
        )
        if not math.isclose(
            activation,
            float(instance["clean_activation"]),
            rel_tol=0.03,
            abs_tol=1.0,
        ):
            raise RuntimeError(
                f"Frozen activation drifted at L{key.layer} F{key.feature} "
                f"P{instance['position']}: {activation} vs {instance['clean_activation']}"
            )

    sparse_spider = base_spider
    sparse_ant = base_ant
    position16_spider = replicate_at_positions(base_spider, [args.start_position])
    position16_ant = replicate_at_positions(base_ant, [args.start_position])
    suffix_spider = replicate_at_positions(base_spider, suffix_positions)
    suffix_ant = replicate_at_positions(base_ant, suffix_positions)

    conditions = [
        {
            "condition": "zero_suffix_control",
            "label": "Suffix hooks\nzero strength",
            "scope": "suffix",
            "positions": suffix_positions,
            "spider_instances": suffix_spider,
            "ant_instances": suffix_ant,
            "spider_suppression": 0.0,
            "ant_factor": 0.0,
        },
        {
            "condition": "established_sparse_both",
            "label": "Established sparse\nL7 + L22",
            "scope": "established_sparse",
            "positions": sorted(
                {int(instance["position"]) for instance in sparse_spider + sparse_ant}
            ),
            "spider_instances": sparse_spider,
            "ant_instances": sparse_ant,
            "spider_suppression": args.spider_suppression,
            "ant_factor": args.ant_factor,
        },
        {
            "condition": "position16_both",
            "label": "Position 16 only\nspider + ant",
            "scope": "position16",
            "positions": [args.start_position],
            "spider_instances": position16_spider,
            "ant_instances": position16_ant,
            "spider_suppression": args.spider_suppression,
            "ant_factor": args.ant_factor,
        },
        {
            "condition": "suffix_spider_only",
            "label": "Positions 16–end\nspider only",
            "scope": "suffix",
            "positions": suffix_positions,
            "spider_instances": suffix_spider,
            "ant_instances": suffix_ant,
            "spider_suppression": args.spider_suppression,
            "ant_factor": 0.0,
        },
        {
            "condition": "suffix_ant_only",
            "label": "Positions 16–end\nant only",
            "scope": "suffix",
            "positions": suffix_positions,
            "spider_instances": suffix_spider,
            "ant_instances": suffix_ant,
            "spider_suppression": 0.0,
            "ant_factor": args.ant_factor,
        },
        {
            "condition": "suffix_both",
            "label": "Positions 16–end\nspider + ant",
            "scope": "suffix",
            "positions": suffix_positions,
            "spider_instances": suffix_spider,
            "ant_instances": suffix_ant,
            "spider_suppression": args.spider_suppression,
            "ant_factor": args.ant_factor,
        },
    ]

    common = {
        "model": model,
        "layers": layers,
        "input_ids": input_ids,
        "tokenizer": tokenizer,
        "target_ids": target_ids,
        "rows": rows,
        "tested_layers": requested_layers,
        "top_k": args.top_k,
        "device": device,
    }
    runs: list[dict[str, Any]] = [
        {
            "condition": "clean",
            "label": "Clean",
            "scope": "clean",
            "positions": [],
            "spider_suppression": 0.0,
            "ant_factor": 0.0,
            **baseline,
            "operations": [],
            "role_layer_delta_l2": {},
            "active_feature_position_counts": {},
        }
    ]
    for condition in conditions:
        print(
            f"Running {condition['condition']} at positions "
            f"{condition['positions'][0]}..{condition['positions'][-1]}..."
        )
        metrics = run_intervention(
            spider_instances=condition["spider_instances"],
            ant_instances=condition["ant_instances"],
            spider_suppression=float(condition["spider_suppression"]),
            ant_factor=float(condition["ant_factor"]),
            **common,
        )
        runs.append(
            {
                **{
                    key: value
                    for key, value in condition.items()
                    if key not in ("spider_instances", "ant_instances")
                },
                **metrics,
            }
        )
        print(
            f"  top={metrics['top_token']!r} "
            f"P(Six)={100 * metrics['p_six']:.6f}% "
            f"P(Eight)={100 * metrics['p_eight']:.6f}% "
            f"P(Four)={100 * metrics['p_four']:.6f}%"
        )

    zero = next(run for run in runs if run["condition"] == "zero_suffix_control")
    for key in ("p_six", "p_eight", "p_four"):
        if not numeric_close(float(zero[key]), float(baseline[key])):
            raise RuntimeError(f"Zero-strength suffix control differs for {key}")
    _, post_logits = capture_hidden_and_logits(model, layers, input_ids, requested_layers)
    post_cleanup = probability_record(post_logits, tokenizer, target_ids, args.top_k)
    for key in ("p_six", "p_eight", "p_four"):
        if not numeric_close(float(post_cleanup[key]), float(baseline[key])):
            raise RuntimeError(f"Post-cleanup baseline differs for {key}")

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
    }
    frozen_manifest = {
        "experiment": "L7+L22 repeated spider-to-ant suffix clamp",
        "created_at_unix": time.time(),
        "model": MODEL_ID,
        "source_manifest": str(args.feature_manifest),
        "layers": list(requested_layers),
        "prompt": args.spider_prompt,
        "prompt_tokens": prompt_tokens,
        "start_position": args.start_position,
        "end_position": end_position,
        "suffix_positions": suffix_positions,
        "spider_suppression": args.spider_suppression,
        "ant_factor": args.ant_factor,
        "spider_target_definition": "current feature activation * (1 - suppression)",
        "ant_target_definition": (
            "current feature activation + factor * frozen natural-reference activation"
        ),
        "edit_definition": (
            "original MLP output + selected (target-current) * decoder deltas; "
            "no full transcoder reconstruction"
        ),
        "unique_spider_features": unique_features(base_spider),
        "unique_ant_features": unique_features(base_ant),
        "runtime": runtime,
    }
    manifest_path = args.output_dir / FEATURE_MANIFEST
    manifest_path.write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2))

    csv_path = args.output_dir / OUTPUT_CSV
    json_path = args.output_dir / OUTPUT_JSON
    figure_path = args.output_dir / FIGURE
    write_csv(csv_path, runs)
    plot_probabilities(figure_path, runs)
    result = {
        "experiment": "L7+L22 repeated spider-to-ant suffix clamp",
        "model": MODEL_ID,
        "baseline": baseline,
        "post_cleanup_baseline": post_cleanup,
        "requested": {
            "layers": list(requested_layers),
            "positions": suffix_positions,
            "spider_suppression": args.spider_suppression,
            "ant_factor": args.ant_factor,
        },
        "runtime": runtime,
        "runs": runs,
        "summary": {
            run["condition"]: compact(run) for run in runs
        },
        "validation": {
            "clean_top_is_eight": baseline["top_token"].strip().lower() == "eight",
            "clean_p_eight_above_99_percent": baseline["p_eight"] > 0.99,
            "zero_suffix_matches_clean": all(
                numeric_close(float(zero[key]), float(baseline[key]))
                for key in ("p_six", "p_eight", "p_four")
            ),
            "hooks_removed": all(
                numeric_close(float(post_cleanup[key]), float(baseline[key]))
                for key in ("p_six", "p_eight", "p_four")
            ),
        },
        "artifacts": {
            "json": OUTPUT_JSON,
            "csv": OUTPUT_CSV,
            "manifest": FEATURE_MANIFEST,
            "figure": FIGURE,
        },
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    suffix_both = next(run for run in runs if run["condition"] == "suffix_both")
    print(
        "Finished suffix experiment: "
        f"top={suffix_both['top_token']!r}, "
        f"P(Six)={100 * suffix_both['p_six']:.6f}%, "
        f"P(Eight)={100 * suffix_both['p_eight']:.6f}%, "
        f"P(Four)={100 * suffix_both['p_four']:.6f}%."
    )
    print(f"Saved {json_path}, {csv_path}, {manifest_path}, and {figure_path}.")


if __name__ == "__main__":
    main()
