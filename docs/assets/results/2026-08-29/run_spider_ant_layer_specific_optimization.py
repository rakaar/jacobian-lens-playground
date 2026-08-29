#!/usr/bin/env python3
"""Optimize ant injection strengths independently at L7, L22, and L27.

The experiment keeps all selected spider features fully suppressed in each
tested layer while assigning a separate ant reference factor to every layer.
It runs a coarse three-layer cube, refines around the smallest successful
coarse point (or the best log-odds point if none succeeds), performs anchored
leave-one-layer-out tests, and independently optimizes all three layer pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch
from matplotlib.patches import Rectangle
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_layerwise_spider_ant_sweep import (
    MODEL_ID,
    SPIDER_PROMPT,
    FeatureKey,
    capture_hidden_and_logits,
    comma_floats,
    display_token,
    feature_activation,
    find_text_layers,
    group_by_layer,
    load_all_rows,
    probability_record,
    read_manifest,
    single_token_id,
    token_rows,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LAYERS = (7, 22, 27)
OUTPUT_JSON = "spider_ant_layer_specific_optimization.json"
OUTPUT_CSV = "spider_ant_layer_specific_optimization.csv"
FEATURE_MANIFEST = "spider_ant_layer_specific_manifest.json"
COARSE_FIGURE = "spider_ant_layer_specific_coarse_cube.png"
REFINED_FIGURE = "spider_ant_layer_specific_refined_cube.png"
SUMMARY_FIGURE = "spider_ant_layer_specific_summary.png"


def comma_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Expected comma-separated integers: {value}") from error
    if not result:
        raise argparse.ArgumentTypeError("The layer list cannot be empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent L7/L22/L27 spider-to-ant strength optimization"
    )
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=Path(
            "results/layerwise_spider_ant_sweep/layerwise_feature_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spider_ant_layer_specific_optimization"),
    )
    parser.add_argument("--layers", type=comma_ints, default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--coarse-factors",
        type=comma_floats,
        default=comma_floats("0,1,2,3"),
    )
    parser.add_argument(
        "--pair-coarse-factors",
        type=comma_floats,
        default=comma_floats("0,1,2,3,4"),
    )
    parser.add_argument(
        "--isolated-factors",
        type=comma_floats,
        default=[round(index * 0.25, 2) for index in range(21)],
        help="Per-layer factor grid for the final one-layer minimality control.",
    )
    parser.add_argument("--refine-radius", type=float, default=0.5)
    parser.add_argument("--refine-step", type=float, default=0.25)
    parser.add_argument("--spider-suppression", type=float, default=1.0)
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def factors_key(factors: dict[int, float]) -> tuple[tuple[int, float], ...]:
    return tuple(sorted((int(layer), round(float(value), 8)) for layer, value in factors.items()))


def config_key(
    tested_layers: tuple[int, ...],
    factors: dict[int, float],
    spider_suppression: float,
) -> tuple[Any, ...]:
    return tested_layers, factors_key(factors), round(spider_suppression, 8)


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
    ant_factors: dict[int, float],
    top_k: int,
    device: torch.device,
) -> dict[str, Any]:
    """Apply separate ant factors while preserving the established hook semantics."""
    if not 0 <= spider_suppression <= 1:
        raise RuntimeError("Spider suppression must be in [0, 1]")
    if set(ant_factors) != set(tested_layers):
        raise RuntimeError("Each tested layer must have exactly one ant factor")
    if any(value < 0 for value in ant_factors.values()):
        raise RuntimeError("Ant factors cannot be negative")

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

            for instance in spider_by_layer[layer_index]:
                key = FeatureKey(layer_index, instance["feature"])
                row = rows[key]
                current_preactivation, current_activation = feature_activation(
                    state[layer_index], instance["position"], row, device
                )
                target_activation = max(
                    0.0,
                    instance["clean_activation"] * (1.0 - spider_suppression),
                )
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target_activation - current_activation) * decoder
                position = int(instance["position"])
                layer_delta_by_position[position] = (
                    layer_delta_by_position.get(position, torch.zeros_like(delta)) + delta
                )
                operations.append(
                    {
                        "role": "spider",
                        "layer": layer_index,
                        "feature": int(instance["feature"]),
                        "position": position,
                        "current_preactivation": current_preactivation,
                        "current_activation": current_activation,
                        "reference_activation": float(instance["clean_activation"]),
                        "target_activation": target_activation,
                        "factor": -spider_suppression,
                        "delta_norm": float(delta.norm().item()),
                    }
                )

            layer_ant_factor = float(ant_factors[layer_index])
            for instance in ant_by_layer[layer_index]:
                key = FeatureKey(layer_index, instance["feature"])
                row = rows[key]
                current_preactivation, current_activation = feature_activation(
                    state[layer_index], instance["position"], row, device
                )
                target_activation = (
                    current_activation
                    + layer_ant_factor * float(instance["reference_activation"])
                )
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target_activation - current_activation) * decoder
                position = int(instance["position"])
                layer_delta_by_position[position] = (
                    layer_delta_by_position.get(position, torch.zeros_like(delta)) + delta
                )
                operations.append(
                    {
                        "role": "ant",
                        "layer": layer_index,
                        "feature": int(instance["feature"]),
                        "position": position,
                        "current_preactivation": current_preactivation,
                        "current_activation": current_activation,
                        "reference_activation": float(instance["reference_activation"]),
                        "target_activation": target_activation,
                        "factor": layer_ant_factor,
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
    values = [float(ant_factors[layer]) for layer in tested_layers]
    return {
        "tested_layers": list(tested_layers),
        "scope": "+".join(f"L{layer}" for layer in tested_layers),
        "spider_suppression": spider_suppression,
        "ant_factors": {str(layer): float(ant_factors[layer]) for layer in tested_layers},
        "factor_l1": float(sum(values)),
        "factor_l2": float(math.sqrt(sum(value * value for value in values))),
        "maximum_ant_factor": float(max(values, default=0.0)),
        "overdrive": any(value > 1.5 for value in values),
        "control": "spider_only" if all(value == 0 for value in values) else "joint",
        "success": success,
        "total_delta_norm": float(
            math.sqrt(sum(operation["delta_norm"] ** 2 for operation in operations))
        ),
        "operations": operations,
        **metrics,
    }


def smallest_success(runs: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    successes = [run for run in runs if run["success"]]
    if not successes:
        return None
    return min(
        successes,
        key=lambda run: (
            len(run["tested_layers"]),
            run["factor_l2"],
            run["factor_l1"],
            run["total_delta_norm"],
            -run["p_six"],
        ),
    )


def choose_refinement_anchor(runs: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    selected = smallest_success(runs)
    if selected is not None:
        return selected, "smallest_success"
    return (
        max(runs, key=lambda run: (run["log_p_six_minus_log_p_eight"], -run["p_four"])),
        "best_log_odds",
    )


def centered_values(center: float, radius: float, step: float) -> list[float]:
    count = int(round((2 * radius) / step))
    values = {
        round(max(0.0, center - radius + index * step), 8)
        for index in range(count + 1)
    }
    values.add(round(center, 8))
    return sorted(values)


def run_factor_grid(
    *,
    stage: str,
    tested_layers: tuple[int, ...],
    values_by_layer: dict[int, list[float]],
    cache: dict[tuple[Any, ...], dict[str, Any]],
    ordered_runs: list[dict[str, Any]],
    spider_suppression: float,
    common: dict[str, Any],
) -> list[dict[str, Any]]:
    stage_runs: list[dict[str, Any]] = []
    grids = [values_by_layer[layer] for layer in tested_layers]
    total = math.prod(len(grid) for grid in grids)
    print(f"Starting {stage}: {total} configurations over {tested_layers}.")
    for index, factor_values in enumerate(product(*grids), start=1):
        factor_map = dict(zip(tested_layers, factor_values, strict=True))
        key = config_key(tested_layers, factor_map, spider_suppression)
        if key not in cache:
            result = run_intervention(
                tested_layers=tested_layers,
                spider_suppression=spider_suppression,
                ant_factors=factor_map,
                **common,
            )
            result["stages"] = [stage]
            cache[key] = result
            ordered_runs.append(result)
        else:
            result = cache[key]
            if stage not in result["stages"]:
                result["stages"].append(stage)
        stage_runs.append(result)
        print(
            f"{stage} {index:>3}/{total}: "
            + " ".join(f"a{layer}={factor_map[layer]:g}" for layer in tested_layers)
            + f" P(Six)={result['p_six']:.4%} P(Eight)={result['p_eight']:.4%} "
            + f"top={display_token(result['top_token'])!r}"
        )
    return stage_runs


def compact_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    keys = [
        "tested_layers",
        "scope",
        "spider_suppression",
        "ant_factors",
        "factor_l1",
        "factor_l2",
        "maximum_ant_factor",
        "success",
        "total_delta_norm",
        "top_token",
        "top_probability",
        "p_six",
        "p_eight",
        "p_four",
        "log_p_six_minus_log_p_eight",
        "top_tokens",
    ]
    return {key: run[key] for key in keys}


def annotate_probability_grid(
    axis: Any,
    values: np.ndarray,
    successes: np.ndarray,
) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.1f}" if value >= 0.1 else f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value >= 45 else "black",
            )
            if successes[row, column]:
                axis.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#22c55e",
                        linewidth=2.2,
                    )
                )


def plot_cube_slices(
    *,
    runs: list[dict[str, Any]],
    values_by_layer: dict[int, list[float]],
    output_path: Path,
    title: str,
) -> None:
    slice_layer, x_layer, y_layer = DEFAULT_LAYERS
    slice_values = values_by_layer[slice_layer]
    x_values = values_by_layer[x_layer]
    y_values = values_by_layer[y_layer]
    lookup = {
        tuple(float(run["ant_factors"][str(layer)]) for layer in DEFAULT_LAYERS): run
        for run in runs
        if tuple(run["tested_layers"]) == DEFAULT_LAYERS
    }
    columns = 2 if len(slice_values) == 4 else min(3, len(slice_values))
    rows_count = math.ceil(len(slice_values) / columns)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(5.3 * columns, 4.5 * rows_count),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for index, slice_value in enumerate(slice_values):
        axis = axes.flat[index]
        probabilities = np.asarray(
            [
                [
                    100 * lookup[(slice_value, x_value, y_value)]["p_six"]
                    for x_value in x_values
                ]
                for y_value in y_values
            ]
        )
        successes = np.asarray(
            [
                [
                    lookup[(slice_value, x_value, y_value)]["success"]
                    for x_value in x_values
                ]
                for y_value in y_values
            ],
            dtype=bool,
        )
        image = axis.imshow(
            probabilities,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=100,
        )
        axis.set_xticks(range(len(x_values)), [f"{value:g}" for value in x_values])
        axis.set_yticks(range(len(y_values)), [f"{value:g}" for value in y_values])
        axis.set_xlabel(f"L{x_layer} ant factor")
        axis.set_ylabel(f"L{y_layer} ant factor")
        axis.set_title(f"L{slice_layer} ant factor = {slice_value:g}", loc="left")
        annotate_probability_grid(axis, probabilities, successes)
    for index in range(len(slice_values), len(axes.flat)):
        axes.flat[index].axis("off")
    if image is not None:
        figure.colorbar(image, ax=list(axes.flat), label="P(Six), %", shrink=0.84)
    figure.suptitle(title + " (green outline = Six is top token)", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_summary(
    *,
    baseline: dict[str, Any],
    optimized_triple: dict[str, Any],
    pair_summaries: dict[str, dict[str, Any]],
    isolated_summaries: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    entries = [("optimized triple", optimized_triple)] + [
        (f"optimized {scope}", summary["optimized"])
        for scope, summary in pair_summaries.items()
    ] + [
        (f"isolated {scope}", summary["optimized"])
        for scope, summary in isolated_summaries.items()
    ]
    labels = [label for label, _run in entries]
    runs = [run for _label, run in entries]
    x = np.arange(len(entries))
    figure, axes = plt.subplots(1, 2, figsize=(19, 7.6), constrained_layout=True)

    width = 0.36
    axes[0].bar(
        x - width / 2,
        [100 * run["p_six"] for run in runs],
        width,
        label="P(Six)",
        color="#0f766e",
    )
    axes[0].bar(
        x + width / 2,
        [100 * run["p_eight"] for run in runs],
        width,
        label="P(Eight)",
        color="#b45309",
    )
    axes[0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0].set_ylabel("next-token probability (%)")
    axes[0].set_ylim(0, 110)
    axes[0].set_title("Best low-strength success, or best log-odds point", loc="left")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[0].text(
        0.02,
        0.94,
        f"clean baseline: P(Six)={baseline['p_six']:.6%}, "
        f"P(Eight)={baseline['p_eight']:.4%}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=9,
    )
    for index, run in enumerate(runs):
        marker = "success" if run["success"] else "no flip"
        axes[0].text(
            index,
            max(100 * run["p_six"], 100 * run["p_eight"]) + 1.2,
            marker,
            ha="center",
            fontsize=8,
            color="#15803d" if run["success"] else "#991b1b",
        )

    colors = {7: "#2563eb", 22: "#7c3aed", 27: "#db2777"}
    bottom = np.zeros(len(entries))
    for layer in DEFAULT_LAYERS:
        values = np.asarray(
            [float(run["ant_factors"].get(str(layer), 0.0)) for run in runs]
        )
        axes[1].bar(x, values, bottom=bottom, label=f"L{layer}", color=colors[layer])
        bottom += values
    axes[1].set_xticks(x, labels, rotation=22, ha="right")
    axes[1].set_ylabel("ant reference factor (stacked)")
    axes[1].set_title(
        "Ant factors at the selected point; spider suppression is 1 in every named layer",
        loc="left",
    )
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    for index, run in enumerate(runs):
        factor_text = ", ".join(
            f"a{layer}={float(run['ant_factors'].get(str(layer), 0.0)):g}"
            for layer in DEFAULT_LAYERS
        )
        axes[1].text(index, bottom[index] + 0.12, factor_text, ha="center", fontsize=8)

    figure.suptitle(
        "Independent ant-strength optimization with full spider suppression",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stages",
        "scope",
        "tested_layers",
        "spider_suppression",
        "ant_factor_L7",
        "ant_factor_L22",
        "ant_factor_L27",
        "factor_l1",
        "factor_l2",
        "maximum_ant_factor",
        "success",
        "total_delta_norm",
        "spider_feature_ids",
        "spider_positions",
        "spider_targets",
        "ant_feature_ids",
        "ant_positions",
        "ant_targets",
        "operation_delta_norms",
        "top_token",
        "top_probability",
        "p_six",
        "p_eight",
        "p_four",
        "log_p_six_minus_log_p_eight",
        "top_tokens",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            spider_operations = [op for op in run["operations"] if op["role"] == "spider"]
            ant_operations = [op for op in run["operations"] if op["role"] == "ant"]
            writer.writerow(
                {
                    "stages": json.dumps(run["stages"]),
                    "scope": run["scope"],
                    "tested_layers": json.dumps(run["tested_layers"]),
                    "spider_suppression": run["spider_suppression"],
                    "ant_factor_L7": run["ant_factors"].get("7"),
                    "ant_factor_L22": run["ant_factors"].get("22"),
                    "ant_factor_L27": run["ant_factors"].get("27"),
                    "factor_l1": run["factor_l1"],
                    "factor_l2": run["factor_l2"],
                    "maximum_ant_factor": run["maximum_ant_factor"],
                    "success": run["success"],
                    "total_delta_norm": run["total_delta_norm"],
                    "spider_feature_ids": json.dumps(
                        [op["feature"] for op in spider_operations]
                    ),
                    "spider_positions": json.dumps(
                        [op["position"] for op in spider_operations]
                    ),
                    "spider_targets": json.dumps(
                        [op["target_activation"] for op in spider_operations]
                    ),
                    "ant_feature_ids": json.dumps([op["feature"] for op in ant_operations]),
                    "ant_positions": json.dumps([op["position"] for op in ant_operations]),
                    "ant_targets": json.dumps(
                        [op["target_activation"] for op in ant_operations]
                    ),
                    "operation_delta_norms": json.dumps(
                        [op["delta_norm"] for op in run["operations"]]
                    ),
                    "top_token": display_token(run["top_token"]),
                    "top_probability": run["top_probability"],
                    "p_six": run["p_six"],
                    "p_eight": run["p_eight"],
                    "p_four": run["p_four"],
                    "log_p_six_minus_log_p_eight": run["log_p_six_minus_log_p_eight"],
                    "top_tokens": json.dumps(run["top_tokens"], ensure_ascii=False),
                }
            )


def main() -> None:
    args = parse_args()
    requested_layers = tuple(args.layers)
    if requested_layers != DEFAULT_LAYERS:
        raise RuntimeError(
            f"This visualization and experiment require layers {DEFAULT_LAYERS}, "
            f"received {requested_layers}"
        )
    if not 0 <= args.spider_suppression <= 1:
        raise RuntimeError("--spider-suppression must be in [0, 1]")
    if args.refine_radius < 0 or args.refine_step <= 0:
        raise RuntimeError("Refinement radius must be non-negative and step must be positive")
    if any(
        value < 0
        for value in args.coarse_factors
        + args.pair_coarse_factors
        + args.isolated_factors
    ):
        raise RuntimeError("Ant factor grids cannot contain negative values")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_manifest(args.feature_manifest)
    if source_manifest.get("model") != MODEL_ID:
        raise RuntimeError("The feature manifest does not match the requested model")
    selection_cutoff = float(source_manifest["selection_cutoff"])
    spider_instances = [
        instance
        for instance in source_manifest["spider_instances"]
        if int(instance["layer"]) in requested_layers
    ]
    ant_instances = [
        instance
        for instance in source_manifest["eligible_ant_instances"]
        if int(instance["layer"]) in requested_layers
    ]
    for instance in spider_instances:
        if float(instance["normalized_activation"]) < selection_cutoff:
            raise RuntimeError("A selected spider feature is below the frozen cutoff")
        if int(instance["position"]) not in (16, 21):
            raise RuntimeError("Unexpected spider target position")
    for instance in ant_instances:
        if float(instance["normalized_reference_activation"]) < selection_cutoff:
            raise RuntimeError("A selected ant feature is below the frozen cutoff")
        if int(instance["position"]) != 16:
            raise RuntimeError("This optimization expects ant injection only at position 16")

    spider_by_layer = group_by_layer(spider_instances)
    ant_by_layer = group_by_layer(ant_instances)
    for layer in requested_layers:
        if layer not in spider_by_layer or layer not in ant_by_layer:
            raise RuntimeError(f"L{layer} lacks a frozen spider or ant target")

    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for instance in spider_instances + ant_instances:
        features_by_layer[int(instance["layer"])].add(int(instance["feature"]))
    print(
        f"Loading {len(spider_instances)} spider instances and {len(ant_instances)} ant "
        f"instances across {requested_layers}."
    )
    rows = load_all_rows(dict(features_by_layer), args.download_workers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to download gated Gemma weights")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    target_ids = {
        name: single_token_id(tokenizer, f" {name}")
        for name in ["Six", "Eight", "Four"]
    }
    input_ids = tokenizer(
        args.spider_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    prompt_tokens = token_rows(tokenizer, input_ids)
    if len(prompt_tokens) != 27:
        raise RuntimeError(f"Expected 27 prompt tokens, found {len(prompt_tokens)}")
    if prompt_tokens[16]["token"].strip() != "spinning" or prompt_tokens[21]["token"] != "\n":
        raise RuntimeError("Prompt token alignment changed at position 16 or 21")

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
    input_ids = input_ids.to(device)

    clean_hidden, clean_logits = capture_hidden_and_logits(
        model, layers, input_ids, requested_layers
    )
    baseline = probability_record(clean_logits, tokenizer, target_ids, args.top_k)
    if baseline["top_token"].strip().lower() != "eight" or baseline["p_eight"] < 0.99:
        raise RuntimeError(f"Clean baseline did not reproduce Eight: {baseline}")
    for instance in spider_instances:
        key = FeatureKey(int(instance["layer"]), int(instance["feature"]))
        _preactivation, activation = feature_activation(
            clean_hidden[key.layer], int(instance["position"]), rows[key], device
        )
        reference = float(instance["clean_activation"])
        if not math.isclose(activation, reference, rel_tol=0.03, abs_tol=1.0):
            raise RuntimeError(
                f"Frozen clean activation drifted for L{key.layer} F{key.feature}: "
                f"{activation} vs {reference}"
            )

    frozen_manifest = {
        "model": MODEL_ID,
        "created_at_unix": time.time(),
        "source_manifest": str(args.feature_manifest),
        "selection_cutoff": selection_cutoff,
        "tested_layers": list(requested_layers),
        "spider_suppression": args.spider_suppression,
        "spider_target_definition": "clean_activation * (1 - suppression); never negative",
        "ant_target_definition": "current_activation + layer_factor * natural_reference_activation",
        "spider_prompt": args.spider_prompt,
        "spider_tokens": prompt_tokens,
        "spider_instances": spider_instances,
        "ant_instances": ant_instances,
    }
    manifest_path = args.output_dir / FEATURE_MANIFEST
    manifest_path.write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2))

    common = {
        "model": model,
        "layers": layers,
        "input_ids": input_ids,
        "tokenizer": tokenizer,
        "target_ids": target_ids,
        "rows": rows,
        "spider_by_layer": spider_by_layer,
        "ant_by_layer": ant_by_layer,
        "top_k": args.top_k,
        "device": device,
    }
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    all_runs: list[dict[str, Any]] = []

    coarse_values = sorted(set(float(value) for value in args.coarse_factors))
    triple_coarse = run_factor_grid(
        stage="triple_coarse",
        tested_layers=requested_layers,
        values_by_layer={layer: coarse_values for layer in requested_layers},
        cache=cache,
        ordered_runs=all_runs,
        spider_suppression=args.spider_suppression,
        common=common,
    )
    coarse_anchor, coarse_anchor_reason = choose_refinement_anchor(triple_coarse)
    triple_refine_values = {
        layer: centered_values(
            float(coarse_anchor["ant_factors"][str(layer)]),
            args.refine_radius,
            args.refine_step,
        )
        for layer in requested_layers
    }
    triple_refined = run_factor_grid(
        stage="triple_refined",
        tested_layers=requested_layers,
        values_by_layer=triple_refine_values,
        cache=cache,
        ordered_runs=all_runs,
        spider_suppression=args.spider_suppression,
        common=common,
    )
    triple_runs = list(dict.fromkeys(id(run) for run in triple_coarse + triple_refined))
    triple_lookup = {id(run): run for run in triple_coarse + triple_refined}
    triple_unique = [triple_lookup[run_id] for run_id in triple_runs]
    optimized_triple = smallest_success(triple_unique) or max(
        triple_unique,
        key=lambda run: (run["log_p_six_minus_log_p_eight"], -run["p_four"]),
    )
    maximum_p_six_triple = max(triple_unique, key=lambda run: (run["p_six"], -run["p_four"]))

    anchored_ablations: dict[str, dict[str, Any]] = {}
    for dropped_layer in requested_layers:
        pair = tuple(layer for layer in requested_layers if layer != dropped_layer)
        factors = {
            layer: float(optimized_triple["ant_factors"][str(layer)])
            for layer in pair
        }
        ablation_runs = run_factor_grid(
            stage=f"anchored_ablation_drop_L{dropped_layer}",
            tested_layers=pair,
            values_by_layer={layer: [factors[layer]] for layer in pair},
            cache=cache,
            ordered_runs=all_runs,
            spider_suppression=args.spider_suppression,
            common=common,
        )
        anchored_ablations[f"drop_L{dropped_layer}"] = ablation_runs[0]

    pair_summaries: dict[str, dict[str, Any]] = {}
    pair_coarse_values = sorted(set(float(value) for value in args.pair_coarse_factors))
    for omitted_layer in requested_layers:
        pair = tuple(layer for layer in requested_layers if layer != omitted_layer)
        scope = "+".join(f"L{layer}" for layer in pair)
        pair_coarse = run_factor_grid(
            stage=f"pair_{scope}_coarse",
            tested_layers=pair,
            values_by_layer={layer: pair_coarse_values for layer in pair},
            cache=cache,
            ordered_runs=all_runs,
            spider_suppression=args.spider_suppression,
            common=common,
        )
        pair_anchor, pair_anchor_reason = choose_refinement_anchor(pair_coarse)
        pair_refine_values = {
            layer: centered_values(
                float(pair_anchor["ant_factors"][str(layer)]),
                args.refine_radius,
                args.refine_step,
            )
            for layer in pair
        }
        pair_refined = run_factor_grid(
            stage=f"pair_{scope}_refined",
            tested_layers=pair,
            values_by_layer=pair_refine_values,
            cache=cache,
            ordered_runs=all_runs,
            spider_suppression=args.spider_suppression,
            common=common,
        )
        pair_unique = {id(run): run for run in pair_coarse + pair_refined}
        pair_runs = list(pair_unique.values())
        optimized_pair = smallest_success(pair_runs) or max(
            pair_runs,
            key=lambda run: (run["log_p_six_minus_log_p_eight"], -run["p_four"]),
        )
        pair_summaries[scope] = {
            "omitted_layer": omitted_layer,
            "coarse_anchor_reason": pair_anchor_reason,
            "coarse_anchor": pair_anchor,
            "refinement_values": {
                str(layer): values for layer, values in pair_refine_values.items()
            },
            "unique_run_count": len(pair_runs),
            "success_count": sum(run["success"] for run in pair_runs),
            "optimized": optimized_pair,
            "maximum_p_six": max(pair_runs, key=lambda run: (run["p_six"], -run["p_four"])),
        }

    isolated_summaries: dict[str, dict[str, Any]] = {}
    isolated_values = sorted(set(float(value) for value in args.isolated_factors))
    for layer in requested_layers:
        scope = f"L{layer}"
        isolated_runs = run_factor_grid(
            stage=f"isolated_{scope}",
            tested_layers=(layer,),
            values_by_layer={layer: isolated_values},
            cache=cache,
            ordered_runs=all_runs,
            spider_suppression=args.spider_suppression,
            common=common,
        )
        optimized_isolated = smallest_success(isolated_runs) or max(
            isolated_runs,
            key=lambda run: (run["log_p_six_minus_log_p_eight"], -run["p_four"]),
        )
        isolated_summaries[scope] = {
            "unique_run_count": len(isolated_runs),
            "success_count": sum(run["success"] for run in isolated_runs),
            "optimized": optimized_isolated,
            "maximum_p_six": max(
                isolated_runs, key=lambda run: (run["p_six"], -run["p_four"])
            ),
        }

    _post_hidden, post_logits = capture_hidden_and_logits(
        model, layers, input_ids, requested_layers
    )
    post_sweep_baseline = probability_record(post_logits, tokenizer, target_ids, args.top_k)
    for key in ["p_six", "p_eight", "p_four"]:
        if not math.isclose(post_sweep_baseline[key], baseline[key], rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(f"Post-sweep baseline differs for {key}; a hook may remain")

    global_smallest_success = smallest_success(all_runs)
    global_maximum_p_six = max(all_runs, key=lambda run: (run["p_six"], -run["p_four"]))
    coarse_figure_path = args.output_dir / COARSE_FIGURE
    refined_figure_path = args.output_dir / REFINED_FIGURE
    summary_figure_path = args.output_dir / SUMMARY_FIGURE
    plot_cube_slices(
        runs=triple_coarse,
        values_by_layer={layer: coarse_values for layer in requested_layers},
        output_path=coarse_figure_path,
        title="Coarse independent L7/L22/L27 ant-strength sweep",
    )
    plot_cube_slices(
        runs=triple_refined,
        values_by_layer=triple_refine_values,
        output_path=refined_figure_path,
        title="Refined independent L7/L22/L27 ant-strength sweep",
    )
    plot_summary(
        baseline=baseline,
        optimized_triple=optimized_triple,
        pair_summaries={
            scope: {
                **summary,
                "optimized": compact_run(summary["optimized"]),
            }
            for scope, summary in pair_summaries.items()
        },
        isolated_summaries={
            scope: {
                **summary,
                "optimized": compact_run(summary["optimized"]),
            }
            for scope, summary in isolated_summaries.items()
        },
        output_path=summary_figure_path,
    )

    csv_path = args.output_dir / OUTPUT_CSV
    write_csv(csv_path, all_runs)
    result = {
        "experiment": "independent L7/L22/L27 ant factors under full spider suppression",
        "model": MODEL_ID,
        "source_feature_manifest": str(args.feature_manifest),
        "selection_cutoff": selection_cutoff,
        "tested_layers": list(requested_layers),
        "spider_suppression": args.spider_suppression,
        "coarse_factors": coarse_values,
        "pair_coarse_factors": pair_coarse_values,
        "isolated_factors": isolated_values,
        "refine_radius": args.refine_radius,
        "refine_step": args.refine_step,
        "baseline": baseline,
        "post_sweep_baseline": post_sweep_baseline,
        "coarse_anchor_reason": coarse_anchor_reason,
        "coarse_anchor": compact_run(coarse_anchor),
        "triple_refinement_values": {
            str(layer): values for layer, values in triple_refine_values.items()
        },
        "optimized_triple": compact_run(optimized_triple),
        "maximum_p_six_triple": compact_run(maximum_p_six_triple),
        "anchored_ablations": {
            name: compact_run(run) for name, run in anchored_ablations.items()
        },
        "pair_summaries": {
            scope: {
                "omitted_layer": summary["omitted_layer"],
                "coarse_anchor_reason": summary["coarse_anchor_reason"],
                "coarse_anchor": compact_run(summary["coarse_anchor"]),
                "refinement_values": summary["refinement_values"],
                "unique_run_count": summary["unique_run_count"],
                "success_count": summary["success_count"],
                "optimized": compact_run(summary["optimized"]),
                "maximum_p_six": compact_run(summary["maximum_p_six"]),
            }
            for scope, summary in pair_summaries.items()
        },
        "isolated_summaries": {
            scope: {
                "unique_run_count": summary["unique_run_count"],
                "success_count": summary["success_count"],
                "optimized": compact_run(summary["optimized"]),
                "maximum_p_six": compact_run(summary["maximum_p_six"]),
            }
            for scope, summary in isolated_summaries.items()
        },
        "unique_run_count": len(all_runs),
        "success_count": sum(run["success"] for run in all_runs),
        "smallest_success": compact_run(global_smallest_success),
        "maximum_p_six_run": compact_run(global_maximum_p_six),
        "runs": all_runs,
        "artifacts": {
            "json": OUTPUT_JSON,
            "csv": OUTPUT_CSV,
            "manifest": FEATURE_MANIFEST,
            "coarse_figure": COARSE_FIGURE,
            "refined_figure": REFINED_FIGURE,
            "summary_figure": SUMMARY_FIGURE,
        },
    }
    json_path = args.output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "Finished layer-specific optimization: "
        f"{len(all_runs)} unique runs, {result['success_count']} successes."
    )
    print("Optimized triple:", json.dumps(result["optimized_triple"], ensure_ascii=False))
    print("Smallest success:", json.dumps(result["smallest_success"], ensure_ascii=False))
    print(f"Saved {json_path}, {csv_path}, {manifest_path}, and figures.")


if __name__ == "__main__":
    main()
