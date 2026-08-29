#!/usr/bin/env python3
"""Add one calibrated ant layer to the frozen 95% L7+L22 intervention.

The established base configuration fully suppresses the selected spider
features at L7 and L22 and injects the calibrated ant features at those layers
with factor 4.  This runner holds that base fixed, then adds the L24, L26, or
L27 ant feature at position 16 with a small factor grid.  It does not suppress
spider features in the added layer, so the experiment isolates the incremental
effect of one additional ant-injection site.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_layerwise_spider_ant_sweep import (
    EXPLICIT_ANT_QUESTION,
    MODEL_ID,
    PROMPT_TEMPLATE,
    SPIDER_PROMPT,
    FeatureKey,
    capture_hidden_and_logits,
    comma_floats,
    display_token,
    feature_activation,
    find_explicit_ant_position,
    find_text_layers,
    load_all_rows,
    probability_record,
    read_manifest,
    single_token_id,
    token_rows,
)
from run_spider_ant_layer_specific_optimization import run_intervention


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_LAYERS = (7, 22)
DEFAULT_CANDIDATE_LAYERS = (24, 26, 27)
OUTPUT_JSON = "spider_ant_added_layer_sweep.json"
OUTPUT_CSV = "spider_ant_added_layer_sweep.csv"
FEATURE_MANIFEST = "spider_ant_added_layer_manifest.json"
CURVES_FIGURE = "spider_ant_added_layer_curves.png"
SUMMARY_FIGURE = "spider_ant_added_layer_summary.png"


def comma_ints(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected comma-separated integers: {value}"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("The candidate layer list cannot be empty")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add one ant-injection layer to the frozen L7+L22 intervention"
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path(
            "results/spider_ant_layer_specific_optimization/"
            "spider_ant_layer_specific_manifest.json"
        ),
    )
    parser.add_argument(
        "--coverage-manifest",
        type=Path,
        default=Path(
            "results/spider_ant_full_window_coverage_sweep/"
            "spider_ant_full_window_coverage_manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spider_ant_added_layer_sweep"),
    )
    parser.add_argument(
        "--candidate-layers",
        type=comma_ints,
        default=list(DEFAULT_CANDIDATE_LAYERS),
    )
    parser.add_argument(
        "--candidate-factors",
        type=comma_floats,
        default=comma_floats("0,0.5,1,2,3,4"),
    )
    parser.add_argument("--base-factor-l7", type=float, default=4.0)
    parser.add_argument("--base-factor-l22", type=float, default=4.0)
    parser.add_argument("--spider-suppression", type=float, default=1.0)
    parser.add_argument("--selection-cutoff", type=float, default=0.4)
    parser.add_argument("--target-probability", type=float, default=0.99)
    parser.add_argument("--minimum-reference-p-six", type=float, default=0.9)
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--explicit-ant-question", default=EXPLICIT_ANT_QUESTION)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def grouped_with_empty_layers(
    instances: list[dict[str, Any]], layers: tuple[int, ...]
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[int(instance["layer"])].append(instance)
    return {layer: grouped[layer] for layer in layers}


def compact_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    fields = [
        "candidate_layer",
        "candidate_factor",
        "candidate_feature_ids",
        "candidate_current_activations",
        "candidate_target_activations",
        "candidate_delta_norm",
        "delta_p_six_vs_base",
        "delta_log_odds_vs_base",
        "reaches_target_probability",
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
    return {field: run[field] for field in fields}


def validate_reference_instances(
    *,
    instances: list[dict[str, Any]],
    hidden: dict[int, torch.Tensor],
    rows: dict[FeatureKey, dict[str, Any]],
    reference_position: int,
    cutoff: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for instance in instances:
        key = FeatureKey(int(instance["layer"]), int(instance["feature"]))
        preactivation, activation = feature_activation(
            hidden[key.layer], reference_position, rows[key], device
        )
        max_act = float(instance["max_act_approx"])
        normalized = activation / max_act
        saved_activation = float(instance["reference_activation"])
        if activation <= 0 or normalized < cutoff:
            raise RuntimeError(
                f"L{key.layer} F{key.feature} failed fresh ant calibration: "
                f"activation={activation:.6f}, normalized={normalized:.6f}"
            )
        if not math.isclose(
            activation, saved_activation, rel_tol=0.03, abs_tol=1.0
        ):
            raise RuntimeError(
                f"L{key.layer} F{key.feature} reference activation drifted: "
                f"fresh={activation:.6f}, saved={saved_activation:.6f}"
            )
        validated.append(
            {
                **instance,
                "fresh_reference_position": reference_position,
                "fresh_reference_preactivation": preactivation,
                "fresh_reference_activation": activation,
                "fresh_normalized_reference_activation": normalized,
                "learned_threshold": float(rows[key]["threshold"]),
            }
        )
    return validated


def annotate_candidate_fields(
    run: dict[str, Any],
    *,
    candidate_layer: int,
    candidate_factor: float,
    base: dict[str, Any],
    target_probability: float,
) -> dict[str, Any]:
    candidate_operations = [
        operation
        for operation in run["operations"]
        if operation["role"] == "ant" and operation["layer"] == candidate_layer
    ]
    run.update(
        {
            "candidate_layer": candidate_layer,
            "candidate_factor": candidate_factor,
            "candidate_feature_ids": [
                int(operation["feature"]) for operation in candidate_operations
            ],
            "candidate_current_activations": [
                float(operation["current_activation"])
                for operation in candidate_operations
            ],
            "candidate_target_activations": [
                float(operation["target_activation"])
                for operation in candidate_operations
            ],
            "candidate_delta_norm": float(
                math.sqrt(
                    sum(
                        float(operation["delta_norm"]) ** 2
                        for operation in candidate_operations
                    )
                )
            ),
            "delta_p_six_vs_base": float(run["p_six"] - base["p_six"]),
            "delta_log_odds_vs_base": float(
                run["log_p_six_minus_log_p_eight"]
                - base["log_p_six_minus_log_p_eight"]
            ),
            "reaches_target_probability": bool(
                run["top_token"].strip().lower() == "six"
                and run["p_six"] >= target_probability
            ),
        }
    )
    return run


def plot_probability_curves(
    *,
    runs: list[dict[str, Any]],
    candidate_layers: tuple[int, ...],
    factors: list[float],
    base: dict[str, Any],
    target_probability: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        len(candidate_layers),
        1,
        figsize=(10.5, 4.1 * len(candidate_layers)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, layer in zip(axes.flat, candidate_layers, strict=True):
        layer_runs = {
            float(run["candidate_factor"]): run
            for run in runs
            if int(run["candidate_layer"]) == layer
        }
        ordered = [layer_runs[factor] for factor in factors]
        axis.plot(
            factors,
            [100 * run["p_six"] for run in ordered],
            marker="o",
            linewidth=2.4,
            color="#0f766e",
            label="P(Six)",
        )
        axis.plot(
            factors,
            [100 * run["p_eight"] for run in ordered],
            marker="o",
            linewidth=2.4,
            color="#b45309",
            label="P(Eight)",
        )
        axis.plot(
            factors,
            [100 * run["p_four"] for run in ordered],
            marker="o",
            linewidth=1.5,
            color="#4f46e5",
            label="P(Four)",
        )
        axis.axhline(
            100 * target_probability,
            color="#111827",
            linestyle=":",
            linewidth=1.5,
            label=f"target {100 * target_probability:g}%",
        )
        axis.axhline(
            100 * base["p_six"],
            color="#0f766e",
            linestyle="--",
            alpha=0.45,
            linewidth=1.2,
            label="fixed L7+L22 base",
        )
        maximum = max(ordered, key=lambda run: (run["p_six"], -run["p_four"]))
        axis.annotate(
            f"max {100 * maximum['p_six']:.3f}%\n"
            f"at factor {maximum['candidate_factor']:g}",
            (
                maximum["candidate_factor"],
                100 * maximum["p_six"],
            ),
            xytext=(8, -34),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#334155"},
        )
        feature_ids = maximum["candidate_feature_ids"]
        axis.set_title(
            f"Add L{layer} ant feature(s) {feature_ids} at position 16",
            loc="left",
        )
        axis.set_ylabel("next-token probability (%)")
        axis.set_ylim(-2, 104)
        axis.grid(alpha=0.22)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    axes.flat[-1].set_xlabel("added-layer ant reference factor")
    figure.suptitle(
        "One added ant layer on the fixed L7=4, L22=4 intervention",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_summary(
    *,
    runs: list[dict[str, Any]],
    candidate_layers: tuple[int, ...],
    base: dict[str, Any],
    output_path: Path,
) -> None:
    best_by_layer = {
        layer: max(
            (run for run in runs if int(run["candidate_layer"]) == layer),
            key=lambda run: (run["p_six"], -run["p_four"]),
        )
        for layer in candidate_layers
    }
    labels = ["L7+L22 base"] + [f"+ L{layer}" for layer in candidate_layers]
    selected = [base] + [best_by_layer[layer] for layer in candidate_layers]
    x = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    axes[0].bar(
        x - width / 2,
        [100 * run["p_six"] for run in selected],
        width,
        color="#0f766e",
        label="P(Six)",
    )
    axes[0].bar(
        x + width / 2,
        [100 * run["p_eight"] for run in selected],
        width,
        color="#b45309",
        label="P(Eight)",
    )
    axes[0].axhline(99, color="#111827", linestyle=":", linewidth=1.5)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("next-token probability (%)")
    axes[0].set_ylim(0, 104)
    axes[0].set_title("Best sampled result from each added layer", loc="left")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend()
    for index, run in enumerate(selected[1:], start=1):
        axes[0].text(
            index,
            min(102, 100 * run["p_six"] + 2),
            f"a={run['candidate_factor']:g}",
            ha="center",
            fontsize=9,
        )

    colors = {24: "#2563eb", 26: "#7c3aed", 27: "#db2777"}
    for layer in candidate_layers:
        layer_runs = sorted(
            (run for run in runs if int(run["candidate_layer"]) == layer),
            key=lambda run: run["candidate_factor"],
        )
        axes[1].plot(
            [run["candidate_factor"] for run in layer_runs],
            [run["delta_log_odds_vs_base"] for run in layer_runs],
            marker="o",
            linewidth=2.2,
            color=colors.get(layer),
            label=f"add L{layer}",
        )
    required_delta = math.log(99) - base["log_p_six_minus_log_p_eight"]
    axes[1].axhline(
        required_delta,
        color="#111827",
        linestyle=":",
        linewidth=1.5,
        label="increment needed for 99:1 odds",
    )
    axes[1].axhline(0, color="#64748b", linewidth=1)
    axes[1].set_xlabel("added-layer ant reference factor")
    axes[1].set_ylabel("change in log P(Six) - log P(Eight)")
    axes[1].set_title("Incremental causal effect beyond the fixed base", loc="left")
    axes[1].grid(alpha=0.22)
    axes[1].legend()
    figure.suptitle("Does one additional ant layer push the SAE intervention to 99%?", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "candidate_layer",
        "candidate_feature_ids",
        "candidate_factor",
        "base_factor_l7",
        "base_factor_l22",
        "spider_suppression",
        "candidate_current_activations",
        "candidate_target_activations",
        "candidate_delta_norm",
        "total_delta_norm",
        "delta_p_six_vs_base",
        "delta_log_odds_vs_base",
        "reaches_target_probability",
        "success",
        "top_token",
        "top_probability",
        "p_six",
        "p_eight",
        "p_four",
        "log_p_six_minus_log_p_eight",
        "top_tokens",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "candidate_layer": run["candidate_layer"],
                    "candidate_feature_ids": json.dumps(
                        run["candidate_feature_ids"]
                    ),
                    "candidate_factor": run["candidate_factor"],
                    "base_factor_l7": run["ant_factors"]["7"],
                    "base_factor_l22": run["ant_factors"]["22"],
                    "spider_suppression": run["spider_suppression"],
                    "candidate_current_activations": json.dumps(
                        run["candidate_current_activations"]
                    ),
                    "candidate_target_activations": json.dumps(
                        run["candidate_target_activations"]
                    ),
                    "candidate_delta_norm": run["candidate_delta_norm"],
                    "total_delta_norm": run["total_delta_norm"],
                    "delta_p_six_vs_base": run["delta_p_six_vs_base"],
                    "delta_log_odds_vs_base": run["delta_log_odds_vs_base"],
                    "reaches_target_probability": run[
                        "reaches_target_probability"
                    ],
                    "success": run["success"],
                    "top_token": display_token(run["top_token"]),
                    "top_probability": run["top_probability"],
                    "p_six": run["p_six"],
                    "p_eight": run["p_eight"],
                    "p_four": run["p_four"],
                    "log_p_six_minus_log_p_eight": run[
                        "log_p_six_minus_log_p_eight"
                    ],
                    "top_tokens": json.dumps(
                        run["top_tokens"], ensure_ascii=False
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    candidate_layers = tuple(sorted(set(args.candidate_layers)))
    candidate_factors = sorted(set(float(value) for value in args.candidate_factors))
    if any(layer in BASE_LAYERS for layer in candidate_layers):
        raise RuntimeError("Candidate layers must not include frozen base layers 7 or 22")
    if any(value < 0 for value in candidate_factors):
        raise RuntimeError("Candidate factors cannot be negative")
    if 0.0 not in candidate_factors:
        raise RuntimeError("The candidate factor grid must include 0 for the base control")
    if args.base_factor_l7 < 0 or args.base_factor_l22 < 0:
        raise RuntimeError("Base ant factors cannot be negative")
    if not 0 <= args.spider_suppression <= 1:
        raise RuntimeError("--spider-suppression must be in [0, 1]")
    if not 0 < args.target_probability < 1:
        raise RuntimeError("--target-probability must be between 0 and 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_manifest = read_manifest(args.base_manifest)
    coverage_manifest = read_manifest(args.coverage_manifest)
    if base_manifest.get("model") != MODEL_ID or coverage_manifest.get("model") != MODEL_ID:
        raise RuntimeError("A source manifest does not match the requested model")

    spider_instances = [
        instance
        for instance in base_manifest["spider_instances"]
        if int(instance["layer"]) in BASE_LAYERS
    ]
    base_ant_instances = [
        instance
        for instance in base_manifest["ant_instances"]
        if int(instance["layer"]) in BASE_LAYERS
    ]
    candidate_instances = [
        instance
        for instance in coverage_manifest["selected_ant_instances"]
        if int(instance["layer"]) in candidate_layers
        and float(instance["normalized_reference_activation"])
        >= args.selection_cutoff
    ]
    candidate_instances_by_layer = defaultdict(list)
    for instance in candidate_instances:
        candidate_instances_by_layer[int(instance["layer"])].append(instance)
    missing = sorted(set(candidate_layers) - set(candidate_instances_by_layer))
    if missing:
        raise RuntimeError(
            f"No above-cutoff calibrated ant feature is available in layers {missing}"
        )
    for instance in spider_instances:
        if int(instance["position"]) not in (16, 21):
            raise RuntimeError("The frozen spider base contains an unexpected position")
    for instance in base_ant_instances + candidate_instances:
        if int(instance["position"]) != 16:
            raise RuntimeError("All ant injections in this experiment must be at position 16")

    all_instances = spider_instances + base_ant_instances + candidate_instances
    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for instance in all_instances:
        features_by_layer[int(instance["layer"])].add(int(instance["feature"]))
    rows = load_all_rows(dict(features_by_layer), args.download_workers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to download gated Gemma weights")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
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
    if (
        spider_tokens[16]["token"].strip() != "spinning"
        or spider_tokens[21]["token"] != "\n"
    ):
        raise RuntimeError("Spider prompt alignment changed at position 16 or 21")

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
    spider_ids = spider_ids.to(device)
    all_layers = tuple(sorted(set(BASE_LAYERS + candidate_layers)))

    clean_hidden, clean_logits = capture_hidden_and_logits(
        model, layers, spider_ids, all_layers
    )
    clean_baseline = probability_record(clean_logits, tokenizer, target_ids, args.top_k)
    if (
        clean_baseline["top_token"].strip().lower() != "eight"
        or clean_baseline["p_eight"] < 0.99
    ):
        raise RuntimeError(f"Clean baseline did not reproduce Eight: {clean_baseline}")
    for instance in spider_instances:
        key = FeatureKey(int(instance["layer"]), int(instance["feature"]))
        _preactivation, activation = feature_activation(
            clean_hidden[key.layer], int(instance["position"]), rows[key], device
        )
        if not math.isclose(
            activation,
            float(instance["clean_activation"]),
            rel_tol=0.03,
            abs_tol=1.0,
        ):
            raise RuntimeError(
                f"Frozen spider activation drifted for L{key.layer} F{key.feature}"
            )

    reference_prompt = PROMPT_TEMPLATE.format(question=args.explicit_ant_question)
    reference_ids = tokenizer(
        reference_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    reference_tokens = token_rows(tokenizer, reference_ids)
    reference_position = find_explicit_ant_position(reference_tokens)
    reference_hidden, reference_logits = capture_hidden_and_logits(
        model, layers, reference_ids.to(device), all_layers
    )
    reference_metrics = probability_record(
        reference_logits, tokenizer, target_ids, args.top_k
    )
    if (
        reference_metrics["top_token"].strip().lower() != "six"
        or reference_metrics["p_six"] < args.minimum_reference_p_six
    ):
        raise RuntimeError(f"Explicit ant reference failed: {reference_metrics}")
    validated_ant_instances = validate_reference_instances(
        instances=base_ant_instances + candidate_instances,
        hidden=reference_hidden,
        rows=rows,
        reference_position=reference_position,
        cutoff=args.selection_cutoff,
        device=device,
    )
    validated_by_key = {
        (int(instance["layer"]), int(instance["feature"])): instance
        for instance in validated_ant_instances
    }
    base_ant_instances = [
        validated_by_key[(int(instance["layer"]), int(instance["feature"]))]
        for instance in base_ant_instances
    ]
    candidate_instances = [
        validated_by_key[(int(instance["layer"]), int(instance["feature"]))]
        for instance in candidate_instances
    ]

    base_spider_by_layer = grouped_with_empty_layers(
        spider_instances, BASE_LAYERS
    )
    base_ant_by_layer = grouped_with_empty_layers(base_ant_instances, BASE_LAYERS)
    common = {
        "model": model,
        "layers": layers,
        "input_ids": spider_ids,
        "tokenizer": tokenizer,
        "target_ids": target_ids,
        "rows": rows,
        "top_k": args.top_k,
        "device": device,
    }
    base_factors = {7: args.base_factor_l7, 22: args.base_factor_l22}
    base_run = run_intervention(
        spider_by_layer=base_spider_by_layer,
        ant_by_layer=base_ant_by_layer,
        tested_layers=BASE_LAYERS,
        spider_suppression=args.spider_suppression,
        ant_factors=base_factors,
        **common,
    )
    if base_run["top_token"].strip().lower() != "six" or base_run["p_six"] < 0.9:
        raise RuntimeError(f"Frozen L7+L22 base did not reproduce: {base_run}")
    print(
        "Frozen base: "
        f"P(Six)={base_run['p_six']:.6%}, "
        f"P(Eight)={base_run['p_eight']:.6%}."
    )

    runs: list[dict[str, Any]] = []
    for candidate_layer in candidate_layers:
        tested_layers = tuple(sorted(BASE_LAYERS + (candidate_layer,)))
        layer_spider_instances = spider_instances
        layer_ant_instances = base_ant_instances + [
            instance
            for instance in candidate_instances
            if int(instance["layer"]) == candidate_layer
        ]
        spider_by_layer = grouped_with_empty_layers(
            layer_spider_instances, tested_layers
        )
        ant_by_layer = grouped_with_empty_layers(layer_ant_instances, tested_layers)
        for candidate_factor in candidate_factors:
            ant_factors = {
                7: args.base_factor_l7,
                22: args.base_factor_l22,
                candidate_layer: candidate_factor,
            }
            run = run_intervention(
                spider_by_layer=spider_by_layer,
                ant_by_layer=ant_by_layer,
                tested_layers=tested_layers,
                spider_suppression=args.spider_suppression,
                ant_factors=ant_factors,
                **common,
            )
            annotate_candidate_fields(
                run,
                candidate_layer=candidate_layer,
                candidate_factor=candidate_factor,
                base=base_run,
                target_probability=args.target_probability,
            )
            if candidate_factor == 0:
                for metric in ["p_six", "p_eight", "p_four"]:
                    if not math.isclose(
                        run[metric], base_run[metric], rel_tol=1e-6, abs_tol=1e-9
                    ):
                        raise RuntimeError(
                            f"L{candidate_layer} factor-0 control differs for {metric}"
                        )
            runs.append(run)
            print(
                f"add L{candidate_layer} factor={candidate_factor:g}: "
                f"P(Six)={run['p_six']:.6%}, "
                f"P(Eight)={run['p_eight']:.6%}, "
                f"P(Four)={run['p_four']:.6%}, "
                f"top={display_token(run['top_token'])!r}"
            )

    _post_hidden, post_logits = capture_hidden_and_logits(
        model, layers, spider_ids, all_layers
    )
    post_sweep_baseline = probability_record(
        post_logits, tokenizer, target_ids, args.top_k
    )
    for metric in ["p_six", "p_eight", "p_four"]:
        if not math.isclose(
            post_sweep_baseline[metric],
            clean_baseline[metric],
            rel_tol=1e-6,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"Post-sweep baseline differs for {metric}; hook leak")

    best_by_layer = {
        str(layer): compact_run(
            max(
                (run for run in runs if run["candidate_layer"] == layer),
                key=lambda run: (run["p_six"], -run["p_four"]),
            )
        )
        for layer in candidate_layers
    }
    maximum = max(runs, key=lambda run: (run["p_six"], -run["p_four"]))
    target_runs = [run for run in runs if run["reaches_target_probability"]]
    smallest_target = (
        min(
            target_runs,
            key=lambda run: (
                run["candidate_factor"],
                run["candidate_delta_norm"],
                -run["p_six"],
            ),
        )
        if target_runs
        else None
    )

    frozen_manifest = {
        "model": MODEL_ID,
        "created_at_unix": time.time(),
        "source_base_manifest": str(args.base_manifest),
        "source_coverage_manifest": str(args.coverage_manifest),
        "selection_cutoff": args.selection_cutoff,
        "base_layers": list(BASE_LAYERS),
        "candidate_layers": list(candidate_layers),
        "spider_suppression": args.spider_suppression,
        "base_ant_factors": {"7": args.base_factor_l7, "22": args.base_factor_l22},
        "candidate_factors": candidate_factors,
        "ant_target_definition": (
            "current_activation + layer_factor * natural_explicit_ant_reference_activation"
        ),
        "spider_target_definition": "clean_activation * (1 - suppression); never negative",
        "spider_prompt": args.spider_prompt,
        "spider_tokens": spider_tokens,
        "explicit_ant_reference": {
            "prompt": reference_prompt,
            "tokens": reference_tokens,
            "concept_position": reference_position,
            "mapped_spider_position": 16,
            "metrics": reference_metrics,
        },
        "spider_instances": spider_instances,
        "base_ant_instances": base_ant_instances,
        "candidate_ant_instances": candidate_instances,
    }
    manifest_path = args.output_dir / FEATURE_MANIFEST
    manifest_path.write_text(
        json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n"
    )

    curves_path = args.output_dir / CURVES_FIGURE
    summary_path = args.output_dir / SUMMARY_FIGURE
    plot_probability_curves(
        runs=runs,
        candidate_layers=candidate_layers,
        factors=candidate_factors,
        base=base_run,
        target_probability=args.target_probability,
        output_path=curves_path,
    )
    plot_summary(
        runs=runs,
        candidate_layers=candidate_layers,
        base=base_run,
        output_path=summary_path,
    )
    csv_path = args.output_dir / OUTPUT_CSV
    write_csv(csv_path, runs)
    result = {
        "experiment": "one added ant layer on frozen L7+L22 SAE intervention",
        "model": MODEL_ID,
        "source_base_manifest": str(args.base_manifest),
        "source_coverage_manifest": str(args.coverage_manifest),
        "selection_cutoff": args.selection_cutoff,
        "target_probability": args.target_probability,
        "candidate_layers": list(candidate_layers),
        "candidate_factors": candidate_factors,
        "clean_baseline": clean_baseline,
        "post_sweep_baseline": post_sweep_baseline,
        "explicit_ant_reference": reference_metrics,
        "frozen_base": base_run,
        "best_by_layer": best_by_layer,
        "maximum_p_six_run": compact_run(maximum),
        "smallest_target_run": compact_run(smallest_target),
        "target_run_count": len(target_runs),
        "run_count": len(runs),
        "runs": runs,
        "artifacts": {
            "json": OUTPUT_JSON,
            "csv": OUTPUT_CSV,
            "manifest": FEATURE_MANIFEST,
            "curves_figure": CURVES_FIGURE,
            "summary_figure": SUMMARY_FIGURE,
        },
    }
    json_path = args.output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        "Maximum added-layer result: "
        f"L{maximum['candidate_layer']} factor={maximum['candidate_factor']:g}, "
        f"P(Six)={maximum['p_six']:.6%}, "
        f"P(Eight)={maximum['p_eight']:.6%}."
    )
    print(
        f"Saved {json_path}, {csv_path}, {manifest_path}, {curves_path}, and {summary_path}."
    )


if __name__ == "__main__":
    main()
