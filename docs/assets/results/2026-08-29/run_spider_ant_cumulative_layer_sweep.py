#!/usr/bin/env python3
"""Test cumulative ant-feature injections beyond the frozen L7+L22 base.

The established base intervention fully suppresses the selected spider
features at L7 and L22, then injects the calibrated ant features at those
layers with factor 4.  This runner holds that base fixed and evaluates every
on/off subset of the L24, L25, L26, and L27 ant features at position 16.

L24, L26, and L27 are the primary above-cutoff candidates.  L25 is retained
only as a clearly labelled below-cutoff exploratory candidate.  All enabled
add-on layers use the same natural-reference factor (1 by default).  The full
factorial design lets us compare observed log odds with the sum of singleton
effects rather than assuming that layer effects add independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

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
from run_spider_ant_added_layer_sweep import grouped_with_empty_layers
from run_spider_ant_layer_specific_optimization import run_intervention


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_LAYERS = (7, 22)
DEFAULT_PRIMARY_LAYERS = (24, 26, 27)
DEFAULT_EXPLORATORY_LAYERS = (25,)
OUTPUT_JSON = "spider_ant_cumulative_layer_sweep.json"
OUTPUT_CSV = "spider_ant_cumulative_layer_sweep.csv"
FEATURE_MANIFEST = "spider_ant_cumulative_layer_manifest.json"
FACTORIAL_FIGURE = "spider_ant_cumulative_factorial.png"
CUMULATIVE_FIGURE = "spider_ant_cumulative_path.png"


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
        description="Factorial cumulative ant-layer test on the frozen L7+L22 base"
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
        "--prior-added-results",
        type=Path,
        default=Path(
            "results/spider_ant_added_layer_sweep/"
            "spider_ant_added_layer_sweep.json"
        ),
        help="Prior singleton sweep used as a cross-run reproducibility check.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/spider_ant_cumulative_layer_sweep"),
    )
    parser.add_argument(
        "--primary-layers",
        type=comma_ints,
        default=list(DEFAULT_PRIMARY_LAYERS),
    )
    parser.add_argument(
        "--exploratory-layers",
        type=comma_ints,
        default=list(DEFAULT_EXPLORATORY_LAYERS),
    )
    parser.add_argument("--candidate-factor", type=float, default=1.0)
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


def all_subsets(values: tuple[int, ...]) -> list[tuple[int, ...]]:
    return [
        subset
        for size in range(len(values) + 1)
        for subset in combinations(values, size)
    ]


def subset_name(subset: Iterable[int], exploratory: set[int]) -> str:
    layers = tuple(sorted(subset))
    if not layers:
        return "base"
    return "+".join(
        f"L{layer}{'*' if layer in exploratory else ''}" for layer in layers
    )


def compact_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if run is None:
        return None
    fields = [
        "subset_layers",
        "subset_label",
        "subset_size",
        "includes_below_cutoff",
        "candidate_factor",
        "candidate_feature_ids",
        "candidate_current_activations",
        "candidate_target_activations",
        "candidate_delta_norm",
        "delta_p_six_vs_base",
        "delta_log_odds_vs_base",
        "additive_predicted_log_odds",
        "additive_residual_log_odds",
        "mobius_log_odds_term",
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
    primary_keys: set[tuple[int, int]],
    exploratory_keys: set[tuple[int, int]],
    hidden: dict[int, torch.Tensor],
    rows: dict[FeatureKey, dict[str, Any]],
    reference_position: int,
    cutoff: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for instance in instances:
        key = FeatureKey(int(instance["layer"]), int(instance["feature"]))
        tuple_key = (key.layer, key.feature)
        preactivation, activation = feature_activation(
            hidden[key.layer], reference_position, rows[key], device
        )
        max_act = float(instance["max_act_approx"])
        normalized = activation / max_act
        saved_activation = float(instance["reference_activation"])
        if activation <= 0:
            raise RuntimeError(
                f"L{key.layer} F{key.feature} is not naturally active on the "
                "fresh explicit-ant reference"
            )
        if not math.isclose(
            activation, saved_activation, rel_tol=0.03, abs_tol=1.0
        ):
            raise RuntimeError(
                f"L{key.layer} F{key.feature} reference activation drifted: "
                f"fresh={activation:.6f}, saved={saved_activation:.6f}"
            )
        if tuple_key in primary_keys and normalized < cutoff:
            raise RuntimeError(
                f"Primary L{key.layer} F{key.feature} failed cutoff: {normalized:.6f}"
            )
        if tuple_key in exploratory_keys and normalized >= cutoff:
            raise RuntimeError(
                f"Exploratory L{key.layer} F{key.feature} unexpectedly passed cutoff; "
                "update the candidate tier instead of silently relabelling it"
            )
        tier = (
            "primary_above_cutoff"
            if tuple_key in primary_keys
            else "exploratory_below_cutoff"
            if tuple_key in exploratory_keys
            else "frozen_base"
        )
        validated.append(
            {
                **instance,
                "selection_tier": tier,
                "fresh_reference_position": reference_position,
                "fresh_reference_preactivation": preactivation,
                "fresh_reference_activation": activation,
                "fresh_normalized_reference_activation": normalized,
                "learned_threshold": float(rows[key]["threshold"]),
            }
        )
    return validated


def annotate_run(
    run: dict[str, Any],
    *,
    subset: tuple[int, ...],
    candidate_layers: tuple[int, ...],
    exploratory_layers: set[int],
    candidate_factor: float,
    base: dict[str, Any],
    target_probability: float,
) -> None:
    candidate_operations = [
        operation
        for operation in run["operations"]
        if operation["role"] == "ant"
        and int(operation["layer"]) in candidate_layers
    ]
    enabled_operations = [
        operation
        for operation in candidate_operations
        if int(operation["layer"]) in subset
    ]

    def values_by_layer(field: str) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for operation in candidate_operations:
            grouped[str(int(operation["layer"]))].append(float(operation[field]))
        return dict(grouped)

    feature_ids: dict[str, list[int]] = defaultdict(list)
    for operation in candidate_operations:
        feature_ids[str(int(operation["layer"]))].append(int(operation["feature"]))

    run.update(
        {
            "subset_layers": list(subset),
            "subset_label": subset_name(subset, exploratory_layers),
            "subset_size": len(subset),
            "includes_below_cutoff": bool(set(subset) & exploratory_layers),
            "candidate_factor": candidate_factor,
            "candidate_feature_ids": dict(feature_ids),
            "candidate_current_activations": values_by_layer("current_activation"),
            "candidate_target_activations": values_by_layer("target_activation"),
            "candidate_delta_norm": float(
                math.sqrt(
                    sum(float(operation["delta_norm"]) ** 2 for operation in enabled_operations)
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


def add_factorial_effects(
    runs: list[dict[str, Any]], candidate_layers: tuple[int, ...]
) -> None:
    by_subset = {frozenset(run["subset_layers"]): run for run in runs}
    empty = by_subset[frozenset()]
    base_log_odds = float(empty["log_p_six_minus_log_p_eight"])
    singleton_effects = {
        layer: float(
            by_subset[frozenset((layer,))]["log_p_six_minus_log_p_eight"]
            - base_log_odds
        )
        for layer in candidate_layers
    }

    for run in runs:
        subset = frozenset(run["subset_layers"])
        predicted = base_log_odds + sum(singleton_effects[layer] for layer in subset)
        observed = float(run["log_p_six_minus_log_p_eight"])
        mobius = 0.0
        for size in range(len(subset) + 1):
            for child in combinations(sorted(subset), size):
                sign = -1 if (len(subset) - size) % 2 else 1
                mobius += sign * float(
                    by_subset[frozenset(child)]["log_p_six_minus_log_p_eight"]
                )
        run.update(
            {
                "additive_predicted_log_odds": float(predicted),
                "additive_residual_log_odds": float(observed - predicted),
                "additive_predicted_pairwise_p_six": float(
                    1.0 / (1.0 + math.exp(-predicted))
                ),
                "observed_pairwise_p_six": float(
                    run["p_six"] / max(run["p_six"] + run["p_eight"], 1e-45)
                ),
                "mobius_log_odds_term": None if not subset else float(mobius),
            }
        )


def compare_prior_singletons(
    *,
    runs: list[dict[str, Any]],
    primary_layers: tuple[int, ...],
    prior_path: Path,
    candidate_factor: float,
) -> list[dict[str, Any]]:
    if not prior_path.is_file():
        raise RuntimeError(f"Missing prior singleton results: {prior_path}")
    prior = json.loads(prior_path.read_text())
    checks: list[dict[str, Any]] = []
    for layer in primary_layers:
        current = next(
            run for run in runs if tuple(run["subset_layers"]) == (layer,)
        )
        previous = next(
            run
            for run in prior["runs"]
            if int(run["candidate_layer"]) == layer
            and math.isclose(
                float(run["candidate_factor"]), candidate_factor, abs_tol=1e-12
            )
        )
        metric_differences = {
            metric: float(current[metric] - previous[metric])
            for metric in [
                "p_six",
                "p_eight",
                "p_four",
                "log_p_six_minus_log_p_eight",
            ]
        }
        for metric, difference in metric_differences.items():
            if not math.isclose(
                float(current[metric]),
                float(previous[metric]),
                rel_tol=2e-4,
                abs_tol=2e-6,
            ):
                raise RuntimeError(
                    f"L{layer} factor-{candidate_factor:g} singleton did not reproduce "
                    f"prior {metric}: difference={difference:.8g}"
                )
        checks.append(
            {
                "layer": layer,
                "factor": candidate_factor,
                "passed": True,
                "metric_differences": metric_differences,
            }
        )
    return checks


def plot_factorial_summary(
    *,
    runs: list[dict[str, Any]],
    exploratory_layers: set[int],
    target_probability: float,
    output_path: Path,
) -> None:
    ordered = sorted(
        runs,
        key=lambda run: (
            int(run["includes_below_cutoff"]),
            int(run["subset_size"]),
            tuple(run["subset_layers"]),
        ),
    )
    labels = [run["subset_label"] for run in ordered]
    x = np.arange(len(ordered))
    width = 0.38
    colors = [
        "#a855f7" if run["includes_below_cutoff"] else "#0f766e"
        for run in ordered
    ]

    figure, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)
    axes[0].bar(
        x - width / 2,
        [100 * run["p_six"] for run in ordered],
        width,
        color=colors,
        label="P(Six)",
    )
    axes[0].bar(
        x + width / 2,
        [100 * run["p_eight"] for run in ordered],
        width,
        color="#b45309",
        label="P(Eight)",
    )
    axes[0].axhline(
        100 * target_probability,
        color="#111827",
        linestyle=":",
        linewidth=1.6,
        label=f"target {100 * target_probability:g}%",
    )
    axes[0].set_xticks(x, labels, rotation=55, ha="right")
    axes[0].set_ylabel("next-token probability (%)")
    axes[0].set_ylim(0, 104)
    axes[0].set_title("Every factor-1 add-on subset", loc="left")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend()

    multi = [run for run in runs if run["subset_size"] >= 2]
    label_offsets = {
        (24, 25): (7, -17),
        (24, 26): (7, 8),
        (26, 27): (7, 8),
        (24, 26, 27): (7, -17),
        (24, 25, 26, 27): (7, 8),
    }
    for run in multi:
        color = "#a855f7" if run["includes_below_cutoff"] else "#0f766e"
        axes[1].scatter(
            run["additive_predicted_log_odds"],
            run["log_p_six_minus_log_p_eight"],
            s=90,
            color=color,
            edgecolor="white",
            linewidth=0.8,
        )
        subset_key = tuple(run["subset_layers"])
        if subset_key in label_offsets:
            axes[1].annotate(
                run["subset_label"],
                (
                    run["additive_predicted_log_odds"],
                    run["log_p_six_minus_log_p_eight"],
                ),
                xytext=label_offsets[subset_key],
                textcoords="offset points",
                fontsize=8.5,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#94a3b8",
                    "linewidth": 0.7,
                },
            )
    values = [
        float(run[field])
        for run in multi
        for field in [
            "additive_predicted_log_odds",
            "log_p_six_minus_log_p_eight",
        ]
    ]
    lower = min(values) - 0.12
    upper = max(values) + 0.12
    axes[1].plot([lower, upper], [lower, upper], "--", color="#64748b")
    target_log_odds = math.log(target_probability / (1 - target_probability))
    axes[1].axhline(target_log_odds, color="#111827", linestyle=":", linewidth=1.4)
    axes[1].axvline(target_log_odds, color="#111827", linestyle=":", linewidth=1.4)
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    axes[1].set_xlabel("additive prediction: log P(Six) - log P(Eight)")
    axes[1].set_ylabel("observed log P(Six) - log P(Eight)")
    axes[1].set_title("Do layer effects add?", loc="left")
    axes[1].grid(alpha=0.22)
    axes[1].text(
        0.02,
        0.98,
        "Purple includes L25*, the below-cutoff exploratory feature",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
    )
    figure.suptitle(
        "Cumulative ant-feature intervention on the fixed L7+L22 base",
        fontsize=16,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_cumulative_path(
    *,
    runs: list[dict[str, Any]],
    primary_layers: tuple[int, ...],
    exploratory_layers: tuple[int, ...],
    target_probability: float,
    output_path: Path,
) -> None:
    by_subset = {tuple(run["subset_layers"]): run for run in runs}
    path: list[tuple[int, ...]] = [()]
    accumulated: list[int] = []
    for layer in primary_layers:
        accumulated.append(layer)
        path.append(tuple(sorted(accumulated)))
    for layer in exploratory_layers:
        accumulated.append(layer)
        path.append(tuple(sorted(accumulated)))
    selected = [by_subset[subset] for subset in path]
    labels = [run["subset_label"] for run in selected]
    x = np.arange(len(selected))

    figure, axis = plt.subplots(figsize=(12.5, 6.7), constrained_layout=True)
    axis.plot(
        x,
        [100 * run["p_six"] for run in selected],
        marker="o",
        linewidth=2.6,
        markersize=8,
        color="#0f766e",
        label="P(Six)",
    )
    axis.plot(
        x,
        [100 * run["p_eight"] for run in selected],
        marker="o",
        linewidth=2.4,
        markersize=7,
        color="#b45309",
        label="P(Eight)",
    )
    axis.plot(
        x,
        [100 * run["p_four"] for run in selected],
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
    for index, run in enumerate(selected):
        axis.annotate(
            f"{100 * run['p_six']:.3f}%",
            (index, 100 * run["p_six"]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("next-token probability (%)")
    axis.set_ylim(-1, 103)
    axis.set_title(
        "Primary candidates first; L25* is added last as a below-cutoff test",
        loc="left",
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    figure.suptitle("Cumulative path at natural-reference factor 1", fontsize=15)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "subset_layers",
        "subset_label",
        "subset_size",
        "includes_below_cutoff",
        "candidate_factor",
        "candidate_feature_ids",
        "candidate_current_activations",
        "candidate_target_activations",
        "candidate_delta_norm",
        "total_delta_norm",
        "delta_p_six_vs_base",
        "delta_log_odds_vs_base",
        "additive_predicted_log_odds",
        "additive_residual_log_odds",
        "additive_predicted_pairwise_p_six",
        "observed_pairwise_p_six",
        "mobius_log_odds_term",
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            row = {field: run[field] for field in fieldnames}
            for field in [
                "subset_layers",
                "candidate_feature_ids",
                "candidate_current_activations",
                "candidate_target_activations",
                "top_tokens",
            ]:
                row[field] = json.dumps(row[field], ensure_ascii=False)
            row["top_token"] = display_token(row["top_token"])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    primary_layers = tuple(sorted(set(args.primary_layers)))
    exploratory_layers = tuple(sorted(set(args.exploratory_layers)))
    candidate_layers = tuple(sorted(set(primary_layers + exploratory_layers)))
    if set(primary_layers) & set(exploratory_layers):
        raise RuntimeError("Primary and exploratory layer lists must be disjoint")
    if set(candidate_layers) & set(BASE_LAYERS):
        raise RuntimeError("Candidate layers cannot include frozen base layers 7 or 22")
    if args.candidate_factor <= 0:
        raise RuntimeError("--candidate-factor must be positive")
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
    coverage_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in coverage_manifest["selected_ant_instances"]:
        coverage_by_layer[int(instance["layer"])].append(instance)

    missing = sorted(set(candidate_layers) - set(coverage_by_layer))
    if missing:
        raise RuntimeError(f"No calibrated ant candidate is available in layers {missing}")
    candidate_instances = [
        instance
        for layer in candidate_layers
        for instance in coverage_by_layer[layer]
    ]
    primary_keys = {
        (int(instance["layer"]), int(instance["feature"]))
        for instance in candidate_instances
        if int(instance["layer"]) in primary_layers
    }
    exploratory_keys = {
        (int(instance["layer"]), int(instance["feature"]))
        for instance in candidate_instances
        if int(instance["layer"]) in exploratory_layers
    }
    for instance in candidate_instances:
        normalized = float(instance["normalized_reference_activation"])
        layer = int(instance["layer"])
        if layer in primary_layers and normalized < args.selection_cutoff:
            raise RuntimeError(f"Primary L{layer} candidate is below the cutoff")
        if layer in exploratory_layers and normalized >= args.selection_cutoff:
            raise RuntimeError(f"Exploratory L{layer} candidate is not below the cutoff")
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

    validated = validate_reference_instances(
        instances=base_ant_instances + candidate_instances,
        primary_keys=primary_keys,
        exploratory_keys=exploratory_keys,
        hidden=reference_hidden,
        rows=rows,
        reference_position=reference_position,
        cutoff=args.selection_cutoff,
        device=device,
    )
    validated_by_key = {
        (int(instance["layer"]), int(instance["feature"])): instance
        for instance in validated
    }
    base_ant_instances = [
        validated_by_key[(int(instance["layer"]), int(instance["feature"]))]
        for instance in base_ant_instances
    ]
    candidate_instances = [
        validated_by_key[(int(instance["layer"]), int(instance["feature"]))]
        for instance in candidate_instances
    ]

    all_tested_layers = tuple(sorted(set(BASE_LAYERS + candidate_layers)))
    spider_by_layer = grouped_with_empty_layers(spider_instances, all_tested_layers)
    ant_by_layer = grouped_with_empty_layers(
        base_ant_instances + candidate_instances, all_tested_layers
    )
    base_spider_by_layer = grouped_with_empty_layers(spider_instances, BASE_LAYERS)
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
    exploratory_set = set(exploratory_layers)
    for subset in all_subsets(candidate_layers):
        factors = {
            7: args.base_factor_l7,
            22: args.base_factor_l22,
            **{
                layer: args.candidate_factor if layer in subset else 0.0
                for layer in candidate_layers
            },
        }
        run = run_intervention(
            spider_by_layer=spider_by_layer,
            ant_by_layer=ant_by_layer,
            tested_layers=all_tested_layers,
            spider_suppression=args.spider_suppression,
            ant_factors=factors,
            **common,
        )
        annotate_run(
            run,
            subset=subset,
            candidate_layers=candidate_layers,
            exploratory_layers=exploratory_set,
            candidate_factor=args.candidate_factor,
            base=base_run,
            target_probability=args.target_probability,
        )
        if not subset:
            for metric in ["p_six", "p_eight", "p_four"]:
                if not math.isclose(
                    run[metric], base_run[metric], rel_tol=1e-6, abs_tol=1e-9
                ):
                    raise RuntimeError(
                        f"All-candidate factor-0 control differs from base for {metric}"
                    )
        runs.append(run)
        print(
            f"{run['subset_label']}: "
            f"P(Six)={run['p_six']:.6%}, "
            f"P(Eight)={run['p_eight']:.6%}, "
            f"P(Four)={run['p_four']:.6%}, "
            f"log-odds={run['log_p_six_minus_log_p_eight']:.6f}"
        )

    add_factorial_effects(runs, candidate_layers)
    singleton_checks = compare_prior_singletons(
        runs=runs,
        primary_layers=primary_layers,
        prior_path=args.prior_added_results,
        candidate_factor=args.candidate_factor,
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

    by_subset = {tuple(run["subset_layers"]): run for run in runs}
    full_primary = by_subset[tuple(sorted(primary_layers))]
    full_four_layer = by_subset[tuple(sorted(candidate_layers))]
    maximum = max(runs, key=lambda run: (run["p_six"], -run["p_four"]))
    target_runs = [run for run in runs if run["reaches_target_probability"]]
    smallest_target = (
        min(
            target_runs,
            key=lambda run: (
                run["subset_size"],
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
        "source_prior_added_results": str(args.prior_added_results),
        "selection_cutoff": args.selection_cutoff,
        "base_layers": list(BASE_LAYERS),
        "primary_layers": list(primary_layers),
        "exploratory_below_cutoff_layers": list(exploratory_layers),
        "candidate_factor": args.candidate_factor,
        "spider_suppression": args.spider_suppression,
        "base_ant_factors": {"7": args.base_factor_l7, "22": args.base_factor_l22},
        "ant_target_definition": (
            "current_activation + layer_factor * natural_explicit_ant_reference_activation"
        ),
        "spider_target_definition": "clean_activation * (1 - suppression); never negative",
        "factorial_definition": "all on/off subsets of L24-L27; enabled add-ons use factor 1",
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

    factorial_path = args.output_dir / FACTORIAL_FIGURE
    cumulative_path = args.output_dir / CUMULATIVE_FIGURE
    plot_factorial_summary(
        runs=runs,
        exploratory_layers=exploratory_set,
        target_probability=args.target_probability,
        output_path=factorial_path,
    )
    plot_cumulative_path(
        runs=runs,
        primary_layers=primary_layers,
        exploratory_layers=exploratory_layers,
        target_probability=args.target_probability,
        output_path=cumulative_path,
    )
    csv_path = args.output_dir / OUTPUT_CSV
    write_csv(csv_path, runs)
    result = {
        "experiment": "factorial cumulative ant layers on frozen L7+L22 SAE intervention",
        "model": MODEL_ID,
        "source_base_manifest": str(args.base_manifest),
        "source_coverage_manifest": str(args.coverage_manifest),
        "source_prior_added_results": str(args.prior_added_results),
        "selection_cutoff": args.selection_cutoff,
        "target_probability": args.target_probability,
        "candidate_factor": args.candidate_factor,
        "primary_layers": list(primary_layers),
        "exploratory_below_cutoff_layers": list(exploratory_layers),
        "clean_baseline": clean_baseline,
        "post_sweep_baseline": post_sweep_baseline,
        "explicit_ant_reference": reference_metrics,
        "frozen_base": base_run,
        "all_candidate_zero_control": compact_run(by_subset[()]),
        "prior_singleton_reproduction_checks": singleton_checks,
        "full_primary_run": compact_run(full_primary),
        "full_four_layer_run": compact_run(full_four_layer),
        "maximum_p_six_run": compact_run(maximum),
        "smallest_target_run": compact_run(smallest_target),
        "target_run_count": len(target_runs),
        "run_count": len(runs),
        "runs": runs,
        "artifacts": {
            "json": OUTPUT_JSON,
            "csv": OUTPUT_CSV,
            "manifest": FEATURE_MANIFEST,
            "factorial_figure": FACTORIAL_FIGURE,
            "cumulative_figure": CUMULATIVE_FIGURE,
        },
    }
    json_path = args.output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(
        "Full L24+L25*+L26+L27 result: "
        f"P(Six)={full_four_layer['p_six']:.6%}, "
        f"P(Eight)={full_four_layer['p_eight']:.6%}, "
        f"P(Four)={full_four_layer['p_four']:.6%}."
    )
    print(
        "Maximum factorial result: "
        f"{maximum['subset_label']}, P(Six)={maximum['p_six']:.6%}, "
        f"P(Eight)={maximum['p_eight']:.6%}."
    )
    print(
        f"Saved {json_path}, {csv_path}, {manifest_path}, "
        f"{factorial_path}, and {cumulative_path}."
    )


if __name__ == "__main__":
    main()
