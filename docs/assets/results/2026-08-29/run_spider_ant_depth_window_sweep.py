#!/usr/bin/env python3
"""Test cumulative spider-suppression windows from L22 through L27.

Every selected spider feature-position instance in the active depth window is
suppressed.  Ant features are injected only in layers where the ant reference
prompt naturally activates a saved feature above the normalized cutoff.  This
keeps the additional depth experiment distinct from inventing ant targets in
layers that did not calibrate.
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
    DETAIL_FIGURE,
    EXPLICIT_ANT_QUESTION,
    IMPLICIT_ANT_QUESTION,
    MODEL_ID,
    SPIDER_PROMPT,
    FeatureKey,
    best_run,
    calibrate_ant_reference,
    capture_hidden_and_logits,
    comma_floats,
    find_text_layers,
    group_by_layer,
    load_all_rows,
    measure_current_activations,
    plot_best_detail,
    plot_layer_heatmaps,
    probability_record,
    read_manifest,
    run_grid,
    select_spider_instances,
    single_token_id,
    token_rows,
    write_tidy_csv,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_JSON = "spider_ant_depth_window_sweep.json"
OUTPUT_CSV = "spider_ant_depth_window_sweep.csv"
FEATURE_MANIFEST = "spider_ant_depth_window_manifest.json"
SUMMARY_FIGURE = "spider_ant_depth_window_summary.png"
FULL_HEATMAP = "spider_ant_depth_window_full_heatmap.png"
CURVES_FIGURE = "spider_ant_depth_window_curves.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cumulative L22-L27 spider suppression with calibrated ant injection"
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
        default=Path("results/spider_ant_depth_window_sweep"),
    )
    parser.add_argument("--start-layer", type=int, default=22)
    parser.add_argument("--end-layer", type=int, default=27)
    parser.add_argument("--cutoff", type=float, default=0.4)
    parser.add_argument(
        "--spider-suppressions",
        type=comma_floats,
        default=comma_floats("0,0.2,0.4,0.6,0.8,1.0"),
    )
    parser.add_argument(
        "--ant-factors",
        type=comma_floats,
        default=comma_floats("0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0"),
    )
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--implicit-ant-question", default=IMPLICIT_ANT_QUESTION)
    parser.add_argument("--explicit-ant-question", default=EXPLICIT_ANT_QUESTION)
    parser.add_argument("--min-ant-p-six", type=float, default=0.9)
    parser.add_argument("--saturation-threshold", type=float, default=0.99)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def max_p_six_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(runs, key=lambda run: (run["p_six"], -run["p_four"]))


def plot_cumulative_summary(
    *,
    cumulative_runs: dict[str, list[dict[str, Any]]],
    baseline: dict[str, Any],
    output_path: Path,
) -> None:
    scopes = list(cumulative_runs)
    maxima = [max_p_six_run(cumulative_runs[scope]) for scope in scopes]
    x = np.arange(len(scopes))
    figure, axes = plt.subplots(
        1, 2, figsize=(16, 6.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    probability_axis = axes[0]
    probability_axis.plot(
        x, [100 * run["p_six"] for run in maxima], marker="o", linewidth=2.3,
        label="maximum P(Six)", color="#0f766e",
    )
    probability_axis.plot(
        x, [100 * run["p_eight"] for run in maxima], marker="o", linewidth=2.3,
        label="paired P(Eight)", color="#b45309",
    )
    probability_axis.plot(
        x, [100 * run["p_four"] for run in maxima], marker="o", linewidth=1.6,
        label="paired P(Four)", color="#4f46e5",
    )
    probability_axis.set_xticks(x, scopes, rotation=30, ha="right")
    probability_axis.set_ylabel("next-token probability (%)")
    probability_axis.set_title("Best sampled probability by cumulative depth window", loc="left")
    probability_axis.grid(alpha=0.25)
    probability_axis.legend()
    for index, run in enumerate(maxima):
        probability_axis.annotate(
            f"s={run['spider_suppression']:g}, a={run['ant_factor']:g}",
            (index, 100 * run["p_six"]),
            xytext=(0, 11), textcoords="offset points", ha="center", fontsize=8,
        )

    odds_axis = axes[1]
    baseline_odds = baseline["log_p_six_minus_log_p_eight"]
    improvements = [
        run["log_p_six_minus_log_p_eight"] - baseline_odds for run in maxima
    ]
    colors = ["#16a34a" if run["success"] else "#2563eb" for run in maxima]
    odds_axis.barh(x, improvements, color=colors)
    odds_axis.set_yticks(x, scopes)
    odds_axis.invert_yaxis()
    odds_axis.set_xlabel("improvement in log P(Six) - log P(Eight)")
    odds_axis.set_title("Maximum Six/Eight log-odds improvement", loc="left")
    odds_axis.grid(axis="x", alpha=0.25)
    for index, improvement in enumerate(improvements):
        odds_axis.text(improvement + 0.08, index, f"{improvement:+.2f}", va="center", fontsize=8)
    figure.suptitle(
        "Cumulative depth sweep: suppress spider features throughout the window; "
        "inject only calibrated ant layers",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_full_suppression_curves(
    *,
    cumulative_runs: dict[str, list[dict[str, Any]]],
    ant_factors: list[float],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 6.8), constrained_layout=True)
    color_map = plt.get_cmap("viridis")
    scopes = list(cumulative_runs)
    for index, scope in enumerate(scopes):
        runs = {
            run["ant_factor"]: run
            for run in cumulative_runs[scope]
            if run["spider_suppression"] == 1.0
        }
        color = color_map(index / max(1, len(scopes) - 1))
        axes[0].plot(
            ant_factors,
            [100 * runs[factor]["p_six"] for factor in ant_factors],
            marker="o", label=scope, color=color,
        )
        axes[1].plot(
            ant_factors,
            [100 * runs[factor]["p_eight"] for factor in ant_factors],
            marker="o", label=scope, color=color,
        )
    for axis, title in zip(axes, ["P(Six), %", "P(Eight), %"], strict=True):
        axis.set_xlabel("ant reference factor")
        axis.set_ylabel("next-token probability (%)")
        axis.set_title(title, loc="left")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Full spider suppression across cumulative L22-L27 windows", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_window_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    # Keep the fully detailed operation-level CSV produced by the shared writer.
    write_tidy_csv(path, runs)


def main() -> None:
    args = parse_args()
    if args.start_layer > args.end_layer:
        raise RuntimeError("--start-layer must be less than or equal to --end-layer")
    if not 0 <= args.cutoff:
        raise RuntimeError("--cutoff must be non-negative")
    if 1.0 not in args.spider_suppressions:
        raise RuntimeError("The cumulative curve requires spider suppression 1.0")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    requested_layers = list(range(args.start_layer, args.end_layer + 1))
    spider_manifest = read_manifest(args.spider_manifest)
    ant_manifest = read_manifest(args.ant_manifest)
    spider_instances = [
        instance
        for instance in select_spider_instances(spider_manifest, args.cutoff)
        if instance["layer"] in requested_layers
    ]
    spider_layers = sorted({instance["layer"] for instance in spider_instances})
    if spider_layers != requested_layers:
        missing = sorted(set(requested_layers) - set(spider_layers))
        raise RuntimeError(f"No qualifying spider instance in requested layers {missing}")
    ant_candidates = [
        feature
        for feature in ant_manifest["features"]
        if int(feature["layer"]) in requested_layers
    ]
    if not ant_candidates:
        raise RuntimeError("No saved ant candidate is present in the requested window")

    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for instance in spider_instances:
        features_by_layer[instance["layer"]].add(instance["feature"])
    for candidate in ant_candidates:
        features_by_layer[int(candidate["layer"])].add(int(candidate["feature"]))
    print(
        f"Selected {len(spider_instances)} spider instances across L{args.start_layer}-"
        f"L{args.end_layer}; loading saved ant candidates for calibration."
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
    spider_ids = tokenizer(
        args.spider_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    spider_tokens = token_rows(tokenizer, spider_ids)
    if len(spider_tokens) != 27:
        raise RuntimeError(f"Expected 27 spider-prompt tokens, found {len(spider_tokens)}")
    if spider_tokens[16]["token"].strip() != "spinning" or spider_tokens[21]["token"] != "\n":
        raise RuntimeError("Spider prompt positions 16 and 21 are not aligned as expected")

    print("Loading Gemma-3-4B-IT in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    layers = find_text_layers(model)
    device = next(model.parameters()).device
    spider_ids = spider_ids.to(device)
    spider_hidden, spider_logits = capture_hidden_and_logits(
        model, layers, spider_ids, requested_layers
    )
    baseline = probability_record(spider_logits, tokenizer, target_ids, args.top_k)
    if baseline["top_token"].strip().lower() != "eight" or baseline["p_eight"] < 0.99:
        raise RuntimeError(f"Spider baseline failed: {baseline}")

    calibration, ant_instances = calibrate_ant_reference(
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
    ant_instances = measure_current_activations(
        spider_hidden=spider_hidden,
        instances=ant_instances,
        rows=rows,
        device=device,
    )
    ant_layers = sorted({instance["layer"] for instance in ant_instances})
    print(
        f"Calibrated {len(ant_instances)} ant instances in layers {ant_layers}; "
        "other window layers receive spider suppression only."
    )

    spider_by_layer = group_by_layer(spider_instances)
    ant_by_layer = group_by_layer(ant_instances)
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

    cumulative_runs: dict[str, list[dict[str, Any]]] = {}
    all_runs: list[dict[str, Any]] = []
    for final_layer in requested_layers:
        tested_layers = tuple(range(args.start_layer, final_layer + 1))
        scope = f"L{args.start_layer}-L{final_layer}"
        print(f"Running cumulative window {scope}...")
        runs = run_grid(
            tested_layers=tested_layers,
            spider_suppressions=args.spider_suppressions,
            ant_factors=args.ant_factors,
            common=common,
            baseline=baseline,
        )
        for run in runs:
            run["scope"] = scope
        cumulative_runs[scope] = runs
        all_runs.extend(runs)

    full_scope = f"L{args.start_layer}-L{args.end_layer}"
    full_runs = cumulative_runs[full_scope]
    full_best = best_run(full_runs)
    full_maximum = max_p_six_run(full_runs)
    global_maximum = max_p_six_run(all_runs)
    successes = [run for run in all_runs if run["success"]]
    smallest_success = best_run(all_runs) if successes else None
    saturated = (
        full_maximum["success"]
        and full_maximum["p_six"] >= args.saturation_threshold
    )

    manifest = {
        "model": MODEL_ID,
        "created_at_unix": time.time(),
        "window_layers": requested_layers,
        "selection_cutoff": args.cutoff,
        "spider_prompt": args.spider_prompt,
        "spider_tokens": spider_tokens,
        "spider_instances": spider_instances,
        "ant_calibration": calibration,
        "eligible_ant_instances": ant_instances,
        "eligible_ant_layers": ant_layers,
        "spider_only_layers": sorted(set(requested_layers) - set(ant_layers)),
        "source_manifests": {
            "spider": str(args.spider_manifest),
            "ant": str(args.ant_manifest),
        },
    }
    manifest_path = args.output_dir / FEATURE_MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    summary_path = args.output_dir / SUMMARY_FIGURE
    plot_cumulative_summary(
        cumulative_runs=cumulative_runs,
        baseline=baseline,
        output_path=summary_path,
    )
    heatmap_path = args.output_dir / FULL_HEATMAP
    plot_layer_heatmaps(
        runs=full_runs,
        suppressions=args.spider_suppressions,
        factors=args.ant_factors,
        scope=full_scope,
        output_path=heatmap_path,
    )
    curves_path = args.output_dir / CURVES_FIGURE
    plot_full_suppression_curves(
        cumulative_runs=cumulative_runs,
        ant_factors=args.ant_factors,
        output_path=curves_path,
    )
    detail_path = args.output_dir / DETAIL_FIGURE
    plot_best_detail(
        runs=full_runs,
        suppressions=args.spider_suppressions,
        factors=args.ant_factors,
        selected=full_maximum,
        output_path=detail_path,
    )

    result = {
        "model": MODEL_ID,
        "experiment": "cumulative_spider_suppression_calibrated_ant_injection",
        "window_layers": requested_layers,
        "baseline": baseline,
        "selection_cutoff": args.cutoff,
        "spider_suppressions": args.spider_suppressions,
        "ant_factors": args.ant_factors,
        "eligible_ant_layers": ant_layers,
        "spider_only_layers": sorted(set(requested_layers) - set(ant_layers)),
        "saturation_threshold": args.saturation_threshold,
        "full_window_saturated": saturated,
        "scope_summaries": [
            {
                "scope": scope,
                "layers": cumulative_runs[scope][0]["tested_layers"],
                "spider_instance_count": sum(
                    len(spider_by_layer[layer])
                    for layer in cumulative_runs[scope][0]["tested_layers"]
                ),
                "ant_instance_count": sum(
                    len(ant_by_layer.get(layer, []))
                    for layer in cumulative_runs[scope][0]["tested_layers"]
                ),
                "success_count": sum(run["success"] for run in cumulative_runs[scope]),
                "smallest_success": (
                    best_run(cumulative_runs[scope])
                    if any(run["success"] for run in cumulative_runs[scope])
                    else None
                ),
                "maximum_p_six_run": max_p_six_run(cumulative_runs[scope]),
            }
            for scope in cumulative_runs
        ],
        "smallest_success": smallest_success,
        "global_maximum_p_six_run": global_maximum,
        "full_window_smallest_success": full_best if full_best["success"] else None,
        "full_window_maximum_p_six_run": full_maximum,
        "runs": all_runs,
        "artifacts": {
            "manifest": str(manifest_path),
            "csv": str(args.output_dir / OUTPUT_CSV),
            "summary": str(summary_path),
            "full_heatmap": str(heatmap_path),
            "curves": str(curves_path),
            "detail": str(detail_path),
        },
    }
    json_path = args.output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    csv_path = args.output_dir / OUTPUT_CSV
    write_window_csv(csv_path, all_runs)

    print(
        json.dumps(
            {
                "baseline_p_eight": baseline["p_eight"],
                "eligible_ant_layers": ant_layers,
                "spider_only_layers": result["spider_only_layers"],
                "full_window_saturated": saturated,
                "full_window_maximum": {
                    key: full_maximum[key]
                    for key in [
                        "scope", "spider_suppression", "ant_factor", "success",
                        "top_token", "p_six", "p_eight", "p_four",
                    ]
                },
                "json": str(json_path),
                "csv": str(csv_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
