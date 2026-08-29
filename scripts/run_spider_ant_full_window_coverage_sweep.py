#!/usr/bin/env python3
"""Sweep L22-L27 with spider suppression and ant injection in every layer.

The L22 and L27 ant targets are held fixed from the calibrated experiment.  For
L23-L26, a broader bounded Neuronpedia search is evaluated on the explicit ant
reference.  The strongest naturally active candidate in each missing layer is
added.  Candidates below the normalized 0.4 cutoff remain explicitly labelled
as coverage fallbacks; no target magnitude is invented from maxActApprox.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import requests
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
    feature_activation,
    find_explicit_ant_position,
    find_text_layers,
    find_turn_boundary_position,
    group_by_layer,
    load_all_rows,
    measure_current_activations,
    plot_best_detail,
    plot_layer_heatmaps,
    probability_record,
    read_manifest,
    request_with_retries,
    run_grid,
    select_spider_instances,
    single_token_id,
    token_rows,
    write_tidy_csv,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEARCH_URL = "https://www.neuronpedia.org/api/explanation/search-model"
SOURCE_SUFFIX = "gemmascope-2-transcoder-262k"
ANT_PATTERN = re.compile(r"(^|[^A-Za-z])ants?([^A-Za-z]|$)|myrmec|formicid", re.I)
SEARCH_QUERIES = (
    "ant ants insect six-legged colony eusocial",
    "ants ant colonies social insects",
    "ant insect formicid myrmecology",
)
SEARCH_OFFSETS = tuple(range(0, 401, 20))
PRESERVED_TARGETS = {22: (1703, 197021), 27: (248182,)}
OUTPUT_JSON = "spider_ant_full_window_coverage_sweep.json"
OUTPUT_CSV = "spider_ant_full_window_coverage_sweep.csv"
FEATURE_MANIFEST = "spider_ant_full_window_coverage_manifest.json"
HEATMAP_FIGURE = "spider_ant_full_window_coverage_heatmap.png"
DETAIL_FIGURE = "spider_ant_full_window_coverage_detail.png"
CURVE_FIGURE = "spider_ant_full_window_coverage_curve.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="L22-L27 spider suppression plus explicit-ant coverage injection"
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
        default=Path("results/spider_ant_full_window_coverage_sweep"),
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
        default=comma_floats("0,0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0,4.0,5.0"),
    )
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--explicit-ant-question", default=EXPLICIT_ANT_QUESTION)
    parser.add_argument("--min-ant-p-six", type=float, default=0.9)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--search-workers", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def search_page(query: str, offset: int) -> list[dict[str, Any]]:
    response = requests.post(
        SEARCH_URL,
        json={"modelId": "gemma-3-4b-it", "query": query, "offset": offset},
        timeout=45,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def discover_expanded_candidates(
    *,
    start_layer: int,
    end_layer: int,
    existing_manifest: dict[str, Any],
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: dict[FeatureKey, dict[str, Any]] = {}
    for feature in existing_manifest["features"]:
        layer = int(feature["layer"])
        if not start_layer <= layer <= end_layer:
            continue
        candidates[FeatureKey(layer, int(feature["feature"]))] = {
            "layer": layer,
            "feature": int(feature["feature"]),
            "description": feature["description"],
            "cosine_similarity": feature.get("cosine_similarity"),
            "max_act_approx": feature.get("max_act_approx"),
            "source": feature.get("source"),
            "forced": bool(feature.get("forced", False)),
            "discovery": "existing_manifest",
        }

    retrieved_pages = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(search_page, query, offset): (query, offset)
            for query in SEARCH_QUERIES
            for offset in SEARCH_OFFSETS
        }
        for future in as_completed(futures):
            query, offset = futures[future]
            results = future.result()
            retrieved_pages += 1
            for result in results:
                source = str(result.get("layer", ""))
                match = re.fullmatch(rf"(\d+)-{re.escape(SOURCE_SUFFIX)}", source)
                description = str(result.get("description", ""))
                if match is None or ANT_PATTERN.search(description) is None:
                    continue
                layer = int(match.group(1))
                if not start_layer <= layer <= end_layer:
                    continue
                neuron = result.get("neuron") or {}
                max_act = neuron.get("maxActApprox")
                candidate = {
                    "layer": layer,
                    "feature": int(result["index"]),
                    "description": description,
                    "cosine_similarity": float(result["cosine_similarity"]),
                    "max_act_approx": float(max_act) if max_act is not None else None,
                    "source": source,
                    "forced": False,
                    "discovery": "expanded_search",
                    "search_query": query,
                    "search_offset": offset,
                }
                key = FeatureKey(layer, candidate["feature"])
                previous = candidates.get(key)
                if previous is None or (
                    candidate["cosine_similarity"] or 0
                ) > (previous.get("cosine_similarity") or 0):
                    candidates[key] = candidate

    by_layer = defaultdict(int)
    for candidate in candidates.values():
        by_layer[candidate["layer"]] += 1
    missing = [layer for layer in range(start_layer, end_layer + 1) if not by_layer[layer]]
    if missing:
        raise RuntimeError(f"Expanded search found no ant candidate in layers {missing}")
    return sorted(
        candidates.values(), key=lambda item: (item["layer"], item["feature"])
    ), {
        "url": SEARCH_URL,
        "queries": list(SEARCH_QUERIES),
        "offsets": list(SEARCH_OFFSETS),
        "retrieved_page_count": retrieved_pages,
        "candidate_count_by_layer": dict(sorted(by_layer.items())),
    }


def evaluate_explicit_reference(
    *,
    model: torch.nn.Module,
    layers: Any,
    tokenizer: Any,
    device: torch.device,
    candidates: list[dict[str, Any]],
    rows: dict[FeatureKey, dict[str, Any]],
    target_ids: dict[str, int],
    question: str,
    minimum_p_six: float,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prompt = PROMPT_TEMPLATE.format(question=question)
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    tokens = token_rows(tokenizer, ids)
    concept_position = find_explicit_ant_position(tokens)
    boundary_position = find_turn_boundary_position(tokens)
    candidate_layers = sorted({candidate["layer"] for candidate in candidates})
    hidden, logits = capture_hidden_and_logits(
        model, layers, ids.to(device), candidate_layers
    )
    metrics = probability_record(logits, tokenizer, target_ids, top_k)
    if metrics["top_token"].strip().lower() != "six" or metrics["p_six"] < minimum_p_six:
        raise RuntimeError(f"Explicit ant reference failed: {metrics}")

    measurements: list[dict[str, Any]] = []
    for candidate in candidates:
        key = FeatureKey(candidate["layer"], candidate["feature"])
        max_act = candidate.get("max_act_approx")
        for target_position, source_position in [
            (16, concept_position),
            (21, boundary_position),
        ]:
            preactivation, activation = feature_activation(
                hidden[key.layer], source_position, rows[key], device
            )
            normalized = (
                activation / float(max_act)
                if max_act is not None and float(max_act) > 0
                else None
            )
            measurements.append(
                {
                    **candidate,
                    "role": "ant_candidate",
                    "position": target_position,
                    "reference_position": source_position,
                    "reference_token": tokens[source_position]["token"],
                    "reference_preactivation": preactivation,
                    "reference_activation": activation,
                    "normalized_reference_activation": normalized,
                    "learned_threshold": float(rows[key]["threshold"]),
                    "naturally_active": activation > 0,
                }
            )
    return {
        "prompt": prompt,
        "tokens": tokens,
        "metrics": metrics,
        "concept_position": concept_position,
        "boundary_position": boundary_position,
    }, measurements


def select_coverage_targets(
    measurements: list[dict[str, Any]],
    cutoff: float,
    requested_layers: list[int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    by_key = {
        (item["layer"], item["feature"], item["position"]): item
        for item in measurements
    }
    for layer, features in PRESERVED_TARGETS.items():
        if layer not in requested_layers:
            continue
        for feature in features:
            item = by_key.get((layer, feature, 16))
            if item is None or not item["naturally_active"]:
                raise RuntimeError(f"Preserved ant target L{layer} F{feature} is not naturally active")
            selected.append(
                {
                    **item,
                    "role": "ant",
                    "selection": "preserved_calibrated_target",
                    "passes_normalized_cutoff": (
                        item["normalized_reference_activation"] is not None
                        and item["normalized_reference_activation"] >= cutoff
                    ),
                }
            )

    missing_layers = sorted(set(requested_layers) - set(PRESERVED_TARGETS))
    for layer in missing_layers:
        pool = [
            item for item in measurements
            if item["layer"] == layer
            and item["position"] == 16
            and item["naturally_active"]
            and item["normalized_reference_activation"] is not None
        ]
        if not pool:
            raise RuntimeError(f"No naturally active explicit-ant candidate at L{layer}")
        strict = [item for item in pool if item["normalized_reference_activation"] >= cutoff]
        chosen = max(
            strict or pool,
            key=lambda item: (
                item["normalized_reference_activation"],
                item.get("cosine_similarity") or 0,
            ),
        )
        selected.append(
            {
                **chosen,
                "role": "ant",
                "selection": (
                    "expanded_search_above_cutoff"
                    if strict
                    else "expanded_search_below_cutoff_coverage"
                ),
                "passes_normalized_cutoff": bool(strict),
            }
        )
    return sorted(
        selected, key=lambda item: (item["layer"], item["feature"], item["position"])
    )


def max_p_six_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(runs, key=lambda run: (run["p_six"], -run["p_four"]))


def smallest_success(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    successes = [run for run in runs if run["success"]]
    if not successes:
        return None
    return min(
        successes,
        key=lambda run: (
            math.hypot(run["spider_suppression"], run["ant_factor"]),
            run["total_delta_norm"],
        ),
    )


def plot_strength_curve(
    *, runs: list[dict[str, Any]], ant_factors: list[float], output_path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    full = {
        run["ant_factor"]: run for run in runs if run["spider_suppression"] == 1.0
    }
    axis.plot(ant_factors, [100 * full[f]["p_six"] for f in ant_factors], marker="o", linewidth=2.3, label="P(Six)", color="#0f766e")
    axis.plot(ant_factors, [100 * full[f]["p_eight"] for f in ant_factors], marker="o", linewidth=2.3, label="P(Eight)", color="#b45309")
    axis.plot(ant_factors, [100 * full[f]["p_four"] for f in ant_factors], marker="o", linewidth=1.7, label="P(Four)", color="#4f46e5")
    crossing = next((full[f] for f in ant_factors if full[f]["success"]), None)
    if crossing:
        axis.axvline(crossing["ant_factor"], linestyle="--", color="#111827", alpha=0.7)
        axis.annotate(
            f"first sampled success: a={crossing['ant_factor']:g}",
            (crossing["ant_factor"], 100 * crossing["p_six"]),
            xytext=(10, 18), textcoords="offset points", arrowprops={"arrowstyle": "->"},
        )
    axis.set_xlabel("ant reference factor")
    axis.set_ylabel("next-token probability (%)")
    axis.set_title("L22-L27 ant coverage with full spider suppression", loc="left")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    requested_layers = list(range(args.start_layer, args.end_layer + 1))
    if requested_layers != list(range(22, 28)):
        raise RuntimeError("This controlled fallback currently preserves targets for L22-L27 only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spider_manifest = read_manifest(args.spider_manifest)
    existing_ant_manifest = read_manifest(args.ant_manifest)
    spider_instances = [
        item for item in select_spider_instances(spider_manifest, args.cutoff)
        if item["layer"] in requested_layers
    ]
    expanded_candidates, discovery = discover_expanded_candidates(
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        existing_manifest=existing_ant_manifest,
        workers=args.search_workers,
    )
    print(
        f"Discovered {len(expanded_candidates)} unique ant-labelled candidates across "
        f"L{args.start_layer}-L{args.end_layer}."
    )

    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for item in spider_instances:
        features_by_layer[item["layer"]].add(item["feature"])
    for item in expanded_candidates:
        if item.get("max_act_approx") is not None:
            features_by_layer[item["layer"]].add(item["feature"])
    expanded_candidates = [
        item for item in expanded_candidates
        if FeatureKey(item["layer"], item["feature"]).feature
        in features_by_layer[item["layer"]]
    ]
    rows = load_all_rows(dict(features_by_layer), args.download_workers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    target_ids = {name: single_token_id(tokenizer, f" {name}") for name in ["Six", "Eight", "Four"]}
    spider_ids = tokenizer(args.spider_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    spider_tokens = token_rows(tokenizer, spider_ids)
    if len(spider_tokens) != 27 or spider_tokens[16]["token"].strip() != "spinning" or spider_tokens[21]["token"] != "\n":
        raise RuntimeError("Spider prompt token alignment failed")

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
    spider_hidden, spider_logits = capture_hidden_and_logits(model, layers, spider_ids, requested_layers)
    baseline = probability_record(spider_logits, tokenizer, target_ids, args.top_k)
    if baseline["top_token"].strip().lower() != "eight" or baseline["p_eight"] < 0.99:
        raise RuntimeError(f"Spider baseline failed: {baseline}")

    reference, measurements = evaluate_explicit_reference(
        model=model,
        layers=layers,
        tokenizer=tokenizer,
        device=device,
        candidates=expanded_candidates,
        rows=rows,
        target_ids=target_ids,
        question=args.explicit_ant_question,
        minimum_p_six=args.min_ant_p_six,
        top_k=args.top_k,
    )
    ant_instances = select_coverage_targets(measurements, args.cutoff, requested_layers)
    ant_instances = measure_current_activations(
        spider_hidden=spider_hidden,
        instances=ant_instances,
        rows=rows,
        device=device,
    )
    if sorted({item["layer"] for item in ant_instances}) != requested_layers:
        raise RuntimeError("Ant coverage selection did not cover every requested layer")
    print("Selected ant coverage targets:")
    for item in ant_instances:
        print(
            f"  L{item['layer']} F{item['feature']} normalized="
            f"{item['normalized_reference_activation']:.4f} {item['selection']}"
        )

    common = {
        "model": model,
        "layers": layers,
        "input_ids": spider_ids,
        "tokenizer": tokenizer,
        "target_ids": target_ids,
        "rows": rows,
        "spider_by_layer": group_by_layer(spider_instances),
        "ant_by_layer": group_by_layer(ant_instances),
        "top_k": args.top_k,
        "device": device,
    }
    tested_layers = tuple(requested_layers)
    runs = run_grid(
        tested_layers=tested_layers,
        spider_suppressions=args.spider_suppressions,
        ant_factors=args.ant_factors,
        common=common,
        baseline=baseline,
    )
    for run in runs:
        run["scope"] = "L22-L27-ant-coverage"
    maximum = max_p_six_run(runs)
    first_success = smallest_success(runs)

    manifest = {
        "model": MODEL_ID,
        "created_at_unix": time.time(),
        "window_layers": requested_layers,
        "selection_cutoff": args.cutoff,
        "discovery": discovery,
        "spider_prompt": args.spider_prompt,
        "spider_tokens": spider_tokens,
        "spider_instances": spider_instances,
        "explicit_ant_reference": reference,
        "expanded_candidate_count": len(expanded_candidates),
        "expanded_candidate_measurements": measurements,
        "selected_ant_instances": ant_instances,
        "below_cutoff_coverage_layers": [
            item["layer"] for item in ant_instances
            if not item["passes_normalized_cutoff"]
        ],
    }
    manifest_path = args.output_dir / FEATURE_MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    heatmap_path = args.output_dir / HEATMAP_FIGURE
    plot_layer_heatmaps(
        runs=runs,
        suppressions=args.spider_suppressions,
        factors=args.ant_factors,
        scope="L22-L27 ant coverage",
        output_path=heatmap_path,
    )
    detail_path = args.output_dir / DETAIL_FIGURE
    plot_best_detail(
        runs=runs,
        suppressions=args.spider_suppressions,
        factors=args.ant_factors,
        selected=maximum,
        output_path=detail_path,
    )
    curve_path = args.output_dir / CURVE_FIGURE
    plot_strength_curve(runs=runs, ant_factors=args.ant_factors, output_path=curve_path)

    result = {
        "model": MODEL_ID,
        "experiment": "L22_L27_spider_suppression_and_ant_coverage_injection",
        "window_layers": requested_layers,
        "baseline": baseline,
        "selection_cutoff": args.cutoff,
        "spider_suppressions": args.spider_suppressions,
        "ant_factors": args.ant_factors,
        "selected_ant_layers": sorted({item["layer"] for item in ant_instances}),
        "below_cutoff_coverage_layers": manifest["below_cutoff_coverage_layers"],
        "smallest_success": first_success,
        "maximum_p_six_run": maximum,
        "success_count": sum(run["success"] for run in runs),
        "runs": runs,
        "artifacts": {
            "manifest": str(manifest_path),
            "csv": str(args.output_dir / OUTPUT_CSV),
            "heatmap": str(heatmap_path),
            "detail": str(detail_path),
            "curve": str(curve_path),
        },
    }
    json_path = args.output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    csv_path = args.output_dir / OUTPUT_CSV
    write_tidy_csv(csv_path, runs)
    print(
        json.dumps(
            {
                "below_cutoff_coverage_layers": result["below_cutoff_coverage_layers"],
                "success_count": result["success_count"],
                "smallest_success": (
                    {key: first_success[key] for key in ["spider_suppression", "ant_factor", "p_six", "p_eight", "p_four"]}
                    if first_success else None
                ),
                "maximum": {key: maximum[key] for key in ["spider_suppression", "ant_factor", "top_token", "p_six", "p_eight", "p_four"]},
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
