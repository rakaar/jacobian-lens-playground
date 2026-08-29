#!/usr/bin/env python3
"""Find concept-labelled transcoders across layers and test them on the prompt."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
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


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ID = "google/gemma-3-4b-it"
NEURONPEDIA_MODEL_ID = "gemma-3-4b-it"
SOURCE_SUFFIX = "gemmascope-2-transcoder-262k"
SEARCH_URL = "https://www.neuronpedia.org/api/explanation/search-model"
SEARCH_QUERY = "spider arachnid web-spinning animal"
SEARCH_OFFSETS = (0, 20, 40, 60, 80)
EXPLICIT_SPIDER_TERMS = re.compile(r"spider|arachnid|web", re.IGNORECASE)
TRANSCODER_ROOT = (
    "https://huggingface.co/mwhanna/gemma-scope-2-4b-it/resolve/main/"
    "transcoder_all/width_262k_l0_small_affine"
)
PROMPT = (
    "<bos><start_of_turn>user\n"
    "Answer in one word: How many legs does a web-spinning animal have?"
    "<end_of_turn>\n<start_of_turn>model\nAnswer:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/spider_layerwise_search"),
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--concept-name", default="spider")
    parser.add_argument("--candidate-label", default="spider/arachnid/web")
    parser.add_argument("--search-query", default=SEARCH_QUERY)
    parser.add_argument(
        "--description-regex", default=EXPLICIT_SPIDER_TERMS.pattern
    )
    parser.add_argument(
        "--force-feature",
        action="append",
        default=[],
        metavar="LAYER:FEATURE",
        help="Always include a feature, even if semantic search does not return it.",
    )
    parser.add_argument("--json-name", default="spider_layerwise_search.json")
    parser.add_argument("--csv-name", default="spider_layerwise_features.csv")
    parser.add_argument("--figure-name", default="spider_layerwise_search.png")
    return parser.parse_args()


def request_with_retries(
    method: str,
    url: str,
    *,
    timeout: int = 90,
    attempts: int = 4,
    **kwargs: Any,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def search_page(offset: int, search_query: str) -> list[dict[str, Any]]:
    response = request_with_retries(
        "POST",
        SEARCH_URL,
        json={
            "modelId": NEURONPEDIA_MODEL_ID,
            "query": search_query,
            "offset": offset,
        },
        timeout=45,
    )
    payload = response.json()
    return payload.get("results", [])


def fetch_forced_candidate(layer: int, feature: int) -> dict[str, Any]:
    source = f"{layer}-{SOURCE_SUFFIX}"
    url = (
        f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_MODEL_ID}/"
        f"{source}/{feature}"
    )
    payload = request_with_retries("GET", url, timeout=45).json()
    explanations = payload.get("explanations") or []
    description = (
        str(explanations[0].get("description"))
        if explanations
        else f"forced feature L{layer} F{feature}"
    )
    return {
        "layer": layer,
        "feature": feature,
        "description": description,
        "cosine_similarity": None,
        "max_act_approx": (
            float(payload["maxActApprox"])
            if payload.get("maxActApprox") is not None
            else None
        ),
        "search_rank": None,
        "source": source,
        "forced": True,
    }


def discover_candidates(
    search_query: str,
    description_pattern: re.Pattern[str],
    forced_features: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pages: dict[int, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=len(SEARCH_OFFSETS)) as executor:
        futures = {
            executor.submit(search_page, offset, search_query): offset
            for offset in SEARCH_OFFSETS
        }
        for future in as_completed(futures):
            pages[futures[future]] = future.result()

    raw_results = [result for offset in SEARCH_OFFSETS for result in pages[offset]]
    candidates_by_id: dict[tuple[int, int], dict[str, Any]] = {}
    source_pattern = re.compile(rf"^(\d+)-{re.escape(SOURCE_SUFFIX)}$")

    for rank, result in enumerate(raw_results):
        source = str(result.get("layer", ""))
        source_match = source_pattern.fullmatch(source)
        description = str(result.get("description", ""))
        if source_match is None or description_pattern.search(description) is None:
            continue

        neuron = result.get("neuron") or {}
        candidate = {
            "layer": int(source_match.group(1)),
            "feature": int(result["index"]),
            "description": description,
            "cosine_similarity": float(result["cosine_similarity"]),
            "max_act_approx": (
                float(neuron["maxActApprox"])
                if neuron.get("maxActApprox") is not None
                else None
            ),
            "search_rank": rank,
            "source": source,
            "forced": False,
        }
        key = (candidate["layer"], candidate["feature"])
        previous = candidates_by_id.get(key)
        if previous is None or candidate["cosine_similarity"] > previous["cosine_similarity"]:
            candidates_by_id[key] = candidate

    for forced_spec in forced_features:
        try:
            layer_text, feature_text = forced_spec.split(":", maxsplit=1)
            layer = int(layer_text)
            feature = int(feature_text)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid --force-feature {forced_spec!r}; expected LAYER:FEATURE"
            ) from error
        forced_candidate = fetch_forced_candidate(layer, feature)
        key = (layer, feature)
        if key in candidates_by_id:
            candidates_by_id[key]["forced"] = True
        else:
            candidates_by_id[key] = forced_candidate

    candidates = sorted(
        candidates_by_id.values(), key=lambda item: (item["layer"], item["feature"])
    )
    if not candidates:
        raise RuntimeError("Neuronpedia search returned no explicit concept candidates")
    return candidates, raw_results


def fetch_range(url: str, start: int, end: int) -> bytes:
    response = request_with_retries(
        "GET",
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=120,
    )
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"Server returned {len(response.content)} bytes for a {expected}-byte range"
        )
    return response.content


def tensor_dtype(metadata: dict[str, Any]) -> np.dtype:
    dtype = metadata["dtype"]
    if dtype != "F32":
        raise RuntimeError(f"Expected F32 transcoder tensors, found {dtype}")
    return np.dtype("<f4")


def load_layer_encoders(
    layer: int, candidates: list[dict[str, Any]]
) -> dict[tuple[int, int], dict[str, Any]]:
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
    encoder_metadata = header["W_enc"]
    encoder_dtype = tensor_dtype(encoder_metadata)
    encoder_start = data_base + encoder_metadata["data_offsets"][0]
    feature_count, width = encoder_metadata["shape"]
    row_bytes = width * encoder_dtype.itemsize

    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate in candidates:
        feature = candidate["feature"]
        if not 0 <= feature < feature_count:
            raise RuntimeError(f"Feature {feature} is outside layer {layer} width")
        start = encoder_start + feature * row_bytes
        raw = fetch_range(url, start, start + row_bytes - 1)
        rows[layer, feature] = {
            "W_enc": np.frombuffer(raw, dtype=encoder_dtype).copy(),
            "b_enc": float(b_enc[feature]),
            "threshold": float(thresholds[feature]),
        }
    return rows


def load_candidate_encoders(
    candidates: list[dict[str, Any]], workers: int
) -> dict[tuple[int, int], dict[str, Any]]:
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_layer[candidate["layer"]].append(candidate)

    rows: dict[tuple[int, int], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(load_layer_encoders, layer, layer_candidates): layer
            for layer, layer_candidates in by_layer.items()
        }
        for future in as_completed(futures):
            layer = futures[future]
            layer_rows = future.result()
            rows.update(layer_rows)
            print(f"Loaded {len(layer_rows)} candidate encoder rows from L{layer}")
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


def display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t") or "<empty>"


def write_feature_csv(path: Path, measurements: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layer",
        "feature",
        "description",
        "cosine_similarity",
        "max_act_approx",
        "threshold",
        "active",
        "peak_position",
        "peak_token",
        "peak_activation",
        "peak_fraction_of_reference_max",
        "nonzero_positions",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(
                {
                    key: (
                        json.dumps(measurement[key])
                        if key == "nonzero_positions"
                        else measurement[key]
                    )
                    for key in fieldnames
                }
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    description_pattern = re.compile(args.description_regex, re.IGNORECASE)
    print(f"Discovering {args.concept_name}-labelled Gemma transcoder candidates...")
    candidates, raw_search_results = discover_candidates(
        args.search_query,
        description_pattern,
        args.force_feature,
    )
    candidate_layers = sorted({candidate["layer"] for candidate in candidates})
    print(f"Found {len(candidates)} explicit candidates across {len(candidate_layers)} layers")

    print("Fetching only the selected transcoder encoder rows...")
    encoder_rows = load_candidate_encoders(candidates, args.download_workers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required to download gated Gemma weights")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    input_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    tokens = [tokenizer.decode([int(token_id)]) for token_id in input_ids[0]]
    if len(tokens) != 27:
        raise RuntimeError(f"Expected 27 prompt tokens, found {len(tokens)}")

    print("Loading Gemma-3-4B-IT and running the clean prompt once...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    layers = find_text_layers(model)
    if max(candidate_layers) >= len(layers):
        raise RuntimeError(
            f"Candidate L{max(candidate_layers)} exceeds the model's {len(layers)} layers"
        )

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in candidate_layers:
        layer = layers[layer_index]

        def capture(_module, _args, output, layer_index=layer_index):
            captured[layer_index] = output.detach()

        handles.append(layer.pre_feedforward_layernorm.register_forward_hook(capture))

    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()

    probabilities = torch.softmax(logits, dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=10)
    top_predictions = [
        {
            "token": tokenizer.decode([int(token_id)]),
            "token_id": int(token_id),
            "probability": float(probability),
        }
        for probability, token_id in zip(
            top_probabilities.cpu(), top_ids.cpu(), strict=True
        )
    ]
    if top_predictions[0]["token"].strip().lower() != "eight":
        raise RuntimeError(f"Clean model did not answer Eight: {top_predictions[0]}")

    measurements: list[dict[str, Any]] = []
    for candidate in candidates:
        layer = candidate["layer"]
        feature = candidate["feature"]
        row = encoder_rows[layer, feature]
        hidden = captured[layer][0].float()
        encoder = torch.from_numpy(row["W_enc"]).to(device=device, dtype=torch.float32)
        preactivations = hidden @ encoder + row["b_enc"]
        activations = torch.where(
            preactivations > row["threshold"],
            preactivations,
            torch.zeros_like(preactivations),
        )
        activation_values = activations.cpu().tolist()
        peak_position = int(torch.argmax(activations).item())
        peak_activation = float(activations[peak_position])
        nonzero_positions = [
            position for position, activation in enumerate(activation_values) if activation > 0
        ]
        max_act_approx = candidate["max_act_approx"]
        measurements.append(
            {
                **candidate,
                "threshold": row["threshold"],
                "preactivations": preactivations.cpu().tolist(),
                "activations": activation_values,
                "active": bool(nonzero_positions),
                "nonzero_positions": nonzero_positions,
                "peak_position": peak_position if nonzero_positions else None,
                "peak_token": tokens[peak_position] if nonzero_positions else None,
                "peak_activation": peak_activation,
                "peak_fraction_of_reference_max": (
                    peak_activation / max_act_approx
                    if max_act_approx is not None and max_act_approx > 0
                    else None
                ),
            }
        )

    layer_summaries = []
    for layer_index in range(len(layers)):
        layer_measurements = [
            measurement for measurement in measurements if measurement["layer"] == layer_index
        ]
        active_measurements = [
            measurement for measurement in layer_measurements if measurement["active"]
        ]
        top_measurement = max(
            active_measurements,
            key=lambda item: item["peak_fraction_of_reference_max"] or 0.0,
            default=None,
        )
        layer_summaries.append(
            {
                "layer": layer_index,
                "candidate_count": len(layer_measurements),
                "active_feature_count": len(active_measurements),
                "active_positions": sorted(
                    {
                        position
                        for measurement in active_measurements
                        for position in measurement["nonzero_positions"]
                    }
                ),
                "top_feature": (
                    {
                        key: top_measurement[key]
                        for key in [
                            "feature",
                            "description",
                            "peak_position",
                            "peak_token",
                            "peak_activation",
                            "peak_fraction_of_reference_max",
                        ]
                    }
                    if top_measurement is not None
                    else None
                ),
            }
        )

    result = {
        "model": MODEL_ID,
        "concept": args.concept_name,
        "prompt": PROMPT,
        "candidate_definition": {
            "search_url": SEARCH_URL,
            "query": args.search_query,
            "offsets": list(SEARCH_OFFSETS),
            "retrieved_result_count": len(raw_search_results),
            "source_suffix": SOURCE_SUFFIX,
            "explicit_description_regex": description_pattern.pattern,
            "forced_features": args.force_feature,
        },
        "tokens": [
            {"position": position, "token": token}
            for position, token in enumerate(tokens)
        ],
        "top_predictions": top_predictions,
        "candidate_count": len(candidates),
        "candidate_layers": candidate_layers,
        "active_feature_count": sum(measurement["active"] for measurement in measurements),
        "active_layers": [
            summary["layer"]
            for summary in layer_summaries
            if summary["active_feature_count"] > 0
        ],
        "layer_summaries": layer_summaries,
        "features": measurements,
    }
    json_path = args.output_dir / args.json_name
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    csv_path = args.output_dir / args.csv_name
    write_feature_csv(csv_path, measurements)

    layer_count = len(layers)
    token_count = len(tokens)
    heatmap = np.full((layer_count, token_count), np.nan, dtype=np.float32)
    candidate_counts = np.zeros(layer_count, dtype=np.int32)
    active_counts = np.zeros(layer_count, dtype=np.int32)
    for summary in layer_summaries:
        layer = summary["layer"]
        candidate_counts[layer] = summary["candidate_count"]
        active_counts[layer] = summary["active_feature_count"]
        if summary["candidate_count"]:
            heatmap[layer] = 0.0

    for measurement in measurements:
        max_act_approx = measurement["max_act_approx"]
        if max_act_approx is None or max_act_approx <= 0:
            continue
        relative = np.asarray(measurement["activations"], dtype=np.float32) / max_act_approx
        heatmap[measurement["layer"]] = np.maximum(
            heatmap[measurement["layer"]], relative
        )

    positive_values = heatmap[np.isfinite(heatmap) & (heatmap > 0)]
    color_max = float(np.percentile(positive_values, 98)) if positive_values.size else 1.0
    color_max = max(color_max, 0.05)

    figure = plt.figure(figsize=(20, 14), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[5.8, 1.3])
    heat_axis = figure.add_subplot(grid[0, 0])
    count_axis = figure.add_subplot(grid[0, 1], sharey=heat_axis)

    color_map = plt.get_cmap("magma").copy()
    color_map.set_bad("#d1d5db")
    image = heat_axis.imshow(
        np.ma.masked_invalid(heatmap),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=color_map,
        vmin=0.0,
        vmax=color_max,
    )
    tick_labels = [
        f"{position}: {display_token(token)}" for position, token in enumerate(tokens)
    ]
    heat_axis.set_xticks(np.arange(token_count))
    heat_axis.set_xticklabels(tick_labels, rotation=65, ha="right", fontsize=8)
    heat_axis.set_yticks(np.arange(layer_count))
    heat_axis.set_yticklabels([f"L{layer}" for layer in range(layer_count)], fontsize=8)
    heat_axis.set_xlabel("prompt token position")
    heat_axis.set_ylabel("transcoder layer")
    heat_axis.set_title(
        f"Strongest {args.concept_name}-labelled feature activation in each layer and token",
        loc="left",
    )
    for position in [16, 17, 21, 26]:
        heat_axis.axvline(position, color="#22d3ee", linewidth=0.8, alpha=0.8)
    color_bar = figure.colorbar(image, ax=heat_axis, fraction=0.028, pad=0.02)
    color_bar.set_label("activation / Neuronpedia approximate feature maximum")

    y = np.arange(layer_count)
    count_axis.barh(y, candidate_counts, color="#cbd5e1", label="labelled candidates")
    count_axis.barh(y, active_counts, color="#2563eb", label="active on prompt")
    count_axis.set_xlabel("feature count")
    count_axis.set_title("Candidates vs active", loc="left")
    count_axis.grid(axis="x", alpha=0.25)
    count_axis.tick_params(axis="y", labelleft=False)
    count_axis.legend(loc="lower right", fontsize=8)

    active_layer_count = int(np.count_nonzero(active_counts))
    figure.suptitle(
        f"Cross-layer {args.concept_name}-feature search for Gemma-3-4B-IT\n"
        f"{len(candidates)} explicit {args.candidate_label} candidates; "
        f"{int(active_counts.sum())} active across {active_layer_count} layers; "
        f"clean answer {top_predictions[0]['token']!r} "
        f"({top_predictions[0]['probability']:.3%})",
        fontsize=15,
    )
    figure_path = args.output_dir / args.figure_name
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(
        json.dumps(
            {
                "answer": top_predictions[0],
                "candidate_count": len(candidates),
                "candidate_layers": candidate_layers,
                "active_feature_count": result["active_feature_count"],
                "active_layers": result["active_layers"],
                "json": str(json_path),
                "csv": str(csv_path),
                "figure": str(figure_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
