#!/usr/bin/env python3
"""Read J-Lens concepts before and after the frozen L7+L22 intervention.

The causal intervention is the established Gemma Scope transcoder edit:

* fully suppress the selected spider features at L7/P16 and L22/P16+P21;
* inject the calibrated ant features at L7/P16 and L22/P16 with factor 4.

For clean, L7-only, L22-only, and full runs, this script captures the residual
stream after blocks L7 and L22.  It then applies Neuronpedia's fitted Gemma
J-Lens to the three edited cells and records top vocabulary readouts, pinned
concept-family scores, and clean-to-edited changes.  Residual vectors are saved
so a later experiment can compare the SAE delta with a J-Lens swap direction.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

from run_layerwise_spider_ant_sweep import (
    MODEL_ID,
    SPIDER_PROMPT,
    FeatureKey,
    display_token,
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


DEFAULT_FEATURE_MANIFEST = Path(
    "results/spider_ant_layer_specific_optimization/"
    "spider_ant_layer_specific_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("results/sae_jlens_before_after")
DEFAULT_LENS_REPO = "neuronpedia/jacobian-lens"
DEFAULT_LENS_FILENAME = (
    "gemma-3-4b-it/jlens/Salesforce-wikitext/"
    "gemma-3-4b-it_jacobian_lens.pt"
)
CAPTURE_CELLS = ((7, 16), (22, 16), (22, 21))
TESTED_LAYERS = (7, 22)
PRIMARY_CONDITION = "L7_L22"

OUTPUT_JSON = "sae_jlens_before_after.json"
FEATURE_MANIFEST = "sae_jlens_before_after_manifest.json"
TOP_CONCEPTS_CSV = "sae_jlens_top_concepts.csv"
TRACKED_CONCEPTS_CSV = "sae_jlens_tracked_concepts.csv"
CONTRASTS_CSV = "sae_jlens_contrasts.csv"
DELTA_CONCEPTS_CSV = "sae_jlens_delta_concepts.csv"
RESIDUAL_STATES = "sae_jlens_residual_states.pt"
TOP_CONCEPTS_FIGURE = "sae_jlens_top_concepts_before_after.png"
TRACKED_CHANGES_FIGURE = "sae_jlens_tracked_concept_changes.png"


CONCEPT_SURFACES: dict[str, tuple[str, ...]] = {
    "spider": (" spider", "spider", " Spider", "Spider", " spiders", "spiders"),
    "ant": (" ant", "ant", " Ant", "Ant", " ants", "ants"),
    "insect": (" insect", "insect", " insects", "insects"),
    "six": (" Six", "Six", " six", "six", " 6", "6"),
    "eight": (" Eight", "Eight", " eight", "eight", " 8", "8"),
    "four": (" Four", "Four", " four", "four", " 4", "4"),
    "web": (" web", "web", " webs", "webs", " spinning", "spinning"),
}

CONTRAST_PAIRS: dict[str, tuple[str, str]] = {
    "ant_minus_spider": ("ant", "spider"),
    "insect_minus_spider": ("insect", "spider"),
    "six_minus_eight": ("six", "eight"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare J-Lens concepts before and after the frozen SAE edit"
    )
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--lens-filename", default=DEFAULT_LENS_FILENAME)
    parser.add_argument("--lens-path", type=Path, default=None)
    parser.add_argument("--spider-suppression", type=float, default=1.0)
    parser.add_argument("--ant-factor-l7", type=float, default=4.0)
    parser.add_argument("--ant-factor-l22", type=float, default=4.0)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--figure-top-k", type=int, default=12)
    return parser.parse_args()


def output_tensor(output: Any) -> torch.Tensor:
    tensor = output if torch.is_tensor(output) else output[0]
    if not torch.is_tensor(tensor):
        raise RuntimeError(f"Expected a tensor block output, found {type(tensor)!r}")
    return tensor


def is_readable_token(token: str) -> bool:
    stripped = token.strip()
    if not stripped or not stripped.isascii():
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        return False
    return any(character.isalnum() for character in stripped)


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def snapshot_revision(path: Path) -> str | None:
    parts = path.parts
    if "snapshots" not in parts:
        return None
    index = parts.index("snapshots")
    return parts[index + 1] if index + 1 < len(parts) else None


def load_selected_jacobians(
    *,
    lens_repo: str,
    lens_filename: str,
    lens_path: Path | None,
    selected_layers: Iterable[int],
) -> tuple[Path, dict[int, torch.Tensor], int, int]:
    if lens_path is None:
        resolved = Path(
            hf_hub_download(
                repo_id=lens_repo,
                filename=lens_filename,
                token=os.environ.get("HF_TOKEN") or None,
            )
        )
    else:
        resolved = lens_path.expanduser().resolve()
        if not resolved.is_file():
            raise RuntimeError(f"Missing local J-Lens checkpoint: {resolved}")

    checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    jacobian_map = checkpoint.get("J") or checkpoint.get("jacobians")
    if not isinstance(jacobian_map, dict):
        raise RuntimeError(
            f"J-Lens checkpoint has no J mapping; keys={sorted(checkpoint)}"
        )

    selected: dict[int, torch.Tensor] = {}
    for layer in sorted(set(selected_layers)):
        value = jacobian_map.get(layer)
        if value is None:
            value = jacobian_map.get(str(layer))
        if value is None:
            raise RuntimeError(f"J-Lens checkpoint does not contain L{layer}")
        matrix = value.float().contiguous()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise RuntimeError(f"Unexpected L{layer} Jacobian shape: {tuple(matrix.shape)}")
        if not torch.isfinite(matrix).all():
            raise RuntimeError(f"L{layer} Jacobian contains non-finite values")
        selected[layer] = matrix

    d_model = int(checkpoint.get("d_model", next(iter(selected.values())).shape[0]))
    n_prompts = int(checkpoint.get("n_prompts", -1))
    del checkpoint, jacobian_map
    return resolved, selected, n_prompts, d_model


def surface_ids(tokenizer: Any) -> dict[str, list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = {}
    for family, surfaces in CONCEPT_SURFACES.items():
        unique: dict[int, str] = {}
        for surface in surfaces:
            encoded = tokenizer.encode(surface, add_special_tokens=False)
            if len(encoded) == 1:
                unique.setdefault(int(encoded[0]), surface)
        if not unique:
            raise RuntimeError(f"No single-token variants found for concept {family!r}")
        families[family] = [
            {
                "token_id": token_id,
                "surface": surface,
                "decoded": decode_token(tokenizer, token_id),
            }
            for token_id, surface in sorted(unique.items())
        ]
    return families


def group_instances(
    instances: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[int(instance["layer"])].append(instance)
    return {
        layer: sorted(values, key=lambda item: (int(item["position"]), int(item["feature"])))
        for layer, values in grouped.items()
    }


def run_condition(
    *,
    condition: str,
    active_layers: tuple[int, ...],
    ant_factors: dict[int, float],
    spider_suppression: float,
    model: torch.nn.Module,
    layers: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    target_ids: dict[str, int],
    rows: dict[FeatureKey, dict[str, Any]],
    spider_by_layer: dict[int, list[dict[str, Any]]],
    ant_by_layer: dict[int, list[dict[str, Any]]],
    device: torch.device,
    top_k: int,
) -> dict[str, Any]:
    states: dict[int, torch.Tensor] = {}
    residuals: dict[tuple[int, int], torch.Tensor] = {}
    operations: list[dict[str, Any]] = []
    handles: list[Any] = []

    cells_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, position in CAPTURE_CELLS:
        cells_by_layer[layer].append(position)

    for layer_index, positions in cells_by_layer.items():
        def capture_block(_module, _args, output, layer_index=layer_index, positions=positions):
            hidden = output_tensor(output)
            for position in positions:
                residuals[(layer_index, position)] = (
                    hidden[0, position].detach().float().cpu().clone()
                )

        handles.append(layers[layer_index].register_forward_hook(capture_block))

    for layer_index in active_layers:
        if layer_index not in ant_factors:
            raise RuntimeError(f"{condition}: no ant factor supplied for L{layer_index}")

        def capture_feature_input(_module, _args, output, layer_index=layer_index):
            states[layer_index] = output_tensor(output)

        def alter_mlp_write(_module, _args, output, layer_index=layer_index):
            if layer_index not in states:
                raise RuntimeError(f"{condition}: L{layer_index} edit ran before feature capture")
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"{condition}: expected tensor from L{layer_index} post-FF norm"
                )
            changed = output.clone()
            delta_by_position: dict[int, torch.Tensor] = {}

            for instance in spider_by_layer[layer_index]:
                key = FeatureKey(layer_index, int(instance["feature"]))
                row = rows[key]
                preactivation, activation = feature_activation(
                    states[layer_index], int(instance["position"]), row, device
                )
                target = max(
                    0.0,
                    float(instance["clean_activation"]) * (1.0 - spider_suppression),
                )
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target - activation) * decoder
                position = int(instance["position"])
                delta_by_position[position] = (
                    delta_by_position.get(position, torch.zeros_like(delta)) + delta
                )
                operations.append(
                    {
                        "role": "spider",
                        "layer": layer_index,
                        "feature": key.feature,
                        "position": position,
                        "current_preactivation": preactivation,
                        "current_activation": activation,
                        "target_activation": target,
                        "factor": -spider_suppression,
                        "delta_norm": float(delta.norm().item()),
                    }
                )

            factor = float(ant_factors[layer_index])
            for instance in ant_by_layer[layer_index]:
                key = FeatureKey(layer_index, int(instance["feature"]))
                row = rows[key]
                preactivation, activation = feature_activation(
                    states[layer_index], int(instance["position"]), row, device
                )
                target = activation + factor * float(instance["reference_activation"])
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target - activation) * decoder
                position = int(instance["position"])
                delta_by_position[position] = (
                    delta_by_position.get(position, torch.zeros_like(delta)) + delta
                )
                operations.append(
                    {
                        "role": "ant",
                        "layer": layer_index,
                        "feature": key.feature,
                        "position": position,
                        "current_preactivation": preactivation,
                        "current_activation": activation,
                        "reference_activation": float(instance["reference_activation"]),
                        "target_activation": target,
                        "factor": factor,
                        "delta_norm": float(delta.norm().item()),
                    }
                )

            for position, delta in delta_by_position.items():
                changed[0, position] = changed[0, position] + delta.to(output.dtype)
            return changed

        handles.append(
            layers[layer_index].pre_feedforward_layernorm.register_forward_hook(
                capture_feature_input
            )
        )
        handles.append(
            layers[layer_index].post_feedforward_layernorm.register_forward_hook(
                alter_mlp_write
            )
        )

    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(CAPTURE_CELLS) - set(residuals))
    if missing:
        raise RuntimeError(f"{condition}: failed to capture residual cells {missing}")

    return {
        "condition": condition,
        "active_layers": list(active_layers),
        "ant_factors": {str(layer): float(value) for layer, value in ant_factors.items()},
        "metrics": probability_record(logits, tokenizer, target_ids, top_k),
        "operations": operations,
        "residuals": residuals,
    }


def rank_for_id(logits: torch.Tensor, token_id: int) -> int:
    value = logits[int(token_id)]
    return int((logits > value).sum().item()) + 1


def lens_record(
    *,
    logits: torch.Tensor,
    tokenizer: Any,
    families: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> dict[str, Any]:
    logits = logits.detach().float().cpu()
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    count = min(top_k, logits.numel())
    top_values, top_ids = torch.topk(log_probabilities, k=count)
    top_tokens = []
    for rank, (log_probability, token_id) in enumerate(
        zip(top_values, top_ids, strict=True), start=1
    ):
        token_id_int = int(token_id)
        token = decode_token(tokenizer, token_id_int)
        top_tokens.append(
            {
                "rank": rank,
                "token_id": token_id_int,
                "token": token,
                "display_token": display_token(token),
                "logit": float(logits[token_id_int]),
                "log_probability": float(log_probability),
                "probability": float(probabilities[token_id_int]),
                "readable": is_readable_token(token),
            }
        )

    readable_top = [item for item in top_tokens if item["readable"]]
    if len(readable_top) < min(12, top_k):
        extra_count = min(max(top_k * 8, 200), logits.numel())
        _, extra_ids = torch.topk(log_probabilities, k=extra_count)
        seen = {item["token_id"] for item in readable_top}
        for token_id in extra_ids:
            token_id_int = int(token_id)
            token = decode_token(tokenizer, token_id_int)
            if token_id_int in seen or not is_readable_token(token):
                continue
            readable_top.append(
                {
                    "rank": rank_for_id(logits, token_id_int),
                    "token_id": token_id_int,
                    "token": token,
                    "display_token": display_token(token),
                    "logit": float(logits[token_id_int]),
                    "log_probability": float(log_probabilities[token_id_int]),
                    "probability": float(probabilities[token_id_int]),
                    "readable": True,
                }
            )
            seen.add(token_id_int)
            if len(readable_top) >= top_k:
                break

    tracked: dict[str, dict[str, Any]] = {}
    for family, variants in families.items():
        ids = torch.tensor([item["token_id"] for item in variants], dtype=torch.long)
        family_log_probability = float(torch.logsumexp(log_probabilities[ids], dim=0))
        best_index = int(torch.argmax(logits[ids]).item())
        best = variants[best_index]
        token_id = int(best["token_id"])
        tracked[family] = {
            "family": family,
            "variant_count": len(variants),
            "token_ids": [int(item["token_id"]) for item in variants],
            "best_token_id": token_id,
            "best_token": decode_token(tokenizer, token_id),
            "best_surface": best["surface"],
            "best_rank": rank_for_id(logits, token_id),
            "best_logit": float(logits[token_id]),
            "family_log_probability": family_log_probability,
            "family_probability": math.exp(family_log_probability),
        }

    contrasts = {
        name: {
            "name": name,
            "positive_family": positive,
            "negative_family": negative,
            "log_probability_difference": (
                tracked[positive]["family_log_probability"]
                - tracked[negative]["family_log_probability"]
            ),
        }
        for name, (positive, negative) in CONTRAST_PAIRS.items()
    }

    return {
        "top_tokens": top_tokens,
        "readable_top_tokens": readable_top[:top_k],
        "tracked": tracked,
        "contrasts": contrasts,
    }


def delta_record(
    *,
    before_logits: torch.Tensor,
    after_logits: torch.Tensor,
    tokenizer: Any,
    top_k: int,
) -> dict[str, Any]:
    difference = after_logits.detach().float().cpu() - before_logits.detach().float().cpu()
    count = min(max(top_k * 8, 200), difference.numel())
    positive_values, positive_ids = torch.topk(difference, k=count)
    negative_values, negative_ids = torch.topk(-difference, k=count)

    def collect(values: torch.Tensor, ids: torch.Tensor, sign: float) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for magnitude, token_id in zip(values, ids, strict=True):
            token_id_int = int(token_id)
            token = decode_token(tokenizer, token_id_int)
            if not is_readable_token(token):
                continue
            result.append(
                {
                    "token_id": token_id_int,
                    "token": token,
                    "display_token": display_token(token),
                    "delta_logit": sign * float(magnitude),
                }
            )
            if len(result) >= top_k:
                break
        return result

    return {
        "top_increases": collect(positive_values, positive_ids, 1.0),
        "top_decreases": collect(negative_values, negative_ids, -1.0),
    }


def cell_label(layer: int, position: int, prompt_token: str) -> str:
    return f"L{layer} / position {position} ({display_token(prompt_token)})"


def plot_top_concepts(
    *,
    readouts: dict[str, dict[tuple[int, int], dict[str, Any]]],
    prompt_tokens: dict[int, str],
    full_metrics: dict[str, Any],
    output_path: Path,
    top_k: int,
) -> None:
    figure, axes = plt.subplots(len(CAPTURE_CELLS), 2, figsize=(15, 13))
    for row, (layer, position) in enumerate(CAPTURE_CELLS):
        for column, condition in enumerate(("clean", PRIMARY_CONDITION)):
            axis = axes[row, column]
            axis.axis("off")
            records = readouts[condition][(layer, position)]["readable_top_tokens"][:top_k]
            lines = [
                f"{item['rank']:>4}  {item['display_token'][:22]:<22}  "
                f"logP {item['log_probability']:>8.3f}"
                for item in records
            ]
            state_name = "Clean residual" if condition == "clean" else "After full SAE edit"
            axis.set_title(
                f"{state_name}\n{cell_label(layer, position, prompt_tokens[position])}",
                fontsize=12,
                loc="left",
            )
            axis.text(
                0.01,
                0.96,
                "\n".join(lines),
                transform=axis.transAxes,
                va="top",
                ha="left",
                family="monospace",
                fontsize=10,
            )
    figure.suptitle(
        "J-Lens concepts before and after the frozen L7+L22 transcoder intervention\n"
        f"Edited output: P(Six)={full_metrics['p_six']:.2%}, "
        f"P(Eight)={full_metrics['p_eight']:.2%}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_tracked_changes(
    *,
    readouts: dict[str, dict[tuple[int, int], dict[str, Any]]],
    prompt_tokens: dict[int, str],
    output_path: Path,
) -> None:
    contrasts = list(CONTRAST_PAIRS)
    labels = {
        "ant_minus_spider": "ant - spider",
        "insect_minus_spider": "insect - spider",
        "six_minus_eight": "Six - Eight",
    }
    figure, axes = plt.subplots(1, len(CAPTURE_CELLS), figsize=(18, 5.8), sharey=True)
    for axis, (layer, position) in zip(axes, CAPTURE_CELLS, strict=True):
        clean = readouts["clean"][(layer, position)]["contrasts"]
        edited = readouts[PRIMARY_CONDITION][(layer, position)]["contrasts"]
        x = np.arange(len(contrasts))
        width = 0.36
        clean_bars = axis.bar(
            x - width / 2,
            [clean[name]["log_probability_difference"] for name in contrasts],
            width,
            label="clean",
            color="#94a3b8",
        )
        edited_bars = axis.bar(
            x + width / 2,
            [edited[name]["log_probability_difference"] for name in contrasts],
            width,
            label="full SAE edit",
            color="#0f766e",
        )
        axis.axhline(0, color="#111827", linewidth=0.8)
        axis.set_xticks(x, [labels[name] for name in contrasts], rotation=25, ha="right")
        axis.set_title(cell_label(layer, position, prompt_tokens[position]))
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(clean_bars, fmt="%.1f", padding=2, fontsize=8, rotation=90)
        axis.bar_label(edited_bars, fmt="%.1f", padding=2, fontsize=8, rotation=90)
    axes[0].set_ylabel("J-Lens log-probability contrast\n(above zero favors the first concept)")
    axes[-1].legend(loc="lower right")
    figure.suptitle("Semantic contrasts before and after the successful SAE intervention", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_top_csv(
    path: Path,
    readouts: dict[str, dict[tuple[int, int], dict[str, Any]]],
    prompt_tokens: dict[int, str],
) -> None:
    fields = [
        "condition",
        "layer",
        "position",
        "prompt_token",
        "rank",
        "token_id",
        "token",
        "logit",
        "log_probability",
        "probability",
        "readable",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, cells in readouts.items():
            for (layer, position), record in cells.items():
                for item in record["top_tokens"]:
                    writer.writerow(
                        {
                            "condition": condition,
                            "layer": layer,
                            "position": position,
                            "prompt_token": prompt_tokens[position],
                            **{field: item[field] for field in fields if field in item},
                        }
                    )


def write_tracked_csv(
    path: Path,
    readouts: dict[str, dict[tuple[int, int], dict[str, Any]]],
    prompt_tokens: dict[int, str],
) -> None:
    fields = [
        "condition",
        "layer",
        "position",
        "prompt_token",
        "family",
        "variant_count",
        "best_token_id",
        "best_token",
        "best_surface",
        "best_rank",
        "best_logit",
        "family_log_probability",
        "family_probability",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, cells in readouts.items():
            for (layer, position), record in cells.items():
                for tracked in record["tracked"].values():
                    writer.writerow(
                        {
                            "condition": condition,
                            "layer": layer,
                            "position": position,
                            "prompt_token": prompt_tokens[position],
                            **{field: tracked[field] for field in fields if field in tracked},
                        }
                    )


def write_contrasts_csv(
    path: Path,
    readouts: dict[str, dict[tuple[int, int], dict[str, Any]]],
    prompt_tokens: dict[int, str],
) -> None:
    fields = [
        "condition",
        "layer",
        "position",
        "prompt_token",
        "contrast",
        "positive_family",
        "negative_family",
        "log_probability_difference",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, cells in readouts.items():
            for (layer, position), record in cells.items():
                for contrast in record["contrasts"].values():
                    writer.writerow(
                        {
                            "condition": condition,
                            "layer": layer,
                            "position": position,
                            "prompt_token": prompt_tokens[position],
                            "contrast": contrast["name"],
                            "positive_family": contrast["positive_family"],
                            "negative_family": contrast["negative_family"],
                            "log_probability_difference": contrast[
                                "log_probability_difference"
                            ],
                        }
                    )


def write_delta_csv(path: Path, comparisons: list[dict[str, Any]]) -> None:
    fields = [
        "comparison",
        "before_condition",
        "after_condition",
        "layer",
        "position",
        "direction",
        "rank",
        "token_id",
        "token",
        "delta_logit",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for comparison in comparisons:
            for direction, key in (("increase", "top_increases"), ("decrease", "top_decreases")):
                for rank, item in enumerate(comparison[key], start=1):
                    writer.writerow(
                        {
                            "comparison": comparison["name"],
                            "before_condition": comparison["before_condition"],
                            "after_condition": comparison["after_condition"],
                            "layer": comparison["layer"],
                            "position": comparison["position"],
                            "direction": direction,
                            "rank": rank,
                            "token_id": item["token_id"],
                            "token": item["token"],
                            "delta_logit": item["delta_logit"],
                        }
                    )


def serializable_readouts(
    readouts: dict[str, dict[tuple[int, int], dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    return {
        condition: {
            f"L{layer}_P{position}": record
            for (layer, position), record in cells.items()
        }
        for condition, cells in readouts.items()
    }


def main() -> None:
    args = parse_args()
    if not 0 <= args.spider_suppression <= 1:
        raise RuntimeError("--spider-suppression must be in [0, 1]")
    if args.ant_factor_l7 < 0 or args.ant_factor_l22 < 0:
        raise RuntimeError("Ant factors cannot be negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_manifest(args.feature_manifest)
    if source_manifest.get("model") != args.model_id:
        raise RuntimeError("Feature manifest and requested model do not match")
    if source_manifest.get("spider_prompt") != args.spider_prompt:
        raise RuntimeError("Feature manifest and requested spider prompt do not match")

    spider_instances = [
        instance
        for instance in source_manifest["spider_instances"]
        if int(instance["layer"]) in TESTED_LAYERS
    ]
    ant_instances = [
        instance
        for instance in source_manifest["ant_instances"]
        if int(instance["layer"]) in TESTED_LAYERS
    ]
    expected_spider_cells = {(7, 16), (22, 16), (22, 21)}
    actual_spider_cells = {
        (int(instance["layer"]), int(instance["position"]))
        for instance in spider_instances
    }
    if actual_spider_cells != expected_spider_cells:
        raise RuntimeError(
            f"Frozen spider cells changed: {sorted(actual_spider_cells)}"
        )
    expected_ant_cells = {(7, 16), (22, 16)}
    actual_ant_cells = {
        (int(instance["layer"]), int(instance["position"]))
        for instance in ant_instances
    }
    if actual_ant_cells != expected_ant_cells:
        raise RuntimeError(f"Frozen ant cells changed: {sorted(actual_ant_cells)}")

    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for instance in spider_instances + ant_instances:
        features_by_layer[int(instance["layer"])].add(int(instance["feature"]))
    rows = load_all_rows(features_by_layer, args.download_workers)
    spider_by_layer = group_instances(spider_instances)
    ant_by_layer = group_instances(ant_instances)

    print(f"Loading model {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, token=os.environ.get("HF_TOKEN") or None
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN") or None,
    )
    model.eval()
    layers = find_text_layers(model)
    device = model.get_input_embeddings().weight.device
    input_ids = tokenizer(
        args.spider_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    prompt_rows = token_rows(tokenizer, input_ids.cpu())
    prompt_tokens = {int(row["position"]): row["token"] for row in prompt_rows}
    runtime = {
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "jlens_version": importlib.metadata.version("jlens"),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
    }
    for layer, position in CAPTURE_CELLS:
        if position >= len(prompt_rows):
            raise RuntimeError(f"L{layer}/P{position} is outside the prompt")
    if prompt_tokens[16] != "spinning" or prompt_tokens[21] != "\n":
        raise RuntimeError(
            f"Prompt alignment drifted: P16={prompt_tokens[16]!r}, P21={prompt_tokens[21]!r}"
        )

    target_ids = {
        name: single_token_id(tokenizer, f" {name}")
        for name in ("Six", "Eight", "Four")
    }
    families = surface_ids(tokenizer)

    lens_path, jacobians_cpu, lens_n_prompts, lens_d_model = load_selected_jacobians(
        lens_repo=args.lens_repo,
        lens_filename=args.lens_filename,
        lens_path=args.lens_path,
        selected_layers=TESTED_LAYERS,
    )
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    if lens_d_model != lens_model.d_model:
        raise RuntimeError(
            f"J-Lens d_model={lens_d_model}, model d_model={lens_model.d_model}"
        )
    jacobians = {
        layer: matrix.to(device=device, dtype=torch.float32)
        for layer, matrix in jacobians_cpu.items()
    }

    conditions_spec = {
        "clean": ((), {}),
        "L7_only": ((7,), {7: args.ant_factor_l7}),
        "L22_only": ((22,), {22: args.ant_factor_l22}),
        PRIMARY_CONDITION: (
            TESTED_LAYERS,
            {7: args.ant_factor_l7, 22: args.ant_factor_l22},
        ),
    }
    runs: dict[str, dict[str, Any]] = {}
    for condition, (active_layers, ant_factors) in conditions_spec.items():
        print(f"Running {condition}")
        runs[condition] = run_condition(
            condition=condition,
            active_layers=active_layers,
            ant_factors=ant_factors,
            spider_suppression=args.spider_suppression,
            model=model,
            layers=layers,
            input_ids=input_ids,
            tokenizer=tokenizer,
            target_ids=target_ids,
            rows=rows,
            spider_by_layer=spider_by_layer,
            ant_by_layer=ant_by_layer,
            device=device,
            top_k=args.top_k,
        )

    clean_metrics = runs["clean"]["metrics"]
    full_metrics = runs[PRIMARY_CONDITION]["metrics"]
    if clean_metrics["top_token"].strip().lower() != "eight" or clean_metrics["p_eight"] < 0.99:
        raise RuntimeError(f"Clean baseline did not reproduce Eight: {clean_metrics}")
    if full_metrics["top_token"].strip().lower() != "six" or full_metrics["p_six"] < 0.9:
        raise RuntimeError(f"Frozen full intervention did not reproduce Six: {full_metrics}")

    cleanup_run = run_condition(
        condition="clean_after_cleanup",
        active_layers=(),
        ant_factors={},
        spider_suppression=args.spider_suppression,
        model=model,
        layers=layers,
        input_ids=input_ids,
        tokenizer=tokenizer,
        target_ids=target_ids,
        rows=rows,
        spider_by_layer=spider_by_layer,
        ant_by_layer=ant_by_layer,
        device=device,
        top_k=args.top_k,
    )
    for key in ("p_six", "p_eight", "p_four"):
        if not math.isclose(
            clean_metrics[key], cleanup_run["metrics"][key], rel_tol=1e-7, abs_tol=1e-10
        ):
            raise RuntimeError(f"Hook cleanup validation failed for {key}")
    for cell in CAPTURE_CELLS:
        if not torch.equal(runs["clean"]["residuals"][cell], cleanup_run["residuals"][cell]):
            raise RuntimeError(f"Clean residual changed after hook cleanup at {cell}")

    l7_repeat_error = float(
        (
            runs["L7_only"]["residuals"][(7, 16)]
            - runs[PRIMARY_CONDITION]["residuals"][(7, 16)]
        )
        .abs()
        .max()
        .item()
    )
    if l7_repeat_error != 0.0:
        raise RuntimeError(
            f"L7 residual should match between L7-only and full runs; max error={l7_repeat_error}"
        )

    readouts: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    lens_logits: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
    for condition, run in runs.items():
        readouts[condition] = {}
        lens_logits[condition] = {}
        for layer, position in CAPTURE_CELLS:
            residual = run["residuals"][(layer, position)].to(
                device=device, dtype=torch.float32
            )
            transported = residual @ jacobians[layer].T
            logits = lens_model.unembed(transported).detach().float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite J-Lens logits at {condition} L{layer}/P{position}")
            lens_logits[condition][(layer, position)] = logits
            readouts[condition][(layer, position)] = lens_record(
                logits=logits,
                tokenizer=tokenizer,
                families=families,
                top_k=args.top_k,
            )

    comparison_specs = [
        ("L7_P16_clean_to_full", "clean", PRIMARY_CONDITION, 7, 16),
        ("L22_P16_clean_to_full", "clean", PRIMARY_CONDITION, 22, 16),
        ("L22_P21_clean_to_full", "clean", PRIMARY_CONDITION, 22, 21),
        ("L22_P16_upstream_L7", "clean", "L7_only", 22, 16),
        ("L22_P21_upstream_L7", "clean", "L7_only", 22, 21),
        ("L22_P16_local_after_L7", "L7_only", PRIMARY_CONDITION, 22, 16),
        ("L22_P21_local_after_L7", "L7_only", PRIMARY_CONDITION, 22, 21),
        ("L22_P16_local_clean_context", "clean", "L22_only", 22, 16),
        ("L22_P21_local_clean_context", "clean", "L22_only", 22, 21),
    ]
    comparisons: list[dict[str, Any]] = []
    for name, before, after, layer, position in comparison_specs:
        delta = delta_record(
            before_logits=lens_logits[before][(layer, position)],
            after_logits=lens_logits[after][(layer, position)],
            tokenizer=tokenizer,
            top_k=args.top_k,
        )
        comparisons.append(
            {
                "name": name,
                "before_condition": before,
                "after_condition": after,
                "layer": layer,
                "position": position,
                "prompt_token": prompt_tokens[position],
                "residual_delta_norm": float(
                    (
                        runs[after]["residuals"][(layer, position)]
                        - runs[before]["residuals"][(layer, position)]
                    )
                    .norm()
                    .item()
                ),
                **delta,
            }
        )

    top_figure = args.output_dir / TOP_CONCEPTS_FIGURE
    tracked_figure = args.output_dir / TRACKED_CHANGES_FIGURE
    plot_top_concepts(
        readouts=readouts,
        prompt_tokens=prompt_tokens,
        full_metrics=full_metrics,
        output_path=top_figure,
        top_k=args.figure_top_k,
    )
    plot_tracked_changes(
        readouts=readouts,
        prompt_tokens=prompt_tokens,
        output_path=tracked_figure,
    )

    top_csv = args.output_dir / TOP_CONCEPTS_CSV
    tracked_csv = args.output_dir / TRACKED_CONCEPTS_CSV
    contrasts_csv = args.output_dir / CONTRASTS_CSV
    delta_csv = args.output_dir / DELTA_CONCEPTS_CSV
    write_top_csv(top_csv, readouts, prompt_tokens)
    write_tracked_csv(tracked_csv, readouts, prompt_tokens)
    write_contrasts_csv(contrasts_csv, readouts, prompt_tokens)
    write_delta_csv(delta_csv, comparisons)

    residual_path = args.output_dir / RESIDUAL_STATES
    torch.save(
        {
            "model": args.model_id,
            "prompt": args.spider_prompt,
            "cells": list(CAPTURE_CELLS),
            "conditions": {
                condition: {
                    f"L{layer}_P{position}": tensor
                    for (layer, position), tensor in run["residuals"].items()
                }
                for condition, run in runs.items()
            },
        },
        residual_path,
    )

    frozen_manifest = {
        "experiment": "J-Lens readout before and after frozen L7+L22 transcoder intervention",
        "created_at_unix": time.time(),
        "model": args.model_id,
        "source_feature_manifest": str(args.feature_manifest),
        "spider_prompt": args.spider_prompt,
        "prompt_tokens": prompt_rows,
        "capture_cells": [
            {"layer": layer, "position": position, "token": prompt_tokens[position]}
            for layer, position in CAPTURE_CELLS
        ],
        "spider_suppression": args.spider_suppression,
        "ant_factors": {"7": args.ant_factor_l7, "22": args.ant_factor_l22},
        "spider_instances": spider_instances,
        "ant_instances": ant_instances,
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "resolved_path": str(lens_path),
            "resolved_revision": snapshot_revision(lens_path),
            "n_prompts": lens_n_prompts,
            "d_model": lens_d_model,
            "source_layers_used": list(TESTED_LAYERS),
        },
        "runtime": runtime,
        "tracked_token_variants": families,
    }
    manifest_path = args.output_dir / FEATURE_MANIFEST
    manifest_path.write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2))

    result = {
        "experiment": frozen_manifest["experiment"],
        "model": args.model_id,
        "feature_manifest": str(args.feature_manifest),
        "lens": frozen_manifest["lens"],
        "runtime": runtime,
        "capture_cells": frozen_manifest["capture_cells"],
        "conditions": {
            condition: {
                "active_layers": run["active_layers"],
                "ant_factors": run["ant_factors"],
                "metrics": run["metrics"],
                "operations": run["operations"],
            }
            for condition, run in runs.items()
        },
        "cleanup_metrics": cleanup_run["metrics"],
        "validations": {
            "clean_top_token_is_eight": True,
            "full_top_token_is_six": True,
            "full_p_six_above_90_percent": True,
            "hook_cleanup_exact": True,
            "L7_only_equals_full_at_L7_P16": True,
            "selected_jacobians_finite": True,
        },
        "readouts": serializable_readouts(readouts),
        "comparisons": comparisons,
        "artifacts": {
            "json": OUTPUT_JSON,
            "manifest": FEATURE_MANIFEST,
            "top_concepts_csv": TOP_CONCEPTS_CSV,
            "tracked_concepts_csv": TRACKED_CONCEPTS_CSV,
            "contrasts_csv": CONTRASTS_CSV,
            "delta_concepts_csv": DELTA_CONCEPTS_CSV,
            "residual_states": RESIDUAL_STATES,
            "top_concepts_figure": TOP_CONCEPTS_FIGURE,
            "tracked_changes_figure": TRACKED_CHANGES_FIGURE,
        },
    }
    json_path = args.output_dir / OUTPUT_JSON
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    print(
        "Clean output: "
        f"P(Six)={clean_metrics['p_six']:.6%} "
        f"P(Eight)={clean_metrics['p_eight']:.6%}"
    )
    print(
        "Full edit output: "
        f"P(Six)={full_metrics['p_six']:.6%} "
        f"P(Eight)={full_metrics['p_eight']:.6%}"
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
