#!/usr/bin/env python3
"""Steer J-Lens concepts only at the three cells selected by the SAE edit.

The experiment freezes the successful SAE causal localization:

* L7/P16: suppress a spider J-Lens direction and inject a target direction;
* L22/P16: suppress a spider J-Lens direction and inject a target direction;
* L22/P21: suppress a spider J-Lens direction only.

No other layer or token position is edited.  The primary intervention is
additive steering along unit rows of ``W_U J_l``.  Strengths are dimensionless
fractions of the clean prompt's mean residual norm at the corresponding layer.
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
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

from run_layerwise_spider_ant_sweep import (
    MODEL_ID,
    SPIDER_PROMPT,
    FeatureKey,
    display_token,
    find_text_layers,
    load_all_rows,
    probability_record,
    read_manifest,
    single_token_id,
    token_rows,
)
from run_sae_jlens_before_after import (
    CAPTURE_CELLS,
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_LENS_FILENAME,
    DEFAULT_LENS_REPO,
    DEFAULT_MODEL_REVISION,
    TESTED_LAYERS,
    group_instances,
    lens_record,
    load_selected_jacobians,
    output_tensor,
    run_condition as run_sae_condition,
    snapshot_revision,
    surface_ids,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("results/sae_site_jlens_steering")
DEFAULT_LENS_REVISION = "0731326edff4ae730ffc5356fe1a4728c748b3a6"
DEFAULT_STRENGTHS = (0.0, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
DEFAULT_OVERDRIVE = (1.5, 2.0)
DEFAULT_REFINE_MULTIPLIERS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

OUTPUT_JSON = "sae_site_jlens_steering.json"
OUTPUT_MANIFEST = "sae_site_jlens_steering_manifest.json"
OUTPUT_CSV = "sae_site_jlens_steering.csv"
HEATMAP_FIGURE = "sae_site_jlens_primary_heatmaps.png"
SUMMARY_FIGURE = "sae_site_jlens_candidate_summary.png"
CURVES_FIGURE = "sae_site_jlens_best_curves.png"
DOWNSTREAM_FIGURE = "sae_site_jlens_best_downstream_top5.png"
DOWNSTREAM_CSV = "sae_site_jlens_best_downstream_top5.csv"

SOURCE_SPECS: dict[tuple[int, int], dict[str, Any]] = {
    (7, 16): {"surface": "Spider", "token_id": 86757},
    (22, 16): {"surface": " spider", "token_id": 32261},
    (22, 21): {"surface": " spiders", "token_id": 92664},
}

PRIMARY_TARGETS: tuple[dict[str, Any], ...] = (
    {"name": "ants", "surface": " ants", "token_id": 63272, "tier": "ant"},
    {"name": "ant", "surface": " ant", "token_id": 2314, "tier": "ant"},
    {"name": "Ants", "surface": " Ants", "token_id": 190788, "tier": "ant"},
)

SECONDARY_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "name": "insects",
        "surface": " insects",
        "token_id": 30348,
        "tier": "insect_surrogate",
    },
    {
        "name": "insect",
        "surface": " insect",
        "token_id": 16368,
        "tier": "insect_surrogate",
    },
    {
        "name": "termite",
        "surface": " termite",
        "token_id": 212513,
        "tier": "insect_surrogate",
    },
    {
        "name": "wasp",
        "surface": " wasp",
        "token_id": 186153,
        "tier": "insect_surrogate",
    },
    {
        "name": "bee",
        "surface": " bee",
        "token_id": 32737,
        "tier": "insect_surrogate",
    },
)

LIZARD_CONTROL = {
    "name": "lizards_control",
    "surface": " lizards",
    "token_id": 82404,
    "tier": "off_target_control",
}


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0.0 or not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("strength lists must contain finite nonnegative values")
    return tuple(sorted(set(values)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep fixed-site spider-down/target-up J-Lens steering"
    )
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--lens-filename", default=DEFAULT_LENS_FILENAME)
    parser.add_argument("--lens-revision", default=DEFAULT_LENS_REVISION)
    parser.add_argument("--lens-path", type=Path, default=None)
    parser.add_argument(
        "--strengths",
        type=parse_float_list,
        default=DEFAULT_STRENGTHS,
        help="Comma-separated shared ant/spider strengths.",
    )
    parser.add_argument(
        "--overdrive-strengths",
        type=parse_float_list,
        default=DEFAULT_OVERDRIVE,
    )
    parser.add_argument(
        "--refine-multipliers",
        type=parse_float_list,
        default=DEFAULT_REFINE_MULTIPLIERS,
    )
    parser.add_argument("--disable-overdrive", action="store_true")
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Require the pinned model/tokenizer to be present in the pod cache.",
    )
    parser.add_argument(
        "--replot-only",
        action="store_true",
        help="Regenerate figures from the saved raw JSON without loading the model.",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().float().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def validate_surface(tokenizer: Any, surface: str, expected_id: int) -> None:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if token_ids != [expected_id]:
        raise RuntimeError(
            f"Surface {surface!r} tokenized as {token_ids}, expected [{expected_id}]"
        )
    if tokenizer.decode([expected_id]) != surface:
        raise RuntimeError(
            f"Token {expected_id} decodes as {tokenizer.decode([expected_id])!r}, "
            f"expected {surface!r}"
        )


def direction_for_token(
    *,
    layer: int,
    token_id: int,
    jacobians: dict[int, torch.Tensor],
    model: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    unembedding = model.lm_head.weight[int(token_id)].detach().to(
        device=device, dtype=torch.float32
    )
    direction = jacobians[layer].T @ unembedding
    norm = direction.norm()
    if not torch.isfinite(direction).all() or not torch.isfinite(norm) or norm <= 0:
        raise RuntimeError(f"Invalid J-Lens direction for L{layer}/token {token_id}")
    result = direction / norm
    if not math.isclose(float(result.norm().item()), 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"J-Lens direction is not unit-normalized at L{layer}")
    return result


def make_direction_tables(
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    jacobians: dict[int, torch.Tensor],
    device: torch.device,
) -> tuple[
    dict[tuple[int, int], torch.Tensor],
    dict[str, dict[int, torch.Tensor]],
    dict[str, Any],
]:
    all_targets = list(PRIMARY_TARGETS) + list(SECONDARY_TARGETS) + [LIZARD_CONTROL]
    for spec in SOURCE_SPECS.values():
        validate_surface(tokenizer, spec["surface"], int(spec["token_id"]))
    for spec in all_targets:
        validate_surface(tokenizer, spec["surface"], int(spec["token_id"]))

    source_directions = {
        cell: direction_for_token(
            layer=cell[0],
            token_id=int(spec["token_id"]),
            jacobians=jacobians,
            model=model,
            device=device,
        )
        for cell, spec in SOURCE_SPECS.items()
    }
    target_directions: dict[str, dict[int, torch.Tensor]] = {}
    for spec in all_targets:
        target_directions[spec["name"]] = {
            layer: direction_for_token(
                layer=layer,
                token_id=int(spec["token_id"]),
                jacobians=jacobians,
                model=model,
                device=device,
            )
            for layer in TESTED_LAYERS
        }

    composite_members = [item["name"] for item in PRIMARY_TARGETS + SECONDARY_TARGETS]
    target_directions["family_composite"] = {}
    for layer in TESTED_LAYERS:
        summed = torch.stack(
            [target_directions[name][layer] for name in composite_members]
        ).sum(dim=0)
        target_directions["family_composite"][layer] = summed / summed.norm()

    metadata = {
        "sources": {
            f"L{layer}_P{position}": {
                **SOURCE_SPECS[(layer, position)],
                "direction_norm": float(source_directions[(layer, position)].norm().item()),
            }
            for layer, position in CAPTURE_CELLS
        },
        "targets": [*PRIMARY_TARGETS, *SECONDARY_TARGETS, LIZARD_CONTROL],
        "family_composite": {
            "members": composite_members,
            "formula": "unit(sum(unit(direction_for_each_target)))",
            "direction_norms": {
                str(layer): float(
                    target_directions["family_composite"][layer].norm().item()
                )
                for layer in TESTED_LAYERS
            },
        },
    }
    return source_directions, target_directions, metadata


def metrics_from_logits(
    *,
    logits: torch.Tensor,
    clean_log_probs: torch.Tensor | None,
    tokenizer: Any,
    target_ids: dict[str, int],
    top_k: int,
) -> dict[str, Any]:
    metrics = probability_record(logits, tokenizer, target_ids, top_k)
    probabilities = torch.softmax(logits.detach().float().cpu(), dim=-1)
    log_probabilities = torch.log_softmax(logits.detach().float().cpu(), dim=-1)
    if clean_log_probs is None:
        kl_from_clean = 0.0
    else:
        kl_from_clean = float(
            (probabilities * (log_probabilities - clean_log_probs)).sum().item()
        )
    numeric_mass = metrics["p_six"] + metrics["p_eight"] + metrics["p_four"]
    coherent = (
        metrics["p_six"] + metrics["p_eight"] >= 0.95
        and metrics["p_four"] < 0.01
        and metrics["top_token"].strip().lower() in {"six", "eight"}
    )
    return {
        **metrics,
        "output_kl_from_clean": kl_from_clean,
        "numeric_mass_six_eight_four": numeric_mass,
        "semantic_coherence": coherent,
        "causal_flip": (
            metrics["top_token"].strip().lower() == "six"
            and metrics["p_six"] > metrics["p_eight"]
        ),
        "strong_reproduction": metrics["p_six"] >= 0.95,
    }


def pair_coordinates(
    residual: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor | None,
) -> dict[str, float]:
    source_dot = float(torch.dot(residual, source).item())
    if target is None:
        return {"source_dot": source_dot}
    matrix = torch.stack([source, target], dim=1)
    pair = torch.linalg.pinv(matrix) @ residual
    gram_condition = float(torch.linalg.cond(matrix.T @ matrix).item())
    return {
        "source_dot": source_dot,
        "target_dot": float(torch.dot(residual, target).item()),
        "source_pair_coordinate": float(pair[0].item()),
        "target_pair_coordinate": float(pair[1].item()),
        "pair_gram_condition": gram_condition,
    }


def cosine_or_zero(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.norm() == 0 or right.norm() == 0:
        return 0.0
    return float(F.cosine_similarity(left[None], right[None]).item())


def projection_record(
    *,
    sae_delta: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor | None,
) -> dict[str, float]:
    matrix = source[:, None] if target is None else torch.stack([source, target], dim=1)
    projection = matrix @ (torch.linalg.pinv(matrix) @ sae_delta)
    return {
        "sae_delta_norm": float(sae_delta.norm().item()),
        "sae_projection_norm": float(projection.norm().item()),
        "sae_projection_norm_fraction": float(
            projection.norm().item() / max(sae_delta.norm().item(), 1e-12)
        ),
        "sae_projection_cosine": cosine_or_zero(sae_delta, projection),
    }


def capture_clean_residuals(
    *,
    model: torch.nn.Module,
    layers: Any,
    input_ids: torch.Tensor,
    capture_layers: Iterable[int],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Capture clean block outputs without installing any intervention hook."""
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []
    for layer_index in sorted(set(capture_layers)):
        def capture(_module, _args, output, layer_index=layer_index):
            captured[layer_index] = output_tensor(output).detach().float().cpu().clone()

        handles.append(layers[layer_index].register_forward_hook(capture))
    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
    missing = sorted(set(capture_layers) - set(captured))
    if missing:
        raise RuntimeError(f"Clean capture missed layers {missing}")
    return logits.detach().cpu(), captured


def fixed_strengths(
    *,
    ant_l7: float,
    spider_l7: float,
    ant_l22: float,
    spider_l22: float,
) -> dict[str, float]:
    return {
        "ant_L7": float(ant_l7),
        "spider_L7": float(spider_l7),
        "ant_L22": float(ant_l22),
        "spider_L22": float(spider_l22),
    }


def delta_for_cell(
    *,
    cell: tuple[int, int],
    strengths: dict[str, float],
    layer_scales: dict[int, float],
    source_direction: torch.Tensor,
    target_direction: torch.Tensor | None,
    method: str,
) -> torch.Tensor:
    layer, position = cell
    spider_strength = strengths[f"spider_L{layer}"]
    ant_strength = strengths[f"ant_L{layer}"] if position == 16 else 0.0
    scale = float(layer_scales[layer])
    if target_direction is None:
        return -scale * spider_strength * source_direction
    if method == "additive":
        return scale * (
            ant_strength * target_direction - spider_strength * source_direction
        )
    if method == "coordinate_clamp":
        matrix = torch.stack([source_direction, target_direction], dim=1)
        desired_dot_changes = torch.tensor(
            [-scale * spider_strength, scale * ant_strength],
            device=matrix.device,
            dtype=torch.float32,
        )
        coefficients = torch.linalg.pinv(matrix.T @ matrix) @ desired_dot_changes
        return matrix @ coefficients
    raise RuntimeError(f"Unknown steering method {method!r}")


def run_jlens_condition(
    *,
    condition: str,
    phase: str,
    target_spec: dict[str, Any],
    strengths: dict[str, float],
    method: str,
    model: torch.nn.Module,
    layers: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    target_ids: dict[str, int],
    source_directions: dict[tuple[int, int], torch.Tensor],
    target_directions: dict[str, dict[int, torch.Tensor]],
    layer_scales: dict[int, float],
    sae_deltas: dict[tuple[int, int], torch.Tensor],
    clean_log_probs: torch.Tensor,
    top_k: int,
    capture_layers: Iterable[int] = (),
    custom_deltas: dict[tuple[int, int], torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Run one fixed-site intervention and retain tensors under private keys."""
    handles: list[Any] = []
    block_residuals: dict[tuple[int, int], torch.Tensor] = {}
    hypothetical_block_residuals: dict[tuple[int, int], torch.Tensor] = {}
    pre_ff_residuals: dict[int, torch.Tensor] = {}
    downstream_residuals: dict[int, torch.Tensor] = {}
    intended_deltas: dict[tuple[int, int], torch.Tensor] = {}
    post_ff_deltas: dict[tuple[int, int], torch.Tensor] = {}
    target_name = str(target_spec["name"])

    for cell in CAPTURE_CELLS:
        layer, position = cell
        target_direction = (
            target_directions[target_name][layer] if position == 16 else None
        )
        intended = (
            custom_deltas[cell].to(
                device=source_directions[cell].device, dtype=torch.float32
            )
            if custom_deltas is not None
            else delta_for_cell(
                cell=cell,
                strengths=strengths,
                layer_scales=layer_scales,
                source_direction=source_directions[cell],
                target_direction=target_direction,
                method=method,
            )
        )
        if not torch.isfinite(intended).all():
            raise RuntimeError(f"{condition}: non-finite intended delta at {cell}")
        intended_deltas[cell] = intended

    cells_by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, position in CAPTURE_CELLS:
        cells_by_layer[layer].append(position)

    for layer_index, positions in cells_by_layer.items():
        def capture_pre_ff_residual(
            _module,
            args,
            layer_index=layer_index,
        ):
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError(
                    f"{condition}: could not capture L{layer_index} pre-FF residual"
                )
            pre_ff_residuals[layer_index] = args[0].detach().clone()

        def edit_post_ff(
            _module,
            _args,
            output,
            layer_index=layer_index,
            positions=positions,
        ):
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"{condition}: expected tensor at L{layer_index} post-FF hook"
                )
            changed = output.clone()
            for position in positions:
                cell = (layer_index, position)
                intended = intended_deltas[cell]
                if layer_index not in pre_ff_residuals:
                    raise RuntimeError(
                        f"{condition}: L{layer_index} post-FF edit preceded residual capture"
                    )
                hypothetical_block_residuals[cell] = (
                    pre_ff_residuals[layer_index][0, position] + output[0, position]
                ).detach().float().cpu()
                changed[0, position] = (
                    output[0, position] + intended.to(dtype=output.dtype)
                )
                post_ff_deltas[cell] = (
                    changed[0, position].detach().float()
                    - output[0, position].detach().float()
                ).cpu()
            return changed

        handles.append(
            layers[layer_index].pre_feedforward_layernorm.register_forward_pre_hook(
                capture_pre_ff_residual
            )
        )
        handles.append(
            layers[layer_index].post_feedforward_layernorm.register_forward_hook(
                edit_post_ff
            )
        )

    all_capture_layers = sorted(set(TESTED_LAYERS) | set(capture_layers))
    for layer_index in all_capture_layers:
        positions = cells_by_layer.get(layer_index, [])

        def capture_block(
            _module,
            _args,
            output,
            layer_index=layer_index,
            positions=positions,
        ):
            hidden = output_tensor(output)
            for position in positions:
                block_residuals[(layer_index, position)] = (
                    hidden[0, position].detach().float().cpu().clone()
                )
            if layer_index in capture_layers:
                downstream_residuals[layer_index] = (
                    hidden[0, 16].detach().float().cpu().clone()
                )

        handles.append(layers[layer_index].register_forward_hook(capture_block))

    try:
        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()

    missing_cells = sorted(set(CAPTURE_CELLS) - set(block_residuals))
    if missing_cells:
        raise RuntimeError(f"{condition}: missing block residuals {missing_cells}")
    if sorted(downstream_residuals) != sorted(set(capture_layers)):
        raise RuntimeError(f"{condition}: downstream capture was incomplete")
    missing_hypotheticals = sorted(
        set(CAPTURE_CELLS) - set(hypothetical_block_residuals)
    )
    if missing_hypotheticals:
        raise RuntimeError(
            f"{condition}: missing counterfactual block outputs {missing_hypotheticals}"
        )
    actual_deltas = {
        cell: block_residuals[cell] - hypothetical_block_residuals[cell]
        for cell in CAPTURE_CELLS
    }

    metrics = metrics_from_logits(
        logits=logits,
        clean_log_probs=clean_log_probs,
        tokenizer=tokenizer,
        target_ids=target_ids,
        top_k=top_k,
    )
    cell_records: list[dict[str, Any]] = []
    relative_norms: list[float] = []
    coordinate_checks: list[bool] = []
    for cell in CAPTURE_CELLS:
        layer, position = cell
        source = source_directions[cell].detach().float().cpu()
        target = (
            target_directions[target_name][layer].detach().float().cpu()
            if position == 16
            else None
        )
        intended = intended_deltas[cell].detach().float().cpu()
        actual = actual_deltas[cell]
        after = block_residuals[cell]
        before = hypothetical_block_residuals[cell]
        before_coordinates = pair_coordinates(before, source, target)
        after_coordinates = pair_coordinates(after, source, target)
        coordinate_changes = {
            key: after_coordinates[key] - before_coordinates[key]
            for key in before_coordinates
            if key.endswith("dot") or key.endswith("coordinate")
        }
        spider_strength = strengths[f"spider_L{layer}"]
        ant_strength = strengths[f"ant_L{layer}"] if position == 16 else 0.0
        source_ok = (
            spider_strength == 0.0
            or coordinate_changes["source_dot"] < -1e-6
        )
        target_ok = (
            target is None
            or ant_strength == 0.0
            or coordinate_changes["target_dot"] > 1e-6
        )
        coordinate_checks.extend([source_ok, target_ok])
        normalized_norm = float(actual.norm().item() / layer_scales[layer])
        relative_norms.append(normalized_norm)
        block_transport_error = float(
            ((after - before) - actual).abs().max().item()
        )
        if block_transport_error > 1e-6:
            raise RuntimeError(
                f"{condition}: block-output delta mismatch at {cell}: "
                f"{block_transport_error}"
            )
        projection = projection_record(
            sae_delta=sae_deltas[cell], source=source, target=target
        )
        cell_records.append(
            {
                "layer": layer,
                "position": position,
                "prompt_token": "spinning" if position == 16 else "\\n",
                "source_surface": SOURCE_SPECS[cell]["surface"],
                "source_token_id": SOURCE_SPECS[cell]["token_id"],
                "target_surface": target_spec.get("surface") if target is not None else None,
                "target_token_id": target_spec.get("token_id") if target is not None else None,
                "layer_scale": float(layer_scales[layer]),
                "spider_strength": spider_strength,
                "target_strength": ant_strength,
                "intended_delta_norm": float(intended.norm().item()),
                "post_feedforward_delta_norm": float(post_ff_deltas[cell].norm().item()),
                "actual_delta_norm": float(actual.norm().item()),
                "actual_delta_normalized_by_clean_mean_norm": normalized_norm,
                "intended_actual_cosine": cosine_or_zero(intended, actual),
                "intended_actual_norm_error": float(
                    abs(intended.norm().item() - actual.norm().item())
                ),
                "block_output_transport_max_abs_error": block_transport_error,
                "source_coordinate_decreased": source_ok,
                "target_coordinate_increased": target_ok,
                "coordinates_before": before_coordinates,
                "coordinates_after": after_coordinates,
                "coordinate_changes": coordinate_changes,
                "cosine_jlens_edit_to_sae_edit": cosine_or_zero(actual, sae_deltas[cell]),
                **projection,
            }
        )

    total_actual_norm = math.sqrt(
        sum(float(actual_deltas[cell].norm().item()) ** 2 for cell in CAPTURE_CELLS)
    )
    total_normalized_norm = math.sqrt(sum(value * value for value in relative_norms))
    return {
        "condition": condition,
        "phase": phase,
        "method": method,
        "target_name": target_name,
        "target_surface": target_spec.get("surface"),
        "target_token_id": target_spec.get("token_id"),
        "target_tier": target_spec.get("tier"),
        "strengths": dict(strengths),
        "metrics": metrics,
        "cells": cell_records,
        "all_requested_coordinate_changes_have_expected_sign": all(coordinate_checks),
        "total_residual_delta_norm": total_actual_norm,
        "total_normalized_residual_delta_norm": total_normalized_norm,
        "_logits": logits.detach().float().cpu(),
        "_residuals": block_residuals,
        "_actual_deltas": actual_deltas,
        "_downstream_residuals": downstream_residuals,
    }


def public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if not key.startswith("_")}


def selection_key(run: dict[str, Any]) -> tuple[float, float, float, float]:
    metrics = run["metrics"]
    return (
        float(metrics["semantic_coherence"]),
        float(metrics["log_p_six_minus_log_p_eight"]),
        float(metrics["p_six"] + metrics["p_eight"]),
        -float(metrics["output_kl_from_clean"]),
    )


def best_run(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(runs)
    if not candidates:
        raise RuntimeError("Cannot select a best run from an empty collection")
    return max(candidates, key=selection_key)


def shared_grid(
    *,
    target_spec: dict[str, Any],
    ant_values: Iterable[float],
    spider_values: Iterable[float],
    phase: str,
    method: str,
    execute: Any,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    ant_values = tuple(ant_values)
    spider_values = tuple(spider_values)
    total = len(ant_values) * len(spider_values)
    print(
        f"Running {phase} for {target_spec['name']} with {total} "
        f"{method} cells...",
        flush=True,
    )
    for ant_strength in ant_values:
        for spider_strength in spider_values:
            strengths = fixed_strengths(
                ant_l7=ant_strength,
                spider_l7=spider_strength,
                ant_l22=ant_strength,
                spider_l22=spider_strength,
            )
            runs.append(
                execute(
                    target_spec=target_spec,
                    strengths=strengths,
                    method=method,
                    phase=phase,
                    condition=(
                        f"{phase}_{target_spec['name']}_"
                        f"a{ant_strength:g}_s{spider_strength:g}"
                    ),
                )
            )
    return runs


def improving_boundary_axes(
    runs: list[dict[str, Any]], strengths: tuple[float, ...]
) -> tuple[bool, bool]:
    if len(strengths) < 2:
        return False, False
    high, previous = strengths[-1], strengths[-2]
    by_pair = {
        (
            run["strengths"]["ant_L7"],
            run["strengths"]["spider_L7"],
        ): run
        for run in runs
    }

    ant_improves = False
    spider_improves = False
    for spider in strengths:
        boundary = by_pair.get((high, spider))
        inner = by_pair.get((previous, spider))
        if (
            boundary is not None
            and inner is not None
            and boundary["metrics"]["semantic_coherence"]
            and selection_key(boundary) > selection_key(inner)
        ):
            ant_improves = True
    for ant in strengths:
        boundary = by_pair.get((ant, high))
        inner = by_pair.get((ant, previous))
        if (
            boundary is not None
            and inner is not None
            and boundary["metrics"]["semantic_coherence"]
            and selection_key(boundary) > selection_key(inner)
        ):
            spider_improves = True
    return ant_improves, spider_improves


def coordinate_descent(
    *,
    initial: dict[str, Any],
    target_spec: dict[str, Any],
    multipliers: tuple[float, ...],
    method: str,
    execute: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    current = initial
    generated: list[dict[str, Any]] = []
    coordinates = ("ant_L7", "spider_L7", "ant_L22", "spider_L22")
    for pass_index in range(2):
        for coordinate in coordinates:
            center = float(current["strengths"][coordinate])
            anchor = center if center > 0 else 1 / 32
            values = sorted(set([center] + [anchor * value for value in multipliers]))
            candidates = [current]
            for value in values:
                strengths = dict(current["strengths"])
                strengths[coordinate] = float(value)
                candidate = execute(
                    target_spec=target_spec,
                    strengths=strengths,
                    method=method,
                    phase="coordinate_descent",
                    condition=(
                        f"refine_p{pass_index + 1}_{coordinate}_{value:g}_"
                        f"{target_spec['name']}_{method}"
                    ),
                )
                generated.append(candidate)
                candidates.append(candidate)
            current = best_run(candidates)
            print(
                f"Refinement pass {pass_index + 1}, {coordinate}: "
                f"{current['strengths'][coordinate]:g}; "
                f"P(Six)={current['metrics']['p_six']:.4%}",
                flush=True,
            )
    return current, generated


def make_heatmaps(
    *,
    runs: list[dict[str, Any]],
    strengths: tuple[float, ...],
    output_path: Path,
) -> None:
    lookup = {
        (run["strengths"]["ant_L7"], run["strengths"]["spider_L7"]): run
        for run in runs
    }
    six = np.full((len(strengths), len(strengths)), np.nan)
    eight = np.full_like(six, np.nan)
    for row, ant_strength in enumerate(strengths):
        for column, spider_strength in enumerate(strengths):
            run = lookup[(ant_strength, spider_strength)]
            six[row, column] = 100 * run["metrics"]["p_six"]
            eight[row, column] = 100 * run["metrics"]["p_eight"]

    figure, axes = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    for axis, values, title, cmap in (
        (axes[0], six, "P(Six), %", "viridis"),
        (axes[1], eight, "P(Eight), %", "magma"),
    ):
        image = axis.imshow(values, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=100)
        axis.set_xticks(range(len(strengths)), [f"{value:g}" for value in strengths])
        axis.set_yticks(range(len(strengths)), [f"{value:g}" for value in strengths])
        axis.set_xlabel("shared spider suppression strength")
        axis.set_ylabel("shared target injection strength")
        axis.set_title(title)
        for row in range(len(strengths)):
            for column in range(len(strengths)):
                value = values[row, column]
                color = "white" if value < 25 or value > 75 else "black"
                label = f"{value:.4f}" if value < 0.1 else f"{value:.1f}"
                axis.text(column, row, label, ha="center", va="center", fontsize=7, color=color)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(
        "Fixed SAE sites, additive J-Lens steering toward ' ants'\n"
        "L7/P16 + L22/P16 target-up; L7/P16 + L22/P16/P21 spider-down",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_candidate_summary(
    *,
    candidate_runs: dict[str, list[dict[str, Any]]],
    sae_p_six: float,
    output_path: Path,
) -> None:
    names = list(candidate_runs)
    bests = [best_run(candidate_runs[name]) for name in names]
    values = [100 * run["metrics"]["p_six"] for run in bests]
    colors = [
        "#0f766e" if run["target_tier"] == "ant" else "#d97706"
        for run in bests
    ]
    figure, axis = plt.subplots(figsize=(max(9, 0.9 * len(names)), 6.2))
    bars = axis.bar(range(len(names)), values, color=colors)
    axis.axhline(100 * sae_p_six, color="#7c3aed", linestyle="--", linewidth=2, label="SAE positive control")
    axis.axhline(50, color="#64748b", linestyle=":", linewidth=1)
    axis.set_xticks(range(len(names)), names, rotation=30, ha="right")
    axis.set_ylabel("best P(Six), %")
    axis.set_ylim(0, 101)
    axis.set_title("Best coherent fixed-site J-Lens result by target direction")
    axis.legend(loc="upper left")
    for bar, value in zip(bars, values, strict=True):
        label = f"{value:.4f}" if value < 0.1 else f"{value:.2f}"
        axis.text(bar.get_x() + bar.get_width() / 2, min(value + 1.5, 98), label, ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def make_best_curves(
    *,
    grid_runs: list[dict[str, Any]],
    best: dict[str, Any],
    output_path: Path,
) -> None:
    shared_best = best_run(grid_runs)
    fixed_spider = shared_best["strengths"]["spider_L7"]
    fixed_ant = shared_best["strengths"]["ant_L7"]
    ant_slice = sorted(
        [
            run for run in grid_runs
            if run["strengths"]["spider_L7"] == fixed_spider
        ],
        key=lambda run: run["strengths"]["ant_L7"],
    )
    spider_slice = sorted(
        [
            run for run in grid_runs
            if run["strengths"]["ant_L7"] == fixed_ant
        ],
        key=lambda run: run["strengths"]["spider_L7"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for axis, slice_runs, x_key, x_label, fixed_label in (
        (axes[0], ant_slice, "ant_L7", "shared target strength", f"spider={fixed_spider:g}"),
        (axes[1], spider_slice, "spider_L7", "shared spider strength", f"target={fixed_ant:g}"),
    ):
        x = [run["strengths"][x_key] for run in slice_runs]
        p_six = [100 * run["metrics"]["p_six"] for run in slice_runs]
        p_eight = [100 * run["metrics"]["p_eight"] for run in slice_runs]
        axis.plot(x, p_six, "o-", color="#0f766e", label="P(Six)")
        axis.plot(x, p_eight, "o-", color="#b91c1c", label="P(Eight)")
        axis.axhline(50, color="#94a3b8", linewidth=0.8)
        axis.axhline(
            100 * best["metrics"]["p_six"],
            color="#7c3aed",
            linestyle="--",
            linewidth=1.4,
            label=f"refined best P(Six)={100 * best['metrics']['p_six']:.2f}%",
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel("probability, %")
        axis.set_ylim(0, 101)
        axis.set_title(f"{fixed_label}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        for left, right in zip(slice_runs, slice_runs[1:]):
            left_gap = left["metrics"]["p_six"] - left["metrics"]["p_eight"]
            right_gap = right["metrics"]["p_six"] - right["metrics"]["p_eight"]
            if left_gap <= 0 < right_gap:
                midpoint = (left["strengths"][x_key] + right["strengths"][x_key]) / 2
                axis.axvline(midpoint, color="#7c3aed", linestyle="--", linewidth=1.2)
                axis.text(midpoint, 4, "Six/Eight crossover", rotation=90, va="bottom", ha="right", fontsize=8)
                break
    figure.suptitle(
        f"Best target: {shared_best['target_surface'] or shared_best['target_name']} — "
        "shared-grid slices plus refined outcome\n"
        f"refined strengths: L7 target={best['strengths']['ant_L7']:g}, "
        f"L7 spider={best['strengths']['spider_L7']:g}, "
        f"L22 target={best['strengths']['ant_L22']:g}, "
        f"L22 spider={best['strengths']['spider_L22']:g}",
        fontsize=14,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def downstream_trajectory(
    *,
    capture_layers: list[int],
    last_layer: int,
    clean_residuals: dict[int, torch.Tensor],
    edited_residuals: dict[int, torch.Tensor],
    jacobians: dict[int, torch.Tensor],
    lens_model: Any,
    tokenizer: Any,
    families: dict[str, list[dict[str, Any]]],
    device: torch.device,
) -> list[dict[str, Any]]:
    trajectory: list[dict[str, Any]] = []
    for layer in capture_layers:
        clean = clean_residuals[layer][0, 16].float()
        edited = edited_residuals[layer].float()
        layer_records: dict[str, Any] = {}
        for condition, residual in (("clean", clean), ("edited", edited)):
            residual_device = residual.to(device=device, dtype=torch.float32)
            transported = (
                residual_device
                if layer == last_layer
                else jacobians[layer] @ residual_device
            )
            logits = lens_model.unembed(transported).detach().float().cpu()
            layer_records[condition] = lens_record(
                logits=logits,
                tokenizer=tokenizer,
                families=families,
                top_k=5,
            )
        delta = edited - clean
        trajectory.append(
            {
                "layer": layer,
                "position": 16,
                "prompt_token": "spinning",
                "transport": "identity_final" if layer == last_layer else "published_jlens",
                "residual_delta_norm": float(delta.norm().item()),
                "relative_residual_delta_norm": float(
                    delta.norm().item() / max(clean.norm().item(), 1e-12)
                ),
                "clean_edited_cosine": cosine_or_zero(clean, edited),
                "clean_top5": layer_records["clean"]["readable_top_tokens"],
                "edited_top5": layer_records["edited"]["readable_top_tokens"],
                "clean_contrasts": layer_records["clean"]["contrasts"],
                "edited_contrasts": layer_records["edited"]["contrasts"],
            }
        )
    return trajectory


def make_downstream_figure(
    *,
    trajectory: list[dict[str, Any]],
    best: dict[str, Any],
    output_path: Path,
) -> None:
    rows: list[list[str]] = []
    for record in trajectory:
        clean_lines = "\n".join(
            f"#{item['rank']:<3} {display_token(item['token'])[:24]}"
            for item in record["clean_top5"]
        )
        edited_lines = "\n".join(
            f"#{item['rank']:<3} {display_token(item['token'])[:24]}"
            for item in record["edited_top5"]
        )
        rows.append(
            [
                f"L{record['layer']}",
                f"{record['residual_delta_norm']:.1f}",
                f"{record['clean_edited_cosine']:.4f}",
                clean_lines,
                edited_lines,
            ]
        )
    figure_height = max(13.0, 1.0 + 1.25 * len(rows))
    figure, axis = plt.subplots(figsize=(19, figure_height))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=(
            "Layer",
            "||delta h||",
            "cos(clean, edited)",
            "Clean: top 5 readable J-Lens tokens",
            "After fixed-site J-Lens edit: top 5 readable tokens",
        ),
        colWidths=(0.07, 0.10, 0.12, 0.35, 0.35),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    row_height = 0.92 / (len(rows) + 1)
    for (row, _column), cell in table.get_celld().items():
        cell.set_height(row_height)
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
    figure.suptitle(
        "Downstream P16 J-Lens trajectory after the best fixed-site J-Lens edit\n"
        f"target={best['target_surface'] or best['target_name']!r}; "
        f"P(Six)={best['metrics']['p_six']:.2%}, "
        f"P(Eight)={best['metrics']['p_eight']:.2%}",
        fontsize=15,
        y=0.985,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_downstream_csv(path: Path, trajectory: list[dict[str, Any]]) -> None:
    fields = (
        "layer",
        "position",
        "condition",
        "list_index",
        "vocabulary_rank",
        "token_id",
        "token",
        "log_probability",
        "probability",
        "residual_delta_norm",
        "relative_residual_delta_norm",
        "clean_edited_cosine",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in trajectory:
            for condition in ("clean", "edited"):
                for list_index, item in enumerate(record[f"{condition}_top5"], start=1):
                    writer.writerow(
                        {
                            "layer": record["layer"],
                            "position": record["position"],
                            "condition": condition,
                            "list_index": list_index,
                            "vocabulary_rank": item["rank"],
                            "token_id": item["token_id"],
                            "token": item["token"],
                            "log_probability": item["log_probability"],
                            "probability": item["probability"],
                            "residual_delta_norm": record["residual_delta_norm"],
                            "relative_residual_delta_norm": record["relative_residual_delta_norm"],
                            "clean_edited_cosine": record["clean_edited_cosine"],
                        }
                    )


def write_tidy_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = (
        "condition",
        "phase",
        "method",
        "target_name",
        "target_surface",
        "target_token_id",
        "target_tier",
        "ant_L7",
        "spider_L7",
        "ant_L22",
        "spider_L22",
        "p_six",
        "p_eight",
        "p_four",
        "log_p_six_minus_log_p_eight",
        "output_kl_from_clean",
        "top_token",
        "top_token_id",
        "causal_flip",
        "strong_reproduction",
        "semantic_coherence",
        "all_coordinate_signs_valid",
        "total_residual_delta_norm",
        "total_normalized_residual_delta_norm",
        "L7_P16_delta_norm",
        "L22_P16_delta_norm",
        "L22_P21_delta_norm",
        "L7_P16_sae_cosine",
        "L22_P16_sae_cosine",
        "L22_P21_sae_cosine",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            metrics = run["metrics"]
            strengths = run["strengths"]
            cells = {
                (cell["layer"], cell["position"]): cell for cell in run["cells"]
            }
            writer.writerow(
                {
                    "condition": run["condition"],
                    "phase": run["phase"],
                    "method": run["method"],
                    "target_name": run["target_name"],
                    "target_surface": run["target_surface"],
                    "target_token_id": run["target_token_id"],
                    "target_tier": run["target_tier"],
                    **{key: strengths[key] for key in ("ant_L7", "spider_L7", "ant_L22", "spider_L22")},
                    **{key: metrics[key] for key in (
                        "p_six", "p_eight", "p_four",
                        "log_p_six_minus_log_p_eight", "output_kl_from_clean",
                        "top_token", "top_token_id", "causal_flip",
                        "strong_reproduction", "semantic_coherence",
                    )},
                    "all_coordinate_signs_valid": run["all_requested_coordinate_changes_have_expected_sign"],
                    "total_residual_delta_norm": run["total_residual_delta_norm"],
                    "total_normalized_residual_delta_norm": run["total_normalized_residual_delta_norm"],
                    "L7_P16_delta_norm": cells[(7, 16)]["actual_delta_norm"],
                    "L22_P16_delta_norm": cells[(22, 16)]["actual_delta_norm"],
                    "L22_P21_delta_norm": cells[(22, 21)]["actual_delta_norm"],
                    "L7_P16_sae_cosine": cells[(7, 16)]["cosine_jlens_edit_to_sae_edit"],
                    "L22_P16_sae_cosine": cells[(22, 16)]["cosine_jlens_edit_to_sae_edit"],
                    "L22_P21_sae_cosine": cells[(22, 21)]["cosine_jlens_edit_to_sae_edit"],
                }
            )


def replot_saved_result(output_dir: Path, strengths: tuple[float, ...]) -> None:
    raw_path = output_dir / OUTPUT_JSON
    if not raw_path.is_file():
        raise RuntimeError(f"Cannot replot without {raw_path}")
    result = json.loads(raw_path.read_text())
    runs = result["runs"]
    best = result["best_run"]
    candidate_names = list(result["candidate_best"])
    candidate_runs = {
        name: [
            run for run in runs
            if run["target_name"] == name
            and ("grid" in run["phase"] or run["phase"] == "coordinate_descent")
        ]
        for name in candidate_names
    }
    primary_ants_grid = [
        run for run in runs
        if run["target_name"] == "ants" and run["phase"] == "primary_grid"
    ]
    best_target_grid = [
        run for run in runs
        if run["target_name"] == best["target_name"] and "grid" in run["phase"]
    ]
    make_heatmaps(
        runs=primary_ants_grid,
        strengths=strengths,
        output_path=output_dir / HEATMAP_FIGURE,
    )
    make_candidate_summary(
        candidate_runs=candidate_runs,
        sae_p_six=result["sae_positive_control"]["L7_L22"]["p_six"],
        output_path=output_dir / SUMMARY_FIGURE,
    )
    make_best_curves(
        grid_runs=best_target_grid,
        best=best,
        output_path=output_dir / CURVES_FIGURE,
    )
    make_downstream_figure(
        trajectory=result["downstream_trajectory"],
        best=best,
        output_path=output_dir / DOWNSTREAM_FIGURE,
    )
    print(f"Regenerated figures from {raw_path}", flush=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.replot_only:
        replot_saved_result(args.output_dir, args.strengths)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a CUDA GPU")
    source_manifest = read_manifest(args.feature_manifest)
    if source_manifest.get("model") != args.model_id:
        raise RuntimeError("Feature manifest and requested model do not match")
    if source_manifest.get("spider_prompt") != args.spider_prompt:
        raise RuntimeError("Feature manifest and spider prompt do not match")

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
    actual_spider_cells = {
        (int(instance["layer"]), int(instance["position"]))
        for instance in spider_instances
    }
    actual_ant_cells = {
        (int(instance["layer"]), int(instance["position"]))
        for instance in ant_instances
    }
    if actual_spider_cells != set(CAPTURE_CELLS):
        raise RuntimeError(f"Frozen spider cells drifted: {sorted(actual_spider_cells)}")
    if actual_ant_cells != {(7, 16), (22, 16)}:
        raise RuntimeError(f"Frozen ant cells drifted: {sorted(actual_ant_cells)}")

    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for instance in spider_instances + ant_instances:
        features_by_layer[int(instance["layer"])].add(int(instance["feature"]))
    print("Loading the frozen transcoder rows for the SAE positive control...", flush=True)
    rows = load_all_rows(features_by_layer, args.download_workers)
    spider_by_layer = group_instances(spider_instances)
    ant_by_layer = group_instances(ant_instances)

    token = os.environ.get("HF_TOKEN") or None
    print(f"Loading pinned eager-attention model {args.model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=token,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=token,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).eval()
    layers = find_text_layers(model)
    device = model.get_input_embeddings().weight.device
    last_layer = len(layers) - 1
    if last_layer < 22:
        raise RuntimeError(f"Unexpected model depth: {len(layers)} blocks")
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if resolved_model_revision not in (None, args.model_revision):
        raise RuntimeError(
            f"Resolved model revision {resolved_model_revision} != {args.model_revision}"
        )

    input_ids = tokenizer(
        args.spider_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)
    prompt_rows = token_rows(tokenizer, input_ids.detach().cpu())
    if len(prompt_rows) != 27:
        raise RuntimeError(f"Expected 27 prompt tokens, found {len(prompt_rows)}")
    if prompt_rows[16]["token"] != "spinning" or prompt_rows[21]["token"] != "\n":
        raise RuntimeError(
            f"Prompt alignment drifted: P16={prompt_rows[16]['token']!r}, "
            f"P21={prompt_rows[21]['token']!r}"
        )
    target_ids = {
        name: single_token_id(tokenizer, f" {name}")
        for name in ("Six", "Eight", "Four")
    }
    if any(
        int(spec["token_id"]) in {target_ids["Six"], target_ids["Eight"]}
        for spec in PRIMARY_TARGETS + SECONDARY_TARGETS
    ):
        raise RuntimeError("Six/Eight was accidentally included as a steering target")

    downstream_layers = list(range(22, last_layer + 1))
    published_layers = sorted(set([7, *[layer for layer in downstream_layers if layer != last_layer]]))
    if args.lens_path is None:
        resolved_lens = Path(
            hf_hub_download(
                repo_id=args.lens_repo,
                filename=args.lens_filename,
                revision=args.lens_revision,
                token=token,
            )
        )
    else:
        resolved_lens = args.lens_path.expanduser().resolve()
    lens_path, jacobians_cpu, lens_n_prompts, lens_d_model = load_selected_jacobians(
        lens_repo=args.lens_repo,
        lens_filename=args.lens_filename,
        lens_path=resolved_lens,
        selected_layers=published_layers,
    )
    resolved_lens_revision = snapshot_revision(lens_path)
    if resolved_lens_revision is None and args.lens_path is None:
        # hf_hub_download was called with this immutable commit hash.  Newer
        # huggingface_hub versions can return a blob path rather than a path
        # containing the snapshot revision, so retain the pinned resolution.
        resolved_lens_revision = args.lens_revision
    if resolved_lens_revision not in (None, args.lens_revision):
        raise RuntimeError(
            f"Resolved lens revision {resolved_lens_revision} != {args.lens_revision}"
        )
    jacobians = {
        layer: matrix.to(device=device, dtype=torch.float32)
        for layer, matrix in jacobians_cpu.items()
    }
    lens_model = jlens.from_hf(model, tokenizer, force_bos=False)
    if lens_d_model != lens_model.d_model:
        raise RuntimeError(
            f"Published J-Lens d_model={lens_d_model}, model d_model={lens_model.d_model}"
        )

    source_directions, target_directions, direction_metadata = make_direction_tables(
        tokenizer=tokenizer,
        model=model,
        jacobians=jacobians,
        device=device,
    )
    families = surface_ids(tokenizer)

    print("Capturing clean residual norms and downstream states...", flush=True)
    clean_logits, clean_layer_outputs = capture_clean_residuals(
        model=model,
        layers=layers,
        input_ids=input_ids,
        capture_layers=sorted(set([7, *downstream_layers])),
    )
    clean_log_probs = torch.log_softmax(clean_logits.float(), dim=-1)
    clean_metrics = metrics_from_logits(
        logits=clean_logits,
        clean_log_probs=None,
        tokenizer=tokenizer,
        target_ids=target_ids,
        top_k=args.top_k,
    )
    layer_scales = {
        layer: float(
            torch.linalg.vector_norm(clean_layer_outputs[layer][0], dim=-1).mean().item()
        )
        for layer in TESTED_LAYERS
    }
    if clean_metrics["top_token"].strip().lower() != "eight":
        raise RuntimeError(f"Clean top token was not Eight: {clean_metrics}")
    if not math.isclose(
        clean_metrics["p_eight"], 0.99998248, rel_tol=0.0, abs_tol=5e-6
    ):
        raise RuntimeError(
            f"Clean P(Eight) did not reproduce 99.998248%: {clean_metrics['p_eight']:.8%}"
        )

    sae_common = {
        "model": model,
        "layers": layers,
        "input_ids": input_ids,
        "tokenizer": tokenizer,
        "target_ids": target_ids,
        "rows": rows,
        "spider_by_layer": spider_by_layer,
        "ant_by_layer": ant_by_layer,
        "device": device,
        "top_k": args.top_k,
        "spider_suppression": 1.0,
    }
    print("Reproducing the eager-attention SAE positive control...", flush=True)
    sae_clean = run_sae_condition(
        condition="sae_clean", active_layers=(), ant_factors={}, **sae_common
    )
    sae_l7 = run_sae_condition(
        condition="sae_L7_only", active_layers=(7,), ant_factors={7: 4.0}, **sae_common
    )
    sae_full = run_sae_condition(
        condition="sae_L7_L22",
        active_layers=TESTED_LAYERS,
        ant_factors={7: 4.0, 22: 4.0},
        **sae_common,
    )
    if not math.isclose(
        sae_clean["metrics"]["p_eight"], clean_metrics["p_eight"],
        rel_tol=1e-6, abs_tol=1e-8,
    ):
        raise RuntimeError(
            "SAE clean control and hook-free clean run disagree beyond the "
            "CPU/GPU softmax tolerance: "
            f"SAE={sae_clean['metrics']['p_eight']:.12g}, "
            f"hook_free={clean_metrics['p_eight']:.12g}"
        )
    if sae_full["metrics"]["top_token"].strip().lower() != "six" or not math.isclose(
        sae_full["metrics"]["p_six"], 0.95222110, rel_tol=0.0, abs_tol=5e-5
    ):
        raise RuntimeError(
            "Frozen SAE positive control did not reproduce P(Six)=95.222110%: "
            f"{sae_full['metrics']}"
        )
    sae_deltas = {
        (7, 16): sae_l7["residuals"][(7, 16)] - sae_clean["residuals"][(7, 16)],
        (22, 16): sae_full["residuals"][(22, 16)] - sae_l7["residuals"][(22, 16)],
        (22, 21): sae_full["residuals"][(22, 21)] - sae_l7["residuals"][(22, 21)],
    }

    all_runs: list[dict[str, Any]] = []

    def execute(
        *,
        target_spec: dict[str, Any],
        strengths: dict[str, float],
        method: str,
        phase: str,
        condition: str,
        custom_deltas: dict[tuple[int, int], torch.Tensor] | None = None,
        capture_layers: Iterable[int] = (),
    ) -> dict[str, Any]:
        run = run_jlens_condition(
            condition=condition,
            phase=phase,
            target_spec=target_spec,
            strengths=strengths,
            method=method,
            model=model,
            layers=layers,
            input_ids=input_ids,
            tokenizer=tokenizer,
            target_ids=target_ids,
            source_directions=source_directions,
            target_directions=target_directions,
            layer_scales=layer_scales,
            sae_deltas=sae_deltas,
            clean_log_probs=clean_log_probs,
            top_k=args.top_k,
            capture_layers=capture_layers,
            custom_deltas=custom_deltas,
        )
        all_runs.append(run)
        return run

    zero_strengths = fixed_strengths(
        ant_l7=0, spider_l7=0, ant_l22=0, spider_l22=0
    )
    hook_transport_validations: list[dict[str, Any]] = []
    for index, cell in enumerate(CAPTURE_CELLS):
        custom = {
            other: torch.zeros_like(source_directions[other])
            for other in CAPTURE_CELLS
        }
        layer, position = cell
        custom[cell] = 0.125 * layer_scales[layer] * (
            target_directions["ants"][layer] - source_directions[cell]
            if position == 16
            else -source_directions[cell]
        )
        validation_run = execute(
            target_spec=PRIMARY_TARGETS[0],
            strengths=zero_strengths,
            method="additive",
            phase="hook_transport_validation",
            condition=f"hook_transport_L{layer}_P{position}",
            custom_deltas=custom,
        )
        observed = (
            validation_run["_residuals"][cell]
            - clean_layer_outputs[layer][0, position]
        )
        realized = validation_run["_actual_deltas"][cell]
        intended = custom[cell].detach().float().cpu()
        internal_error = float((observed - realized).abs().max().item())
        relative_quantization_error = float(
            (realized - intended).norm().item() / max(intended.norm().item(), 1e-12)
        )
        direction_cosine = cosine_or_zero(realized, intended)
        if internal_error > 1e-5:
            raise RuntimeError(
                f"Counterfactual block measurement failed at {cell}: {internal_error}"
            )
        if relative_quantization_error > 0.10 or direction_cosine < 0.995:
            raise RuntimeError(
                f"Post-feedforward hook distorted the intended J-Lens delta at {cell}: "
                f"relative error={relative_quantization_error:.6f}, "
                f"cosine={direction_cosine:.6f}"
            )
        hook_transport_validations.append(
            {
                "layer": layer,
                "position": position,
                "counterfactual_max_abs_error": internal_error,
                "intended_realized_relative_l2_error": relative_quantization_error,
                "intended_realized_cosine": direction_cosine,
                "precision": "bfloat16 model residual stream",
            }
        )

    candidate_runs: dict[str, list[dict[str, Any]]] = {}
    target_specs_by_name = {
        spec["name"]: dict(spec)
        for spec in PRIMARY_TARGETS + SECONDARY_TARGETS
    }
    target_specs_by_name["family_composite"] = {
        "name": "family_composite",
        "surface": None,
        "token_id": None,
        "tier": "insect_family_composite",
    }
    target_specs_by_name[LIZARD_CONTROL["name"]] = dict(LIZARD_CONTROL)

    def run_target_grid(spec: dict[str, Any], phase: str) -> list[dict[str, Any]]:
        runs = shared_grid(
            target_spec=spec,
            ant_values=args.strengths,
            spider_values=args.strengths,
            phase=phase,
            method="additive",
            execute=execute,
        )
        ant_improves, spider_improves = improving_boundary_axes(runs, args.strengths)
        if not args.disable_overdrive and (ant_improves or spider_improves):
            if ant_improves:
                runs.extend(
                    shared_grid(
                        target_spec=spec,
                        ant_values=args.overdrive_strengths,
                        spider_values=args.strengths,
                        phase=f"{phase}_overdrive_ant",
                        method="additive",
                        execute=execute,
                    )
                )
            if spider_improves:
                runs.extend(
                    shared_grid(
                        target_spec=spec,
                        ant_values=args.strengths,
                        spider_values=args.overdrive_strengths,
                        phase=f"{phase}_overdrive_spider",
                        method="additive",
                        execute=execute,
                    )
                )
            if ant_improves and spider_improves:
                runs.extend(
                    shared_grid(
                        target_spec=spec,
                        ant_values=args.overdrive_strengths,
                        spider_values=args.overdrive_strengths,
                        phase=f"{phase}_overdrive_both",
                        method="additive",
                        execute=execute,
                    )
                )
        candidate_runs[spec["name"]] = runs
        candidate_best = best_run(runs)
        print(
            f"Best {spec['name']}: P(Six)={candidate_best['metrics']['p_six']:.4%}, "
            f"P(Eight)={candidate_best['metrics']['p_eight']:.4%}, "
            f"strengths={candidate_best['strengths']}",
            flush=True,
        )
        return runs

    for spec in PRIMARY_TARGETS:
        run_target_grid(dict(spec), "primary_grid")

    primary_matches_sae = any(
        run["metrics"]["strong_reproduction"]
        and run["metrics"]["semantic_coherence"]
        for name in [spec["name"] for spec in PRIMARY_TARGETS]
        for run in candidate_runs[name]
    )
    tested_individual_names = [spec["name"] for spec in PRIMARY_TARGETS]
    if not primary_matches_sae:
        for spec in SECONDARY_TARGETS:
            run_target_grid(dict(spec), "secondary_grid")
            tested_individual_names.append(spec["name"])

    individual_succeeded = any(
        run["metrics"]["causal_flip"]
        for name in tested_individual_names
        for run in candidate_runs[name]
    )
    if not individual_succeeded:
        composite = target_specs_by_name["family_composite"]
        run_target_grid(composite, "family_composite_grid")

    all_candidate_grid_runs = [
        run for runs in candidate_runs.values() for run in runs
    ]
    best_additive = best_run(all_candidate_grid_runs)
    best_target = target_specs_by_name[best_additive["target_name"]]
    simultaneous = any(
        best_additive["strengths"][f"ant_L{layer}"] > 0
        and best_additive["strengths"][f"spider_L{layer}"] > 0
        for layer in TESTED_LAYERS
    )
    clamp_runs: list[dict[str, Any]] = []
    if simultaneous and not best_additive[
        "all_requested_coordinate_changes_have_expected_sign"
    ]:
        print(
            "Additive directions failed a requested coordinate sign; "
            "running the pre-registered pseudoinverse clamp fallback.",
            flush=True,
        )
        clamp_runs = shared_grid(
            target_spec=best_target,
            ant_values=args.strengths,
            spider_values=args.strengths,
            phase="coordinate_clamp_grid",
            method="coordinate_clamp",
            execute=execute,
        )
    initial_best = best_run([*all_candidate_grid_runs, *clamp_runs])
    chosen_method = initial_best["method"]
    best_target = target_specs_by_name[initial_best["target_name"]]
    refined_best, refinement_runs = coordinate_descent(
        initial=initial_best,
        target_spec=best_target,
        multipliers=args.refine_multipliers,
        method=chosen_method,
        execute=execute,
    )
    best = best_run([initial_best, refined_best, *refinement_runs])

    controls: dict[str, dict[str, Any]] = {}
    controls["zero_strength_hooks"] = execute(
        target_spec=best_target,
        strengths=zero_strengths,
        method="additive",
        phase="control",
        condition="zero_strength_hooks",
    )
    zero_strength_logits_exact = torch.equal(
        controls["zero_strength_hooks"]["_logits"], clean_logits
    )
    zero_strength_logit_max_abs_error = float(
        (
            controls["zero_strength_hooks"]["_logits"] - clean_logits
        ).abs().max().item()
    )
    if not zero_strength_logits_exact:
        raise RuntimeError(
            "Zero-strength hook logits did not exactly match clean: "
            f"max abs error={zero_strength_logit_max_abs_error}"
        )
    controls["target_only"] = execute(
        target_spec=best_target,
        strengths=fixed_strengths(
            ant_l7=best["strengths"]["ant_L7"], spider_l7=0,
            ant_l22=best["strengths"]["ant_L22"], spider_l22=0,
        ),
        method=best["method"],
        phase="control",
        condition="target_only",
    )
    controls["spider_only"] = execute(
        target_spec=best_target,
        strengths=fixed_strengths(
            ant_l7=0, spider_l7=best["strengths"]["spider_L7"],
            ant_l22=0, spider_l22=best["strengths"]["spider_L22"],
        ),
        method=best["method"],
        phase="control",
        condition="spider_only",
    )
    controls["L7_only"] = execute(
        target_spec=best_target,
        strengths=fixed_strengths(
            ant_l7=best["strengths"]["ant_L7"],
            spider_l7=best["strengths"]["spider_L7"],
            ant_l22=0, spider_l22=0,
        ),
        method=best["method"],
        phase="control",
        condition="L7_only",
    )
    controls["L22_only"] = execute(
        target_spec=best_target,
        strengths=fixed_strengths(
            ant_l7=0, spider_l7=0,
            ant_l22=best["strengths"]["ant_L22"],
            spider_l22=best["strengths"]["spider_L22"],
        ),
        method=best["method"],
        phase="control",
        condition="L22_only",
    )
    controls["lizards_target"] = execute(
        target_spec=target_specs_by_name[LIZARD_CONTROL["name"]],
        strengths=best["strengths"],
        method=best["method"],
        phase="off_target_control",
        condition="lizards_matched_strength",
    )

    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(20260830)
    random_deltas: dict[tuple[int, int], torch.Tensor] = {}
    for cell in CAPTURE_CELLS:
        reference = best["_actual_deltas"][cell]
        random_direction = torch.randn(reference.shape, generator=random_generator)
        random_direction = random_direction / random_direction.norm()
        random_deltas[cell] = random_direction * reference.norm()
    controls["matched_random"] = execute(
        target_spec=best_target,
        strengths=best["strengths"],
        method="additive",
        phase="matched_random_control",
        condition="matched_norm_random",
        custom_deltas=random_deltas,
    )

    best_repeat = execute(
        target_spec=best_target,
        strengths=best["strengths"],
        method=best["method"],
        phase="best_repeat_and_downstream",
        condition="best_exact_repeat",
        capture_layers=downstream_layers,
    )
    for key in ("p_six", "p_eight", "p_four", "output_kl_from_clean"):
        if not math.isclose(
            best["metrics"][key], best_repeat["metrics"][key],
            rel_tol=1e-7, abs_tol=1e-10,
        ):
            raise RuntimeError(f"Best-run repeat failed for {key}")

    cleanup_logits, cleanup_outputs = capture_clean_residuals(
        model=model,
        layers=layers,
        input_ids=input_ids,
        capture_layers=sorted(set([7, *downstream_layers])),
    )
    cleanup_metrics = metrics_from_logits(
        logits=cleanup_logits,
        clean_log_probs=clean_log_probs,
        tokenizer=tokenizer,
        target_ids=target_ids,
        top_k=args.top_k,
    )
    for key in ("p_six", "p_eight", "p_four"):
        if not math.isclose(
            clean_metrics[key], cleanup_metrics[key], rel_tol=1e-7, abs_tol=1e-10
        ):
            raise RuntimeError(f"Hook cleanup failed for {key}")
    for layer in sorted(set([7, *downstream_layers])):
        if not torch.equal(clean_layer_outputs[layer], cleanup_outputs[layer]):
            raise RuntimeError(f"Hook cleanup changed clean residuals at L{layer}")

    trajectory = downstream_trajectory(
        capture_layers=downstream_layers,
        last_layer=last_layer,
        clean_residuals=clean_layer_outputs,
        edited_residuals=best_repeat["_downstream_residuals"],
        jacobians=jacobians,
        lens_model=lens_model,
        tokenizer=tokenizer,
        families=families,
        device=device,
    )

    eligible_efficiency_runs = [
        run for run in all_runs
        if run["phase"] not in {
            "hook_transport_validation", "matched_random_control", "off_target_control"
        }
        and run["target_tier"] != "off_target_control"
    ]
    efficiency: dict[str, Any] = {}
    success_predicates = {
        "causal_flip": lambda run: run["metrics"]["causal_flip"],
        "strong_reproduction": lambda run: (
            run["metrics"]["causal_flip"] and run["metrics"]["strong_reproduction"]
        ),
        "semantic_coherence": lambda run: (
            run["metrics"]["causal_flip"] and run["metrics"]["semantic_coherence"]
        ),
    }
    for name, predicate in success_predicates.items():
        successful = [run for run in eligible_efficiency_runs if predicate(run)]
        efficiency[name] = (
            public_run(min(
                successful,
                key=lambda run: run["total_normalized_residual_delta_norm"],
            ))
            if successful else None
        )

    primary_ants_grid = [
        run for run in candidate_runs["ants"]
        if run["phase"] == "primary_grid"
    ]
    best_target_grid = candidate_runs[best["target_name"]]
    make_heatmaps(
        runs=primary_ants_grid,
        strengths=args.strengths,
        output_path=args.output_dir / HEATMAP_FIGURE,
    )
    figure_candidate_runs = {
        name: list(runs) for name, runs in candidate_runs.items()
    }
    for run in refinement_runs:
        figure_candidate_runs[run["target_name"]].append(run)
    make_candidate_summary(
        candidate_runs=figure_candidate_runs,
        sae_p_six=sae_full["metrics"]["p_six"],
        output_path=args.output_dir / SUMMARY_FIGURE,
    )
    make_best_curves(
        grid_runs=best_target_grid,
        best=best,
        output_path=args.output_dir / CURVES_FIGURE,
    )
    make_downstream_figure(
        trajectory=trajectory,
        best=best,
        output_path=args.output_dir / DOWNSTREAM_FIGURE,
    )
    write_downstream_csv(args.output_dir / DOWNSTREAM_CSV, trajectory)
    write_tidy_csv(args.output_dir / OUTPUT_CSV, all_runs)

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "jlens": package_version("jlens"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "attention_implementation": "eager",
    }
    manifest = {
        "experiment": "SAE-site-matched direct J-Lens steering",
        "created_at_unix": time.time(),
        "model": args.model_id,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": resolved_model_revision,
        "prompt": args.spider_prompt,
        "prompt_tokens": prompt_rows,
        "fixed_edit_cells": [
            {"layer": layer, "position": position, **SOURCE_SPECS[(layer, position)]}
            for layer, position in CAPTURE_CELLS
        ],
        "layer_scale_definition": (
            "mean clean block-output residual norm across this prompt's token positions; "
            "prompt-derived approximation because corpus means are absent from the checkpoint"
        ),
        "layer_scales": layer_scales,
        "direction_definition": "unit(J_layer.T @ unembedding[token])",
        "directions": direction_metadata,
        "strengths": list(args.strengths),
        "overdrive_strengths": list(args.overdrive_strengths),
        "refine_multipliers": list(args.refine_multipliers),
        "primary_targets": list(PRIMARY_TARGETS),
        "secondary_targets": list(SECONDARY_TARGETS),
        "off_target_control": LIZARD_CONTROL,
        "source_feature_manifest": str(args.feature_manifest),
        "sae_spider_instances": spider_instances,
        "sae_ant_instances": ant_instances,
        "lens": {
            "repo": args.lens_repo,
            "filename": args.lens_filename,
            "revision_requested": args.lens_revision,
            "revision_resolved": resolved_lens_revision,
            "resolved_path": str(lens_path),
            "n_prompts": lens_n_prompts,
            "d_model": lens_d_model,
            "published_layers_used": published_layers,
        },
        "runtime": runtime,
    }
    (args.output_dir / OUTPUT_MANIFEST).write_text(
        json.dumps(json_ready(manifest), ensure_ascii=False, indent=2) + "\n"
    )

    result = {
        "experiment": manifest["experiment"],
        "manifest": OUTPUT_MANIFEST,
        "clean": clean_metrics,
        "sae_positive_control": {
            "clean": sae_clean["metrics"],
            "L7_only": sae_l7["metrics"],
            "L7_L22": sae_full["metrics"],
            "local_residual_delta_norms": {
                f"L{layer}_P{position}": float(sae_deltas[(layer, position)].norm().item())
                for layer, position in CAPTURE_CELLS
            },
        },
        "tier_decisions": {
            "primary_matched_sae_95_percent": primary_matches_sae,
            "secondary_tier_was_run": not primary_matches_sae,
            "an_individual_target_causally_flipped": individual_succeeded,
            "family_composite_was_run": "family_composite" in candidate_runs,
        },
        "candidate_best": {
            name: public_run(best_run(runs))
            for name, runs in candidate_runs.items()
        },
        "best_run": public_run(best),
        "best_repeat": public_run(best_repeat),
        "controls": {
            name: public_run(run) for name, run in controls.items()
        },
        "efficiency": efficiency,
        "hook_transport_validations": hook_transport_validations,
        "downstream_trajectory": trajectory,
        "runs": [public_run(run) for run in all_runs],
        "validations": {
            "clean_p_eight_reproduced": True,
            "sae_p_six_reproduced": True,
            "candidate_surfaces_are_single_tokens": True,
            "directions_finite_and_unit_normalized": True,
            "only_fixed_cells_edited": True,
            "post_ff_delta_reaches_block_output": True,
            "zero_strength_hook_logits_exactly_match_clean": zero_strength_logits_exact,
            "zero_strength_hook_logit_max_abs_error": zero_strength_logit_max_abs_error,
            "best_repeat_matches": True,
            "hook_cleanup_matches_clean": True,
        },
        "artifacts": {
            "json": OUTPUT_JSON,
            "manifest": OUTPUT_MANIFEST,
            "csv": OUTPUT_CSV,
            "primary_heatmaps": HEATMAP_FIGURE,
            "candidate_summary": SUMMARY_FIGURE,
            "best_curves": CURVES_FIGURE,
            "downstream_top5": DOWNSTREAM_FIGURE,
            "downstream_csv": DOWNSTREAM_CSV,
        },
    }
    (args.output_dir / OUTPUT_JSON).write_text(
        json.dumps(json_ready(result), ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Completed fixed-site J-Lens experiment: "
        f"target={best['target_surface'] or best['target_name']!r}, "
        f"method={best['method']}, P(Six)={best['metrics']['p_six']:.6%}, "
        f"P(Eight)={best['metrics']['p_eight']:.6%}, "
        f"normalized delta={best['total_normalized_residual_delta_norm']:.6f}",
        flush=True,
    )
    print(f"Artifacts: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
