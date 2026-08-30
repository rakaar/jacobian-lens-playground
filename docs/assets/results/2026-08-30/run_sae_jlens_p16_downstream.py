#!/usr/bin/env python3
"""Trace the SAE edit through downstream J-Lens readouts at one position.

The frozen intervention is the established eager-attention L7+L22 edit:

* suppress the selected spider features at L7/P16 and L22/P16+P21;
* inject the calibrated ant features at L7/P16 and L22/P16 with factor 4.

For clean and edited runs, capture the residual at the requested position
after every block from L22 through the final model block. Published
layer-specific J-Lens matrices transport L22 through the penultimate source
layer. The final block uses identity transport because its output is already
the terminal residual. No full residual tensors are written to disk.
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
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

from run_layerwise_spider_ant_sweep import (
    MODEL_ID,
    SPIDER_PROMPT,
    FeatureKey,
    feature_activation,
    find_text_layers,
    load_all_rows,
    probability_record,
    read_manifest,
    single_token_id,
    token_rows,
)
from run_sae_jlens_before_after import (
    DEFAULT_FEATURE_MANIFEST,
    DEFAULT_LENS_FILENAME,
    DEFAULT_LENS_REPO,
    DEFAULT_MODEL_REVISION,
    lens_record,
    load_selected_jacobians,
    output_tensor,
    snapshot_revision,
    surface_ids,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUTPUT_DIR = Path("results/sae_jlens_p16_downstream")
DEFAULT_START_LAYER = 22
DEFAULT_POSITION = 16
INTERVENTION_LAYERS = (7, 22)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace J-Lens concepts at one prompt position from L22 through "
            "the final block"
        )
    )
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--spider-prompt", default=SPIDER_PROMPT)
    parser.add_argument("--lens-repo", default=DEFAULT_LENS_REPO)
    parser.add_argument("--lens-filename", default=DEFAULT_LENS_FILENAME)
    parser.add_argument("--lens-path", type=Path, default=None)
    parser.add_argument("--start-layer", type=int, default=DEFAULT_START_LAYER)
    parser.add_argument(
        "--end-layer",
        type=int,
        default=-1,
        help="Inclusive end layer; -1 means the final model block.",
    )
    parser.add_argument("--position", type=int, default=DEFAULT_POSITION)
    parser.add_argument("--spider-suppression", type=float, default=1.0)
    parser.add_argument("--ant-factor-l7", type=float, default=4.0)
    parser.add_argument("--ant-factor-l22", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--download-workers", type=int, default=6)
    return parser.parse_args()


def group_instances(
    instances: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[int(instance["layer"])].append(instance)
    return {
        layer: sorted(
            values,
            key=lambda item: (int(item["position"]), int(item["feature"])),
        )
        for layer, values in grouped.items()
    }


def run_condition(
    *,
    condition: str,
    active_layers: tuple[int, ...],
    capture_layers: list[int],
    capture_position: int,
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
    feature_inputs: dict[int, torch.Tensor] = {}
    residuals: dict[int, torch.Tensor] = {}
    operations: list[dict[str, Any]] = []
    handles: list[Any] = []

    for layer_index in capture_layers:
        def capture_block(_module, _args, output, layer_index=layer_index):
            hidden = output_tensor(output)
            residuals[layer_index] = (
                hidden[0, capture_position].detach().float().cpu().clone()
            )

        handles.append(layers[layer_index].register_forward_hook(capture_block))

    for layer_index in active_layers:
        if layer_index not in ant_factors:
            raise RuntimeError(f"{condition}: missing ant factor for L{layer_index}")

        def capture_feature_input(_module, _args, output, layer_index=layer_index):
            feature_inputs[layer_index] = output_tensor(output)

        def alter_mlp_write(_module, _args, output, layer_index=layer_index):
            if layer_index not in feature_inputs:
                raise RuntimeError(
                    f"{condition}: L{layer_index} edit ran before feature capture"
                )
            if not torch.is_tensor(output):
                raise RuntimeError(
                    f"{condition}: expected tensor at L{layer_index} MLP output"
                )
            changed = output.clone()
            delta_by_position: dict[int, torch.Tensor] = {}

            for instance in spider_by_layer[layer_index]:
                key = FeatureKey(layer_index, int(instance["feature"]))
                row = rows[key]
                position = int(instance["position"])
                preactivation, activation = feature_activation(
                    feature_inputs[layer_index], position, row, device
                )
                target = max(
                    0.0,
                    float(instance["clean_activation"])
                    * (1.0 - spider_suppression),
                )
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target - activation) * decoder
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

            ant_factor = float(ant_factors[layer_index])
            for instance in ant_by_layer[layer_index]:
                key = FeatureKey(layer_index, int(instance["feature"]))
                row = rows[key]
                position = int(instance["position"])
                preactivation, activation = feature_activation(
                    feature_inputs[layer_index], position, row, device
                )
                reference_activation = float(instance["reference_activation"])
                target = activation + ant_factor * reference_activation
                decoder = torch.from_numpy(row["W_dec"]).to(
                    device=device, dtype=torch.float32
                )
                delta = (target - activation) * decoder
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
                        "reference_activation": reference_activation,
                        "target_activation": target,
                        "factor": ant_factor,
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

    missing = sorted(set(capture_layers) - set(residuals))
    if missing:
        raise RuntimeError(f"{condition}: missing captured layers {missing}")

    return {
        "condition": condition,
        "active_layers": list(active_layers),
        "metrics": probability_record(logits, tokenizer, target_ids, top_k),
        "operations": operations,
        "residuals": residuals,
    }


def write_top_concepts_csv(
    path: Path,
    readouts: dict[str, dict[int, dict[str, Any]]],
) -> None:
    fields = [
        "condition",
        "layer",
        "list_index",
        "vocabulary_rank",
        "token_id",
        "token",
        "log_probability",
        "probability",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for condition, layer_records in readouts.items():
            for layer, record in layer_records.items():
                for list_index, item in enumerate(
                    record["readable_top_tokens"], start=1
                ):
                    writer.writerow(
                        {
                            "condition": condition,
                            "layer": layer,
                            "list_index": list_index,
                            "vocabulary_rank": item["rank"],
                            "token_id": item["token_id"],
                            "token": item["token"],
                            "log_probability": item["log_probability"],
                            "probability": item["probability"],
                        }
                    )


def write_trajectory_csv(path: Path, trajectory: list[dict[str, Any]]) -> None:
    fields = [
        "layer",
        "transport",
        "residual_delta_norm",
        "relative_residual_delta_norm",
        "clean_edited_cosine",
        "top5_token_overlap",
        "clean_ant_minus_spider",
        "edited_ant_minus_spider",
        "clean_insect_minus_spider",
        "edited_insect_minus_spider",
        "clean_six_minus_eight",
        "edited_six_minus_eight",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in trajectory:
            writer.writerow({field: record[field] for field in fields})


def concept_lines(record: dict[str, Any]) -> str:
    return "\n".join(
        f"#{item['rank']:<3} {item['display_token'][:24]}"
        for item in record["readable_top_tokens"]
    )


def plot_trajectory(
    *,
    output_path: Path,
    capture_position: int,
    prompt_token: str,
    capture_layers: list[int],
    readouts: dict[str, dict[int, dict[str, Any]]],
    trajectory: list[dict[str, Any]],
    edited_metrics: dict[str, Any],
) -> None:
    rows = []
    for layer, metrics in zip(capture_layers, trajectory, strict=True):
        rows.append(
            [
                f"L{layer}",
                f"{metrics['residual_delta_norm']:.1f}",
                f"{metrics['clean_edited_cosine']:.4f}",
                concept_lines(readouts["clean"][layer]),
                concept_lines(readouts["edited"][layer]),
            ]
        )

    figure_height = max(13.0, 1.0 + 1.25 * len(capture_layers))
    figure, axis = plt.subplots(figsize=(19, figure_height))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=[
            "Layer",
            "||delta h||",
            "cos(clean, edited)",
            "Clean: top 5 readable J-Lens tokens",
            "After SAE edit: top 5 readable J-Lens tokens",
        ],
        colWidths=[0.07, 0.10, 0.12, 0.35, 0.35],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    row_height = 0.92 / (len(capture_layers) + 1)
    for (row, column), cell in table.get_celld().items():
        cell.set_height(row_height)
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
        if column in (3, 4) and row > 0:
            cell.set_text_props(family="monospace")

    figure.suptitle(
        "Downstream J-Lens trajectory of the L7+L22 SAE edit\n"
        f"position {capture_position} ({prompt_token!r}); "
        f"edited output P(Six)={edited_metrics['p_six']:.2%}, "
        f"P(Eight)={edited_metrics['p_eight']:.2%}",
        fontsize=15,
        y=0.985,
    )
    figure.text(
        0.5,
        0.015,
        "Published J-Lens transport is used through L32; the final block uses "
        "identity transport before final normalization and unembedding.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0.01, 0.035, 0.99, 0.965))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def contrast_value(record: dict[str, Any], name: str) -> float:
    return float(record["contrasts"][name]["log_probability_difference"])


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.spider_suppression <= 1.0:
        raise RuntimeError("--spider-suppression must be in [0, 1]")
    if args.ant_factor_l7 < 0.0 or args.ant_factor_l22 < 0.0:
        raise RuntimeError("Ant factors cannot be negative")
    if args.top_k < 1:
        raise RuntimeError("--top-k must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = read_manifest(args.feature_manifest)
    if source_manifest.get("model") != args.model_id:
        raise RuntimeError("Feature manifest and requested model do not match")
    if source_manifest.get("spider_prompt") != args.spider_prompt:
        raise RuntimeError("Feature manifest and requested prompt do not match")

    spider_instances = [
        instance
        for instance in source_manifest["spider_instances"]
        if int(instance["layer"]) in INTERVENTION_LAYERS
    ]
    ant_instances = [
        instance
        for instance in source_manifest["ant_instances"]
        if int(instance["layer"]) in INTERVENTION_LAYERS
    ]
    expected_spider_cells = {(7, 16), (22, 16), (22, 21)}
    actual_spider_cells = {
        (int(instance["layer"]), int(instance["position"]))
        for instance in spider_instances
    }
    if actual_spider_cells != expected_spider_cells:
        raise RuntimeError(f"Frozen spider cells changed: {actual_spider_cells}")
    expected_ant_cells = {(7, 16), (22, 16)}
    actual_ant_cells = {
        (int(instance["layer"]), int(instance["position"]))
        for instance in ant_instances
    }
    if actual_ant_cells != expected_ant_cells:
        raise RuntimeError(f"Frozen ant cells changed: {actual_ant_cells}")

    features_by_layer: dict[int, set[int]] = defaultdict(set)
    for instance in spider_instances + ant_instances:
        features_by_layer[int(instance["layer"])].add(int(instance["feature"]))
    rows = load_all_rows(features_by_layer, args.download_workers)
    spider_by_layer = group_instances(spider_instances)
    ant_by_layer = group_instances(ant_instances)

    hf_token = os.environ.get("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=hf_token,
        revision=args.model_revision,
    )
    input_ids = tokenizer(
        args.spider_prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    prompt_rows = token_rows(tokenizer, input_ids)
    if args.position >= len(prompt_rows):
        raise RuntimeError(f"Position {args.position} is outside the prompt")
    prompt_token = str(prompt_rows[args.position]["token"])
    if args.position == 16 and prompt_token != "spinning":
        raise RuntimeError(f"Expected P16='spinning', found {prompt_token!r}")

    print(f"Loading pinned {args.model_id} with eager attention")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        token=hf_token,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    layers = find_text_layers(model)
    last_layer = len(layers) - 1
    end_layer = last_layer if args.end_layer < 0 else args.end_layer
    if not 0 <= args.start_layer <= end_layer <= last_layer:
        raise RuntimeError(
            f"Capture range L{args.start_layer}-L{end_layer} is outside "
            f"the model's L0-L{last_layer} blocks"
        )
    capture_layers = list(range(args.start_layer, end_layer + 1))
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)

    target_ids = {
        name: single_token_id(tokenizer, f" {name}")
        for name in ("Six", "Eight", "Four")
    }
    families = surface_ids(tokenizer)
    published_layers = [layer for layer in capture_layers if layer != last_layer]
    lens_path, jacobians_cpu, lens_n_prompts, lens_d_model = load_selected_jacobians(
        lens_repo=args.lens_repo,
        lens_filename=args.lens_filename,
        lens_path=args.lens_path,
        selected_layers=published_layers,
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

    common = {
        "capture_layers": capture_layers,
        "capture_position": args.position,
        "spider_suppression": args.spider_suppression,
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
    }
    print("Running clean trajectory")
    clean = run_condition(
        condition="clean",
        active_layers=(),
        ant_factors={},
        **common,
    )
    print("Running full L7+L22 intervention trajectory")
    edited = run_condition(
        condition="edited",
        active_layers=INTERVENTION_LAYERS,
        ant_factors={7: args.ant_factor_l7, 22: args.ant_factor_l22},
        **common,
    )

    clean_metrics = clean["metrics"]
    edited_metrics = edited["metrics"]
    if clean_metrics["top_token"].strip().lower() != "eight":
        raise RuntimeError(f"Clean output is not Eight: {clean_metrics}")
    if clean_metrics["p_eight"] < 0.99:
        raise RuntimeError(f"Clean P(Eight) is below 99%: {clean_metrics}")
    if edited_metrics["top_token"].strip().lower() != "six":
        raise RuntimeError(f"Edited output is not Six: {edited_metrics}")
    if edited_metrics["p_six"] < 0.95:
        raise RuntimeError(f"Edited P(Six) is below 95%: {edited_metrics}")

    cleanup = run_condition(
        condition="clean_after_cleanup",
        active_layers=(),
        ant_factors={},
        **common,
    )
    for key in ("p_six", "p_eight", "p_four"):
        if not math.isclose(
            float(clean_metrics[key]),
            float(cleanup["metrics"][key]),
            rel_tol=1e-7,
            abs_tol=1e-10,
        ):
            raise RuntimeError(f"Hook cleanup differs for {key}")
    for layer in capture_layers:
        if not torch.equal(clean["residuals"][layer], cleanup["residuals"][layer]):
            raise RuntimeError(f"Hook cleanup residual differs at L{layer}/P{args.position}")

    readouts: dict[str, dict[int, dict[str, Any]]] = {
        "clean": {},
        "edited": {},
    }
    for condition, run in (("clean", clean), ("edited", edited)):
        for layer in capture_layers:
            residual = run["residuals"][layer].to(device=device, dtype=torch.float32)
            if layer == last_layer:
                transported = residual
            else:
                transported = residual @ jacobians[layer].T
            logits = lens_model.unembed(transported).detach().float().cpu()
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"Non-finite readout at {condition} L{layer}")
            readouts[condition][layer] = lens_record(
                logits=logits,
                tokenizer=tokenizer,
                families=families,
                top_k=args.top_k,
            )

    trajectory: list[dict[str, Any]] = []
    for layer in capture_layers:
        clean_residual = clean["residuals"][layer]
        edited_residual = edited["residuals"][layer]
        delta_norm = float((edited_residual - clean_residual).norm().item())
        clean_norm = float(clean_residual.norm().item())
        cosine = float(
            F.cosine_similarity(
                clean_residual.unsqueeze(0),
                edited_residual.unsqueeze(0),
            ).item()
        )
        if delta_norm <= 0.0:
            raise RuntimeError(f"No downstream residual change at L{layer}/P{args.position}")
        clean_readout = readouts["clean"][layer]
        edited_readout = readouts["edited"][layer]
        clean_ids = {
            int(item["token_id"]) for item in clean_readout["readable_top_tokens"]
        }
        edited_ids = {
            int(item["token_id"]) for item in edited_readout["readable_top_tokens"]
        }
        trajectory.append(
            {
                "layer": layer,
                "transport": "identity_final" if layer == last_layer else "published_jlens",
                "residual_delta_norm": delta_norm,
                "relative_residual_delta_norm": delta_norm / clean_norm,
                "clean_edited_cosine": cosine,
                "top5_token_overlap": len(clean_ids & edited_ids),
                "clean_ant_minus_spider": contrast_value(
                    clean_readout, "ant_minus_spider"
                ),
                "edited_ant_minus_spider": contrast_value(
                    edited_readout, "ant_minus_spider"
                ),
                "clean_insect_minus_spider": contrast_value(
                    clean_readout, "insect_minus_spider"
                ),
                "edited_insect_minus_spider": contrast_value(
                    edited_readout, "insect_minus_spider"
                ),
                "clean_six_minus_eight": contrast_value(
                    clean_readout, "six_minus_eight"
                ),
                "edited_six_minus_eight": contrast_value(
                    edited_readout, "six_minus_eight"
                ),
            }
        )
        clean_tokens = ", ".join(
            item["display_token"]
            for item in clean_readout["readable_top_tokens"]
        )
        edited_tokens = ", ".join(
            item["display_token"]
            for item in edited_readout["readable_top_tokens"]
        )
        print(
            f"L{layer}: ||delta h||={delta_norm:.3f}, cosine={cosine:.6f}; "
            f"clean=[{clean_tokens}] -> edited=[{edited_tokens}]"
        )

    artifact_prefix = f"sae_jlens_p{args.position}_downstream"
    figure_name = f"{artifact_prefix}_top5.png"
    trajectory_csv_name = f"{artifact_prefix}_trajectory.csv"
    top_concepts_csv_name = f"{artifact_prefix}_top_concepts.csv"
    manifest_name = f"{artifact_prefix}_manifest.json"
    json_name = f"{artifact_prefix}.json"
    figure_path = args.output_dir / figure_name
    trajectory_csv_path = args.output_dir / trajectory_csv_name
    top_csv_path = args.output_dir / top_concepts_csv_name
    plot_trajectory(
        output_path=figure_path,
        capture_position=args.position,
        prompt_token=prompt_token,
        capture_layers=capture_layers,
        readouts=readouts,
        trajectory=trajectory,
        edited_metrics=edited_metrics,
    )
    write_trajectory_csv(trajectory_csv_path, trajectory)
    write_top_concepts_csv(top_csv_path, readouts)

    runtime = {
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "jlens_version": importlib.metadata.version("jlens"),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "attention_implementation": "eager",
    }
    manifest = {
        "experiment": (
            f"P{args.position} downstream J-Lens trajectory after L7+L22 SAE edit"
        ),
        "created_at_unix": time.time(),
        "model": args.model_id,
        "source_feature_manifest": str(args.feature_manifest),
        "spider_prompt": args.spider_prompt,
        "position": args.position,
        "prompt_token": prompt_token,
        "capture_layers": capture_layers,
        "final_model_layer": last_layer,
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
            "published_source_layers": published_layers,
            "terminal_identity_layer": last_layer,
        },
        "runtime": runtime,
        "residual_tensor_exported": False,
    }
    manifest_path = args.output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    result = {
        "experiment": manifest["experiment"],
        "model": args.model_id,
        "position": args.position,
        "prompt_token": prompt_token,
        "capture_layers": capture_layers,
        "runtime": runtime,
        "lens": manifest["lens"],
        "conditions": {
            "clean": {
                "metrics": clean_metrics,
                "active_layers": clean["active_layers"],
            },
            "edited": {
                "metrics": edited_metrics,
                "active_layers": edited["active_layers"],
                "operations": edited["operations"],
            },
        },
        "trajectory": trajectory,
        "readouts": {
            condition: {str(layer): record for layer, record in records.items()}
            for condition, records in readouts.items()
        },
        "validation": {
            "clean_top_is_eight": True,
            "clean_p_eight_above_99_percent": True,
            "edited_top_is_six": True,
            "edited_p_six_at_least_95_percent": True,
            "hook_cleanup_exact": True,
            "all_downstream_residual_deltas_nonzero": True,
            "selected_jacobians_finite": True,
            "terminal_layer_uses_identity_transport": True,
            "full_residual_tensors_not_exported": True,
        },
        "artifacts": {
            "json": json_name,
            "manifest": manifest_name,
            "trajectory_csv": trajectory_csv_name,
            "top_concepts_csv": top_concepts_csv_name,
            "figure": figure_name,
        },
    }
    json_path = args.output_dir / json_name
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "Finished: "
        f"P(Six)={100 * edited_metrics['p_six']:.6f}%, "
        f"P(Eight)={100 * edited_metrics['p_eight']:.6f}%."
    )
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
