#!/usr/bin/env python3
"""Test whether the frozen spider-to-ant SAE edit changes animal identity.

The established task asks for a leg count, so a successful ``Eight -> Six``
flip could reflect either a precise ``spider -> ant`` substitution or only a
broader ``arachnid -> insect`` shift.  This control changes the question while
preserving the original 27-token layout:

    How many legs does a web-spinning animal have?
    What animal is famous for web-spinning behavior worldwide?

Consequently, position 16 is still ``spinning`` and position 21 is still the
turn-boundary newline.  The exact frozen L7+L22 feature sites and strengths can
therefore be reused without an ambiguous positional remapping.
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
    feature_activation,
    find_text_layers,
    load_all_rows,
    read_manifest,
    token_rows,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
DEFAULT_FEATURE_MANIFEST = Path(
    "results/spider_ant_layer_specific_optimization/"
    "spider_ant_layer_specific_manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("results/spider_ant_identity_control")
IDENTITY_PROMPT = (
    "<bos><start_of_turn>user\n"
    "Answer in one word: What animal is famous for web-spinning behavior worldwide?"
    "<end_of_turn>\n<start_of_turn>model\nAnswer:"
)
TESTED_LAYERS = (7, 22)
SPIDER_SUPPRESSION = 1.0
ANT_FACTOR = 4.0

OUTPUT_JSON = "spider_ant_identity_control.json"
OUTPUT_CSV = "spider_ant_identity_control.csv"
OUTPUT_MANIFEST = "spider_ant_identity_control_manifest.json"
FIGURE = "spider_ant_identity_control.png"

ANIMAL_SURFACES: dict[str, tuple[str, ...]] = {
    "spider": (" Spider", " spider", "Spider", "spider", " Spiders", " spiders"),
    "ant": (" Ant", " ant", "Ant", "ant", " Ants", " ants"),
    "wasp": (" Wasp", " wasp", "Wasp", "wasp", " Wasps", " wasps"),
    "termite": (
        " Termite",
        " termite",
        "Termite",
        "termite",
        " Termites",
        " termites",
    ),
    "bee": (" Bee", " bee", "Bee", "bee", " Bees", " bees"),
    "insect": (" Insect", " insect", "Insect", "insect", " Insects", " insects"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a matched-token animal-identity control for the frozen SAE edit"
    )
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_FEATURE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--identity-prompt", default=IDENTITY_PROMPT)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--download-workers", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the pinned model/tokenizer only from the pod cache.",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def grouped_instances(
    instances: Iterable[dict[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        grouped[int(instance["layer"])].append(dict(instance))
    return {
        layer: sorted(values, key=lambda item: (int(item["position"]), int(item["feature"])))
        for layer, values in grouped.items()
    }


def animal_token_ids(tokenizer: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for family, surfaces in ANIMAL_SURFACES.items():
        unique: dict[int, str] = {}
        for surface in surfaces:
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                unique.setdefault(int(ids[0]), surface)
        if not unique:
            raise RuntimeError(f"No single-token variants found for {family}")
        result[family] = [
            {
                "token_id": token_id,
                "surface": surface,
                "decoded": tokenizer.decode([token_id]),
            }
            for token_id, surface in sorted(unique.items())
        ]
    return result


def rank_for_probability(probabilities: torch.Tensor, token_id: int) -> int:
    value = probabilities[int(token_id)]
    return int((probabilities > value).sum().item()) + 1


def identity_record(
    logits: torch.Tensor,
    tokenizer: Any,
    family_ids: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> dict[str, Any]:
    probabilities = torch.softmax(logits.detach().float().cpu(), dim=-1)
    top_probabilities, top_ids = torch.topk(probabilities, k=top_k)
    top_tokens = [
        {
            "rank": rank,
            "token": tokenizer.decode([int(token_id)]),
            "token_id": int(token_id),
            "probability": float(probability),
        }
        for rank, (probability, token_id) in enumerate(
            zip(top_probabilities, top_ids, strict=True), start=1
        )
    ]

    families: dict[str, dict[str, Any]] = {}
    for family, variants in family_ids.items():
        ids = torch.tensor([item["token_id"] for item in variants], dtype=torch.long)
        values = probabilities[ids]
        best_index = int(torch.argmax(values).item())
        best = variants[best_index]
        families[family] = {
            "family": family,
            "family_probability": float(values.sum().item()),
            "best_token": tokenizer.decode([int(best["token_id"])]),
            "best_token_id": int(best["token_id"]),
            "best_token_probability": float(values[best_index].item()),
            "best_token_rank": rank_for_probability(probabilities, int(best["token_id"])),
            "variants": variants,
        }

    animal_ranking = sorted(
        families.values(), key=lambda item: item["family_probability"], reverse=True
    )
    return {
        "answer": top_tokens[0]["token"].strip(),
        "top_token": top_tokens[0]["token"],
        "top_token_id": top_tokens[0]["token_id"],
        "top_probability": top_tokens[0]["probability"],
        "top_tokens": top_tokens,
        "animal_families": families,
        "animal_family_ranking": [item["family"] for item in animal_ranking],
        "top_animal_family": animal_ranking[0]["family"],
        "top_animal_family_probability": animal_ranking[0]["family_probability"],
    }


def run_forward(
    *,
    condition: str,
    model: torch.nn.Module,
    layers: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    family_ids: dict[str, list[dict[str, Any]]],
    rows: dict[FeatureKey, dict[str, Any]],
    spider_by_layer: dict[int, list[dict[str, Any]]],
    ant_by_layer: dict[int, list[dict[str, Any]]],
    spider_suppression: float,
    ant_factor: float,
    top_k: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    states: dict[int, torch.Tensor] = {}
    operations: list[dict[str, Any]] = []
    handles: list[Any] = []

    for layer_index in TESTED_LAYERS:
        def capture(_module, _args, output, layer_index=layer_index):
            states[layer_index] = output

        def alter(_module, _args, output, layer_index=layer_index):
            if layer_index not in states:
                raise RuntimeError(f"{condition}: L{layer_index} edit ran before capture")
            changed = output.clone()
            delta_by_position: dict[int, torch.Tensor] = {}

            for role, instances in (
                ("spider", spider_by_layer.get(layer_index, [])),
                ("ant", ant_by_layer.get(layer_index, [])),
            ):
                for instance in instances:
                    key = FeatureKey(layer_index, int(instance["feature"]))
                    position = int(instance["position"])
                    preactivation, activation = feature_activation(
                        states[layer_index], position, rows[key], device
                    )
                    if role == "spider":
                        target = activation * (1.0 - spider_suppression)
                        factor = -spider_suppression
                        reference_activation = activation
                    else:
                        reference_activation = float(instance["reference_activation"])
                        target = activation + ant_factor * reference_activation
                        factor = ant_factor
                    decoder = torch.from_numpy(rows[key]["W_dec"]).to(
                        device=device, dtype=torch.float32
                    )
                    delta = (target - activation) * decoder
                    delta_by_position[position] = (
                        delta_by_position.get(position, torch.zeros_like(delta)) + delta
                    )
                    operations.append(
                        {
                            "role": role,
                            "layer": layer_index,
                            "feature": key.feature,
                            "position": position,
                            "current_preactivation": preactivation,
                            "current_activation": activation,
                            "reference_activation": reference_activation,
                            "target_activation": target,
                            "factor": factor,
                            "delta_norm": float(delta.norm().item()),
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

    total_delta_norm = math.sqrt(sum(item["delta_norm"] ** 2 for item in operations))
    return (
        {
            "condition": condition,
            "spider_suppression": spider_suppression,
            "ant_factor": ant_factor,
            "total_delta_norm": total_delta_norm,
            "operations": operations,
            **identity_record(logits, tokenizer, family_ids, top_k),
        },
        logits.detach().float().cpu(),
    )


def clean_forward(
    *,
    condition: str,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tokenizer: Any,
    family_ids: dict[str, list[dict[str, Any]]],
    top_k: int,
) -> tuple[dict[str, Any], torch.Tensor]:
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    return (
        {
            "condition": condition,
            "spider_suppression": 0.0,
            "ant_factor": 0.0,
            "total_delta_norm": 0.0,
            "operations": [],
            **identity_record(logits, tokenizer, family_ids, top_k),
        },
        logits.detach().float().cpu(),
    )


def add_greedy_completion(
    *,
    run: dict[str, Any],
    first_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    common: dict[str, Any],
) -> dict[str, Any]:
    """Decode the short answer while reapplying the same edit on every step."""
    current_ids = input_ids
    logits = first_logits
    generated: list[int] = []
    stop_ids = {
        int(token_id)
        for token_id in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        )
        if token_id is not None and int(token_id) >= 0
    }
    for step in range(max_new_tokens):
        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)
        if next_id in stop_ids or "\n" in tokenizer.decode([next_id]):
            break
        current_ids = torch.cat(
            [
                current_ids,
                torch.tensor([[next_id]], dtype=current_ids.dtype, device=current_ids.device),
            ],
            dim=1,
        )
        if step + 1 >= max_new_tokens:
            break
        if run["condition"] == "clean":
            _, logits = clean_forward(
                condition="clean_generation_step",
                model=common["model"],
                input_ids=current_ids,
                tokenizer=tokenizer,
                family_ids=common["family_ids"],
                top_k=common["top_k"],
            )
        else:
            _, logits = run_forward(
                condition=f"{run['condition']}_generation_step",
                input_ids=current_ids,
                spider_suppression=float(run["spider_suppression"]),
                ant_factor=float(run["ant_factor"]),
                **common,
            )
    completion = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return {
        **run,
        "greedy_completion": completion,
        "greedy_token_ids": generated,
        "greedy_tokens": [tokenizer.decode([token_id]) for token_id in generated],
    }


def write_csv(path: Path, prompt_name: str, runs: list[dict[str, Any]]) -> None:
    fields = [
        "prompt",
        "condition",
        "answer",
        "top_probability",
        "top_animal_family",
        "family",
        "family_probability",
        "best_token",
        "best_token_probability",
        "best_token_rank",
        "spider_suppression",
        "ant_factor",
        "total_delta_norm",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            for family, record in run["animal_families"].items():
                writer.writerow(
                    {
                        "prompt": prompt_name,
                        "condition": run["condition"],
                        "answer": run.get("greedy_completion") or run["answer"],
                        "top_probability": run["top_probability"],
                        "top_animal_family": run["top_animal_family"],
                        "family": family,
                        "family_probability": record["family_probability"],
                        "best_token": record["best_token"],
                        "best_token_probability": record["best_token_probability"],
                        "best_token_rank": record["best_token_rank"],
                        "spider_suppression": run["spider_suppression"],
                        "ant_factor": run["ant_factor"],
                        "total_delta_norm": run["total_delta_norm"],
                    }
                )


def plot_identity(path: Path, runs: list[dict[str, Any]]) -> None:
    families = list(ANIMAL_SURFACES)
    conditions = [run["condition"] for run in runs]
    values = np.array(
        [
            [100.0 * run["animal_families"][family]["family_probability"] for family in families]
            for run in runs
        ]
    )

    figure, axis = plt.subplots(figsize=(15.2, 6.4))
    image = axis.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=max(100.0, float(values.max())))
    axis.set_xticks(np.arange(len(families)), [name.title() for name in families])
    axis.set_yticks(
        np.arange(len(conditions)),
        [name.replace("_", " ").title() for name in conditions],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            text = f"{value:.2f}%" if value >= 0.01 else "<0.01%"
            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > 45 else "#111827",
                fontsize=9,
            )
        answer = runs[row].get("greedy_completion") or runs[row]["answer"] or "<empty>"
        probability = 100.0 * float(runs[row]["top_probability"])
        axis.text(
            len(families) - 0.25,
            row,
            f"  answer: {answer} ({probability:.2f}%)",
            ha="left",
            va="center",
            fontsize=9,
            clip_on=False,
        )
    axis.set_xlim(-0.5, len(families) + 1.65)
    axis.set_title(
        "Does the frozen L7+L22 SAE edit replace spider identity with ant?\n"
        "Matched prompt keeps 'spinning' at P16 and the turn boundary at P21",
        loc="left",
    )
    axis.set_xlabel("tracked next-token animal family probability")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.03)
    colorbar.set_label("probability (%)")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def compact(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key != "operations"}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_manifest(args.feature_manifest)
    if source.get("model") != MODEL_ID:
        raise RuntimeError("The feature manifest does not match Gemma-3-4B-IT")

    spider_instances = [
        dict(item)
        for item in source["spider_instances"]
        if int(item["layer"]) in TESTED_LAYERS
    ]
    ant_instances = [
        dict(item)
        for item in source["ant_instances"]
        if int(item["layer"]) in TESTED_LAYERS
    ]
    spider_by_layer = grouped_instances(spider_instances)
    ant_by_layer = grouped_instances(ant_instances)
    if set(spider_by_layer) != set(TESTED_LAYERS) or set(ant_by_layer) != set(TESTED_LAYERS):
        raise RuntimeError("The frozen manifest lacks L7 or L22 spider/ant instances")

    feature_ids: dict[int, set[int]] = defaultdict(set)
    for instance in spider_instances + ant_instances:
        feature_ids[int(instance["layer"])].add(int(instance["feature"]))
    rows = load_all_rows(dict(feature_ids), args.download_workers)

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token and not args.local_files_only:
        raise RuntimeError("HF_TOKEN is required to load gated Gemma weights")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=hf_token or None,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    identity_ids = tokenizer(
        args.identity_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    legs_ids = tokenizer(SPIDER_PROMPT, add_special_tokens=False, return_tensors="pt").input_ids
    identity_tokens = token_rows(tokenizer, identity_ids)
    legs_tokens = token_rows(tokenizer, legs_ids)
    if len(identity_tokens) != len(legs_tokens) or len(identity_tokens) != 27:
        raise RuntimeError(
            f"Matched prompts must both contain 27 tokens; got "
            f"identity={len(identity_tokens)}, legs={len(legs_tokens)}"
        )
    edited_positions = sorted(
        {int(item["position"]) for item in spider_instances + ant_instances}
    )
    alignment = []
    for position in edited_positions:
        identity_token = identity_tokens[position]["token"]
        legs_token = legs_tokens[position]["token"]
        alignment.append(
            {
                "position": position,
                "legs_token": legs_token,
                "identity_token": identity_token,
                "exact_match": identity_token == legs_token,
            }
        )
    if not all(item["exact_match"] for item in alignment):
        raise RuntimeError(f"Edited token sites are not exactly aligned: {alignment}")
    family_ids = animal_token_ids(tokenizer)

    print("Matched identity-prompt tokens:")
    for row in identity_tokens:
        print(f"  P{row['position']:02d}: {row['token']!r}")
    print("Loading pinned Gemma-3-4B-IT in bfloat16 with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=hf_token or None,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    layers = find_text_layers(model)
    device = next(model.parameters()).device
    identity_ids = identity_ids.to(device)
    legs_ids = legs_ids.to(device)

    common = {
        "model": model,
        "layers": layers,
        "tokenizer": tokenizer,
        "family_ids": family_ids,
        "rows": rows,
        "spider_by_layer": spider_by_layer,
        "ant_by_layer": ant_by_layer,
        "top_k": args.top_k,
        "device": device,
    }
    identity_runs: list[dict[str, Any]] = []
    identity_logits: dict[str, torch.Tensor] = {}
    clean, clean_logits = clean_forward(
        condition="clean",
        model=model,
        input_ids=identity_ids,
        tokenizer=tokenizer,
        family_ids=family_ids,
        top_k=args.top_k,
    )
    clean = add_greedy_completion(
        run=clean,
        first_logits=clean_logits,
        input_ids=identity_ids,
        tokenizer=tokenizer,
        max_new_tokens=4,
        common=common,
    )
    identity_runs.append(clean)
    identity_logits["clean"] = clean_logits

    for condition, spider_suppression, ant_factor in (
        ("zero_strength_control", 0.0, 0.0),
        ("spider_suppression_only", SPIDER_SUPPRESSION, 0.0),
        ("ant_injection_only", 0.0, ANT_FACTOR),
        ("full_edit", SPIDER_SUPPRESSION, ANT_FACTOR),
    ):
        run, logits = run_forward(
            condition=condition,
            input_ids=identity_ids,
            spider_suppression=spider_suppression,
            ant_factor=ant_factor,
            **common,
        )
        run = add_greedy_completion(
            run=run,
            first_logits=logits,
            input_ids=identity_ids,
            tokenizer=tokenizer,
            max_new_tokens=4,
            common=common,
        )
        identity_runs.append(run)
        identity_logits[condition] = logits
        print(
            f"Identity {condition}: answer={run['greedy_completion']!r} "
            f"P={100 * run['top_probability']:.6f}% "
            f"top tracked family={run['top_animal_family']}"
        )

    legs_clean, legs_clean_logits = clean_forward(
        condition="clean",
        model=model,
        input_ids=legs_ids,
        tokenizer=tokenizer,
        family_ids=family_ids,
        top_k=args.top_k,
    )
    legs_full, _ = run_forward(
        condition="full_edit",
        input_ids=legs_ids,
        spider_suppression=SPIDER_SUPPRESSION,
        ant_factor=ANT_FACTOR,
        **common,
    )
    post_cleanup, post_cleanup_logits = clean_forward(
        condition="post_cleanup",
        model=model,
        input_ids=identity_ids,
        tokenizer=tokenizer,
        family_ids=family_ids,
        top_k=args.top_k,
    )

    zero_matches = torch.allclose(
        clean_logits, identity_logits["zero_strength_control"], rtol=1e-6, atol=1e-7
    )
    cleanup_matches = torch.allclose(
        clean_logits, post_cleanup_logits, rtol=1e-6, atol=1e-7
    )
    if not zero_matches:
        raise RuntimeError("Zero-strength identity hooks do not match the clean logits")
    if not cleanup_matches:
        raise RuntimeError("Identity logits changed after hook cleanup")

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
    validation = {
        "both_prompts_have_27_tokens": len(identity_tokens) == len(legs_tokens) == 27,
        "edited_positions_exactly_aligned": all(item["exact_match"] for item in alignment),
        "clean_identity_answer_is_spider": clean["greedy_completion"].lower() == "spider",
        "zero_strength_matches_clean_logits": bool(zero_matches),
        "hooks_removed": bool(cleanup_matches),
        "legs_clean_answer_is_eight": legs_clean["answer"].lower() == "eight",
        "legs_full_answer_is_six": legs_full["answer"].lower() == "six",
    }
    manifest = {
        "experiment": "matched-token spider-to-ant animal-identity control",
        "created_at_unix": time.time(),
        "model": MODEL_ID,
        "source_manifest": str(args.feature_manifest),
        "identity_prompt": args.identity_prompt,
        "legs_prompt": SPIDER_PROMPT,
        "identity_prompt_tokens": identity_tokens,
        "legs_prompt_tokens": legs_tokens,
        "edited_position_alignment": alignment,
        "layers": list(TESTED_LAYERS),
        "spider_suppression": SPIDER_SUPPRESSION,
        "ant_factor": ANT_FACTOR,
        "spider_instances": spider_instances,
        "ant_instances": ant_instances,
        "animal_token_variants": family_ids,
        "runtime": runtime,
    }
    result = {
        "experiment": "matched-token spider-to-ant animal-identity control",
        "model": MODEL_ID,
        "identity_prompt": args.identity_prompt,
        "identity_runs": identity_runs,
        "positive_control": {
            "prompt": SPIDER_PROMPT,
            "clean": compact(legs_clean),
            "full_edit": compact(legs_full),
        },
        "post_cleanup_identity": compact(post_cleanup),
        "runtime": runtime,
        "validation": validation,
        "artifacts": {
            "json": OUTPUT_JSON,
            "csv": OUTPUT_CSV,
            "manifest": OUTPUT_MANIFEST,
            "figure": FIGURE,
        },
    }

    json_path = args.output_dir / OUTPUT_JSON
    csv_path = args.output_dir / OUTPUT_CSV
    manifest_path = args.output_dir / OUTPUT_MANIFEST
    figure_path = args.output_dir / FIGURE
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    write_csv(csv_path, "identity", identity_runs)
    plot_identity(figure_path, identity_runs)

    print(
        f"Leg-count positive control: clean={legs_clean['answer']!r} "
        f"({100 * legs_clean['top_probability']:.6f}%), "
        f"full_edit={legs_full['answer']!r} "
        f"({100 * legs_full['top_probability']:.6f}%)"
    )
    print(f"Validation: {validation}")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
