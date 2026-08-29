#!/usr/bin/env python3
"""Plot selected spider transcoder features across every prompt token."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import matplotlib
import numpy as np
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ID = "google/gemma-3-4b-it"
PROMPT = (
    "<bos><start_of_turn>user\n"
    "Answer in one word: How many legs does a web-spinning animal have?"
    "<end_of_turn>\n<start_of_turn>model\nAnswer:"
)
FEATURES = [
    {"layer": 22, "feature": 23422, "label": "spider and spiders"},
    {"layer": 22, "feature": 60476, "label": "spider and webs"},
    {"layer": 23, "feature": 56900, "label": "spiders"},
    {"layer": 24, "feature": 184534, "label": "spiders and webs"},
]
TRANSCODER_ROOT = (
    "https://huggingface.co/mwhanna/gemma-scope-2-4b-it/resolve/main/"
    "transcoder_all/width_262k_l0_small_affine"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/spider_feature_activation_history"),
    )
    return parser.parse_args()


def fetch_range(url: str, start: int, end: int) -> bytes:
    response = requests.get(
        url,
        headers={"Range": f"bytes={start}-{end}"},
        timeout=90,
        allow_redirects=True,
    )
    response.raise_for_status()
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"Server returned {len(response.content)} bytes for a {expected}-byte range"
        )
    return response.content


def load_feature_encoder(layer: int, feature: int) -> dict[str, np.ndarray | float]:
    url = f"{TRANSCODER_ROOT}/layer_{layer}.safetensors?download=true"
    header_size = struct.unpack("<Q", fetch_range(url, 0, 7))[0]
    header = json.loads(fetch_range(url, 8, 8 + header_size - 1))
    data_base = 8 + header_size

    def read_f32(name: str, flat_start: int, count: int) -> np.ndarray:
        metadata = header[name]
        tensor_start = data_base + metadata["data_offsets"][0]
        byte_start = tensor_start + flat_start * 4
        raw = fetch_range(url, byte_start, byte_start + count * 4 - 1)
        return np.frombuffer(raw, dtype="<f4").copy()

    width = header["W_enc"]["shape"][1]
    return {
        "W_enc": read_f32("W_enc", feature * width, width),
        "b_enc": float(read_f32("b_enc", feature, 1)[0]),
        "threshold": float(
            read_f32("activation_function.threshold", feature, 1)[0]
        ),
    }


def find_text_layers(model):
    candidates = [model, getattr(model, "model", None)]
    for candidate in list(candidates):
        if candidate is not None:
            candidates.append(getattr(candidate, "language_model", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise RuntimeError("Could not locate Gemma text decoder layers")


def single_token_id(tokenizer, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def display_token(token: str) -> str:
    return token.replace("\n", "\\n").replace("\t", "\\t") or "<empty>"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching the four selected feature encoders...")
    feature_rows = {
        (spec["layer"], spec["feature"]): load_feature_encoder(
            spec["layer"], spec["feature"]
        )
        for spec in FEATURES
    }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=True)
    input_ids = tokenizer(
        PROMPT, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    tokens = [
        tokenizer.decode([int(token_id)]) for token_id in input_ids[0]
    ]
    if len(tokens) != 27:
        raise RuntimeError(f"Expected 27 prompt tokens, found {len(tokens)}")

    print("Loading Gemma-3-4B-IT in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=True,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    layers = find_text_layers(model)
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in sorted({spec["layer"] for spec in FEATURES}):
        layer = layers[layer_index]

        def capture(_module, _args, output, layer_index=layer_index):
            captured[layer_index] = output.detach()

        handles.append(
            layer.pre_feedforward_layernorm.register_forward_hook(capture)
        )

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
    top_answer = top_predictions[0]["token"].strip().lower()
    if top_answer != "eight":
        raise RuntimeError(
            f"The clean model did not answer Eight; top token was {top_predictions[0]}"
        )

    named_probabilities = {}
    for answer in ["Eight", " Eight", "Six", " Six"]:
        token_id = single_token_id(tokenizer, answer)
        named_probabilities[answer] = (
            float(probabilities[token_id]) if token_id is not None else None
        )

    measurements = []
    for spec in FEATURES:
        row = feature_rows[spec["layer"], spec["feature"]]
        hidden = captured[spec["layer"]][0].float()
        encoder = torch.from_numpy(row["W_enc"]).to(
            device=device, dtype=torch.float32
        )
        preactivations = hidden @ encoder + float(row["b_enc"])
        threshold = float(row["threshold"])
        activations = torch.where(
            preactivations > threshold, preactivations, torch.zeros_like(preactivations)
        )
        measurements.append(
            {
                **spec,
                "threshold": threshold,
                "preactivations": preactivations.cpu().tolist(),
                "activations": activations.cpu().tolist(),
                "nonzero_positions": [
                    index
                    for index, activation in enumerate(activations)
                    if float(activation) > 0.0
                ],
            }
        )

    result = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "tokens": [
            {"position": index, "token": token}
            for index, token in enumerate(tokens)
        ],
        "top_predictions": top_predictions,
        "named_probabilities": named_probabilities,
        "features": measurements,
    }
    json_path = args.output_dir / "spider_feature_activation_history.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    x = np.arange(len(tokens))
    tick_labels = [
        f"{index}: {display_token(token)}" for index, token in enumerate(tokens)
    ]
    figure, axes = plt.subplots(
        len(measurements), 1, figsize=(18, 13), sharex=True, constrained_layout=True
    )
    for axis, measurement in zip(axes, measurements, strict=True):
        values = np.asarray(measurement["activations"])
        axis.plot(x, values, color="#475569", linewidth=1.5, marker="o", markersize=4)
        axis.fill_between(x, values, color="#38bdf8", alpha=0.28)
        nonzero = np.flatnonzero(values > 0)
        axis.scatter(nonzero, values[nonzero], color="#dc2626", s=30, zorder=3)
        for position in nonzero:
            axis.annotate(
                f"{values[position]:.0f}",
                (position, values[position]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_ylabel("activation")
        axis.set_title(
            f"L{measurement['layer']} F{measurement['feature']} — "
            f"{measurement['label']} (threshold {measurement['threshold']:.1f})",
            loc="left",
            fontsize=11,
        )
        axis.grid(axis="y", alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tick_labels, rotation=65, ha="right", fontsize=8)
    axes[-1].set_xlabel("prompt token position")
    figure.suptitle(
        "Spider-related transcoder feature activation over prompt history\n"
        f"Clean next-token answer: {top_predictions[0]['token']!r} "
        f"({top_predictions[0]['probability']:.1%})",
        fontsize=15,
    )
    figure_path = args.output_dir / "spider_feature_activation_history.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    print(json.dumps({
        "answer": top_predictions[0],
        "json": str(json_path),
        "figure": str(figure_path),
        "nonzero_positions": {
            f"L{item['layer']}F{item['feature']}": item["nonzero_positions"]
            for item in measurements
        },
    }, indent=2))


if __name__ == "__main__":
    main()
