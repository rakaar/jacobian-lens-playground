#!/usr/bin/env python3
"""Compare downstream SAE-edit effects across prompt positions P16-P21."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


DEFAULT_POSITIONS = (16, 17, 18, 19, 20, 21)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean-versus-edited J-Lens trajectories by position"
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/sae_jlens_position_comparison"),
    )
    parser.add_argument(
        "--positions",
        type=int,
        nargs="+",
        default=list(DEFAULT_POSITIONS),
    )
    return parser.parse_args()


def display_token(position: int, token: str) -> str:
    if token == "\n":
        token = "newline"
    elif token == "<end_of_turn>":
        token = "<end_of_turn>"
    else:
        token = token.strip()
    direct = " *" if position in (16, 21) else ""
    return f"P{position}  {token}{direct}"


def load_rows(results_root: Path, positions: list[int]) -> tuple[list[dict], list[int]]:
    rows: list[dict] = []
    layers: list[int] | None = None
    for position in positions:
        result_path = (
            results_root
            / f"sae_jlens_p{position}_downstream"
            / f"sae_jlens_p{position}_downstream.json"
        )
        result = json.loads(result_path.read_text())
        trajectory = result["trajectory"]
        current_layers = [int(item["layer"]) for item in trajectory]
        if layers is None:
            layers = current_layers
        elif current_layers != layers:
            raise RuntimeError(f"Layer grid differs in {result_path}")
        for item in trajectory:
            rows.append(
                {
                    "position": position,
                    "prompt_token": result["prompt_token"],
                    "directly_edited_position": position in (16, 21),
                    "layer": int(item["layer"]),
                    "top5_token_overlap": int(item["top5_token_overlap"]),
                    "residual_delta_norm": float(item["residual_delta_norm"]),
                    "relative_residual_delta_norm": float(
                        item["relative_residual_delta_norm"]
                    ),
                    "clean_edited_cosine": float(item["clean_edited_cosine"]),
                }
            )
    if layers is None:
        raise RuntimeError("No positions were supplied")
    return rows, layers


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def annotate(
    ax: plt.Axes,
    values: np.ndarray,
    fmt: str,
    threshold: float,
    white_below: float | None = None,
) -> None:
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            use_white = value >= threshold or (
                white_below is not None and value <= white_below
            )
            color = "white" if use_white else "black"
            ax.text(
                column_index,
                row_index,
                format(value, fmt),
                ha="center",
                va="center",
                fontsize=8.5,
                color=color,
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positions = list(args.positions)
    rows, layers = load_rows(args.results_root, positions)

    by_cell = {(row["position"], row["layer"]): row for row in rows}
    overlap = np.array(
        [
            [by_cell[(position, layer)]["top5_token_overlap"] for layer in layers]
            for position in positions
        ],
        dtype=float,
    )
    relative_delta_percent = 100.0 * np.array(
        [
            [
                by_cell[(position, layer)]["relative_residual_delta_norm"]
                for layer in layers
            ]
            for position in positions
        ],
        dtype=float,
    )
    tokens = {
        row["position"]: str(row["prompt_token"])
        for row in rows
    }
    ylabels = [display_token(position, tokens[position]) for position in positions]
    xlabels = [f"L{layer}" for layer in layers]

    figure, axes = plt.subplots(2, 1, figsize=(15, 8.6), constrained_layout=True)
    layout_engine = figure.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.055, 1.0, 0.94), hspace=0.08)
    figure.suptitle(
        "Where the frozen L7+L22 SAE edit changes downstream J-Lens readouts",
        fontsize=17,
        fontweight="bold",
    )

    overlap_cmap = ListedColormap(
        ["#7f0000", "#b30000", "#e34a33", "#fdbb84", "#bdd7e7", "#2b8cbe"]
    )
    overlap_norm = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), overlap_cmap.N)
    image_overlap = axes[0].imshow(
        overlap,
        cmap=overlap_cmap,
        norm=overlap_norm,
        aspect="auto",
    )
    axes[0].set_title(
        "Clean/edited top-five token overlap (0 = fully changed, 5 = identical)",
        fontsize=12,
    )
    annotate(axes[0], overlap, ".0f", threshold=4.5, white_below=1.0)
    colorbar_overlap = figure.colorbar(
        image_overlap,
        ax=axes[0],
        ticks=range(6),
        pad=0.015,
        fraction=0.025,
    )
    colorbar_overlap.set_label("Shared top-five tokens")

    max_delta = float(np.ceil(relative_delta_percent.max()))
    image_delta = axes[1].imshow(
        relative_delta_percent,
        cmap="magma",
        vmin=0.0,
        vmax=max_delta,
        aspect="auto",
    )
    axes[1].set_title(
        "Residual change as a percentage of the clean residual norm",
        fontsize=12,
    )
    annotate(
        axes[1],
        relative_delta_percent,
        ".1f",
        threshold=0.58 * max_delta,
    )
    colorbar_delta = figure.colorbar(
        image_delta,
        ax=axes[1],
        pad=0.015,
        fraction=0.025,
    )
    colorbar_delta.set_label("Relative residual delta (%)")

    for ax in axes:
        ax.set_xticks(np.arange(len(layers)), labels=xlabels)
        ax.set_yticks(np.arange(len(positions)), labels=ylabels)
        ax.set_xlabel("Downstream block output")
        ax.tick_params(axis="x", rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    figure.text(
        0.5,
        0.012,
        (
            "* Directly edited position. P16 receives L7 and L22 edits; P21 "
            "receives L22 spider suppression only. Other rows show propagation."
        ),
        ha="center",
        fontsize=10,
    )

    figure_path = args.output_dir / "sae_jlens_position_comparison.png"
    csv_path = args.output_dir / "sae_jlens_position_comparison.csv"
    manifest_path = args.output_dir / "sae_jlens_position_comparison_manifest.json"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    write_csv(csv_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "experiment": "P16-P21 downstream J-Lens position comparison",
                "positions": positions,
                "layers": layers,
                "directly_edited_positions": {
                    "16": "L7 and L22 spider suppression plus ant injection",
                    "21": "L22 spider suppression only",
                },
                "source_results": [
                    str(
                        args.results_root
                        / f"sae_jlens_p{position}_downstream"
                        / f"sae_jlens_p{position}_downstream.json"
                    )
                    for position in positions
                ],
                "artifacts": {
                    "figure": figure_path.name,
                    "csv": csv_path.name,
                },
            },
            indent=2,
        )
    )
    print(f"Saved comparison figure to {figure_path}")
    print(f"Saved tidy comparison data to {csv_path}")


if __name__ == "__main__":
    main()
