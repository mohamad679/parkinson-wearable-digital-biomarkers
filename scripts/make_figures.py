"""Render a deterministic SVG summary from baseline benchmark JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any


class FigureInputError(ValueError):
    """Raised when benchmark JSON cannot produce a valid figure."""


def make_benchmark_figure(
    benchmark_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a reproducible grouped-bar SVG for AUROC and AUPRC."""
    benchmark = _load_benchmark(benchmark_path)
    models = _validated_model_metrics(benchmark)
    svg = _render_svg(models)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def _load_benchmark(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FigureInputError(f"Unable to read benchmark JSON: {error}") from error
    if not isinstance(value, dict):
        raise FigureInputError("Benchmark JSON must contain an object")
    return value


def _validated_model_metrics(
    benchmark: dict[str, Any],
) -> tuple[tuple[str, float, float], ...]:
    raw_models = benchmark.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise FigureInputError("Benchmark JSON must contain non-empty model results")

    models: list[tuple[str, float, float]] = []
    for model_name in sorted(raw_models):
        metrics = raw_models[model_name]
        if not isinstance(model_name, str) or not isinstance(metrics, dict):
            raise FigureInputError("Model results must be named objects")
        models.append(
            (
                model_name,
                _unit_metric(metrics.get("auroc"), "AUROC"),
                _unit_metric(metrics.get("auprc"), "AUPRC"),
            )
        )
    return tuple(models)


def _unit_metric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise FigureInputError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise FigureInputError(f"{name} must be finite and between 0 and 1")
    return normalized


def _render_svg(models: tuple[tuple[str, float, float], ...]) -> str:
    width = 760
    height = 440
    plot_left = 80
    plot_top = 70
    plot_height = 270
    plot_bottom = plot_top + plot_height
    group_width = (width - plot_left - 40) / 2
    bar_width = min(60.0, group_width / (len(models) + 1))
    colors = ("#35618f", "#c46b32", "#4f8a5b", "#8b5ca5")
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            '<text x="380" y="30" text-anchor="middle" font-family="sans-serif" '
            'font-size="20">Synthetic baseline discrimination metrics</text>'
        ),
    ]

    for tick in range(5):
        value = tick / 4
        y = plot_bottom - value * plot_height
        lines.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="720" y2="{y:.1f}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="68" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{value:.2f}</text>'
        )

    for metric_index, metric_name in enumerate(("AUROC", "AUPRC")):
        group_center = plot_left + group_width * (metric_index + 0.5)
        total_bar_width = bar_width * len(models)
        for model_index, (_model_name, auroc_value, auprc_value) in enumerate(models):
            value = auroc_value if metric_index == 0 else auprc_value
            x = group_center - total_bar_width / 2 + model_index * bar_width
            bar_height = value * plot_height
            y = plot_bottom - bar_height
            color = colors[model_index % len(colors)]
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 6:.1f}" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{x + (bar_width - 6) / 2:.1f}" y="{y - 6:.1f}" '
                f'text-anchor="middle" font-family="sans-serif" font-size="11">'
                f'{value:.3f}</text>'
            )
        lines.append(
            f'<text x="{group_center:.1f}" y="365" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{metric_name}</text>'
        )

    legend_x = 110
    for model_index, (model_name, _, _) in enumerate(models):
        color = colors[model_index % len(colors)]
        label = html.escape(model_name.replace("_", " "))
        x = legend_x + model_index * 260
        lines.append(f'<rect x="{x}" y="390" width="14" height="14" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 21}" y="402" font-family="sans-serif" '
            f'font-size="12">{label}</text>'
        )

    lines.append(
        '<text x="380" y="430" text-anchor="middle" font-family="sans-serif" '
        'font-size="11" fill="#555555">Synthetic, non-diagnostic research output</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a reproducible SVG from benchmark JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="Benchmark JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/synthetic_benchmark.svg"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the figure generation command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        output = make_benchmark_figure(arguments.input, arguments.output)
    except FigureInputError as error:
        parser.error(str(error))
    print(json.dumps({"figure": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
