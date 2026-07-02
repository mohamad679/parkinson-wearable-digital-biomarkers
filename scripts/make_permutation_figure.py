"""Render a side-by-side null-distribution histogram SVG from a permutation JSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


class PermFigureInputError(ValueError):
    """Raised when the permutation JSON cannot produce a valid figure."""


# ── Public API ────────────────────────────────────────────────────────────────


def make_permutation_figure(
    benchmark_path: str | Path,
    output_path: str | Path,
    *,
    n_bins: int = 20,
) -> Path:
    """Create a two-panel null-distribution histogram SVG.

    The left panel shows the logistic regression null distribution; the right
    panel shows the random forest null distribution.  Each panel contains grey
    histogram bars and a vertical dashed red line at the observed AUROC, with
    an annotation showing the observed value and empirical p-value.

    Parameters
    ----------
    benchmark_path:
        Path to the permutation JSON produced by ``permutation_test.py``.
    output_path:
        Destination for the rendered SVG file.
    n_bins:
        Number of equal-width histogram bins spanning [0, 1].
    """
    data = _load_benchmark(benchmark_path)
    models = _extract_model_data(data)
    svg = _render_svg(models, n_bins=n_bins)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


# ── Data extraction ───────────────────────────────────────────────────────────


def _load_benchmark(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PermFigureInputError(f"Unable to read permutation JSON: {error}") from error
    if not isinstance(value, dict):
        raise PermFigureInputError("Permutation JSON must contain an object")
    return value


def _extract_model_data(
    data: dict[str, Any],
) -> list[tuple[str, float, float, list[float]]]:
    """Return list of (model_name, observed_auroc, p_value, null_auroc) tuples."""
    raw_models = data.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise PermFigureInputError("Permutation JSON must contain non-empty model results")

    result: list[tuple[str, float, float, list[float]]] = []
    for model_name in sorted(raw_models):
        model = raw_models[model_name]
        if not isinstance(model, dict):
            raise PermFigureInputError(f"Model {model_name!r} data must be an object")
        if "error" in model:
            raise PermFigureInputError(
                f"Model {model_name!r} has an error: {model['error']}"
            )

        observed = model.get("observed_auroc")
        p_value = model.get("p_value")
        null_auroc = model.get("null_auroc")

        if not isinstance(observed, int | float) or isinstance(observed, bool):
            raise PermFigureInputError(f"observed_auroc for {model_name!r} must be numeric")
        if not isinstance(p_value, int | float) or isinstance(p_value, bool):
            raise PermFigureInputError(f"p_value for {model_name!r} must be numeric")
        if not isinstance(null_auroc, list) or not null_auroc:
            raise PermFigureInputError(
                f"null_auroc for {model_name!r} must be a non-empty list"
            )

        result.append((model_name, float(observed), float(p_value), [float(v) for v in null_auroc]))
    return result


# ── Histogram helper ──────────────────────────────────────────────────────────


def _histogram_bins(
    values: list[float], n_bins: int
) -> list[tuple[float, float, int]]:
    """Return (lower, upper, count) for each equal-width bin over [0, 1]."""
    bins: list[tuple[float, float, int]] = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i < n_bins - 1:
            count = sum(1 for v in values if lo <= v < hi)
        else:
            # Last bin is closed on the right to capture exactly 1.0.
            count = sum(1 for v in values if lo <= v <= hi)
        bins.append((lo, hi, count))
    return bins


# ── SVG renderer ──────────────────────────────────────────────────────────────


def _render_svg(
    models: list[tuple[str, float, float, list[float]]],
    n_bins: int,
) -> str:
    n_models = len(models)
    panel_w = 370
    gap = 40
    margin_left = 65
    margin_right = 20
    margin_top = 65
    margin_bottom = 75
    plot_h = 260

    total_w = margin_left + n_models * panel_w + (n_models - 1) * gap + margin_right
    total_h = margin_top + plot_h + margin_bottom

    bar_color = "#94a3b8"
    bar_stroke = "#64748b"
    obs_color = "#dc2626"
    grid_color = "#e5e7eb"
    axis_color = "#374151"
    label_font = "font-family='sans-serif' font-size='11' fill='#374151'"
    title_font = "font-family='sans-serif' font-size='13' font-weight='bold' fill='#1e293b'"
    note_font = "font-family='sans-serif' font-size='10' fill='#64748b'"
    caption_font = "font-family='sans-serif' font-size='18' font-weight='bold' fill='#111827'"

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" '
            f'height="{total_h}" viewBox="0 0 {total_w} {total_h}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{total_w // 2}" y="28" text-anchor="middle" {caption_font}>'
            f'Permutation test \u2014 null AUROC distributions</text>'
        ),
    ]

    for panel_index, (model_name, observed, p_value, null_auroc) in enumerate(models):
        panel_x = margin_left + panel_index * (panel_w + gap)
        plot_top = margin_top
        plot_bottom = margin_top + plot_h
        plot_left = panel_x
        plot_right = panel_x + panel_w

        bins = _histogram_bins(null_auroc, n_bins)
        max_count = max(count for _, _, count in bins) or 1
        bin_px_w = panel_w / n_bins

        # Grid lines (y-axis: 0, 25, 50, 75, 100 % of max)
        for tick_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            count_val = tick_frac * max_count
            y = plot_bottom - tick_frac * plot_h
            lines.append(
                f'<line x1="{plot_left}" y1="{y:.1f}" '
                f'x2="{plot_right}" y2="{y:.1f}" '
                f'stroke="{grid_color}" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{plot_left - 5}" y="{y + 4:.1f}" '
                f'text-anchor="end" {label_font}>{int(count_val)}</text>'
            )

        # Histogram bars
        for lo, _hi, count in bins:
            bar_h = (count / max_count) * plot_h
            bar_x = plot_left + lo * panel_w
            bar_y = plot_bottom - bar_h
            lines.append(
                f'<rect x="{bar_x:.2f}" y="{bar_y:.2f}" '
                f'width="{bin_px_w - 1:.2f}" height="{bar_h:.2f}" '
                f'fill="{bar_color}" stroke="{bar_stroke}" stroke-width="0.5"/>'
            )

        # Observed AUROC vertical line
        obs_x = plot_left + observed * panel_w
        lines.append(
            f'<line x1="{obs_x:.1f}" y1="{plot_top}" '
            f'x2="{obs_x:.1f}" y2="{plot_bottom}" '
            f'stroke="{obs_color}" stroke-width="2" stroke-dasharray="6 3"/>'
        )
        # Label the observed line
        label_x = obs_x + 4 if obs_x < plot_left + panel_w * 0.75 else obs_x - 4
        anchor = "start" if obs_x < plot_left + panel_w * 0.75 else "end"
        lines.append(
            f'<text x="{label_x:.1f}" y="{plot_top + 14:.1f}" '
            f'text-anchor="{anchor}" font-family="sans-serif" font-size="10" '
            f'fill="{obs_color}">obs={observed:.3f}</text>'
        )
        lines.append(
            f'<text x="{label_x:.1f}" y="{plot_top + 27:.1f}" '
            f'text-anchor="{anchor}" font-family="sans-serif" font-size="10" '
            f'fill="{obs_color}">p={p_value:.4f}</text>'
        )

        # Axis borders
        lines.append(
            f'<line x1="{plot_left}" y1="{plot_top}" '
            f'x2="{plot_left}" y2="{plot_bottom}" '
            f'stroke="{axis_color}" stroke-width="1.5"/>'
        )
        lines.append(
            f'<line x1="{plot_left}" y1="{plot_bottom}" '
            f'x2="{plot_right}" y2="{plot_bottom}" '
            f'stroke="{axis_color}" stroke-width="1.5"/>'
        )

        # X-axis tick labels (0.0, 0.5, 1.0)
        for tick_val in (0.0, 0.25, 0.5, 0.75, 1.0):
            tick_x = plot_left + tick_val * panel_w
            lines.append(
                f'<text x="{tick_x:.1f}" y="{plot_bottom + 16:.1f}" '
                f'text-anchor="middle" {label_font}>{tick_val:.2f}</text>'
            )

        # X-axis label
        lines.append(
            f'<text x="{(plot_left + plot_right) / 2:.1f}" '
            f'y="{plot_bottom + 34:.1f}" '
            f'text-anchor="middle" {label_font} font-size="12">AUROC</text>'
        )

        # Y-axis label (rotated)
        axis_label_x = plot_left - 48
        axis_label_y = (plot_top + plot_bottom) // 2
        lines.append(
            f'<text x="{axis_label_x}" y="{axis_label_y}" '
            f'text-anchor="middle" {label_font} font-size="12" '
            f'transform="rotate(-90 {axis_label_x} {axis_label_y})">Count</text>'
        )

        # Panel title
        model_label = html.escape(model_name.replace("_", " ").title())
        lines.append(
            f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{plot_top - 12:.1f}" '
            f'text-anchor="middle" {title_font}>{model_label}</text>'
        )

    # Legend
    legend_y = total_h - 22
    legend_x = margin_left
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y - 8}" width="14" height="10" '
        f'fill="{bar_color}" stroke="{bar_stroke}" stroke-width="0.5"/>'
    )
    lines.append(
        f'<text x="{legend_x + 19}" y="{legend_y}" {note_font}>Null AUROC (permuted labels)</text>'
    )
    lines.append(
        f'<line x1="{legend_x + 165}" y1="{legend_y - 4}" '
        f'x2="{legend_x + 185}" y2="{legend_y - 4}" '
        f'stroke="{obs_color}" stroke-width="2" stroke-dasharray="5 3"/>'
    )
    lines.append(
        f'<text x="{legend_x + 190}" y="{legend_y}" {note_font}>'
        f'Observed AUROC</text>'
    )
    lines.append(
        f'<text x="{total_w - margin_right}" y="{legend_y}" '
        f'text-anchor="end" {note_font}>'
        f'Non-diagnostic research output</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a two-panel null-distribution histogram SVG "
            "from a permutation test JSON."
        )
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Permutation test JSON path."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/daphnet_permutation_null.svg"),
    )
    parser.add_argument("--n-bins", type=int, default=20, help="Histogram bin count.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the permutation figure generation command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        output = make_permutation_figure(
            arguments.input, arguments.output, n_bins=arguments.n_bins
        )
    except PermFigureInputError as error:
        parser.error(str(error))
        return 1
    print(json.dumps({"figure": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
