"""Render a per-subject AUROC dot-plot SVG from a LOSO benchmark JSON."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


class LOSOFigureInputError(ValueError):
    """Raised when the LOSO benchmark JSON cannot produce a valid figure."""


def make_loso_figure(
    benchmark_path: str | Path,
    output_path: str | Path,
    *,
    model_name: str = "random_forest",
) -> Path:
    """Create a per-subject AUROC dot-plot SVG with a mean line.

    Parameters
    ----------
    benchmark_path:
        Path to the LOSO benchmark JSON produced by ``run_baselines.py``.
    output_path:
        Destination for the rendered SVG file.
    model_name:
        Key inside ``models`` to plot (default: ``random_forest``).
    """
    benchmark = _load_benchmark(benchmark_path)
    subject_aurocs, mean_auroc = _extract_per_fold_aurocs(benchmark, model_name)
    svg = _render_svg(subject_aurocs, mean_auroc, model_name)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


# ── I/O helpers ──────────────────────────────────────────────────────────────


def _load_benchmark(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LOSOFigureInputError(f"Unable to read benchmark JSON: {error}") from error
    if not isinstance(value, dict):
        raise LOSOFigureInputError("Benchmark JSON must contain an object")
    return value


def _extract_per_fold_aurocs(
    benchmark: dict[str, Any],
    model_name: str,
) -> tuple[list[tuple[str, float]], float]:
    """Return a list of (subject_id, auroc) pairs and the mean AUROC."""
    raw_models = benchmark.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise LOSOFigureInputError("Benchmark JSON must contain non-empty model results")
    if model_name not in raw_models:
        available = sorted(raw_models)
        raise LOSOFigureInputError(f"Model {model_name!r} not found; available: {available}")

    model_data = raw_models[model_name]
    per_fold = model_data.get("per_fold")
    if not isinstance(per_fold, list) or not per_fold:
        raise LOSOFigureInputError(
            "Model results must contain a non-empty 'per_fold' list. "
            "Re-run run_baselines.py -- this JSON may be from an older run."
        )

    subject_aurocs: list[tuple[str, float]] = []
    for fold in per_fold:
        if "error" in fold:
            continue
        sid = str(fold.get("subject_id", f"fold_{fold.get('fold_index', '?')}"))
        raw_auroc = fold.get("auroc")
        if not isinstance(raw_auroc, int | float) or isinstance(raw_auroc, bool):
            raise LOSOFigureInputError(f"AUROC for fold {sid!r} must be numeric")
        auroc_value = float(raw_auroc)
        if not math.isfinite(auroc_value) or not 0.0 <= auroc_value <= 1.0:
            raise LOSOFigureInputError(f"AUROC for fold {sid!r} must be finite and between 0 and 1")
        subject_aurocs.append((sid, auroc_value))

    if not subject_aurocs:
        raise LOSOFigureInputError("No valid per-fold AUROC values found")

    mean_auroc = math.fsum(v for _, v in subject_aurocs) / len(subject_aurocs)
    return subject_aurocs, mean_auroc


# ── SVG renderer ─────────────────────────────────────────────────────────────


def _render_svg(
    subject_aurocs: list[tuple[str, float]],
    mean_auroc: float,
    model_name: str,
) -> str:
    n = len(subject_aurocs)
    width = max(600, n * 70 + 120)
    height = 460
    plot_left = 75
    plot_right = width - 40
    plot_top = 60
    plot_bottom = 360
    plot_height = plot_bottom - plot_top
    plot_width = plot_right - plot_left
    dot_r = 7
    dot_color = "#2563eb"
    mean_color = "#dc2626"
    grid_color = "#e5e7eb"
    axis_color = "#374151"
    label_font = "font-family='sans-serif' fill='#374151'"
    title_font = "font-family='sans-serif' font-size='17' font-weight='bold' fill='#111827'"
    note_font = "font-family='sans-serif' font-size='10' fill='#6b7280'"

    def x_pos(i: int) -> float:
        if n == 1:
            return plot_left + plot_width / 2
        return plot_left + plot_width * i / (n - 1)

    def y_pos(auroc: float) -> float:
        return plot_bottom - auroc * plot_height

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    model_label = html.escape(model_name.replace("_", " ").title())
    lines.append(
        f'<text x="{width // 2}" y="35" text-anchor="middle" {title_font}>'
        f"Per-subject AUROC \u2014 {model_label}</text>"
    )

    for tick_num in range(6):
        auroc_val = tick_num / 5
        y = y_pos(auroc_val)
        lines.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" '
            f'x2="{plot_right}" y2="{y:.1f}" '
            f'stroke="{grid_color}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" '
            f'text-anchor="end" {label_font}>{auroc_val:.1f}</text>'
        )

    lines.append(
        f'<text x="18" y="{(plot_top + plot_bottom) // 2}" '
        f'text-anchor="middle" transform="rotate(-90 18 {(plot_top + plot_bottom) // 2})" '
        f'{label_font} font-size="13">AUROC</text>'
    )

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

    mean_y = y_pos(mean_auroc)
    lines.append(
        f'<line x1="{plot_left}" y1="{mean_y:.1f}" '
        f'x2="{plot_right}" y2="{mean_y:.1f}" '
        f'stroke="{mean_color}" stroke-width="2" stroke-dasharray="6 3"/>'
    )
    lines.append(
        f'<text x="{plot_right + 5}" y="{mean_y + 4:.1f}" '
        f'font-family="sans-serif" font-size="11" fill="{mean_color}">'
        f"mean={mean_auroc:.3f}</text>"
    )

    for i, (sid, auroc_val) in enumerate(subject_aurocs):
        x = x_pos(i)
        y = y_pos(auroc_val)
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{dot_r}" '
            f'fill="{dot_color}" stroke="#1e40af" stroke-width="1.2"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{y - dot_r - 5:.1f}" '
            f'text-anchor="middle" {label_font} font-size="10">'
            f"{auroc_val:.3f}</text>"
        )
        lines.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 18:.1f}" '
            f'text-anchor="middle" {label_font}>'
            f"{html.escape(sid)}</text>"
        )

    legend_y = plot_bottom + 45
    lines.append(f'<circle cx="{plot_left}" cy="{legend_y}" r="6" fill="{dot_color}"/>')
    lines.append(
        f'<text x="{plot_left + 14}" y="{legend_y + 4}" {label_font}>Per-subject AUROC</text>'
    )
    lines.append(
        f'<line x1="{plot_left + 140}" y1="{legend_y}" '
        f'x2="{plot_left + 165}" y2="{legend_y}" '
        f'stroke="{mean_color}" stroke-width="2" stroke-dasharray="6 3"/>'
    )
    lines.append(
        f'<text x="{plot_left + 170}" y="{legend_y + 4}" {label_font}>Mean across subjects</text>'
    )

    lines.append(
        f'<text x="{width // 2}" y="{height - 10}" '
        f'text-anchor="middle" {note_font}>'
        f"Non-diagnostic research output \u2014 N={n} subjects</text>"
    )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a per-subject AUROC dot-plot SVG from a LOSO benchmark JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="LOSO benchmark JSON path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/daphnet_loso_benchmark.svg"),
    )
    parser.add_argument(
        "--model",
        default="random_forest",
        help="Model key to plot (default: random_forest).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the LOSO figure generation command."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        output = make_loso_figure(arguments.input, arguments.output, model_name=arguments.model)
    except LOSOFigureInputError as error:
        parser.error(str(error))
        return 1
    print(json.dumps({"figure": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
