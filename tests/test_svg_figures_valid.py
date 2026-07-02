"""Regression tests for committed SVG figures."""

from pathlib import Path
import xml.etree.ElementTree as ET


def test_committed_svg_figures_are_valid_xml() -> None:
    """All committed SVG figures should be parseable XML."""

    figure_dir = Path("docs/figures")
    svg_files = sorted(figure_dir.glob("*.svg"))

    assert svg_files, "Expected at least one committed SVG figure."

    for svg_file in svg_files:
        try:
            ET.parse(svg_file)
        except ET.ParseError as exc:
            raise AssertionError(f"Invalid SVG XML: {svg_file}") from exc
