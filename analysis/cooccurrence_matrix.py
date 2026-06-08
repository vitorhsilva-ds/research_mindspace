#!/usr/bin/env python3
"""
Name: cooccurrence_matrix_visualization_pipeline
Input: interface table and complaint table
Output: cooccurrence matrix chart files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from visualization_pipeline_core import (
    COMPONENT_CODES,
    COMPONENT_NAMES,
    BLUE_CMAP,
    ORANGE_CMAP,
    ChartDisplayTextPolicy,
    MatplotlibPublicationStyleService,
    TabularFileRepository,
    AnalyticalTableLoader,
    AnalyticalMetricService,
    OutputDirectoryService,
    component_display_labels,
    component_axis_labels,
)

THEORETICAL_PAIRS = {("S", "A"), ("S", "I"), ("I", "A"), ("P", "A"), ("P", "S"), ("N", "S"), ("N", "M"), ("D", "C")}


@dataclass(frozen=True)
class CooccurrenceExecutionContext:
    base_dir: Path
    output_dir: Path
    year_filter: int | None
    dpi: int


class CooccurrencePlotService:
    """Builds co-occurrence matrix visualizations."""

    def __init__(self, display_policy: ChartDisplayTextPolicy) -> None:
        self._display_policy = display_policy

    def plot_matrix(self, axis, matrix: np.ndarray, sample_count: int, cmap, title: str, footer: str, max_value: float, show_y_labels: bool = True):
        image = axis.imshow(matrix, cmap=cmap, vmin=0, vmax=max_value, aspect="equal")
        for row_index, left_code in enumerate(COMPONENT_CODES):
            for column_index, right_code in enumerate(COMPONENT_CODES):
                value = matrix[row_index, column_index]
                color = "white" if value > max_value * 0.60 else "#222222"
                axis.text(column_index, row_index, f"{value:.2f}" if value else "0", ha="center", va="center", fontsize=7.5, color=color if value else "#CCCCCC", fontweight="bold" if row_index == column_index else "normal")
                pair = (left_code, right_code)
                if pair in THEORETICAL_PAIRS or (right_code, left_code) in THEORETICAL_PAIRS:
                    axis.add_patch(plt.Rectangle((column_index - 0.5, row_index - 0.5), 1, 1, fill=False, edgecolor="#333333", linewidth=1.3, zorder=5))
        axis.set_xticks(range(len(COMPONENT_CODES)))
        axis.set_xticklabels(component_display_labels(), fontsize=7.5)
        axis.set_yticks(range(len(COMPONENT_CODES)))
        axis.set_yticklabels(component_axis_labels() if show_y_labels else [], fontsize=8)
        axis.set_title(f"{title}\n(N = {sample_count:,})", fontsize=10, fontweight="bold")
        axis.text(0.5, -0.17, footer, transform=axis.transAxes, fontsize=7, color="gray", style="italic", ha="center", va="top")
        return image

    def render(self, interface_matrix: np.ndarray, interface_count: int, complaint_matrix: np.ndarray, complaint_count: int, output_dir: Path, dpi: int, year_filter: int | None) -> None:
        suffix = f" — {year_filter}" if year_filter else " — 2020–2025"
        off_interface = interface_matrix.copy(); np.fill_diagonal(off_interface, 0)
        off_complaint = complaint_matrix.copy(); np.fill_diagonal(off_complaint, 0)
        max_value = max(float(off_interface.max()), float(off_complaint.max()), 0.01)
        max_value = np.ceil(max_value / 0.05) * 0.05
        figure = plt.figure(figsize=(18, 8))
        figure.suptitle(f"Matrizes de co-ocorrência MINDSPACE{suffix}\n{self._display_policy.combined_context_label()}", fontsize=12, fontweight="bold", y=1.01)
        grid = GridSpec(1, 3, figure=figure, width_ratios=[1, 1, 0.06], wspace=0.32)
        axis_a = figure.add_subplot(grid[0, 0])
        axis_b = figure.add_subplot(grid[0, 1])
        axis_c = figure.add_subplot(grid[0, 2])
        image = self.plot_matrix(axis_a, interface_matrix, interface_count, BLUE_CMAP, "Painel A — Interface", "Diagonal = p_I | Off-diagonal = proporção conjunta", max_value, True)
        self.plot_matrix(axis_b, complaint_matrix, complaint_count, ORANGE_CMAP, f"Painel B — {self._display_policy.complaint_source_label()}", "Diagonal = p_R | Off-diagonal = proporção conjunta", max_value, False)
        colorbar = figure.colorbar(image, cax=axis_c)
        colorbar.set_label("Proporção conjunta")
        output_file = output_dir / "fig4_6_cooccurrence.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")
        for matrix, count, cmap, title, file_name in (
            (interface_matrix, interface_count, BLUE_CMAP, "Interface", "fig4_6A_cooccurrence_ti.png"),
            (complaint_matrix, complaint_count, ORANGE_CMAP, self._display_policy.complaint_source_label(), "fig4_6B_cooccurrence_tr.png"),
        ):
            figure_single, (axis_matrix, axis_colorbar) = plt.subplots(1, 2, figsize=(10, 7.5), gridspec_kw={"width_ratios": [1, 0.05], "wspace": 0.08})
            image = self.plot_matrix(axis_matrix, matrix, count, cmap, f"Co-ocorrência MINDSPACE — {title}{suffix}", "Diagonal = frequência individual | Off-diagonal = proporção conjunta", max_value, True)
            figure_single.colorbar(image, cax=axis_colorbar).set_label("Proporção")
            output_single = output_dir / file_name
            figure_single.savefig(output_single, dpi=dpi, bbox_inches="tight")
            plt.close(figure_single)
            print(f"Output saved: {output_single}")


class CooccurrenceVisualizationOrchestrator:
    """Coordinates co-occurrence matrix visualization generation."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._metric_service = AnalyticalMetricService()
        self._style_service = MatplotlibPublicationStyleService()
        self._output_service = OutputDirectoryService()
        self._plot_service = CooccurrencePlotService(ChartDisplayTextPolicy())

    def execute(self, context: CooccurrenceExecutionContext) -> None:
        self._output_service.ensure(context.output_dir)
        self._style_service.apply()
        interface_frame = self._loader.load_interface_table(context.base_dir, required=True)
        complaint_frame = self._loader.load_complaint_table(context.base_dir, required=True)
        interface_matrix, interface_count = self._metric_service.calculate_cooccurrence_matrix(interface_frame, context.year_filter)
        complaint_matrix, complaint_count = self._metric_service.calculate_cooccurrence_matrix(complaint_frame, context.year_filter)
        self._plot_service.render(interface_matrix, interface_count, complaint_matrix, complaint_count, context.output_dir, context.dpi, context.year_filter)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MINDSPACE co-occurrence matrix visualizations.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-23"
    CooccurrenceVisualizationOrchestrator().execute(CooccurrenceExecutionContext(base_dir, output_dir, args.year, args.dpi))


if __name__ == "__main__":
    main()
