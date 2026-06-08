#!/usr/bin/env python3
"""
Name: interface_component_analysis_visualization_pipeline
Input: analytical base table
Output: interface component analysis chart files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from visualization_pipeline_core import (
    DEFAULT_YEARS,
    COMPONENT_CODES,
    COMPONENT_NAMES,
    YEAR_COLORS,
    COMPONENT_COLORS,
    HEAT_CMAP,
    ChartDisplayTextPolicy,
    MatplotlibPublicationStyleService,
    TabularFileRepository,
    AnalyticalTableLoader,
    AnalyticalMetricService,
    OutputDirectoryService,
    component_display_labels,
    component_axis_labels,
)


@dataclass(frozen=True)
class InterfaceAnalysisExecutionContext:
    base_dir: Path
    output_dir: Path
    figure: str | None
    dpi: int


class InterfaceAnalysisPlotService:
    """Builds interface component analysis figures."""

    def __init__(self, display_policy: ChartDisplayTextPolicy, metric_service: AnalyticalMetricService) -> None:
        self._display_policy = display_policy
        self._metric_service = metric_service

    def plot_presence_bars(self, frame, output_dir: Path, dpi: int) -> None:
        presence = self._metric_service.pivot(frame, "p_I")
        width = 0.13
        x = np.arange(len(COMPONENT_CODES))
        figure, axis = plt.subplots(figsize=(14, 5.5))
        for index, year in enumerate(DEFAULT_YEARS):
            offset = (index - len(DEFAULT_YEARS) / 2 + 0.5) * width
            values = presence[year].values
            bars = axis.bar(x + offset, values, width, color=YEAR_COLORS[year], label=str(year), zorder=3)
            for bar, component in zip(bars, COMPONENT_CODES):
                if component in {"A", "P"}:
                    bar.set_edgecolor("#8C2D04")
                    bar.set_linewidth(1.4)
        axis.axhline(0.30, color="#555555", linewidth=0.9, linestyle=":", zorder=2)
        axis.set_xticks(x)
        axis.set_xticklabels(component_display_labels(), fontsize=9)
        axis.set_ylabel("p_I")
        axis.set_ylim(0, 1.0)
        axis.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        axis.legend(loc="upper left", ncol=4, title="Ano", fontsize=8)
        axis.set_title(f"Frequência de uso dos componentes MINDSPACE por ano\n{self._display_policy.commerce_event_label()} | TI", fontweight="bold")
        axis.text(0.01, -0.15, "p_I = proporção de evidências de interface com componente presente", transform=axis.transAxes, fontsize=7.5, color="gray", style="italic", va="top")
        output_file = output_dir / "fig4_2_1_pi_barras.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")

    def plot_density_heatmap(self, frame, output_dir: Path, dpi: int) -> None:
        density = self._metric_service.pivot(frame, "density_media")
        presence = self._metric_service.pivot(frame, "p_I")
        figure, axis = plt.subplots(figsize=(10, 6))
        vmax = max(float(np.nanmax(density.values)), 0.01)
        image = axis.imshow(density.values, cmap=HEAT_CMAP, aspect="auto", vmin=0, vmax=vmax)
        for row_index, component in enumerate(COMPONENT_CODES):
            for column_index, year in enumerate(DEFAULT_YEARS):
                value = density.loc[component, year]
                if presence.loc[component, year] == 0:
                    axis.text(column_index, row_index, "—", ha="center", va="center", fontsize=11, color="#AAAAAA", fontweight="bold")
                else:
                    color = "white" if value > vmax * 0.55 else "#333333"
                    axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=9, color=color, fontweight="bold")
        axis.set_xticks(range(len(DEFAULT_YEARS)))
        axis.set_xticklabels([str(year) for year in DEFAULT_YEARS])
        axis.set_yticks(range(len(COMPONENT_CODES)))
        axis.set_yticklabels(component_axis_labels())
        axis.set_xlabel("Ano")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
        colorbar.set_label("density_media")
        axis.set_title(f"Intensidade de uso dos componentes MINDSPACE\n{self._display_policy.commerce_event_label()} | TI", fontweight="bold")
        output_file = output_dir / "fig4_2_2_density_heatmap.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")

    def plot_average_density(self, frame, output_dir: Path, dpi: int) -> None:
        density = self._metric_service.pivot(frame, "density_media")
        presence = self._metric_service.pivot(frame, "p_I")
        averages = {}
        for component in COMPONENT_CODES:
            valid_years = [year for year in DEFAULT_YEARS if presence.loc[component, year] > 0]
            averages[component] = float(density.loc[component, valid_years].mean()) if valid_years else 0.0
        ordered_components = sorted(COMPONENT_CODES, key=lambda code: averages[code], reverse=True)
        values = [averages[component] for component in ordered_components]
        figure, axis = plt.subplots(figsize=(9, 5.5))
        y = np.arange(len(ordered_components))
        axis.barh(y, values, color=[COMPONENT_COLORS[component] for component in ordered_components], edgecolor="#888888", zorder=3)
        for y_value, value in zip(y, values):
            if value > 0:
                axis.text(value + 0.04, y_value, f"{value:.2f}", va="center", fontsize=9, fontweight="bold")
        axis.set_yticks(y)
        axis.set_yticklabels([f"{COMPONENT_NAMES[component]} ({component})" for component in ordered_components])
        axis.invert_yaxis()
        axis.set_xlabel("density_media média")
        axis.axvline(1.0, color="#AAAAAA", linewidth=0.8, linestyle=":", zorder=2)
        axis.set_title(f"Intensidade média de uso por componente MINDSPACE\n{self._display_policy.commerce_event_label()} | TI", fontweight="bold")
        output_file = output_dir / "fig4_2_3_density_media_barras.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")


class InterfaceAnalysisVisualizationOrchestrator:
    """Coordinates interface analysis visualization generation."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._metric_service = AnalyticalMetricService()
        self._style_service = MatplotlibPublicationStyleService()
        self._output_service = OutputDirectoryService()
        self._plot_service = InterfaceAnalysisPlotService(ChartDisplayTextPolicy(), self._metric_service)

    def execute(self, context: InterfaceAnalysisExecutionContext) -> None:
        self._output_service.ensure(context.output_dir)
        self._style_service.apply()
        frame = self._loader.load_abt(context.base_dir)
        selected = [context.figure] if context.figure else ["1", "2", "3"]
        if "1" in selected:
            self._plot_service.plot_presence_bars(frame, context.output_dir, context.dpi)
        if "2" in selected:
            self._plot_service.plot_density_heatmap(frame, context.output_dir, context.dpi)
        if "3" in selected:
            self._plot_service.plot_average_density(frame, context.output_dir, context.dpi)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interface component analysis visualizations.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--figure", choices=["1", "2", "3"], default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-2"
    InterfaceAnalysisVisualizationOrchestrator().execute(InterfaceAnalysisExecutionContext(base_dir, output_dir, args.figure, args.dpi))


if __name__ == "__main__":
    main()
