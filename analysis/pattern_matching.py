#!/usr/bin/env python3
"""
Name: pattern_matching_visualization_pipeline
Input: analytical base table
Output: pattern matching chart files
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
from matplotlib.colors import TwoSlopeNorm

from visualization_pipeline_core import (
    DEFAULT_YEARS,
    COMPONENT_CODES,
    COMPONENT_NAMES,
    DIVERGENCE_CMAP,
    ChartDisplayTextPolicy,
    MatplotlibPublicationStyleService,
    TabularFileRepository,
    AnalyticalTableLoader,
    AnalyticalMetricService,
    OutputDirectoryService,
    component_axis_labels,
)

PATTERN_BY_COMPONENT = {
    "I": "direct_convergence",
    "S": "dynamic_convergence",
    "A": "usage_without_verbalization",
    "P": "usage_without_verbalization",
    "N": "usage_without_verbalization",
    "C": "inverse_convergence",
    "M": "residual_pattern",
    "D": "residual_pattern",
    "E": "residual_pattern",
}
PATTERN_COLORS = {
    "direct_convergence": "#2171B5",
    "dynamic_convergence": "#41AB5D",
    "usage_without_verbalization": "#F16913",
    "inverse_convergence": "#9E3B7A",
    "residual_pattern": "#969696",
}
PATTERN_LABELS = {
    "direct_convergence": "Convergência direta",
    "dynamic_convergence": "Convergência dinâmica",
    "usage_without_verbalization": "Uso sem verbalização",
    "inverse_convergence": "Convergência inversa",
    "residual_pattern": "Pontual / residual",
}
HIGH_POSITIVE_GAP = 0.30
HIGH_NEGATIVE_GAP = -0.15


@dataclass(frozen=True)
class PatternMatchingExecutionContext:
    base_dir: Path
    output_dir: Path
    figure: str | None
    dpi: int


class PatternMatchingPlotService:
    """Builds pattern matching visualizations."""

    def __init__(self, display_policy: ChartDisplayTextPolicy, metric_service: AnalyticalMetricService) -> None:
        self._display_policy = display_policy
        self._metric_service = metric_service

    def plot_comparison_grid(self, presence, perception, gap, output_dir: Path, dpi: int) -> None:
        figure, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
        figure.suptitle(f"Comparação p_I × p_R por componente MINDSPACE\n{self._display_policy.combined_context_label()}", fontsize=12, fontweight="bold", y=1.01)
        x = np.arange(len(DEFAULT_YEARS))
        for index, component in enumerate(COMPONENT_CODES):
            axis = axes[index // 3][index % 3]
            presence_values = presence.loc[component].values
            perception_values = perception.loc[component].values
            gap_values = gap.loc[component].values
            axis.fill_between(x, presence_values, perception_values, where=(presence_values >= perception_values), alpha=0.18, color="#2171B5")
            axis.fill_between(x, presence_values, perception_values, where=(perception_values > presence_values), alpha=0.25, color="#F16913")
            axis.plot(x, presence_values, "o-", color="#2171B5", linewidth=1.8, markersize=5, label="p_I")
            axis.plot(x, perception_values, "s--", color="#D94801", linewidth=1.6, markersize=4, label="p_R")
            for cursor, value in enumerate(gap_values):
                if value > HIGH_POSITIVE_GAP:
                    axis.annotate("▲", (x[cursor], max(presence_values[cursor], perception_values[cursor]) + 0.03), ha="center", fontsize=8, color="#8C2D04", fontweight="bold")
                elif value < HIGH_NEGATIVE_GAP:
                    axis.annotate("▼", (x[cursor], max(presence_values[cursor], perception_values[cursor]) + 0.03), ha="center", fontsize=8, color="#084594", fontweight="bold")
            pattern = PATTERN_BY_COMPONENT[component]
            axis.set_title(f"{COMPONENT_NAMES[component]} ({component})", fontsize=10, fontweight="bold", color=PATTERN_COLORS[pattern])
            axis.set_xticks(x)
            if index // 3 == 2:
                axis.set_xticklabels([str(year) for year in DEFAULT_YEARS], fontsize=8)
            if index % 3 == 0:
                axis.set_ylabel("Proporção", fontsize=8)
            axis.set_ylim(0, 1.0)
        handles = [mlines.Line2D([0], [0], color="#2171B5", marker="o", label="p_I"), mlines.Line2D([0], [0], color="#D94801", linestyle="--", marker="s", label="p_R")]
        figure.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
        output_file = output_dir / "fig4_4_7_pi_pr_comparacao.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")

    def plot_gap_heatmap(self, gap, output_dir: Path, dpi: int) -> None:
        matrix = gap.values
        value_abs = max(abs(matrix[~np.isnan(matrix)]).max(), 0.01)
        norm = TwoSlopeNorm(vmin=-value_abs, vcenter=0, vmax=value_abs)
        figure, axis = plt.subplots(figsize=(11, 6))
        image = axis.imshow(matrix, cmap=DIVERGENCE_CMAP, norm=norm, aspect="auto")
        for row_index, component in enumerate(COMPONENT_CODES):
            for column_index, year in enumerate(DEFAULT_YEARS):
                value = matrix[row_index, column_index]
                color = "white" if abs(value) > value_abs * 0.55 else "#222222"
                sign = "+" if value > 0 else ""
                axis.text(column_index, row_index, f"{sign}{value:.3f}", ha="center", va="center", fontsize=8, color=color, fontweight="bold")
        axis.set_xticks(range(len(DEFAULT_YEARS)))
        axis.set_xticklabels([str(year) for year in DEFAULT_YEARS])
        axis.set_yticks(range(len(COMPONENT_CODES)))
        axis.set_yticklabels(component_axis_labels())
        axis.set_xlabel("Ano")
        figure.colorbar(image, ax=axis, fraction=0.035, pad=0.02).set_label("gap_IR = p_I − p_R")
        axis.set_title(f"Mapa de divergência interface–percepção\n{self._display_policy.combined_context_label()}", fontweight="bold")
        output_file = output_dir / "fig4_4_8_gap_heatmap.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")

    def plot_pattern_map(self, presence, perception, gap, output_dir: Path, dpi: int) -> None:
        figure, axes = plt.subplots(3, 3, figsize=(12, 8))
        figure.suptitle("Mapa de padrões de convergência MINDSPACE", fontsize=12, fontweight="bold", y=1.01)
        for index, component in enumerate(COMPONENT_CODES):
            axis = axes[index // 3][index % 3]
            pattern = PATTERN_BY_COMPONENT[component]
            color = PATTERN_COLORS[pattern]
            axis.set_facecolor(color + "22")
            for spine in axis.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.text(0.5, 0.86, f"{COMPONENT_NAMES[component]} ({component})", transform=axis.transAxes, ha="center", va="top", fontsize=12, fontweight="bold", color=color)
            axis.text(0.5, 0.66, PATTERN_LABELS[pattern], transform=axis.transAxes, ha="center", va="top", fontsize=9, style="italic")
            axis.text(0.5, 0.43, f"gap̄ = {gap.loc[component].mean():+.3f}", transform=axis.transAxes, ha="center", va="top", fontsize=11, fontweight="bold", color=color)
            axis.text(0.5, 0.26, f"p_Ī = {presence.loc[component].mean():.3f}   p_R̄ = {perception.loc[component].mean():.3f}", transform=axis.transAxes, ha="center", va="top", fontsize=8)
        handles = [mpatches.Patch(facecolor=PATTERN_COLORS[key] + "44", edgecolor=PATTERN_COLORS[key], label=label) for key, label in PATTERN_LABELS.items()]
        figure.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8)
        output_file = output_dir / "fig4_4_9_padroes_convergencia.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")


class PatternMatchingVisualizationOrchestrator:
    """Coordinates pattern matching visualization generation."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._metric_service = AnalyticalMetricService()
        self._style_service = MatplotlibPublicationStyleService()
        self._output_service = OutputDirectoryService()
        self._plot_service = PatternMatchingPlotService(ChartDisplayTextPolicy(), self._metric_service)

    def execute(self, context: PatternMatchingExecutionContext) -> None:
        self._output_service.ensure(context.output_dir)
        self._style_service.apply()
        frame = self._loader.load_abt(context.base_dir)
        presence = self._metric_service.pivot(frame, "p_I")
        perception = self._metric_service.pivot(frame, "p_R")
        gap = self._metric_service.pivot(frame, "gap_IR")
        selected = [context.figure] if context.figure else ["7", "8", "9"]
        if "7" in selected:
            self._plot_service.plot_comparison_grid(presence, perception, gap, context.output_dir, context.dpi)
        if "8" in selected:
            self._plot_service.plot_gap_heatmap(gap, context.output_dir, context.dpi)
        if "9" in selected:
            self._plot_service.plot_pattern_map(presence, perception, gap, context.output_dir, context.dpi)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pattern matching visualizations.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--figure", choices=["7", "8", "9"], default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-4"
    PatternMatchingVisualizationOrchestrator().execute(PatternMatchingExecutionContext(base_dir, output_dir, args.figure, args.dpi))


if __name__ == "__main__":
    main()
