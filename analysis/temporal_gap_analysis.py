#!/usr/bin/env python3
"""
Name: temporal_gap_analysis_pipeline
Input: analytical base table
Output: temporal gap chart and trajectory files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd

from visualization_pipeline_core import (
    DEFAULT_YEARS,
    COMPONENT_CODES,
    COMPONENT_NAMES,
    ChartDisplayTextPolicy,
    MatplotlibPublicationStyleService,
    TabularFileRepository,
    AnalyticalTableLoader,
    AnalyticalMetricService,
    OutputDirectoryService,
)

TRAJECTORY_RULES = {
    "M": ("gradual_convergence", "gap positivo reduzindo ao longo do período"),
    "I": ("irregular", "oscila em torno de zero"),
    "N": ("increasing_divergence", "gap positivo com crescimento moderado"),
    "D": ("stable_residual", "gap próximo de zero"),
    "S": ("increasing_distance", "inversão temporal seguida de aumento"),
    "P": ("increasing_divergence", "gap positivo ampliado"),
    "A": ("increasing_divergence", "gap positivo ampliado"),
    "C": ("stable_negative", "gap negativo estável"),
    "E": ("irregular", "pico pontual sem sustentação"),
}
TRAJECTORY_COLORS = {
    "gradual_convergence": "#2CA02C",
    "irregular": "#7F7F7F",
    "increasing_divergence": "#D94801",
    "stable_residual": "#9ECAE1",
    "increasing_distance": "#17BECF",
    "stable_negative": "#9467BD",
}
TRAJECTORY_LABELS = {
    "gradual_convergence": "Convergência gradual",
    "irregular": "Irregular",
    "increasing_divergence": "Divergência crescente",
    "stable_residual": "Estável / residual",
    "increasing_distance": "Afastamento crescente",
    "stable_negative": "Estável negativo",
}
HIGH_POSITIVE_GAP = 0.30
HIGH_NEGATIVE_GAP = -0.15


@dataclass(frozen=True)
class TemporalAnalysisExecutionContext:
    base_dir: Path
    output_dir: Path
    figure_only: bool
    dpi: int


class TrajectoryClassificationService:
    """Classifies component trajectories using configured rules."""

    def classify(self, gap_pivot: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for component in COMPONENT_CODES:
            series = gap_pivot.loc[component].values.astype(float)
            first_value = series[0]
            last_value = series[-1]
            variation = last_value - first_value
            trajectory, description = TRAJECTORY_RULES[component]
            rows.append({
                "comp": component,
                "name": COMPONENT_NAMES[component],
                "gap_2020": round(first_value, 3),
                "gap_2025": round(last_value, 3),
                "variation": round(variation, 3),
                "gap_mean": round(series.mean(), 3),
                "gap_std": round(series.std(), 3),
                "gap_min": round(series.min(), 3),
                "gap_max": round(series.max(), 3),
                "trajectory": trajectory,
                "description": description,
                "label": TRAJECTORY_LABELS[trajectory],
            })
        return pd.DataFrame(rows)


class TemporalAnalysisPlotService:
    """Builds temporal gap analysis visualizations."""

    def __init__(self, display_policy: ChartDisplayTextPolicy) -> None:
        self._display_policy = display_policy

    def plot_gap_series(self, gap_pivot: pd.DataFrame, trajectory_frame: pd.DataFrame, output_dir: Path, dpi: int) -> None:
        figure, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
        figure.suptitle(f"Séries temporais do gap_IR por componente MINDSPACE\n{self._display_policy.combined_context_label()}", fontsize=12, fontweight="bold", y=1.01)
        x = np.arange(len(DEFAULT_YEARS))
        for index, component in enumerate(COMPONENT_CODES):
            axis = axes[index // 3][index % 3]
            gap_values = gap_pivot.loc[component].values.astype(float)
            trajectory = TRAJECTORY_RULES[component][0]
            color = TRAJECTORY_COLORS[trajectory]
            axis.axhline(0, color="#888888", linewidth=0.9, zorder=2)
            axis.fill_between(x, gap_values, 0, where=(gap_values >= 0), alpha=0.15, color=color)
            axis.fill_between(x, gap_values, 0, where=(gap_values < 0), alpha=0.25, color="#4292C6")
            axis.plot(x, gap_values, "o-", color=color, linewidth=2.0, markersize=6, zorder=4)
            for cursor, value in enumerate(gap_values):
                if value > HIGH_POSITIVE_GAP:
                    axis.annotate("▲", xy=(x[cursor], value), xytext=(x[cursor], value + 0.04), ha="center", fontsize=8, color="#8C2D04", fontweight="bold")
                elif value < HIGH_NEGATIVE_GAP:
                    axis.annotate("▼", xy=(x[cursor], value), xytext=(x[cursor], value - 0.06), ha="center", fontsize=8, color="#084594", fontweight="bold")
            row = trajectory_frame[trajectory_frame["comp"] == component].iloc[0]
            axis.set_title(f"{COMPONENT_NAMES[component]} ({component})", fontsize=10, fontweight="bold", color=color)
            axis.text(0.03, 0.97, row["label"], transform=axis.transAxes, fontsize=7, ha="left", va="top", color=color, style="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.8, linewidth=0.8))
            axis.text(0.97, 0.05, f"Δ = {row['variation']:+.3f}", transform=axis.transAxes, fontsize=7.5, ha="right", va="bottom", color="#555555")
            axis.set_xticks(x)
            axis.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
            axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:+.1f}"))
            if index // 3 == 2:
                axis.set_xticklabels([str(year) for year in DEFAULT_YEARS], fontsize=8)
            if index % 3 == 0:
                axis.set_ylabel("gap_IR", fontsize=8)
        handles = [mpatches.Patch(facecolor=TRAJECTORY_COLORS[key], label=label) for key, label in TRAJECTORY_LABELS.items()]
        figure.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=8)
        output_file = output_dir / "fig4_6_10_gap_series.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")


class IrregularCaseReportService:
    """Builds a compact technical report for irregular trajectories."""

    def build(self, gap_pivot: pd.DataFrame) -> str:
        lines = ["IRREGULAR TRAJECTORY CASES", "=" * 60, ""]
        for component in COMPONENT_CODES:
            if TRAJECTORY_RULES[component][0] != "irregular":
                continue
            values = gap_pivot.loc[component].values.astype(float)
            lines.append(f"{COMPONENT_NAMES[component]} ({component})")
            lines.append(f"gap_mean={values.mean():+.3f} | gap_std={values.std():.3f} | range=[{values.min():+.3f}, {values.max():+.3f}]")
            for year, value in zip(DEFAULT_YEARS, values):
                lines.append(f"  {year}: gap_IR={value:+.3f}")
            lines.append("")
        return "\n".join(lines)


class TemporalAnalysisOrchestrator:
    """Coordinates temporal gap analysis outputs."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._metric_service = AnalyticalMetricService()
        self._style_service = MatplotlibPublicationStyleService()
        self._output_service = OutputDirectoryService()
        self._classification_service = TrajectoryClassificationService()
        self._plot_service = TemporalAnalysisPlotService(ChartDisplayTextPolicy())
        self._report_service = IrregularCaseReportService()

    def execute(self, context: TemporalAnalysisExecutionContext) -> None:
        self._output_service.ensure(context.output_dir)
        self._style_service.apply()
        frame = self._loader.load_abt(context.base_dir)
        gap_pivot = self._metric_service.pivot(frame, "gap_IR")
        trajectory_frame = self._classification_service.classify(gap_pivot)
        self._plot_service.plot_gap_series(gap_pivot, trajectory_frame, context.output_dir, context.dpi)
        if not context.figure_only:
            trajectory_frame.to_csv(context.output_dir / "trajectories.csv", index=False, encoding="utf-8")
            (context.output_dir / "irregular_cases.txt").write_text(self._report_service.build(gap_pivot), encoding="utf-8")
            print(f"Output saved: {context.output_dir / 'trajectories.csv'}")
            print(f"Output saved: {context.output_dir / 'irregular_cases.txt'}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate temporal gap analysis outputs.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--figure-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-6"
    TemporalAnalysisOrchestrator().execute(TemporalAnalysisExecutionContext(base_dir, output_dir, args.figure_only, args.dpi))


if __name__ == "__main__":
    main()
