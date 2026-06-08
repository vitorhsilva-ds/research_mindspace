#!/usr/bin/env python3
"""
Name: complaint_component_analysis_visualization_pipeline
Input: analytical base table and complaint table
Output: complaint component analysis chart files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from visualization_pipeline_core import (
    DEFAULT_YEARS,
    COMPONENT_CODES,
    COMPONENT_NAMES,
    YEAR_COLORS,
    ChartDisplayTextPolicy,
    MatplotlibPublicationStyleService,
    TabularFileRepository,
    AnalyticalTableLoader,
    AnalyticalMetricService,
    OutputDirectoryService,
    component_display_labels,
)

CATEGORY_ORDER = ("Descontos", "Escassez", "Pressão de tempo", "Dificuldade de navegação", "Outros sinais", "Indeterminado")
CATEGORY_COLORS = {
    "Descontos": "#2171B5",
    "Escassez": "#6BAED6",
    "Pressão de tempo": "#F16913",
    "Dificuldade de navegação": "#74C476",
    "Outros sinais": "#BDBDBD",
    "Indeterminado": "#D9D9D9",
}
FALLBACK_PRIMARY_COUNTS = {
    2020: {"I": 93, "S": 52, "indeterminado": 65, "C": 3, "N": 2},
    2021: {"I": 53, "S": 14, "indeterminado": 23, "C": 5, "A": 1},
    2022: {"I": 44, "S": 10, "indeterminado": 30, "C": 3, "M": 3},
    2023: {"I": 96, "S": 37, "indeterminado": 77, "C": 6, "D": 1},
    2024: {"I": 104, "S": 24, "indeterminado": 78, "C": 8, "D": 2},
    2025: {"I": 63, "S": 18, "indeterminado": 44, "C": 5, "D": 2},
}
CATEGORY_CMAP = LinearSegmentedColormap.from_list("category_heat", ["#FFFFFF", "#C6DBEF", "#4292C6", "#084594"], N=256)


@dataclass(frozen=True)
class ComplaintAnalysisExecutionContext:
    base_dir: Path
    output_dir: Path
    figure: str | None
    dpi: int


class CategoryMappingService:
    """Maps primary component values into analytical categories."""

    def map_primary(self, primary_value) -> str:
        if primary_value in (None, "") or pd.isna(primary_value):
            return "Indeterminado"
        normalized_value = str(primary_value).strip().upper()
        if normalized_value == "INDETERMINADO":
            return "Indeterminado"
        if normalized_value == "I":
            return "Descontos"
        if normalized_value == "S":
            return "Escassez"
        if normalized_value == "A":
            return "Pressão de tempo"
        if normalized_value in ("D", "C"):
            return "Dificuldade de navegação"
        return "Outros sinais"

    def calculate_category_by_year(self, complaint_frame) -> pd.DataFrame:
        if complaint_frame is not None and "primary" in complaint_frame.columns:
            frame = complaint_frame.copy()
            frame["category"] = frame["primary"].apply(self.map_primary)
            return frame.groupby(["ano", "category"]).size().unstack(fill_value=0).reindex(index=DEFAULT_YEARS, columns=CATEGORY_ORDER, fill_value=0)
        rows = {}
        for year, distribution in FALLBACK_PRIMARY_COUNTS.items():
            row = {category: 0 for category in CATEGORY_ORDER}
            for primary, count in distribution.items():
                row[self.map_primary(primary)] += count
            rows[year] = row
        return pd.DataFrame(rows).T.reindex(index=DEFAULT_YEARS, columns=CATEGORY_ORDER, fill_value=0)

    def calculate_category_component_matrix(self, complaint_frame) -> pd.DataFrame:
        if complaint_frame is not None and "primary" in complaint_frame.columns:
            columns = [component for component in COMPONENT_CODES if component in complaint_frame.columns]
            frame = complaint_frame.copy()
            frame["category"] = frame["primary"].apply(self.map_primary)
            rows = []
            for category in CATEGORY_ORDER:
                subset = frame[frame["category"] == category]
                count = len(subset)
                row = {"category": category, "n_count": count}
                for component in COMPONENT_CODES:
                    row[component] = subset[component].sum() / count if count and component in columns else 0.0
                rows.append(row)
            return pd.DataFrame(rows).set_index("category")
        fallback = {
            "Descontos": {"I": 1.00, "S": 0.30, "A": 0.00, "N": 0.10},
            "Escassez": {"I": 0.60, "S": 1.00, "A": 0.20, "N": 0.10},
            "Pressão de tempo": {"I": 0.30, "S": 0.70, "A": 1.00, "N": 0.00},
            "Dificuldade de navegação": {"I": 0.40, "S": 0.20, "D": 0.80, "C": 1.00},
            "Outros sinais": {"M": 0.50, "N": 0.50, "P": 0.30, "E": 0.30},
            "Indeterminado": {},
        }
        rows = []
        for category in CATEGORY_ORDER:
            row = {component: fallback.get(category, {}).get(component, 0.0) for component in COMPONENT_CODES}
            row["category"] = category
            row["n_count"] = 0
            rows.append(row)
        return pd.DataFrame(rows).set_index("category")


class ComplaintAnalysisPlotService:
    """Builds complaint component analysis figures."""

    def __init__(self, display_policy: ChartDisplayTextPolicy, metric_service: AnalyticalMetricService) -> None:
        self._display_policy = display_policy
        self._metric_service = metric_service

    def plot_perception_bars(self, abt_frame, output_dir: Path, dpi: int) -> None:
        perception = self._metric_service.pivot(abt_frame, "p_R")
        width = 0.13
        x = np.arange(len(COMPONENT_CODES))
        figure, axis = plt.subplots(figsize=(14, 5.5))
        for index, year in enumerate(DEFAULT_YEARS):
            offset = (index - len(DEFAULT_YEARS) / 2 + 0.5) * width
            bars = axis.bar(x + offset, perception[year].values, width, color=YEAR_COLORS[year], label=str(year), zorder=3)
            for bar, component in zip(bars, COMPONENT_CODES):
                if component in {"I", "S"}:
                    bar.set_edgecolor("#084594")
                    bar.set_linewidth(1.4)
        axis.axhline(0.10, color="#555555", linewidth=0.9, linestyle=":", zorder=2)
        axis.set_xticks(x)
        axis.set_xticklabels(component_display_labels(), fontsize=9)
        axis.set_ylabel("p_R")
        axis.set_ylim(0, 0.80)
        axis.legend(loc="upper right", ncol=4, title="Ano", fontsize=8)
        axis.set_title(f"Frequência de percepção dos componentes MINDSPACE por ano\n{self._display_policy.complaint_source_label()} | TR", fontweight="bold")
        output_file = output_dir / "fig4_3_4_pr_barras.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")

    def plot_category_mapping(self, category_by_year, category_component_matrix, output_dir: Path, dpi: int, real_source: bool) -> None:
        figure = plt.figure(figsize=(16, 6))
        grid = GridSpec(1, 2, figure=figure, width_ratios=[1.1, 1], wspace=0.35)
        axis_a = figure.add_subplot(grid[0, 0])
        axis_b = figure.add_subplot(grid[0, 1])
        x = np.arange(len(DEFAULT_YEARS))
        bottom = np.zeros(len(DEFAULT_YEARS))
        for category in CATEGORY_ORDER:
            values = category_by_year[category].values.astype(float)
            axis_a.bar(x, values, 0.6, bottom=bottom, color=CATEGORY_COLORS[category], label=category, zorder=3)
            bottom += values
        for x_value, total in zip(x, category_by_year.sum(axis=1).values):
            axis_a.text(x_value, total + 1.5, str(int(total)), ha="center", va="bottom", fontsize=8, fontweight="bold")
        axis_a.set_xticks(x)
        axis_a.set_xticklabels([str(year) for year in DEFAULT_YEARS])
        axis_a.set_xlabel("Ano")
        axis_a.set_ylabel("N de reclamações")
        axis_a.legend(loc="upper left", fontsize=8, title="Categoria")
        axis_a.set_title("Painel A — Volume por categoria e ano", fontweight="bold")
        matrix = category_component_matrix.loc[CATEGORY_ORDER, list(COMPONENT_CODES)].values
        image = axis_b.imshow(matrix, cmap=CATEGORY_CMAP, vmin=0, vmax=1, aspect="auto")
        for row_index, category in enumerate(CATEGORY_ORDER):
            for column_index, component in enumerate(COMPONENT_CODES):
                value = matrix[row_index, column_index]
                axis_b.text(column_index, row_index, f"{value:.2f}" if value else "—", ha="center", va="center", fontsize=8, color="white" if value > 0.55 else "#222222")
        axis_b.set_xticks(range(len(COMPONENT_CODES)))
        axis_b.set_xticklabels(component_display_labels(), fontsize=8)
        axis_b.set_yticks(range(len(CATEGORY_ORDER)))
        axis_b.set_yticklabels([f"{category} (n={int(category_component_matrix.loc[category, 'n_count'])})" for category in CATEGORY_ORDER], fontsize=8)
        figure.colorbar(image, ax=axis_b, fraction=0.035, pad=0.02).set_label("P(comp | categoria)")
        axis_b.set_title("Painel B — Ativação de componentes por categoria", fontweight="bold")
        figure.suptitle(f"Mapeamento categoria → MINDSPACE\n{self._display_policy.complaint_source_label()} 2020–2025", fontweight="bold", y=1.02)
        output_file = output_dir / "fig4_3_5_category_mapping.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")


class ComplaintAnalysisVisualizationOrchestrator:
    """Coordinates complaint analysis visualization generation."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._metric_service = AnalyticalMetricService()
        self._category_service = CategoryMappingService()
        self._style_service = MatplotlibPublicationStyleService()
        self._output_service = OutputDirectoryService()
        self._plot_service = ComplaintAnalysisPlotService(ChartDisplayTextPolicy(), self._metric_service)

    def execute(self, context: ComplaintAnalysisExecutionContext) -> None:
        self._output_service.ensure(context.output_dir)
        self._style_service.apply()
        abt_frame = self._loader.load_abt(context.base_dir)
        complaint_frame = self._loader.load_complaint_table(context.base_dir, required=False)
        category_by_year = self._category_service.calculate_category_by_year(complaint_frame)
        category_component_matrix = self._category_service.calculate_category_component_matrix(complaint_frame)
        selected = [context.figure] if context.figure else ["4", "5"]
        if "4" in selected:
            self._plot_service.plot_perception_bars(abt_frame, context.output_dir, context.dpi)
        if "5" in selected:
            self._plot_service.plot_category_mapping(category_by_year, category_component_matrix, context.output_dir, context.dpi, complaint_frame is not None)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate complaint component analysis visualizations.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--figure", choices=["4", "5"], default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-3"
    ComplaintAnalysisVisualizationOrchestrator().execute(ComplaintAnalysisExecutionContext(base_dir, output_dir, args.figure, args.dpi))


if __name__ == "__main__":
    main()
