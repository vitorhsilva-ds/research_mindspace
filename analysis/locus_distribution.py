#!/usr/bin/env python3
"""
Name: interface_locus_distribution_visualization_pipeline
Input: interface table
Output: locus distribution chart files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from visualization_pipeline_core import (
    COMPONENT_CODES,
    COMPONENT_NAMES,
    COMPONENT_COLORS,
    BLUE_CMAP,
    ChartDisplayTextPolicy,
    MatplotlibPublicationStyleService,
    TabularFileRepository,
    AnalyticalTableLoader,
    OutputDirectoryService,
)

LOCUS_LABELS = {
    "pagina_principal": "Página principal",
    "topo_campanha": "Banner / hero\n(campanha)",
    "card_produto": "Card de produto",
    "corpo_campanha": "Corpo da\ncampanha",
    "grade_campanha": "Grade de produtos\n(campanha)",
    "rodape_campanha": "Rodapé\n(campanha)",
    "topo_principal": "Banner / hero\n(principal)",
    "corpo_principal": "Corpo da\npágina principal",
    "grade_principal": "Grade de produtos\n(principal)",
    "topo_produto": "Topo\n(produto)",
    "corpo_produto": "Corpo\n(produto)",
    "grade_produto": "Grade / specs\n(produto)",
    "campanha": "Página de campanha",
    "principal": "Página principal",
    "produto": "Página de produto",
}
LOCUS_ORDER = (
    "topo_campanha", "pagina_principal", "card_produto",
    "corpo_campanha", "grade_campanha", "rodape_campanha",
    "topo_principal", "corpo_principal", "grade_principal",
    "topo_produto", "corpo_produto", "grade_produto",
    "campanha", "principal", "produto",
)


@dataclass(frozen=True)
class LocusDistributionExecutionContext:
    base_dir: Path
    output_dir: Path
    field: str | None
    dpi: int


class LocusFieldResolutionService:
    """Resolves the locus field to use in the interface table."""

    def resolve(self, frame: pd.DataFrame, forced_field: str | None) -> str:
        if forced_field:
            return forced_field
        if "locus" in frame.columns and frame["locus"].dropna().nunique() >= 3:
            return "locus"
        return "tipo_pagina"


class LocusDistributionComputationService:
    """Computes component distribution by locus."""

    def compute(self, frame: pd.DataFrame, field: str) -> pd.DataFrame:
        component_columns = [component for component in COMPONENT_CODES if component in frame.columns]
        ordered_loci = [locus for locus in LOCUS_ORDER if locus in set(frame[field].dropna())]
        extra_loci = [locus for locus in frame[field].dropna().unique() if locus not in ordered_loci]
        loci = ordered_loci + sorted(extra_loci)
        rows = []
        for locus in loci:
            subset = frame[frame[field] == locus]
            count = len(subset)
            row = {"locus": locus, "n_slices": count}
            for component in component_columns:
                row[component] = subset[component].sum() / count if count else 0.0
            rows.append(row)
        return pd.DataFrame(rows).set_index("locus")


class LocusDistributionPlotService:
    """Builds locus distribution visualizations."""

    def __init__(self, display_policy: ChartDisplayTextPolicy) -> None:
        self._display_policy = display_policy

    def plot_grouped_bars(self, distribution: pd.DataFrame, field: str, output_dir: Path, dpi: int) -> None:
        loci = list(distribution.index)
        components = [component for component in COMPONENT_CODES if component in distribution.columns]
        width = 0.08
        x = np.arange(len(loci))
        figure, axis = plt.subplots(figsize=(max(12, len(loci) * 1.5), 5.5))
        for index, component in enumerate(components):
            offset = (index - len(components) / 2 + 0.5) * width
            axis.bar(x + offset, distribution[component].values, width, color=COMPONENT_COLORS[component], label=f"{COMPONENT_NAMES[component]} ({component})", zorder=3, alpha=0.92)
        axis.set_xticks(x)
        axis.set_xticklabels([LOCUS_LABELS.get(locus, locus) for locus in loci], fontsize=8.5)
        axis.set_ylabel("P(componente | locus)")
        axis.set_ylim(0, 1.0)
        axis.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:.1f}"))
        axis.legend(loc="upper right", ncol=3, fontsize=8, title="Componente")
        for x_value, locus in zip(x, loci):
            axis.text(x_value, -0.07, f"n={int(distribution.loc[locus, 'n_slices'])}", ha="center", va="top", fontsize=7, color="gray", transform=axis.get_xaxis_transform())
        axis.set_title(f"Distribuição dos componentes MINDSPACE por locus\n{self._display_policy.commerce_event_label()} | campo: {field}", fontsize=11, fontweight="bold")
        output_file = output_dir / "fig4_locus_componentes.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")

    def plot_heatmap(self, distribution: pd.DataFrame, field: str, output_dir: Path, dpi: int) -> None:
        loci = list(distribution.index)
        components = [component for component in COMPONENT_CODES if component in distribution.columns]
        matrix = distribution[components].values
        figure, (axis, color_axis) = plt.subplots(1, 2, figsize=(11, max(4, len(loci) * 0.7 + 2)), gridspec_kw={"width_ratios": [1, 0.04], "wspace": 0.06})
        image = axis.imshow(matrix, cmap=BLUE_CMAP, vmin=0, vmax=1, aspect="auto")
        for row_index, locus in enumerate(loci):
            for column_index, component in enumerate(components):
                value = matrix[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.2f}" if value else "—", ha="center", va="center", fontsize=8, color="white" if value > 0.55 else "#222222")
        axis.set_xticks(range(len(components)))
        axis.set_xticklabels([f"{COMPONENT_NAMES[component]}\n({component})" for component in components], fontsize=8)
        axis.set_yticks(range(len(loci)))
        axis.set_yticklabels([f"{LOCUS_LABELS.get(locus, locus)} (n={int(distribution.loc[locus, 'n_slices'])})" for locus in loci], fontsize=8.5)
        axis.set_title(f"Heatmap — Componentes MINDSPACE por locus\n{self._display_policy.commerce_event_label()} | campo: {field}", fontsize=11, fontweight="bold")
        figure.colorbar(image, cax=color_axis).set_label("P(comp | locus)")
        output_file = output_dir / "fig4_locus_heatmap.png"
        figure.savefig(output_file, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Output saved: {output_file}")


class LocusDistributionVisualizationOrchestrator:
    """Coordinates locus distribution visualization generation."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._style_service = MatplotlibPublicationStyleService()
        self._output_service = OutputDirectoryService()
        self._field_service = LocusFieldResolutionService()
        self._computation_service = LocusDistributionComputationService()
        self._plot_service = LocusDistributionPlotService(ChartDisplayTextPolicy())

    def execute(self, context: LocusDistributionExecutionContext) -> None:
        self._output_service.ensure(context.output_dir)
        self._style_service.apply()
        interface_frame = self._loader.load_interface_table(context.base_dir, required=True)
        field = self._field_service.resolve(interface_frame, context.field)
        print(f"Locus field selected: {field}")
        distribution = self._computation_service.compute(interface_frame, field)
        self._plot_service.plot_grouped_bars(distribution, field, context.output_dir, context.dpi)
        self._plot_service.plot_heatmap(distribution, field, context.output_dir, context.dpi)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate interface locus distribution visualizations.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--field", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-24"
    LocusDistributionVisualizationOrchestrator().execute(LocusDistributionExecutionContext(base_dir, output_dir, args.field, args.dpi))


if __name__ == "__main__":
    main()
