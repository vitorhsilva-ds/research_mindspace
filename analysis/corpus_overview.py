#!/usr/bin/env python3
"""
Name: corpus_overview_visualization_pipeline
Input: analytical base table, interface table, complaint table, capture inventory
Output: corpus overview chart files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from visualization_pipeline_core import (
    DEFAULT_YEARS,
    FALLBACK_COUNTS,
    PAGE_TYPE_COLORS,
    PAGE_TYPE_LABELS,
    WINDOW_COLORS,
    WINDOW_LABELS,
    CLASSIFICATION_COLORS,
    ChartDisplayTextPolicy,
    MatplotlibPublicationStyleService,
    TabularFileRepository,
    AnalyticalTableLoader,
    AnalyticalMetricService,
    AxisAnnotationService,
    OutputDirectoryService,
)


@dataclass(frozen=True)
class CorpusOverviewInputContract:
    base_dir: Path
    capture_plan_path: Path | None
    capture_summary_path: Path | None


@dataclass(frozen=True)
class CorpusOverviewOutputContract:
    output_dir: Path
    dpi: int
    panel: str | None


@dataclass(frozen=True)
class CorpusOverviewExecutionContext:
    input_contract: CorpusOverviewInputContract
    output_contract: CorpusOverviewOutputContract


class CaptureCoverageExtractionService:
    """Extracts archive capture coverage by year and page type."""

    def extract(self, capture_plan_frame, capture_summary: dict | None) -> dict:
        result: dict[int, dict] = {}
        page_types = ("principal", "campanha", "produto")

        if capture_plan_frame is not None:
            year_column = next((column for column in capture_plan_frame.columns if column.lower() in ("ano", "year")), None)
            type_column = next((column for column in capture_plan_frame.columns if column.lower() in ("tipo_pagina", "page_type", "tipo", "type")), None)
            if year_column and type_column:
                capture_plan_frame[year_column] = capture_plan_frame[year_column].astype(int)
                capture_plan_frame = capture_plan_frame[capture_plan_frame[year_column].isin(DEFAULT_YEARS)]
                for year in DEFAULT_YEARS:
                    year_frame = capture_plan_frame[capture_plan_frame[year_column] == year]
                    result[year] = {
                        page_type: {
                            "planned": int((year_frame[type_column] == page_type).sum()),
                            "captured": 0,
                            "available": 0,
                        }
                        for page_type in page_types
                    }

        if not result:
            for year in DEFAULT_YEARS:
                result[year] = {
                    page_type: {"planned": 3, "captured": 0, "available": 0}
                    for page_type in page_types
                }

        if capture_summary:
            for year in DEFAULT_YEARS:
                entry = capture_summary.get(str(year), {})
                for page_type in page_types:
                    if page_type in entry and isinstance(entry[page_type], dict):
                        result[year][page_type]["captured"] = int(entry[page_type].get("slots", 0))
                        result[year][page_type]["available"] = int(entry[page_type].get("available", 0))

        return result


class InterfaceVolumeExtractionService:
    """Extracts interface volume by year, page type, and temporal window."""

    def extract(self, interface_frame, fallback_counts: dict[int, int]) -> dict:
        if interface_frame is not None:
            year_column = next((column for column in interface_frame.columns if column.lower() in ("ano", "year")), None)
            type_column = next((column for column in interface_frame.columns if column.lower() in ("tipo_pagina", "page_type", "tipo", "locus")), None)
            window_column = next((column for column in interface_frame.columns if column.lower() in ("janela", "window", "periodo")), None)
            if year_column:
                result = {}
                for year in DEFAULT_YEARS:
                    year_frame = interface_frame[interface_frame[year_column] == year]
                    result[year] = {"total": len(year_frame), "type": None, "window": None}
                    if type_column:
                        result[year]["type"] = {page_type: int((year_frame[type_column] == page_type).sum()) for page_type in PAGE_TYPE_LABELS}
                    if window_column:
                        result[year]["window"] = {window: int((year_frame[window_column] == window).sum()) for window in WINDOW_LABELS}
                return result

        return {year: {"total": fallback_counts.get(year, 0), "type": None, "window": None} for year in DEFAULT_YEARS}


class ComplaintVolumeExtractionService:
    """Extracts complaint volume by classification state."""

    def extract(self, complaint_frame, fallback_counts: dict[int, int]) -> dict:
        if complaint_frame is not None:
            year_column = next((column for column in complaint_frame.columns if column.lower() in ("ano", "year")), None)
            primary_column = next((column for column in complaint_frame.columns if column.lower() in ("primary", "primary_comp", "componente_primario", "primary_mindspace")), None)
            if year_column:
                result = {}
                for year in DEFAULT_YEARS:
                    year_frame = complaint_frame[complaint_frame[year_column] == year]
                    total = len(year_frame)
                    if primary_column:
                        indeterminate_count = int((year_frame[primary_column].astype(str).str.lower() == "indeterminado").sum())
                    else:
                        indeterminate_count = FALLBACK_COUNTS["indeterminate"].get(year, 0)
                    result[year] = {
                        "total": total,
                        "classified": total - indeterminate_count,
                        "indeterminate": indeterminate_count,
                    }
                return result

        return {
            year: {
                "total": fallback_counts.get(year, 0),
                "classified": fallback_counts.get(year, 0) - FALLBACK_COUNTS["indeterminate"].get(year, 0),
                "indeterminate": FALLBACK_COUNTS["indeterminate"].get(year, 0),
            }
            for year in DEFAULT_YEARS
        }


class CorpusOverviewPlotService:
    """Builds the corpus overview chart panels."""

    def __init__(self, display_policy: ChartDisplayTextPolicy, annotation_service: AxisAnnotationService) -> None:
        self._display_policy = display_policy
        self._annotation_service = annotation_service

    def plot_archive_panel(self, axis, coverage: dict) -> None:
        page_types = ("principal", "campanha", "produto")
        x = np.arange(len(DEFAULT_YEARS))
        width = 0.55
        has_captured_values = any(coverage[year][page_type].get("captured", 0) > 0 for year in DEFAULT_YEARS for page_type in page_types)
        bottom = np.zeros(len(DEFAULT_YEARS))
        for page_type in page_types:
            value_key = "captured" if has_captured_values else "planned"
            values = np.array([coverage[year][page_type].get(value_key, 0) for year in DEFAULT_YEARS])
            bars = axis.bar(x, values, width, bottom=bottom, color=PAGE_TYPE_COLORS[page_type], label=PAGE_TYPE_LABELS[page_type], zorder=3)
            for index, (bar, value) in enumerate(zip(bars, values)):
                if value >= 1:
                    axis.text(bar.get_x() + bar.get_width() / 2, bottom[index] + value / 2, str(int(value)), ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            bottom += values
        totals = [sum(coverage[year][page_type].get("captured" if has_captured_values else "planned", 0) for page_type in page_types) for year in DEFAULT_YEARS]
        for x_value, total in zip(x, totals):
            axis.text(x_value, total + 0.15, str(int(total)), ha="center", va="bottom", fontsize=9, fontweight="bold")
        self._annotation_service.configure_year_axis(axis)
        axis.set_xlabel("Ano")
        axis.set_ylabel("Capturas")
        axis.set_title("Painel A — Cobertura de capturas por ano e tipo de página", fontweight="bold")
        axis.legend(loc="upper right", fontsize=8)
        axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    def plot_interface_panel(self, axis, interface_data: dict) -> None:
        x = np.arange(len(DEFAULT_YEARS))
        width = 0.55
        has_page_type = all(interface_data[year]["type"] is not None for year in DEFAULT_YEARS)
        if has_page_type:
            bottom = np.zeros(len(DEFAULT_YEARS))
            for page_type in PAGE_TYPE_LABELS:
                values = np.array([interface_data[year]["type"].get(page_type, 0) for year in DEFAULT_YEARS])
                axis.bar(x, values, width, bottom=bottom, color=PAGE_TYPE_COLORS[page_type], label=PAGE_TYPE_LABELS[page_type], zorder=3)
                bottom += values
            totals = [interface_data[year]["total"] for year in DEFAULT_YEARS]
            for x_value, total in zip(x, totals):
                axis.text(x_value, total + 1, str(int(total)), ha="center", va="bottom", fontsize=8, fontweight="bold")
            axis.legend(loc="upper right", fontsize=8)
        else:
            totals = [interface_data[year]["total"] for year in DEFAULT_YEARS]
            bars = axis.bar(x, totals, width, color=PAGE_TYPE_COLORS["campanha"], zorder=3, label="Total")
            self._annotation_service.add_bar_labels(axis, bars, offset=1.5)
            axis.legend(loc="upper right", fontsize=8)
        self._annotation_service.configure_year_axis(axis)
        axis.set_xlabel("Ano")
        axis.set_ylabel("N de evidências de interface")
        axis.set_title("Painel B — Volume de evidências de interface por ano", fontweight="bold")
        axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    def plot_complaint_panel(self, axis, complaint_data: dict) -> None:
        x = np.arange(len(DEFAULT_YEARS))
        width = 0.55
        classified = np.array([complaint_data[year]["classified"] for year in DEFAULT_YEARS])
        indeterminate = np.array([complaint_data[year]["indeterminate"] for year in DEFAULT_YEARS])
        totals = classified + indeterminate
        axis.bar(x, classified, width, color=CLASSIFICATION_COLORS["classified"], label="Com sinal MINDSPACE", zorder=3)
        axis.bar(x, indeterminate, width, bottom=classified, color=CLASSIFICATION_COLORS["indeterminate"], label="Indeterminado", zorder=3)
        for x_value, total, indeterminate_value in zip(x, totals, indeterminate):
            percentage = indeterminate_value / total * 100 if total else 0
            axis.text(x_value, total + 1.5, f"{int(total)}\n({percentage:.0f}% ind.)", ha="center", va="bottom", fontsize=8, fontweight="bold")
        self._annotation_service.configure_year_axis(axis)
        axis.set_xlabel("Ano")
        axis.set_ylabel("N de reclamações elegíveis")
        axis.set_title(f"Painel C — Volume de reclamações {self._display_policy.complaint_source_label()} por ano", fontweight="bold")
        axis.legend(loc="upper right", fontsize=8)
        axis.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))


class CorpusOverviewVisualizationOrchestrator:
    """Coordinates corpus overview visualization generation."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._metric_service = AnalyticalMetricService()
        self._style_service = MatplotlibPublicationStyleService()
        self._output_service = OutputDirectoryService()
        self._coverage_service = CaptureCoverageExtractionService()
        self._interface_service = InterfaceVolumeExtractionService()
        self._complaint_service = ComplaintVolumeExtractionService()
        self._plot_service = CorpusOverviewPlotService(ChartDisplayTextPolicy(), AxisAnnotationService())

    def execute(self, context: CorpusOverviewExecutionContext) -> None:
        self._output_service.ensure(context.output_contract.output_dir)
        self._style_service.apply()
        print("Loading input data...")
        abt_frame = self._loader.load_abt(context.input_contract.base_dir)
        interface_frame = self._loader.load_interface_table(context.input_contract.base_dir, required=False)
        complaint_frame = self._loader.load_complaint_table(context.input_contract.base_dir, required=False)
        capture_plan = self._loader.load_capture_plan(context.input_contract.base_dir, context.input_contract.capture_plan_path)
        capture_summary = self._loader.load_capture_summary(context.input_contract.base_dir, context.input_contract.capture_summary_path)
        interface_counts, complaint_counts = self._metric_service.extract_total_counts(abt_frame)
        coverage = self._coverage_service.extract(capture_plan, capture_summary)
        interface_data = self._interface_service.extract(interface_frame, interface_counts)
        complaint_data = self._complaint_service.extract(complaint_frame, complaint_counts)
        self._render(context.output_contract, coverage, interface_data, complaint_data)

    def _render(self, output_contract: CorpusOverviewOutputContract, coverage: dict, interface_data: dict, complaint_data: dict) -> None:
        selected_panels = [output_contract.panel] if output_contract.panel else ["A", "B", "C"]
        panel_map = {"A": (self._plot_service.plot_archive_panel, coverage, "fig4_1A_archive_coverage.png"), "B": (self._plot_service.plot_interface_panel, interface_data, "fig4_1B_interface_volume.png"), "C": (self._plot_service.plot_complaint_panel, complaint_data, "fig4_1C_complaint_volume.png")}
        if set(selected_panels) == {"A", "B", "C"}:
            figure = plt.figure(figsize=(18, 5.5))
            figure.suptitle("Panorama descritivo do corpus\nCobertura de capturas · Volume de interface · Volume de reclamações", fontsize=12, fontweight="bold", y=1.02)
            grid = GridSpec(1, 3, figure=figure, wspace=0.38)
            axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
            for axis, panel in zip(axes, selected_panels):
                plotter, data, _ = panel_map[panel]
                plotter(axis, data)
            output_file = output_contract.output_dir / "fig4_1_corpus_overview.png"
            figure.savefig(output_file, dpi=output_contract.dpi, bbox_inches="tight")
            plt.close(figure)
            print(f"Output saved: {output_file}")
        for panel in selected_panels:
            plotter, data, file_name = panel_map[panel]
            figure, axis = plt.subplots(figsize=(6.5, 4.8))
            plotter(axis, data)
            output_file = output_contract.output_dir / file_name
            figure.savefig(output_file, dpi=output_contract.dpi, bbox_inches="tight")
            plt.close(figure)
            print(f"Output saved: {output_file}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corpus overview visualizations.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--capture-plan", type=Path, default=None)
    parser.add_argument("--capture-summary", type=Path, default=None)
    parser.add_argument("--panel", choices=["A", "B", "C"], default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def build_context(args: argparse.Namespace) -> CorpusOverviewExecutionContext:
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-1"
    return CorpusOverviewExecutionContext(
        input_contract=CorpusOverviewInputContract(base_dir=base_dir, capture_plan_path=args.capture_plan, capture_summary_path=args.capture_summary),
        output_contract=CorpusOverviewOutputContract(output_dir=output_dir, dpi=args.dpi, panel=args.panel),
    )


def main() -> None:
    args = parse_arguments()
    CorpusOverviewVisualizationOrchestrator().execute(build_context(args))


if __name__ == "__main__":
    main()
