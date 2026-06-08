#!/usr/bin/env python3
"""
Name: divergence_case_export_pipeline
Input: analytical base table, interface table, complaint table
Output: divergence case CSV and text files
Usage: run as a Python script
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from visualization_pipeline_core import (
    COMPONENT_NAMES,
    COMPONENT_CODES,
    TabularFileRepository,
    AnalyticalTableLoader,
    OutputDirectoryService,
)


@dataclass(frozen=True)
class DivergenceExportExecutionContext:
    base_dir: Path
    output_dir: Path
    positive_threshold: float
    negative_threshold: float
    component_filter: str | None
    year_filter: int | None
    max_evidence: int


class DivergenceCaseIdentificationService:
    """Identifies extreme divergence cases from the analytical base table."""

    def identify(self, frame: pd.DataFrame, positive_threshold: float, negative_threshold: float, component_filter: str | None, year_filter: int | None) -> pd.DataFrame:
        mask = (frame["gap_IR"] > positive_threshold) | (frame["gap_IR"] < negative_threshold)
        if component_filter:
            mask &= frame["comp"] == component_filter.upper()
        if year_filter:
            mask &= frame["ano"] == year_filter
        cases = frame[mask].copy()
        cases["gap_type"] = cases["gap_IR"].apply(lambda value: "positive" if value > positive_threshold else "negative")
        cases["component_name"] = cases["comp"].map(COMPONENT_NAMES)
        cases = cases.reindex(cases["gap_IR"].abs().sort_values(ascending=False).index)
        return cases[["comp", "component_name", "ano", "N_I", "n_I", "p_I", "N_R", "n_R", "p_R", "gap_IR", "gap_type"]]


class DivergenceEvidenceExtractionService:
    """Extracts interface and complaint evidence for selected divergence cases."""

    def extract_interface_evidence(self, interface_frame: pd.DataFrame | None, cases: pd.DataFrame, max_evidence: int) -> pd.DataFrame:
        if interface_frame is None:
            return pd.DataFrame()
        rows = []
        for _, case in cases.iterrows():
            component = case["comp"]
            year = case["ano"]
            evidence_column = f"{component}_ev"
            instance_column = f"{component}_inst"
            if component not in interface_frame.columns:
                continue
            subset = interface_frame[(interface_frame["ano"] == year) & (interface_frame[component] == 1)].copy()
            if instance_column in subset.columns:
                subset = subset.sort_values(instance_column, ascending=False)
            for _, row in subset.head(max_evidence).iterrows():
                rows.append({
                    "comp": component,
                    "component_name": COMPONENT_NAMES[component],
                    "ano": year,
                    "gap_IR": case["gap_IR"],
                    "gap_type": case["gap_type"],
                    "source_id": row.get("id", ""),
                    "page_type": row.get("tipo_pagina", ""),
                    "locus": row.get("locus", ""),
                    "window": row.get("janela", ""),
                    "instances": row.get(instance_column, 0),
                    "evidence": str(row.get(evidence_column, "") or "").strip(),
                    "filepath": row.get("filepath", ""),
                })
        return pd.DataFrame(rows)

    def extract_complaint_evidence(self, complaint_frame: pd.DataFrame | None, cases: pd.DataFrame, max_evidence: int) -> pd.DataFrame:
        if complaint_frame is None:
            return pd.DataFrame()
        rows = []
        for _, case in cases.iterrows():
            component = case["comp"]
            year = case["ano"]
            if component in complaint_frame.columns:
                subset = complaint_frame[(complaint_frame["ano"] == year) & (complaint_frame[component] == 1)].head(max_evidence)
            elif component == "C" and "primary" in complaint_frame.columns:
                subset = complaint_frame[(complaint_frame["ano"] == year) & (complaint_frame["primary"].astype(str).str.upper() == "C")].head(max_evidence)
            else:
                continue
            for _, row in subset.iterrows():
                rows.append({
                    "comp": component,
                    "component_name": COMPONENT_NAMES[component],
                    "ano": year,
                    "gap_IR": case["gap_IR"],
                    "gap_type": case["gap_type"],
                    "source_id": row.get("id", ""),
                    "window": row.get("janela", ""),
                    "complaint_date": row.get("complaint_date", ""),
                    "primary": row.get("primary", ""),
                    "confidence": row.get("confidence", ""),
                    "justification": str(row.get("justificativa", "") or "").strip(),
                    "filename": row.get("filename", ""),
                })
        return pd.DataFrame(rows)


class DivergenceReportBuilder:
    """Builds a technical text report without organizational references."""

    def build(self, cases: pd.DataFrame, interface_evidence: pd.DataFrame, complaint_evidence: pd.DataFrame, positive_threshold: float, negative_threshold: float) -> str:
        lines = [
            "=" * 72,
            "DIVERGENCE CASE EXPORT",
            "=" * 72,
            f"Positive threshold: gap_IR > +{positive_threshold}",
            f"Negative threshold: gap_IR < {negative_threshold}",
            f"Total cases: {len(cases)}",
            "",
        ]
        for _, case in cases.iterrows():
            component = case["comp"]
            year = case["ano"]
            lines.extend([
                f"{COMPONENT_NAMES[component]} ({component}) | {year}",
                f"p_I={case['p_I']:.3f} | p_R={case['p_R']:.3f} | gap_IR={case['gap_IR']:+.3f}",
                f"N_I={int(case['N_I'])} | n_I={int(case['n_I'])} | N_R={int(case['N_R'])} | n_R={int(case['n_R'])}",
                "Interface evidence:",
            ])
            interface_subset = interface_evidence[(interface_evidence["comp"] == component) & (interface_evidence["ano"] == year)] if not interface_evidence.empty else pd.DataFrame()
            if interface_subset.empty:
                lines.append("  none")
            else:
                for _, row in interface_subset.iterrows():
                    lines.append(f"  [{row.get('source_id', '')}] {row.get('locus', '')} | instances={row.get('instances', 0)}")
                    evidence = row.get("evidence", "")
                    if evidence:
                        lines.extend(f"    {part}" for part in textwrap.wrap(evidence, width=64))
            lines.append("Complaint evidence:")
            complaint_subset = complaint_evidence[(complaint_evidence["comp"] == component) & (complaint_evidence["ano"] == year)] if not complaint_evidence.empty else pd.DataFrame()
            if complaint_subset.empty:
                lines.append("  none")
            else:
                for _, row in complaint_subset.iterrows():
                    lines.append(f"  [{row.get('source_id', '')}] primary={row.get('primary', '')} | confidence={row.get('confidence', '')}")
                    justification = row.get("justification", "")
                    if justification:
                        lines.extend(f"    {part}" for part in textwrap.wrap(justification, width=64))
            lines.extend(["-" * 72, ""])
        return "\n".join(lines)


class DivergenceCaseExportOrchestrator:
    """Coordinates divergence case exports."""

    def __init__(self) -> None:
        repository = TabularFileRepository()
        self._loader = AnalyticalTableLoader(repository)
        self._output_service = OutputDirectoryService()
        self._case_service = DivergenceCaseIdentificationService()
        self._evidence_service = DivergenceEvidenceExtractionService()
        self._report_builder = DivergenceReportBuilder()

    def execute(self, context: DivergenceExportExecutionContext) -> None:
        self._output_service.ensure(context.output_dir)
        abt_frame = self._loader.load_abt(context.base_dir)
        interface_frame = self._loader.load_interface_table(context.base_dir, required=False)
        complaint_frame = self._loader.load_complaint_table(context.base_dir, required=False)
        cases = self._case_service.identify(abt_frame, context.positive_threshold, context.negative_threshold, context.component_filter, context.year_filter)
        interface_evidence = self._evidence_service.extract_interface_evidence(interface_frame, cases, context.max_evidence)
        complaint_evidence = self._evidence_service.extract_complaint_evidence(complaint_frame, cases, context.max_evidence)
        cases.to_csv(context.output_dir / "divergence_abt_summary.csv", index=False, encoding="utf-8")
        if not interface_evidence.empty:
            interface_evidence.to_csv(context.output_dir / "divergence_ti_slices.csv", index=False, encoding="utf-8")
        if not complaint_evidence.empty:
            complaint_evidence.to_csv(context.output_dir / "divergence_tr_complaints.csv", index=False, encoding="utf-8")
        report = self._report_builder.build(cases, interface_evidence, complaint_evidence, context.positive_threshold, context.negative_threshold)
        (context.output_dir / "divergence_report.txt").write_text(report, encoding="utf-8")
        print(f"Output directory: {context.output_dir}")
        print(f"Cases exported: {len(cases)}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export divergence cases and supporting evidence.")
    parser.add_argument("--base", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--gap-pos", type=float, default=0.30)
    parser.add_argument("--gap-neg", type=float, default=-0.15)
    parser.add_argument("--comp", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--max-ev", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    base_dir = Path(args.base).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else base_dir / "analysis" / "outputs_4-5"
    DivergenceCaseExportOrchestrator().execute(DivergenceExportExecutionContext(base_dir, output_dir, args.gap_pos, args.gap_neg, args.comp, args.year, args.max_ev))


if __name__ == "__main__":
    main()
