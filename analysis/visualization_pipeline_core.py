#!/usr/bin/env python3
"""
Name: visualization_pipeline_core
Input: tabular analytical files
Output: reusable plotting and data-loading services
Usage: import from advanced visualization scripts
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
COMPONENT_CODES = ("M", "I", "N", "D", "S", "P", "A", "C", "E")
COMPONENT_NAMES = {
    "M": "Messenger",
    "I": "Incentives",
    "N": "Norms",
    "D": "Defaults",
    "S": "Salience",
    "P": "Priming",
    "A": "Affect",
    "C": "Commitments",
    "E": "Ego",
}

COMPONENT_LABELS = {code: f"{name} ({code})" for code, name in COMPONENT_NAMES.items()}

PAGE_TYPE_LABELS = {
    "principal": "Página principal",
    "campanha": "Página de campanha",
    "produto": "Página de produto",
}

WINDOW_LABELS = {
    "pre": "Pré-BF",
    "bf": "Janela BF",
    "pos": "Pós-BF",
}

FALLBACK_COUNTS = {
    "N_I": {2020: 195, 2021: 95, 2022: 91, 2023: 98, 2024: 116, 2025: 80},
    "N_R": {2020: 221, 2021: 97, 2022: 91, 2023: 218, 2024: 219, 2025: 133},
    "indeterminate": {2020: 65, 2021: 23, 2022: 30, 2023: 77, 2024: 78, 2025: 44},
}

PAGE_TYPE_COLORS = {
    "principal": "#1F77B4",
    "campanha": "#FF7F0E",
    "produto": "#2CA02C",
}
WINDOW_COLORS = {"pre": "#AEC7E8", "bf": "#1F77B4", "pos": "#17BECF"}
CLASSIFICATION_COLORS = {"classified": "#2E75B6", "indeterminate": "#BFBFBF"}
YEAR_COLORS = {
    2020: "#C6DBEF",
    2021: "#9ECAE1",
    2022: "#6BAED6",
    2023: "#4292C6",
    2024: "#2171B5",
    2025: "#084594",
}
COMPONENT_COLORS = {
    "A": "#D94801",
    "P": "#F16913",
    "I": "#2171B5",
    "N": "#4292C6",
    "S": "#6BAED6",
    "M": "#9ECAE1",
    "E": "#BDBDBD",
    "D": "#D9D9D9",
    "C": "#F0F0F0",
}

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "technical_heat",
    ["#FFFFFF", "#FEE6CE", "#FDAE6B", "#F16913", "#D94801", "#8C2D04"],
    N=256,
)
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "technical_blue",
    ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#084594"],
    N=256,
)
ORANGE_CMAP = LinearSegmentedColormap.from_list(
    "technical_orange",
    ["#FFFFFF", "#FEE6CE", "#FDAE6B", "#F16913", "#8C2D04"],
    N=256,
)
DIVERGENCE_CMAP = LinearSegmentedColormap.from_list(
    "technical_divergence",
    ["#084594", "#4292C6", "#C6DBEF", "#FFFFFF", "#FEE6CE", "#F16913", "#8C2D04"],
    N=512,
)

@dataclass(frozen=True)
class FileDiscoveryResult:
    file_path: Path | None
    label: str


@dataclass(frozen=True)
class BasePathContract:
    base_dir: Path
    output_dir: Path


class ChartDisplayTextPolicy:
    """Provides display-only chart labels allowed by the output exception."""

    def commerce_event_label(self) -> str:
        return "KaBuM! Black Friday 2020–2025"

    def complaint_source_label(self) -> str:
        return "Reclame Aqui"

    def combined_context_label(self) -> str:
        return "KaBuM! / Reclame Aqui — Black Friday 2020–2025"


class MatplotlibPublicationStyleService:
    """Applies deterministic publication styling."""

    def apply(self) -> None:
        plt.rcParams.update(
            {
                "font.family": "DejaVu Sans",
                "font.size": 10,
                "axes.titlesize": 11,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8.5,
                "legend.framealpha": 0.9,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
                "grid.linestyle": "--",
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
                "savefig.facecolor": "white",
            }
        )


class TabularFileRepository:
    """Loads CSV and JSON analytical inputs."""

    def find_latest(self, base_dir: Path, patterns: Sequence[str]) -> FileDiscoveryResult:
        for pattern in patterns:
            matches = sorted(base_dir.glob(pattern))
            if matches:
                return FileDiscoveryResult(file_path=matches[-1], label=pattern)
        return FileDiscoveryResult(file_path=None, label="not_found")

    def load_csv(self, file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path, low_memory=False)

    def load_json(self, file_path: Path) -> dict:
        with file_path.open(encoding="utf-8") as input_file:
            return json.load(input_file)

    def save_text(self, file_path: Path, content: str) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


class AnalyticalTableLoader:
    """Loads common analytical tables using neutral discovery rules."""

    def __init__(self, repository: TabularFileRepository) -> None:
        self._repository = repository

    def load_abt(self, base_dir: Path) -> pd.DataFrame:
        result = self._repository.find_latest(base_dir, ("corpus/abt.csv", "abt.csv"))
        if result.file_path is None:
            raise FileNotFoundError("abt.csv was not found.")
        frame = self._repository.load_csv(result.file_path)
        frame["ano"] = frame["ano"].astype(int)
        frame["comp"] = frame["comp"].astype(str).str.strip().str.upper()
        print(f"Analytical base table loaded: {result.file_path} ({len(frame)} rows)")
        return frame

    def load_interface_table(self, base_dir: Path, required: bool = False) -> pd.DataFrame | None:
        result = self._repository.find_latest(
            base_dir,
            ("corpus/ti_completa_revisado_*.csv", "corpus/ti_completa.csv"),
        )
        if result.file_path is None:
            if required:
                raise FileNotFoundError("Interface table was not found.")
            print("Interface table not found; fallback values will be used where available.")
            return None
        frame = self._repository.load_csv(result.file_path)
        if "ano" in frame.columns:
            frame["ano"] = frame["ano"].astype(int)
        print(f"Interface table loaded: {result.file_path} ({len(frame)} rows)")
        return frame

    def load_complaint_table(self, base_dir: Path, required: bool = False) -> pd.DataFrame | None:
        result = self._repository.find_latest(
            base_dir,
            ("corpus/tr_completa_revisado_*.csv", "corpus/tr_completa.csv"),
        )
        if result.file_path is None:
            if required:
                raise FileNotFoundError("Complaint table was not found.")
            print("Complaint table not found; fallback values will be used where available.")
            return None
        frame = self._repository.load_csv(result.file_path)
        if "ano" in frame.columns:
            frame["ano"] = frame["ano"].astype(int)
        print(f"Complaint table loaded: {result.file_path} ({len(frame)} rows)")
        return frame

    def load_capture_plan(self, base_dir: Path, explicit_path: Path | None = None) -> pd.DataFrame | None:
        if explicit_path is not None and explicit_path.exists():
            frame = self._repository.load_csv(explicit_path)
            print(f"Capture plan loaded: {explicit_path} ({len(frame)} rows)")
            return frame
        result = self._repository.find_latest(
            base_dir,
            ("archive_inventory/capture_plan.csv", "capture_inventory/capture_plan.csv"),
        )
        if result.file_path is None:
            print("Capture plan not found; fallback values will be used where available.")
            return None
        frame = self._repository.load_csv(result.file_path)
        print(f"Capture plan loaded: {result.file_path} ({len(frame)} rows)")
        return frame

    def load_capture_summary(self, base_dir: Path, explicit_path: Path | None = None) -> dict | None:
        if explicit_path is not None and explicit_path.exists():
            print(f"Capture summary loaded: {explicit_path}")
            return self._repository.load_json(explicit_path)
        result = self._repository.find_latest(
            base_dir,
            ("archive_inventory/capture_summary.json", "capture_inventory/capture_summary.json"),
        )
        if result.file_path is None:
            print("Capture summary not found; fallback values will be used where available.")
            return None
        print(f"Capture summary loaded: {result.file_path}")
        return self._repository.load_json(result.file_path)


class AnalyticalMetricService:
    """Computes shared analytical metrics."""

    def pivot(self, frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
        return (
            frame.pivot(index="comp", columns="ano", values=value_column)
            .reindex(index=COMPONENT_CODES, columns=DEFAULT_YEARS)
            .fillna(0)
        )

    def extract_total_counts(self, abt_frame: pd.DataFrame | None) -> tuple[dict[int, int], dict[int, int]]:
        if abt_frame is not None:
            interface_counts = abt_frame.groupby("ano")["N_I"].first().astype(int).to_dict()
            complaint_counts = abt_frame.groupby("ano")["N_R"].first().astype(int).to_dict()
            return (
                {year: interface_counts.get(year, FALLBACK_COUNTS["N_I"][year]) for year in DEFAULT_YEARS},
                {year: complaint_counts.get(year, FALLBACK_COUNTS["N_R"][year]) for year in DEFAULT_YEARS},
            )
        return FALLBACK_COUNTS["N_I"].copy(), FALLBACK_COUNTS["N_R"].copy()

    def calculate_cooccurrence_matrix(self, frame: pd.DataFrame, year: int | None = None) -> tuple[np.ndarray, int]:
        if year is not None:
            frame = frame[frame["ano"] == year]
        available_columns = [code for code in COMPONENT_CODES if code in frame.columns]
        component_frame = frame[available_columns].fillna(0).astype(int)
        total_rows = len(component_frame)
        matrix = np.zeros((len(COMPONENT_CODES), len(COMPONENT_CODES)))
        for i, left_code in enumerate(COMPONENT_CODES):
            if left_code not in available_columns:
                continue
            matrix[i, i] = component_frame[left_code].sum() / total_rows if total_rows else 0.0
            for j, right_code in enumerate(COMPONENT_CODES):
                if j <= i or right_code not in available_columns:
                    continue
                value = ((component_frame[left_code] == 1) & (component_frame[right_code] == 1)).sum()
                matrix[i, j] = value / total_rows if total_rows else 0.0
                matrix[j, i] = matrix[i, j]
        return matrix, total_rows


class AxisAnnotationService:
    """Adds common chart annotations."""

    def add_bar_labels(self, axis, bars, offset: float = 1.0, fmt: str = "{:.0f}") -> None:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + offset,
                    fmt.format(height),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    def configure_year_axis(self, axis) -> None:
        axis.set_xticks(np.arange(len(DEFAULT_YEARS)))
        axis.set_xticklabels([str(year) for year in DEFAULT_YEARS])


class OutputDirectoryService:
    """Ensures output directories exist."""

    def ensure(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)


def component_display_labels() -> list[str]:
    return [f"{COMPONENT_NAMES[code]}\n({code})" for code in COMPONENT_CODES]


def component_axis_labels() -> list[str]:
    return [f"{COMPONENT_NAMES[code]} ({code})" for code in COMPONENT_CODES]
