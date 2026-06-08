#!/usr/bin/env python3
"""
Name: archive_capture_plan_builder
Input: inventory CSV files
Output: capture_plan.csv, capture_plan_summary.json
Usage: run as a Python script
"""
from __future__ import annotations
import argparse
from pathlib import Path
from archive_collection_core import CapturePlanInputContract, CapturePlanOrchestrator, CapturePlanOutputContract, CaptureSlotSelectionPolicy, CsvRepository, DateComputationService, DEFAULT_INVENTORY_DIR, InventoryRepository, JsonRepository
class CapturePlanDependencyFactory:
    """Builds dependencies for capture plan generation."""
    def build_orchestrator(self) -> CapturePlanOrchestrator:
        csv_repository = CsvRepository(); date_service = DateComputationService()
        return CapturePlanOrchestrator(InventoryRepository(csv_repository), CaptureSlotSelectionPolicy(date_service), date_service, csv_repository, JsonRepository())
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build capture_plan.csv using three slots per year and page type around the event date.")
    parser.add_argument("--inventory-dir", default=str(DEFAULT_INVENTORY_DIR), help=f"Inventory directory. Default: {DEFAULT_INVENTORY_DIR}")
    parser.add_argument("--years", nargs="+", type=int, default=[], help="Filter years. Default: all available inventory files.")
    return parser.parse_args()
def main() -> None:
    args = parse_arguments(); orchestrator = CapturePlanDependencyFactory().build_orchestrator()
    orchestrator.execute(CapturePlanInputContract(Path(args.inventory_dir), tuple(sorted(set(args.years)))), CapturePlanOutputContract(Path(args.inventory_dir)))
if __name__ == "__main__":
    main()
