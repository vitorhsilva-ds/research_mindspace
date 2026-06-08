#!/usr/bin/env python3
"""
Name: archived_page_capture_pipeline
Input: capture_plan.csv or a single archived URL
Output: MHTML files, PNG files, capture_log.jsonl, capture_summary.json
Usage: run as a Python script
"""
from __future__ import annotations
import argparse
from pathlib import Path
from archive_collection_core import BrowserCaptureService, BrowserDriverFactory, CaptureExecutionContract, CaptureExecutionOrchestrator, CapturePlanRepository, CaptureRowsFactory, CsvRepository, DEFAULT_CAPTURE_OUTPUT_DIR, DEFAULT_INVENTORY_DIR, JsonLinesRepository, JsonRepository, PageOverlaySuppressionService, PAGE_TYPES, SingleSnapshotCaptureService
class ArchivedPageCaptureDependencyFactory:
    """Builds dependencies for archived page capture."""
    def build_orchestrator(self) -> CaptureExecutionOrchestrator:
        overlay_service = PageOverlaySuppressionService()
        return CaptureExecutionOrchestrator(CapturePlanRepository(CsvRepository()), CaptureRowsFactory(), BrowserDriverFactory(), SingleSnapshotCaptureService(BrowserCaptureService(overlay_service)), JsonLinesRepository(), JsonRepository())
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture MHTML and PNG artifacts from archived page snapshots.")
    parser.add_argument("--years", nargs="+", type=int, default=[], help="Filter years. Default: all plan records.")
    parser.add_argument("--types", nargs="+", default=[], choices=list(PAGE_TYPES), help="Filter page types. Default: all page types.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip snapshots already captured successfully.")
    parser.add_argument("--single-url", type=str, default=None, help="Capture one archived URL for validation.")
    parser.add_argument("--inventory-dir", default=str(DEFAULT_INVENTORY_DIR), help=f"Inventory directory. Default: {DEFAULT_INVENTORY_DIR}")
    parser.add_argument("--output-dir", default=str(DEFAULT_CAPTURE_OUTPUT_DIR), help=f"Output directory. Default: {DEFAULT_CAPTURE_OUTPUT_DIR}")
    return parser.parse_args()
def main() -> None:
    args = parse_arguments(); contract = CaptureExecutionContract(Path(args.inventory_dir), Path(args.output_dir), tuple(sorted(set(args.years))), tuple(args.types), args.skip_existing, args.single_url)
    ArchivedPageCaptureDependencyFactory().build_orchestrator().execute(contract)
if __name__ == "__main__":
    main()
