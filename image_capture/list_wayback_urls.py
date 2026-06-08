#!/usr/bin/env python3
"""
Name: archive_inventory_listing
Input: target archive URL patterns and year window configuration
Output: inventory_YYYY.csv, inventory_all.csv, inventory_summary.json
Usage: run as a Python script
"""
from __future__ import annotations
import argparse
from pathlib import Path
from archive_collection_core import ArchiveCdxGateway, ArchiveInventoryQueryContract, ArchiveUrlConstructionService, CsvRepository, DateComputationService, DEFAULT_COLLAPSE_DIGITS, DEFAULT_INVENTORY_DIR, InventoryOrchestrator, InventoryPersistenceService, JsonRepository, TARGET_SITE_PAGE_TARGETS
DEFAULT_YEARS = (2019,2020,2021,2022,2023,2024,2025)
class ArchiveInventoryDependencyFactory:
    """Builds dependencies for archive inventory listing."""
    def build_orchestrator(self) -> InventoryOrchestrator:
        date_service = DateComputationService(); url_service = ArchiveUrlConstructionService()
        return InventoryOrchestrator(cdx_gateway=ArchiveCdxGateway(url_service, date_service), persistence_service=InventoryPersistenceService(CsvRepository(), JsonRepository()))
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build archive CDX inventory files for the configured target patterns.")
    parser.add_argument("--years", nargs="+", type=int, default=list(DEFAULT_YEARS), help=f"Years to inventory. Default: {list(DEFAULT_YEARS)}")
    parser.add_argument("--collapse", type=int, default=DEFAULT_COLLAPSE_DIGITS, help=f"Timestamp collapse digits. Default: {DEFAULT_COLLAPSE_DIGITS}. Use 8 for daily collapse.")
    parser.add_argument("--output-dir", default=str(DEFAULT_INVENTORY_DIR), help=f"Output directory. Default: {DEFAULT_INVENTORY_DIR}")
    return parser.parse_args()
def build_query_contract(args: argparse.Namespace) -> ArchiveInventoryQueryContract:
    return ArchiveInventoryQueryContract(output_dir=Path(args.output_dir), years=tuple(sorted(set(args.years))), collapse_digits=args.collapse, page_targets=TARGET_SITE_PAGE_TARGETS)
def main() -> None:
    args = parse_arguments(); ArchiveInventoryDependencyFactory().build_orchestrator().execute(build_query_contract(args))
if __name__ == "__main__":
    main()
