#!/usr/bin/env python3
"""
Name: archive_collection_core
Input: archive index records and capture plan records
Output: inventory, capture plan, capture logs, and captured artifacts
Usage: imported by archive collection scripts
"""
from __future__ import annotations
import csv, json, logging, time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_ENCODING = "utf-8"
PAGE_TYPES: tuple[str, ...] = ("principal", "campanha", "produto")
ARCHIVE_CDX_ENDPOINT = "https://web.archive.org/cdx/search/cdx"
WINDOW_START_MMDD = "1118"
WINDOW_END_MMDD = "1202"
CDX_FILTER_STATUS = "statuscode:200"
CDX_FILTER_MIME = "mimetype:text/html"
CDX_FIELDS: tuple[str, ...] = ("timestamp", "original", "statuscode", "mimetype", "digest", "length")
INVENTORY_FIELDS: tuple[str, ...] = ("year", "page_type", "pattern", "timestamp", "datetime", "original", "statuscode", "mimetype", "digest", "length", "wayback_url")
CAPTURE_PLAN_FIELDS: tuple[str, ...] = ("year", "page_type", "slot", "bf_date", "timestamp", "datetime", "days_from_bf", "direction", "original", "pattern", "wayback_url")
DEFAULT_INVENTORY_DIR = Path("wayback_inventory")
DEFAULT_CAPTURE_OUTPUT_DIR = Path("wayback_captures")
TARGET_SITE_OUTPUT_STEM = "kabum"
TARGET_SITE_USER_AGENT_TOKEN = "archive-collection-bot"
TARGET_SITE_PAGE_TARGETS: dict[str, list[str]] = {
    "principal": [],
    "campanha": [],
    "produto": ["kabum.com.br/produto/10*"],
}
DEFAULT_COLLAPSE_DIGITS = 12
DEFAULT_CDX_MAX_RETRIES = 3
DEFAULT_CDX_RETRY_DELAY_SECONDS = 30
DEFAULT_CDX_SLEEP_BETWEEN_REQUESTS_SECONDS = 1.5
PAGE_LOAD_TIMEOUT_SECONDS = 60
WAIT_AFTER_LOAD_SECONDS = 4
SLEEP_BETWEEN_CAPTURES_SECONDS = 5
SLEEP_ON_ERROR_SECONDS = 20
MAX_CAPTURE_RETRIES = 3
VIEWPORT_WIDTH = 1440
VIEWPORT_HEIGHT = 900

def configure_logging() -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger(__name__)
LOGGER = configure_logging()

class ArchiveCollectionError(Exception):
    """Base exception for archive collection failures."""
class InventoryNotFoundError(ArchiveCollectionError):
    """Raised when inventory files are not available."""
class CapturePlanNotFoundError(ArchiveCollectionError):
    """Raised when the capture plan file is not available."""

@dataclass(frozen=True)
class ArchiveInventoryQueryContract:
    output_dir: Path
    years: tuple[int, ...]
    collapse_digits: int
    page_targets: dict[str, list[str]]
    window_start_mmdd: str = WINDOW_START_MMDD
    window_end_mmdd: str = WINDOW_END_MMDD

@dataclass(frozen=True)
class CapturePlanInputContract:
    inventory_dir: Path
    years: tuple[int, ...]

@dataclass(frozen=True)
class CapturePlanOutputContract:
    inventory_dir: Path
    plan_file_name: str = "capture_plan.csv"
    summary_file_name: str = "capture_plan_summary.json"

@dataclass(frozen=True)
class CaptureExecutionContract:
    inventory_dir: Path
    output_dir: Path
    years: tuple[int, ...]
    page_types: tuple[str, ...]
    skip_existing: bool
    single_url: str | None

@dataclass(frozen=True)
class ArchiveSnapshotRecord:
    year: int
    page_type: str
    pattern: str
    timestamp: str
    datetime_text: str
    original: str
    statuscode: str
    mimetype: str
    digest: str
    length: str
    wayback_url: str
    def to_row(self) -> dict[str, str | int]:
        return {"year": self.year, "page_type": self.page_type, "pattern": self.pattern, "timestamp": self.timestamp, "datetime": self.datetime_text, "original": self.original, "statuscode": self.statuscode, "mimetype": self.mimetype, "digest": self.digest, "length": self.length, "wayback_url": self.wayback_url}

@dataclass(frozen=True)
class CaptureArtifactRecord:
    timestamp: str
    year: str
    page_type: str
    wayback_url: str
    original: str
    datetime_text: str
    mhtml_path: Path
    png_path: Path
    status: str | None
    error: str | None
    mhtml_bytes: int | None
    png_bytes: int | None
    captured_at: str | None
    def to_log_payload(self) -> dict[str, str | int | None]:
        return {"timestamp": self.timestamp, "year": self.year, "page_type": self.page_type, "wayback_url": self.wayback_url, "original": self.original, "datetime": self.datetime_text, "mhtml_path": str(self.mhtml_path), "png_path": str(self.png_path), "status": self.status, "error": self.error, "mhtml_bytes": self.mhtml_bytes, "png_bytes": self.png_bytes, "captured_at": self.captured_at}

class JsonRepository:
    """Persists and loads JSON payloads."""
    def load(self, file_path: Path) -> dict:
        with file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            return json.load(input_file)
    def save(self, file_path: Path, payload: dict) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding=DEFAULT_ENCODING)

class CsvRepository:
    """Persists and loads CSV records."""
    def load_rows(self, file_path: Path) -> list[dict[str, str]]:
        with file_path.open(newline="", encoding=DEFAULT_ENCODING) as input_file:
            return list(csv.DictReader(input_file))
    def save_rows(self, file_path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", newline="", encoding=DEFAULT_ENCODING) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader(); writer.writerows(rows)
    def save_header_only(self, file_path: Path, fieldnames: Iterable[str]) -> None:
        self.save_rows(file_path, fieldnames, [])

class JsonLinesRepository:
    """Appends and reads JSON Lines records."""
    def append(self, file_path: Path, payload: dict) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding=DEFAULT_ENCODING) as output_file:
            output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    def load_success_keys(self, file_path: Path) -> set[str]:
        success_keys: set[str] = set()
        if not file_path.exists():
            return success_keys
        with file_path.open(encoding=DEFAULT_ENCODING) as input_file:
            for line in input_file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("status") == "ok":
                    success_keys.add(str(payload.get("timestamp", "")) + str(payload.get("page_type", "")))
        return success_keys

class DateComputationService:
    """Computes event dates and timestamp conversions."""
    def get_event_date(self, year: int) -> datetime:
        december_first = datetime(year, 12, 1)
        days_back = (december_first.weekday() - 4) % 7
        if days_back == 0:
            days_back = 7
        return december_first - timedelta(days=days_back)
    def timestamp_to_datetime(self, timestamp: str) -> datetime:
        return datetime.strptime(timestamp[:14].ljust(14, "0"), "%Y%m%d%H%M%S")
    def timestamp_to_iso_text(self, timestamp: str) -> str:
        if len(timestamp) < 14:
            return timestamp
        return f"{timestamp[0:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}:{timestamp[12:14]}"
    def get_snapshot_date(self, row: dict) -> datetime:
        return self.timestamp_to_datetime(str(row["timestamp"])).replace(hour=0, minute=0, second=0, microsecond=0)

class ArchiveUrlConstructionService:
    """Builds CDX query URLs and archived snapshot URLs."""
    def build_cdx_url(self, pattern: str, year: int, collapse_digits: int, window_start_mmdd: str, window_end_mmdd: str) -> str:
        from_timestamp = f"{year}{window_start_mmdd}000000"
        to_timestamp = f"{year}{window_end_mmdd}235959"
        query_parts = [f"url={pattern}", "output=json", f"from={from_timestamp}", f"to={to_timestamp}", f"collapse=timestamp:{collapse_digits}", f"fl={','.join(CDX_FIELDS)}", f"filter={CDX_FILTER_STATUS}", f"filter={CDX_FILTER_MIME}"]
        return f"{ARCHIVE_CDX_ENDPOINT}?" + "&".join(query_parts)
    def build_archived_url(self, timestamp: str, original_url: str) -> str:
        return f"https://web.archive.org/web/{timestamp}/{original_url}"

class ArchiveCdxGateway:
    """Fetches snapshot metadata from the archive CDX endpoint."""
    def __init__(self, url_service: ArchiveUrlConstructionService, date_service: DateComputationService, max_retries: int = DEFAULT_CDX_MAX_RETRIES, retry_delay_seconds: int = DEFAULT_CDX_RETRY_DELAY_SECONDS, sleep_between_requests_seconds: float = DEFAULT_CDX_SLEEP_BETWEEN_REQUESTS_SECONDS) -> None:
        self._url_service = url_service; self._date_service = date_service; self._max_retries = max_retries; self._retry_delay_seconds = retry_delay_seconds; self._sleep_between_requests_seconds = sleep_between_requests_seconds
    def fetch_records_for_type(self, page_type: str, patterns: list[str], year: int, collapse_digits: int, window_start_mmdd: str, window_end_mmdd: str) -> list[ArchiveSnapshotRecord]:
        seen_keys: set[tuple[str, str]] = set(); records: list[ArchiveSnapshotRecord] = []
        for pattern in patterns:
            LOGGER.info("    %s", pattern)
            cdx_url = self._url_service.build_cdx_url(pattern, year, collapse_digits, window_start_mmdd, window_end_mmdd)
            timeout_seconds = 20 if any(character.isupper() for character in pattern) else 30
            fetched_rows = self._fetch_cdx_rows(cdx_url, timeout=timeout_seconds)
            time.sleep(self._sleep_between_requests_seconds)
            new_count = 0
            for row in fetched_rows:
                unique_key = (row["timestamp"], row["original"])
                if unique_key in seen_keys:
                    continue
                seen_keys.add(unique_key)
                records.append(ArchiveSnapshotRecord(year=year, page_type=page_type, pattern=pattern, timestamp=row["timestamp"], datetime_text=self._date_service.timestamp_to_iso_text(row["timestamp"]), original=row["original"], statuscode=row.get("statuscode", ""), mimetype=row.get("mimetype", ""), digest=row.get("digest", ""), length=row.get("length", ""), wayback_url=self._url_service.build_archived_url(row["timestamp"], row["original"])))
                new_count += 1
            LOGGER.info("      -> fetched=%s new=%s total_type=%s", len(fetched_rows), new_count, len(records))
        return records
    def _fetch_cdx_rows(self, url: str, timeout: int) -> list[dict]:
        request = Request(url, headers={"User-Agent": f"Mozilla/5.0 ({TARGET_SITE_USER_AGENT_TOKEN}; archive-collection)"})
        for attempt in range(1, self._max_retries + 1):
            try:
                with urlopen(request, timeout=timeout) as response:
                    data = json.loads(response.read().decode(DEFAULT_ENCODING))
                if not data or len(data) < 2:
                    return []
                fields = data[0]
                return [dict(zip(fields, row)) for row in data[1:]]
            except HTTPError as error:
                LOGGER.warning("    HTTP %s attempt=%s/%s", error.code, attempt, self._max_retries)
            except URLError as error:
                LOGGER.warning("    URL error=%s attempt=%s/%s", error.reason, attempt, self._max_retries)
            except (TimeoutError, OSError) as error:
                LOGGER.warning("    Network timeout/error=%s attempt=%s/%s", error, attempt, self._max_retries)
            except json.JSONDecodeError:
                LOGGER.warning("    Invalid JSON attempt=%s/%s", attempt, self._max_retries)
            if attempt < self._max_retries:
                LOGGER.info("    Waiting %ss before retry.", self._retry_delay_seconds); time.sleep(self._retry_delay_seconds)
        LOGGER.warning("    Pattern skipped after all retry attempts."); return []

class InventoryPersistenceService:
    """Persists inventory outputs."""
    def __init__(self, csv_repository: CsvRepository, json_repository: JsonRepository) -> None:
        self._csv_repository = csv_repository; self._json_repository = json_repository
    def save_year_inventory(self, output_dir: Path, year: int, rows: list[dict]) -> Path:
        output_path = output_dir / f"inventory_{year}.csv"
        self._csv_repository.save_rows(output_path, INVENTORY_FIELDS, rows) if rows else self._csv_repository.save_header_only(output_path, INVENTORY_FIELDS)
        return output_path
    def save_all_inventory(self, output_dir: Path, rows: list[dict]) -> Path:
        output_path = output_dir / "inventory_all.csv"
        self._csv_repository.save_rows(output_path, INVENTORY_FIELDS, rows) if rows else self._csv_repository.save_header_only(output_path, INVENTORY_FIELDS)
        return output_path
    def save_summary(self, output_dir: Path, summary: dict) -> Path:
        output_path = output_dir / "inventory_summary.json"; self._json_repository.save(output_path, summary); return output_path

class InventoryOrchestrator:
    """Coordinates archive inventory generation."""
    def __init__(self, cdx_gateway: ArchiveCdxGateway, persistence_service: InventoryPersistenceService) -> None:
        self._cdx_gateway = cdx_gateway; self._persistence_service = persistence_service
    def execute(self, query_contract: ArchiveInventoryQueryContract) -> None:
        query_contract.output_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict] = []; summary: dict[int, dict[str, int]] = {}
        for year in query_contract.years:
            LOGGER.info("\n%s", "=" * 60); LOGGER.info("  %s | window: %s%s -> %s%s", year, year, query_contract.window_start_mmdd, year, query_contract.window_end_mmdd); LOGGER.info("%s", "=" * 60)
            year_rows: list[dict] = []; summary[year] = {}
            for page_type, patterns in query_contract.page_targets.items():
                LOGGER.info("\n  [%s]", page_type.upper())
                records = self._cdx_gateway.fetch_records_for_type(page_type, patterns, year, query_contract.collapse_digits, query_contract.window_start_mmdd, query_contract.window_end_mmdd)
                rows = [record.to_row() for record in records]
                summary[year][page_type] = len(rows); year_rows.extend(rows)
                LOGGER.info("  -> %s: %s snapshots", page_type, len(rows))
            year_path = self._persistence_service.save_year_inventory(query_contract.output_dir, year, year_rows)
            LOGGER.info("  Saved: %s (%s snapshots)", year_path, len(year_rows)); all_rows.extend(year_rows)
        consolidated_path = self._persistence_service.save_all_inventory(query_contract.output_dir, all_rows)
        summary_path = self._persistence_service.save_summary(query_contract.output_dir, summary)
        LOGGER.info("Consolidated inventory: %s (%s snapshots)", consolidated_path, len(all_rows)); LOGGER.info("Summary: %s", summary_path)
        self._print_inventory_table(query_contract.years, summary, len(all_rows))
    def _print_inventory_table(self, years: tuple[int, ...], summary: dict[int, dict[str, int]], total_all: int) -> None:
        column_width = 10; line_width = 6 + column_width * len(PAGE_TYPES) + 10
        print("\n" + "=" * line_width); print(f"{'YEAR':<6}" + "".join(f"{p:>{column_width}}" for p in PAGE_TYPES) + f"{'TOTAL':>10}"); print("-" * line_width)
        for year in years:
            row_summary = summary.get(year, {}); year_total = sum(row_summary.values())
            print(f"{year:<6}" + "".join(f"{row_summary.get(p, 0):>{column_width}}" for p in PAGE_TYPES) + f"{year_total:>10}")
        grand_totals = {p: sum(summary.get(y, {}).get(p, 0) for y in years) for p in PAGE_TYPES}
        print("-" * line_width); print(f"{'TOTAL':<6}" + "".join(f"{grand_totals[p]:>{column_width}}" for p in PAGE_TYPES) + f"{total_all:>10}"); print("=" * line_width); print()

class InventoryRepository:
    """Loads inventory CSV files for capture planning."""
    def __init__(self, csv_repository: CsvRepository) -> None:
        self._csv_repository = csv_repository
    def load_inventory_buckets(self, input_contract: CapturePlanInputContract) -> dict[tuple[int, str], list[dict[str, str]]]:
        csv_files = sorted(input_contract.inventory_dir.glob("inventory_*.csv")); csv_files = [p for p in csv_files if p.name != "inventory_all.csv"]
        if not csv_files:
            raise InventoryNotFoundError(f"No inventory_YYYY.csv files found in {input_contract.inventory_dir}.")
        buckets: dict[tuple[int, str], list[dict[str, str]]] = {}
        for csv_file_path in csv_files:
            try:
                year = int(csv_file_path.stem.split("_")[1])
            except (IndexError, ValueError):
                LOGGER.warning("Skipping file with unexpected name: %s", csv_file_path.name); continue
            if input_contract.years and year not in input_contract.years:
                continue
            rows = self._csv_repository.load_rows(csv_file_path); LOGGER.info("  %s: %s rows", csv_file_path.name, len(rows))
            for row in rows:
                row["year"] = str(year); page_type = row.get("page_type", "").strip()
                if page_type:
                    buckets.setdefault((year, page_type), []).append(row)
        for key in buckets:
            buckets[key].sort(key=lambda row: row.get("timestamp", ""))
        return buckets

class CaptureSlotSelectionPolicy:
    """Selects capture slots around the configured event date."""
    def __init__(self, date_service: DateComputationService) -> None:
        self._date_service = date_service
    def select_slots(self, rows: list[dict], event_date: datetime) -> list[dict]:
        if not rows:
            return []
        event_day = event_date.replace(hour=0, minute=0, second=0, microsecond=0)
        rows_by_day: dict[datetime, dict] = {}
        for row in rows:
            rows_by_day.setdefault(self._date_service.get_snapshot_date(row), row)
        sorted_days = sorted(rows_by_day.keys())
        slot_a_day = min(sorted_days, key=lambda day: abs((day - event_day).days)); slot_a = rows_by_day[slot_a_day]
        before_days = [day for day in sorted_days if day < slot_a_day]; after_days = [day for day in sorted_days if day > slot_a_day]
        slot_b = rows_by_day[max(before_days)] if before_days else None; slot_c = rows_by_day[min(after_days)] if after_days else None
        if slot_a_day != event_day and event_day in rows_by_day:
            slot_a = rows_by_day[event_day]; before_days = [day for day in sorted_days if day < event_day]; after_days = [day for day in sorted_days if day > event_day]
            slot_b = rows_by_day[max(before_days)] if before_days else None; slot_c = rows_by_day[min(after_days)] if after_days else None
        selected: list[dict] = []
        for slot_label, row in (("A_bf_day", slot_a), ("B_before", slot_b), ("C_after", slot_c)):
            if row is None: continue
            delta_days = (self._date_service.get_snapshot_date(row) - event_day).days
            enriched = dict(row); enriched["slot"] = slot_label; enriched["bf_date"] = event_date.strftime("%Y-%m-%d"); enriched["days_from_bf"] = abs(delta_days); enriched["direction"] = "on" if delta_days == 0 else ("before" if delta_days < 0 else "after")
            selected.append(enriched)
        return selected

class CapturePlanOrchestrator:
    """Coordinates capture plan generation."""
    def __init__(self, inventory_repository: InventoryRepository, slot_policy: CaptureSlotSelectionPolicy, date_service: DateComputationService, csv_repository: CsvRepository, json_repository: JsonRepository) -> None:
        self._inventory_repository = inventory_repository; self._slot_policy = slot_policy; self._date_service = date_service; self._csv_repository = csv_repository; self._json_repository = json_repository
    def execute(self, input_contract: CapturePlanInputContract, output_contract: CapturePlanOutputContract) -> None:
        LOGGER.info("Loading inventories from: %s", input_contract.inventory_dir)
        buckets = self._inventory_repository.load_inventory_buckets(input_contract)
        plan_rows: list[dict] = []; summary: dict[int, dict[str, dict]] = {}; active_years = sorted({year for (year, _) in buckets.keys()})
        for year in active_years:
            event_date = self._date_service.get_event_date(year); LOGGER.info("\n%s", "=" * 60); LOGGER.info("  %s | event date: %s", year, event_date.strftime("%Y-%m-%d")); LOGGER.info("%s", "=" * 60); summary[year] = {}
            for page_type in PAGE_TYPES:
                rows = buckets.get((year, page_type), []); LOGGER.info("  [%s] available snapshots=%s", page_type.upper(), len(rows))
                if not rows:
                    LOGGER.warning("  No snapshots for %s/%s", year, page_type); summary[year][page_type] = {"slots": 0, "available": 0}; continue
                slots = self._slot_policy.select_slots(rows, event_date)
                for slot in slots:
                    LOGGER.info("    %-12s %-20s (%s, %sd) %s", slot["slot"], slot["datetime"], slot["direction"], slot["days_from_bf"], slot.get("original", "")[:50]); plan_rows.append(slot)
                summary[year][page_type] = {"slots": len(slots), "available": len(rows), "slots_detail": [{"slot": s["slot"], "date": s["datetime"], "direction": s["direction"]} for s in slots]}
        plan_path = output_contract.inventory_dir / output_contract.plan_file_name; summary_path = output_contract.inventory_dir / output_contract.summary_file_name
        self._csv_repository.save_rows(plan_path, CAPTURE_PLAN_FIELDS, plan_rows); self._json_repository.save(summary_path, summary)
        LOGGER.info("Capture plan saved: %s (%s snapshots)", plan_path, len(plan_rows)); LOGGER.info("Plan summary saved: %s", summary_path)
        self._print_plan_summary(active_years, summary)
    def _print_plan_summary(self, active_years: list[int], summary: dict[int, dict[str, dict]]) -> None:
        print("\n" + "=" * 72); print(f"{'YEAR':<6} {'TYPE':<12} {'SLOTS':>6}  {'AVAILABLE':>12}  DIRECTIONS"); print("-" * 72)
        total_slots = 0; incomplete: list[tuple[int, str]] = []
        for year in active_years:
            for page_type in PAGE_TYPES:
                page_summary = summary[year].get(page_type, {}); slot_count = page_summary.get("slots", 0); available_count = page_summary.get("available", 0)
                direction_text = ", ".join(f"{slot['slot']}={slot['direction']}" for slot in page_summary.get("slots_detail", [])); flag = "! " if slot_count < 3 else "  "
                print(f"{flag}{year:<4} {page_type:<12} {slot_count:>6}  {available_count:>12}  {direction_text}"); total_slots += slot_count
                if slot_count < 3: incomplete.append((year, page_type))
        print("-" * 72); print(f"{'TOTAL':<18} {total_slots:>6} snapshots in plan"); print("=" * 72)
        if incomplete:
            print(f"\nIncomplete slots (< 3): {len(incomplete)} year/type combinations")
            for year, page_type in incomplete:
                page_summary = summary[year].get(page_type, {}); print(f"   {year}/{page_type}: {page_summary.get('slots', 0)} slot(s), {page_summary.get('available', 0)} available snapshots")
        print()

class BrowserDriverFactory:
    """Builds the browser driver used for archived page capture."""
    def build(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        try:
            from selenium.webdriver.chrome.service import Service
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())
        except ImportError:
            service = None
        options = Options(); options.add_argument("--headless=new"); options.add_argument("--no-sandbox"); options.add_argument("--disable-dev-shm-usage"); options.add_argument("--disable-gpu"); options.add_argument(f"--window-size={VIEWPORT_WIDTH},{VIEWPORT_HEIGHT}"); options.add_argument("--disable-extensions"); options.add_argument("--disable-infobars"); options.add_argument("--enable-features=NetworkService,NetworkServiceInProcess"); options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
        driver = webdriver.Chrome(service=service, options=options) if service else webdriver.Chrome(options=options); driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT_SECONDS); return driver

class PageOverlaySuppressionService:
    """Suppresses overlays before page persistence."""
    def suppress(self, driver) -> None:
        driver.execute_script("""
        var style = document.getElementById('__capture_suppress__');
        if (!style) {
            style = document.createElement('style'); style.id = '__capture_suppress__';
            style.innerHTML = `[class*="cookie"], [id*="cookie"], [class*="Cookie"], [id*="Cookie"], [class*="lgpd"], [id*="lgpd"], [class*="consent"], [id*="consent"], [class*="gdpr"], [id*="gdpr"], [class*="modal"], [class*="Modal"], [class*="popup"], [class*="Popup"], [class*="overlay"], [class*="Overlay"], [class*="sticky"]:not(header):not(nav), [class*="fixed-bottom"], #wm-ipp-base, #wm-ipp, .wb-autocomplete-suggestions { display: none !important; visibility: hidden !important; }`;
            document.head.appendChild(style);
        }
        document.querySelectorAll('*').forEach(function(el) { try { var s = window.getComputedStyle(el); if ((s.position === 'fixed' || s.position === 'sticky') && el.id !== '__capture_suppress__' && !el.matches('header, nav, #header, .header')) { el.style.setProperty('display', 'none', 'important'); } } catch(e) {} });
        """)

class BrowserCaptureService:
    """Captures MHTML and full-page PNG artifacts."""
    def __init__(self, overlay_service: PageOverlaySuppressionService) -> None:
        self._overlay_service = overlay_service
    def capture_mhtml(self, driver) -> bytes:
        result = driver.execute_cdp_cmd("Page.captureSnapshot", {"format": "mhtml"}); return result["data"].encode(DEFAULT_ENCODING)
    def capture_full_page_png(self, driver) -> bytes:
        from PIL import Image
        import io
        overlap = 50; prescroll_step = 600; prescroll_wait = 0.15; load_wait = 3.0; stitch_wait = 0.4
        driver.set_window_size(VIEWPORT_WIDTH, VIEWPORT_HEIGHT); time.sleep(0.5)
        def get_total_height() -> int:
            return int(driver.execute_script("return Math.max(document.body.scrollHeight,document.documentElement.scrollHeight,document.body.offsetHeight,document.documentElement.offsetHeight)"))
        total_height = get_total_height(); LOGGER.info("    Total page height: %spx", total_height)
        scroll_position = 0
        while scroll_position < total_height:
            driver.execute_script(f"window.scrollTo(0, {scroll_position});"); time.sleep(prescroll_wait); scroll_position += prescroll_step
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);"); time.sleep(prescroll_wait); LOGGER.info("    Waiting %ss for lazy-loaded content.", load_wait); time.sleep(load_wait)
        total_height = get_total_height(); LOGGER.info("    Height after lazy-load: %spx", total_height)
        self._overlay_service.suppress(driver); driver.execute_script("window.scrollTo(0, 0);"); time.sleep(0.5); self._overlay_service.suppress(driver)
        image_slices = []; scroll_y = 0; step = None; last_actual_y = -1
        while True:
            driver.execute_script(f"window.scrollTo(0, {scroll_y});"); time.sleep(stitch_wait); self._overlay_service.suppress(driver)
            actual_y = int(driver.execute_script("return window.pageYOffset;")); png_bytes = driver.get_screenshot_as_png(); image = Image.open(io.BytesIO(png_bytes))
            if step is None:
                step = image.height; LOGGER.info("    Screenshot real height: %spx (viewport=%spx)", step, VIEWPORT_HEIGHT)
            crop_top = overlap if scroll_y > 0 else 0; scroll_stuck = actual_y == last_actual_y; page_covered = actual_y + image.height; is_last = page_covered >= total_height or scroll_stuck
            crop_bottom = max(crop_top + 1, min(image.height, crop_top + (total_height - actual_y))) if is_last else image.height
            cropped = image.crop((0, crop_top, image.width, crop_bottom)); image_slices.append(cropped); LOGGER.info("    Slice: scroll=%s actual_y=%s crop=[%s:%s] -> %spx%s", scroll_y, actual_y, crop_top, crop_bottom, cropped.height, " [last]" if is_last else "")
            if is_last: break
            last_actual_y = actual_y; scroll_y = actual_y + step - overlap
        final = Image.new("RGB", (image_slices[0].width, sum(s.height for s in image_slices)), (255, 255, 255)); y_offset = 0
        for image_slice in image_slices:
            final.paste(image_slice, (0, y_offset)); y_offset += image_slice.height
        buffer = io.BytesIO(); final.save(buffer, format="PNG", optimize=False); return buffer.getvalue()

class SingleSnapshotCaptureService:
    """Captures one archived snapshot with retry handling."""
    def __init__(self, capture_service: BrowserCaptureService) -> None:
        self._capture_service = capture_service
    def capture_one(self, driver, row: dict, output_dir: Path) -> CaptureArtifactRecord:
        from selenium.common.exceptions import TimeoutException, WebDriverException
        timestamp = str(row["timestamp"]); year = str(row["year"]); page_type = str(row.get("page_type", "pagina")); archived_url = str(row["wayback_url"])
        stem = f"{TARGET_SITE_OUTPUT_STEM}_{year}_{page_type}_{timestamp}"; mhtml_path = output_dir / f"{stem}.mhtml"; png_path = output_dir / f"{stem}.png"
        base = {"timestamp": timestamp, "year": year, "page_type": page_type, "wayback_url": archived_url, "original": str(row.get("original", "")), "datetime_text": str(row.get("datetime", "")), "mhtml_path": mhtml_path, "png_path": png_path}
        last_error: str | None = None
        for attempt in range(1, MAX_CAPTURE_RETRIES + 1):
            try:
                LOGGER.info("  [%s/%s] %s", attempt, MAX_CAPTURE_RETRIES, archived_url); driver.get(archived_url); time.sleep(WAIT_AFTER_LOAD_SECONDS)
                self._capture_service._overlay_service.suppress(driver)
                mhtml_data = self._capture_service.capture_mhtml(driver); mhtml_path.write_bytes(mhtml_data)
                png_data = self._capture_service.capture_full_page_png(driver); png_path.write_bytes(png_data)
                LOGGER.info("  ok mhtml_bytes=%s png_bytes=%s", len(mhtml_data), len(png_data))
                return CaptureArtifactRecord(**base, status="ok", error=None, mhtml_bytes=len(mhtml_data), png_bytes=len(png_data), captured_at=datetime.now().isoformat())
            except TimeoutException:
                last_error = f"timeout attempt={attempt}"; LOGGER.warning("  Timeout attempt=%s", attempt)
            except WebDriverException as error:
                last_error = str(error)[:200]; LOGGER.warning("  WebDriverException: %s", str(error)[:80])
            except Exception as error:
                last_error = str(error)[:200]; LOGGER.warning("  Unexpected error: %s", str(error)[:80])
            if attempt < MAX_CAPTURE_RETRIES:
                LOGGER.info("  Waiting %ss before retry.", SLEEP_ON_ERROR_SECONDS); time.sleep(SLEEP_ON_ERROR_SECONDS)
        LOGGER.error("  Capture failed after all retry attempts."); return CaptureArtifactRecord(**base, status="failed", error=last_error, mhtml_bytes=None, png_bytes=None, captured_at=None)

class CapturePlanRepository:
    """Loads capture plan records."""
    def __init__(self, csv_repository: CsvRepository) -> None:
        self._csv_repository = csv_repository
    def load_rows(self, inventory_dir: Path, years: tuple[int, ...], page_types: tuple[str, ...]) -> list[dict[str, str]]:
        plan_file_path = inventory_dir / "capture_plan.csv"
        if not plan_file_path.exists():
            raise CapturePlanNotFoundError(f"capture_plan.csv not found in {inventory_dir}.")
        rows = self._csv_repository.load_rows(plan_file_path)
        if years: rows = [row for row in rows if int(row["year"]) in years]
        if page_types: rows = [row for row in rows if row["page_type"] in page_types]
        LOGGER.info("%s snapshots loaded from capture_plan.csv.", len(rows)); return rows

class CaptureRowsFactory:
    """Builds capture rows for single URL execution."""
    def build_single_url_row(self, single_url: str) -> dict[str, str]:
        return {"timestamp": "test", "year": "test", "page_type": "teste", "wayback_url": single_url, "original": single_url, "datetime": datetime.now().isoformat()}

class CaptureExecutionOrchestrator:
    """Coordinates archived page capture execution."""
    def __init__(self, plan_repository: CapturePlanRepository, rows_factory: CaptureRowsFactory, driver_factory: BrowserDriverFactory, snapshot_capture_service: SingleSnapshotCaptureService, jsonl_repository: JsonLinesRepository, json_repository: JsonRepository) -> None:
        self._plan_repository = plan_repository; self._rows_factory = rows_factory; self._driver_factory = driver_factory; self._snapshot_capture_service = snapshot_capture_service; self._jsonl_repository = jsonl_repository; self._json_repository = json_repository
    def execute(self, execution_contract: CaptureExecutionContract) -> None:
        output_dir = execution_contract.output_dir; log_path = output_dir / "capture_log.jsonl"; summary_path = output_dir / "capture_summary.json"; output_dir.mkdir(parents=True, exist_ok=True)
        rows = [self._rows_factory.build_single_url_row(execution_contract.single_url)] if execution_contract.single_url else self._plan_repository.load_rows(execution_contract.inventory_dir, execution_contract.years, execution_contract.page_types)
        for row in rows: self._get_capture_output_dir(output_dir, row, execution_contract.single_url).mkdir(parents=True, exist_ok=True)
        completed_keys = self._jsonl_repository.load_success_keys(log_path) if execution_contract.skip_existing else set()
        if execution_contract.skip_existing: LOGGER.info("skip_existing enabled: previous_success_count=%s", len(completed_keys))
        stats: dict[str, dict[str, int]] = {}; LOGGER.info("Initializing browser driver."); driver = self._driver_factory.build()
        try:
            total = len(rows)
            for index, row in enumerate(rows, 1):
                timestamp = str(row.get("timestamp", "test")); year = str(row.get("year", "test")); page_type = str(row.get("page_type", "pagina")); capture_key = timestamp + page_type
                stats.setdefault(year, {"ok": 0, "failed": 0, "skipped": 0})
                LOGGER.info("\n%s", "-" * 60); LOGGER.info("[%s/%s] %s / %s | %s", index, total, year, page_type, row.get("datetime", timestamp)); LOGGER.info("  %s", str(row.get("original", ""))[:70])
                if execution_contract.skip_existing and capture_key in completed_keys:
                    LOGGER.info("  Existing successful capture found. Skipping."); stats[year]["skipped"] += 1; continue
                capture_record = self._snapshot_capture_service.capture_one(driver, row, self._get_capture_output_dir(output_dir, row, execution_contract.single_url))
                self._jsonl_repository.append(log_path, capture_record.to_log_payload()); stats[year]["ok" if capture_record.status == "ok" else "failed"] += 1
                if index < total: time.sleep(SLEEP_BETWEEN_CAPTURES_SECONDS)
        finally:
            driver.quit()
        totals = self._compute_totals(stats); self._json_repository.save(summary_path, {"generated_at": datetime.now().isoformat(), "by_year": stats, "totals": totals}); self._print_capture_summary(stats, totals, log_path, summary_path, output_dir)
    def _get_capture_output_dir(self, output_dir: Path, row: dict, single_url: str | None) -> Path:
        return output_dir / "test" if single_url else output_dir / str(row["year"]) / str(row["page_type"])
    def _compute_totals(self, stats: dict[str, dict[str, int]]) -> dict[str, int]:
        totals = {"ok": 0, "failed": 0, "skipped": 0}
        for year_stats in stats.values():
            for key in totals: totals[key] += year_stats[key]
        return totals
    def _print_capture_summary(self, stats: dict[str, dict[str, int]], totals: dict[str, int], log_path: Path, summary_path: Path, output_dir: Path) -> None:
        print("\n" + "=" * 50); print(f"{'YEAR':<8}  {'OK':>6}  {'FAILED':>6}  {'SKIPPED':>8}"); print("-" * 50)
        for year, year_stats in sorted(stats.items()): print(f"{year:<8}  {year_stats['ok']:>6}  {year_stats['failed']:>6}  {year_stats['skipped']:>8}")
        print("-" * 50); print(f"{'TOTAL':<8}  {totals['ok']:>6}  {totals['failed']:>6}  {totals['skipped']:>8}"); print("=" * 50); print(f"\nLog: {log_path}"); print(f"Summary: {summary_path}"); print(f"Output: {output_dir}/YYYY/page_type/\n")
