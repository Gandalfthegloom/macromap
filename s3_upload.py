"""
s3_upload.py
Upload BACI .csv.gz files to S3 using safe creds (AWS profile / env),
matching the MacroMap key structure:
  baci/hs6/year=<YYYY>/<FILENAME>

Highlights
- No hard-coded credentials. Use --profile or your default AWS config.
- Sets ContentType=text/csv and ContentEncoding=gzip.
- Private objects with SSE-S3 (AES256).
- Skips existing keys unless --overwrite is passed.
- Dry-run mode to preview without uploading.
- Flexible source discovery with a sensible default; override via --source.
- Optional normalization to strict part names: part-0000.csv.gz, etc.
- NEW: Selective upload by year via --years / --min-year / --max-year (and INI config)

Examples
  # Preview what would be uploaded (uses default bucket/region/source)
  python s3_upload.py --dry-run

  # Upload with a specific AWS profile and overwrite existing keys
  python s3_upload.py --profile your-profile --overwrite

  # Custom source folder and a custom prefix
  python s3_upload.py --source "D:/world trade/macromap/data/zip/BACI_HS92_V202501/compressed" \
      --prefix baci/hs6/

  # Normalize filenames to part-0000.csv.gz per year partition
  python s3_upload.py --normalize-parts

  # NEW: Upload only 2013 and 2015–2017
  python s3_upload.py --years 2013,2015-2017

  # NEW: Upload only within a range (inclusive)
  python s3_upload.py --min-year 2018 --max-year 2020
"""

from __future__ import annotations
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Iterator, Tuple, Dict, List, Optional, Set
import configparser

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

# ----------------------------
# Built-in Defaults (lowest precedence)
# ----------------------------
DEFAULT_BUCKET = "macromap-raws"
DEFAULT_REGION = "ap-southeast-2"
DEFAULT_SOURCE = Path(__file__).resolve().parent / "data" / "zip" / "BACI_HS92_V202501" / "compressed"
DEFAULT_PREFIX = "baci/hs6/"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("s3_upload.ini")

# Match BACI_HS92_Y<YEAR>_V....csv.gz
FILENAME_RX = re.compile(r"^BACI_HS92_Y(\d{4})_V.*\.csv\.gz$", re.IGNORECASE)

# ----------------------------
# Helpers
# ----------------------------

def iter_csvgz(folder: Path) -> Iterator[Tuple[int, Path]]:
    """Yield (year, path) for .csv.gz files that match naming convention."""
    for p in folder.glob("*.csv.gz"):
        m = FILENAME_RX.match(p.name)
        if m:
            yield int(m.group(1)), p


def human_size(bytes_: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(bytes_)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}EB"


class ProgressPrinter:
    """A lightweight progress callback for boto3 uploads."""
    def __init__(self, filename: str, total: int) -> None:
        self.filename = filename
        self.total = total
        self.seen = 0

    def __call__(self, bytes_amount: int) -> None:
        self.seen += bytes_amount
        pct = (self.seen / self.total) * 100 if self.total else 0
        print(f"  → {self.filename}: {self.seen}/{self.total} bytes ({pct:.1f}%)", end="\r", flush=True)
        if self.seen >= self.total:
            print()  # newline on completion


# ----------------------------
# Year filtering helpers (NEW)
# ----------------------------

def parse_years_expr(expr: str) -> List[int]:
    """Parse a comma/space-separated list of years and ranges into a sorted list of ints.
    Examples: "2013", "2013,2015-2017", "2019 2021-2023".
    """
    years: Set[int] = set()
    for token in re.split(r"[\s,]+", expr.strip()):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            try:
                start, end = int(a), int(b)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid year range: {token}")
            if start > end:
                start, end = end, start
            for y in range(start, end + 1):
                years.add(y)
        else:
            try:
                years.add(int(token))
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid year value: {token}")
    return sorted(years)


def filter_files_by_year(
    files: List[Tuple[int, Path]],
    include_years: Optional[Set[int]] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
) -> List[Tuple[int, Path]]:
    def allowed(y: int) -> bool:
        if include_years is not None and y not in include_years:
            return False
        if min_year is not None and y < min_year:
            return False
        if max_year is not None and y > max_year:
            return False
        return True

    return [(y, p) for (y, p) in files if allowed(y)]


# ----------------------------
# Config loading & precedence
# ----------------------------

def _to_bool(s: Optional[str], default: bool) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config(config_path: Path) -> dict:
    """Load config values from an INI file. Returns a flat dict of defaults.
    Missing file returns an empty dict.
    """
    if not config_path.exists():
        return {}

    cp = configparser.ConfigParser()
    cp.read(config_path, encoding="utf-8")

    cfg: Dict[str, Optional[str] | int | bool] = {}

    # [aws]
    bucket = cp.get("aws", "bucket", fallback=DEFAULT_BUCKET)
    region = cp.get("aws", "region", fallback=DEFAULT_REGION)
    profile = cp.get("aws", "profile", fallback=None)

    # [paths]
    source = cp.get("paths", "source", fallback=str(DEFAULT_SOURCE))
    prefix = cp.get("paths", "prefix", fallback=DEFAULT_PREFIX)

    # [upload]
    normalize_parts = _to_bool(cp.get("upload", "normalize_parts", fallback=None), False)
    overwrite = _to_bool(cp.get("upload", "overwrite", fallback=None), False)
    dry_run = _to_bool(cp.get("upload", "dry_run", fallback=None), False)
    storage_class = cp.get("upload", "storage_class", fallback=None) or None

    # [transfer]
    mth_mb = cp.getint("transfer", "multipart_threshold_mb", fallback=8)
    mcs_mb = cp.getint("transfer", "multipart_chunksize_mb", fallback=8)
    max_conc = cp.getint("transfer", "max_concurrency", fallback=8)
    use_threads = _to_bool(cp.get("transfer", "use_threads", fallback=None), True)

    # [headers]
    acl = cp.get("headers", "acl", fallback="private")
    content_type = cp.get("headers", "content_type", fallback="text/csv")
    content_encoding = cp.get("headers", "content_encoding", fallback="gzip")
    sse = cp.get("headers", "sse", fallback="AES256")

    # [filter]  (NEW)
    include_years_expr = cp.get("filter", "include_years", fallback="").strip()
    min_year = cp.get("filter", "min_year", fallback="").strip()
    max_year = cp.get("filter", "max_year", fallback="").strip()

    cfg.update({
        "bucket": bucket,
        "region": region,
        "profile": profile,
        "source": source,
        "prefix": prefix,
        "normalize_parts": normalize_parts,
        "overwrite": overwrite,
        "dry_run": dry_run,
        "storage_class": storage_class,
        "multipart_threshold_mb": mth_mb,
        "multipart_chunksize_mb": mcs_mb,
        "max_concurrency": max_conc,
        "use_threads": use_threads,
        "acl": acl,
        "content_type": content_type,
        "content_encoding": content_encoding,
        "sse": sse,
        # filters
        "include_years_expr": include_years_expr or None,
        "min_year": int(min_year) if min_year else None,
        "max_year": int(max_year) if max_year else None,
    })
    return cfg


# ----------------------------
# S3 Upload
# ----------------------------

def build_key(prefix: str, year: int, filename: str, normalize_parts: bool,
              year_to_part_counter: Dict[int, int]) -> str:
    """Build the S3 key under the required partitioning scheme.

    Default: baci/hs6/year=<YEAR>/<FILENAME>
    If normalize_parts=True: baci/hs6/year=<YEAR>/part-0000.csv.gz, etc.
    """
    prefix = prefix.rstrip("/")
    if normalize_parts:
        idx = year_to_part_counter.setdefault(year, 0)
        key_name = f"part-{idx:04d}.csv.gz"
        year_to_part_counter[year] = idx + 1
    else:
        key_name = filename
    return f"{prefix}/year={year}/{key_name}"


def upload_file(s3_client, file_path: Path, bucket: str, key: str, dry_run: bool,
                extra_args: Dict[str, str], config: TransferConfig) -> None:
    if dry_run:
        print(f"[DRY RUN] s3://{bucket}/{key}  <-  {file_path}")
        return

    size = file_path.stat().st_size
    print(f"Uploading {file_path.name} ({human_size(size)}) → s3://{bucket}/{key}")
    cb = ProgressPrinter(file_path.name, size)
    s3_client.upload_file(
        Filename=str(file_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs=extra_args,
        Config=config,
        Callback=cb,
    )


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    # Stage 1: parse only --config to know which file to read
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to INI config file")
    base_args, _ = base_parser.parse_known_args()

    # Load config (if present) and use it to seed argparse defaults
    cfg_path = Path(base_args.config)
    cfg = load_config(cfg_path)

    parser = argparse.ArgumentParser(description="Upload BACI .csv.gz files to S3 with config + safe defaults.",
                                     parents=[base_parser])

    # Arg defaults come from config (if provided), else built-ins
    parser.set_defaults(
        bucket=cfg.get("bucket", DEFAULT_BUCKET),
        region=cfg.get("region", DEFAULT_REGION),
        profile=cfg.get("profile", None),
        source=cfg.get("source", str(DEFAULT_SOURCE)),
        prefix=cfg.get("prefix", DEFAULT_PREFIX),
        normalize_parts=cfg.get("normalize_parts", False),
        overwrite=cfg.get("overwrite", False),
        dry_run=cfg.get("dry_run", False),
        storage_class=cfg.get("storage_class", None),
        multipart_threshold_mb=cfg.get("multipart_threshold_mb", 8),
        multipart_chunksize_mb=cfg.get("multipart_chunksize_mb", 8),
        max_concurrency=cfg.get("max_concurrency", 8),
        use_threads=cfg.get("use_threads", True),
        acl=cfg.get("acl", "private"),
        content_type=cfg.get("content_type", "text/csv"),
        content_encoding=cfg.get("content_encoding", "gzip"),
        sse=cfg.get("sse", "AES256"),
        # filters (NEW)
        include_years_expr=cfg.get("include_years_expr", None),
        min_year=cfg.get("min_year", None),
        max_year=cfg.get("max_year", None),
    )

    # Full CLI
    parser.add_argument("--bucket", help="Target S3 bucket")
    parser.add_argument("--region", help="AWS region")
    parser.add_argument("--profile", help="AWS profile name (uses default creds if omitted)")
    parser.add_argument("--source", help="Local folder with .csv.gz files")
    parser.add_argument("--prefix", help="Key prefix (e.g., baci/hs6/)")
    parser.add_argument("--normalize-parts", dest="normalize_parts", action="store_true",
                        help="Normalize filenames to part-0000.csv.gz per year")
    parser.add_argument("--no-normalize-parts", dest="normalize_parts", action="store_false",
                        help="Use original filenames (default)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing objects if present")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false",
                        help="Do not overwrite existing objects")
    parser.add_argument("--dry-run", action="store_true", help="Preview uploads without sending data")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                        help="Disable dry run (actually upload)")
    parser.add_argument("--storage-class", choices=[
        "STANDARD", "STANDARD_IA", "ONEZONE_IA", "INTELLIGENT_TIERING",
        "REDUCED_REDUNDANCY", "GLACIER", "DEEP_ARCHIVE",
    ], help="Optional S3 storage class")

    # Transfer tuning
    parser.add_argument("--multipart-threshold-mb", type=int, help="Multipart threshold in MB (default 8)")
    parser.add_argument("--multipart-chunksize-mb", type=int, help="Multipart chunk size in MB (default 8)")
    parser.add_argument("--max-concurrency", type=int, help="Max concurrent threads (default 8)")
    parser.add_argument("--use-threads", dest="use_threads", action="store_true", help="Use threaded uploads")
    parser.add_argument("--no-use-threads", dest="use_threads", action="store_false", help="Disable threads")

    # Headers / encryption
    parser.add_argument("--acl", help="Object ACL (default private)")
    parser.add_argument("--content-type", dest="content_type", help="Content-Type header")
    parser.add_argument("--content-encoding", dest="content_encoding", help="Content-Encoding header")
    parser.add_argument("--sse", help="ServerSideEncryption value (e.g., AES256)")

    # Year filters (NEW)
    parser.add_argument("--years", dest="include_years_expr",
                        help="Comma/space-separated years or ranges (e.g., '2013,2015-2017')")
    parser.add_argument("--min-year", dest="min_year", type=int, help="Minimum year to include (inclusive)")
    parser.add_argument("--max-year", dest="max_year", type=int, help="Maximum year to include (inclusive)")

    args = parser.parse_args()

    # Friendly banner
    if cfg_path.exists():
        print(f"Using config: {cfg_path}")
    else:
        print(f"No config found at: {cfg_path} (using built-ins/CLI)")

    src = Path(args.source)
    if not src.exists() or not src.is_dir():
        print(f"Source folder not found: {src}")
        candidates: List[Path] = []
        project_root = Path(__file__).resolve().parent
        try:
            candidates = [p for p in project_root.rglob("compressed") if p.is_dir()]
        except Exception:
            candidates = []
        if candidates:
            print("Did you mean one of these?")
            for p in candidates[:10]:
                print(f"  - {p}")
        sys.exit(1)

    # Create a session WITHOUT embedding credentials in code
    if args.profile:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
    else:
        session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")

    # Transfer configuration (multipart uploads, concurrency)
    config = TransferConfig(
        multipart_threshold=int(args.multipart_threshold_mb) * 1024 * 1024,
        multipart_chunksize=int(args.multipart_chunksize_mb) * 1024 * 1024,
        max_concurrency=int(args.max_concurrency),
        use_threads=bool(args.use_threads),
    )

    # Upload headers and encryption
    extra_args: Dict[str, str] = {
        "ACL": args.acl,
        "ContentType": args.content_type,
        "ContentEncoding": args.content_encoding,
        "ServerSideEncryption": args.sse,
    }
    if args.storage_class:
        extra_args["StorageClass"] = args.storage_class

    # Discover files
    files = sorted(iter_csvgz(src), key=lambda t: (t[0], t[1].name))
    if not files:
        print(f"No matching .csv.gz files found in: {src}")
        sys.exit(0)

    # Apply year filtering (NEW)
    include_years: Optional[Set[int]] = None
    if args.include_years_expr:
        include_years = set(parse_years_expr(args.include_years_expr))
        if not include_years:
            print("No valid years parsed from --years.")
            sys.exit(1)

    files = filter_files_by_year(files, include_years, args.min_year, args.max_year)

    if not files:
        print("No files remaining after year filtering. Nothing to do.")
        sys.exit(0)

    # Pretty print the plan
    years_present = sorted({y for y, _ in files})
    print(f"Found {len(files)} file(s) in: {src}")
    print(f"Years selected: {years_present}")
    print(f"Target: s3://{args.bucket}/{args.prefix} (region: {args.region})")
    print(f"Overwrite existing: {'YES' if args.overwrite else 'NO'} | Dry-run: {'YES' if args.dry_run else 'NO'}")

    year_to_part_counter: Dict[int, int] = {}
    uploaded, skipped = 0, 0

    for year, path in files:
        key = build_key(args.prefix, year, path.name, args.normalize_parts, year_to_part_counter)

        # Skip if exists and --overwrite not set
        if not args.overwrite:
            try:
                s3.head_object(Bucket=args.bucket, Key=key)
                print(f"SKIP (exists): s3://{args.bucket}/{key}")
                skipped += 1
                continue
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code")
                if code not in ("404", "NoSuchKey", "NotFound"):
                    raise

        upload_file(s3, path, args.bucket, key, args.dry_run, extra_args, config)
        uploaded += 1

    print(f"\nDone. Uploaded: {uploaded}, Skipped: {skipped}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
