"""
s3_upload.py
Upload BACI .csv.gz files to S3 using safe creds (AWS profile / env),
matching the MacroMap key structure:
  raw/baci/hs6/year=<YYYY>/<FILENAME>

Highlights
- No hard-coded credentials. Use --profile or your default AWS config.
- Sets ContentType=text/csv and ContentEncoding=gzip.
- Private objects with SSE-S3 (AES256).
- Skips existing keys unless --overwrite is passed.
- Dry-run mode to preview without uploading.
- Flexible source discovery with a sensible default; override via --source.
- Optional normalization to strict part names: part-0000.csv.gz, etc.

Examples
  # Preview what would be uploaded (uses default bucket/region/source)
  python upload_csvgz_to_s3.py --dry-run

  # Upload with a specific AWS profile and overwrite existing keys
  python upload_csvgz_to_s3.py --profile macromap --overwrite

  # Custom source folder and a custom prefix
  python upload_csvgz_to_s3.py --source "D:/world trade/macromap/data/zip/BACI_HS92_V202501/compressed" \
      --prefix raw/baci/hs6/

  # Normalize filenames to part-0000.csv.gz per year partition
  python upload_csvgz_to_s3.py --normalize-parts
"""

from __future__ import annotations
import os
import re
import sys
import argparse
from pathlib import Path
from typing import Iterator, Tuple, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

# ----------------------------
# Configuration & Defaults
# ----------------------------
# Default bucket/region per your environment. Safe to publish :D
DEFAULT_BUCKET = "macromap-prod-ap-southeast-2"
DEFAULT_REGION = "ap-southeast-2"

# Default local folder where the compression script writes outputs.
# You can override with --source if your layout differs.
DEFAULT_SOURCE = Path(__file__).resolve().parent / "data" / "zip" / "BACI_HS92_V202501" / "compressed"

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
        # Print on the same line (works on most terminals)
        print(f"  → {self.filename}: {self.seen}/{self.total} bytes ({pct:.1f}%)", end="\r", flush=True)
        if self.seen >= self.total:
            print()  # newline on completion


# ----------------------------
# S3 Upload
# ----------------------------

def build_key(prefix: str, year: int, filename: str, normalize_parts: bool,
              year_to_part_counter: Dict[int, int]) -> str:
    """Build the S3 key under the required partitioning scheme.

    Default: raw/baci/hs6/year=<YEAR>/<FILENAME>
    If normalize_parts=True: raw/baci/hs6/year=<YEAR>/part-0000.csv.gz, etc.
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
    parser = argparse.ArgumentParser(description="Upload BACI .csv.gz files to S3 with safe defaults.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="Target S3 bucket")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--profile", default=None, help="AWS profile name (uses default creds if omitted)")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Local folder with .csv.gz files")
    parser.add_argument("--prefix", default="raw/baci/hs6/", help="Key prefix (default: raw/baci/hs6/)")
    parser.add_argument("--normalize-parts", action="store_true",
                        help="Normalize filenames to part-0000.csv.gz per year")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing objects if present")
    parser.add_argument("--dry-run", action="store_true", help="Preview uploads without sending data")
    parser.add_argument("--storage-class", default=None,
                        choices=[
                            "STANDARD", "STANDARD_IA", "ONEZONE_IA", "INTELLIGENT_TIERING",
                            "REDUCED_REDUNDANCY", "GLACIER", "DEEP_ARCHIVE",
                        ], help="Optional S3 storage class")

    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists() or not src.is_dir():
        print(f"Source folder not found: {src}")
        # Soft assist: try to locate a nearby 'compressed' folder as a fallback
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

    # Create a session **WITHOUT** embedding credentials in code
    # Credentials are sourced from: --profile, env vars, or default config chain
    if args.profile:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
    else:
        session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")

    # Transfer configuration (multipart uploads, concurrency)
    config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,      # 8 MB
        multipart_chunksize=8 * 1024 * 1024,
        max_concurrency=8,
        use_threads=True,
    )

    # Upload headers and encryption; bucket also enforces SSE-S3 by policy
    extra_args: Dict[str, str] = {
        "ACL": "private",
        "ContentType": "text/csv",
        "ContentEncoding": "gzip",
        "ServerSideEncryption": "AES256",
    }
    if args.storage_class:
        extra_args["StorageClass"] = args.storage_class

    # Discover files
    files = sorted(iter_csvgz(src), key=lambda t: (t[0], t[1].name))
    if not files:
        print(f"No matching .csv.gz files found in: {src}")
        sys.exit(0)

    print(f"Found {len(files)} file(s) in: {src}")
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
