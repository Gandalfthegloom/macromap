"""
csv-to-csv.gz
simple script converting csv files to csv.gz to reduce AWS S3 costs 
pls fund my cloud bills for education :")

Merged + user-friendly version:
- Prioritizes user's directory path variables
- Uses os and pandas for compression
- Auto-detects BACI_HS92_Y<year> file pattern starting from 1995
- Finds highest year present and converts all matching CSVs up to that year
- Saves compressed files into a 'compressed' subfolder
- Skips non-matching files and ones already compressed
- Interactive prompts in main():
  • Shows the configured folder and lets the user override it
  • Validates naming format and shows non-matching examples
  • Confirms before proceeding and tells user where the compressed files will be saved
"""

import os
import re
from typing import List, Tuple
import pandas as pd

# Gets filepath (from root) of this script's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "zip", "BACI_HS92_V202501")

# Basic config for csv file scan
PREFIX = r"BACI_HS92_Y"
YEAR_MIN = 1995
DRY_RUN = False

pattern = re.compile(rf"^{re.escape(PREFIX)}(\d{{4}})_V.*\.csv$", re.IGNORECASE)


def list_candidate_files(folder: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if not os.path.isfile(full) or not name.lower().endswith('.csv'):
            continue
        m = pattern.match(name)
        if not m:
            continue
        year = int(m.group(1))
        if year >= YEAR_MIN:
            out.append((year, full))
    return out


def compress_csv_to_gz(src_csv: str, output_folder: str) -> str:
    os.makedirs(output_folder, exist_ok=True)
    dst_gz = os.path.join(output_folder, os.path.basename(src_csv) + ".gz")
    if os.path.exists(dst_gz):
        print(f"SKIP (exists): {os.path.basename(dst_gz)}")
        return dst_gz

    print(f"Compressing (pandas): {os.path.basename(src_csv)} -> {os.path.basename(dst_gz)}")
    if not DRY_RUN:
        try:
            df = pd.read_csv(src_csv)
            df.to_csv(dst_gz, index=False, compression='gzip')
        except Exception as e:
            print(f"ERROR: Failed to compress {src_csv} with pandas: {e}")
    return dst_gz


def check_naming_and_report(folder: str, candidates: List[Tuple[int, str]]) -> None:
    all_csvs = [p for p in os.listdir(folder) if os.path.isfile(os.path.join(folder, p)) and p.lower().endswith('.csv')]
    matched = {os.path.basename(p) for _, p in candidates}
    non_matching = [name for name in all_csvs if name not in matched]

    if non_matching:
        print("\nWARNING: Some .csv files do not match the required naming pattern and will be ignored:")
        preview = non_matching[:10]
        for nm in preview:
            print(f"  - {nm}")
        if len(non_matching) > 10:
            print(f"  ... and {len(non_matching) - 10} more")
        print("\nRequired format: BACI_HS92_Y<YEAR>_V<...>.csv  e.g., BACI_HS92_Y1995_V202501.csv")
        print("Regex used:", pattern.pattern)
    else:
        print("All CSV files match the required naming pattern. ✅")


def prompt_for_folder(default_folder: str) -> str:
    print("\nConfigured folder for raw CSVs:")
    print(f"  {default_folder}")
    ans = input("Are the raw files already in this path? [Y/n]: ").strip().lower()
    folder = default_folder
    if ans in {'n', 'no'}:
        entered = input("Enter the full path to the folder containing your BACI CSV files: ").strip()
        if entered:
            folder = os.path.expanduser(entered)
    return folder


def main():
    folder = prompt_for_folder(file_path)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    candidates = list_candidate_files(folder)
    if not candidates:
        print("No matching CSVs found.")
        print("Required format: BACI_HS92_Y<YEAR>_V<...>.csv  e.g., BACI_HS92_Y1995_V202501.csv")
        return

    check_naming_and_report(folder, candidates)

    max_year = max(y for y, _ in candidates)
    compressed_folder = os.path.join(folder, "compressed")
    print(f"\nDetected highest year in folder: {max_year}")
    print(f"Compressed files will be saved in: {compressed_folder}")

    print(f"This will compress all matching files from {YEAR_MIN}..{max_year} (inclusive).")
    go = input("Proceed? [Y/n]: ").strip().lower()
    if go in {'n', 'no'}:
        print("Aborted by user.")
        return

    targets = sorted([(y, p) for y, p in candidates if YEAR_MIN <= y <= max_year], key=lambda t: (t[0], t[1]))
    print(f"\nFound {len(targets)} file(s) to process in: {folder}\n")

    for year, src in targets:
        compress_csv_to_gz(src, compressed_folder)

    print("\nDone.")


if __name__ == "__main__":
    main()
