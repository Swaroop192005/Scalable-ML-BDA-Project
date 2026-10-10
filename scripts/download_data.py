#!/usr/bin/env python3
"""Fetch the Online Retail dataset.

The dataset is the UCI *Online Retail* set (Chen, Sain & Hong, 2012):
541,909 line items from a UK-based online gift retailer, 1 Dec 2010 to
9 Dec 2011. Kaggle redistributes the identical file as "E-Commerce Data" /
"Online Retail Dataset"; this script pulls from UCI because that needs no
API token, and falls back to a Kaggle download if you have the CLI set up.

    python scripts/download_data.py              # UCI (default)
    python scripts/download_data.py --kaggle     # via the Kaggle CLI
    python scripts/download_data.py --sample     # also write the small sample

The .xlsx is converted to CSV on first download: the Excel parse costs ~40 s
and ~1 GB of peak memory, the CSV read about 1.5 s.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
KAGGLE_DATASET = "carrie1/ecommerce-data"


def download(url: str, target: Path, chunk: int = 1 << 20) -> Path:
    """Stream a URL to disk with a simple progress line."""
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urlopen(url) as response, target.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        seen = 0
        while block := response.read(chunk):
            handle.write(block)
            seen += len(block)
            if total:
                print(f"\r  {seen / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\r  {target.stat().st_size / 1e6:.1f} MB -> {target}")
    return target


def from_uci() -> Path:
    archive = RAW_DIR / "online_retail.zip"
    download(UCI_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".csv"))]
        if not names:
            raise RuntimeError(f"no data file inside {archive}")
        zf.extract(names[0], RAW_DIR)
    archive.unlink()
    return RAW_DIR / names[0]


def from_kaggle() -> Path:
    """Download via the Kaggle CLI, which must already be authenticated."""
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "the `kaggle` CLI is not installed. `pip install kaggle` and place "
            "your API token at ~/.kaggle/kaggle.json, or drop the --kaggle flag "
            "to use UCI instead."
        )
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR), "--unzip"],
        check=True,
    )
    candidates = sorted(RAW_DIR.glob("*.csv")) + sorted(RAW_DIR.glob("*.xlsx"))
    if not candidates:
        raise RuntimeError(f"Kaggle download produced no data file in {RAW_DIR}")
    return candidates[0]


def to_csv(source: Path) -> Path:
    """Convert the Excel workbook to CSV so later runs load in ~1.5 s."""
    target = RAW_DIR / "online_retail.csv"
    if source.suffix.lower() == ".csv":
        if source != target:
            shutil.copy(source, target)
        return target

    import pandas as pd

    print(f"converting {source.name} -> {target.name} (one-off, ~40 s)")
    frame = pd.read_excel(source, dtype={"InvoiceNo": str, "StockCode": str})
    frame.to_csv(target, index=False)
    print(f"  {len(frame):,} rows, {target.stat().st_size / 1e6:.1f} MB")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kaggle", action="store_true", help="use the Kaggle CLI instead of UCI")
    parser.add_argument("--sample", action="store_true",
                        help="also write data/sample/sample_retail.csv")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    csv = RAW_DIR / "online_retail.csv"
    if csv.exists() and not args.force:
        print(f"already present: {csv} ({csv.stat().st_size / 1e6:.1f} MB)")
        print("use --force to re-download")
    else:
        source = from_kaggle() if args.kaggle else from_uci()
        csv = to_csv(source)

    print("\nverifying...")
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    import pandas as pd

    head = pd.read_csv(csv, nrows=5)
    expected = {"InvoiceNo", "StockCode", "Description", "Quantity",
                "InvoiceDate", "UnitPrice", "CustomerID", "Country"}
    missing = expected - set(head.columns)
    if missing:
        print(f"WARNING: expected columns are missing: {sorted(missing)}")
        return 1
    print(f"  columns OK: {list(head.columns)}")

    if args.sample:
        from mba.cli import main as cli_main

        cli_main(["sample", "-q"])

    print("\nReady. Next:  mba run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
