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
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

UCI_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
KAGGLE_DATASET = "carrie1/ecommerce-data"


class DownloadFailed(RuntimeError):
    """Raised when the dataset could not be fetched, with advice attached."""


def _diagnose(error: Exception) -> str:
    """Turn a urllib failure into something a person can act on.

    A bare ``gaierror`` traceback tells the reader nothing about what to do
    next, and the three common causes need three different fixes.
    """
    if isinstance(error, HTTPError):
        return (
            f"The server answered {error.code} ({error.reason}). UCI's archive "
            "is occasionally down or rate-limiting; wait a minute and retry, or "
            "use --kaggle, or download it by hand (see below)."
        )
    reason = getattr(error, "reason", error)
    text = str(reason).lower()
    if isinstance(reason, socket.gaierror) or "nodename nor servname" in text or "name or service" in text:
        return (
            "DNS could not resolve the host, so the machine could not reach the "
            "internet at all for this request. Check your connection, VPN, or "
            "whether a campus/corporate network is blocking it. On a managed "
            "network, downloading by hand in a browser usually works."
        )
    if "certificate" in text or "ssl" in text:
        return (
            "TLS certificate verification failed. On macOS this normally means "
            "the python.org installer's certificates were never installed — run:\n"
            "    /Applications/Python\\ 3.x/Install\\ Certificates.command\n"
            "substituting your Python version."
        )
    if "timed out" in text or isinstance(error, TimeoutError):
        return "The connection timed out. UCI can be slow; retry, or download by hand."
    return "Check your network connection, then retry."


MANUAL_INSTRUCTIONS = """\
To download it by hand instead:

  1. Open  https://archive.ics.uci.edu/dataset/352/online+retail
     (or the Kaggle mirror: https://www.kaggle.com/datasets/carrie1/ecommerce-data)
  2. Save the file into  data/raw/
  3. Run  python scripts/download_data.py  again — it will find the file and
     just do the CSV conversion.

Or skip the dataset entirely: everything runs on the committed sample with

  mba run -i data/sample/sample_retail.csv
"""


def download(url: str, target: Path, chunk: int = 1 << 20,
             retries: int = 3, timeout: int = 60) -> Path:
    """Stream a URL to disk, retrying on transient network failures.

    UCI's archive is intermittently slow or unreachable, so a single attempt
    is not enough to call it a real failure. Retries back off; a partial file
    is removed so a later run never reads a truncated download as valid.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        suffix = f"  (attempt {attempt}/{retries})" if attempt > 1 else ""
        print(f"downloading {url}{suffix}")
        try:
            request = Request(url, headers={"User-Agent": "scalable-mba/1.0"})
            with urlopen(request, timeout=timeout) as response, target.open("wb") as handle:
                total = int(response.headers.get("Content-Length") or 0)
                seen = 0
                while block := response.read(chunk):
                    handle.write(block)
                    seen += len(block)
                    if total:
                        print(f"\r  {seen / 1e6:6.1f} / {total / 1e6:.1f} MB",
                              end="", flush=True)
            print(f"\r  {target.stat().st_size / 1e6:.1f} MB -> {target}")
            return target
        except (URLError, HTTPError, TimeoutError, OSError) as exc:
            last_error = exc
            target.unlink(missing_ok=True)   # never leave a partial file behind
            if attempt < retries:
                wait = 2 ** attempt
                print(f"\n  failed: {exc}\n  retrying in {wait}s...")
                time.sleep(wait)

    assert last_error is not None
    raise DownloadFailed(
        f"Could not download {url} after {retries} attempts.\n\n"
        f"{_diagnose(last_error)}\n\n{MANUAL_INSTRUCTIONS}"
    )


def find_existing() -> Path | None:
    """A dataset file already dropped into data/raw by hand."""
    for pattern in ("*.xlsx", "*.csv", "*.zip"):
        for candidate in sorted(RAW_DIR.glob(pattern)):
            if candidate.stat().st_size > 1_000_000:   # ignore stray small files
                return candidate
    return None


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
        existing = None if args.force else find_existing()
        if existing is not None:
            print(f"found a file already in {RAW_DIR}: {existing.name}")
            if existing.suffix.lower() == ".zip":
                with zipfile.ZipFile(existing) as zf:
                    names = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".csv"))]
                    if not names:
                        print(f"no data file inside {existing}", file=sys.stderr)
                        return 1
                    zf.extract(names[0], RAW_DIR)
                existing = RAW_DIR / names[0]
            source = existing
        else:
            try:
                source = from_kaggle() if args.kaggle else from_uci()
            except (DownloadFailed, RuntimeError) as exc:
                print(f"\n{exc}", file=sys.stderr)
                return 1
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
