"""Reading the raw transaction file and persisting processed artifacts.

The UCI *Online Retail* dataset ships as a 23 MB ``.xlsx``. Parsing that with
openpyxl costs ~40 s and ~1 GB of peak memory, so the loader transparently
caches a CSV/Parquet copy next to it and prefers the cheap format on every
later run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from mba.config import Config

log = logging.getLogger(__name__)

#: Column dtypes forced at read time. InvoiceNo/StockCode are *not* numeric —
#: cancellations look like 'C536379' and codes like '85123A'.
RAW_DTYPES: dict[str, str] = {
    "InvoiceNo": "string",
    "StockCode": "string",
    "Description": "string",
    "Country": "string",
}


def resolve_source(cfg: Config) -> Path:
    """Return the first existing raw-data path from the configured candidates.

    Raises
    ------
    FileNotFoundError
        When neither the primary path nor any fallback exists. The message
        points at ``scripts/download_data.py``, which fetches the dataset.
    """
    candidates: list[Path] = [cfg.path("paths.raw_file")]
    for extra in cfg.get("paths.raw_file_fallbacks", []) or []:
        p = Path(extra)
        candidates.append(p if p.is_absolute() else cfg.path("paths.raw_dir").parent.parent / p)

    for candidate in candidates:
        if candidate.exists():
            log.info("using raw source: %s", candidate)
            return candidate

    listed = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "No raw transaction file found. Looked for:\n  "
        f"{listed}\n\nRun `python scripts/download_data.py` to fetch the "
        "UCI Online Retail dataset."
    )


def _read_any(path: Path, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    """Dispatch on file extension and return a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(
            path,
            dtype={k: v for k, v in RAW_DTYPES.items() if usecols is None or k in usecols},
            usecols=list(usecols) if usecols else None,
            parse_dates=["InvoiceDate"] if usecols is None or "InvoiceDate" in usecols else None,
            encoding="utf-8",
            on_bad_lines="warn",
        )
    if suffix == ".parquet":
        return pd.read_parquet(path, columns=list(usecols) if usecols else None)
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, dtype={"InvoiceNo": str, "StockCode": str})
        return frame[list(usecols)] if usecols else frame
    raise ValueError(f"unsupported raw file format: {path.suffix!r} ({path})")


def load_raw(
    cfg: Config,
    path: str | Path | None = None,
    nrows: int | None = None,
    cache_csv: bool = True,
) -> pd.DataFrame:
    """Load the raw transaction table.

    Parameters
    ----------
    cfg:
        Loaded configuration (supplies paths and the column schema).
    path:
        Override the configured source file.
    nrows:
        Read only the first *n* rows — handy for smoke tests.
    cache_csv:
        When the source is an Excel workbook, write a CSV twin beside it so
        subsequent runs skip the expensive parse.
    """
    source = Path(path) if path is not None else resolve_source(cfg)
    log.info("loading transactions from %s", source)

    frame = _read_any(source)

    if cache_csv and source.suffix.lower() in {".xlsx", ".xls"}:
        twin = source.with_name("online_retail.csv")
        if not twin.exists():
            log.info("caching CSV twin at %s (speeds up later runs)", twin)
            frame.to_csv(twin, index=False)

    # Normalise the date column no matter which backend produced it.
    date_col = cfg.get("schema.date_col", "InvoiceDate")
    if date_col in frame.columns and not pd.api.types.is_datetime64_any_dtype(frame[date_col]):
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")

    if nrows is not None:
        frame = frame.head(nrows).copy()

    log.info("loaded %s rows x %s columns", f"{len(frame):,}", frame.shape[1])
    return frame


def iter_raw_chunks(
    cfg: Config,
    chunk_size: int,
    path: str | Path | None = None,
) -> Iterable[pd.DataFrame]:
    """Yield the raw table in row chunks, for out-of-core processing.

    Only CSV and Parquet sources stream; an Excel workbook is read once and
    then sliced, because openpyxl cannot seek.
    """
    source = Path(path) if path is not None else resolve_source(cfg)
    suffix = source.suffix.lower()

    if suffix in {".csv", ".txt"}:
        reader = pd.read_csv(
            source,
            dtype=RAW_DTYPES,
            parse_dates=["InvoiceDate"],
            chunksize=chunk_size,
            encoding="utf-8",
            on_bad_lines="warn",
        )
        yield from reader
        return

    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.ParquetFile(source)
        for batch in table.iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()
        return

    frame = load_raw(cfg, path=source)
    for start in range(0, len(frame), chunk_size):
        yield frame.iloc[start : start + chunk_size].copy()


def save_processed(
    frame: pd.DataFrame,
    cfg: Config,
    stem: str = "transactions_clean",
    formats: Sequence[str] = ("parquet", "csv"),
) -> dict[str, Path]:
    """Persist a processed frame as Parquet (compact) and/or CSV (portable)."""
    out_dir = cfg.path("paths.processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for fmt in formats:
        target = out_dir / f"{stem}.{fmt}"
        if fmt == "parquet":
            try:
                frame.to_parquet(target, index=False, compression="snappy")
            except (ImportError, ValueError) as exc:  # pragma: no cover - env dependent
                log.warning("parquet write skipped (%s)", exc)
                continue
        elif fmt == "csv":
            frame.to_csv(target, index=False)
        else:
            raise ValueError(f"unsupported output format: {fmt!r}")
        written[fmt] = target
        log.info("wrote %s (%.1f MB)", target, target.stat().st_size / 1e6)

    return written
