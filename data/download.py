import sys
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW, START_DATE, END_DATE,
    YAHOO_TICKERS, FRED_SERIES, FRED_API_KEY,
)

MANIFEST_PATH = DATA_RAW / "manifest.json"

# Series that require API key (ICE BofA proprietary)
_NEEDS_API_KEY = {"BAMLC0A1CAAA", "BAMLC0A4CBBB", "BAMLMKV2Y5Y"}


def _sha256(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        df.to_csv().encode("utf-8")
    ).hexdigest()[:16]


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest: dict):
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


# ── Yahoo Finance ────────────────────────────────────────────

def download_yahoo(name: str, ticker: str,
                   start: str = START_DATE,
                   end: str = END_DATE) -> tuple:
    path = DATA_RAW / f"{name}.parquet"

    print(f"  Downloading {name} ({ticker}) from Yahoo Finance...")
    df = yf.download(
        ticker, start=start, end=end,
        auto_adjust=True, progress=False,
    )

    if df.empty:
        print(f"  WARNING: No data returned for {name} ({ticker})")
        return df, "empty"

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df.to_parquet(path)
    sha = _sha256(df)

    print(f"  ✓ {name}: {len(df)} rows, "
          f"{df.index[0].date()} to {df.index[-1].date()}, "
          f"sha256={sha}")
    return df, sha


def download_all_yahoo() -> dict:
    results = {}
    for name, ticker in YAHOO_TICKERS.items():
        try:
            df, sha = download_yahoo(name, ticker)
            results[name] = {
                "rows": len(df),
                "start": str(df.index[0].date()),
                "end": str(df.index[-1].date()),
                "sha256": sha,
                "source": "yahoo",
                "ticker": ticker,
            }
        except Exception as e:
            print(f"  ERROR downloading {name}: {e}")
            results[name] = {"error": str(e)}
    return results


# ── FRED ─────────────────────────────────────────────────────

def _get_fred_client():
    """Create fredapi client with API key from config."""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError(
            "fredapi is required for ICE BofA series. "
            "Install with: pip install fredapi"
        )

    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY not set in config.py. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    return Fred(api_key=FRED_API_KEY)


def download_fred_public_csv(series_id: str,
                             start: str = START_DATE,
                             end: str = END_DATE) -> tuple:
    """
    Download from FRED public CSV endpoint (no API key).
    Only works for non-proprietary series (DGS10, etc.).
    """
    print(f"  Downloading {series_id} from FRED (public CSV)...")
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )

    df = pd.read_csv(url, index_col=0, parse_dates=True)
    df.columns = [series_id]
    df.index.name = "Date"
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    n_missing = df[series_id].isna().sum()
    df[series_id] = df[series_id].ffill()

    path = DATA_RAW / f"{series_id}.parquet"
    df.to_parquet(path)
    sha = _sha256(df)

    print(f"  ✓ {series_id}: {len(df)} rows, "
          f"{n_missing} missing filled, sha256={sha}")
    return df, sha


def download_fred_api(series_id: str,
                      start: str = START_DATE,
                      end: str = END_DATE) -> tuple:
    """
    Download from FRED using fredapi (requires API key).
    Required for ICE BofA proprietary series (BAML*).
    """
    print(f"  Downloading {series_id} from FRED (API key)...")
    fred = _get_fred_client()

    s = fred.get_series(
        series_id,
        observation_start=start,
        observation_end=end,
    )

    df = pd.DataFrame({series_id: s})
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    n_missing = df[series_id].isna().sum()
    df[series_id] = df[series_id].ffill()

    path = DATA_RAW / f"{series_id}.parquet"
    df.to_parquet(path)
    sha = _sha256(df)

    print(f"  ✓ {series_id}: {len(df)} rows, "
          f"{n_missing} missing filled, sha256={sha}")
    return df, sha


def download_all_fred() -> dict:
    """Download all FRED series, using API key for BAML* series."""
    results = {}

    for name, series_id in FRED_SERIES.items():
        try:
            if series_id in _NEEDS_API_KEY:
                # ICE BofA series: must use API key
                df, sha = download_fred_api(series_id)
            else:
                # Public series: try CSV first, fallback to API
                try:
                    df, sha = download_fred_public_csv(series_id)
                except Exception:
                    print(f"  Public CSV failed, trying API...")
                    df, sha = download_fred_api(series_id)

            results[name] = {
                "rows": len(df),
                "sha256": sha,
                "source": "fred",
                "series_id": series_id,
            }
        except Exception as e:
            print(f"  ERROR downloading {series_id}: {e}")
            results[name] = {"error": str(e)}

            # Special fallback for MOVE only
            if series_id == "BAMLMKV2Y5Y":
                print("  NOTE: MOVE will be proxied from DGS10 vol "
                      "in clean.py")
                results[name] = {"status": "proxy_from_dgs10"}

    return results


# ── Master Download ──────────────────────────────────────────

def download_all():
    print("=" * 60)
    print("DATA INGESTION")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("=" * 60)

    manifest = {
        "download_time": datetime.now(timezone.utc).isoformat(),
        "period": {"start": START_DATE, "end": END_DATE},
        "datasets": {},
    }

    print("\n── Yahoo Finance ──")
    yahoo_results = download_all_yahoo()
    manifest["datasets"].update(yahoo_results)

    print("\n── FRED ──")
    fred_results = download_all_fred()
    manifest["datasets"].update(fred_results)

    _save_manifest(manifest)
    print(f"\n✓ Manifest saved to {MANIFEST_PATH}")
    print(f"  Total datasets: {len(manifest['datasets'])}")

    return manifest


def verify_checksums():
    manifest = _load_manifest()
    all_ok = True
    for name, info in manifest.get("datasets", {}).items():
        if "error" in info or "status" in info:
            continue
        path = DATA_RAW / f"{name}.parquet"
        if not path.exists():
            print(f"  MISSING: {name}")
            all_ok = False
            continue
        df = pd.read_parquet(path)
        sha = _sha256(df)
        expected = info.get("sha256", "")
        if sha == expected:
            print(f"  ✓ {name}: OK")
        else:
            print(f"  ✗ {name}: MISMATCH "
                  f"(expected {expected}, got {sha})")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    if "--check" in sys.argv:
        print("Verifying data integrity...\n")
        ok = verify_checksums()
        sys.exit(0 if ok else 1)
    else:
        download_all()
