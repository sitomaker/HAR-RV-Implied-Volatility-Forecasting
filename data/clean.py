import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_RAW, DATA_PROC


def load_raw(name: str) -> pd.DataFrame:
    """Load a raw parquet file."""
    path = DATA_RAW / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m data.download' first."
        )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df


def build_master_frame() -> pd.DataFrame:
    """
    Build master DataFrame aligned to SPY trading calendar.
    
    Series mapping (matches LaTeX Table 1):
        AAA yield:  BAMLC0A1CAAA  (ICE BofA AAA, daily)
        BBB yield:  BAMLC0A4CBBB  (ICE BofA BBB, daily)
        Spread:     BBB - AAA
        MOVE:       BAMLMKV2Y5Y or proxy from DGS10
    """
    # ── Load SPY (anchor calendar) ──
    spy = load_raw("SPY")
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    spy = spy[["Open", "High", "Low", "Close", "Volume"]]
    spy.columns = [f"SPY_{c}" for c in spy.columns]
    spy["SPY_ret"] = np.log(
        spy["SPY_Close"] / spy["SPY_Close"].shift(1)
    )

    calendar = spy.index

    # ── Load VIX family ──
    vix_frames = {}
    for name in ["VIX", "VIX9D", "VIX3M", "VIX6M"]:
        try:
            df = load_raw(name)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Close" in df.columns:
                vix_frames[name] = df["Close"]
            else:
                vix_frames[name] = df.iloc[:, 0]
        except FileNotFoundError:
            print(f"  WARNING: {name} not found, will be NaN")
            vix_frames[name] = pd.Series(np.nan, index=calendar, name=name)

    # ── Load FRED series ──
    fred_frames = {}

    # DGS10
    try:
        df = load_raw("DGS10")
        fred_frames["DGS10"] = df.iloc[:, 0]
    except FileNotFoundError:
        print("  WARNING: DGS10 not found")
        fred_frames["DGS10"] = pd.Series(np.nan, index=calendar, name="DGS10")

    # AAA corporate yield (ICE BofA)
    try:
        df = load_raw("BAMLC0A1CAAA")
        fred_frames["AAA"] = df.iloc[:, 0]
        print("  ✓ AAA: loaded from BAMLC0A1CAAA (ICE BofA AAA, daily)")
    except FileNotFoundError:
        print("  WARNING: BAMLC0A1CAAA not found — AAA will be NaN")
        fred_frames["AAA"] = pd.Series(np.nan, index=calendar, name="AAA")

    # BBB corporate yield (ICE BofA)
    try:
        df = load_raw("BAMLC0A4CBBB")
        fred_frames["BBB"] = df.iloc[:, 0]
        print("  ✓ BBB: loaded from BAMLC0A4CBBB (ICE BofA BBB, daily)")
    except FileNotFoundError:
        print("  WARNING: BAMLC0A4CBBB not found — BBB will be NaN")
        fred_frames["BBB"] = pd.Series(np.nan, index=calendar, name="BBB")

    # MOVE Index: BAMLMKV2Y5Y if available, else proxy
    try:
        df = load_raw("BAMLMKV2Y5Y")
        fred_frames["MOVE"] = df.iloc[:, 0]
        print("  ✓ MOVE: loaded from BAMLMKV2Y5Y (ICE BofA MOVE Index)")
    except FileNotFoundError:
        if "DGS10" in fred_frames:
            dy10 = fred_frames["DGS10"].diff(1)
            fred_frames["MOVE"] = dy10.rolling(22).std() * np.sqrt(252)
            print("  NOTE: MOVE proxied by 22-day rolling vol of ΔDGS10 "
                  "(BAMLMKV2Y5Y not available)")
        else:
            fred_frames["MOVE"] = pd.Series(np.nan, index=calendar, name="MOVE")
            print("  WARNING: MOVE could not be computed")

    # ── Load VXX ──
    try:
        vxx = load_raw("VXX")
        if isinstance(vxx.columns, pd.MultiIndex):
            vxx.columns = vxx.columns.get_level_values(0)
        vxx_ret = np.log(vxx["Close"] / vxx["Close"].shift(1))
        vxx_ret.name = "VXX_ret"
    except FileNotFoundError:
        print("  WARNING: VXX not found.")
        vxx_ret = pd.Series(np.nan, index=calendar, name="VXX_ret")

    # ── Assemble master frame ──
    master = spy.copy()

    for name, series in vix_frames.items():
        master[name] = series.reindex(calendar).ffill()

    if "DGS10" in fred_frames:
        master["DGS10"] = fred_frames["DGS10"].reindex(calendar).ffill()

    # Credit spread: BBB - AAA (LaTeX eq. credit_spread)
    bbb = fred_frames["BBB"].reindex(calendar).ffill()
    aaa = fred_frames["AAA"].reindex(calendar).ffill()
    master["spread"] = bbb - aaa
    print("  NOTE: spread = BBB - AAA (ICE BofA credit quality spread)")

    # MOVE
    master["MOVE"] = fred_frames["MOVE"].reindex(calendar).ffill()

    # VXX
    master["VXX_ret"] = vxx_ret.reindex(calendar)

    # ── Quality checks ──
    print(f"\n── Master Frame Summary ──")
    print(f"  Date range: {master.index[0].date()} "
          f"to {master.index[-1].date()}")
    print(f"  Trading days: {len(master)}")
    print(f"  Columns: {list(master.columns)}")
    print(f"\n  Missing values:")
    missing = master.isna().sum()
    for col in missing[missing > 0].index:
        pct = 100 * missing[col] / len(master)
        print(f"    {col}: {missing[col]} ({pct:.1f}%)")

    out_path = DATA_PROC / "master.parquet"
    master.to_parquet(out_path)
    print(f"\n  ✓ Saved to {out_path}")

    return master


if __name__ == "__main__":
    build_master_frame()
