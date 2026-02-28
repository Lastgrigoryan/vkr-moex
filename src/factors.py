import numpy as np
import pandas as pd


def compute_factors_one(candles: pd.DataFrame) -> pd.DataFrame:
    """
    candles: date, close, (value/volume)
    returns: 1-row df with factors:
      mom_6m, mom_12m, vol, mdd, liq
    """
    df = candles.sort_values("date").copy()
    df["ret"] = df["close"].pct_change()

    def momentum(k: int) -> float:
        if len(df) <= k:
            return np.nan
        return float(df["close"].iloc[-1] / df["close"].iloc[-1 - k] - 1.0)

    mom_6m = momentum(126)
    mom_12m = momentum(252)

    vol = float(df["ret"].std(ddof=1)) if df["ret"].notna().sum() > 2 else np.nan

    close = df["close"].astype(float).to_numpy()
    if len(close) >= 2:
        peak = np.maximum.accumulate(close)
        dd = close / peak - 1.0
        mdd = float(np.min(dd))  # negative
    else:
        mdd = np.nan

    # liquidity proxy: log(mean value or volume, last 60 days)
    liq = np.nan
    if "value" in df.columns and df["value"].notna().any():
        liq = float(np.log(df["value"].tail(60).mean()))
    elif "volume" in df.columns and df["volume"].notna().any():
        liq = float(np.log(df["volume"].tail(60).mean()))

    return pd.DataFrame([{
        "secid": str(df["secid"].iloc[0]),
        "mom_6m": mom_6m,
        "mom_12m": mom_12m,
        "vol": vol,
        "mdd": mdd,
        "liq": liq,
    }])