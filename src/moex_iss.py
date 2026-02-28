import requests
import pandas as pd

ISS_BASE = "https://iss.moex.com/iss"


def fetch_candles(secid: str, date_from: str, date_till: str, interval: int = 24) -> pd.DataFrame:
    """
    Daily candles for shares: /engines/stock/markets/shares/securities/{SECID}/candles.json
    interval=24 часто соответствует дневным свечам.
    """
    url = f"{ISS_BASE}/engines/stock/markets/shares/securities/{secid}/candles.json"
    params = {
        "from": date_from,
        "till": date_till,
        "interval": interval,
        "iss.meta": "off",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()

    candles = js.get("candles", {})
    cols = candles.get("columns", [])
    data = candles.get("data", [])
    df = pd.DataFrame(data, columns=cols)

    if df.empty:
        return df

    # date column
    if "begin" in df.columns:
        df["date"] = pd.to_datetime(df["begin"]).dt.date
    elif "end" in df.columns:
        df["date"] = pd.to_datetime(df["end"]).dt.date
    else:
        df["date"] = pd.to_datetime(df.iloc[:, 0]).dt.date

    keep = ["date"]
    for c in ["open", "high", "low", "close", "value", "volume", "numtrades"]:
        if c in df.columns:
            keep.append(c)

    df = df[keep].sort_values("date").dropna(subset=["close"])
    df["secid"] = secid.upper()
    return df