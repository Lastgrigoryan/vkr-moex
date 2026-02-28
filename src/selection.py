import numpy as np
import pandas as pd


def _z(s: pd.Series) -> pd.Series:
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def score_and_select_top3(
    factors: pd.DataFrame,
    sectors: pd.DataFrame,
    w_mom: float = 0.35,
    w_stab: float = 0.30,
    w_dd: float = 0.20,
    w_liq: float = 0.15,
    require_positive: bool = True,   # оставим фильтр score>0, можно выключить
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ГЛОБАЛЬНОЕ скоринг-ранжирование:
    - Z-score по всему списку акций (не по отрасли)
    - Top-3 по каждой отрасли выбираются по глобальному score
    """
    df = factors.merge(sectors, on="secid", how="inner").copy()

    # базовые факторы
    df["mom"] = (df["mom_6m"] + df["mom_12m"]) / 2.0
    df["stab"] = -df["vol"]          # ниже вола -> выше стабильность
    df["dd_good"] = -df["mdd"]       # меньше просадка -> лучше (mdd отриц.)

    # ✅ Z-score по ВСЕМ акциям (глобально)
    df["z_mom"] = _z(df["mom"])
    df["z_stab"] = _z(df["stab"])
    df["z_dd"] = _z(df["dd_good"])
    df["z_liq"] = _z(df["liq"])

    # интегральный рейтинг
    df["score"] = (
        w_mom * df["z_mom"] +
        w_stab * df["z_stab"] +
        w_dd * df["z_dd"] +
        w_liq * df["z_liq"]
    )

    # сортировка по глобальному скору
    scored = df.sort_values(["score"], ascending=False).copy()

    # Top-3 по каждой отрасли — по глобальному скору
    base = scored[scored["score"] > 0] if require_positive else scored
    top3 = (
        base.sort_values(["sector", "score"], ascending=[True, False])
            .groupby("sector", as_index=False)
            .head(3)
            .copy()
    )
    top1 = (
        scored[scored["score"] > 0]
        .sort_values(["sector", "score"], ascending=[True, False])
        .groupby("sector", as_index=False)
        .head(1)
        .copy()
    )

    return scored, top3