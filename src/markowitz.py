import numpy as np
import pandas as pd
from scipy.optimize import minimize


def build_returns_matrix(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует candles_df в матрицу доходностей:
    index = date, columns = secid, values = daily returns
    """
    px = candles_df.pivot(index="date", columns="secid", values="close").sort_index()
    rets = px.pct_change().dropna(how="all")
    return rets


def markowitz_min_variance(returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимизация дисперсии портфеля.
    Ограничения:
    - сумма весов = 1
    - short selling запрещён: 0 <= w_i <= 1
    """
    # удаляем бумаги с слишком большим количеством пропусков
    returns_df = returns_df.dropna(axis=1, thresh=max(20, int(len(returns_df) * 0.7)))
    returns_df = returns_df.dropna(axis=0, how="any")

    if returns_df.empty or returns_df.shape[1] < 2:
        raise ValueError("Недостаточно данных для расчета портфеля Марковица")

    mu = returns_df.mean().values
    cov = returns_df.cov().values
    tickers = returns_df.columns.tolist()
    n = len(tickers)

    def portfolio_variance(w):
        return float(np.dot(w.T, np.dot(cov, w)))

    x0 = np.array([1.0 / n] * n)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    ]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        portfolio_variance,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if not result.success:
        raise ValueError(f"Оптимизация Марковица не сошлась: {result.message}")

    w = result.x
    port_return = float(np.dot(w, mu))
    port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))

    out = pd.DataFrame({
        "secid": tickers,
        "weight_markowitz": w,
        "exp_return": mu,
    })

    # индивидуальная волатильность акций
    out["volatility"] = returns_df.std().values

    # убираем почти нулевые веса для аккуратности
    out = out[out["weight_markowitz"] > 1e-4].copy()
    out = out.sort_values("weight_markowitz", ascending=False).reset_index(drop=True)

    summary = pd.DataFrame([{
        "portfolio_expected_return": port_return,
        "portfolio_volatility": port_vol,
        "n_assets": len(out)
    }])

    return out, summary