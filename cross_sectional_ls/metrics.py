from __future__ import annotations

import math

import pandas as pd


def _annualized_return(returns: pd.Series, annualization_factor: int) -> float:
    if returns.empty:
        return 0.0

    equity = (1.0 + returns).cumprod()
    total_return = equity.iloc[-1] - 1.0
    years = len(returns) / annualization_factor

    if years <= 0:
        return 0.0
    if equity.iloc[-1] <= 0:
        return -1.0

    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0

    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def compute_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    gross_exposure: pd.Series,
    annualization_factor: int = 252,
) -> dict[str, float]:
    returns = returns.fillna(0.0)
    turnover = turnover.fillna(0.0)
    gross_exposure = gross_exposure.fillna(0.0)

    annual_return = _annualized_return(returns, annualization_factor)
    annual_volatility = float(returns.std(ddof=0) * math.sqrt(annualization_factor))
    sharpe = 0.0
    if annual_volatility > 0:
        sharpe = float(
            returns.mean() / returns.std(ddof=0) * math.sqrt(annualization_factor)
        )

    max_drawdown = _max_drawdown(returns)
    calmar = 0.0
    if max_drawdown < 0:
        calmar = float(annual_return / abs(max_drawdown))

    equity = (1.0 + returns).cumprod()
    metrics = {
        "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": float((returns > 0).mean()) if not returns.empty else 0.0,
        "average_daily_turnover": float(turnover.mean()) if not turnover.empty else 0.0,
        "average_gross_exposure": (
            float(gross_exposure.mean()) if not gross_exposure.empty else 0.0
        ),
    }
    return metrics
