from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from .system import BacktestResult


def save_backtest_outputs(result: BacktestResult, output_dir: str | Path) -> dict[str, Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "daily_results": target_dir / "daily_results.csv",
        "rebalance_weights": target_dir / "rebalance_weights.csv",
        "daily_weights": target_dir / "daily_weights.csv",
        "metrics": target_dir / "metrics.json",
        "equity_curve": target_dir / "equity_curve.png",
    }

    result.daily_results.to_csv(files["daily_results"], index_label="date")
    result.target_weights.to_csv(files["rebalance_weights"], index_label="date")
    result.daily_weights.to_csv(files["daily_weights"], index_label="date")
    files["metrics"].write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    _plot_equity_curve(result, files["equity_curve"])
    return files


def format_metrics(metrics: dict[str, float]) -> str:
    ordered_keys = [
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "calmar",
        "win_rate",
        "average_daily_turnover",
        "average_gross_exposure",
    ]
    lines = []
    for key in ordered_keys:
        value = metrics.get(key)
        if value is None:
            continue
        lines.append(f"{key:>24}: {value:>10.4f}")
    return "\n".join(lines)


def _plot_equity_curve(result: BacktestResult, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(result.daily_results.index, result.daily_results["equity"], linewidth=2.0)
    axis.set_title("Cross-Sectional Long-Short Equity Curve")
    axis.set_xlabel("Date")
    axis.set_ylabel("Equity")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
