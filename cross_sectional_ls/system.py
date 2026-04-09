from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import compute_metrics


@dataclass(slots=True)
class StrategyConfig:
    lookback_days: int = 60
    skip_recent_days: int = 5
    rebalance_frequency: str = "ME"
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    gross_leverage: float = 1.0
    min_assets: int = 20
    transaction_cost_bps: float = 10.0
    annualization_factor: int = 252


@dataclass(slots=True)
class BacktestResult:
    prices: pd.DataFrame
    factor: pd.DataFrame
    target_weights: pd.DataFrame
    daily_weights: pd.DataFrame
    daily_results: pd.DataFrame
    metrics: dict[str, float]


class CrossSectionalLongShortSystem:
    """Research-oriented cross-sectional long-short engine."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        self._validate_config()

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        clean_prices = self.prepare_prices(prices)
        factor = self.compute_momentum_factor(clean_prices)
        target_weights = self.build_target_weights(factor)
        daily_weights = self.expand_weights_to_daily(target_weights, clean_prices.index)
        asset_returns = clean_prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

        gross_returns = (daily_weights * asset_returns).sum(axis=1)
        turnover = daily_weights.diff().abs().sum(axis=1).fillna(daily_weights.abs().sum(axis=1))
        transaction_cost = turnover * (self.config.transaction_cost_bps / 10_000.0)
        net_returns = gross_returns - transaction_cost
        gross_exposure = daily_weights.abs().sum(axis=1)

        daily_results = pd.DataFrame(
            {
                "gross_return": gross_returns,
                "transaction_cost": transaction_cost,
                "net_return": net_returns,
                "turnover": turnover,
                "gross_exposure": gross_exposure,
                "net_exposure": daily_weights.sum(axis=1),
                "active_positions": (daily_weights != 0.0).sum(axis=1),
            }
        )
        daily_results["equity"] = (1.0 + daily_results["net_return"]).cumprod()

        metrics = compute_metrics(
            returns=daily_results["net_return"],
            turnover=daily_results["turnover"],
            gross_exposure=daily_results["gross_exposure"],
            annualization_factor=self.config.annualization_factor,
        )

        return BacktestResult(
            prices=clean_prices,
            factor=factor,
            target_weights=target_weights,
            daily_weights=daily_weights,
            daily_results=daily_results,
            metrics=metrics,
        )

    def prepare_prices(self, prices: pd.DataFrame) -> pd.DataFrame:
        clean = prices.copy()
        clean.index = pd.to_datetime(clean.index)
        clean = clean.sort_index()
        clean = clean.loc[~clean.index.duplicated(keep="last")]
        clean = clean.apply(pd.to_numeric, errors="coerce")
        clean = clean.replace([np.inf, -np.inf], np.nan)

        required_history = min(
            len(clean),
            self.config.lookback_days + self.config.skip_recent_days + 2,
        )
        clean = clean.loc[:, clean.notna().sum(axis=0) >= required_history]

        if clean.empty:
            raise ValueError("Price matrix is empty after cleaning.")

        return clean

    def compute_momentum_factor(self, prices: pd.DataFrame) -> pd.DataFrame:
        reference_price = prices.shift(self.config.skip_recent_days)
        base_price = prices.shift(self.config.lookback_days + self.config.skip_recent_days)
        factor = reference_price / base_price - 1.0
        return factor.replace([np.inf, -np.inf], np.nan)

    def build_target_weights(self, factor: pd.DataFrame) -> pd.DataFrame:
        rebalance_dates = self._get_rebalance_dates(factor.index)
        weights = pd.DataFrame(0.0, index=rebalance_dates, columns=factor.columns)

        for rebalance_date in rebalance_dates:
            signal = factor.loc[rebalance_date].dropna()
            if len(signal) < self.config.min_assets:
                continue
            row_weights = self._signal_to_weights(signal)
            weights.loc[rebalance_date, row_weights.index] = row_weights

        return weights

    def expand_weights_to_daily(
        self, target_weights: pd.DataFrame, trading_index: pd.Index
    ) -> pd.DataFrame:
        scheduled_weights = target_weights.reindex(trading_index).ffill().fillna(0.0)
        effective_weights = scheduled_weights.shift(1).fillna(0.0)
        return effective_weights

    def _get_rebalance_dates(self, index: pd.Index) -> pd.DatetimeIndex:
        marker = pd.Series(pd.DatetimeIndex(index), index=pd.DatetimeIndex(index))
        rebalance_frequency = self._normalize_frequency(self.config.rebalance_frequency)
        rebalance_dates = marker.groupby(pd.Grouper(freq=rebalance_frequency)).last().dropna()
        return pd.DatetimeIndex(rebalance_dates.to_list())

    def _signal_to_weights(self, signal: pd.Series) -> pd.Series:
        n_assets = len(signal)
        n_long = max(1, int(np.floor(n_assets * self.config.long_quantile)))
        n_short = max(1, int(np.floor(n_assets * self.config.short_quantile)))

        if n_long + n_short > n_assets:
            raise ValueError("Long and short buckets overlap; reduce quantiles.")

        ranked = signal.sort_values(ascending=False)
        long_assets = ranked.head(n_long).index
        short_assets = ranked.tail(n_short).index

        weights = pd.Series(0.0, index=signal.index)
        weights.loc[long_assets] = self.config.gross_leverage / 2.0 / n_long
        weights.loc[short_assets] = -self.config.gross_leverage / 2.0 / n_short
        return weights

    def _validate_config(self) -> None:
        config = self.config
        if config.lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")
        if config.skip_recent_days < 0:
            raise ValueError("skip_recent_days must be non-negative.")
        if not 0 < config.long_quantile < 1:
            raise ValueError("long_quantile must be in (0, 1).")
        if not 0 < config.short_quantile < 1:
            raise ValueError("short_quantile must be in (0, 1).")
        if config.long_quantile + config.short_quantile >= 1:
            raise ValueError("long_quantile + short_quantile must be < 1.")
        if config.gross_leverage <= 0:
            raise ValueError("gross_leverage must be positive.")
        if config.min_assets < 2:
            raise ValueError("min_assets must be at least 2.")
        if config.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be non-negative.")

    @staticmethod
    def _normalize_frequency(frequency: str) -> str:
        legacy_aliases = {
            "M": "ME",
            "Q": "QE",
            "Y": "YE",
            "A": "YE",
            "BM": "BME",
        }
        upper_frequency = frequency.upper()
        return legacy_aliases.get(upper_frequency, frequency)
