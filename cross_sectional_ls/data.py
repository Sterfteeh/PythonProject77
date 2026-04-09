from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_prices_from_csv(
    path: str | Path,
    date_col: str = "date",
    asset_col: str = "asset",
    price_col: str = "close",
) -> pd.DataFrame:
    """Load prices from either long-format or wide-format CSV."""

    csv_path = Path(path)
    raw = pd.read_csv(csv_path)
    lowered = {column.lower(): column for column in raw.columns}

    resolved_date = lowered.get(date_col.lower())
    resolved_asset = lowered.get(asset_col.lower())
    resolved_price = lowered.get(price_col.lower())

    if resolved_date and resolved_asset and resolved_price:
        frame = raw[[resolved_date, resolved_asset, resolved_price]].copy()
        frame[resolved_date] = pd.to_datetime(frame[resolved_date])
        prices = frame.pivot_table(
            index=resolved_date,
            columns=resolved_asset,
            values=resolved_price,
            aggfunc="last",
        )
    else:
        index_column = resolved_date or raw.columns[0]
        prices = raw.copy()
        prices[index_column] = pd.to_datetime(prices[index_column])
        prices = prices.set_index(index_column)

    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.replace([np.inf, -np.inf], np.nan)
    prices = prices.loc[:, prices.notna().any(axis=0)]
    prices.columns = [str(column) for column in prices.columns]

    if prices.empty:
        raise ValueError(f"No usable price data found in {csv_path}.")

    return prices


def generate_synthetic_prices(
    n_assets: int = 60,
    n_days: int = 756,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate synthetic daily prices with persistent cross-sectional structure."""

    if n_assets < 4:
        raise ValueError("n_assets must be at least 4.")
    if n_days < 40:
        raise ValueError("n_days must be at least 40.")

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)

    market_component = rng.normal(loc=0.0002, scale=0.0080, size=n_days)
    alpha_drift = rng.normal(loc=0.00005, scale=0.00025, size=n_assets)
    momentum_state = np.zeros((n_days, n_assets), dtype=float)

    for day in range(1, n_days):
        momentum_state[day] = (
            0.94 * momentum_state[day - 1]
            + rng.normal(loc=0.0, scale=0.0030, size=n_assets)
        )

    idiosyncratic_noise = rng.normal(loc=0.0, scale=0.0100, size=(n_days, n_assets))
    returns = (
        market_component[:, None]
        + alpha_drift[None, :]
        + 0.35 * momentum_state
        + idiosyncratic_noise
    )

    log_prices = np.log(start_price) + np.cumsum(returns, axis=0)
    prices = pd.DataFrame(
        np.exp(log_prices),
        index=dates,
        columns=[f"Asset_{asset_id:03d}" for asset_id in range(1, n_assets + 1)],
    )

    listing_offsets = rng.integers(0, max(2, n_days // 5), size=n_assets)
    for column, offset in zip(prices.columns, listing_offsets, strict=True):
        prices.loc[prices.index[:offset], column] = np.nan

    return prices
