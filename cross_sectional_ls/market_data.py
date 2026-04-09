from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha1
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


YAHOO_PRESETS: dict[str, list[str]] = {
    "us-sector-etfs": [
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    ],
    "dow30": [
        "AAPL",
        "AMGN",
        "AXP",
        "BA",
        "CAT",
        "CRM",
        "CSCO",
        "CVX",
        "DIS",
        "GS",
        "HD",
        "HON",
        "IBM",
        "JNJ",
        "JPM",
        "KO",
        "MCD",
        "MMM",
        "MRK",
        "MSFT",
        "NKE",
        "NVDA",
        "PG",
        "SHW",
        "TRV",
        "UNH",
        "V",
        "VZ",
        "WMT",
    ],
}


@dataclass(slots=True)
class MarketDataBundle:
    prices: pd.DataFrame
    description: str
    metadata: dict[str, Any]
    buy_blocked: pd.DataFrame | None = None
    sell_blocked: pd.DataFrame | None = None


class TushareHttpClient:
    """Minimal Tushare HTTP client based on the documented REST protocol."""

    def __init__(self, token: str, base_url: str = "http://api.tushare.pro") -> None:
        if not token:
            raise ValueError("Tushare token is required.")
        self.token = token
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def query(
        self,
        api_name: str,
        params: dict[str, Any] | None = None,
        fields: str = "",
    ) -> pd.DataFrame:
        payload = {
            "api_name": api_name,
            "token": self.token,
            "params": params or {},
            "fields": fields,
        }
        response = self.session.post(self.base_url, json=payload, timeout=60)
        response.raise_for_status()
        payload = response.json()

        if payload.get("code") != 0:
            raise RuntimeError(
                f"Tushare API {api_name} failed with code={payload.get('code')}: "
                f"{payload.get('msg')}"
            )

        data = payload.get("data") or {}
        items = data.get("items") or []
        columns = data.get("fields") or []
        return pd.DataFrame(items, columns=columns)


def parse_symbol_inputs(
    symbols: str | None = None,
    symbols_file: str | Path | None = None,
    preset: str | None = None,
    preset_map: dict[str, list[str]] | None = None,
) -> list[str]:
    resolved: list[str] = []
    available_presets = preset_map or {}

    if preset:
        if preset not in available_presets:
            supported = ", ".join(sorted(available_presets))
            raise ValueError(f"Unknown preset {preset!r}. Supported presets: {supported}")
        resolved.extend(available_presets[preset])

    if symbols_file:
        file_path = Path(symbols_file)
        file_symbols = [
            line.strip()
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        resolved.extend(file_symbols)

    if symbols:
        resolved.extend(part.strip() for part in symbols.split(",") if part.strip())

    deduplicated = list(dict.fromkeys(symbol.upper() for symbol in resolved))
    if not deduplicated:
        raise ValueError("No symbols were provided.")
    return deduplicated


def list_yahoo_presets() -> dict[str, list[str]]:
    return {key: value[:] for key, value in YAHOO_PRESETS.items()}


def download_yahoo_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = ".cache",
    refresh_cache: bool = False,
    max_workers: int = 8,
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one Yahoo ticker is required.")

    resolved_cache_dir = Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    start_timestamp = int(pd.Timestamp(start_date).timestamp())
    end_timestamp = int((pd.Timestamp(end_date) + pd.Timedelta(days=1)).timestamp())

    results: list[pd.Series] = []
    failures: list[str] = []
    worker_count = min(max_workers, max(1, len(tickers)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                _download_single_yahoo_symbol,
                ticker=ticker,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                cache_dir=resolved_cache_dir,
                refresh_cache=refresh_cache,
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                series = future.result()
            except Exception:
                failures.append(ticker)
                continue
            if series.empty:
                failures.append(ticker)
                continue
            results.append(series.rename(ticker))

    if not results:
        raise RuntimeError(
            "Yahoo download returned no usable series. "
            "This usually means the symbols were invalid or the remote endpoint blocked the request."
        )

    prices = pd.concat(results, axis=1).sort_index()
    prices = prices.loc[:, prices.notna().any(axis=0)]
    prices.index.name = "date"

    if failures and len(failures) == len(tickers):
        raise RuntimeError("Yahoo download failed for all requested symbols.")

    return prices


def build_tushare_ashare_universe(
    token: str,
    as_of_date: str,
    universe_size: int = 300,
    min_listed_days: int = 120,
    markets: list[str] | None = None,
    exchanges: list[str] | None = None,
    exclude_st: bool = True,
    cache_dir: str | Path = ".cache",
    refresh_cache: bool = False,
) -> pd.DataFrame:
    cache_path = _cache_path(
        cache_dir,
        "tushare_universe",
        {
            "as_of_date": as_of_date,
            "universe_size": universe_size,
            "min_listed_days": min_listed_days,
            "markets": markets or [],
            "exchanges": exchanges or [],
            "exclude_st": exclude_st,
        },
    )
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path)
        cached["list_date"] = pd.to_datetime(cached["list_date"])
        return cached

    client = TushareHttpClient(token)
    stock_basic = client.query(
        "stock_basic",
        params={"exchange": "", "list_status": "L"},
        fields="ts_code,symbol,name,market,exchange,list_date",
    )
    if stock_basic.empty:
        raise RuntimeError("Tushare stock_basic returned no active A-share symbols.")

    stock_basic["list_date"] = pd.to_datetime(stock_basic["list_date"], format="%Y%m%d")
    cutoff_date = pd.Timestamp(as_of_date) - pd.Timedelta(days=min_listed_days)
    filtered = stock_basic.loc[stock_basic["list_date"] <= cutoff_date].copy()

    if exclude_st:
        filtered = filtered.loc[
            ~filtered["name"].fillna("").str.upper().str.contains("ST", regex=False)
        ]
    if markets:
        filtered = filtered.loc[filtered["market"].isin(markets)]
    if exchanges:
        filtered = filtered.loc[filtered["exchange"].isin(exchanges)]

    last_trade_date = get_last_tushare_trade_date(token, as_of_date)
    liquidity_snapshot = client.query(
        "daily",
        params={"trade_date": last_trade_date},
        fields="ts_code,trade_date,amount,vol,close,pct_chg",
    )
    if liquidity_snapshot.empty:
        raise RuntimeError(
            f"Tushare daily returned no liquidity snapshot for {last_trade_date}."
        )

    universe = filtered.merge(liquidity_snapshot, on="ts_code", how="inner")
    universe = universe.loc[universe["amount"].fillna(0) > 0].copy()
    universe = universe.sort_values(["amount", "vol"], ascending=[False, False])
    if universe_size > 0:
        universe = universe.head(universe_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(cache_path, index=False)
    return universe


def download_tushare_bundle(
    token: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = ".cache",
    refresh_cache: bool = False,
    include_trade_constraints: bool = True,
) -> MarketDataBundle:
    if not tickers:
        raise ValueError("At least one Tushare symbol is required.")

    trading_dates = get_tushare_trade_dates(token, start_date, end_date)
    if trading_dates.empty:
        raise RuntimeError(
            f"No Tushare trade dates returned between {start_date} and {end_date}."
        )

    daily_data = download_tushare_daily_frame(
        token=token,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
    )
    prices = _build_adjusted_prices_from_tushare_returns(
        daily_data=daily_data,
        trading_dates=trading_dates,
    )
    prices.index.name = "date"

    buy_blocked = None
    sell_blocked = None
    metadata: dict[str, Any] = {
        "symbol_count": len(tickers),
        "trading_day_count": len(trading_dates),
    }

    if include_trade_constraints:
        limit_data = download_tushare_limit_frame(
            token=token,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
        )
        buy_blocked, sell_blocked, constraint_metadata = build_tushare_tradeability_masks(
            daily_data=daily_data,
            limit_data=limit_data,
            trading_dates=trading_dates,
        )
        metadata.update(constraint_metadata)

    return MarketDataBundle(
        prices=prices,
        description=(
            f"downloaded from Tushare daily API for {len(tickers)} A-share symbols "
            f"between {start_date} and {end_date}"
        ),
        metadata=metadata,
        buy_blocked=buy_blocked,
        sell_blocked=sell_blocked,
    )


def download_tushare_prices(
    token: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = ".cache",
    refresh_cache: bool = False,
) -> pd.DataFrame:
    return download_tushare_bundle(
        token=token,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        include_trade_constraints=False,
    ).prices


def download_tushare_daily_frame(
    token: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = ".cache",
    refresh_cache: bool = False,
) -> pd.DataFrame:
    cache_path = _cache_path(
        cache_dir,
        "tushare_daily_frame",
        {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path)
        cached["trade_date"] = pd.to_datetime(cached["trade_date"])
        return cached

    client = TushareHttpClient(token)
    trading_dates = get_tushare_trade_dates(token, start_date, end_date)
    batch_size = max(1, 5500 // max(1, len(trading_dates)))
    daily_frames: list[pd.DataFrame] = []

    for batch in _batched(tickers, batch_size):
        daily = client.query(
            "daily",
            params={
                "ts_code": ",".join(batch),
                "start_date": _compact_date(start_date),
                "end_date": _compact_date(end_date),
            },
            fields="ts_code,trade_date,open,high,low,close,pre_close,pct_chg,vol,amount",
        )
        if not daily.empty:
            daily_frames.append(daily)

    if not daily_frames:
        raise RuntimeError("Tushare daily returned no historical prices for the selected symbols.")

    daily_data = pd.concat(daily_frames, ignore_index=True)
    daily_data["trade_date"] = pd.to_datetime(daily_data["trade_date"], format="%Y%m%d")
    daily_data = daily_data.sort_values(["trade_date", "ts_code"])
    daily_data.to_csv(cache_path, index=False)
    return daily_data


def download_tushare_limit_frame(
    token: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    cache_dir: str | Path = ".cache",
    refresh_cache: bool = False,
) -> pd.DataFrame:
    cache_path = _cache_path(
        cache_dir,
        "tushare_stk_limit",
        {
            "tickers": tickers,
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path)
        cached["trade_date"] = pd.to_datetime(cached["trade_date"])
        return cached

    client = TushareHttpClient(token)
    limit_frames: list[pd.DataFrame] = []
    for ticker in tickers:
        limit_frame = client.query(
            "stk_limit",
            params={
                "ts_code": ticker,
                "start_date": _compact_date(start_date),
                "end_date": _compact_date(end_date),
            },
            fields="trade_date,ts_code,up_limit,down_limit",
        )
        if not limit_frame.empty:
            limit_frames.append(limit_frame)

    if not limit_frames:
        raise RuntimeError("Tushare stk_limit returned no price-limit data.")

    limit_data = pd.concat(limit_frames, ignore_index=True)
    limit_data["trade_date"] = pd.to_datetime(limit_data["trade_date"], format="%Y%m%d")
    limit_data = limit_data.sort_values(["trade_date", "ts_code"])
    limit_data.to_csv(cache_path, index=False)
    return limit_data


def build_tushare_tradeability_masks(
    daily_data: pd.DataFrame,
    limit_data: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    open_matrix = _pivot_market_field(daily_data, "open", trading_dates)
    close_matrix = _pivot_market_field(daily_data, "close", trading_dates)
    vol_matrix = _pivot_market_field(daily_data, "vol", trading_dates)
    amount_matrix = _pivot_market_field(daily_data, "amount", trading_dates)
    up_limit_matrix = _pivot_market_field(limit_data, "up_limit", trading_dates)
    down_limit_matrix = _pivot_market_field(limit_data, "down_limit", trading_dates)

    suspended = (
        close_matrix.isna()
        | vol_matrix.fillna(0.0).le(0.0)
        | amount_matrix.fillna(0.0).le(0.0)
    )
    buy_blocked = suspended | _near_price_level(open_matrix, up_limit_matrix)
    sell_blocked = suspended | _near_price_level(open_matrix, down_limit_matrix)

    metadata = {
        "average_buy_block_ratio": float(buy_blocked.mean().mean()),
        "average_sell_block_ratio": float(sell_blocked.mean().mean()),
    }
    return buy_blocked, sell_blocked, metadata


def get_tushare_trade_dates(token: str, start_date: str, end_date: str) -> pd.DatetimeIndex:
    client = TushareHttpClient(token)
    calendar = client.query(
        "trade_cal",
        params={
            "exchange": "SSE",
            "start_date": _compact_date(start_date),
            "end_date": _compact_date(end_date),
            "is_open": "1",
        },
        fields="exchange,cal_date,is_open,pretrade_date",
    )
    if calendar.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(calendar["cal_date"], format="%Y%m%d"))


def get_last_tushare_trade_date(token: str, as_of_date: str) -> str:
    start_date = (pd.Timestamp(as_of_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    trading_dates = get_tushare_trade_dates(token, start_date, as_of_date)
    if trading_dates.empty:
        raise RuntimeError(f"Could not resolve a Tushare trading day on or before {as_of_date}.")
    return trading_dates.max().strftime("%Y%m%d")


def _download_single_yahoo_symbol(
    ticker: str,
    start_timestamp: int,
    end_timestamp: int,
    cache_dir: Path,
    refresh_cache: bool,
) -> pd.Series:
    cache_path = _cache_path(
        cache_dir,
        "yahoo_symbol",
        {
            "ticker": ticker,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
        },
    )
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return cached.iloc[:, 0]

    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        params={
            "period1": start_timestamp,
            "period2": end_timestamp,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {ticker}: {error}")

    results = chart.get("result") or []
    if not results:
        return pd.Series(dtype=float)

    result = results[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        return pd.Series(dtype=float)

    adjclose_block = result.get("indicators", {}).get("adjclose", [])
    quote_block = result.get("indicators", {}).get("quote", [])
    prices = []
    if adjclose_block and "adjclose" in adjclose_block[0]:
        prices = adjclose_block[0]["adjclose"]
    elif quote_block and "close" in quote_block[0]:
        prices = quote_block[0]["close"]

    series = pd.Series(
        prices,
        index=pd.to_datetime(timestamps, unit="s").normalize(),
        dtype=float,
    ).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return series

    series.to_frame(name=ticker).to_csv(cache_path, index=True)
    return series


def _build_adjusted_prices_from_tushare_returns(
    daily_data: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    close_matrix = daily_data.pivot(index="trade_date", columns="ts_code", values="close")
    return_matrix = (
        daily_data.pivot(index="trade_date", columns="ts_code", values="pct_chg").div(100.0)
    )

    close_matrix = close_matrix.reindex(trading_dates)
    return_matrix = return_matrix.reindex(trading_dates)

    price_columns: dict[str, pd.Series] = {}
    for symbol in close_matrix.columns:
        raw_close = close_matrix[symbol]
        daily_return = return_matrix[symbol]
        first_valid_date = raw_close.first_valid_index()
        adjusted = pd.Series(np.nan, index=trading_dates, dtype=float)
        if first_valid_date is None:
            price_columns[symbol] = adjusted
            continue

        start_close = float(raw_close.loc[first_valid_date])
        growth = 1.0 + daily_return.loc[first_valid_date:].fillna(0.0)
        growth.iloc[0] = 1.0
        adjusted.loc[first_valid_date:] = start_close * growth.cumprod()
        price_columns[symbol] = adjusted

    return pd.DataFrame(price_columns, index=trading_dates)


def _pivot_market_field(
    frame: pd.DataFrame,
    field: str,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    matrix = frame.pivot(index="trade_date", columns="ts_code", values=field)
    return matrix.reindex(trading_dates).sort_index()


def _near_price_level(
    price_matrix: pd.DataFrame,
    level_matrix: pd.DataFrame,
    tolerance: float = 0.011,
) -> pd.DataFrame:
    return (price_matrix - level_matrix).abs().le(tolerance) & level_matrix.notna()


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _compact_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _cache_path(cache_dir: str | Path, prefix: str, payload: dict[str, Any]) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{prefix}_{digest}.csv"


def resolve_tushare_token(explicit_token: str | None = None) -> str:
    token = explicit_token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Tushare token is missing. Pass --tushare-token or set TUSHARE_TOKEN.")
    return token
