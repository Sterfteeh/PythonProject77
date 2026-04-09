from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import pandas as pd

from cross_sectional_ls.data import generate_synthetic_prices, load_prices_from_csv
from cross_sectional_ls.market_data import (
    MarketDataBundle,
    build_tushare_ashare_universe,
    download_tushare_bundle,
    download_yahoo_prices,
    list_yahoo_presets,
    parse_symbol_inputs,
    resolve_tushare_token,
)
from cross_sectional_ls.reporting import format_metrics, save_backtest_outputs
from cross_sectional_ls.system import (
    CrossSectionalLongShortSystem,
    ExecutionConstraints,
    StrategyConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-sectional long-short trading system."
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="demo",
        choices=["demo", "csv", "yahoo", "tushare"],
        help="Data source used for the backtest.",
    )
    parser.add_argument("--prices", type=str, default=None, help="Path to a CSV file.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output folder.")
    parser.add_argument("--cache-dir", type=str, default=".cache", help="Local cache folder.")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached downloads and fetch data again.",
    )
    parser.add_argument("--date-col", type=str, default="date", help="Date column name.")
    parser.add_argument("--asset-col", type=str, default="asset", help="Asset column name.")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name.")
    parser.add_argument("--start-date", type=str, default=None, help="Backtest start date.")
    parser.add_argument("--end-date", type=str, default=None, help="Backtest end date.")
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols or tickers.",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="Text file with one symbol per line.",
    )
    parser.add_argument(
        "--universe-preset",
        type=str,
        default=None,
        help=f"Preset universe for Yahoo. Supported: {', '.join(sorted(list_yahoo_presets()))}",
    )
    parser.add_argument(
        "--portfolio-mode",
        type=str,
        default="auto",
        choices=["auto", "long_short", "long_only"],
        help="Portfolio construction mode. 'auto' uses long_only for Tushare and long_short otherwise.",
    )

    parser.add_argument("--lookback", type=int, default=60, help="Momentum lookback window.")
    parser.add_argument(
        "--skip-recent",
        type=int,
        default=5,
        help="Recent days skipped to reduce short-term reversal contamination.",
    )
    parser.add_argument(
        "--rebalance-frequency",
        type=str,
        default="ME",
        help="Pandas resample alias, such as ME, W-FRI, or QE.",
    )
    parser.add_argument("--long-quantile", type=float, default=0.2, help="Long bucket size.")
    parser.add_argument("--short-quantile", type=float, default=0.2, help="Short bucket size.")
    parser.add_argument("--gross-leverage", type=float, default=1.0, help="Gross exposure.")
    parser.add_argument("--min-assets", type=int, default=20, help="Minimum assets to trade.")
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=10.0,
        help="One-way transaction cost in basis points per unit turnover.",
    )

    parser.add_argument("--demo-assets", type=int, default=60, help="Synthetic asset count.")
    parser.add_argument("--demo-days", type=int, default=756, help="Synthetic day count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for demo data.")

    parser.add_argument(
        "--tushare-token",
        type=str,
        default=None,
        help="Tushare token. If omitted, TUSHARE_TOKEN is used.",
    )
    parser.add_argument(
        "--tushare-universe-size",
        type=int,
        default=300,
        help="Universe size for auto-selected A-share stocks.",
    )
    parser.add_argument(
        "--tushare-min-listed-days",
        type=int,
        default=120,
        help="Minimum listed days for A-share universe selection.",
    )
    parser.add_argument(
        "--tushare-markets",
        type=str,
        default="",
        help="Comma-separated A-share markets, such as main,gem,star,bse.",
    )
    parser.add_argument(
        "--tushare-exchanges",
        type=str,
        default="",
        help="Comma-separated Tushare exchange codes such as SSE,SZSE,BSE.",
    )
    parser.add_argument(
        "--keep-st",
        action="store_true",
        help="Keep ST stocks in the Tushare auto universe.",
    )
    parser.add_argument(
        "--disable-trade-constraints",
        action="store_true",
        help="Disable A-share open-limit / suspension execution constraints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    end_date = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    start_date = args.start_date or (
        pd.Timestamp(end_date) - pd.DateOffset(years=3)
    ).strftime("%Y-%m-%d")

    market_data = load_market_data(args, start_date, end_date)
    portfolio_mode = resolve_portfolio_mode(args.portfolio_mode, args.data_source)
    effective_min_assets = min(args.min_assets, market_data.prices.shape[1])

    config = StrategyConfig(
        lookback_days=args.lookback,
        skip_recent_days=args.skip_recent,
        rebalance_frequency=args.rebalance_frequency,
        long_quantile=args.long_quantile,
        short_quantile=args.short_quantile,
        gross_leverage=args.gross_leverage,
        min_assets=effective_min_assets,
        transaction_cost_bps=args.transaction_cost_bps,
        portfolio_mode=portfolio_mode,
    )
    system = CrossSectionalLongShortSystem(config)
    constraints = build_execution_constraints(market_data)
    result = system.run(market_data.prices, constraints=constraints)

    output_dir = Path(args.output_dir)
    output_paths = save_backtest_outputs(result, output_dir)
    used_prices_path = output_dir / "used_prices.csv"
    result.prices.to_csv(used_prices_path, index_label="date")
    output_paths["used_prices"] = used_prices_path

    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "data": market_data.metadata,
                "strategy": asdict(config),
                "environment": {"cwd": os.getcwd()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_paths["run_metadata"] = metadata_path

    print("Cross-sectional long-short backtest completed.")
    print(f"Data source: {market_data.description}")
    print(f"Trading days: {len(result.prices)}")
    print(f"Asset count: {result.prices.shape[1]}")
    print(f"Portfolio mode: {config.portfolio_mode}")
    print(f"Minimum assets to trade: {config.min_assets}")
    print()
    print(format_metrics(result.metrics))
    print()
    print("Output files:")
    for name, path in output_paths.items():
        print(f"  {name:>20}: {path.resolve()}")


def load_market_data(
    args: argparse.Namespace,
    start_date: str,
    end_date: str,
) -> MarketDataBundle:
    if args.data_source == "csv":
        if not args.prices:
            raise ValueError("--data-source csv requires --prices.")
        prices = load_prices_from_csv(
            args.prices,
            date_col=args.date_col,
            asset_col=args.asset_col,
            price_col=args.price_col,
        )
        return MarketDataBundle(
            prices=prices,
            description=f"loaded from {Path(args.prices).resolve()}",
            metadata={
                "data_source": "csv",
                "path": str(Path(args.prices).resolve()),
            },
        )

    if args.data_source == "demo":
        prices = generate_synthetic_prices(
            n_assets=args.demo_assets,
            n_days=args.demo_days,
            seed=args.seed,
        )
        return MarketDataBundle(
            prices=prices,
            description="generated from the built-in synthetic data generator",
            metadata={
                "data_source": "demo",
                "demo_assets": args.demo_assets,
                "demo_days": args.demo_days,
                "seed": args.seed,
            },
        )

    if args.data_source == "yahoo":
        symbols = parse_symbol_inputs(
            symbols=args.symbols,
            symbols_file=args.symbols_file,
            preset=args.universe_preset or "us-sector-etfs",
            preset_map=list_yahoo_presets(),
        )
        prices = download_yahoo_prices(
            tickers=symbols,
            start_date=start_date,
            end_date=end_date,
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
        )
        return MarketDataBundle(
            prices=prices,
            description=(
                f"downloaded from Yahoo chart API for {len(symbols)} tickers "
                f"between {start_date} and {end_date}"
            ),
            metadata={
                "data_source": "yahoo",
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

    token = resolve_tushare_token(args.tushare_token)
    markets = normalize_tushare_markets(args.tushare_markets)
    exchanges = normalize_csv_list(args.tushare_exchanges)

    if args.symbols or args.symbols_file:
        symbols = parse_symbol_inputs(
            symbols=args.symbols,
            symbols_file=args.symbols_file,
        )
        universe = None
    else:
        universe = build_tushare_ashare_universe(
            token=token,
            as_of_date=end_date,
            universe_size=args.tushare_universe_size,
            min_listed_days=args.tushare_min_listed_days,
            markets=markets,
            exchanges=exchanges,
            exclude_st=not args.keep_st,
            cache_dir=args.cache_dir,
            refresh_cache=args.refresh_cache,
        )
        symbols = universe["ts_code"].tolist()

    market_data = download_tushare_bundle(
        token=token,
        tickers=symbols,
        start_date=start_date,
        end_date=end_date,
        cache_dir=args.cache_dir,
        refresh_cache=args.refresh_cache,
        include_trade_constraints=not args.disable_trade_constraints,
    )
    market_data.metadata.update(
        {
            "data_source": "tushare",
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "markets": markets,
            "exchanges": exchanges,
            "trade_constraints_enabled": not args.disable_trade_constraints,
        }
    )
    if universe is not None:
        market_data.metadata["auto_universe_size"] = len(universe)
        market_data.metadata["auto_universe_preview"] = universe[
            ["ts_code", "name", "market", "amount"]
        ].head(20).to_dict(orient="records")
    return market_data


def build_execution_constraints(
    market_data: MarketDataBundle,
) -> ExecutionConstraints | None:
    if market_data.buy_blocked is None and market_data.sell_blocked is None:
        return None
    return ExecutionConstraints(
        buy_blocked=market_data.buy_blocked,
        sell_blocked=market_data.sell_blocked,
    )


def resolve_portfolio_mode(portfolio_mode: str, data_source: str) -> str:
    if portfolio_mode != "auto":
        return portfolio_mode
    if data_source == "tushare":
        return "long_only"
    return "long_short"


def normalize_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_tushare_markets(value: str | None) -> list[str]:
    if not value:
        return []
    aliases = {
        "main": "\u4e3b\u677f",
        "gem": "\u521b\u4e1a\u677f",
        "star": "\u79d1\u521b\u677f",
        "bse": "\u5317\u4ea4\u6240",
        "cdr": "CDR",
    }
    resolved = []
    for item in normalize_csv_list(value):
        resolved.append(aliases.get(item.lower(), item))
    return resolved


if __name__ == "__main__":
    main()
