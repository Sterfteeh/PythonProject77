# Cross-Sectional Trading System

This repository contains a research-oriented cross-sectional trading system in Python.

It supports:

- synthetic demo data
- local CSV files
- real US market data from Yahoo
- real A-share market data from Tushare

## What is implemented

The strategy is a cross-sectional momentum model:

1. compute each asset's momentum signal with a lookback window and a recent-days skip window
2. rank assets on each rebalance date
3. build either:
   - a `long_short` portfolio: long the strongest bucket and short the weakest bucket
   - a `long_only` portfolio: hold only the strongest bucket
4. shift signals by one trading day to avoid look-ahead bias
5. simulate daily execution and deduct transaction costs

For A-shares, the system also supports more realistic execution constraints:

- suspended or zero-liquidity days block trading
- stocks opening at the daily upper limit block buy orders
- stocks opening at the daily lower limit block sell orders
- blocked orders keep the old position instead of forcing the target weight

## Project structure

- `cross_sectional_ls/data.py`: local CSV loading and synthetic data generation
- `cross_sectional_ls/market_data.py`: Yahoo and Tushare data downloaders, A-share universe selection, tradeability masks
- `cross_sectional_ls/system.py`: factor calculation, portfolio construction, execution simulation, and backtest engine
- `cross_sectional_ls/metrics.py`: performance statistics
- `cross_sectional_ls/reporting.py`: CSV, JSON, and PNG outputs
- `main.py`: command-line entry point
- `tests/test_system.py`: unit tests

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Quick start

Run the demo:

```powershell
python main.py
```

Run with a local CSV:

```powershell
python main.py --data-source csv --prices path\to\prices.csv
```

Run with Yahoo data:

```powershell
python main.py --data-source yahoo --universe-preset dow30 --start-date 2021-01-01 --end-date 2025-12-31
```

Run with A-share data from Tushare:

```powershell
set TUSHARE_TOKEN=your_token_here
python main.py --data-source tushare --start-date 2021-01-01 --end-date 2025-12-31 --tushare-universe-size 200 --tushare-markets main,gem
```

## Important CLI options

- `--data-source demo|csv|yahoo|tushare`
- `--portfolio-mode auto|long_short|long_only`
- `--start-date YYYY-MM-DD`
- `--end-date YYYY-MM-DD`
- `--symbols AAPL,MSFT,...`
- `--symbols-file path\to\symbols.txt`
- `--universe-preset us-sector-etfs|dow30`
- `--lookback`
- `--skip-recent`
- `--rebalance-frequency`
- `--long-quantile`
- `--short-quantile`
- `--gross-leverage`
- `--min-assets`
- `--transaction-cost-bps`
- `--cache-dir`
- `--refresh-cache`
- `--disable-trade-constraints`

## A-share workflow

When `--data-source tushare` is used:

- if you do not pass `--symbols` or `--symbols-file`, the system builds an automatic universe
- the automatic universe keeps listed stocks only
- ST stocks are excluded by default
- the universe applies a minimum listing-age filter
- the universe sorts by the latest daily turnover amount and keeps the top `N`
- `portfolio_mode=auto` becomes `long_only`, which is the more realistic default for A-shares

## Supported CSV formats

Wide format:

```csv
date,A,B,C
2024-01-01,10,20,30
2024-01-02,10.2,19.8,30.5
```

Long format:

```csv
date,asset,close
2024-01-01,A,10
2024-01-01,B,20
2024-01-02,A,10.2
2024-01-02,B,19.8
```

## Outputs

Each run writes files under the selected output directory:

- `daily_results.csv`
- `rebalance_weights.csv`
- `desired_daily_weights.csv`
- `daily_weights.csv`
- `metrics.json`
- `equity_curve.png`
- `used_prices.csv`
- `run_metadata.json`

## Notes on realism

This is still a research backtest, not a full production execution engine.

What is modeled:

- one-day signal lag
- transaction costs
- A-share open-limit trading blocks
- A-share zero-liquidity / suspension-style blocks

What is not fully modeled:

- borrow fees and financing costs
- exact intraday fill probability at price limits
- dynamic shortable lists for A-shares
- delisting and corporate-action edge cases beyond return-based price reconstruction
- sector-neutral or market-cap-neutral optimization

## Testing

Run unit tests:

```powershell
python -m unittest discover -s tests -v
```
