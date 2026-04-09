# Cross-Sectional Long-Short Trading System

This project implements a research-oriented cross-sectional long-short system in Python.

It is designed to be simple enough to understand end-to-end, but structured enough to extend into a real research workflow.

## What the system does

1. Load a daily close-price matrix from CSV, generate a synthetic demo dataset, or download real data from Yahoo / Tushare.
2. Compute a cross-sectional momentum factor:
   - signal date `t`
   - momentum = `price[t - skip_recent_days] / price[t - lookback_days - skip_recent_days] - 1`
3. Rebalance on a fixed schedule such as monthly or weekly.
4. Rank assets by the factor on each rebalance date.
5. Go long the strongest bucket and short the weakest bucket.
6. Keep the portfolio dollar-neutral:
   - long book = `+0.5`
   - short book = `-0.5`
   - gross exposure = `1.0`
7. Shift positions by one trading day to avoid look-ahead bias.
8. Deduct transaction costs based on turnover.
9. Output equity curve, weights, daily returns, and summary metrics.

## Project structure

- `cross_sectional_ls/data.py`: price loading and demo data generation.
- `cross_sectional_ls/market_data.py`: Yahoo / Tushare real-market downloaders and universe builders.
- `cross_sectional_ls/system.py`: factor, ranking, weighting, and backtest engine.
- `cross_sectional_ls/metrics.py`: performance statistics.
- `cross_sectional_ls/reporting.py`: CSV / JSON / PNG output.
- `main.py`: command-line entry point.
- `tests/test_system.py`: smoke tests.

## How to run

Run the demo system:

```bash
python main.py
```

Run with your own CSV:

```bash
python main.py --data-source csv --prices path/to/prices.csv
```

Run with real US market data from Yahoo:

```bash
python main.py --data-source yahoo --universe-preset us-sector-etfs --start-date 2021-01-01 --end-date 2025-12-31
```

Run with real A-share data from Tushare:

```bash
set TUSHARE_TOKEN=your_token_here
python main.py --data-source tushare --start-date 2021-01-01 --end-date 2025-12-31 --tushare-universe-size 200 --tushare-markets 主板,创业板
```

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

## Key assumptions

- Prices are daily close prices.
- Signals are computed at the close of the rebalance day.
- Execution starts on the next trading day.
- Missing asset returns are treated as zero after weights are formed.
- Shorting, financing, borrow costs, and slippage are not modeled separately.
- Transaction cost is modeled as:

```text
turnover * cost_bps / 10000
```

## Useful parameters

- `--data-source`
- `--start-date`
- `--end-date`
- `--symbols`
- `--symbols-file`
- `--universe-preset`
- `--cache-dir`
- `--refresh-cache`
- `--lookback`
- `--skip-recent`
- `--rebalance-frequency`
- `--long-quantile`
- `--short-quantile`
- `--min-assets`
- `--transaction-cost-bps`

## Real-market notes

- Yahoo mode uses the public chart API and prefers adjusted close when available.
- Tushare mode uses `stock_basic`, `trade_cal`, and `daily`.
- For A-shares, the code rebuilds an adjusted-price-like series from `pct_chg`, which already reflects ex-right / ex-dividend handling in Tushare daily data.
- The automatic A-share universe:
  - keeps listed stocks only
  - excludes ST by default
  - applies a minimum listing age filter
  - ranks by latest daily turnover amount and keeps the top `N`
- All downloaded price matrices are cached under `.cache/`.

## Output files

- `used_prices.csv`: the actual matrix fed into the strategy.
- `run_metadata.json`: data source, symbols, and strategy parameters.

## Extension ideas

- Replace momentum with value, quality, residual reversal, or blended factors.
- Add sector / industry neutrality constraints.
- Add volatility scaling or risk-parity weighting.
- Load adjusted prices and delisting-aware universes.
- Add benchmark comparison and factor attribution.
- Replace synthetic data with Wind / Tushare / JoinQuant exports.
