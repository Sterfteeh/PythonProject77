from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cross_sectional_ls.data import generate_synthetic_prices, load_prices_from_csv
from cross_sectional_ls.market_data import (
    YAHOO_PRESETS,
    _build_adjusted_prices_from_tushare_returns,
    parse_symbol_inputs,
)
from cross_sectional_ls.system import CrossSectionalLongShortSystem, StrategyConfig


class CrossSectionalSystemTests(unittest.TestCase):
    def test_backtest_runs_end_to_end(self) -> None:
        prices = generate_synthetic_prices(n_assets=24, n_days=320, seed=7)
        config = StrategyConfig(
            lookback_days=40,
            skip_recent_days=5,
            rebalance_frequency="ME",
            min_assets=10,
            transaction_cost_bps=5.0,
        )
        system = CrossSectionalLongShortSystem(config)

        result = system.run(prices)

        self.assertFalse(result.daily_results.empty)
        self.assertIn("equity", result.daily_results.columns)
        self.assertIn("sharpe", result.metrics)
        self.assertFalse(result.daily_results["net_return"].isna().any())

        active_rebalances = result.target_weights.abs().sum(axis=1) > 0
        self.assertTrue(
            (
                result.target_weights.loc[active_rebalances].sum(axis=1).abs()
                < 1e-10
            ).all()
        )
        self.assertTrue(
            (
                result.target_weights.loc[active_rebalances].abs().sum(axis=1) - 1.0
            ).abs().max()
            < 1e-10
        )

    def test_load_long_format_csv(self) -> None:
        long_prices = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "asset": ["A", "B", "A", "B"],
                "close": [10.0, 20.0, 11.0, 19.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "prices.csv"
            long_prices.to_csv(csv_path, index=False)
            prices = load_prices_from_csv(csv_path)

        self.assertEqual(prices.shape, (2, 2))
        self.assertEqual(list(prices.columns), ["A", "B"])
        self.assertAlmostEqual(prices.loc[pd.Timestamp("2024-01-02"), "A"], 11.0)

    def test_parse_symbol_inputs_merges_file_string_and_preset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            symbols_path = Path(temp_dir) / "symbols.txt"
            symbols_path.write_text("spy\nqqq\n", encoding="utf-8")

            resolved = parse_symbol_inputs(
                symbols="iwm,spy",
                symbols_file=symbols_path,
                preset="us-sector-etfs",
                preset_map=YAHOO_PRESETS,
            )

        self.assertIn("SPY", resolved)
        self.assertIn("QQQ", resolved)
        self.assertIn("IWM", resolved)
        self.assertEqual(resolved.count("SPY"), 1)

    def test_tushare_pct_change_rebuilds_adjusted_prices(self) -> None:
        trading_dates = pd.date_range("2024-01-02", periods=4, freq="B")
        daily_data = pd.DataFrame(
            {
                "trade_date": [
                    trading_dates[0],
                    trading_dates[1],
                    trading_dates[3],
                    trading_dates[1],
                    trading_dates[2],
                    trading_dates[3],
                ],
                "ts_code": ["AAA", "AAA", "AAA", "BBB", "BBB", "BBB"],
                "close": [10.0, 11.0, 12.1, 20.0, 18.0, 18.9],
                "pct_chg": [0.0, 10.0, 10.0, 0.0, -10.0, 5.0],
            }
        )

        prices = _build_adjusted_prices_from_tushare_returns(daily_data, trading_dates)

        self.assertAlmostEqual(prices.loc[trading_dates[0], "AAA"], 10.0)
        self.assertAlmostEqual(prices.loc[trading_dates[1], "AAA"], 11.0)
        self.assertAlmostEqual(prices.loc[trading_dates[2], "AAA"], 11.0)
        self.assertAlmostEqual(prices.loc[trading_dates[3], "AAA"], 12.1)
        self.assertTrue(np.isnan(prices.loc[trading_dates[0], "BBB"]))
        self.assertAlmostEqual(prices.loc[trading_dates[1], "BBB"], 20.0)
        self.assertAlmostEqual(prices.loc[trading_dates[2], "BBB"], 18.0)
        self.assertAlmostEqual(prices.loc[trading_dates[3], "BBB"], 18.9)


if __name__ == "__main__":
    unittest.main()
