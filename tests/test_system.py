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
    build_tushare_tradeability_masks,
    parse_symbol_inputs,
)
from cross_sectional_ls.system import (
    CrossSectionalLongShortSystem,
    ExecutionConstraints,
    StrategyConfig,
)


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
        self.assertIn("execution_fill_rate", result.metrics)
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

    def test_long_only_weights_are_non_negative(self) -> None:
        prices = generate_synthetic_prices(n_assets=20, n_days=260, seed=9)
        config = StrategyConfig(
            lookback_days=20,
            skip_recent_days=3,
            rebalance_frequency="ME",
            min_assets=8,
            portfolio_mode="long_only",
        )
        system = CrossSectionalLongShortSystem(config)
        result = system.run(prices)

        active_rebalances = result.target_weights.abs().sum(axis=1) > 0
        self.assertTrue((result.target_weights.loc[active_rebalances] >= -1e-12).all().all())
        self.assertTrue(
            (
                result.target_weights.loc[active_rebalances].sum(axis=1) - 1.0
            ).abs().max()
            < 1e-10
        )

    def test_execution_constraints_block_buys_and_sells(self) -> None:
        index = pd.date_range("2024-01-02", periods=4, freq="B")
        desired = pd.DataFrame(
            {
                "A": [0.0, 1.0, 0.0, 0.0],
                "B": [0.0, 0.0, 0.0, 0.0],
            },
            index=index,
        )
        constraints = ExecutionConstraints(
            buy_blocked=pd.DataFrame(
                {"A": [False, True, False, False], "B": [False, False, False, False]},
                index=index,
            ),
            sell_blocked=pd.DataFrame(
                {"A": [False, False, True, False], "B": [False, False, False, False]},
                index=index,
            ),
        )
        system = CrossSectionalLongShortSystem(StrategyConfig(min_assets=2))

        actual, summary = system.simulate_daily_weights(desired, constraints)

        self.assertAlmostEqual(actual.loc[index[1], "A"], 0.0)
        self.assertAlmostEqual(summary.loc[index[1], "blocked_turnover"], 1.0)

        desired.loc[index[1], "A"] = 1.0
        desired.loc[index[2], "A"] = 1.0
        desired.loc[index[3], "A"] = 0.0
        actual, summary = system.simulate_daily_weights(desired, constraints)
        self.assertAlmostEqual(actual.loc[index[1], "A"], 0.0)
        self.assertAlmostEqual(actual.loc[index[2], "A"], 1.0)
        self.assertAlmostEqual(actual.loc[index[3], "A"], 0.0)
        self.assertGreater(summary["blocked_turnover"].sum(), 0.0)

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

    def test_tushare_tradeability_masks_block_open_limits_and_suspensions(self) -> None:
        trading_dates = pd.date_range("2024-01-02", periods=2, freq="B")
        daily_data = pd.DataFrame(
            {
                "trade_date": [trading_dates[0], trading_dates[1], trading_dates[0], trading_dates[1]],
                "ts_code": ["AAA", "AAA", "BBB", "BBB"],
                "open": [11.0, 9.0, 20.0, np.nan],
                "high": [11.0, 9.5, 20.0, np.nan],
                "low": [11.0, 9.0, 20.0, np.nan],
                "close": [11.0, 9.1, 20.0, np.nan],
                "pre_close": [10.0, 10.0, 20.0, 20.0],
                "pct_chg": [10.0, -9.0, 0.0, np.nan],
                "vol": [1000.0, 1000.0, 1000.0, 0.0],
                "amount": [10000.0, 10000.0, 10000.0, 0.0],
            }
        )
        limit_data = pd.DataFrame(
            {
                "trade_date": [trading_dates[0], trading_dates[1], trading_dates[0], trading_dates[1]],
                "ts_code": ["AAA", "AAA", "BBB", "BBB"],
                "up_limit": [11.0, 11.0, 22.0, 22.0],
                "down_limit": [9.0, 9.0, 18.0, 18.0],
            }
        )

        buy_blocked, sell_blocked, metadata = build_tushare_tradeability_masks(
            daily_data=daily_data,
            limit_data=limit_data,
            trading_dates=trading_dates,
        )

        self.assertTrue(buy_blocked.loc[trading_dates[0], "AAA"])
        self.assertTrue(sell_blocked.loc[trading_dates[1], "AAA"])
        self.assertTrue(buy_blocked.loc[trading_dates[1], "BBB"])
        self.assertTrue(sell_blocked.loc[trading_dates[1], "BBB"])
        self.assertGreaterEqual(metadata["average_buy_block_ratio"], 0.0)


if __name__ == "__main__":
    unittest.main()
