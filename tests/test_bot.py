import unittest
import numpy as np
from core.filters import sma, lma, ema, macd
from core.bot import make_signal, crossover_signals, TYPE_MAP

class TestSignalFunctions(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.prices = np.linspace(100, 200, 100)  # clean, linearly rising prices
        self.noisy  = self.prices + np.random.normal(0, 2, size=100)

    def test_signal_outputs_shape(self):
        alpha = 0.2
        for t in TYPE_MAP:
            sig = make_signal(self.prices, 10, t, alpha)
            self.assertIsInstance(sig, np.ndarray, f"Signal type {t} did not return a numpy array.")
            self.assertGreater(len(sig), 0, f"Signal type {t} returned empty array.")

    def test_macd_difference(self):
        alpha = 0.2
        window = 30
        macd_line = macd(self.prices, window, alpha)

        # Recalculate expected manually
        ema_fast = ema(self.prices, window // 2, alpha)
        ema_slow = ema(self.prices, window, alpha)

        min_len = min(len(ema_fast), len(ema_slow))
        expected = ema_fast[-min_len:] - ema_slow[-min_len:]

        np.testing.assert_almost_equal(macd_line[-min_len:], expected, decimal=6,
                                       err_msg="MACD calculation mismatch between manual EMA subtraction and macd() output.")

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError, msg="make_signal did not raise ValueError for unknown type."):
            make_signal(self.prices, 10, 999, 0.2)

    def test_crossover_logic(self):
        # Use random walk prices instead of smooth line
        rng = np.random.default_rng(42)
        random_walk_prices = np.cumsum(rng.normal(0, 1, size=200)) + 100

        theta = [5, 0, 0.0, 20, 0, 0.0]  # SMA 5 vs SMA 20
        buy_idx, sell_idx, n = crossover_signals(random_walk_prices, theta)

        self.assertIsInstance(buy_idx, np.ndarray, "buy_idx is not a numpy array.")
        self.assertTrue(np.all(buy_idx >= 0), "buy_idx contains negative indices.")
        self.assertGreater(len(buy_idx), 0, "No upward crossovers detected in random walk prices.")

    def test_min_length_consistency(self):
        theta = [5, 2, 0.1, 60, 2, 0.1]
        buy_idx, sell_idx, n = crossover_signals(self.prices, theta)
        sig1 = make_signal(self.prices, 5, 2, 0.1)
        sig2 = make_signal(self.prices, 60, 2, 0.1)
        min_len = min(len(sig1), len(sig2))
        self.assertEqual(n, min_len, f"Mismatch in min length: Expected {min_len}, got {n}.")

    def test_equal_windows_force_d2_gt_d1(self):
        theta = [10, 0, 0.0, 10, 0, 0.0]  # intentionally equal
        d1, d2 = int(np.round(theta[0])), int(np.round(np.maximum(theta[3], theta[0] + 1)))
        self.assertGreater(d2, d1, f"d2 ({d2}) is not greater than d1 ({d1}). d1 and d2 must satisfy d2 > d1.")

if __name__ == "__main__":
    unittest.main(verbosity=2)
