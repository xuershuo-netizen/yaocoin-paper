import tempfile
import unittest
from pathlib import Path

import pandas as pd

from account import Costs, PaperAccount, new_account
from server import Ledger, PaperService
from strategy import YaoCoinSniper


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.strategy = YaoCoinSniper()
        self.interval = self.strategy.bar_ms
        self.rows = []
        for value in [100] * 86 + [122] * 4 + [112] + [122] * 4:
            self.bar(value, value + 1, value - 1, value)
        self.rows[0][3] = 60
        self.rows[-2][2] = 132
        self.rows[-2][5] = 50
        self.bar(112, 119, 99, 100)
        self.state = new_account()
        self.account = PaperAccount(self.strategy, Costs(), self.state)

    def bar(self, op, hi, lo, cl, vol=100):
        self.rows.append([len(self.rows) * self.interval, op, hi, lo, cl, vol])

    def frame(self):
        return pd.DataFrame(self.rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

    def step(self, funding=None):
        self.account.step(self.strategy.calculate_indicators(self.frame()), funding)

    def test_entry_and_stop_priority(self):
        self.assertEqual(self.strategy.check_entry_signal(self.frame(), 0.0003), (False, None))
        self.step(0.00031)
        self.assertIsNone(self.state["position"])
        self.assertAlmostEqual(self.state["pending"]["stop"], 133.32)
        self.assertEqual(self.state["pending"]["target"], 30)
        self.bar(124, 131, 123, 130)
        self.step()
        self.step()
        self.assertAlmostEqual(self.state["cash"], 99.8)
        self.assertEqual(self.state["position"]["margin"], 100)
        self.assertAlmostEqual(self.state["position"]["qty"] * 124, 1000)
        self.bar(130, 134, 29, 40)
        self.step()
        self.assertIsNone(self.state["position"])
        self.assertEqual(self.state["events"][-1]["kind"], "stop")

    def test_no_trade_through_expires(self):
        self.step(0.00031)
        self.bar(124, 124, 120, 122)
        self.step()
        self.assertIsNone(self.state["pending"])
        self.assertIsNone(self.state["position"])
        self.assertEqual(self.state["events"][-1]["kind"], "entry_expired")

    def test_tp1_at_next_open_then_break_even(self):
        self.step(0.00031)
        self.bar(124, 131, 123, 130)
        self.step()
        self.bar(120, 121, 99, 100)
        self.step()
        self.assertTrue(self.state["position"]["tp1_pending"])
        self.bar(101, 107, 96, 98)
        self.step()
        self.assertEqual(self.state["events"][-1]["kind"], "tp1")
        self.assertAlmostEqual(self.state["position"]["qty"] * 124, 500)
        self.assertEqual(self.state["position"]["stop"], 124)
        self.bar(98, 124, 95, 120)
        self.step()
        self.assertEqual(self.state["events"][-1]["kind"], "stop")

    def test_stop_beyond_liquidation_cancels(self):
        self.step(0.00031)
        self.bar(112, 116, 106, 108)
        self.step()
        self.assertEqual(self.state["events"][-1]["reason"], "stop_beyond_model_liquidation")
        self.assertEqual(self.state["cash"], 200)

    def test_gap_blocks_signal(self):
        del self.rows[30]
        self.assertEqual(self.strategy.check_entry_signal(self.frame(), 0.00031), (False, None))

    def test_live_poll_persists_only_new_bars_and_point_in_time_funding(self):
        bars = self.frame()
        duration = self.interval
        last_open = int(bars.timestamp.iloc[-1])

        class Feed:
            now = last_open + duration - 3000

            def snapshot(self, symbol):
                return {"time": self.now, "rate": 0.00031, "mark": 124, "next_funding": 0}

            def candles(self, symbol, now, limit):
                return bars[bars.timestamp + duration <= now]

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "paper.sqlite3"
            feed = Feed()
            ledger = Ledger(path, "test")
            service = PaperService(feed, ledger, "AKEUSDT", self.strategy, Costs())
            service.poll()
            self.assertEqual(service.snapshot()["equity"], 200)
            self.assertEqual(service.snapshot()["events"], [])
            feed.now += 15_000
            service.poll()
            service.poll()
            restored = Ledger(path, "test").read()
            self.assertEqual(restored["last_bar"], last_open)
            self.assertEqual([event["kind"] for event in restored["events"]], ["short_signal"])

    def test_auto_selects_best_funding_and_shares_single_account(self):
        frame = self.frame()
        duration = self.interval
        closing = int(frame.timestamp.iloc[-1]) + duration

        class Market:
            now = closing - 1000

            def server_time(self):
                return self.now

            def universe(self):
                return {"ONEUSDT", "TWOUSDT", "LOWUSDT"}

            def snapshot(self, symbol):
                return self.snapshots()[symbol]

            def snapshots(self):
                return {symbol: {"time": self.now, "rate": rate, "mark": 124, "next_funding": 0}
                        for symbol, rate in {"ONEUSDT": 0.0004, "TWOUSDT": 0.0005,
                                             "LOWUSDT": 0.0002, "NOTACTIVEUSDT": 0.1}.items()}

            def candles(self, symbol, now, limit):
                assert symbol in {"ONEUSDT", "TWOUSDT"}
                return frame[frame.timestamp + duration <= now]

            def settlements(self, symbol, start, end):
                return []

        with tempfile.TemporaryDirectory() as temp:
            feed = Market()
            ledger = Ledger(Path(temp) / "paper.sqlite3", "auto")
            service = PaperService(feed, ledger, "AUTO", self.strategy, Costs())
            service.poll()
            feed.now = closing + 1000
            service.poll()
            status = service.snapshot()
            self.assertEqual(status["scan"]["signals"], ["TWOUSDT", "ONEUSDT"])
            self.assertEqual(status["symbol"], "TWOUSDT")
            self.assertEqual(status["pending"]["bar"], closing)
            self.assertEqual(status["events"][0]["symbol"], "TWOUSDT")
            self.bar(124, 131, 123, 130)
            frame = self.frame()
            feed.now += duration
            service.poll()
            status = service.snapshot()
            self.assertEqual(status["position"]["margin"], 100)
            self.assertAlmostEqual(sum(status["daily_net"].values()), -0.2)
            self.assertEqual([event["kind"] for event in status["events"]], ["short_signal", "short_fill"])


if __name__ == "__main__":
    unittest.main()
