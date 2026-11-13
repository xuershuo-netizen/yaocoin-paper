"""用历史收盘 K 线和逐根对齐的资金费率回放模拟盘。"""
import argparse
import json
import math

import pandas as pd

from account import Costs, PaperAccount, new_account
from strategy import YaoCoinSniper


def replay(frame, tp2_mode="half_low", costs=Costs(), settlements=()):
    strategy = YaoCoinSniper(tp2_mode)
    bars = strategy.calculate_indicators(frame)
    rates = pd.to_numeric(frame["funding_rate"], errors="raise")
    rates = dict(zip(pd.to_numeric(frame["timestamp"], errors="raise"), rates))
    events_by_bar = {}
    for event in sorted(settlements, key=lambda event: event["time"]):
        timestamp = event["time"] // strategy.bar_ms * strategy.bar_ms
        events_by_bar.setdefault(timestamp, []).append(event)
    account = PaperAccount(strategy, costs, new_account())
    for idx, row in enumerate(bars.itertuples(index=False)):
        rate = float(rates[row.timestamp])
        account.step(bars.iloc[:idx + 1], rate if math.isfinite(rate) else None,
                     events_by_bar.get(row.timestamp, ()))
    return {**account.state, "equity": account.equity(account.state["last_close"])}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="OHLCV CSV，另含每根收盘时已知的 funding_rate 列")
    parser.add_argument("--tp2-mode", choices=["half_low", "half_retracement"], default="half_low")
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--settlements", help="可选 CSV，列为 time（毫秒）、rate、mark")
    args = parser.parse_args()
    settlements = pd.read_csv(args.settlements).to_dict("records") if args.settlements else ()
    state = replay(pd.read_csv(args.csv), args.tp2_mode, Costs(slippage=args.slippage),
                   settlements)
    print(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False))
