"""妖币末日狙击系统：只读行情与本地模拟撮合。"""
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Entry:
    stop: float
    target: float
    wick_timestamp: int


class YaoCoinSniper:
    initial_cash = 200.0
    leverage = 10
    margin_fraction = 0.50

    def __init__(self, tp2_mode="half_low", bar_minutes=15):
        if tp2_mode not in {"half_low", "half_retracement"}:
            raise ValueError("TP2_MODE 必须为 half_low 或 half_retracement")
        if not isinstance(bar_minutes, int) or bar_minutes <= 0 or 1440 % bar_minutes:
            raise ValueError("K 线周期必须能整除 24 小时")
        self.tp2_mode = tp2_mode
        self.bar_ms = bar_minutes * 60_000
        self.window = 1440 // bar_minutes

    def calculate_indicators(self, df):
        """timestamp 是 UTC 开盘时间，单位毫秒；输入只含已收盘 K 线。"""
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        bars = df.loc[:, columns].apply(pd.to_numeric, errors="raise").copy()
        if not np.isfinite(bars.to_numpy(dtype=float)).all():
            raise ValueError("OHLCV 包含非有限值")
        if not bars.timestamp.mod(self.bar_ms).eq(0).all():
            raise ValueError("K 线时间戳与周期不对齐")
        prices = bars[["open", "high", "low", "close"]]
        if ((prices <= 0).any(axis=None) or (bars.volume < 0).any()
                or (bars.high < bars[["open", "close"]].max(axis=1)).any()
                or (bars.low > bars[["open", "close"]].min(axis=1)).any()):
            raise ValueError("OHLCV 范围无效")
        bars = bars.sort_values("timestamp").reset_index(drop=True)
        if bars.timestamp.duplicated().any():
            raise ValueError("K 线时间戳重复")
        # 缺 K 线时重新预热指标，避免把不连续的 96 根当成连续 24 小时。
        segments = bars.timestamp.diff().ne(self.bar_ms).cumsum()
        for period in (5, 10, 20):
            bars[f"MA{period}"] = bars.groupby(segments).close.transform(
                lambda values: values.rolling(period).mean()
            )
        bars["vol_ma5"] = bars.groupby(segments).volume.transform(
            lambda values: values.rolling(5).mean()
        )
        bars["body_size"] = (bars.close - bars.open).abs()
        bars["upper_shadow"] = bars.high - np.maximum(bars.open, bars.close)
        bars["low_24h"] = bars.groupby(segments).low.transform(
            lambda values: values.rolling(self.window).min()
        )
        bars["high_24h"] = bars.groupby(segments).high.transform(
            lambda values: values.rolling(self.window).max()
        )
        # 上影线与缩量必须同时出现在同一根 K 线上。
        bars["wick_setup"] = ((bars.upper_shadow > 2 * bars.body_size)
                              & (bars.volume < 0.8 * bars.vol_ma5))
        bars["overextended"] = bars.high > 1.15 * bars.MA20
        return bars

    def entry_details(self, bars, funding_rate):
        if len(bars) < max(22, self.window) or not math.isfinite(funding_rate):
            return None
        now, previous = bars.iloc[-1], bars.iloc[-2]
        recent = bars.iloc[-3:]
        wicks = recent[recent.wick_setup]
        if not (
            now.high_24h > 1.80 * now.low_24h
            and recent.overextended.any()
            and not wicks.empty
            # 上一根尚未死叉，当前才由上向下穿越。
            and previous.MA5 >= previous.MA10
            and now.MA5 < now.MA10
            and now.close < now.MA10
            and now.high >= 0.98 * now.MA10
            and funding_rate > 0.0003
        ):
            return None
        wick = wicks.iloc[-1]
        target = (0.5 * now.low_24h if self.tp2_mode == "half_low"
                  else now.low_24h + 0.5 * (now.high_24h - now.low_24h))
        return Entry(float(wick.high * 1.01), float(target), int(wick.timestamp))

    def check_entry_signal(self, df, funding_rate):
        bars = df if "wick_setup" in df else self.calculate_indicators(df)
        entry = self.entry_details(bars, funding_rate)
        return (entry is not None, entry.stop if entry else None)

    def check_exit_signal(self, df, position, *, check_intrabar=True):
        """返回当前收盘 K 线的退出指令；账户负责成交和更新仓位。"""
        bars = df if "wick_setup" in df else self.calculate_indicators(df)
        if bars.empty:
            return None
        now = bars.iloc[-1]
        if check_intrabar and now.high >= position["stop"]:
            return "stop"
        if check_intrabar and now.low < position["target"]:
            return "tp2"
        if len(bars) > 1 and not position["partial"] and not position["tp1_pending"]:
            previous = bars.iloc[-2]
            if previous.close >= previous.MA20 and now.close < now.MA20:
                return "tp1_next_open"
        return None
