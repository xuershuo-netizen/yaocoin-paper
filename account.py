"""逐仓模拟账本。以收盘后的 OHLC 回放撮合，每根 K 线只处理一次。"""
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from zoneinfo import ZoneInfo

from strategy import YaoCoinSniper


@dataclass(frozen=True)
class Costs:
    maker: float = 0.0002
    taker: float = 0.0005
    slippage: float = 0.0005
    maintenance: float = 0.005

    def __post_init__(self):
        if any(not math.isfinite(x) or not 0 <= x < 0.05 for x in vars(self).values()):
            raise ValueError("模拟费率、滑点和维持保证金率必须在 [0, 5%)")


def new_account():
    return {
        "cash": YaoCoinSniper.initial_cash, "position": None, "pending": None,
        "last_bar": None, "last_close": None,
        "last_funding": None, "funding_sample": None, "events": [], "daily_net": {},
    }


class PaperAccount:
    def __init__(self, strategy, costs, state):
        self.strategy, self.costs, self.state = strategy, costs, state

    def log(self, kind, timestamp, **fields):
        self.state["events"].append({"kind": kind, "timestamp": int(timestamp),
                                     "symbol": self.state.get("active_symbol"), **fields})
        self.state["events"] = self.state["events"][-300:]

    def book(self, amount, timestamp):
        day = datetime.fromtimestamp(timestamp / 1000, timezone.utc).astimezone(
            ZoneInfo("Australia/Sydney")
        ).date().isoformat()
        daily = self.state.setdefault("daily_net", {})
        daily[day] = daily.get(day, 0.0) + amount

    def equity(self, price):
        pos = self.state["position"]
        if not pos or price is None:
            return self.state["cash"]
        pnl = (pos["entry"] - price) * pos["qty"]
        return self.state["cash"] + max(0.0, pos["margin"] + pnl)

    def liquidation_price(self, pos):
        # 近似模型：逐仓余额减浮亏低于估算的维持保证金及平仓手续费。
        return (pos["entry"] + pos["margin"] / pos["qty"]) / (
            1 + self.costs.maintenance + self.costs.taker
        )

    def close(self, fraction, price, reason, ts):
        pos = self.state["position"]
        qty = pos["qty"] * fraction
        margin = pos["margin"] * fraction
        fee = price * qty * self.costs.taker
        pnl = (pos["entry"] - price) * qty
        released = 0.0 if reason == "liquidation" else max(0.0, margin + pnl - fee)
        self.state["cash"] += released
        self.book(released - margin, ts)
        self.log(reason, ts, price=price, qty=qty, fee=fee,
                 pnl_after_close_fee=released - margin, cash=self.state["cash"])
        if fraction == 1:
            self.state["position"] = None
        else:
            pos["qty"] -= qty
            pos["margin"] -= margin
            pos.update(partial=True, tp1_pending=False, stop=pos["entry"])

    def settle_funding(self, settlement):
        pos = self.state["position"]
        if not pos:
            return
        # 正费率时空头收款。把结算计入该笔逐仓资金。
        amount = settlement["rate"] * settlement["mark"] * pos["qty"]
        pos["margin"] += amount
        self.book(amount, settlement["time"])
        self.log("funding", settlement["time"], amount=amount, rate=settlement["rate"])

    def protect(self, price, ts, gap=False):
        pos = self.state["position"]
        liquidation = self.liquidation_price(pos)
        if price >= liquidation and (gap or pos["stop"] >= liquidation):
            self.close(1, price if gap else liquidation, "liquidation", ts)
            return True
        if price >= pos["stop"]:
            fill = (price if gap else pos["stop"]) * (1 + self.costs.slippage)
            if fill >= liquidation:
                self.close(1, max(liquidation, price) if gap else liquidation, "liquidation", ts)
            else:
                self.close(1, fill, "stop", ts)
            return True
        return False

    def enter_next_bar(self, row, ts):
        order = self.state["pending"]
        if order is None:
            return False
        self.state["pending"] = None
        if ts != order["bar"]:
            self.log("entry_expired", ts, reason="missed_next_bar")
            return False
        limit = float(row.open)
        if not order["target"] < limit < order["stop"]:
            self.log("entry_cancelled", ts, reason="open_outside_risk_levels", price=limit)
            return False
        # 限价取下一根开盘价。只触价无法证明排队成交，需最高价严格穿过限价。
        if row.high <= limit:
            self.log("entry_expired", ts, reason="no_trade_through", price=limit)
            return False
        margin = self.state["cash"] * self.strategy.margin_fraction
        qty = margin * self.strategy.leverage / limit
        fee = qty * limit * self.costs.maker
        if margin <= 0 or margin + fee > self.state["cash"]:
            self.log("entry_cancelled", ts, reason="insufficient_collateral")
            return False
        pos = {"entry": limit, "qty": qty, "margin": margin, "stop": order["stop"],
               "target": order["target"], "partial": False, "tp1_pending": False}
        if pos["stop"] >= self.liquidation_price(pos):
            self.log("entry_cancelled", ts, reason="stop_beyond_model_liquidation", price=limit)
            return False
        self.state["cash"] -= margin + fee
        self.state["position"] = pos
        self.book(-fee, ts)
        self.log("short_fill", ts, price=limit, qty=qty, margin=margin, fee=fee,
                 stop=pos["stop"], target=pos["target"])
        return True

    def step(self, bars, funding_rate=None, settlements=()):
        row = bars.iloc[-1]
        ts = int(row.timestamp)
        previous = self.state["last_bar"]
        if previous is not None:
            if ts <= previous:
                return
            if ts != previous + self.strategy.bar_ms:
                raise ValueError("撮合中断：缺少连续 K 线，保留账本等待补齐")
        # 资金费结算在本根开盘动作之前，开盘后才入场的仓位不参与这次结算。
        for settlement in settlements:
            self.settle_funding(settlement)
        open_price = float(row.open)
        existing = self.state["position"]
        if existing and not self.protect(open_price, ts, gap=True):
            if open_price < existing["target"]:
                self.close(1, open_price * (1 + self.costs.slippage), "tp2", ts)
            elif existing["tp1_pending"]:
                self.close(0.5, open_price * (1 + self.costs.slippage), "tp1", ts)
        entered = self.enter_next_bar(row, ts) if existing is None else False
        pos = self.state["position"]
        if pos and not self.protect(float(row.high), ts):
            if not entered and row.low < pos["target"]:
                self.close(1, pos["target"] * (1 + self.costs.slippage), "tp2", ts)
            elif self.strategy.check_exit_signal(bars, pos, check_intrabar=not entered) == "tp1_next_open":
                pos["tp1_pending"] = True
        # 只在有该信号收盘之前的资金费率快照时产生新信号。
        if funding_rate is not None:
            details = self.strategy.entry_details(bars, funding_rate)
            if details and existing is None and not entered and not self.state["position"]:
                self.state["pending"] = {"bar": ts + self.strategy.bar_ms,
                                         "stop": details.stop, "target": details.target}
                self.log("short_signal", ts + self.strategy.bar_ms, stop=details.stop,
                         target=details.target, funding_rate=funding_rate,
                         wick_timestamp=details.wick_timestamp)
        self.state["last_bar"] = ts
        self.state["last_close"] = float(row.close)
        self.state["last_funding"] = funding_rate
