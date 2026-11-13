"""公开行情轮询、SQLite 账本和只读模拟盘面板。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen

import pandas as pd

from account import Costs, PaperAccount, new_account
from strategy import YaoCoinSniper


class BinanceMarket:
    root = "https://fapi.binance.com/fapi/v1/"

    def get(self, path, **params):
        # 仅调用公开市场数据；不存在账户凭证和下单接口。
        with urlopen(self.root + path + "?" + urlencode(params), timeout=10) as response:
            return json.load(response)

    def snapshot(self, symbol):
        return self.parse_snapshot(self.get("premiumIndex", symbol=symbol))

    @staticmethod
    def parse_snapshot(raw):
        sample = {"time": int(raw["time"]), "rate": float(raw["lastFundingRate"]),
                  "mark": float(raw["markPrice"]), "next_funding": int(raw["nextFundingTime"])}
        if not math.isfinite(sample["rate"]) or not (0 < sample["mark"] < math.inf):
            raise ValueError("无效行情快照")
        return sample

    def universe(self):
        return {
            item["symbol"] for item in self.get("exchangeInfo")["symbols"]
            if item["contractType"] == "PERPETUAL" and item["quoteAsset"] == "USDT"
            and item["status"] == "TRADING"
        }

    def snapshots(self):
        return {item["symbol"]: self.parse_snapshot(item) for item in self.get("premiumIndex")
                if float(item["markPrice"]) > 0}

    def server_time(self):
        return int(self.get("time")["serverTime"])

    def candles(self, symbol, now, limit):
        raw = self.get("klines", symbol=symbol, interval="15m", limit=limit)
        return pd.DataFrame(
            [bar[:6] for bar in raw if int(bar[6]) < now],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

    def settlements(self, symbol, start, end):
        raw = self.get("fundingRate", symbol=symbol, startTime=start, endTime=end, limit=1000)
        result = []
        for item in raw:
            settlement = {"time": int(item["fundingTime"]), "rate": float(item["fundingRate"]),
                          "mark": float(item["markPrice"])}
            if not math.isfinite(settlement["rate"]) or not 0 < settlement["mark"] < math.inf:
                raise ValueError("无效资金费结算")
            result.append(settlement)
        return sorted(result, key=lambda item: item["time"])


class Ledger:
    def __init__(self, path, identity):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS ledger (id INTEGER PRIMARY KEY, identity TEXT, data TEXT)")
            db.execute("INSERT OR IGNORE INTO ledger VALUES (1, ?, ?)",
                       (identity, json.dumps(new_account())))
            if db.execute("SELECT identity FROM ledger WHERE id=1").fetchone()[0] != identity:
                raise ValueError("模拟账本与配置不匹配，请为新配置指定独立 STATE_DB")

    def connection(self):
        from contextlib import closing, contextmanager

        @contextmanager
        def transaction():
            with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
                yield db

        return transaction()

    def read(self):
        with self.connection() as db:
            return json.loads(db.execute("SELECT data FROM ledger WHERE id=1").fetchone()[0])

    def write(self, state):
        with self.connection() as db:
            db.execute("UPDATE ledger SET data=? WHERE id=1", (json.dumps(state, allow_nan=False),))


class PaperService:
    def __init__(self, market, ledger, symbol, strategy, costs):
        self.market, self.ledger, self.symbol = market, ledger, symbol
        self.strategy, self.costs = strategy, costs
        self.lock = threading.Lock()
        self.status = {"last_success": None, "error": None, "market": None}

    def advance(self, state, symbol, market, *, allow_entry):
        now = market["time"]
        # 在资金费整点后的第一分钟等待结算发布，避免用预估值入账。
        if (state["funding_sample"] and state["funding_sample"]["next_funding"] > 0
                and 0 <= now - state["funding_sample"]["next_funding"] < 60_000):
            return False
        cursor = state["last_bar"]
        interval = self.strategy.bar_ms
        limit = 125 if cursor is None else max(125, (now - cursor) // interval + 110)
        if limit > 1500:
            raise ValueError("停机超过单次行情回补上限，暂停撮合并保留账本")
        bars = self.strategy.calculate_indicators(self.market.candles(symbol, now, limit))
        if bars.empty:
            raise ValueError("未收到已收盘 K 线")
        if int(bars.timestamp.iloc[-1]) != now // interval * interval - interval:
            raise ValueError("行情缺少最新已收盘 K 线")
        if cursor is None:
            # 首次启动只预热，不把启动前的信号冒充为实时模拟成交。
            state["last_bar"] = int(bars.timestamp.iloc[-1])
            state["last_close"] = float(bars.close.iloc[-1])
        else:
            account = PaperAccount(self.strategy, self.costs, state)
            new_bars = bars[bars.timestamp > cursor]
            due = []
            if not new_bars.empty and (state["position"] or state["pending"] or len(new_bars) > 1):
                due = self.market.settlements(symbol, cursor + interval, now)
                previous_sample = state["funding_sample"]
                expected = previous_sample["next_funding"] if previous_sample else 0
                if (expected and cursor + interval <= expected <= int(new_bars.timestamp.iloc[-1])
                        and not any(expected <= item["time"] < expected + 60_000 for item in due)):
                    raise ValueError("等待资金费结算记录")
            previous_sample = state["funding_sample"]
            for index in new_bars.index:
                opening = int(bars.at[index, "timestamp"])
                closing = opening + interval
                # 只有足够接近且早于收盘的公开快照可用于信号。回补时不复用今天的费率。
                rate = (previous_sample["rate"] if allow_entry and previous_sample
                        and 0 < closing - previous_sample["time"] <= 120_000 else None)
                settlements = [item for item in due if opening <= item["time"] < closing]
                account.step(bars.iloc[:index + 1], rate, settlements)
        # 资金费结算尚未落地时保留原始预告时间，供下一次完整撮合检查。
        state["funding_sample"] = market
        return True

    def scan(self, state):
        now = self.market.server_time()
        interval = self.strategy.bar_ms
        closing = now // interval * interval
        latest = closing - interval
        active = state.get("active_symbol")
        mark = None
        if active:
            mark = self.market.snapshot(active)
            self.advance(state, active, mark, allow_entry=False)
            if not state["position"] and not state["pending"]:
                state["active_symbol"] = None

        # 全市场先筛资金费率，其余合约必然无法满足六项开仓条件。
        previous = state.get("scan_rates", {})
        current = self.market.snapshots()
        # 保留每个合约最新一笔属于当前未收盘 K 线的费率，避免收盘后的报价覆盖它。
        state["scan_rates"] = {symbol: sample for symbol, sample in current.items()
                               if closing <= sample["time"] <= now}
        if state["last_bar"] is not None and state["last_bar"] < latest:
            for symbol, sample in previous.items():
                if symbol not in state["scan_rates"] and 0 < closing - sample["time"] <= 120_000:
                    state["scan_rates"][symbol] = sample
        if state["last_bar"] is None:
            state["last_bar"] = latest
        elif active is None and state["last_bar"] < latest:
            rates = {symbol: sample["rate"] for symbol, sample in previous.items()
                     if sample["rate"] > 0.0003 and 0 < closing - sample["time"] <= 120_000}
            universe = self.market.universe()
            eligible = sorted(universe & rates.keys())

            def evaluate(symbol):
                bars = self.strategy.calculate_indicators(self.market.candles(symbol, now, 100))
                if bars.empty or int(bars.timestamp.iloc[-1]) != latest:
                    return symbol, None, None
                return symbol, bars, self.strategy.entry_details(bars, rates[symbol])

            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(evaluate, eligible))
            signals = [(symbol, bars) for symbol, bars, entry in results if entry]
            signals.sort(key=lambda item: (-rates[item[0]], item[0]))
            state["scan_summary"] = {"time": closing, "universe": len(universe),
                                     "funding_candidates": len(eligible),
                                     "checked": sum(bars is not None for _, bars, _ in results),
                                     "signals": [symbol for symbol, _ in signals]}
            if signals:
                symbol, bars = signals[0]
                state.update(last_bar=None, active_symbol=symbol)
                PaperAccount(self.strategy, self.costs, state).step(bars, rates[symbol])
                state["funding_sample"] = current.get(symbol, previous[symbol])
                mark = current.get(symbol)
            else:
                state["last_bar"] = latest
        return mark

    def poll(self):
        state = self.ledger.read()
        if self.symbol == "AUTO":
            market = self.scan(state)
        else:
            market = self.market.snapshot(self.symbol)
            if not self.advance(state, self.symbol, market, allow_entry=True):
                return
        self.ledger.write(state)
        with self.lock:
            self.status = {"last_success": int(time.time() * 1000), "error": None, "market": market}

    def loop(self, stop, poll_seconds):
        delay = poll_seconds
        while not stop.is_set():
            try:
                self.poll()
                delay = poll_seconds
            except Exception as exc:
                logging.warning("行情或账本暂不可用：%s", exc)
                with self.lock:
                    self.status["error"] = str(exc)
                delay = min(120, delay * 2)
            stop.wait(delay)

    def snapshot(self):
        with self.lock:
            status = deepcopy(self.status)
        state = self.ledger.read()
        account = PaperAccount(self.strategy, self.costs, state)
        market = status["market"]
        price = market["mark"] if market else state["last_close"]
        return {
            "symbol": state.get("active_symbol") or self.symbol,
            "scan": state.get("scan_summary"), "mode": "paper", "leverage": self.strategy.leverage,
            "margin_fraction": self.strategy.margin_fraction, "tp2_mode": self.strategy.tp2_mode,
            "initial_cash": self.strategy.initial_cash, "cash": state["cash"],
            "equity": account.equity(price), "position": state["position"], "pending": state["pending"],
            "last_bar": state["last_bar"], "last_signal_funding": state["last_funding"],
            "daily_net": state.get("daily_net", {}),
            "events": state["events"], **status,
        }


def make_handler(service):
    page = Path(__file__).with_name("dashboard.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                code, mime, body = 200, "text/html; charset=utf-8", page
            elif path in {"/api/status", "/healthz"}:
                if path == "/healthz":
                    status = {"process": "ready", "mode": "paper"}
                else:
                    try:
                        status = service.snapshot()
                    except (ValueError, sqlite3.Error):
                        self.send_error(503, "Ledger unavailable")
                        return
                code, mime, body = 200, "application/json", json.dumps(status).encode()
            else:
                self.send_error(404)
                return
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--once", action="store_true", help="只拉取并处理一次行情")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    symbol = os.environ.get("PAPER_SYMBOL", "AUTO").strip().upper()
    if symbol != "AUTO" and not re.fullmatch(r"[A-Z0-9]{2,20}USDT", symbol):
        raise ValueError("PAPER_SYMBOL 必须是 AUTO 或 USDT 永续合约代码")
    strategy = YaoCoinSniper(os.environ.get("TP2_MODE", "half_low"))
    costs = Costs(slippage=float(os.environ.get("PAPER_SLIPPAGE", "0.0005")))
    identity = json.dumps([2, symbol, strategy.tp2_mode, strategy.bar_ms, asdict(costs)], sort_keys=True)
    ledger = Ledger(os.environ.get("STATE_DB", "data/paper.sqlite3"), identity)
    service = PaperService(BinanceMarket(), ledger, symbol, strategy, costs)
    if args.once:
        service.poll()
        print(json.dumps(service.snapshot(), ensure_ascii=False))
        return
    stop = threading.Event()
    with ThreadingHTTPServer((args.host, args.port), make_handler(service)) as server:
        worker = threading.Thread(target=service.loop, args=(stop, 15), daemon=True)
        worker.start()
        logging.info("模拟盘 http://%s:%s", args.host, args.port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            worker.join(timeout=15)


if __name__ == "__main__":
    main()
