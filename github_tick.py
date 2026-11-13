"""执行一次公开行情轮询并生成 GitHub Pages 静态快照。"""
import json
from pathlib import Path

from account import Costs
from server import BinanceMarket, Ledger, PaperService
from strategy import YaoCoinSniper


def tick(root, market):
    state_path = root / "paper-state.json"
    public = root / "_site"
    public.mkdir(exist_ok=True)
    service = PaperService(
        market, Ledger(root / "data" / "github.sqlite3", "github-auto-v1"),
        "AUTO", YaoCoinSniper(), Costs(),
    )
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        service.ledger.write(previous["ledger"])
        service.status["last_success"] = previous["last_success"]
    try:
        service.poll()
    except Exception as error:
        # 保留最后一份账本，并在页面显示本轮行情失败。
        print(f"Market update failed: {type(error).__name__}: {ascii(str(error))}")
        service.status["error"] = f"行情更新失败（{type(error).__name__}）"
    else:
        persisted = {"ledger": service.ledger.read(), "last_success": service.status["last_success"]}
        state_path.write_text(json.dumps(persisted, allow_nan=False) + "\n", encoding="utf-8")
    snapshot = service.snapshot()
    snapshot["snapshot_mode"] = True
    (public / "status.json").write_text(json.dumps(snapshot, allow_nan=False), encoding="utf-8")
    html = (root / "dashboard.html").read_text(encoding="utf-8")
    (public / "index.html").write_text(
        html.replace('<html lang="zh-CN">', '<html lang="zh-CN" data-status-url="./status.json">', 1),
        encoding="utf-8",
    )
    return snapshot


if __name__ == "__main__":
    tick(Path.cwd(), BinanceMarket())
