#!/usr/bin/env python
"""轮询 /status/{task_id}，记录节点完成时间线并输出 JSON。

用法（项目根目录）：
    PYTHONIOENCODING=utf-8 \
      F:/Program/miniconda/envs/graph-rag/python.exe scripts/poll_task.py <task_id> [timeout_sec]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8000"


def get_status(task_id: str) -> dict:
    with urllib.request.urlopen(f"{BASE}/status/{task_id}", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: poll_task.py <task_id> [timeout_sec]", file=sys.stderr)
        return 2
    task_id = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 1800

    start = time.time()
    seen_done: list[str] = []
    timeline: list[dict] = []
    last = None

    while True:
        elapsed = round(time.time() - start, 1)
        try:
            info = get_status(task_id)
        except Exception as exc:
            info = {"error": f"{type(exc).__name__}: {exc}"}

        current = json.dumps(info, ensure_ascii=False, sort_keys=True)
        if current != last:
            last = current
            for node in info.get("done_list", []):
                if node not in seen_done:
                    seen_done.append(node)
                    timeline.append({"elapsed_s": elapsed, "event": "done", "node": node})
            for node in info.get("running_list", []):
                timeline.append({"elapsed_s": elapsed, "event": "running", "node": node})
            print(json.dumps({"elapsed_s": elapsed, "status": info.get("status"),
                              "done": info.get("done_list"), "running": info.get("running_list")},
                             ensure_ascii=False), flush=True)

        if info.get("status") in ("completed", "failed"):
            break
        if time.time() - start > timeout:
            print(json.dumps({"error": "timeout", "elapsed_s": elapsed}, ensure_ascii=False))
            break
        time.sleep(3)

    result = {
        "task_id": task_id,
        "total_s": round(time.time() - start, 1),
        "final_status": info.get("status"),
        "done_list": info.get("done_list"),
        "running_list": info.get("running_list"),
        "timeline": timeline,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "reports" / f"task_{task_id}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["final_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
