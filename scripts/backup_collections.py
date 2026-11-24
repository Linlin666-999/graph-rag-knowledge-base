#!/usr/bin/env python
"""备份 Milvus 三个集合的现有数据到 JSON（导入前保护用）。

用法（项目根目录）：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. \
      F:/Program/miniconda/envs/graph-rag/python.exe scripts/backup_collections.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=ROOT / ".env")


def dump_collection(client, name: str) -> dict:
    if not client.has_collection(name):
        return {"exists": False, "rows": []}
    try:
        rows = client.query(name, filter="", output_fields=["*"], limit=16384)
    except Exception as exc:
        return {"exists": True, "error": f"{type(exc).__name__}: {exc}", "rows": []}
    return {"exists": True, "count": len(rows), "rows": rows}


def main() -> int:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.getenv("MILVUS_URL"), timeout=10)
    names = {
        "chunks": os.getenv("CHUNKS_COLLECTION", "chunks_collection"),
        "item_name": os.getenv("ITEM_NAME_COLLECTION", "item_name_collection"),
        "entity": os.getenv("ENTITY_NAME_COLLECTION", "entity_collection"),
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "reports" / f"milvus_backup_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for key, name in names.items():
        data = dump_collection(client, name)
        (out_dir / f"{key}__{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8"
        )
        summary[key] = {"collection": name, "exists": data.get("exists"), "count": data.get("count", 0)}
        if data.get("error"):
            summary[key]["error"] = data["error"]

    print(json.dumps({"dir": str(out_dir), "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
