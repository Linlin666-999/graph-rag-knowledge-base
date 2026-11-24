#!/usr/bin/env python
"""导入后结果核查：Milvus / Neo4j / MinIO 明细。

用法（项目根目录）：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. \
      F:/Program/miniconda/envs/graph-rag/python.exe scripts/inspect_import_result.py [item_name关键字]
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=ROOT / ".env")


def milvus_detail() -> dict:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.getenv("MILVUS_URL"), timeout=10)
    out = {}
    for key, env_name in (
        ("chunks", "CHUNKS_COLLECTION"),
        ("item_name", "ITEM_NAME_COLLECTION"),
        ("entity", "ENTITY_NAME_COLLECTION"),
    ):
        name = os.getenv(env_name)
        if not name or not client.has_collection(name):
            out[key] = {"exists": False}
            continue
        rows = client.query(name, filter="", output_fields=["count(*)"])
        count = int(rows[0]["count(*)"]) if rows else 0
        detail = {"exists": True, "count": count}
        if key == "item_name":
            detail["rows"] = client.query(
                name, filter="", output_fields=["file_title", "item_name"], limit=100
            )
        elif key == "chunks":
            detail["file_titles"] = client.query(
                name, filter="", output_fields=["file_title", "item_name"],
                limit=16384,
            )
            sample = client.query(
                name, filter="", output_fields=["chunk_id", "title", "file_title", "item_name", "content"],
                limit=3,
            )
            for s in sample:
                s["content"] = (s.get("content") or "")[:300]
            detail["sample"] = sample
        elif key == "entity":
            detail["item_names"] = client.query(
                name, filter="", output_fields=["item_name", "entity_name"], limit=16384
            )
        out[key] = detail
    return out


def neo4j_detail() -> dict:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as s:
            out = {}
            out["nodes"] = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            out["rels"] = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            out["by_item"] = s.run(
                "MATCH (n) WHERE n.item_name IS NOT NULL "
                "RETURN n.item_name AS item_name, count(*) AS c ORDER BY c DESC"
            ).data()
            out["rel_types"] = s.run(
                "MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC"
            ).data()
            out["sample_rels"] = s.run(
                "MATCH (h:Entity)-[r]->(t) RETURN h.name AS head, type(r) AS type, t.name AS tail "
                "LIMIT 15"
            ).data()
            return out
    finally:
        driver.close()


def minio_detail() -> dict:
    from minio import Minio

    client = Minio(
        os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    objs = list(itertools.islice(client.list_objects(os.getenv("MINIO_BUCKET_NAME"), recursive=True), 5000))
    return {"objects": len(objs), "latest": [o.object_name for o in objs[-6:]]}


def md_files() -> list:
    base = ROOT / "knowledge" / "processor" / "import_process" / "temp_data"
    found = []
    for p in base.rglob("*.md"):
        if p.stat().st_size > 0:
            text = p.read_text(encoding="utf-8", errors="replace")
            found.append({
                "path": str(p),
                "size": p.stat().st_size,
                "chars": len(text),
                "head": text[:300],
            })
    return found[-3:]


if __name__ == "__main__":
    result = {
        "milvus": milvus_detail(),
        "neo4j": neo4j_detail(),
        "minio": minio_detail(),
        "md": md_files(),
    }
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))
