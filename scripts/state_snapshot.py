#!/usr/bin/env python
"""导入冒烟测试用的状态快照。

统计 Milvus 三个集合行数、Neo4j 节点/关系数、MinIO 对象数、Mongo 会话数，
输出一行 JSON，便于导入前后做差。

用法（项目根目录）：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. \
      F:/Program/miniconda/envs/graph-rag/python.exe scripts/state_snapshot.py
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


def milvus_counts() -> dict:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=os.getenv("MILVUS_URL"), timeout=10)
    out = {}
    for key, env_name in (
        ("chunks", "CHUNKS_COLLECTION"),
        ("item_name", "ITEM_NAME_COLLECTION"),
        ("entity", "ENTITY_NAME_COLLECTION"),
    ):
        name = os.getenv(env_name, key + "_collection")
        try:
            rows = client.query(name, filter="", output_fields=["count(*)"])
            out[key] = int(rows[0]["count(*)"]) if rows else 0
        except Exception:
            try:
                out[key] = int(client.get_collection_stats(name).get("row_count", 0))
            except Exception:
                out[key] = -1
    return out


def neo4j_counts() -> dict:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            labels = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c ORDER BY c DESC"
            ).data()
        return {
            "nodes": int(nodes),
            "rels": int(rels),
            "labels": {row["label"]: int(row["c"]) for row in labels},
        }
    finally:
        driver.close()


def minio_count() -> dict:
    from minio import Minio

    client = Minio(
        os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    bucket = os.getenv("MINIO_BUCKET_NAME")
    if not client.bucket_exists(bucket):
        return {"objects": 0, "latest": []}
    objs = list(itertools.islice(client.list_objects(bucket, recursive=True), 5000))
    return {"objects": len(objs), "latest": [o.object_name for o in objs[-5:]]}


def mongo_count() -> dict:
    from pymongo import MongoClient

    client = MongoClient(os.getenv("MONGO_URL"), serverSelectionTimeoutMS=3000)
    try:
        db = client[os.getenv("MONGO_DB_NAME")]
        return {"chat_message": db["chat_message"].count_documents({})}
    finally:
        client.close()


def main() -> int:
    snapshot = {
        "milvus": milvus_counts(),
        "neo4j": neo4j_counts(),
        "minio": minio_count(),
        "mongo": mongo_count(),
    }
    print(json.dumps(snapshot, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
