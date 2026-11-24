#!/usr/bin/env python
"""环境基线体检脚本（Phase 0）。

一次检查本项目运行所需的全部依赖，并输出可读表格 + Markdown 报告：

    config       .env 关键变量是否齐全（含 DEEPSEEK_API_BASE 变量名坑）
    milvus       Milvus 连接、三个 collection 是否存在、实际行数
    neo4j        Neo4j 连接、节点/关系数量、主要标签
    mongo        MongoDB ping、chat_message 数量
    minio        MinIO 连接、bucket 是否存在、对象数量
    llm          LLM 代理 /models + 一次最小 chat completion
    http:8000    导入服务是否在跑
    http:8001    查询服务是否在跑

用法（项目根目录）：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. \
      F:/Program/miniconda/envs/graph-rag/python.exe scripts/health_check.py

常用参数：
    --no-services   跳过 8000/8001 检查（只查依赖服务）
    --json          额外输出 JSON 结果
    --no-report     不写 reports/ 下的 Markdown 报告

退出码：0 = 无 FAIL（允许 WARN）；1 = 存在 FAIL。
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Windows 控制台默认 GBK，统一成 UTF-8 避免中文详情炸掉
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=ROOT / ".env")

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str
    elapsed_ms: int = 0


@dataclass
class Report:
    results: List[CheckResult] = field(default_factory=list)
    started_at: str = ""

    def add(self, r: CheckResult) -> None:
        self.results.append(r)

    @property
    def counts(self) -> Dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for r in self.results:
            out[r.status] = out.get(r.status, 0) + 1
        return out

    @property
    def green(self) -> bool:
        return self.counts[FAIL] == 0


def _elapsed(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _short(value: Any, limit: int = 400) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _mask(value: Optional[str]) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "***" + value[-2:]


def _run(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    start = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # 单点失败不能拖垮整张体检表
        result = CheckResult(name=name, status=FAIL, detail=f"{type(exc).__name__}: {_short(exc)}")
    result.name = name
    result.elapsed_ms = _elapsed(start)
    result.detail = _short(result.detail)
    return result


# ---------------------------------------------------------------------------
# 各项检查
# ---------------------------------------------------------------------------


def check_config() -> CheckResult:
    required = [
        "DEEPSEEK_API_BASE", "MODEL",
        "MILVUS_URL", "CHUNKS_COLLECTION", "ITEM_NAME_COLLECTION", "ENTITY_NAME_COLLECTION",
        "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
        "MINIO_ENDPOINT", "MINIO_BUCKET_NAME",
        "MONGO_URL", "MONGO_DB_NAME",
    ]
    missing = [k for k in required if not os.getenv(k)]
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        missing.append("DEEPSEEK_API_KEY/API_KEY")

    base = os.getenv("DEEPSEEK_API_BASE", "")
    model = os.getenv("MODEL", "")
    detail = (
        f"base={base or '<empty>'} model={model or '<empty>'} "
        f"key={_mask(api_key)} collections="
        f"{os.getenv('CHUNKS_COLLECTION')}/{os.getenv('ITEM_NAME_COLLECTION')}/"
        f"{os.getenv('ENTITY_NAME_COLLECTION')}"
    )
    if missing:
        return CheckResult("config", FAIL, f"缺少变量: {', '.join(missing)}；{detail}")
    if "api.deepseek.com" in base:
        return CheckResult(
            "config", WARN,
            f"DEEPSEEK_API_BASE 仍指向官方地址，若本意走本地代理请改成 http://127.0.0.1:7863/v1；{detail}",
        )
    return CheckResult("config", PASS, detail)


def check_milvus() -> CheckResult:
    from pymilvus import MilvusClient

    uri = os.getenv("MILVUS_URL")
    client = MilvusClient(uri=uri, timeout=10)
    collections = set(client.list_collections())

    expected = {
        "chunks": os.getenv("CHUNKS_COLLECTION", "chunks_collection"),
        "item_name": os.getenv("ITEM_NAME_COLLECTION", "item_name_collection"),
        "entity": os.getenv("ENTITY_NAME_COLLECTION", "entity_collection"),
    }
    parts: List[str] = []
    status = PASS
    for label, name in expected.items():
        if name not in collections:
            status = WARN
            parts.append(f"{name}: MISSING")
            continue
        count = _milvus_count(client, name)
        if count is None:
            status = WARN
            parts.append(f"{name}: count-unknown")
        else:
            if count == 0:
                status = WARN
            parts.append(f"{name}={count}")
    detail = f"uri={uri}; " + ", ".join(parts) + f"; all={sorted(collections)}"
    return CheckResult("milvus", status, detail)


def _milvus_count(client: Any, collection: str) -> Optional[int]:
    try:
        rows = client.query(collection, filter="", output_fields=["count(*)"])
        if rows:
            return int(rows[0].get("count(*)", 0))
    except Exception:
        pass
    try:
        stats = client.get_collection_stats(collection)
        return int(stats.get("row_count", 0))
    except Exception:
        return None


def check_neo4j() -> CheckResult:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI")
    driver = GraphDatabase.driver(
        uri, auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
    )
    try:
        driver.verify_connectivity()
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        with driver.session(database=database) as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            labels = session.run(
                "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS c "
                "ORDER BY c DESC LIMIT 8"
            ).data()
        label_text = ", ".join(f"{row['label']}={row['c']}" for row in labels) or "none"
        status = PASS if nodes > 0 else WARN
        return CheckResult(
            "neo4j", status,
            f"uri={uri} db={database} nodes={nodes} rels={rels} labels=[{label_text}]",
        )
    finally:
        driver.close()


def check_mongo() -> CheckResult:
    from pymongo import MongoClient

    url = os.getenv("MONGO_URL")
    db_name = os.getenv("MONGO_DB_NAME")
    client = MongoClient(url, serverSelectionTimeoutMS=3000)
    try:
        client.admin.command("ping")
        count = client[db_name]["chat_message"].count_documents({})
        status = PASS if count >= 0 else WARN
        return CheckResult("mongo", status, f"url={url} db={db_name} chat_message={count}")
    finally:
        client.close()


def check_minio() -> CheckResult:
    from minio import Minio

    endpoint = os.getenv("MINIO_ENDPOINT")
    bucket = os.getenv("MINIO_BUCKET_NAME")
    client = Minio(
        endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=False,
    )
    exists = client.bucket_exists(bucket)
    if not exists:
        return CheckResult("minio", WARN, f"endpoint={endpoint} bucket={bucket} 不存在")
    objects = list(itertools.islice(client.list_objects(bucket, recursive=True), 1000))
    status = PASS if objects else WARN
    sample = ", ".join(o.object_name for o in objects[-3:])
    return CheckResult(
        "minio", status,
        f"endpoint={endpoint} bucket={bucket} objects(前1000)={len(objects)}"
        + (f" 最近={sample}" if sample else ""),
    )


def _http_json(
    url: str,
    method: str = "GET",
    payload: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: int = 30,
) -> tuple[int, Any]:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=req_headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        try:
            return resp.status, json.loads(body)
        except json.JSONDecodeError:
            return resp.status, body


def check_llm() -> CheckResult:
    base = (os.getenv("DEEPSEEK_API_BASE") or "").rstrip("/")
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("API_KEY") or ""
    model = os.getenv("MODEL") or "deepseek-chat"
    if not base or not key:
        return CheckResult("llm", FAIL, "DEEPSEEK_API_BASE / API_KEY 未配置")

    headers = {"Authorization": f"Bearer {key}"}
    try:
        status, models = _http_json(f"{base}/models", headers=headers, timeout=10)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        return CheckResult("llm", FAIL, f"GET /models -> HTTP {exc.code}: {body}")
    except Exception as exc:
        return CheckResult("llm", FAIL, f"GET /models 连接失败: {type(exc).__name__}: {_short(exc)}")

    model_names: List[str] = []
    if isinstance(models, dict):
        model_names = [str(m.get("id")) for m in models.get("data", []) if isinstance(m, dict)]
    model_ok = (not model_names) or (model in model_names)
    detail = f"base={base} models_http={status} model={model}"
    if model_names:
        detail += f" available={model_names[:5]}{'…' if len(model_names) > 5 else ''}"
    if not model_ok:
        detail += " [MODEL 不在 /models 列表]"

    try:
        status, chat = _http_json(
            f"{base}/chat/completions",
            method="POST",
            headers=headers,
            payload={
                "model": model,
                "messages": [{"role": "user", "content": "只回复：ok"}],
                "max_tokens": 5,
                "stream": False,
            },
            timeout=45,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        # 代理池偶发 503 no_healthy_account 是已知可恢复问题，记为 WARN
        status_code = WARN if exc.code == 503 else FAIL
        return CheckResult("llm", status_code, f"POST /chat/completions -> HTTP {exc.code}: {body}; {detail}")
    except Exception as exc:
        return CheckResult("llm", FAIL, f"POST /chat/completions 失败: {type(exc).__name__}: {_short(exc)}; {detail}")

    content = ""
    if isinstance(chat, dict):
        try:
            content = chat["choices"][0]["message"]["content"] or ""
        except Exception:
            content = ""
    return CheckResult(
        "llm", PASS if model_ok else WARN,
        f"chat_http={status} reply={_short(content, 40)!r}; {detail}",
    )


def check_http(name: str, url: str) -> CheckResult:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return CheckResult(name, PASS, f"{url} -> HTTP {resp.status}")
    except urllib.error.HTTPError as exc:
        status = PASS if exc.code < 500 else FAIL
        return CheckResult(name, status, f"{url} -> HTTP {exc.code}")
    except Exception as exc:
        return CheckResult(name, FAIL, f"{url} 未就绪: {type(exc).__name__}: {_short(exc)}")


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def print_table(report: Report) -> None:
    name_width = max([len("检查项")] + [len(r.name) for r in report.results])
    print()
    print(f"{'检查项'.ljust(name_width)}  结果   耗时     详情")
    print("-" * (name_width + 60))
    for r in report.results:
        print(f"{r.name.ljust(name_width)}  {r.status:<4}  {r.elapsed_ms:>5}ms  {r.detail}")
    counts = report.counts
    verdict = "GREEN ✅" if report.green else "RED ❌"
    print("-" * (name_width + 60))
    print(
        f"结论: {verdict}  "
        f"PASS={counts[PASS]} WARN={counts[WARN]} FAIL={counts[FAIL]} SKIP={counts[SKIP]}"
    )
    if counts[FAIL]:
        print("失败项:")
        for r in report.results:
            if r.status == FAIL:
                print(f"  - {r.name}: {r.detail}")


def write_markdown(report: Report) -> Path:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"baseline_{stamp}.md"
    counts = report.counts
    lines = [
        "# 环境基线体检报告",
        "",
        f"- 时间: {report.started_at}",
        f"- 结论: **{'GREEN' if report.green else 'RED'}**",
        f"- 汇总: PASS={counts[PASS]} WARN={counts[WARN]} FAIL={counts[FAIL]} SKIP={counts[SKIP]}",
        "",
        "| 检查项 | 结果 | 耗时(ms) | 详情 |",
        "|---|---|---:|---|",
    ]
    for r in report.results:
        detail = r.detail.replace("|", "\\|")
        lines.append(f"| {r.name} | {r.status} | {r.elapsed_ms} | {detail} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Graph RAG 环境基线体检")
    parser.add_argument("--no-services", action="store_true", help="跳过 8000/8001 服务检查")
    parser.add_argument("--json", action="store_true", help="输出 JSON 结果")
    parser.add_argument("--no-report", action="store_true", help="不写 Markdown 报告")
    args = parser.parse_args()

    report = Report(started_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    checks: List[tuple[str, Callable[[], CheckResult]]] = [
        ("config", check_config),
        ("milvus", check_milvus),
        ("neo4j", check_neo4j),
        ("mongo", check_mongo),
        ("minio", check_minio),
        ("llm", check_llm),
    ]
    if not args.no_services:
        checks += [
            ("http:8000", lambda: check_http("http:8000", "http://127.0.0.1:8000/docs")),
            ("http:8001", lambda: check_http("http:8001", "http://127.0.0.1:8001/health")),
        ]

    for name, fn in checks:
        result = _run(name, fn)
        report.add(result)
        print(f"[{result.status}] {name} ({result.elapsed_ms}ms)")

    print_table(report)

    if args.json:
        print(json.dumps(
            {
                "started_at": report.started_at,
                "green": report.green,
                "counts": report.counts,
                "results": [r.__dict__ for r in report.results],
            },
            ensure_ascii=False, indent=2,
        ))

    if not args.no_report:
        path = write_markdown(report)
        print(f"\n报告已写入: {path}")

    return 0 if report.green else 1


if __name__ == "__main__":
    raise SystemExit(main())
