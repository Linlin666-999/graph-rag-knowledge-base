#!/usr/bin/env python
"""运行 data/qa_set.json，调用查询服务并做关键词评分。

用法（项目根目录，查询服务 :8001 已启动）：
    PYTHONIOENCODING=utf-8 PYTHONPATH=. \
      F:/Program/miniconda/envs/graph-rag/python.exe scripts/run_qa_set.py

常用参数：
    --ids DW-01,WM-03     只跑指定题号
    --limit 5             只跑前 5 题
    --base-url http://127.0.0.1:8001
    --timeout 900         单题超时秒数
    --retries 3           连接失败重试次数
    --output-dir reports  结果输出目录

输出：
    reports/qa_run_<timestamp>.json  逐题答案、耗时、命中情况
    reports/qa_run_<timestamp>.md    人读摘要
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def contains(text: str, token: str) -> bool:
    """大小写不敏感的子串判断；中文与数字不受影响。"""
    return token in text or token.casefold() in text.casefold()


def score_item(item: dict, answer: str) -> dict:
    must = item.get("must_contain") or []
    groups = item.get("must_contain_any") or []
    refusal_markers = item.get("_refusal_markers") or []

    missing = [token for token in must if not contains(answer, token)]
    group_results = [
        {"any_of": group, "hit": any(contains(answer, token) for token in group)}
        for group in groups
    ]
    failed_groups = [g for g in group_results if not g["hit"]]
    refused = any(marker in answer for marker in refusal_markers)

    total_checks = len(must) + len(groups)
    passed_checks = (len(must) - len(missing)) + (len(groups) - len(failed_groups))

    if refused:
        status = "REFUSAL"
    elif total_checks and passed_checks == total_checks:
        status = "PASS"
    elif total_checks and passed_checks / total_checks >= 0.5:
        status = "PARTIAL"
    elif total_checks == 0:
        status = "UNSCORED"
    else:
        status = "FAIL"

    return {
        "status": status,
        "missing": missing,
        "failed_groups": failed_groups,
        "group_results": group_results,
        "refused": refused,
        "checks_passed": passed_checks,
        "checks_total": total_checks,
    }


def call_query(base_url: str, question: str, timeout: int, retries: int) -> tuple[dict, float]:
    payload = json.dumps({"query": question, "is_stream": False}, ensure_ascii=False).encode("utf-8")
    last_error = None
    started = time.time()
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            base_url.rstrip("/") + "/query",
            data=payload,
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8")), time.time() - started
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            last_error = f"HTTP {exc.code}: {body}"
        except Exception as exc:  # noqa: BLE001 - 网络/解析错误统一重试
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            wait = 5 * (attempt + 1)
            print(f"    重试 {attempt + 1}/{retries}（{last_error[:100]}），{wait}s 后再试 ...")
            time.sleep(wait)
    raise RuntimeError(last_error or "query failed")


def write_markdown(path: Path, run: dict) -> None:
    lines = [
        f"# QA Set 运行报告（{run['started_at']}）",
        "",
        f"- 数据文件：`{run['qa_set']}`",
        f"- 查询服务：`{run['base_url']}`",
        f"- 题数：**{len(run['results'])}**",
        f"- 通过 / 部分 / 失败 / 拒答 / 错误："
        f"**{run['summary']['PASS']} / {run['summary']['PARTIAL']} / "
        f"{run['summary']['FAIL']} / {run['summary']['REFUSAL']} / {run['summary']['ERROR']}**",
        "",
        "| 题号 | 商品 | 状态 | 耗时(s) | 问题 | 命中 |",
        "|---|---|---|---:|---|---|",
    ]
    for r in run["results"]:
        hit = f"{r['score']['checks_passed']}/{r['score']['checks_total']}" if r["score"] else "-"
        question = r["question"].replace("|", "\\|")
        lines.append(
            f"| {r['id']} | {r['item_name']} | {r['status']} | {r['elapsed_s']:.1f} | {question} | {hit} |"
        )
    lines += ["", "## 逐题答案", ""]
    for r in run["results"]:
        lines += [
            f"### {r['id']} · {r['status']}",
            "",
            f"**问题**：{r['question']}",
            "",
            f"**答案**：",
            "",
            (r.get("answer") or "（空）").strip(),
            "",
        ]
        if r.get("error"):
            lines += [f"**错误**：`{r['error']}`", ""]
        if r["score"] and (r["score"]["missing"] or r["score"]["failed_groups"]):
            lines += [f"**未命中**：`{json.dumps(r['score']['missing'], ensure_ascii=False)}` / "
                      f"`{json.dumps(r['score']['failed_groups'], ensure_ascii=False)}`", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 qa_set 并评分")
    parser.add_argument("--qa-set", default=str(ROOT / "data" / "qa_set.json"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--ids", default="", help="逗号分隔的题号，如 DW-01,WM-03")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--output-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    qa_path = Path(args.qa_set)
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    items = data["items"]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [i for i in items if i["id"] in wanted]
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("没有匹配的题目")
        return 2

    refusal_markers = (data.get("meta", {}).get("scoring", {}) or {}).get("refusal_markers", [])
    for item in items:
        item["_refusal_markers"] = refusal_markers

    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] {item['id']} {item['question']}")
        result = {
            "id": item["id"],
            "item_name": item["item_name"],
            "question": item["question"],
            "expected_answer": item.get("expected_answer", ""),
            "source": item.get("source", {}),
            "answer": "",
            "error": "",
            "elapsed_s": 0.0,
            "status": "ERROR",
            "score": None,
        }
        try:
            resp, elapsed = call_query(args.base_url, item["question"], args.timeout, args.retries)
            answer = (resp.get("answer") or "").strip()
            result["answer"] = answer
            result["elapsed_s"] = elapsed
            result["error"] = (resp.get("error") or "").strip()
            if result["error"]:
                result["status"] = "ERROR"
            else:
                result["score"] = score_item(item, answer)
                result["status"] = result["score"]["status"]
        except Exception as exc:  # noqa: BLE001
            result["error"] = str(exc)
            result["elapsed_s"] = 0.0
            result["status"] = "ERROR"
        print(f"    -> {result['status']}  {result['elapsed_s']:.1f}s  "
              f"{(result['answer'] or result['error'])[:120]}")
        results.append(result)

    summary = {k: 0 for k in ("PASS", "PARTIAL", "FAIL", "REFUSAL", "ERROR", "UNSCORED")}
    for r in results:
        summary[r["status"]] = summary.get(r["status"], 0) + 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "qa_set": str(qa_path),
        "base_url": args.base_url,
        "started_at": started_at,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "results": results,
    }
    json_path = out_dir / f"qa_run_{stamp}.json"
    md_path = out_dir / f"qa_run_{stamp}.md"
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, run)

    print("\n汇总:", json.dumps(summary, ensure_ascii=False))
    print("JSON:", json_path)
    print("MD  :", md_path)
    return 0 if summary.get("PASS", 0) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
