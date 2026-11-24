# 本地修改日志

> 记录本机对项目做的修改，便于回滚与追溯。

## 2026-11-20 18:24:41
- 更新 `.env` 的 API_KEY，改为当前运行中的本地 LLM 代理（`D:/userapi/w2api-panel-v1.15.8-windows-amd64/config.json`）使用的密钥 `sk-Lv6***ZD0a`。
- 原因：原 `.env` 密钥调用 `http://127.0.0.1:7863/v1/models` 返回 401 invalid_api_key；新密钥实测 `/models` 与 `/chat/completions` 均 200。
- 新增 `scripts/health_check.py`：环境基线体检（config / Milvus / Neo4j / Mongo / MinIO / LLM / 8000 / 8001），输出 Markdown 到 `reports/`。

## 2026-11-20 18:30:37
- Phase 0 基线跑通：启动 milvus-etcd / minio / milvus-standalone / neo4j / mongo，启动导入服务 :8000 与查询服务 :8001。
- `scripts/health_check.py` 全量结果：7 PASS / 1 WARN / 0 FAIL（WARN 仅为 `cn:hy3` 不在代理 `/models` 列表，实际 chat 200 可用）。
- 冒烟查询「数字万用表如何测量直流电压？」非流式返回 HTTP 200，耗时约 40s，`error` 为空，答案内容正常。
- 基线报告：`reports/baseline_20261120_182659.md`。
