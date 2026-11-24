"""Rerank 之后的动态 TopK 截断节点。

插入位置：``rerank_node -> truncate_node -> answer_output_node``。

截断同时施加三个约束（按顺序执行）：

1. 分数阈值：``score < threshold`` 的条目直接丢弃；
   可选自适应阈值（``TRUNCATE_ADAPTIVE_ENABLED=1``）：
   ``score >= mean - factor * std``，其中 factor 默认 0.5；
2. 数量上限：最多保留 ``max_k`` 条；
3. token 预算：累计 token 数超过 ``token_budget`` 即停止追加。

token 计数说明：优先复用项目已有依赖 ``transformers`` 的分词器
（依次尝试 ``BGE_M3_MODEL``、``BGE_RERANKER_LARGE``，要求词表正常），
避免为了计数重复加载 BGE 重排模型；若都不可用，则回退到字符估算
（中文按 1 字 ≈ 1 token、其它字符按 4 字符 ≈ 1 token 的保守估算）。
计数只统计 ``content`` 字段，不包含 Prompt 中的元信息标签。

输入/输出均使用 state 里的 ``reranked_docs``，元素结构与 RerankNode 一致::

    {"content": str, "title": str, "chunk_id": str, "url": str,
     "source": str, "score": float}
"""

import os
import logging
from typing import Any, Dict, List, Tuple

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.state import QueryGraphState

logger = logging.getLogger(__name__)

# 分词器单例（懒加载，避免每次节点执行重复初始化）
_tokenizer = None
_tokenizer_loaded = False


def _get_tokenizer():
    """懒加载 tokenizer：优先复用项目已配置的 BGE 模型分词器。"""
    global _tokenizer, _tokenizer_loaded
    if _tokenizer_loaded:
        return _tokenizer
    _tokenizer_loaded = True

    try:
        from transformers import AutoTokenizer
    except Exception as e:  # transformers 缺失时直接走字符估算
        logger.warning(f"未加载到 transformers({e})，token 计数回退为字符估算")
        return None

    # 1. BGE-M3 与 Reranker 共用 XLM-R 词表，任一可用即可
    candidates = [
        os.getenv("BGE_M3_MODEL", "").strip(),
        os.getenv("BGE_RERANKER_LARGE", "").strip(),
    ]
    for path in candidates:
        if not path:
            continue
        try:
            # 2. 只读本地缓存，避免截断节点触发网络下载
            tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        except Exception:
            continue

        # 3. 防御：缓存不完整时词表会异常小（如 vocab_size=5），不可用于计数
        if getattr(tok, "vocab_size", 0) < 1000:
            logger.warning(f"分词器 {path} 词表异常(vocab_size={getattr(tok, 'vocab_size', 0)})，跳过")
            continue

        _tokenizer = tok
        logger.info(f"截断节点使用分词器: {path}")
        return _tokenizer

    logger.warning("未找到可用分词器，token 计数回退为字符估算")
    return None


def _estimate_tokens(text: str) -> int:
    """字符估算：CJK 按 1 字 1 token，其它字符按 4 字符 1 token（偏保守）。"""
    cjk = sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF)
    other = len(text) - cjk
    return cjk + (other + 3) // 4


class TruncateNode(BaseNode):
    """动态 TopK 截断节点。"""

    name = "truncate_node"

    def __init__(self, config=None):
        super().__init__(config)
        self._threshold = self.config.truncate_score_threshold
        self._max_k = self.config.truncate_max_k
        self._token_budget = self.config.truncate_token_budget
        self._adaptive_enabled = self.config.truncate_adaptive_enabled
        self._adaptive_std_factor = self.config.truncate_adaptive_std_factor

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取 Rerank 后的候选列表（含 score）
        candidates = state.get("reranked_docs") or []

        # 2. 动态截断
        kept_docs = self._truncate(candidates)

        # 3. 写回同一字段，Prompt 组装节点无需改动
        state["reranked_docs"] = kept_docs
        return state

    def _truncate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按「分数阈值 -> 数量上限 -> token 预算」依次截断。"""
        # 0. 过滤无效条目，并预先算好每条内容的 token 数
        docs: List[Dict[str, Any]] = []
        token_counts: List[int] = []
        for doc in candidates:
            if not isinstance(doc, dict):
                continue
            content = str(doc.get("content") or "").strip()
            if not content:
                continue
            docs.append(doc)
            token_counts.append(self._count_tokens(content))

        before_count = len(docs)
        before_tokens = sum(token_counts)
        if not docs:
            self.logger.info("截断前 0 条(tokens=0) | 截断后 0 条(tokens=0)")
            return []

        # 1. 防御性重排：Rerank 已降序，这里保证输入顺序异常时仍然稳定
        order = sorted(
            range(before_count),
            key=lambda i: self._score_of(docs[i]),
            reverse=True,
        )
        docs = [docs[i] for i in order]
        token_counts = [token_counts[i] for i in order]

        # 2. 分数阈值（固定阈值 / 自适应阈值）
        threshold, threshold_desc = self._resolve_threshold(docs)
        after_score_docs: List[Dict[str, Any]] = []
        after_score_tokens: List[int] = []
        dropped_by_score = 0
        for doc, tokens in zip(docs, token_counts):
            score = doc.get("score")
            if score is not None and self._score_of(doc) < threshold:
                dropped_by_score += 1
                continue
            after_score_docs.append(doc)
            after_score_tokens.append(tokens)

        # 3. 数量上限（max_k <= 0 视为不限制）
        if self._max_k and self._max_k > 0 and len(after_score_docs) > self._max_k:
            dropped_by_max_k = len(after_score_docs) - self._max_k
            after_max_k_docs = after_score_docs[: self._max_k]
            after_max_k_tokens = after_score_tokens[: self._max_k]
        else:
            dropped_by_max_k = 0
            after_max_k_docs = after_score_docs
            after_max_k_tokens = after_score_tokens

        # 4. token 预算（token_budget <= 0 视为不限制）
        kept_docs: List[Dict[str, Any]] = []
        kept_tokens = 0
        if self._token_budget and self._token_budget > 0:
            for idx, (doc, tokens) in enumerate(
                zip(after_max_k_docs, after_max_k_tokens)
            ):
                if kept_tokens + tokens > self._token_budget:
                    # 4.1 至少保留 1 条，避免整个上下文为空
                    if idx == 0:
                        kept_docs.append(doc)
                        kept_tokens += tokens
                        self.logger.warning(
                            f"首条内容 tokens={tokens} 已超过预算 "
                            f"{self._token_budget}，仍保留 1 条"
                        )
                    break
                kept_docs.append(doc)
                kept_tokens += tokens
        else:
            kept_docs = list(after_max_k_docs)
            kept_tokens = sum(after_max_k_tokens)

        dropped_by_token = len(after_max_k_docs) - len(kept_docs)

        # 5. 输出统计日志
        self.logger.info(
            f"截断前 {before_count} 条(tokens={before_tokens})"
            f" | {threshold_desc}"
            f" | 分数阈值丢弃 {dropped_by_score} 条"
            f" | 数量上限({self._max_k})丢弃 {dropped_by_max_k} 条"
            f" | token 预算({self._token_budget})截断 {dropped_by_token} 条"
            f" | 截断后 {len(kept_docs)} 条(tokens={kept_tokens})"
        )

        return kept_docs

    def _resolve_threshold(self, docs: List[Dict[str, Any]]) -> Tuple[float, str]:
        """返回本次生效的阈值及说明（自适应 / 固定）。"""
        if self._adaptive_enabled:
            scores = [
                self._score_of(doc)
                for doc in docs
                if doc.get("score") is not None
            ]
            if len(scores) >= 2:
                mean = sum(scores) / len(scores)
                variance = sum((s - mean) ** 2 for s in scores) / len(scores)
                std = variance ** 0.5
                threshold = mean - self._adaptive_std_factor * std
                return threshold, (
                    f"自适应阈值 threshold={threshold:.4f}"
                    f"(mean={mean:.4f}, std={std:.4f}, factor={self._adaptive_std_factor})"
                )
            # 分数不足以计算统计量时回退固定阈值
            return self._threshold, (
                f"固定阈值 threshold={self._threshold}(样本不足，自适应回退)"
            )

        return self._threshold, f"固定阈值 threshold={self._threshold}"

    @staticmethod
    def _score_of(doc: Dict[str, Any]) -> float:
        """安全取分数；缺失/非法分数排到最后。"""
        try:
            return float(doc.get("score"))
        except (TypeError, ValueError):
            return float("-inf")

    def _count_tokens(self, text: str) -> int:
        """统计文本 token 数：优先分词器，失败时字符估算。"""
        tokenizer = _get_tokenizer()
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass
        return _estimate_tokens(text)


if __name__ == "__main__":
    import copy

    from dotenv import load_dotenv

    load_dotenv()
    from knowledge.processor.query_process.base import setup_logging
    from knowledge.processor.query_process.config import QueryConfig

    setup_logging()

    print("=" * 60)
    print("开始测试：动态截断节点（TruncateNode）")
    print("=" * 60)

    mock_state = {
        "reranked_docs": [
            {"chunk_id": "c1", "source": "local", "score": 6.82,
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量各供电电感对地阻值。"},
            {"chunk_id": "c2", "source": "local", "score": 5.15,
             "content": "测量短路时先断开电源，用万用表蜂鸣档分别测量主供电电感的对地阻值，阻值偏低说明存在短路。"},
            {"chunk_id": "c3", "source": "local", "score": 4.37,
             "content": "数字万用表的蜂鸣档用于通断测试，测量前需确认表笔连接正确。"},
            {"chunk_id": "c4", "source": "web", "score": 3.92,
             "content": "主板维修的一般流程：外观检查、供电检测、短路排查、更换损坏元件。"},
            {"chunk_id": "c5", "source": "local", "score": 0.28,
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
            {"chunk_id": "c6", "source": "local", "score": -1.28,
             "content": "万用表使用说明书第 3 页目录。"},
        ]
    }

    def run(title, cfg):
        print(f"\n【{title}】")
        result = TruncateNode(cfg).process(copy.deepcopy(mock_state))
        for i, doc in enumerate(result["reranked_docs"], 1):
            print(f"[{i}] score={doc['score']:.4f} | {doc['content'][:40]}...")

    # 1. 固定阈值 + 数量上限 + 宽松 token 预算
    run("固定阈值 threshold=0.0, max_k=5, token_budget=2000", QueryConfig(
        truncate_score_threshold=0.0,
        truncate_max_k=5,
        truncate_token_budget=2000,
        truncate_adaptive_enabled=False,
    ))

    # 2. 自适应阈值：score >= mean - 0.5*std
    run("自适应阈值 mean-0.5*std, max_k=5, token_budget=2000", QueryConfig(
        truncate_score_threshold=0.0,
        truncate_max_k=5,
        truncate_token_budget=2000,
        truncate_adaptive_enabled=True,
    ))

    # 3. 固定阈值 + 收紧 token 预算，观察预算截断
    run("固定阈值 threshold=0.0, max_k=5, token_budget=100", QueryConfig(
        truncate_score_threshold=0.0,
        truncate_max_k=5,
        truncate_token_budget=100,
        truncate_adaptive_enabled=False,
    ))
