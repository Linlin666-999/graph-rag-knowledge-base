"""统一的实体识别（NER）模块。

导入侧（文档切片之后）和查询侧（问题改写之后）复用同一个 ``NERNode``：

1. LLM 按固定 JSON Schema 抽取七类业务实体；
2. ``config/entity_dict.json`` 领域词典做实体对齐与补召回；
3. LLM 不可用时自动降级为纯词典召回，不阻断导入/问答主链路。

实体输出格式固定为::

    [{"type": "商品", "text": "RS-12", "normalized": "苏泊尔RS-12数字万用表"}]
"""

import hashlib
import json
import logging
import os
import re
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from knowledge.utils.llm_client import get_llm_client
from knowledge.utils.sse_util import push_sse_event
from knowledge.utils.task_util import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
)

# ==================== 固定实体类型体系 ====================
ENTITY_TYPES: Tuple[str, ...] = (
    "商品",
    "商户",
    "积分规则",
    "订单",
    "售后",
    "活动",
    "账户安全",
)
ENTITY_TYPE_SET = set(ENTITY_TYPES)

# 实体名/实体片段的最大长度，避免 LLM 把整句当实体
MAX_ENTITY_TEXT_LENGTH = 50

# 词典默认路径：<项目根>/config/entity_dict.json，可用 ENTITY_DICT_PATH 覆盖
DEFAULT_ENTITY_DICT_PATH = Path(__file__).resolve().parents[2] / "config" / "entity_dict.json"

# 供 LLM 使用的 JSON Schema（写在提示词里约束输出；客户端保持 json_mode）
NER_OUTPUT_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "text": {"type": "string", "description": "原文中出现的实体片段"},
                    "normalized": {"type": "string", "description": "对齐后的标准实体名"},
                },
                "required": ["type", "text", "normalized"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

NER_SYSTEM_PROMPT = """你是电商/零售知识库的实体识别器（NER）。请从给定文本中抽取业务实体，并严格按 JSON 输出。

【实体类型（只能使用这七类）】
- 商品：具体商品、型号、品类名称
- 商户：店铺、商家、品牌方
- 积分规则：积分获取、累计、兑换、抵扣等规则或术语
- 订单：订单状态、下单、支付、取消、物流等订单相关对象
- 售后：退货、换货、退款、维修、保修等售后服务
- 活动：满减、秒杀、优惠券、促销等营销活动
- 账户安全：登录、密码、实名认证、账户冻结、风控等

【抽取规则】
1. text 必须是原文中出现的实体片段，不要改写，长度不超过 50 字；normalized 是标准名称，优先使用下方词典中的标准名。
2. 同一个实体只输出一次；type 必须是上述七类之一。
3. 只抽取文本中明确出现的实体，禁止臆造。
4. 没有实体时输出 {{"entities": []}}。
5. 只输出合法 JSON，不要 Markdown 代码块，不要解释。

【领域词典提示（优先对齐这些标准名）】
{dictionary_hint}

【输出 JSON Schema】
{schema}

【示例】
输入：用户在苏泊尔官方旗舰店购买RS-12数字万用表，使用满减活动后支付了199元，订单显示已发货；用户咨询七天无理由退货和积分兑换规则。
输出：
{{"entities":[
  {{"type":"商户","text":"苏泊尔官方旗舰店","normalized":"苏泊尔官方旗舰店"}},
  {{"type":"商品","text":"RS-12数字万用表","normalized":"苏泊尔RS-12数字万用表"}},
  {{"type":"活动","text":"满减活动","normalized":"满减活动"}},
  {{"type":"订单","text":"订单","normalized":"订单"}},
  {{"type":"售后","text":"七天无理由退货","normalized":"七天无理由退货"}},
  {{"type":"积分规则","text":"积分兑换规则","normalized":"积分兑换规则"}}
]}}
"""


def _normalize_match_text(text: str) -> str:
    """词典匹配用的归一化：去空白 + 转小写（中文不受影响）。"""
    return re.sub(r"\s+", "", str(text or "")).lower()


def content_hash(content: str) -> str:
    """计算 chunk 内容哈希，用于把 NER 结果和原 chunk 稳定关联起来。"""
    return hashlib.sha1(str(content or "").strip().encode("utf-8")).hexdigest()


def get_ner_entities_for_chunk(ner_entities_by_chunk: Optional[Dict[str, List[Dict[str, str]]]],
                               content: str) -> List[Dict[str, str]]:
    """按 chunk 内容哈希取出 NER 结果（导入侧 kg_graph_node 使用）。"""
    if not ner_entities_by_chunk:
        return []
    return ner_entities_by_chunk.get(content_hash(content)) or []


class EntityDictionary:
    """领域词典：实体对齐 + 补召回。

    词典文件结构见 ``config/entity_dict.json``，支持两种写法：

    - ``entities``：{类型: [{canonical, aliases}]} 或 {类型: {canonical: [aliases]}}
    - ``business_terms``：{类型: [术语, ...]}，术语自身作为 canonical
    """

    def __init__(self, path: Optional[str] = None):
        self._path = Path(path) if path else Path(os.getenv("ENTITY_DICT_PATH", "") or DEFAULT_ENTITY_DICT_PATH)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._entries: List[Dict[str, Any]] = []
        self._alias_index: Dict[str, Tuple[str, str]] = {}
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> None:
        if not self._path.exists():
            self._logger.warning(f"领域词典不存在，NER 仅使用 LLM 结果: {self._path}")
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            self._logger.error(f"领域词典加载失败，NER 仅使用 LLM 结果: {e}")
            return

        if not isinstance(raw, dict):
            self._logger.error("领域词典格式错误：根节点必须是 JSON 对象")
            return

        self._add_section(raw.get("entities"))
        self._add_section(raw.get("business_terms"))
        self._logger.info(f"领域词典加载完成: {len(self._entries)} 条实体/术语 ({self._path})")

    def _add_section(self, section: Any) -> None:
        if not isinstance(section, dict):
            return

        for type_name, items in section.items():
            if type_name not in ENTITY_TYPE_SET:
                self._logger.warning(f"词典中出现未知实体类型，已忽略: {type_name}")
                continue

            # 写法 1：{"canonical": ["alias1", "alias2"]}
            if isinstance(items, dict):
                for canonical, aliases in items.items():
                    self._add_entry(type_name, canonical, aliases if isinstance(aliases, list) else [])
                continue

            # 写法 2：[{"canonical": "...", "aliases": [...]}] 或 ["术语1", "术语2"]
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, str):
                        self._add_entry(type_name, item, [])
                    elif isinstance(item, dict):
                        canonical = item.get("canonical") or item.get("name") or item.get("term")
                        aliases = item.get("aliases") or item.get("alias") or []
                        if isinstance(aliases, str):
                            aliases = [aliases]
                        self._add_entry(type_name, canonical, aliases if isinstance(aliases, list) else [])

    def _add_entry(self, type_name: str, canonical: Any, aliases: Iterable[Any]) -> None:
        canonical = str(canonical or "").strip()
        if not canonical:
            return

        alias_list: List[str] = []
        for alias in [canonical, *list(aliases or [])]:
            alias = str(alias or "").strip()
            if alias and alias not in alias_list:
                alias_list.append(alias)

        self._entries.append({
            "type": type_name,
            "canonical": canonical,
            "aliases": alias_list,
        })

        for alias in alias_list:
            key = _normalize_match_text(alias)
            # 单字符别名误召回风险太高，不进入索引
            if len(key) < 2 or key in self._alias_index:
                continue
            self._alias_index[key] = (type_name, canonical)

    def align(self, type_name: str, text: str, normalized: str) -> Tuple[str, str]:
        """把 LLM 抽到的实体对齐到词典标准名；词典优先于 LLM 的类型/名称。"""
        for candidate in (normalized, text):
            key = _normalize_match_text(candidate)
            if key in self._alias_index:
                return self._alias_index[key]

        # 名称不是词典别名时，再尝试“文本中包含某个别名”
        matches = self.recall(text)
        if matches:
            same_type = [m for m in matches if m["type"] == type_name]
            picked = same_type[0] if same_type else matches[0]
            return picked["type"], picked["normalized"]

        return type_name, normalized or text

    def recall(self, text: str) -> List[Dict[str, str]]:
        """从文本中扫描词典别名，返回补召回实体（保持去重）。"""
        if not text:
            return []

        normalized_text = _normalize_match_text(text)
        result: List[Dict[str, str]] = []
        seen = set()

        for entry in self._entries:
            hit_alias = None
            # 长别名优先，避免 "积分" 先命中 "积分兑换规则"
            for alias in sorted(entry["aliases"], key=len, reverse=True):
                alias_key = _normalize_match_text(alias)
                if len(alias_key) < 2:
                    continue
                if alias_key in normalized_text:
                    hit_alias = alias
                    break

            if not hit_alias:
                continue

            key = (entry["type"], entry["canonical"])
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "type": entry["type"],
                "text": hit_alias,
                "normalized": entry["canonical"],
            })

        # 同类实体中，短别名被更长实体覆盖时丢弃，避免"积分/兑换/退货"这类碎片刷屏
        filtered: List[Dict[str, str]] = []
        for entity in result:
            entity_text = _normalize_match_text(entity["text"])
            covered = any(
                entity is not other
                and entity["type"] == other["type"]
                and entity_text != _normalize_match_text(other["text"])
                and entity_text in _normalize_match_text(other["text"])
                for other in result
            )
            if not covered:
                filtered.append(entity)

        # 同一类型下相同 text 只保留一条（防止同一别名配置到多个 canonical）
        final: List[Dict[str, str]] = []
        seen_text = set()
        for entity in filtered:
            key = (entity["type"], _normalize_match_text(entity["text"]))
            if key in seen_text:
                continue
            seen_text.add(key)
            final.append(entity)

        return final

    def prompt_hint(self, limit_per_type: int = 12) -> str:
        """生成给 LLM 的词典提示，只放标准名，控制提示词长度。"""
        lines: List[str] = []
        for type_name in ENTITY_TYPES:
            names: List[str] = []
            for entry in self._entries:
                if entry["type"] != type_name:
                    continue
                if entry["canonical"] not in names:
                    names.append(entry["canonical"])
                if len(names) >= limit_per_type:
                    break
            if names:
                lines.append(f"- {type_name}：{'、'.join(names)}")
        return "\n".join(lines) or "（词典为空）"


class NERService:
    """NER 核心实现：LLM 抽取 + 词典对齐/补召回。"""

    def __init__(self, dictionary: Optional[EntityDictionary] = None,
                 logger: Optional[logging.Logger] = None):
        self._dictionary = dictionary or EntityDictionary()
        self._logger = logger or logging.getLogger(self.__class__.__name__)

    def extract(self, text: str) -> List[Dict[str, str]]:
        """输入一段文本，输出 ``[{"type", "text", "normalized"}]``。"""
        text = str(text or "").strip()
        if not text:
            return []

        llm_entities = self._extract_by_llm(text)
        dict_entities = self._dictionary.recall(text)
        merged = self._merge(llm_entities, dict_entities)

        self._logger.info(
            "NER 完成: LLM %d 个, 词典补召回 %d 个, 合并去重后 %d 个",
            len(llm_entities), len(dict_entities), len(merged),
        )
        return merged

    def _extract_by_llm(self, text: str) -> List[Dict[str, str]]:
        llm_client = get_llm_client(json_mode=True)
        if llm_client is None:
            self._logger.warning("LLM 客户端不可用，NER 降级为词典召回")
            return []

        system_prompt = NER_SYSTEM_PROMPT.format(
            dictionary_hint=self._dictionary.prompt_hint(),
            schema=json.dumps(NER_OUTPUT_JSON_SCHEMA, ensure_ascii=False, indent=2),
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"待识别文本：\n{text}"),
        ]

        # 轻量重试：失败不阻断主链路，最终由词典兜底
        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            try:
                response = llm_client.invoke(messages)
                content = getattr(response, "content", "")
                if not isinstance(content, str):
                    content = str(content)
                entities = self._parse_llm_entities(content.strip())
                # 解析成功（包括合法的空列表）就直接返回；只有解析失败才重试
                if entities is not None:
                    return entities
            except Exception as e:
                self._logger.warning(f"NER LLM 第 {attempt} 次调用失败: {e}")
                if attempt < max_attempts:
                    time.sleep(0.5 * attempt)

        return []

    def _parse_llm_entities(self, content: str) -> Optional[List[Dict[str, str]]]:
        """清洗围栏、反序列化、校验七类实体并去重。

        返回 ``None`` 表示 LLM 输出无法解析（可重试）；返回列表表示解析成功，
        列表可能为空（文本中确实没有实体）。
        """
        if not content:
            return None

        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        parsed: Any = None
        try:
            parsed = json.loads(cleaned)
        except JSONDecodeError:
            # 模型偶尔会在 JSON 前后带解释文字，退化为截取第一个 { 到最后一个 }
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    parsed = json.loads(cleaned[start:end + 1])
                except JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, dict):
            self._logger.warning("NER LLM 输出不是合法 JSON，已忽略")
            return None

        raw_entities = parsed.get("entities")
        if not isinstance(raw_entities, list):
            return None

        result: List[Dict[str, str]] = []
        seen = set()
        for item in raw_entities:
            if not isinstance(item, dict):
                continue

            type_name = str(item.get("type") or "").strip()
            if type_name not in ENTITY_TYPE_SET:
                continue

            text = str(item.get("text") or "").strip()
            normalized = str(item.get("normalized") or "").strip() or text
            if not text or not normalized:
                continue

            text = text[:MAX_ENTITY_TEXT_LENGTH]
            normalized = normalized[:MAX_ENTITY_TEXT_LENGTH]

            key = (type_name, normalized)
            if key in seen:
                continue
            seen.add(key)
            result.append({"type": type_name, "text": text, "normalized": normalized})

        return result

    def _merge(self, llm_entities: List[Dict[str, str]],
               dict_entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """LLM 结果先按词典对齐，再补入词典召回但 LLM 漏掉的实体。"""
        result: List[Dict[str, str]] = []
        seen = set()

        for entity in llm_entities:
            type_name, normalized = self._dictionary.align(
                entity["type"], entity["text"], entity["normalized"]
            )
            if type_name not in ENTITY_TYPE_SET:
                continue

            key = (type_name, normalized)
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "type": type_name,
                "text": entity["text"],
                "normalized": normalized,
            })

        for entity in dict_entities:
            key = (entity["type"], entity["normalized"])
            if key in seen:
                continue
            seen.add(key)
            result.append(entity)

        return result


class NERNode:
    """LangGraph 通用 NER 节点，导入侧和查询侧共用同一个实现。

    - ``mode="import"``：读取 ``state["chunks"]``，逐 chunk 抽取，写入
      ``state["ner_entities_by_chunk"]``（content_hash -> 实体列表）和
      ``state["ner_entities"]``（全局去重实体列表）。
    - ``mode="query"``：读取 ``state["rewritten_query"]``（缺失时回退
      ``original_query``），写入 ``state["ner_entities"]``。
    """

    name = "ner_node"

    def __init__(self, mode: str = "query", dict_path: Optional[str] = None,
                 service: Optional[NERService] = None):
        if mode not in ("import", "query"):
            raise ValueError(f"NERNode mode 只支持 import/query，当前为: {mode}")

        self.mode = mode
        self.logger = logging.getLogger(f"ner.{mode}")
        self._service = service or NERService(
            dictionary=EntityDictionary(dict_path),
            logger=self.logger,
        )

    def __call__(self, state: Any) -> Any:
        """LangGraph 节点入口：上报任务进度后再执行 NER。

        NERNode 同时被导入图和查询图复用，而两条流程的 BaseNode 并不是同一个类，
        因此这里直接复用 task_util / sse_util，实现与 BaseNode 一致的进度上报：

        - 导入侧：写入 running / done 列表，供 ``/status/{task_id}`` 轮询；
        - 查询侧：``is_stream=True`` 时额外推送 SSE ``progress`` 事件。

        注意：``process`` 内部已对 NER 异常做降级，不会阻断导入/问答主链路。
        """
        task_id = state.get("task_id", "") if isinstance(state, dict) else ""
        is_stream = bool(state.get("is_stream", False)) if isinstance(state, dict) else False

        if task_id:
            add_running_task(task_id, self.name)
            if is_stream:
                self._push_progress(task_id)

        try:
            return self.process(state)
        finally:
            # 无论成功、降级还是意外异常，都要把节点移出 running，避免进度条卡住
            if task_id:
                add_done_task(task_id, self.name)
                if is_stream:
                    self._push_progress(task_id)

    @staticmethod
    def _push_progress(task_id: str) -> None:
        """推送节点进度（与查询流程 BaseNode._push_progress 保持一致）。"""
        push_sse_event(task_id, "progress", {
            "status": get_task_status(task_id),
            "done_list": get_done_task_list(task_id),
            "running_list": get_running_task_list(task_id),
        })

    def process(self, state: Any) -> Any:
        try:
            if self.mode == "import":
                return self._process_import(state)
            return self._process_query(state)
        except Exception as e:
            # NER 是增强环节，任何异常都不应阻断导入/问答主链路
            self.logger.error(f"NER 节点执行失败，降级为空实体: {e}", exc_info=True)
            state["ner_entities"] = state.get("ner_entities") or []
            if self.mode == "import":
                state["ner_entities_by_chunk"] = state.get("ner_entities_by_chunk") or {}
            return state

    def _process_query(self, state: Any) -> Any:
        text = str(state.get("rewritten_query") or state.get("original_query") or "").strip()
        entities = self._service.extract(text)
        state["ner_entities"] = entities
        self.logger.info(
            "查询侧 NER 抽取到 %d 个实体: %s",
            len(entities), json.dumps(entities, ensure_ascii=False),
        )
        return state

    def _process_import(self, state: Any) -> Any:
        chunks = state.get("chunks") or []
        ner_entities_by_chunk: Dict[str, List[Dict[str, str]]] = {}
        all_entities: List[Dict[str, str]] = []
        seen = set()

        for chunk in self._iter_chunks(chunks):
            content = str(chunk.get("content") or "").strip()
            if not content:
                continue

            entities = self._service.extract(content)
            ner_entities_by_chunk[content_hash(content)] = entities

            for entity in entities:
                key = (entity["type"], entity["normalized"])
                if key in seen:
                    continue
                seen.add(key)
                all_entities.append(entity)

        state["ner_entities_by_chunk"] = ner_entities_by_chunk
        state["ner_entities"] = all_entities
        self.logger.info(
            "导入侧 NER 完成: %d 个切片, %d 个去重实体",
            len(ner_entities_by_chunk), len(all_entities),
        )
        return state

    @staticmethod
    def _iter_chunks(chunks: Any):
        """兼容扁平的 chunk dict 列表和 MinerU 的按页嵌套列表。"""
        for item in chunks or []:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, dict):
                        yield sub
