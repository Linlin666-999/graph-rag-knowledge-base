"""NER 节点离线测试（不依赖 LLM / Milvus / Neo4j）。

运行方式（项目根目录）：
    python knowledge/test/ner/test_ner_node.py

说明：测试里把 LLM 抽取替换为空列表，只验证词典兜底、输出格式和两侧节点接线。
真实 LLM 抽取需要 .env 中的 API 配置可用。
"""

import json
import sys
from pathlib import Path

# 允许从项目根目录直接运行
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.utils.ner_node import (  # noqa: E402
    EntityDictionary,
    NERNode,
    NERService,
    get_ner_entities_for_chunk,
)

TEXT = (
    "用户在苏泊尔官方旗舰店购买RS-12数字万用表，使用满减活动后支付了199元，"
    "订单显示已发货；用户咨询七天无理由退货和积分兑换规则。"
)


def main() -> None:
    # 1. 词典兜底：强制 LLM 返回空，验证词典召回与输出格式
    service = NERService(dictionary=EntityDictionary())
    service._extract_by_llm = lambda _text: []

    entities = service.extract(TEXT)
    print("=== 输入 ===")
    print(TEXT)
    print("=== NER 输出（词典兜底） ===")
    print(json.dumps(entities, ensure_ascii=False, indent=2))

    assert isinstance(entities, list) and entities
    for entity in entities:
        assert set(entity.keys()) == {"type", "text", "normalized"}
        assert entity["type"] in {
            "商品", "商户", "积分规则", "订单", "售后", "活动", "账户安全"
        }

    # 2. 导入侧节点：chunk 内容哈希 -> 实体列表
    import_node = NERNode(mode="import")
    import_node._service.extract = lambda _text: entities
    import_state = import_node.process({"chunks": [{"content": TEXT}]})
    by_chunk = import_state["ner_entities_by_chunk"]
    assert get_ner_entities_for_chunk(by_chunk, TEXT) == entities
    print("\n=== 导入侧 state ===")
    print(json.dumps({
        "ner_entities": import_state["ner_entities"],
        "ner_entities_by_chunk_keys": list(by_chunk.keys()),
    }, ensure_ascii=False, indent=2))

    # 3. 查询侧节点：rewritten_query -> ner_entities
    query_node = NERNode(mode="query")
    query_node._service.extract = lambda _text: entities
    query_state = query_node.process({"rewritten_query": TEXT})
    assert query_state["ner_entities"] == entities
    print("\n=== 查询侧 state ===")
    print(json.dumps({"ner_entities": query_state["ner_entities"]}, ensure_ascii=False, indent=2))

    # 4. 进度上报：running/done 列表 + 中文映射 + SSE progress 事件
    from knowledge.utils import task_util
    from knowledge.utils.sse_util import create_sse_queue, get_sse_queue

    task_id = "test_task_ner"
    create_sse_queue(task_id)
    streaming_node = NERNode(mode="query")
    streaming_node._service.extract = lambda _text: entities
    streaming_node({
        "task_id": task_id,
        "is_stream": True,
        "rewritten_query": TEXT,
    })

    assert task_util.get_done_task_list(task_id) == ["实体识别"]
    assert task_util.get_running_task_list(task_id) == []

    queue = get_sse_queue(task_id)
    progress_events = []
    while not queue.empty():
        event = queue.get()
        assert event["event"] == "progress"
        progress_events.append(event["data"])
    assert len(progress_events) == 2
    assert progress_events[0]["running_list"] == ["实体识别"]
    assert progress_events[-1]["done_list"] == ["实体识别"]
    print("\n=== 进度上报 ===")
    print(json.dumps(progress_events, ensure_ascii=False, indent=2))

    # 5. 导入侧：有 task_id 但非流式，只写 running/done，不建 SSE 队列
    import_task_id = "test_task_ner_import"
    import_node_with_task = NERNode(mode="import")
    import_node_with_task._service.extract = lambda _text: entities
    import_node_with_task({"task_id": import_task_id, "chunks": [{"content": TEXT}]})
    assert task_util.get_done_task_list(import_task_id) == ["实体识别"]
    assert get_sse_queue(import_task_id) is None
    print("\n=== 导入侧进度 ===")
    print(json.dumps({"done_list": task_util.get_done_task_list(import_task_id)}, ensure_ascii=False))

    print("\nALL OK")


if __name__ == "__main__":
    main()
