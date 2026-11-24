"""查询流程配置管理模块

集中管理所有配置项，支持环境变量覆盖。所有属性均采用懒加载模式。
"""

from dataclasses import dataclass, field
from typing import Optional
import os

from dotenv import load_dotenv
load_dotenv()


@dataclass
class QueryConfig:
    """查询流程配置。"""

    # ==================== 文本处理配置 ====================
    max_context_chars: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
    )

    # ==================== Rerank 配置 ====================
    rerank_max_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MAX_TOP_K", "10"))
    )
    rerank_min_top_k: int = field(
        default_factory=lambda: int(os.getenv("RERANK_MIN_TOP_K", "3"))
    )
    rerank_gap_ratio: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_RATIO", "0.25"))
    )
    rerank_gap_abs: float = field(
        default_factory=lambda: float(os.getenv("RERANK_GAP_ABS", "0.5"))
    )

    # ==================== 动态截断（Rerank 之后）配置 ====================
    # 分数阈值：低于该值的候选直接丢弃（bge-reranker 原始 logits）
    truncate_score_threshold: float = field(
        default_factory=lambda: float(os.getenv("TRUNCATE_SCORE_THRESHOLD", "0.0"))
    )
    # 数量上限：最多保留条数（<=0 表示不限制）
    truncate_max_k: int = field(
        default_factory=lambda: int(os.getenv("TRUNCATE_MAX_K", "5"))
    )
    # token 预算：累计 token 超过该值即停止追加（<=0 表示不限制）
    truncate_token_budget: int = field(
        default_factory=lambda: int(os.getenv("TRUNCATE_TOKEN_BUDGET", "2000"))
    )
    # 自适应阈值开关：启用后用 mean - factor*std 取代固定阈值
    truncate_adaptive_enabled: bool = field(
        default_factory=lambda: os.getenv("TRUNCATE_ADAPTIVE_ENABLED", "0")
        .strip()
        .lower() in ("1", "true", "yes", "on")
    )
    truncate_adaptive_std_factor: float = field(
        default_factory=lambda: float(os.getenv("TRUNCATE_ADAPTIVE_STD_FACTOR", "0.5"))
    )

    # ==================== RRF 配置 ====================
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )
    rrf_kg_weight: float = field(
        default_factory=lambda: float(os.getenv("RRF_KG_WEIGHT", "0.7"))
    )
    rrf_max_results: int = field(
        default_factory=lambda: int(os.getenv("RRF_MAX_RESULTS", "10"))
    )

    # ==================== 检索配置 ====================
    embedding_search_limit: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_SEARCH_LIMIT", "10"))
    )
    hyde_search_limit: int = field(
        default_factory=lambda: int(os.getenv("HYDE_SEARCH_LIMIT", "5"))
    )

    # ==================== 商品确认节点配置 ====================
    item_name_high_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_HIGH_CONFIDENCE", "0.7")) # 直接给的（压测给到）--->RAG评估（了解）
    )
    item_name_mid_confidence: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_MID_CONFIDENCE", "0.6"))  # 直接给的（压测给到）--->RAG评估（了解）
    )
    item_name_max_options: int = field(
        default_factory=lambda: int(os.getenv("ITEM_NAME_MAX_OPTIONS", "5"))
    )
    item_name_dense_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_DENSE_WEIGHT", "0.5"))
    )
    item_name_sparse_weight: float = field(
        default_factory=lambda: float(os.getenv("ITEM_NAME_SPARSE_WEIGHT", "0.5"))
    )

    # ==================== 知识图谱配置 ====================
    kg_entity_align_min_score: Optional[float] = field(
        default_factory=lambda: (
            float(os.getenv("KG_ENTITY_ALIGN_MIN_SCORE"))
            if os.getenv("KG_ENTITY_ALIGN_MIN_SCORE")
            else None
        )
    )
    kg_max_seed_candidates: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_SEED_CANDIDATES", "3"))
    )
    kg_max_total_seeds: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TOTAL_SEEDS", "30"))
    )
    kg_max_triples_per_seed: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TRIPLES_PER_SEED", "50"))
    )
    kg_max_total_triples: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TOTAL_TRIPLES", "50"))
    )
    kg_max_total_chunks: int = field(
        default_factory=lambda: int(os.getenv("KG_MAX_TOTAL_CHUNKS", "50"))
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("API_KEY", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "") or "deepseek-chat"
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "") or os.getenv("MODEL", "") or "deepseek-chat"
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )

    # ==================== Neo4j 配置 ====================
    neo4j_uri: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "")
    )
    neo4j_username: str = field(
        default_factory=lambda: os.getenv("NEO4J_USERNAME", "")
    )
    neo4j_password: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "")
    )
    neo4j_database: str = field(
        default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j")
    )

    # ==================== MCP 配置 ====================
    mcp_dashscope_base_url: str = field(
        default_factory=lambda: os.getenv("MCP_DASHSCOPE_BASE_URL", "")
    )

    @classmethod
    def from_env(cls) -> "QueryConfig":
        """从环境变量加载配置。

        Returns:
            配置实例。
        """
        return cls()


_config: Optional[QueryConfig] = None


def get_config() -> QueryConfig:
    """获取配置单例。

    Returns:
        全局配置实例。
    """
    global _config
    if _config is None:
        _config = QueryConfig.from_env()
    return _config