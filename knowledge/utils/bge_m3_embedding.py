import os
import threading

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("BGE_M3_MODEL", "") or r"BAAI/bge-m3"
DEVICE = os.getenv("BGE_M3_DEVICE", "cpu") or "cpu"
USE_FP16 = False  # CPU 环境下禁用 FP16

_init_lock = threading.Lock()
_model = None
_encode_lock = threading.Lock()


def get_bge_m3_embedding():
    """进程内单例，双重检查加锁。别用 lru_cache 代替——它在函数体执行期间不持锁。"""
    global _model
    if _model is None:
        with _init_lock:
            if _model is None:
                _model = BGEM3EmbeddingFunction(
                    model_name=MODEL_NAME, device=DEVICE, use_fp16=USE_FP16,
                )
    return _model


def encode_documents(documents):
    """线程安全入口。所有调用方都走这里，不要再写 model.encode_documents(...)。"""
    with _encode_lock:
        return get_bge_m3_embedding().encode_documents(documents)
