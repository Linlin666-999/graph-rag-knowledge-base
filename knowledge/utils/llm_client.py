import json

from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv
load_dotenv()

DEFAULT_API_BASE = "https://api.deepseek.com"


def get_llm_client(model_name: str = None, json_mode: bool = False):
    """
    创建 DeepSeek 客户端（通过 API，base_url / key / 模型名均在 .env 中配置）

    Args:
        model_name: 模型名。为 None 时读取 .env 的 MODEL（回落 deepseek-chat）。
        json_mode: 是否强制模型输出 JSON。
                   注意: DeepSeek 要求开启 json_object 时, 提示词里必须出现
                   "json" 这个单词, 否则会直接返回 400。
                   本项目的商品名识别要求"纯净输出", 所以默认关闭。

    Returns: llm客户端对象

    """

    if model_name is None:
        model_name = os.getenv("MODEL", "") or "deepseek-chat"

    api_base = os.getenv("DEEPSEEK_API_BASE", "") or DEFAULT_API_BASE
    api_key = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("API_KEY", "")

    model_kwargs = {}
    if json_mode:
        model_kwargs["response_format"] = {"type": "json_object"}

    client = ChatDeepSeek(
        model=model_name,
        api_base=api_base,
        api_key=api_key,
        model_kwargs=model_kwargs
    )

    return client


