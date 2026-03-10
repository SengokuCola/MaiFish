"""
MaiSaka - 记忆模块

提供记忆存储抽象接口和查询功能。
使用本地 JSON 文件存储，支持简单的关键词匹配检索。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum


def _try_load_file_store() -> Optional["BaseMemoryStore"]:
    """
    尝试加载本地文件存储。

    Returns:
        FileMemoryStore 实例，加载失败时返回 None
    """
    try:
        from file_memory_store import FileMemoryStore
        return FileMemoryStore()
    except Exception:
        return None


class QueryType(Enum):
    """查询类型枚举"""
    SEARCH = "search"           # 语义检索
    ENTITY = "entity"           # 实体查询
    RELATION = "relation"       # 关系查询
    TIME = "time"               # 时序检索


@dataclass
class MemoryQuery:
    """记忆查询请求"""
    query_type: QueryType
    query: str                  # 查询内容
    top_k: int = 5              # 返回结果数量
    person_id: Optional[str] = None  # 人物ID（可选）


@dataclass
class MemoryResult:
    """记忆查询结果"""
    content: str                # 结果内容
    score: float = 0.0          # 相关性分数
    metadata: Optional[Dict[str, Any]] = None  # 元数据


class BaseMemoryStore(ABC):
    """
    记忆存储抽象基类。

    定义记忆存储和查询的接口，实现可对接 A_Memorix 或其他存储系统。
    """

    @abstractmethod
    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        存储一条记忆。

        Args:
            content: 记忆内容
            metadata: 元数据（如时间戳、来源等）

        Returns:
            是否存储成功
        """
        ...

    @abstractmethod
    async def query_memory(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        查询记忆。

        Args:
            query: 记忆查询请求

        Returns:
            记忆结果列表
        """
        ...


class MockMemoryStore(BaseMemoryStore):
    """
    模拟记忆存储实现。

    当前为空实现，仅提供接口形式。
    后续可对接 A_Memorix 或其他存储系统。
    """

    def __init__(self):
        """初始化模拟存储"""
        self._memories: List[Dict[str, Any]] = []

    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """存储一条记忆（当前为空实现，返回 False 表示使用 Mock）"""
        # TODO: 对接实际存储系统
        self._memories.append({
            "content": content,
            "metadata": metadata or {},
        })
        return False  # 返回 False 表示使用 Mock 存储

    async def query_memory(self, query: MemoryQuery) -> List[MemoryResult]:
        """查询记忆（当前为空实现）"""
        # TODO: 对接实际存储系统
        return []


# 全局记忆存储实例
_memory_store: Optional[BaseMemoryStore] = None
_tried_load: bool = False  # 是否已尝试加载存储


def get_memory_store() -> BaseMemoryStore:
    """获取记忆存储实例，优先使用文件存储"""
    global _memory_store, _tried_load

    if _memory_store is None:
        # 优先尝试文件存储
        file_store = _try_load_file_store()
        if file_store is not None:
            _memory_store = file_store
            _tried_load = True
            return _memory_store

        # 回退到 Mock 存储
        _memory_store = MockMemoryStore()

    return _memory_store


def set_memory_store(store: BaseMemoryStore) -> None:
    """设置记忆存储实例"""
    global _memory_store, _tried_load
    _memory_store = store
    _tried_load = True  # 手动设置后不再尝试自动加载
