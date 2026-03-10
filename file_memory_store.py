"""
MaiSaka - 本地文件记忆存储
不依赖 A_Memorix，使用 JSON 文件持久化 + 简单关键词匹配。
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from difflib import SequenceMatcher

from memory import BaseMemoryStore, MemoryQuery, MemoryResult, QueryType

# 数据目录 - 项目根目录下的 mai_memory
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DATA_DIR = PROJECT_ROOT / "mai_memory"
MEMORY_FILE = MEMORY_DATA_DIR / "memories.json"


class SimpleKeywordScorer:
    """简单的关键词相似度评分器"""

    @staticmethod
    def score(query: str, content: str) -> float:
        """
        计算查询与内容的相似度分数

        使用多种策略组合：
        1. 精确匹配关键词
        2. 连续子串匹配
        3. SequenceRatio 相似度
        """
        if not query or not content:
            return 0.0

        query_lower = query.lower()
        content_lower = content.lower()

        # 策略 1: 精确匹配（查询中的每个词都在内容中）
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        exact_matches = len(query_words & content_words)
        exact_score = exact_matches / len(query_words) if query_words else 0

        # 策略 2: 子串匹配（查询作为整体出现在内容中）
        substring_score = 0.3 if query_lower in content_lower else 0

        # 策略 3: SequenceRatio 相似度
        sequence_score = SequenceMatcher(None, query_lower, content_lower).ratio() * 0.4

        # 组合分数
        return min(1.0, exact_score * 0.5 + substring_score + sequence_score)


class FileMemoryStore(BaseMemoryStore):
    """
    基于本地 JSON 文件的记忆存储。

    特性：
    - 持久化到 JSON 文件
    - 简单的关键词匹配检索
    - 不依赖外部服务
    - 支持元数据过滤
    """

    def __init__(self):
        """初始化文件存储"""
        self._memories: List[Dict[str, Any]] = []
        self._loaded = False
        self._ensure_data_dir()
        self._load()

    def _ensure_data_dir(self):
        """确保数据目录存在"""
        MEMORY_DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self):
        """从文件加载记忆"""
        if not MEMORY_FILE.exists():
            self._memories = []
            self._loaded = True
            return

        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                self._memories = json.load(f)
            self._loaded = True
        except Exception:
            self._memories = []
            self._loaded = True

    def _save(self):
        """保存记忆到文件"""
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._memories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[warning]保存记忆失败: {e}[/warning]")

    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        存储一条记忆。

        Args:
            content: 记忆内容
            metadata: 元数据

        Returns:
            是否存储成功
        """
        try:
            memory = {
                "id": f"mem_{datetime.now().timestamp()}",
                "content": content,
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat(),
            }
            self._memories.append(memory)
            self._save()
            return True
        except Exception:
            return False

    async def query_memory(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        查询记忆。

        Args:
            query: 记忆查询请求

        Returns:
            记忆结果列表
        """
        if not self._memories:
            return []

        results = []

        for memory in self._memories:
            score = 0.0
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})

            # 根据查询类型处理
            if query.query_type == QueryType.SEARCH:
                # 关键词匹配
                score = SimpleKeywordScorer.score(query.query, content)

            elif query.query_type == QueryType.ENTITY:
                # 实体查询：在内容中查找实体名称
                if query.query.lower() in content.lower():
                    score = 0.8

            elif query.query_type == QueryType.RELATION:
                # 关系查询：查找包含关系的内容
                score = SimpleKeywordScorer.score(query.query, content)

            elif query.query_type == QueryType.TIME:
                # 时序检索：匹配时间相关内容
                score = SimpleKeywordScorer.score(query.query, content)

            # 元数据过滤（如果有 person_id）
            if query.person_id:
                person_meta = metadata.get("person_id", "")
                if person_meta and str(person_meta) != str(query.person_id):
                    score = 0

            # 只返回分数大于阈值的结果
            if score > 0.1:
                results.append(MemoryResult(
                    content=content,
                    score=score,
                    metadata={
                        **metadata,
                        "id": memory.get("id"),
                        "created_at": memory.get("created_at"),
                    },
                ))

        # 按分数排序并返回 top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:query.top_k]

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "total_memories": len(self._memories),
            "data_file": str(MEMORY_FILE),
            "data_exists": MEMORY_FILE.exists(),
            "data_size_kb": MEMORY_FILE.stat().st_size / 1024 if MEMORY_FILE.exists() else 0,
        }
