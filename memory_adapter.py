"""
MaiSaka - A_Memorix 记忆适配器

提供 A_Memorix 记忆系统的适配器实现，将 A_Memorix 对接到 BaseMemoryStore 接口。

直接初始化 A_Memorix 存储组件，不依赖插件系统。
"""

import os
import sys
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

# 添加 A_memorix 路径
memorix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "A_memorix")
if memorix_path not in sys.path:
    sys.path.insert(0, memorix_path)

from memory import BaseMemoryStore, MemoryQuery, MemoryResult, QueryType

# A_Memorix 数据目录
MEMORIX_DATA_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "maidairy_data", "memorix"))


class AMemorixAdapter(BaseMemoryStore):
    """
    A_Memorix 记忆适配器。

    直接初始化 A_Memorix 存储组件（不依赖插件系统），适配到 BaseMemoryStore 接口。
    """

    def __init__(self):
        """初始化适配器，延迟加载 A_Memorix 组件"""
        self._vector_store = None
        self._graph_store = None
        self._metadata_store = None
        self._embedding_manager = None
        self._sparse_index = None
        self._dual_path_retriever = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """
        确保 A_Memorix 组件已初始化。

        直接初始化存储组件，不依赖插件系统。

        Returns:
            是否初始化成功
        """
        if self._initialized:
            return True

        try:
            # 导入 A_Memorix 核心组件
            from A_memorix.core import (
                VectorStore,
                GraphStore,
                MetadataStore,
                EmbeddingAPIAdapter,
                SparseBM25Index,
                DualPathRetriever,
                DualPathRetrieverConfig,
                RetrievalStrategy,
            )

            # 确保数据目录存在
            MEMORIX_DATA_DIR.mkdir(parents=True, exist_ok=True)

            # 初始化向量存储
            vector_path = MEMORIX_DATA_DIR / "vectors"
            vector_path.mkdir(parents=True, exist_ok=True)
            self._vector_store = VectorStore(
                dimension=1024,  # 默认维度，会根据模型自动调整
                save_path=str(vector_path),
            )

            # 初始化图存储
            graph_path = MEMORIX_DATA_DIR / "graph"
            graph_path.mkdir(parents=True, exist_ok=True)
            self._graph_store = GraphStore(save_path=str(graph_path))

            # 初始化元数据存储
            self._metadata_store = MetadataStore(
                db_path=str(MEMORIX_DATA_DIR / "metadata.db")
            )

            # 初始化嵌入管理器（使用环境变量配置）
            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL", "")
            model = os.getenv("OPENAI_MODEL", "text-embedding-3-small")

            if not api_key:
                # 没有 API key，无法初始化嵌入管理器
                return False

            self._embedding_manager = EmbeddingAPIAdapter(
                api_key=api_key,
                base_url=base_url if base_url else None,
                model=model,
            )

            # 初始化稀疏索引（BM25）
            sparse_path = MEMORIX_DATA_DIR / "sparse"
            sparse_path.mkdir(parents=True, exist_ok=True)
            self._sparse_index = SparseBM25Index(save_path=str(sparse_path))

            # 创建检索器
            retriever_config = DualPathRetrieverConfig()
            self._dual_path_retriever = DualPathRetriever(
                vector_store=self._vector_store,
                graph_store=self._graph_store,
                metadata_store=self._metadata_store,
                sparse_index=self._sparse_index,
                config=retriever_config,
            )

            self._initialized = True
            return True

        except Exception as e:
            # 静默失败，避免启动时错误
            print(f"[warning]A_Memorix 初始化失败: {e}[/warning]")
            return False

    async def store_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        存储一条记忆到 A_Memorix。

        Args:
            content: 记忆内容
            metadata: 元数据（如时间戳、来源等）

        Returns:
            是否存储成功
        """
        if not self._ensure_initialized():
            return False

        try:
            # 生成 source 标识
            source = metadata.get("source", "maidairy") if metadata else "maidairy"
            if metadata and metadata.get("type") == "context_summary":
                source = f"maidairy:context_summary:{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 添加段落到元数据存储
            hash_value = self._metadata_store.add_paragraph(
                content=content,
                source=source,
                knowledge_type="narrative",
                time_meta=metadata.get("time_meta") if metadata else None,
            )

            # 生成嵌入向量并存储
            embedding = await self._embedding_manager.encode(content)
            self._vector_store.add(
                vectors=embedding.reshape(1, -1),
                ids=[hash_value]
            )

            # 持久化
            self._vector_store.save()

            return True

        except Exception:
            # 静默失败
            return False

    async def query_memory(self, query: MemoryQuery) -> List[MemoryResult]:
        """
        从 A_Memorix 查询记忆。

        Args:
            query: 记忆查询请求

        Returns:
            记忆结果列表
        """
        if not self._ensure_initialized():
            return []

        try:
            # 根据查询类型选择检索方式
            if query.query_type == QueryType.SEARCH:
                return await self._search_memory(query.query, query.top_k)
            elif query.query_type == QueryType.ENTITY:
                return await self._query_entity(query.query, query.top_k)
            elif query.query_type == QueryType.RELATION:
                return await self._query_relation(query.query, query.top_k)
            elif query.query_type == QueryType.TIME:
                return await self._query_time(query.query, query.top_k)
            else:
                return await self._search_memory(query.query, query.top_k)

        except Exception:
            return []

    async def _search_memory(self, query_text: str, top_k: int) -> List[MemoryResult]:
        """语义检索"""
        from A_memorix.core import RetrievalStrategy

        results = await self._dual_path_retriever.retrieve(
            query_text=query_text,
            strategy=RetrievalStrategy.FUSION,
            top_k=top_k,
        )

        memory_results = []
        for result in results:
            memory_results.append(MemoryResult(
                content=result.content or "",
                score=result.score or 0.0,
                metadata={
                    "hash": result.hash,
                    "source": result.source,
                    "knowledge_type": result.knowledge_type,
                } if result.hash else None,
            ))

        return memory_results

    async def _query_entity(self, entity_name: str, top_k: int) -> List[MemoryResult]:
        """实体查询"""
        # 获取实体的相关段落
        from A_memorix.core import create_ppr_from_graph, PageRankConfig

        # 使用 Personalized PageRank 获取实体相关内容
        ppr_config = PageRankConfig()
        ppr = create_ppr_from_graph(
            self._graph_store,
            self._metadata_store,
            ppr_config
        )

        # 获取实体相关的关系
        relations = self._metadata_store.search_relations_by_subject_or_object(
            entity_name,
            limit=top_k,
            include_deleted=False,
        )

        memory_results = []
        for rel in relations:
            source_paragraph = rel.get("source_paragraph")
            if source_paragraph:
                memory_results.append(MemoryResult(
                    content=source_paragraph,
                    score=rel.get("confidence", 0.5),
                    metadata={
                        "subject": rel.get("subject"),
                        "predicate": rel.get("predicate"),
                        "object": rel.get("object"),
                    },
                ))

        return memory_results[:top_k]

    async def _query_relation(self, query_text: str, top_k: int) -> List[MemoryResult]:
        """关系查询"""
        # 简化实现：使用语义检索
        return await self._search_memory(query_text, top_k)

    async def _query_time(self, query_text: str, top_k: int) -> List[MemoryResult]:
        """时序检索"""
        # 简化实现：使用语义检索
        return await self._search_memory(query_text, top_k)


def create_memorix_adapter() -> Optional[BaseMemoryStore]:
    """
    创建 A_Memorix 适配器实例。

    Returns:
        A_Memorix 适配器实例，如果初始化失败则返回 None
    """
    adapter = AMemorixAdapter()
    if adapter._ensure_initialized():
        return adapter
    return None
