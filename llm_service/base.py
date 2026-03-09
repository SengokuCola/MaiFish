"""
MaiDiary - LLM 服务数据结构与抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


# ──────────────────── 数据结构 ────────────────────

@dataclass
class ModelInfo:
    """模型描述信息"""
    model_name: str
    base_url: str


@dataclass
class ToolCall:
    """工具调用信息"""
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResponse:
    """LLM 对话循环单步响应"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    raw_message: dict  # 可直接追加到对话历史的消息字典


# ──────────────────── 抽象接口 ────────────────────

class BaseLLMService(ABC):
    """
    LLM 服务抽象基类。
    所有 LLM 后端实现都应继承此类，并实现以下方法。
    """

    def set_extra_tools(self, tools: List[dict]) -> None:
        """
        设置额外的工具定义（如 MCP 工具），将与内置工具合并使用。

        Args:
            tools: OpenAI function calling 格式的工具定义列表
        """
        # 默认空实现，子类可覆盖
        pass

    @abstractmethod
    async def chat_loop_step(self, chat_history: List[dict]) -> ChatResponse:
        """
        执行对话循环的一步。

        发送当前对话历史，获取 LLM 响应（可能包含文本和/或工具调用）。
        调用方需要将 raw_message 追加到 chat_history，并根据 tool_calls 执行工具、
        将工具结果追加到 chat_history 后再次调用本方法。

        Args:
            chat_history: 对话历史（含 system / user / assistant / tool 消息）

        Returns:
            ChatResponse
        """
        ...

    @abstractmethod
    def build_chat_context(self, user_text: str) -> List[dict]:
        """根据用户初始输入，构建对话循环的初始上下文（system + user）。"""
        ...

    @abstractmethod
    async def analyze_for_memory(
        self, messages: List[dict],
    ) -> List[dict]:
        """
        分析一段对话记录，返回需要创建的子代理描述列表。

        Returns:
            [{"description": "...", "summary": "..."}, ...]
        """
        ...

    @abstractmethod
    async def analyze_timing(
        self, chat_history: List[dict], timing_info: str,
    ) -> str:
        """
        Timing 模块：分析对话的时间维度信息。

        评估对话已经持续多久、上次回复距今多长时间、建议等待时长、
        以及其他与时间节奏相关的考量。

        Args:
            chat_history: 当前对话历史（与主 Agent 完全一致的上下文）
            timing_info:  系统提供的精确时间戳信息（对话开始时间、各消息时间等）

        Returns:
            时间维度分析文本
        """
        ...

    @abstractmethod
    async def analyze_emotion(self, chat_history: List[dict]) -> str:
        """
        情商模块：分析对话对方（用户）的情绪状态和言语态度。

        接收与主 Agent 相同的上下文，返回一段简洁的情绪分析文本。
        该文本将被注入主 Agent 上下文，帮助主 Agent 更好地理解用户状态。

        Args:
            chat_history: 当前对话历史（与主 Agent 完全一致的上下文）

        Returns:
            情绪分析文本
        """
        ...

    @abstractmethod
    async def create_sub_agent_summary(
        self, chat_history: List[dict], memory_description: str,
    ) -> str:
        """根据记忆描述，对主对话上下文进行总结，作为子代理的初始上下文。"""
        ...

    @abstractmethod
    async def sub_agent_check(self, sub_agent_history: List[dict]) -> ChatResponse:
        """执行子代理检查，返回包含文本和/或 offer_info 工具调用的响应。"""
        ...

    @abstractmethod
    async def rewrite_say(self, text: str) -> str:
        """
        对 say 工具输出的文本进行后处理改写。

        将正式发言改写为口语化、简短、贴吧风格的语句。

        Args:
            text: say 工具原始输出文本

        Returns:
            改写后的文本
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """返回当前使用的模型信息。"""
        ...
