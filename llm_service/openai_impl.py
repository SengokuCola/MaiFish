"""
MaiDiary - OpenAI 兼容 LLM 服务实现
支持所有兼容 OpenAI Chat Completions 接口的服务商。
"""

import json
from typing import Callable, List, Optional

from openai import AsyncOpenAI

from .base import BaseLLMService, ChatResponse, ModelInfo, ToolCall
from .prompts import (
    CHAT_TOOLS,
    DEFAULT_CHAT_SYSTEM_PROMPT,
    EQ_SYSTEM_PROMPT,
    SAY_REWRITE_PROMPT,
    TIMING_SYSTEM_PROMPT,
)
from .utils import format_chat_history


class OpenAILLMService(BaseLLMService):
    """
    基于 OpenAI 兼容 API 的 LLM 服务实现。
    支持所有兼容 OpenAI Chat Completions 接口的服务商。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        chat_system_prompt: Optional[str] = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        enable_thinking: Optional[bool] = None,
    ):
        """
        Args:
            api_key:              API 密钥
            base_url:             API 基地址 (默认 OpenAI 官方)
            model:                模型名称
            chat_system_prompt:   自定义对话系统提示词 (为 None 则使用默认)
            temperature:          生成温度
            max_tokens:           最大输出 token 数
            enable_thinking:      是否启用思考模式 (True/False/None)
        """
        self._base_url = base_url or "https://api.openai.com/v1"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._chat_system_prompt = chat_system_prompt or DEFAULT_CHAT_SYSTEM_PROMPT
        self._enable_thinking = enable_thinking

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=self._base_url,
        )
        self._debug_callback: Optional[Callable] = None
        self._extra_tools: List[dict] = []  # MCP 等外部工具

    def set_extra_tools(self, tools: List[dict]) -> None:
        """设置额外的工具定义（如 MCP 工具），与内置工具合并使用。"""
        self._extra_tools = list(tools)

    def set_debug_callback(self, callback: Callable[[str, list, Optional[list]], None]):
        """
        设置调试回调，每次 LLM 调用前触发。

        callback(label, messages, tools) — tools 可为 None。
        """
        self._debug_callback = callback

    async def _call_llm(self, label: str, messages: list, tools: Optional[list] = None, **kwargs):
        """统一 LLM 调用入口：触发 debug 回调后调用 API。"""
        if self._debug_callback:
            try:
                self._debug_callback(label, messages, tools)
            except Exception:
                pass

        create_kwargs = {"model": self._model, "messages": messages, **kwargs}
        if tools:
            create_kwargs["tools"] = tools

        return await self._client.chat.completions.create(**create_kwargs)

    def _build_extra_body(self) -> dict:
        """构建 extra_body 参数（如 enable_thinking）。"""
        extra_body = {}
        if self._enable_thinking is not None:
            extra_body["enable_thinking"] = self._enable_thinking
        return extra_body

    def _parse_tool_calls(self, msg) -> List[ToolCall]:
        """从 API 响应消息中解析工具调用列表。"""
        tool_calls: List[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))
        return tool_calls

    def _build_raw_message(self, msg) -> dict:
        """从 API 响应消息构建可追加到对话历史的消息字典。"""
        raw_message: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return raw_message

    # ──────── 接口实现 ────────

    async def chat_loop_step(self, chat_history: List[dict]) -> ChatResponse:
        """执行对话循环的一步，返回包含文本和/或工具调用的响应。"""
        extra_body = self._build_extra_body()

        # 合并内置工具与 MCP 等外部工具
        all_tools = CHAT_TOOLS + self._extra_tools

        response = await self._call_llm(
            "主 Agent 对话",
            chat_history,
            tools=all_tools,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        msg = response.choices[0].message
        return ChatResponse(
            content=msg.content,
            tool_calls=self._parse_tool_calls(msg),
            raw_message=self._build_raw_message(msg),
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self._model, base_url=self._base_url)

    # ──────── say 后处理 ────────

    async def rewrite_say(self, text: str) -> str:
        """将 say 工具的输出文本改写为口语化贴吧风格。"""
        messages = [
            {"role": "system", "content": SAY_REWRITE_PROMPT},
            {"role": "user", "content": text},
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                "say 风格改写",
                messages,
                temperature=0.8,
                max_tokens=512,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            result = response.choices[0].message.content or text
            return result.strip()
        except Exception:
            # 改写失败时回退到原文
            return text

    # ──────── Timing 模块 ────────

    async def analyze_timing(
        self, chat_history: List[dict], timing_info: str,
    ) -> str:
        """Timing 模块：分析对话的时间维度信息。"""
        formatted = format_chat_history(chat_history)
        timing_messages = [
            {"role": "system", "content": TIMING_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"【系统时间戳信息】\n{timing_info}\n\n"
                    f"【当前对话记录】\n{formatted}"
                ),
            },
        ]
        extra_body = self._build_extra_body()

        response = await self._call_llm(
            "Timing 模块",
            timing_messages,
            temperature=0.3,
            max_tokens=512,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        return response.choices[0].message.content or ""

    # ──────── 情商模块 (EQ Module) ────────

    async def analyze_emotion(self, chat_history: List[dict]) -> str:
        """情商模块：分析用户的情绪状态和言语态度。"""
        # 获取最近几轮对话（约 8-10 条消息，约 3-5 轮）
        recent_messages = chat_history[-10:] if len(chat_history) > 10 else chat_history
        formatted = format_chat_history(recent_messages)

        eq_messages = [
            {"role": "system", "content": EQ_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"以下是最近几轮对话记录，请分析其中用户的情绪状态和言语态度：\n\n{formatted}",
            },
        ]
        extra_body = self._build_extra_body()

        response = await self._call_llm(
            "情商模块 (EQ)",
            eq_messages,
            temperature=0.3,
            max_tokens=512,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        return response.choices[0].message.content or ""

    # ──────── 对话上下文构建 ────────

    def build_chat_context(self, user_text: str) -> List[dict]:
        """根据用户初始输入构建对话循环的初始上下文。"""
        return [
            {"role": "system", "content": self._chat_system_prompt},
            {"role": "user", "content": user_text},
        ]
