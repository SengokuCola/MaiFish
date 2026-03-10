"""
MaiSaka - OpenAI 兼容 LLM 服务实现
支持所有兼容 OpenAI Chat Completions 接口的服务商。
"""

import json
from typing import Callable, List, Optional

from openai import AsyncOpenAI

from .base import BaseLLMService, ChatResponse, ModelInfo, ToolCall
from .prompts import get_enabled_chat_tools
from .utils import format_chat_history, filter_for_api
from prompt_loader import load_prompt


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
        self._enable_thinking = enable_thinking

        # 如果没有提供自定义提示词，则根据配置动态构建
        if chat_system_prompt is None:
            from config import ENABLE_WRITE_FILE, ENABLE_READ_FILE, ENABLE_LIST_FILES

            # 构建文件工具说明
            tools_parts = []
            if ENABLE_WRITE_FILE:
                tools_parts.append("• write_file(filename, content) — 在 mai_files 目录下写入文件，支持任意格式。")
            if ENABLE_READ_FILE:
                tools_parts.append("• read_file(filename) — 读取 mai_files 目录下的文件内容。")
            if ENABLE_LIST_FILES:
                tools_parts.append("• list_files() — 获取 mai_files 目录下所有文件的元信息列表。")

            # 如果有任何工具启用，添加前缀空行
            if tools_parts:
                tools_section = "\n" + "\n".join(tools_parts) + "\n"
            else:
                tools_section = ""

            # 加载提示词模板并注入工具部分
            self._chat_system_prompt = load_prompt("chat.system", file_tools_section=tools_section)
        else:
            self._chat_system_prompt = chat_system_prompt

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
                        # 确保 arguments 是有效的 JSON 字符串，空参数用 "{}"
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in msg.tool_calls
            ]
        return raw_message

    # ──────── 接口实现 ────────

    async def chat_loop_step(self, chat_history: List[dict]) -> ChatResponse:
        """执行对话循环的一步，返回包含文本和/或工具调用的响应。"""
        extra_body = self._build_extra_body()

        # 延迟导入配置以避免循环导入
        from config import ENABLE_WRITE_FILE, ENABLE_READ_FILE, ENABLE_LIST_FILES

        # 获取根据配置启用的内置工具
        enabled_tools = get_enabled_chat_tools(
            enable_write_file=ENABLE_WRITE_FILE,
            enable_read_file=ENABLE_READ_FILE,
            enable_list_files=ENABLE_LIST_FILES,
        )

        # 合并内置工具与 MCP 等外部工具
        all_tools = enabled_tools + self._extra_tools

        # 过滤内部字段（如 _type），只保留 API 需要的字段
        api_messages = filter_for_api(chat_history)

        response = await self._call_llm(
            "主 Agent 对话",
            api_messages,
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

    # ──────── Timing 模块 ────────

    async def analyze_timing(
        self, chat_history: List[dict], timing_info: str,
    ) -> str:
        """Timing 模块：分析对话的时间维度信息。"""
        # 过滤掉感知消息（AI 的内部感知不需要再分析）
        filtered_history = [msg for msg in chat_history if msg.get("_type") != "perception"]
        formatted = format_chat_history(filtered_history)
        timing_messages = [
            {"role": "system", "content": load_prompt("timing.system")},
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

    # ──────── 记忆需求分析模块 ────────

    async def analyze_memory_need(self, chat_history: List[dict]) -> str:
        """记忆需求分析模块：分析当前对话上下文，思考需要查询什么记忆信息。"""
        # 过滤掉感知消息（AI 的内部感知不需要再分析）
        filtered_history = [msg for msg in chat_history if msg.get("_type") != "perception"]
        # 获取最近几轮对话用于分析
        recent_messages = filtered_history[-10:] if len(filtered_history) > 10 else filtered_history
        formatted = format_chat_history(recent_messages)

        memory_need_messages = [
            {"role": "system", "content": load_prompt("memory_need.system")},
            {
                "role": "user",
                "content": f"以下是最近的对话记录，请分析需要从记忆系统中查询什么信息：\n\n{formatted}",
            },
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                "记忆需求分析",
                memory_need_messages,
                temperature=0.3,
                max_tokens=512,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            return response.choices[0].message.content or ""
        except Exception:
            # 分析失败时返回空字符串（表示无需查询）
            return ""

    # ──────── 上下文总结模块 ────────

    async def summarize_context(self, context_messages: List[dict]) -> str:
        """上下文总结模块：对需要压缩的上下文进行总结。"""
        formatted = format_chat_history(context_messages)

        summarize_messages = [
            {"role": "system", "content": load_prompt("context_summarize.system")},
            {
                "role": "user",
                "content": f"请对以下对话内容进行总结，以便存入记忆系统：\n\n{formatted}",
            },
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                "上下文总结",
                summarize_messages,
                temperature=0.3,
                max_tokens=1024,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            return response.choices[0].message.content or ""
        except Exception:
            # 总结失败时返回空字符串
            return ""

    # ──────── 记忆选择模块 ────────

    async def select_relevant_memories(
        self, questions: List[str], memories: List[dict]
    ) -> List[int]:
        """
        记忆选择模块：从所有记忆中选择与问题相关的记忆。

        将所有记忆（编号）和问句传给 LLM，让 LLM 选择有用的记忆并返回编号。
        """
        if not questions or not memories:
            return []

        # 构建记忆列表文本
        memory_list_text = "\n".join([
            f"{i+1}. {m.get('content', '')}"
            for i, m in enumerate(memories)
        ])

        # 构建问题文本
        questions_text = "\n".join([
            f"- {q}"
            for q in questions
        ])

        select_messages = [
            {"role": "system", "content": load_prompt("memory_select.system")},
            {
                "role": "user",
                "content": (
                    f"【问题】\n{questions_text}\n\n"
                    f"【记忆列表】\n{memory_list_text}\n\n"
                    f"请选择与问题相关的记忆编号。"
                ),
            },
        ]
        extra_body = self._build_extra_body()

        try:
            response = await self._call_llm(
                "记忆选择",
                select_messages,
                temperature=0.3,
                max_tokens=256,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            result = response.choices[0].message.content or ""

            # 解析返回的编号
            result = result.strip()
            if "无" in result or not result:
                return []

            # 解析编号（支持逗号分隔、空格分隔）
            selected_indices = []
            for part in result.replace(",", " ").replace("，", " ").split():
                try:
                    idx = int(part.strip())
                    # 转换为 0-based 索引，并检查范围
                    if 1 <= idx <= len(memories):
                        selected_indices.append(idx - 1)
                except ValueError:
                    continue

            return selected_indices
        except Exception:
            # 选择失败时返回空列表
            return []

    # ──────── 对话上下文构建 ────────

    def build_chat_context(self, user_text: str) -> List[dict]:
        """根据用户初始输入构建对话循环的初始上下文。"""
        return [
            {"role": "system", "content": self._chat_system_prompt},
            {"role": "user", "content": user_text},
        ]
