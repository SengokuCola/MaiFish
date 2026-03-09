"""
MaiDiary - CLI 交互界面与对话引擎
BufferCLI 整合主循环、对话引擎、子代理管理。
"""

import os
import asyncio
from datetime import datetime
from typing import Optional

from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich import box

from config import console
from input_reader import InputReader
from table import Table
from sub_agent import SubAgent
from debug_client import DebugViewer
from timing import build_timing_info
from llm_service import BaseLLMService, OpenAILLMService, format_chat_history
from mcp_client import MCPManager
from tool_handlers import (
    ToolHandlerContext,
    handle_say,
    handle_stop,
    handle_wait,
    handle_create_table,
    handle_list_tables,
    handle_view_table,
    handle_mcp_tool,
    handle_unknown_tool,
)


class BufferCLI:
    """命令行交互界面"""

    CONTEXT_LIMIT = 40       # 上下文消息数上限，触发记忆转化
    CONTEXT_TRIM_COUNT = 10  # 每次移出的旧消息数

    def __init__(self):
        self.llm_service: Optional[BaseLLMService] = None
        self._reader = InputReader()
        self._sub_agents: list[SubAgent] = []
        self._tables: dict[str, Table] = {}  # 名称 → 表格
        self._chat_history: Optional[list] = None  # 持久化的对话历史
        # Timing 模块时间戳跟踪
        self._chat_start_time: Optional[datetime] = None
        self._last_user_input_time: Optional[datetime] = None
        self._last_assistant_response_time: Optional[datetime] = None
        self._user_input_times: list[datetime] = []  # 所有用户输入时间戳
        # MCP 管理器（异步初始化，在 run() 中完成）
        self._mcp_manager: Optional[MCPManager] = None
        # Debug Viewer
        self._debug_viewer = DebugViewer()
        self._init_llm()

    def _init_llm(self):
        """初始化 LLM 服务"""
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        thinking_env = os.getenv("ENABLE_THINKING", "").strip().lower()
        enable_thinking: Optional[bool] = (
            True if thinking_env == "true"
            else False if thinking_env == "false"
            else None
        )

        if not api_key:
            console.print(
                Panel(
                    "[warning]未检测到 OPENAI_API_KEY 环境变量！[/warning]\n\n"
                    "请设置以下环境变量（或在 .env 文件中配置）：\n"
                    "  • OPENAI_API_KEY   - 必填，API 密钥\n"
                    "  • OPENAI_BASE_URL  - 可选，API 基地址\n"
                    "  • OPENAI_MODEL     - 可选，模型名称（默认 gpt-4o）\n\n"
                    "[muted]程序无法运行，请配置后重试。[/muted]",
                    title="⚠️ 配置提示",
                    border_style="yellow",
                )
            )
            return

        self.llm_service = OpenAILLMService(
            api_key=api_key,
            base_url=base_url if base_url else None,
            model=model,
            enable_thinking=enable_thinking,
        )
        # 绑定 debug 回调
        self.llm_service.set_debug_callback(self._debug_viewer.send)
        console.print(f"[success]✓ LLM 服务已初始化[/success] [muted](模型: {model})[/muted]")

    def _build_tool_context(self) -> ToolHandlerContext:
        """构建工具处理器所需的上下文。"""
        ctx = ToolHandlerContext(
            llm_service=self.llm_service,
            reader=self._reader,
            tables=self._tables,
            user_input_times=self._user_input_times,
        )
        ctx.last_user_input_time = self._last_user_input_time
        return ctx

    # ──────── 显示方法 ────────

    def _show_banner(self):
        """显示欢迎横幅"""
        banner = Text()
        banner.append("MaiDiary", style="bold cyan")
        banner.append(" v2.0\n", style="muted")
        banner.append("直接输入文字开始对话 | Ctrl+C 退出", style="muted")

        console.print(Panel(banner, box=box.DOUBLE_EDGE, border_style="cyan", padding=(1, 2)))
        console.print()

    # ──────── LLM 循环架构 ────────

    async def _start_chat(self, user_text: str):
        """接收用户输入并启动/继续 LLM 对话循环"""
        if not self.llm_service:
            console.print("[warning]LLM 服务未初始化，跳过对话。[/warning]")
            return

        now = datetime.now()
        self._last_user_input_time = now
        self._user_input_times.append(now)

        if self._chat_history is None:
            # 首次对话：初始化上下文
            self._chat_start_time = now
            self._last_assistant_response_time = None
            self._chat_history = self.llm_service.build_chat_context(user_text)
        else:
            # 后续对话：追加用户消息到已有上下文
            self._chat_history.append({
                "role": "user",
                "content": user_text,
            })

        await self._run_llm_loop(self._chat_history)

    async def _run_llm_loop(self, chat_history: list):
        """
        LLM 循环架构核心。

        LLM 持续运行，每步可能输出文本（内心思考）和/或调用工具：
        - say(text): 对用户说话
        - wait(seconds): 暂停等待用户输入，超时或收到输入后继续
        - stop(): 结束循环，进入待机，直到用户下次输入
        - 不调用工具: 继续下一轮思考/生成

        每轮流程：
        1. 上下文管理：达到上限时自动压缩为子代理
        2. 子代理检查：并行触发所有子代理，注入提示
        3. 情商 + Timing 模块（并行）：分析用户情绪和对话时间节奏，注入分析结果
        4. 调用主 LLM：基于完整上下文（含情绪 + 时间分析）生成响应
        """
        consecutive_errors = 0

        while True:
            # ── 上下文管理：达到上限时自动转化为子代理 ──
            if len(chat_history) >= self.CONTEXT_LIMIT:
                await self._compress_context(chat_history)

            # ── 触发子代理检查（并行） ──
            if self._sub_agents:
                suggestions = await self._trigger_sub_agents(chat_history)
                if suggestions:
                    chat_history.append({
                        "role": "user",
                        "content": f"[来自记忆子代理的提示信息]\n{suggestions}",
                    })

            # ── 情商模块 + Timing 模块（并行） ──
            timing_info = build_timing_info(
                self._chat_start_time,
                self._last_user_input_time,
                self._last_assistant_response_time,
                self._user_input_times,
            )
            with console.status(
                "[info]🎭⏱️ 情商 + Timing 模块并行分析中...[/info]",
                spinner="dots",
            ):
                eq_task = self.llm_service.analyze_emotion(chat_history)
                timing_task = self.llm_service.analyze_timing(
                    chat_history, timing_info,
                )
                eq_result, timing_result = await asyncio.gather(
                    eq_task, timing_task, return_exceptions=True,
                )

            # 处理情商模块结果
            eq_analysis = ""
            if isinstance(eq_result, Exception):
                console.print(f"[warning]情商模块分析失败: {eq_result}[/warning]")
            elif eq_result:
                eq_analysis = eq_result
                console.print(
                    Panel(
                        Markdown(eq_analysis),
                        title="🎭 情绪感知",
                        border_style="bright_yellow",
                        padding=(0, 1),
                        style="dim",
                    )
                )

            # 处理 Timing 模块结果
            timing_analysis = ""
            if isinstance(timing_result, Exception):
                console.print(f"[warning]Timing 模块分析失败: {timing_result}[/warning]")
            elif timing_result:
                timing_analysis = timing_result
                console.print(
                    Panel(
                        Markdown(timing_analysis),
                        title="⏱️ 时间感知",
                        border_style="bright_blue",
                        padding=(0, 1),
                        style="dim",
                    )
                )

            # 注入情商模块结果
            if eq_analysis:
                chat_history.append({
                    "role": "user",
                    "content": (
                        f"[情商模块 — 情绪感知]\n{eq_analysis}\n"
                        f"请结合以上情绪分析来回应用户，注意语气和态度的适配。"
                    ),
                })

            # 注入 Timing 模块结果（单独一条，只保留最新一条）
            if timing_analysis:
                # 移除上一条 timing 消息（如果存在）
                TIMING_TAG = "[Timing 模块 — 时间感知]"
                for i in range(len(chat_history) - 1, -1, -1):
                    msg = chat_history[i]
                    if (
                        msg.get("role") == "user"
                        and isinstance(msg.get("content"), str)
                        and TIMING_TAG in msg["content"]
                    ):
                        chat_history.pop(i)
                        break
                # 添加最新的 timing 消息
                chat_history.append({
                    "role": "user",
                    "content": (
                        f"{TIMING_TAG}\n{timing_analysis}\n"
                        f"请结合以上时间信息把握对话节奏。"
                    ),
                })

            # ── 调用 LLM ──
            with console.status("[info]💬 AI 正在思考...[/info]", spinner="dots"):
                try:
                    response = await self.llm_service.chat_loop_step(chat_history)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    console.print(f"[error]LLM 调用出错: {e}[/error]")
                    if consecutive_errors >= 3:
                        console.print("[error]连续出错，退出对话[/error]\n")
                        break
                    continue

            # 将 assistant 消息追加到历史
            chat_history.append(response.raw_message)
            self._last_assistant_response_time = datetime.now()

            # 显示内心思考（content 部分，淡色呈现）
            if response.content:
                console.print(
                    Panel(
                        Markdown(response.content),
                        title="💭 内心思考",
                        border_style="dim",
                        padding=(1, 2),
                        style="dim",
                    )
                )

            # ── 处理工具调用 ──
            if response.tool_calls:
                should_stop = False
                ctx = self._build_tool_context()

                for tc in response.tool_calls:
                    if tc.name == "say":
                        await handle_say(tc, chat_history, ctx)

                    elif tc.name == "stop":
                        await handle_stop(tc, chat_history)
                        should_stop = True

                    elif tc.name == "wait":
                        tool_result = await handle_wait(tc, chat_history, ctx)
                        # 同步回 timing 时间戳
                        if ctx.last_user_input_time != self._last_user_input_time:
                            self._last_user_input_time = ctx.last_user_input_time
                        if tool_result.startswith("[[QUIT]]"):
                            should_stop = True

                    elif tc.name == "create_table":
                        await handle_create_table(tc, chat_history, ctx)

                    elif tc.name == "list_tables":
                        await handle_list_tables(tc, chat_history, ctx)

                    elif tc.name == "view_table":
                        await handle_view_table(tc, chat_history, ctx)

                    elif self._mcp_manager and self._mcp_manager.is_mcp_tool(tc.name):
                        await handle_mcp_tool(tc, chat_history, self._mcp_manager)

                    else:
                        await handle_unknown_tool(tc, chat_history)

                if should_stop:
                    console.print("[muted]对话暂停，等待新输入...[/muted]\n")
                    break

            # LLM 未调用任何工具 → 继续下一轮思考
            # （不做任何额外操作，直接回到循环顶部再次调用 LLM）

    # ──────── 子代理管理 ────────

    async def _compress_context(self, chat_history: list):
        """
        上下文压缩：取最早的 CONTEXT_TRIM_COUNT 条对话记录（跳过 system），
        调用 LLM 分析并创建子代理，然后从 chat_history 中移除这些记录。

        注意：不会在 assistant(+tool_calls) 和对应的 tool 结果消息之间切断，
        会自动扩展边界以包含完整的工具调用序列。
        """
        # 保留第一条 system 消息，从 index 1 开始取
        trim_end = 1 + self.CONTEXT_TRIM_COUNT
        if trim_end > len(chat_history):
            return  # 不够取，跳过

        # 确保不在 tool_call 序列中间切断
        while (
            trim_end < len(chat_history)
            and chat_history[trim_end].get("role") == "tool"
        ):
            trim_end += 1

        # 反向检查：如果 trim_end-1 是 assistant(+tool_calls)，
        # 也要把后续的 tool 消息一并纳入
        if (
            trim_end < len(chat_history)
            and trim_end - 1 >= 1
            and chat_history[trim_end - 1].get("role") == "assistant"
            and chat_history[trim_end - 1].get("tool_calls")
        ):
            while (
                trim_end < len(chat_history)
                and chat_history[trim_end].get("role") == "tool"
            ):
                trim_end += 1

        old_messages = chat_history[1:trim_end]

        actual_trim_count = trim_end - 1  # 实际要移除的条数
        console.print(
            f"[info]📦 上下文达到 {len(chat_history)} 条，"
            f"正在将最早 {actual_trim_count} 条对话转为记忆...[/info]"
        )

        # 调用 LLM 分析这些旧消息，决定创建哪些子代理
        try:
            with console.status(
                "[info]🧠 分析旧对话记录...[/info]", spinner="dots",
            ):
                memory_entries = await self.llm_service.analyze_for_memory(
                    old_messages,
                )
        except Exception as e:
            console.print(f"[warning]记忆分析失败: {e}，直接移除旧记录[/warning]")
            memory_entries = []

        # 为每个记忆条目创建子代理
        for entry in memory_entries:
            desc = entry.get("description", "")
            summary = entry.get("summary", "")
            if not desc:
                continue

            sub_agent = SubAgent(desc, summary)
            sub_agent.last_checked_index = len(chat_history) - actual_trim_count
            self._sub_agents.append(sub_agent)

            console.print(
                f"[success]✓ 记忆子代理 #{sub_agent.id} 已创建[/success] "
                f"[muted]({desc[:50]})[/muted]"
            )

        # 移除旧记录（保留 system 消息 + trim_end 之后的内容）
        del chat_history[1:trim_end]

        # 更新所有子代理的 last_checked_index
        for sa in self._sub_agents:
            sa.last_checked_index = max(0, sa.last_checked_index - actual_trim_count)

        console.print(
            f"[info]📦 已压缩上下文: 移除 {actual_trim_count} 条旧记录，"
            f"当前 {len(chat_history)} 条，"
            f"共 {len(self._sub_agents)} 个子代理[/info]"
        )

    async def _trigger_sub_agents(self, chat_history: list) -> str:
        """并行触发所有子代理检查，返回汇总的提示信息（空字符串表示无提示）。"""
        console.print(
            f"[muted]🧠 触发 {len(self._sub_agents)} 个子代理检查...[/muted]"
        )

        tasks = [
            self._check_sub_agent(sa, chat_history)
            for sa in self._sub_agents
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        suggestions: list[str] = []
        for sa, result in zip(self._sub_agents, results):
            if isinstance(result, Exception):
                console.print(
                    f"[warning]子代理 #{sa.id} 检查出错: {result}[/warning]"
                )
                continue
            if result:  # 非 None 表示有提示
                suggestions.append(
                    f"[子代理 #{sa.id} — {sa.memory_description}]\n{result}"
                )

        if suggestions:
            combined = "\n\n".join(suggestions)
            console.print(
                Panel(
                    Markdown(combined),
                    title=f"🧠 子代理提供信息 ({len(suggestions)} 条)",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )
            return combined
        else:
            console.print("[muted]🧠 子代理检查完毕，无需提示[/muted]")
        return ""

    async def _check_sub_agent(
        self, sub_agent: SubAgent, chat_history: list,
    ) -> Optional[str]:
        """检查单个子代理，返回 offer_info 提供的信息或 None。"""
        new_messages = chat_history[sub_agent.last_checked_index:]
        if not new_messages:
            return None

        # 格式化最新一轮主对话内容
        formatted = format_chat_history(new_messages)
        sub_agent.history.append({
            "role": "user",
            "content": (
                f"【主对话最新一轮内容】\n{formatted}\n\n"
                f"请判断是否有与你的记忆领域相关的、需要提供给主 Agent 的信息。"
                f"如果有，使用 offer_info 工具提供；如果没有，不要调用任何工具。"
            ),
        })

        # 调用 LLM 检查（返回 ChatResponse）
        response = await self.llm_service.sub_agent_check(sub_agent.history)

        # 追加 assistant 消息到子代理历史
        sub_agent.history.append(response.raw_message)

        # 显示子代理思考过程
        if response.content:
            console.print(
                Panel(
                    Markdown(response.content),
                    title=f"🧠 子代理 #{sub_agent.id} 思考",
                    border_style="dim cyan",
                    padding=(0, 1),
                    style="dim",
                )
            )

        # 处理 offer_info 工具调用
        offered_info: list[str] = []
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name == "offer_info":
                    info = tc.arguments.get("info", "")
                    console.print(
                        f"[cyan]🧠 子代理 #{sub_agent.id} 调用工具: "
                        f"offer_info(...)[/cyan]"
                    )
                    if info:
                        offered_info.append(info)
                    sub_agent.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": "已传达给主 Agent",
                    })
                else:
                    sub_agent.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"未知工具: {tc.name}",
                    })

        # 更新检查位置
        sub_agent.last_checked_index = len(chat_history)

        if offered_info:
            return "\n".join(offered_info)
        return None

    # ──────── 主循环 ────────

    async def _init_mcp(self):
        """初始化 MCP 服务器连接，发现并注册外部工具。"""
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "mcp_config.json",
        )
        self._mcp_manager = await MCPManager.from_config(config_path)

        if self._mcp_manager and self.llm_service:
            mcp_tools = self._mcp_manager.get_openai_tools()
            if mcp_tools:
                self.llm_service.set_extra_tools(mcp_tools)
                summary = self._mcp_manager.get_tool_summary()
                console.print(
                    Panel(
                        f"已加载 {len(mcp_tools)} 个 MCP 工具:\n{summary}",
                        title="🔌 MCP 工具",
                        border_style="green",
                        padding=(0, 1),
                    )
                )

    async def run(self):
        """主循环：直接输入文本即可对话"""
        # 启动调试窗口
        self._debug_viewer.start()

        # 初始化 MCP 服务器（连接、发现工具、注入到 LLM）
        await self._init_mcp()

        # 启动异步输入读取器
        self._reader.start(asyncio.get_event_loop())

        self._show_banner()

        try:
            while True:
                console.print("[bold cyan]> [/bold cyan]", end="")
                raw_input = await self._reader.get_line()

                if raw_input is None:  # EOF
                    console.print("\n[muted]再见！[/muted]")
                    break

                raw_input = raw_input.strip()
                if not raw_input:
                    continue

                await self._start_chat(raw_input)
        finally:
            self._debug_viewer.close()
            if self._mcp_manager:
                await self._mcp_manager.close()
