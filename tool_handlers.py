"""
MaiDiary - 工具调用处理器
处理 LLM 循环中各工具（say/wait/stop/create_table/list_tables/view_table/MCP）的执行逻辑。
"""

import json as _json
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from rich.panel import Panel
from rich.markdown import Markdown

from config import console
from table import Table
from input_reader import InputReader
from llm_service import BaseLLMService

if TYPE_CHECKING:
    from mcp_client import MCPManager


class ToolHandlerContext:
    """工具处理器所需的共享上下文。"""

    def __init__(
        self,
        llm_service: BaseLLMService,
        reader: InputReader,
        tables: dict[str, Table],
        user_input_times: list[datetime],
    ):
        self.llm_service = llm_service
        self.reader = reader
        self.tables = tables
        self.user_input_times = user_input_times
        self.last_user_input_time: Optional[datetime] = None


async def handle_say(tc, chat_history: list, ctx: ToolHandlerContext):
    """处理 say 工具：改写文本后展示给用户。"""
    say_text = tc.arguments.get("text", "")
    console.print("[accent]🔧 调用工具: say(...)[/accent]")

    if say_text:
        # 原文以淡色展示
        console.print(
            Panel(
                Markdown(say_text),
                title="💭 say 原文",
                border_style="dim",
                padding=(0, 1),
                style="dim",
            )
        )
        # 过一遍 LLM 改写为贴吧风格
        with console.status(
            "[info]✏️ 风格改写中...[/info]",
            spinner="dots",
        ):
            rewritten = await ctx.llm_service.rewrite_say(say_text)
        console.print(
            Panel(
                Markdown(rewritten),
                title="💬 MaiDiary",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        # 改写后的结果作为 tool 结果写入上下文
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"已向用户展示（实际输出）：{rewritten}",
        })
    else:
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": "say 内容为空，未展示",
        })


async def handle_stop(tc, chat_history: list):
    """处理 stop 工具：结束对话循环。"""
    console.print("[accent]🔧 调用工具: stop()[/accent]")
    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": "对话循环已停止，等待用户下次输入。",
    })


async def handle_wait(tc, chat_history: list, ctx: ToolHandlerContext) -> str:
    """
    处理 wait 工具：等待用户输入或超时。

    Returns:
        工具结果字符串。以 "[[QUIT]]" 开头表示用户要求退出对话。
    """
    seconds = tc.arguments.get("seconds", 30)
    seconds = max(5, min(seconds, 300))  # 限制 5-300 秒
    console.print(f"[accent]🔧 调用工具: wait({seconds})[/accent]")

    tool_result = await _do_wait(seconds, ctx)

    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": tool_result,
    })
    return tool_result


async def _do_wait(seconds: int, ctx: ToolHandlerContext) -> str:
    """实际执行等待逻辑。"""
    console.print(f"[muted]⏳ 等待回复 (最多 {seconds} 秒)...[/muted]")
    console.print("[bold magenta]💬 > [/bold magenta]", end="")

    user_input = await ctx.reader.get_line(timeout=seconds)

    if user_input is None:
        # 超时
        console.print()  # 换行
        console.print("[muted]⏳ 等待超时[/muted]")
        return "等待超时，用户未输入任何内容"

    user_input = user_input.strip()

    if not user_input:
        return "用户发送了空消息"

    # 更新 timing 时间戳
    now = datetime.now()
    ctx.last_user_input_time = now
    ctx.user_input_times.append(now)

    if user_input.lower() in ("/quit", "/exit", "/q"):
        return "[[QUIT]] 用户主动退出了对话"

    return f"用户说：{user_input}"


async def handle_create_table(tc, chat_history: list, ctx: ToolHandlerContext):
    """处理 create_table 工具。"""
    tbl_name = tc.arguments.get("name", "未命名")
    tbl_cols = tc.arguments.get("columns", [])
    tbl_rows = tc.arguments.get("rows", [])
    tbl_note = tc.arguments.get("note", "")
    console.print(f"[accent]🔧 调用工具: create_table(\"{tbl_name}\")[/accent]")

    table = Table(
        name=tbl_name,
        columns=tbl_cols,
        rows=tbl_rows,
        note=tbl_note,
    )
    ctx.tables[tbl_name] = table

    console.print(
        Panel(
            table.to_display(),
            title="📋 表格已创建",
            border_style="green",
            padding=(0, 1),
        )
    )
    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": f"表格「{tbl_name}」已创建，共 {len(tbl_cols)} 列 {len(tbl_rows)} 行。",
    })


async def handle_list_tables(tc, chat_history: list, ctx: ToolHandlerContext):
    """处理 list_tables 工具。"""
    console.print("[accent]🔧 调用工具: list_tables()[/accent]")
    if ctx.tables:
        summaries = [t.to_summary() for t in ctx.tables.values()]
        result_text = f"当前共有 {len(ctx.tables)} 个表格：\n" + "\n".join(summaries)
    else:
        result_text = "当前没有任何表格。"
    console.print(
        Panel(
            result_text,
            title="📋 表格列表",
            border_style="blue",
            padding=(0, 1),
        )
    )
    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result_text,
    })


async def handle_view_table(tc, chat_history: list, ctx: ToolHandlerContext):
    """处理 view_table 工具。"""
    view_name = tc.arguments.get("name", "")
    console.print(f"[accent]🔧 调用工具: view_table(\"{view_name}\")[/accent]")
    if view_name in ctx.tables:
        result_text = ctx.tables[view_name].to_display()
    else:
        available = ", ".join(ctx.tables.keys()) if ctx.tables else "无"
        result_text = f"表格「{view_name}」不存在。当前可用表格: {available}"
    console.print(
        Panel(
            result_text,
            title=f"📋 查看表格: {view_name}",
            border_style="blue",
            padding=(0, 1),
        )
    )
    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result_text,
    })


async def handle_mcp_tool(tc, chat_history: list, mcp_manager: "MCPManager"):
    """
    处理 MCP 工具调用。

    将调用转发到 MCPManager，展示结果并写入对话上下文。
    """
    # 格式化参数预览
    args_str = _json.dumps(tc.arguments, ensure_ascii=False)
    args_preview = args_str if len(args_str) <= 120 else args_str[:120] + "..."
    console.print(f"[accent]🔌 调用 MCP 工具: {tc.name}({args_preview})[/accent]")

    with console.status(
        f"[info]🔌 MCP 工具 {tc.name} 执行中...[/info]",
        spinner="dots",
    ):
        result = await mcp_manager.call_tool(tc.name, tc.arguments)

    # 展示结果（截断过长内容）
    display_text = result if len(result) <= 800 else result[:800] + "\n... (已截断)"
    console.print(
        Panel(
            display_text,
            title=f"🔌 MCP: {tc.name}",
            border_style="bright_green",
            padding=(0, 1),
        )
    )

    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result,
    })


async def handle_unknown_tool(tc, chat_history: list):
    """处理未知工具调用。"""
    console.print(f"[accent]🔧 调用工具: {tc.name}({tc.arguments})[/accent]")
    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": f"未知工具: {tc.name}",
    })
