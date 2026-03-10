"""
MaiDiary - 工具调用处理器
处理 LLM 循环中各工具（say/wait/stop/file/MCP）的执行逻辑。
"""

import json as _json
import asyncio
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from pathlib import Path

from rich.panel import Panel
from rich.markdown import Markdown

from config import console
from input_reader import InputReader
from llm_service import BaseLLMService

if TYPE_CHECKING:
    from mcp_client import MCPManager


# mai_files 目录路径
MAI_FILES_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mai_files"))


class ToolHandlerContext:
    """工具处理器所需的共享上下文。"""

    def __init__(
        self,
        llm_service: BaseLLMService,
        reader: InputReader,
        user_input_times: list[datetime],
    ):
        self.llm_service = llm_service
        self.reader = reader
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


async def handle_write_file(tc, chat_history: list):
    """处理 write_file 工具：在 mai_files 目录下写入文件。"""
    filename = tc.arguments.get("filename", "")
    content = tc.arguments.get("content", "")
    console.print(f"[accent]🔧 调用工具: write_file(\"{filename}\")[/accent]")

    # 确保目录存在
    MAI_FILES_DIR.mkdir(parents=True, exist_ok=True)

    # 构建完整文件路径
    file_path = MAI_FILES_DIR / filename

    try:
        # 创建父目录（如果需要）
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # 获取文件大小
        file_size = file_path.stat().st_size

        console.print(
            Panel(
                f"文件已写入: {filename}\n大小: {file_size} 字符",
                title="📁 文件已保存",
                border_style="green",
                padding=(0, 1),
            )
        )

        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"文件「{filename}」已成功写入，共 {file_size} 个字符。",
        })
    except Exception as e:
        error_msg = f"写入文件失败: {e}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })


async def handle_read_file(tc, chat_history: list):
    """处理 read_file 工具：读取 mai_files 目录下的文件。"""
    filename = tc.arguments.get("filename", "")
    console.print(f"[accent]🔧 调用工具: read_file(\"{filename}\")[/accent]")

    # 构建完整文件路径
    file_path = MAI_FILES_DIR / filename

    try:
        if not file_path.exists():
            error_msg = f"文件「{filename}」不存在。"
            console.print(f"[warning]{error_msg}[/warning]")
            chat_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": error_msg,
            })
            return

        if not file_path.is_file():
            error_msg = f"「{filename}」不是一个文件。"
            console.print(f"[warning]{error_msg}[/warning]")
            chat_history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": error_msg,
            })
            return

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        # 截断过长内容用于显示
        display_content = file_content
        if len(file_content) > 1000:
            display_content = file_content[:1000] + "\n... (内容已截断)"

        console.print(
            Panel(
                display_content,
                title=f"📄 文件内容: {filename}",
                border_style="blue",
                padding=(0, 1),
            )
        )

        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"文件「{filename}」内容：\n{file_content}",
        })
    except Exception as e:
        error_msg = f"读取文件失败: {e}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })


async def handle_list_files(tc, chat_history: list):
    """处理 list_files 工具：获取 mai_files 目录下所有文件的元信息。"""
    console.print("[accent]🔧 调用工具: list_files()[/accent]")

    try:
        # 确保目录存在
        MAI_FILES_DIR.mkdir(parents=True, exist_ok=True)

        # 获取所有文件
        files_info = []
        for item in MAI_FILES_DIR.rglob("*"):
            if item.is_file():
                # 获取相对路径
                rel_path = item.relative_to(MAI_FILES_DIR)
                stat = item.stat()
                files_info.append({
                    "name": str(rel_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })

        if not files_info:
            result_text = "mai_files 目录为空，没有任何文件。"
        else:
            # 按名称排序
            files_info.sort(key=lambda x: x["name"])
            # 格式化输出
            lines = [f"📁 mai_files 目录下共有 {len(files_info)} 个文件:\n"]
            for info in files_info:
                lines.append(f"  • {info['name']} ({info['size']} 字节, 修改于 {info['modified']})")
            result_text = "\n".join(lines)

        console.print(
            Panel(
                result_text,
                title="📁 文件列表",
                border_style="cyan",
                padding=(0, 1),
            )
        )

        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_text,
        })
    except Exception as e:
        error_msg = f"获取文件列表失败: {e}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
