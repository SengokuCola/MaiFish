"""
MaiSaka - 工具调用处理器
处理 LLM 循环中各工具（say/wait/file/MCP/QQ）的执行逻辑。
"""

import json as _json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional
from pathlib import Path
import importlib.util

from rich.panel import Panel
from rich.markdown import Markdown

from config import console
from input_reader import InputReader
from llm_service import BaseLLMService
from memory import get_memory_store
from say_rewriter import SayRewriter
from agent_state import agent_state, tool_switch_mode

if TYPE_CHECKING:
    from mcp_client import MCPManager


# 检查 aiohttp 是否可用（放在所有导入之后，避免 lint 警告）
AIOHTTP_AVAILABLE = importlib.util.find_spec("aiohttp") is not None
if AIOHTTP_AVAILABLE:
    import aiohttp


# mai_files 目录路径
MAI_FILES_DIR = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mai_files"))

# 全局 say 改写器
_say_rewriter: Optional[SayRewriter] = None


def get_say_rewriter(llm_service: BaseLLMService) -> SayRewriter:
    """获取 say 改写器实例（单例模式）"""
    global _say_rewriter
    if _say_rewriter is None:
        _say_rewriter = SayRewriter(llm_service)
    elif _say_rewriter._llm_service is None:
        _say_rewriter.set_llm_service(llm_service)
    return _say_rewriter


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
            rewriter = get_say_rewriter(ctx.llm_service)
            rewritten = await rewriter.rewrite(say_text)
        console.print(
            Panel(
                Markdown(rewritten),
                title="💬 MaiSaka",
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
        # 超时：说明当前暂时没有真实用户交互，切回 free 模式
        console.print()  # 换行
        console.print("[muted]⏳ 等待超时[/muted]")
        agent_state.to_free(reason="等待用户输入超时")
        return "等待超时，用户未输入任何内容"

    user_input = user_input.strip()

    if not user_input:
        # 有输入但为空，一般仍视为在对话中，保持 social 模式
        agent_state.to_social(reason="用户发送了空消息")
        return "用户发送了空消息"

    # 有真实的用户输入，切换为 social 模式
    agent_state.to_social(reason="收到用户输入")

    # 更新 timing 时间戳
    now = datetime.now()
    ctx.last_user_input_time = now
    ctx.user_input_times.append(now)

    if user_input.lower() in ("/quit", "/exit", "/q"):
        return "[[QUIT]] 用户主动退出了对话"

    return f"用户说：{user_input}"


async def handle_switch_mode(tc, chat_history: list):
    """
    处理 switch_mode 工具：主动切换主 Agent 的模式。

    参数：
    - mode: "free" 或 "social"
    - reason: 可选的切换原因说明
    """
    mode = tc.arguments.get("mode", "free")
    reason = tc.arguments.get("reason", "")
    console.print(f"[accent]🔧 调用工具: switch_mode(mode=\"{mode}\", reason=\"{reason}\")[/accent]")

    try:
        result = tool_switch_mode(mode, reason or None)
        debug = agent_state.debug_info()
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": f"{result}\n当前状态: {debug}",
        })
    except Exception as e:
        error_msg = f"切换状态失败: {e}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
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


async def handle_store_context(tc, chat_history: list, ctx: ToolHandlerContext):
    """
    处理 store_context 工具：将指定范围的对话上下文存入记忆系统，然后从对话中移除。

    参数：
    - count: 要存入记忆的消息数量（从最早的消息开始）
    - reason: 存入的原因
    """
    count = tc.arguments.get("count", 0)
    reason = tc.arguments.get("reason", "")
    console.print(f"[accent]🔧 调用工具: store_context(count={count}, reason=\"{reason}\")[/accent]")

    if count <= 0:
        error_msg = "count 参数必须大于 0"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
        return

    # 计算实际消息数量（排除 role=tool 的工具返回消息）
    actual_messages = [m for m in chat_history if m.get("role") != "tool"]

    if count > len(actual_messages):
        error_msg = f"count({count}) 超过了当前对话消息数量({len(actual_messages)})"
        console.print(f"[warning]{error_msg}[/warning]")
        count = len(actual_messages)

    # 找到要移除的消息索引（确保 tool_calls 和 tool 响应成对）
    indices_to_remove = []
    removed_count = 0
    i = 0

    while i < len(chat_history) and removed_count < count:
        msg = chat_history[i]
        role = msg.get("role", "")

        # 跳过 role=tool 的消息（它们会被对应的 assistant 消息一起处理）
        if role == "tool":
            i += 1
            continue

        # 检查这是否是一个带 tool_calls 的 assistant 消息
        if role == "assistant" and "tool_calls" in msg:
            # 检查这个消息是否包含当前的 tool_call（store_context 自己）
            # 如果包含，跳过不删除（否则会导致 tool 响应孤儿）
            current_tool_call_id = tc.id
            contains_current_call = any(
                tool_call.get("id") == current_tool_call_id
                for tool_call in msg.get("tool_calls", [])
            )
            if contains_current_call:
                i += 1
                continue

            # 收集这个 assistant 消息及其后续的 tool 响应消息
            block_indices = [i]
            j = i + 1
            while j < len(chat_history):
                next_msg = chat_history[j]
                if next_msg.get("role") == "tool":
                    block_indices.append(j)
                    j += 1
                else:
                    break
            indices_to_remove.extend(block_indices)
            removed_count += 1
            i = j
        elif role in ["user", "assistant"]:
            # 普通消息，可以直接删除
            indices_to_remove.append(i)
            removed_count += 1
            i += 1
        else:
            i += 1

    if not indices_to_remove:
        result_msg = "没有找到可存入记忆的消息"
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_msg,
        })
        return

    # 收集要总结的消息（在删除前）
    to_compress = []
    for i in sorted(indices_to_remove):
        if 0 <= i < len(chat_history):
            to_compress.append(chat_history[i])

    # 总结上下文并存入记忆
    try:
        with console.status(
            "[info]🧠 正在总结上下文并存入记忆...[/info]",
            spinner="dots",
        ):
            summary = await ctx.llm_service.summarize_context(to_compress)

            if summary:
                # 存入记忆
                memory_store = get_memory_store()

                # 调试输出：显示使用的记忆存储类型
                store_type = type(memory_store).__name__
                console.print(f"[muted]📦 使用记忆存储: {store_type}[/muted]")

                # 调试输出：显示要存储的内容
                console.print(f"[muted]📝 存储内容: {summary[:50]}...[/muted]")

                result = await memory_store.store_memory(
                    summary,
                    metadata={
                        "type": "manual_store",
                        "reason": reason,
                        "message_count": len(to_compress),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

                # 调试输出：显示存储结果
                if result:
                    console.print("[success]✅ 记忆存储成功[/success]")
                else:
                    console.print("[warning]⚠️ 记忆存储失败（可能使用了 Mock 存储）[/warning]")

                console.print(
                    Panel(
                        Markdown(summary),
                        title="🧠 上下文已存入记忆",
                        border_style="green",
                        padding=(0, 1),
                        style="dim",
                    )
                )

                result_msg = f"✅ 已将 {len(to_compress)} 条消息存入记忆\n原因: {reason}\n总结: {summary[:100]}..."
            else:
                result_msg = "⚠️ 上下文总结失败，未存入记忆"
                console.print(f"[warning]{result_msg}[/warning]")

    except Exception as e:
        result_msg = f"❌ 存入记忆时出错: {e}"
        console.print(f"[error]{result_msg}[/error]")

    # 从后往前删除消息
    for i in sorted(indices_to_remove, reverse=True):
        if 0 <= i < len(chat_history):
            chat_history.pop(i)

    # 清理"孤儿" tool 消息（没有对应 tool_calls 的 tool 消息）
    # 收集所有有效的 tool_call_id
    valid_tool_call_ids = set()
    for msg in chat_history:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                valid_tool_call_ids.add(tool_call.get("id", ""))

    # 删除无效的 tool 消息（从后往前）
    i = len(chat_history) - 1
    while i >= 0:
        msg = chat_history[i]
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            if tool_call_id not in valid_tool_call_ids:
                chat_history.pop(i)
        i -= 1

    chat_history.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result_msg,
    })


async def handle_get_qq_chat_info(tc, chat_history: list):
    """处理 get_qq_chat_info 工具：通过 HTTP 获取 QQ 聊天内容。"""
    chat = tc.arguments.get("chat", "")
    limit = tc.arguments.get("limit", 20)
    console.print(f"[accent]🔧 调用工具: get_qq_chat_info(\"{chat}\", limit={limit})[/accent]")

    if not AIOHTTP_AVAILABLE:
        error_msg = "aiohttp 模块未安装，请运行: pip install aiohttp"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
        return

    from config import QQ_API_BASE_URL, QQ_API_KEY
    if not QQ_API_BASE_URL:
        error_msg = "QQ_API_BASE_URL 未配置，请在 .env 中设置"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
        return

    try:
        # 构建 API 端点
        url = f"{QQ_API_BASE_URL.rstrip('/')}/api/external/chat/history"

        # 构建请求头（如果配置了 API Key）
        headers = {}
        if QQ_API_KEY:
            headers["Authorization"] = f"Bearer {QQ_API_KEY}"

        # 发送 HTTP 请求
        async with aiohttp.ClientSession() as session:
            params = {"chat": chat, "limit": limit}
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    # 获取纯文本响应
                    text = await response.text()

                    # 格式化显示
                    console.print(
                        Panel(
                            f"聊天标识: {chat}\n获取数量: {limit}\n\n{text if text.strip() else '暂无聊天记录'}",
                            title="💬 QQ 聊天记录",
                            border_style="cyan",
                            padding=(0, 1),
                        )
                    )

                    chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": text if text.strip() else "暂无聊天记录",
                    })
                else:
                    error_text = await response.text()
                    error_msg = f"HTTP 请求失败 (状态码 {response.status}): {error_text}"
                    console.print(f"[error]{error_msg}[/error]")
                    chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_msg,
                    })
    except Exception as e:
        error_msg = f"获取 QQ 聊天记录失败: {e}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })


async def handle_send_info(tc, chat_history: list):
    """处理 send_info 工具：通过 HTTP 发送消息到 QQ。"""
    chat = tc.arguments.get("chat", "")
    message = tc.arguments.get("message", "")
    console.print(f"[accent]🔧 调用工具: send_info(\"{chat}\")[/accent]")

    if not AIOHTTP_AVAILABLE:
        error_msg = "aiohttp 模块未安装，请运行: pip install aiohttp"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
        return

    from config import QQ_API_BASE_URL, QQ_API_KEY
    if not QQ_API_BASE_URL:
        error_msg = "QQ_API_BASE_URL 未配置，请在 .env 中设置"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
        return

    try:
        # 构建 API 端点
        url = f"{QQ_API_BASE_URL.rstrip('/')}/api/external/chat/send"

        # 构建请求头（如果配置了 API Key）
        headers = {}
        if QQ_API_KEY:
            headers["Authorization"] = f"Bearer {QQ_API_KEY}"

        # 发送 HTTP 请求
        async with aiohttp.ClientSession() as session:
            payload = {"chat": chat, "message": message}
            async with session.post(url, json=payload, headers=headers) as response:
                data = await response.json()

                if response.status == 200 and data.get("success"):
                    # 格式化显示
                    console.print(
                        Panel(
                            f"目标: {chat}\n消息: {message}\n\n结果: {data.get('message', '发送成功')}",
                            title="📤 消息已发送",
                            border_style="green",
                            padding=(0, 1),
                        )
                    )

                    chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"消息发送成功: {data.get('message', '发送成功')}",
                    })
                else:
                    error_msg = f"发送失败: {data.get('message', '未知错误')}"
                    console.print(f"[error]{error_msg}[/error]")
                    chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_msg,
                    })
    except Exception as e:
        error_msg = f"发送消息失败: {e}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })


async def handle_list_qq_chats(tc, chat_history: list):
    """处理 list_qq_chats 工具：获取所有可用的 QQ 聊天列表。"""
    console.print("[accent]🔧 调用工具: list_qq_chats()[/accent]")

    if not AIOHTTP_AVAILABLE:
        error_msg = "aiohttp 模块未安装，请运行: pip install aiohttp"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
        return

    from config import QQ_API_BASE_URL, QQ_API_KEY
    if not QQ_API_BASE_URL:
        error_msg = "QQ_API_BASE_URL 未配置，请在 .env 中设置"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })
        return

    try:
        # 构建 API 端点
        url = f"{QQ_API_BASE_URL.rstrip('/')}/api/external/chat/list"

        # 构建请求头（如果配置了 API Key）
        headers = {}
        if QQ_API_KEY:
            headers["Authorization"] = f"Bearer {QQ_API_KEY}"

        # 发送 HTTP 请求
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                data = await response.json()

                if response.status == 200 and data.get("success"):
                    chats = data.get("chats", [])

                    # 格式化聊天列表
                    if chats:
                        chat_list_text = "\n".join([
                            f"  • [{c.get('platform', 'qq')}] {c.get('name', '未知')} (chat: {c.get('chat', 'N/A')})"
                            for c in chats
                        ])
                        result_text = f"可用的聊天 (共 {len(chats)} 个):\n{chat_list_text}"
                    else:
                        result_text = "没有可用的聊天"

                    console.print(
                        Panel(
                            result_text,
                            title="💬 QQ 聊天列表",
                            border_style="cyan",
                            padding=(0, 1),
                        )
                    )

                    chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })
                else:
                    error_msg = f"获取失败: {data.get('message', '未知错误')}"
                    console.print(f"[error]{error_msg}[/error]")
                    chat_history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": error_msg,
                    })
    except Exception as e:
        error_msg = f"获取聊天列表失败: {e}"
        console.print(f"[error]{error_msg}[/error]")
        chat_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": error_msg,
        })


# ──────────────────── 初始化 mai_files 目录 ────────────────────

# 确保程序启动时 mai_files 目录存在
try:
    MAI_FILES_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    console.print(f"[warning]创建 mai_files 目录失败: {e}[/warning]")
