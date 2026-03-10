"""
MaiSaka Debug Viewer — 在独立命令行窗口中显示每次 LLM 调用的完整 Prompt。

由主进程自动启动，通过 TCP socket 接收数据。
"""

import socket
import struct
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()

ROLE_STYLES = {
    "system":    ("📋", "bold blue"),
    "user":      ("👤", "bold green"),
    "assistant": ("🤖", "bold magenta"),
    "tool":      ("🔧", "bold yellow"),
}


def recv_exact(conn: socket.socket, n: int) -> bytes | None:
    """精确接收 n 字节数据。"""
    data = b""
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def format_message(idx: int, msg: dict) -> str:
    """格式化单条消息用于终端展示。"""
    role = msg.get("role", "?")
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls", [])
    tool_call_id = msg.get("tool_call_id", "")

    icon, style = ROLE_STYLES.get(role, ("❓", "white"))

    parts: list[str] = []

    # 消息头
    header = f"[{style}]{icon} [{idx}] {role}[/{style}]"
    if tool_call_id:
        header += f"  [dim](tool_call_id: {tool_call_id})[/dim]"
    parts.append(header)

    # 正文
    if content:
        display = content if len(content) <= 3000 else (
            content[:3000] + f"\n[dim]... (截断, 共 {len(content)} 字符)[/dim]"
        )
        parts.append(display)

    # 工具调用
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "?")
        args = func.get("arguments", "")
        if isinstance(args, str) and len(args) > 500:
            args = args[:500] + "..."
        parts.append(f"  [yellow]→ tool_call: {name}({args})[/yellow]")

    return "\n".join(parts)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 19876

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(1)

    console.print(
        Panel(
            f"[bold cyan]MaiSaka Debug Viewer[/bold cyan]\n"
            f"[dim]监听端口: {port}  等待主进程连接...[/dim]",
            box=box.DOUBLE_EDGE,
            border_style="cyan",
        )
    )

    conn, _ = server.accept()
    console.print("[green]✓ 已连接到主进程[/green]\n")

    call_count = 0
    try:
        while True:
            # 读 4 字节长度前缀
            length_bytes = recv_exact(conn, 4)
            if not length_bytes:
                break

            length = struct.unpack(">I", length_bytes)[0]

            # 读取 payload
            payload_bytes = recv_exact(conn, length)
            if not payload_bytes:
                break

            call_count += 1
            payload = json.loads(payload_bytes.decode("utf-8"))

            label = payload.get("label", "LLM Call")
            messages = payload.get("messages", [])
            tools = payload.get("tools")

            # ── 标题栏 ──
            console.print(f"\n{'═' * 90}")
            console.print(
                f"[bold yellow]#{call_count}  {label}[/bold yellow]  "
                f"[dim]({len(messages)} messages)[/dim]"
            )
            console.print(f"{'═' * 90}")

            # ── 逐条消息 ──
            for i, msg in enumerate(messages):
                console.print(format_message(i, msg))
                if i < len(messages) - 1:
                    console.print("[dim]─ ─ ─[/dim]")

            # ── tools 信息 ──
            if tools:
                tool_names = [
                    t.get("function", {}).get("name", "?") for t in tools
                ]
                console.print(
                    f"\n[dim]可用工具: {', '.join(tool_names)}[/dim]"
                )

            console.print(f"[dim]{'─' * 90}[/dim]")

    except (ConnectionResetError, ConnectionAbortedError):
        pass
    finally:
        conn.close()
        server.close()

    console.print("\n[red]连接已断开[/red]")
    input("按 Enter 关闭窗口...")


if __name__ == "__main__":
    main()
