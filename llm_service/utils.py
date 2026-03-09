"""
MaiDiary - LLM 服务工具函数
"""


def format_chat_history(messages: list) -> str:
    """将聊天消息列表格式化为可读文本，用于子代理上下文构建。"""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "") or ""
        if role == "system":
            parts.append(f"[系统] {content[:500]}")
        elif role == "user":
            parts.append(f"[用户] {content[:500]}")
        elif role == "assistant":
            if content:
                parts.append(f"[助手思考] {content[:500]}")
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "?")
                args = func.get("arguments", "")
                if isinstance(args, str) and len(args) > 200:
                    args = args[:200] + "..."
                parts.append(f"[助手调用 {name}] {args}")
        elif role == "tool":
            parts.append(f"[工具结果] {content[:300]}")
    return "\n".join(parts)
