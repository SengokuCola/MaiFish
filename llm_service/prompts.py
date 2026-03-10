"""
MaiSaka - LLM 工具定义
所有 Tool Schema 集中管理。

注意：所有 Prompt 模板已迁移至 prompts/ 目录，使用 .prompt 文件存储。
使用 prompt_loader.load_prompt() 加载模板。
"""

# ──────────────────── 工具定义 ────────────────────

# 核心工具（始终启用）
CORE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "say",
            "description": (
                "对用户说话。你的所有正式发言都必须通过此工具输出。"
                "直接输出的 content 文本会被视为你的内心思考，用户无法看到。"
                "请你读读之前的聊天记录，把握当前的话题，然后给出日常且简短的回复。最好一次对一个话题进行回复，免得啰嗦或者回复内容太乱。请注意不要输出多余内容(包括不必要的前后缀，冒号，括号，at或 @等 )，只输出发言内容就好。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要对用户说的话，尽量简短且口语化",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": (
                "暂时结束你的发言，把话语权交给用户，等待对方说话。"
                "指定等待的最大秒数。"
                "如果用户在等待期间说了话，你会通过工具结果收到内容；"
                "如果超时对方没有说话，你会收到超时通知。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "等待的秒数（1-24*3600",
                    }
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": (
                "结束当前对话循环，进入待机状态。"
                "调用后主循环会停止，直到用户下次输入新内容时重新唤醒。"
                "适合在对话自然结束、用户不再回复、或深夜等不适合继续聊天时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

# 可选工具（可通过配置启用/禁用）
OPTIONAL_TOOLS = {
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "在 mai_files 目录下写入文件，支持任意格式（文本、代码、Markdown等）。"
                "如果文件已存在，会覆盖原有内容。可用于保存笔记、代码片段、配置等。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，可包含路径，如 'notes.txt' 或 'diary/2024-03-09.md'",
                    },
                    "content": {
                        "type": "string",
                        "description": "要写入的文件内容",
                    },
                },
                "required": ["filename", "content"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "读取 mai_files 目录下的文件内容。"
                "返回文件的完整文本内容。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "要读取的文件名，可包含路径",
                    },
                },
                "required": ["filename"],
            },
        },
    },
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": (
                "获取 mai_files 目录下所有文件的元信息列表。"
                "返回每个文件的名称、大小、修改时间等信息，帮助你了解有哪些文件可用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
}

# 始终启用的工具
ALWAYS_ENABLED_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "store_context",
            "description": (
                "将指定范围的对话上下文存入记忆系统，然后从当前对话中移除这些内容。"
                "适合在以下情况使用："
                "1. 对话上下文过长，需要压缩以保持效率"
                "2. 对话话题已经转换，旧话题的内容可以归档"
                "3. 遇到重要的对话内容，需要保存到长期记忆中"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "要存入记忆的消息数量，从最早的消息开始计算。例如传入10会将最早的10条消息存入记忆并移除。",
                    },
                    "reason": {
                        "type": "string",
                        "description": "说明为什么要存入这段上下文，帮助记忆系统更好地组织信息。例如：「话题从游戏转换到了工作」或「上下文过长需要压缩」。",
                    },
                },
                "required": ["count", "reason"],
            },
        },
    },
]

# ──────────────────── 主 Agent 工具定义 ────────────────────

# 保持原有的 CHAT_TOOLS 用于向后兼容
CHAT_TOOLS = CORE_TOOLS + [
    OPTIONAL_TOOLS["write_file"],
    OPTIONAL_TOOLS["read_file"],
    OPTIONAL_TOOLS["list_files"],
    ALWAYS_ENABLED_TOOLS[0],
]


def get_enabled_chat_tools(
    enable_write_file: bool = True,
    enable_read_file: bool = True,
    enable_list_files: bool = True,
) -> list:
    """
    根据配置获取启用的工具列表。

    Args:
        enable_write_file: 是否启用 write_file 工具
        enable_read_file: 是否启用 read_file 工具
        enable_list_files: 是否启用 list_files 工具

    Returns:
        启用的工具列表
    """
    tools = CORE_TOOLS + ALWAYS_ENABLED_TOOLS

    if enable_write_file:
        tools.append(OPTIONAL_TOOLS["write_file"])
    if enable_read_file:
        tools.append(OPTIONAL_TOOLS["read_file"])
    if enable_list_files:
        tools.append(OPTIONAL_TOOLS["list_files"])

    return tools
