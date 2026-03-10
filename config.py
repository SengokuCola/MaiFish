"""
MaiSaka - 全局配置
环境变量加载、Rich Console 实例、主题定义。
"""

import os
from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme

# ──────────────────── 加载 .env ────────────────────

load_dotenv()

# ──────────────────── 模块开关配置 ────────────────────

ENABLE_EMOTION_MODULE = os.getenv("ENABLE_EMOTION_MODULE", "true").strip().lower() == "true"
ENABLE_TIMING_MODULE = os.getenv("ENABLE_TIMING_MODULE", "true").strip().lower() == "true"

# ──────────────────── Rich 主题 & Console ────────────────────

custom_theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "muted": "dim",
        "accent": "bold magenta",
    }
)

console = Console(theme=custom_theme)
