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

ENABLE_TIMING_MODULE = os.getenv("ENABLE_TIMING_MODULE", "true").strip().lower() == "true"
ENABLE_MCP = os.getenv("ENABLE_MCP", "true").strip().lower() == "true"
ENABLE_WRITE_FILE = os.getenv("ENABLE_WRITE_FILE", "true").strip().lower() == "true"
ENABLE_READ_FILE = os.getenv("ENABLE_READ_FILE", "true").strip().lower() == "true"
ENABLE_LIST_FILES = os.getenv("ENABLE_LIST_FILES", "true").strip().lower() == "true"

# ──────────────────── Twin 模式配置 ────────────────────

ENABLE_TWIN_MODE = os.getenv("ENABLE_TWIN_MODE", "false").strip().lower() == "true"
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
TWIN_MAX_ROUNDS = int(os.getenv("TWIN_MAX_ROUNDS", "50"))
TWIN_TERMINATE_ON_GOAL = os.getenv("TWIN_TERMINATE_ON_GOAL", "true").strip().lower() == "true"

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
