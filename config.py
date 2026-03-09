"""
MaiDiary - 全局配置
环境变量加载、Rich Console 实例、主题定义。
"""

from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme

# ──────────────────── 加载 .env ────────────────────

load_dotenv()

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
