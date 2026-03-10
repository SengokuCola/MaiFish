"""
MaiSaka - 主 Agent 状态管理器

管理主 Agent 的行为模式：
- free  : 自主 / 自由行动模式
- social: 与人类交互、需要情商感知的社交模式
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional


AgentMode = Literal["free", "social"]


class AgentModeEnum(str, Enum):
    FREE = "free"
    SOCIAL = "social"


class AgentStateManager:
    """主 Agent 状态管理器（单例）。"""

    def __init__(self) -> None:
        self._mode: AgentMode = AgentModeEnum.FREE
        self._last_switch_reason: Optional[str] = None

    # ── 基础接口 ──────────────────────────────────────────────
    @property
    def mode(self) -> AgentMode:
        return self._mode

    def set_mode(self, mode: AgentMode, reason: Optional[str] = None) -> None:
        """
        设置当前模式。

        Args:
            mode: "free" 或 "social"
            reason: 可选的切换原因（便于日志与调试）
        """
        if mode not in ("free", "social"):
            raise ValueError(f"invalid agent mode: {mode!r}")
        self._mode = mode
        self._last_switch_reason = reason

    # ── 便捷方法 ──────────────────────────────────────────────
    def to_free(self, reason: Optional[str] = None) -> None:
        self.set_mode("free", reason=reason)

    def to_social(self, reason: Optional[str] = None) -> None:
        self.set_mode("social", reason=reason)

    def is_free(self) -> bool:
        return self._mode == "free"

    def is_social(self) -> bool:
        return self._mode == "social"

    def debug_info(self) -> dict:
        return {
            "mode": self._mode,
            "last_switch_reason": self._last_switch_reason,
        }


# 全局单例
agent_state = AgentStateManager()


def tool_switch_mode(mode: AgentMode, reason: Optional[str] = None) -> str:
    """
    提供给 LLM 的工具接口：主动切换 Agent 模式。

    返回的字符串适合作为 tool 的自然语言结果写回 chat_history。
    """
    agent_state.set_mode(mode, reason=reason)
    return f"已切换主 Agent 模式为: {mode}（原因: {reason or '未指定'}）"

