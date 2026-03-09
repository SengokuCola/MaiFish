"""
MaiDiary - 表格数据结构
用户/LLM 创建的持久化表格，用于存储任意结构化内容。
"""

from dataclasses import dataclass


@dataclass
class Table:
    """用户/LLM 创建的持久化表格，用于存储任意结构化内容。"""
    name: str
    columns: list[str]
    rows: list[list[str]]
    note: str = ""

    def to_display(self) -> str:
        """将表格格式化为可读文本，用于写入上下文和终端展示。"""
        parts: list[str] = []
        parts.append(f"📋 表格: {self.name}")
        if self.note:
            parts.append(f"   注释: {self.note}")
        # 列名
        header = " | ".join(self.columns)
        parts.append(f"   {header}")
        parts.append(f"   {'-+-'.join(['-' * max(len(c), 4) for c in self.columns])}")
        # 数据行
        for row in self.rows:
            # 确保行和列数对齐
            padded = row + [""] * (len(self.columns) - len(row))
            parts.append(f"   {' | '.join(padded[:len(self.columns)])}")
        parts.append(f"   共 {len(self.rows)} 行")
        return "\n".join(parts)

    def to_summary(self) -> str:
        """简要概述，用于 list_tables。"""
        note_part = f"  注释: {self.note}" if self.note else ""
        return (
            f"• {self.name} — 列: [{', '.join(self.columns)}], "
            f"{len(self.rows)} 行{note_part}"
        )
