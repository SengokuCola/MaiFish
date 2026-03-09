"""
MaiDiary - 数据模型定义
统一的多模态内容存储格式
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class ContentType(Enum):
    """内容模态类型"""
    TEXT = "text"
    IMAGE = "image"


@dataclass
class ContentItem:
    """
    统一的内容存储格式，区分不同模态。
amemo
    Attributes:
        id:           唯一标识
        modality:     模态类型 (TEXT / IMAGE)
        content:      文本内容 或 图片描述
        timestamp:    创建时间
        base64_data:  图片的 base64 编码 (仅图片)
        image_format: 图片格式，如 jpeg, png (仅图片)
        file_name:    原始文件名 (仅图片)
    """
    id: int
    modality: ContentType
    content: str
    timestamp: datetime
    base64_data: Optional[str] = None
    image_format: Optional[str] = None
    file_name: Optional[str] = None

    def display_content(self, max_len: int = 80) -> str:
        """返回用于显示的内容摘要"""
        if self.modality == ContentType.IMAGE:
            size_kb = len(self.base64_data) * 3 / 4 / 1024 if self.base64_data else 0
            return f"🖼️  {self.file_name or '未知图片'} ({self.image_format}, {size_kb:.1f}KB)"
        else:
            text = self.content.replace("\n", " ")
            if len(text) > max_len:
                return text[:max_len] + "..."
            return text

    def modality_label(self) -> str:
        """返回模态标签"""
        return "📝 文本" if self.modality == ContentType.TEXT else "🖼️ 图片"
