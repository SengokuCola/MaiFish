"""
MaiDiary - SQLite 存储模块
提供基于 SQLite 的内容持久化存储，支持文本和图片。

数据库结构:
    content_items  - 存储所有内容项 (文本/图片)
    summaries      - 存储 LLM 生成的总结历史
"""

import os
import base64
import sqlite3
from io import BytesIO
from datetime import datetime
from typing import List, Optional

from PIL import Image

from models import ContentItem, ContentType


# ──────────────────── 常量 ────────────────────

SUPPORTED_IMAGE_FORMATS = {"jpg", "jpeg", "png", "webp", "gif", "bmp"}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS content_items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    modality      TEXT    NOT NULL CHECK(modality IN ('text', 'image')),
    content       TEXT    NOT NULL,
    timestamp     TEXT    NOT NULL,
    base64_data   TEXT,
    image_format  TEXT,
    file_name     TEXT
);

CREATE TABLE IF NOT EXISTS summaries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    summary       TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);
"""


# ──────────────────── 存储类 ────────────────────

class ContentStorage:
    """
    基于 SQLite 的内容缓存区持久化存储。

    - 每个 ContentItem 存入 content_items 表
    - LLM 总结存入 summaries 表
    - 支持按 ID 增删查，以及清空操作
    """

    def __init__(self, db_path: str = "media_buffer.db"):
        """
        Args:
            db_path: SQLite 数据库文件路径
        """
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()

    def close(self):
        """关闭数据库连接"""
        self._conn.close()

    # ──────── 添加内容 ────────

    def add_text(self, text: str) -> ContentItem:
        """添加文本内容"""
        now = datetime.now()
        cursor = self._conn.execute(
            """INSERT INTO content_items (modality, content, timestamp)
               VALUES (?, ?, ?)""",
            (ContentType.TEXT.value, text, now.isoformat()),
        )
        self._conn.commit()

        return ContentItem(
            id=cursor.lastrowid,
            modality=ContentType.TEXT,
            content=text,
            timestamp=now,
        )

    def add_image(self, image_path: str) -> ContentItem:
        """
        添加图片文件到存储。
        自动读取、验证、编码为 base64 并存入数据库。

        Args:
            image_path: 图片文件路径

        Raises:
            FileNotFoundError: 文件不存在
            ValueError:        格式不支持 / 文件无效
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"找不到文件: {image_path}")

        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        if ext not in SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"不支持的图片格式: .{ext}\n"
                f"支持的格式: {', '.join(sorted(SUPPORTED_IMAGE_FORMATS))}"
            )

        # 读取文件
        with open(image_path, "rb") as f:
            image_data = f.read()

        # 验证图片
        try:
            img = Image.open(BytesIO(image_data))
            img.verify()
        except Exception as e:
            raise ValueError(f"无法识别为有效图片文件: {e}")

        # 标准化格式
        format_map = {"jpg": "jpeg", "bmp": "png"}
        image_format = format_map.get(ext, ext)

        # bmp → png 转换
        if ext == "bmp":
            img = Image.open(BytesIO(image_data))
            buf = BytesIO()
            img.save(buf, format="PNG")
            image_data = buf.getvalue()
            image_format = "png"

        base64_data = base64.b64encode(image_data).decode("utf-8")
        file_name = os.path.basename(image_path)
        content = f"[图片: {file_name}]"
        now = datetime.now()

        cursor = self._conn.execute(
            """INSERT INTO content_items
               (modality, content, timestamp, base64_data, image_format, file_name)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ContentType.IMAGE.value, content, now.isoformat(),
             base64_data, image_format, file_name),
        )
        self._conn.commit()

        return ContentItem(
            id=cursor.lastrowid,
            modality=ContentType.IMAGE,
            content=content,
            timestamp=now,
            base64_data=base64_data,
            image_format=image_format,
            file_name=file_name,
        )

    def add_image_from_base64(
        self,
        base64_data: str,
        image_format: str = "png",
        name: str = "clipboard",
    ) -> ContentItem:
        """从 base64 数据直接添加图片"""
        content = f"[图片: {name}]"
        now = datetime.now()

        cursor = self._conn.execute(
            """INSERT INTO content_items
               (modality, content, timestamp, base64_data, image_format, file_name)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ContentType.IMAGE.value, content, now.isoformat(),
             base64_data, image_format, name),
        )
        self._conn.commit()

        return ContentItem(
            id=cursor.lastrowid,
            modality=ContentType.IMAGE,
            content=content,
            timestamp=now,
            base64_data=base64_data,
            image_format=image_format,
            file_name=name,
        )

    # ──────── 查询 ────────

    def get_all_items(self) -> List[ContentItem]:
        """获取所有内容项（按 ID 正序）"""
        rows = self._conn.execute(
            "SELECT * FROM content_items ORDER BY id ASC"
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_text_items(self) -> List[ContentItem]:
        """获取所有文本项"""
        rows = self._conn.execute(
            "SELECT * FROM content_items WHERE modality = ? ORDER BY id ASC",
            (ContentType.TEXT.value,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def get_image_items(self) -> List[ContentItem]:
        """获取所有图片项"""
        rows = self._conn.execute(
            "SELECT * FROM content_items WHERE modality = ? ORDER BY id ASC",
            (ContentType.IMAGE.value,),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    @property
    def size(self) -> int:
        """缓存区内容数量"""
        row = self._conn.execute("SELECT COUNT(*) FROM content_items").fetchone()
        return row[0]

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    # ──────── 删除 ────────

    def remove(self, item_id: int) -> bool:
        """按 ID 移除内容项，返回是否成功"""
        cursor = self._conn.execute(
            "DELETE FROM content_items WHERE id = ?", (item_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear(self):
        """清空所有内容项和总结"""
        self._conn.execute("DELETE FROM content_items")
        self._conn.execute("DELETE FROM summaries")
        self._conn.commit()

    # ──────── 总结管理 ────────

    def save_summary(self, summary: str):
        """保存一条 LLM 总结"""
        self._conn.execute(
            "INSERT INTO summaries (summary, created_at) VALUES (?, ?)",
            (summary, datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_latest_summary(self) -> Optional[str]:
        """获取最新一条总结"""
        row = self._conn.execute(
            "SELECT summary FROM summaries ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row["summary"] if row else None

    def get_all_summaries(self) -> List[dict]:
        """获取所有总结历史"""
        rows = self._conn.execute(
            "SELECT id, summary, created_at FROM summaries ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    # ──────── 内部方法 ────────

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> ContentItem:
        """将数据库行转换为 ContentItem"""
        return ContentItem(
            id=row["id"],
            modality=ContentType(row["modality"]),
            content=row["content"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            base64_data=row["base64_data"],
            image_format=row["image_format"],
            file_name=row["file_name"],
        )
