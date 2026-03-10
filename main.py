"""
MaiSaka - 程序入口
使用方法:
    python main.py

环境变量 (可通过 .env 文件设置):
    OPENAI_API_KEY   - API 密钥
    OPENAI_BASE_URL  - API 基地址 (可选, 默认 https://api.openai.com/v1)
    OPENAI_MODEL     - 模型名称 (可选, 默认 gpt-4o)
    ENABLE_THINKING  - 是否启用思考模式 (可选, true/false, 不设置则不发送该参数)

Twin 模式（双LLM自主对话）:
    ENABLE_TWIN_MODE=true  - 启用双LLM自主对话模式
    SUPERVISOR_MODEL       - Supervisor 使用的模型（默认 gpt-4o-mini）
    TWIN_MAX_ROUNDS        - 最大对话轮次（默认 50）
"""

import asyncio

from config import console, ENABLE_TWIN_MODE
from cli import BufferCLI
from twin_coordinator import TwinCoordinator


def main():
    if ENABLE_TWIN_MODE:
        # Twin 模式：启动双LLM自主对话
        coordinator = TwinCoordinator.create()
        asyncio.run(coordinator.run_autonomous_loop())
    else:
        # 正常模式：用户输入驱动
        cli = BufferCLI()
        try:
            asyncio.run(cli.run())
        except KeyboardInterrupt:
            console.print("\n[muted]程序已终止[/muted]")
        finally:
            cli._debug_viewer.close()


if __name__ == "__main__":
    main()
