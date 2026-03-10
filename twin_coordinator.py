"""
MaiSaka - 双LLM对话协调器
管理 Supervisor 和 MainAgent 之间的自主对话循环
"""

import asyncio
import os
from typing import Optional

from rich import box
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text

from config import (
    console,
    SUPERVISOR_MODEL,
    TWIN_MAX_ROUNDS,
)
from agent_state import agent_state
from cli import BufferCLI
from llm_service.supervisor_impl import SupervisorLLMService


class TwinCoordinator:
    """双LLM对话协调器"""

    def __init__(
        self,
        supervisor_llm: SupervisorLLMService,
        main_cli: BufferCLI,
    ):
        """
        初始化协调器。

        Args:
            supervisor_llm: Supervisor LLM 服务
            main_cli: MainAgent CLI 实例
        """
        self.supervisor_llm = supervisor_llm
        self.main_cli = main_cli
        self.conversation_round = 0
        self.last_supervisor_guidance = ""

    @classmethod
    def create(cls) -> "TwinCoordinator":
        """
        创建并初始化 TwinCoordinator 实例。

        Returns:
            配置好的 TwinCoordinator 实例
        """
        # 创建 Supervisor LLM 服务
        supervisor_llm = SupervisorLLMService(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", ""),
            model=SUPERVISOR_MODEL,
        )

        # 创建 MainAgent CLI
        main_cli = BufferCLI()
        # LLM 服务已在 __init__ 中初始化

        return cls(supervisor_llm, main_cli)

    async def run_autonomous_loop(self):
        """自主运行主循环"""
        self._show_twin_banner()

        # 进入 Twin 自主模式，默认切换为 free 模式
        agent_state.to_free(reason="进入 Twin 自主对话模式")

        # 初始化 MCP（如果启用）
        from config import ENABLE_MCP
        if ENABLE_MCP:
            await self.main_cli._init_mcp()
        else:
            console.print("[muted]🔌 MCP 已禁用 (ENABLE_MCP=false)[/muted]")

        try:
            # 1. Supervisor 设定初始目标
            console.print("[info]🎯 Supervisor 正在设定初始目标...[/info]")
            initial_goal = await self.supervisor_llm.get_initial_goal()

            console.print(
                Panel(
                    Markdown(initial_goal),
                    title="🎯 初始目标",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

            # 2. 开始自主对话循环
            await self._autonomous_loop(initial_goal)

        except KeyboardInterrupt:
            console.print("\n[muted]程序已终止[/muted]")
        finally:
            self.main_cli._debug_viewer.close()
            if hasattr(self.main_cli, '_mcp_manager') and self.main_cli._mcp_manager:
                await self.main_cli._mcp_manager.close()

    async def _autonomous_loop(self, initial_goal: str):
        """自主对话循环"""
        current_guidance = initial_goal
        self.conversation_round = 0

        # 自主循环整体以 free 模式为主
        agent_state.to_free(reason="开始 Twin 自主循环")

        while not self._should_terminate():
            self.conversation_round += 1
            # 每轮开始时确保处于 free 模式（Supervisor ↔ Main 的内部对话）
            agent_state.to_free(reason=f"Twin 第 {self.conversation_round} 轮开始（Supervisor ↔ Main 自主对话）")
            console.print(f"\n[muted]═══ 第 {self.conversation_round} 轮 ═══[/muted]")

            # 显示 Supervisor 的指导
            console.print(
                Panel(
                    Markdown(current_guidance),
                    title=f"💬 Supervisor 指导 (轮次 {self.conversation_round})",
                    border_style="bright_blue",
                    padding=(0, 1),
                )
            )

            # 将指导作为 user 消息传递给 MainAgent
            # 模拟用户输入
            await self._run_main_agent_cycle(current_guidance)

            # 检查是否有协商请求（从 chat_history 中检测）
            negotiation_info = self._check_negotiation_request()
            if negotiation_info:
                await self._handle_negotiation(current_guidance, negotiation_info)
                current_guidance = self.last_supervisor_guidance
                # 清除协商标记
                self._clear_negotiation_marker()
                continue

            # Supervisor 分析并生成新的指导（使用完整的对称消息历史）
            supervisor_response = await self.supervisor_llm.get_guidance(
                full_history=self.main_cli.get_chat_history(),
            )

            # 检查是否应该终止
            if supervisor_response.should_terminate:
                console.print("[success]🏁 对话自然结束[/success]")
                console.print(
                    Panel(
                        supervisor_response.guidance,
                        title="👋 结束语",
                        border_style="green",
                        padding=(0, 1),
                    )
                )
                break

            current_guidance = supervisor_response.guidance

            # 检查轮次限制
            if self.conversation_round >= TWIN_MAX_ROUNDS:
                console.print(f"[warning]⚠️ 达到最大轮次限制 ({TWIN_MAX_ROUNDS})[/warning]")
                break

            # 短暂延迟，让观察者看清楚
            await asyncio.sleep(1)

    async def _run_main_agent_cycle(self, user_input: str):
        """运行 MainAgent 的一个对话周期"""
        # 初始化对话历史（如果是第一轮）
        if self.main_cli._chat_history is None:
            self.main_cli._chat_history = self.main_cli.llm_service.build_chat_context(user_input)
        else:
            # 追加用户消息
            self.main_cli._chat_history.append({
                "role": "user",
                "content": user_input,
            })

        # 运行 LLM 循环（Twin 模式：单步执行一次“思考 + 工具”）
        # 这样可以保证：每次主 Agent 完成一轮思考并调用工具后，就会回到 Supervisor 继续指导
        await self.main_cli._run_llm_loop(
            chat_history=self.main_cli._chat_history,
            max_steps=1,
        )

    def _check_negotiation_request(self) -> Optional[str]:
        """检查 chat_history 中是否有协商请求"""
        history = self.main_cli.get_chat_history()
        if not history:
            return None

        for msg in reversed(history):
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if "[NEGOTIATION_REQUEST]" in content:
                    # 提取协商理由
                    return content.replace("[NEGOTIATION_REQUEST]", "").strip()

        return None

    def _clear_negotiation_marker(self):
        """清除 chat_history 中的协商标记"""
        history = self.main_cli.get_chat_history()
        if not history:
            return

        # 移除协商标记消息
        i = len(history) - 1
        while i >= 0:
            msg = history[i]
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if "[NEGOTIATION_REQUEST]" in content:
                    history.pop(i)
                    break
            i -= 1

    async def _handle_negotiation(self, original_guidance: str, negotiation_reason: str):
        """处理协商请求"""
        console.print("[warning]🤔 检测到协商请求，启动协商流程...[/warning]")

        # Supervisor 处理协商
        supervisor_response = await self.supervisor_llm.handle_negotiation(
            negotiation_reason=negotiation_reason,
            your_original_guidance=original_guidance,
        )

        console.print(
            Panel(
                Markdown(supervisor_response),
                title="🤝 协商结果",
                border_style="yellow",
                padding=(0, 1),
            )
        )

        # 更新最后的指导
        self.last_supervisor_guidance = supervisor_response

    def _should_terminate(self) -> bool:
        """检查是否应该终止"""
        return self.conversation_round >= TWIN_MAX_ROUNDS

    def _show_twin_banner(self):
        """显示 Twin 模式横幅"""
        banner = Text()
        banner.append("MaiSaka", style="bold cyan")
        banner.append(" Twin Mode\n", style="magenta")
        banner.append("双LLM自主对话模式", style="muted")
        banner.append(" | Supervisor → MainAgent\n", style="muted")
        banner.append(f"最大轮次: {TWIN_MAX_ROUNDS}", style="muted")

        console.print(Panel(banner, box=box.DOUBLE_EDGE, border_style="magenta", padding=(1, 2)))
        console.print()
