"""
MaiSaka - Supervisor LLM 服务
Supervisor 负责为麦麦设定目标、进行元认知分析和评估表现
"""

import json
from dataclasses import dataclass
from typing import List, Optional
from openai import AsyncOpenAI

from prompt_loader import load_prompt
from llm_service.base import BaseLLMService, ModelInfo


def _build_symmetric_messages(main_history: List[dict]) -> List[dict]:
    """
    构建对称的消息历史。

    从 Supervisor 的视角来看：
    - MainAgent 的 assistant 消息 → user 消息（对方说的话）
    - MainAgent 的 user 消息 → assistant 消息（Supervisor 自己说的话）
    - 过滤掉 tool、perception 等系统消息

    Args:
        main_history: MainAgent 的对话历史

    Returns:
        角色转换后的消息历史
    """
    result = []
    for msg in main_history:
        role = msg.get("role", "")

        # 跳过系统消息
        if role in ["tool", "system"]:
            continue

        # 跳过 perception 消息
        if msg.get("_type") == "perception":
            continue

        # 角色互换
        if role == "assistant":
            result.append({
                "role": "user",
                "content": msg.get("content", ""),
            })
        elif role == "user":
            result.append({
                "role": "assistant",
                "content": msg.get("content", ""),
            })

    return result


@dataclass
class SupervisorResponse:
    """Supervisor 响应"""
    guidance: str  # 指导内容，将作为user消息传递给MainAgent
    should_terminate: bool = False  # 是否应该终止对话
    new_goal: Optional[str] = None  # 新的目标（如果设定了）


class SupervisorLLMService:
    """
    Supervisor 专用的 LLM 服务。

    注意：Supervisor 不继承 BaseLLMService，因为它有不同的职责和接口。
    它专注于分析、指导和评估，而不是直接参与对话。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.8,
    ):
        """
        初始化 Supervisor LLM 服务。

        Args:
            api_key: OpenAI API 密钥
            base_url: API 基地址
            model: 使用的模型名称（默认使用更便宜的模型）
            temperature: 温度参数（略高以增加多样性）
        """
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._enable_thinking = False  # Supervisor不使用思考模式

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # 状态跟踪
        self._current_goal: Optional[str] = None
        self._conversation_round: int = 0
        self._recent_guidances: List[str] = []  # 用于检测重复

    def get_model_info(self) -> ModelInfo:
        """返回当前使用的模型信息"""
        return ModelInfo(model_name=self._model, base_url=self._base_url)

    async def get_initial_goal(self) -> str:
        """
        获取 Supervisor 设定的初始目标。

        Returns:
            初始指导文本，将作为user消息传递给MainAgent
        """
        system_prompt = load_prompt("supervisor_init")

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请开始对话，为麦麦设定一个初始目标。"},
            ],
            temperature=self._temperature,
            max_tokens=500,
            extra_body={"enable_thinking": False},
        )

        initial_goal = response.choices[0].message.content or ""
        self._current_goal = initial_goal
        self._conversation_round = 1

        return initial_goal

    async def get_guidance(
        self,
        full_history: List[dict],
    ) -> SupervisorResponse:
        """
        分析主 Agent 的回复，生成指导。

        Args:
            full_history: 完整对话历史（MainAgent的视角）

        Returns:
            SupervisorResponse 包含指导内容和是否终止的标志
        """
        # 构建对称的消息历史（角色互换）
        symmetric_messages = _build_symmetric_messages(full_history)

        # 简单的对话摘要用于终止检查
        conversation_summary = "\n".join([
            f"{m['role']}: {m['content'][:100]}"
            for m in symmetric_messages[-6:]
        ])

        # 检查是否应该终止
        if self._should_check_termination():
            termination_check = await self._check_termination(conversation_summary)
            if termination_check:
                return SupervisorResponse(
                    guidance="我们的对话今天就到这里吧，很高兴和你聊天！",
                    should_terminate=True,
                )

        # 构建 Supervisor 的系统提示
        system_prompt = load_prompt(
            "supervisor_guidance",
            current_goal=self._current_goal or "无特定目标",
        )

        # 构建完整消息列表：system + 对称的历史对话
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        messages.extend(symmetric_messages)

        # 添加指导请求
        messages.append({
            "role": "user",
            "content": "请分析麦麦的回复，并给出你的下一步指导。",
        })

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=500,
            extra_body={"enable_thinking": False},
        )

        guidance = response.choices[0].message.content or ""

        # 更新状态
        self._conversation_round += 1
        self._recent_guidances.append(guidance)
        if len(self._recent_guidances) > 5:
            self._recent_guidances.pop(0)

        # 检测重复（简单检测）
        if self._is_repeating(guidance):
            guidance = self._break_repetition_pattern()

        return SupervisorResponse(guidance=guidance, should_terminate=False)

    async def handle_negotiation(
        self,
        negotiation_reason: str,
        your_original_guidance: str,
    ) -> str:
        """
        处理麦麦的协商请求。

        Args:
            negotiation_reason: 麦麦的协商理由
            your_original_guidance: 你原来的指导

        Returns:
            回应内容，将作为user消息传递给MainAgent
        """
        system_prompt = load_prompt(
            "supervisor_negotiate",
            main_agent_negotiation=negotiation_reason,
            your_original_guidance=your_original_guidance,
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请回应麦麦的协商请求。"},
            ],
            temperature=self._temperature,
            max_tokens=500,
            extra_body={"enable_thinking": False},
        )

        return response.choices[0].message.content or ""

    def _should_check_termination(self) -> bool:
        """每5轮检查一次是否应该终止"""
        return self._conversation_round % 5 == 0

    async def _check_termination(self, conversation_summary: str) -> bool:
        """
        检查是否应该终止对话。

        Args:
            conversation_summary: 对话摘要

        Returns:
            是否应该终止
        """
        system_prompt = """你是一个对话终止判断者。请分析当前对话状态，判断是否应该自然结束对话。

判断标准：
1. 对话是否已经自然结束（双方都没有更多要说）
2. 当前话题是否已经充分讨论
3. 是否已经达成了明确的目标
4. 继续对话是否还有意义

请只回答 "是" 或 "否"，不要有其他内容。"""

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"对话摘要：{conversation_summary[:500]}"},
                ],
                temperature=0.1,  # 低温度以获得确定性的答案
                max_tokens=10,
                extra_body={"enable_thinking": False},
            )

            answer = response.choices[0].message.content or ""
            return "是" in answer
        except Exception:
            return False

    def _is_repeating(self, new_guidance: str) -> bool:
        """检测是否陷入重复模式"""
        if len(self._recent_guidances) < 3:
            return False

        # 简单的相似度检测：检查最近3条guidance是否有相似的开头
        recent = self._recent_guidances[-3:]
        new_prefix = new_guidance[:50]

        similar_count = sum(1 for g in recent if g[:50] == new_prefix)
        return similar_count >= 2

    def _break_repetition_pattern(self) -> str:
        """打破重复模式，生成一个新的指导"""
        return "我们来聊点别的吧，你最近有什么有趣的经历或想法吗？"

    @property
    def current_goal(self) -> Optional[str]:
        """获取当前目标"""
        return self._current_goal

    @property
    def conversation_round(self) -> int:
        """获取当前对话轮数"""
        return self._conversation_round
