"""
MaiSaka - Say 发言改写器
将正式发言改写为口语化贴吧风格。
"""

from typing import Optional
from prompt_loader import load_prompt
from llm_service import BaseLLMService


class SayRewriter:
    """
    发言改写器。

    将 say 工具的输出文本改写为口语化贴吧风格。
    """

    def __init__(self, llm_service: Optional[BaseLLMService] = None):
        """
        初始化改写器。

        Args:
            llm_service: LLM 服务实例，如果为 None 则需要在调用前设置
        """
        self._llm_service = llm_service
        self._enabled = True

    def set_llm_service(self, llm_service: BaseLLMService) -> None:
        """设置 LLM 服务"""
        self._llm_service = llm_service

    def set_enabled(self, enabled: bool) -> None:
        """启用/禁用改写功能"""
        self._enabled = enabled

    async def rewrite(self, text: str) -> str:
        """
        将文本改写为口语化贴吧风格。

        Args:
            text: 原始文本

        Returns:
            改写后的文本，失败时返回原文
        """
        if not self._enabled or not text or self._llm_service is None:
            return text
        
        text = f"我的想法是，我想根据以下内容进行回复：{text}。"
        text += f"但是我不能直接回复，现在我要参考回复要求改写回复："

        # 构建改写消息
        messages = [
            {"role": "system", "content": load_prompt("say_rewrite.system")},
            {"role": "assistant", "content": text},
        ]

        try:
            # 调用 LLM 改写
            from llm_service.openai_impl import OpenAILLMService
            if isinstance(self._llm_service, OpenAILLMService):
                extra_body = self._llm_service._build_extra_body()
                response = await self._llm_service._call_llm(
                    "say 风格改写",
                    messages,
                    temperature=0.8,
                    max_tokens=512,
                    **({"extra_body": extra_body} if extra_body else {}),
                )
                result = response.choices[0].message.content or text
                return result.strip()
        except Exception:
            pass

        # 改写失败时回退到原文
        return text
