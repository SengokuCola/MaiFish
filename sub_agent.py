"""
MaiDiary - 记忆子代理
当主对话上下文达到上限时自动创建，持续监控主对话中与特定记忆领域相关的信息。
"""


class SubAgent:
    """
    记忆子代理。

    当主对话上下文达到上限时自动创建，持续监控主对话中与特定记忆领域相关的信息。
    每轮主循环会并行触发所有子代理，检查是否有需要提示主 Agent 的信息。
    """
    _counter = 0

    def __init__(self, memory_description: str, initial_summary: str):
        SubAgent._counter += 1
        self.id = SubAgent._counter
        self.memory_description = memory_description
        self.history: list = [
            {
                "role": "system",
                "content": (
                    f"你是记忆监控子代理 #{self.id}。\n"
                    f"你负责监控的记忆领域：{memory_description}\n\n"
                    f"你的工作方式：\n"
                    f"1. 每当主对话产生新一轮内容时，你会收到该轮对话的内容\n"
                    f"2. 结合你的初始上下文，判断新内容是否与你负责的记忆领域相关\n"
                    f"3. 如果有主 Agent 需要知道的信息，使用 offer_info 工具提供\n"
                    f"4. 如果没有需要提供的信息，不要调用任何工具\n\n"
                    f"你可以在直接输出的 content 中进行思考分析（这会被展示出来），\n"
                    f"但只有通过 offer_info 工具提供的信息才会被传达给主 Agent。\n"
                    f"注意：提示要简洁、有针对性，不要重复已经传达过的信息。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"以下是根据你的记忆领域，从主对话中提取的初始上下文总结：\n\n"
                    f"{initial_summary}"
                ),
            },
            {
                "role": "assistant",
                "content": "收到，我已理解我的记忆领域和初始上下文。我会持续关注主对话中与此相关的信息。",
            },
        ]
        self.last_checked_index: int = 0
