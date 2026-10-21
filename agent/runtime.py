"""Agent Runtime：核心循环，完全自研，不依赖任何 Agent 框架。

对应题目"要求 2"的四步循环：
    Step 1 接收用户输入            -> handle(session_id, user_input)
    Step 2 判断直接回复还是调工具  -> LLM 输出经 parser 解析：Final Answer 还是 Action
    Step 3 调用工具                -> ToolRegistry.execute，结果作为 Observation 回灌
    Step 4 依据工具结果决定继续    -> 回到 Step 2，直到 Final Answer 或达到最大步数

容错策略（题目"额外要求"）：
- 模型输出不合规：注入一条格式纠正提示，让它重来（计入步数，不会死循环）；
- 工具失败：错误以 "Error: ..." 作为 Observation 回灌，模型可自行修正参数重试；
- LLM 服务不可用：捕获后给用户可读回复，不让异常穿透到 CLI；
- 每步写 trace（输入、原始输出、解析结果、工具结果），可随时 /trace 查看。
"""
from __future__ import annotations

import logging
import time

from .context import ContextManager
from .llm import LLMClient
from .parser import parse_llm_output
from .session import Session, SessionStore
from .tools import ToolContext, ToolRegistry

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """你是一个具备工具调用能力的助手。你可以使用以下工具：

{tool_schemas}

## 输出格式（必须严格遵守，每一步只输出两种形式之一）

形式一：调用工具
Thought: <你的推理过程>
Action: <工具名称，必须是上面列出的名字>
Action Input: <合法的 JSON 对象，参数需符合该工具的 Schema>

形式二：给出最终答案
Thought: <你的推理过程>
Final Answer: <给用户的最终答复>

## 规则
- 一次只能调用一个工具；调用后系统会返回 Observation（工具结果），再决定下一步。
- 已有信息足够回答时，直接输出 Final Answer，不要无谓地调用工具。
- Action Input 必须是合法 JSON，不要包裹在 markdown 代码块里。
- 用中文回答用户。"""

# 输出格式不合规时的纠正提示，作为一条普通 user 消息注入
_FORMAT_NUDGE = (
    "系统提示：你的上一条输出不符合约定格式。请严格按两种形式之一重新输出："
    "调用工具（Thought/Action/Action Input）或最终回复（Thought/Final Answer）。"
)


class AgentRuntime:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        store: SessionStore,
        context: ContextManager | None = None,
        max_steps: int = 8,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.store = store
        self.context = context or ContextManager()
        # max_steps 是单次 handle 内"模型调用次数"的上限，
        # 防止模型反复调工具不收手（死循环保护）
        self.max_steps = max_steps
        self.system_prompt = _SYSTEM_TEMPLATE.format(
            tool_schemas=tools.schemas_prompt()
        )

    def handle(self, session_id: str, user_input: str) -> str:
        """处理一轮用户输入，返回给用户的最终文本。"""
        session = self.store.get_or_create(session_id)

        # ---- 最大轮次限制（硬上限）----
        if session.user_rounds >= self.context.config.max_rounds:
            msg = (f"本会话已达到最大轮次限制（{self.context.config.max_rounds} 轮），"
                   "请开启新会话继续。")
            logger.info("session=%s 达到最大轮次，拒绝新输入", session_id)
            return msg

        if not user_input or not user_input.strip():
            return "输入为空，请输入有效内容。"

        # Step 1：接收用户输入
        session.add_input(user_input.strip())
        logger.info("session=%s 收到输入: %s", session_id, user_input[:80])

        try:
            answer = self._loop(session)
        except Exception as e:
            # LLM 连续失败等致命异常：不穿透到调用方，给可读回复
            logger.exception("session=%s loop 异常", session_id)
            answer = f"抱歉，处理过程中出现错误（{type(e).__name__}），请稍后重试。"

        self.store.save(session)
        return answer

    # ------------------------------------------------------------ 内部
    def _loop(self, session: Session) -> str:
        nudged = False  # 每轮输入最多纠正一次格式，避免陷入"纠正-再错-再纠正"
        for step in range(1, self.max_steps + 1):
            # 过长先压缩，再组装 messages（压缩时机：每步调用前检查）
            self.context.maybe_compress(session, self.llm)
            messages = self.context.build_messages(session, self.system_prompt)

            raw = self.llm.chat(messages)

            # Step 2：解析模型输出，判断是直接回复还是调用工具
            parsed = parse_llm_output(raw)
            session.add_assistant(raw)
            self._trace(session, step, raw, parsed_action=parsed.action,
                        parsed_final=parsed.final_answer is not None)

            if parsed.final_answer is not None:
                logger.info("session=%s 第 %d 步给出最终答案", session.session_id, step)
                return parsed.final_answer

            if parsed.action is not None:
                # Step 3：调用工具
                observation = self.tools.execute(
                    parsed.action, parsed.action_input, ToolContext(session=session)
                )
                logger.info("session=%s 第 %d 步调用工具 %s -> %s",
                            session.session_id, step, parsed.action, observation[:80])
                # 超长结果先截断再入 history：工具返回几十 KB 文本时不能把 context 撑爆
                observation = self.context.truncate_observation(observation)
                # Step 4 的"继续"分支：把结果回灌，进入下一轮循环
                session.add_observation(observation)
                continue

            # 输出不合规：纠正一次；若已纠正过，把原文当答案兜底返回，
            # 避免用户干等——不合规的输出往往内容本身是可读的
            if not nudged:
                nudged = True
                logger.warning("session=%s 第 %d 步输出不合规，注入纠正提示",
                               session.session_id, step)
                session.add_input(_FORMAT_NUDGE)
                continue
            logger.warning("session=%s 纠正后仍不合规，按原文兜底返回", session.session_id)
            return raw.strip() or "（模型返回了空内容）"

        # 步数耗尽：明确告知用户，而不是无限循环
        logger.warning("session=%s 达到最大推理步数 %d", session.session_id, self.max_steps)
        return f"已达到最大推理步数（{self.max_steps} 步），任务可能过于复杂，请拆分后重试。"

    @staticmethod
    def _trace(session: Session, step: int, raw: str,
               parsed_action: str | None, parsed_final: bool) -> None:
        """每一步留痕：时间、步号、模型原文、解析结论。CLI /trace 可读。"""
        session.trace.append({
            "ts": time.strftime("%H:%M:%S"),
            "step": step,
            "llm_output": raw,
            "action": parsed_action,
            "is_final": parsed_final,
        })
