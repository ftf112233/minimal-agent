"""Context 有效管理。

回答题目要求的四件事：
1. 最大轮次限制：max_rounds 是硬上限，由 Runtime 在入口检查；
2. 追问支持：追问本质就是"带着历史继续对话"，历史原样进 context，
   带工具的追问靠 Observation 也留在历史里实现；
3. 哪些信息塞 context：用户输入、模型输出（含 Thought）、工具结果全量保留
   原文（最近窗口内）；被压缩掉的早期内容以摘要形式保留在 system prompt；
4. 基础压缩：history 超过阈值时，把最旧的若干条交给 LLM 压成摘要，
   合并进 session.summary；LLM 摘要失败时降级为纯文本截断。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .session import Session

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    max_rounds: int = 20          # 单 session 最大用户轮次（硬上限）
    recent_window: int = 6        # 压缩后仍原样保留的最近消息条数
    compress_threshold: int = 12  # history 条数超过它才考虑压缩


# 摘要压缩用的提示词：要求保留事实与结论，丢掉寒暄和中间推理
_SUMMARIZE_PROMPT = (
    "请把以下对话历史压缩成一段简洁的摘要（不超过 200 字），"
    "保留：用户的真实需求、已确认的事实、已完成的操作与关键结果；"
    "丢弃：寒暄、重复的中间推理过程。\n\n"
    "【已有摘要】\n{old_summary}\n\n【待压缩的对话】\n{chunk}\n\n"
    "请输出合并后的新摘要，只输出摘要本身："
)


class ContextManager:
    """负责把 Session 组装成发给 LLM 的 messages，并在过长时压缩。"""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()

    # ------------------------------------------------------------ 组装
    def build_messages(self, session: Session, system_prompt: str) -> list[dict]:
        """构造一次 LLM 调用的 messages。

        结构：[system(prompt + 记忆摘要)] + history 原文（role/content 两字段）。
        摘要放在 system 尾部：位置最靠前、最稳定，不会被窗口截断，
        这就是"memory 的放置方式"。
        """
        system = system_prompt
        if session.summary:
            system += f"\n\n## 早期对话摘要（长期记忆）\n{session.summary}"

        messages = [{"role": "system", "content": system}]
        # 发往 API 的消息只保留 role/content，kind 是内部元数据
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in session.history
        )
        return messages

    # ------------------------------------------------------------ 压缩
    def maybe_compress(self, session: Session, llm) -> bool:
        """若 history 过长则压缩最旧部分。返回是否发生了压缩。"""
        cfg = self.config
        if len(session.history) <= cfg.compress_threshold:
            return False
        if len(session.history) <= cfg.recent_window:
            return False  # 全是最近窗口内的消息，没有可压缩的部分

        # 切分：旧的去压缩，最近的保留原文（保证追问指代不失真）
        cut = len(session.history) - cfg.recent_window
        old_chunk = session.history[:cut]
        chunk_text = "\n".join(f"[{m['role']}] {m['content']}" for m in old_chunk)

        try:
            summary = llm.chat([{
                "role": "user",
                "content": _SUMMARIZE_PROMPT.format(
                    old_summary=session.summary or "（无）", chunk=chunk_text,
                ),
            }]).strip()
            if not summary:
                raise ValueError("LLM 返回了空摘要")
        except Exception as e:
            # 降级策略：压缩失败不该让对话崩掉，用纯文本截断兜底
            logger.warning("LLM 摘要失败，降级为文本截断: %s", e)
            summary = self._fallback_summary(session.summary, chunk_text)

        session.summary = summary
        del session.history[:cut]
        logger.info("context 压缩完成：移除 %d 条旧消息，剩余 %d 条", cut, len(session.history))
        return True

    @staticmethod
    def _fallback_summary(old_summary: str, chunk_text: str) -> str:
        """不依赖 LLM 的兜底压缩：拼接旧摘要 + 截断的原文。"""
        merged = (old_summary + "\n" + chunk_text).strip()
        return merged[-2000:]  # 只留尾部 2000 字符，防止无限膨胀
