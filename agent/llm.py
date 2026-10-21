"""LLM 客户端封装。

设计要点：
- Runtime 只依赖 LLMClient 协议（一个 chat 方法），不关心背后是哪家模型。
  这样测试时可以换成 MockLLM，提交时换真实 DeepSeek API，主流程不变。
- DeepSeek 官方 API 是 OpenAI 兼容协议，所以直接用官方 openai SDK 即可，
  这不是 Agent 框架，只是 HTTP 调用的封装，符合"从零实现"的要求。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Agent Runtime 依赖的最小接口：给一组消息，返回一段文本。

    用 Protocol 而不是抽象基类：任何有 chat(messages) -> str 方法的对象
    都能直接用（鸭子类型），测试里的 MockLLM 不需要继承任何东西。
    """

    def chat(self, messages: list[dict]) -> str:
        ...


class DeepSeekClient:
    """DeepSeek 真实 API 客户端（OpenAI 兼容协议）。

    只负责"把 messages 发出去、把文本拿回来"，不做任何 Agent 逻辑。
    内置有限次指数退避重试，应对网络抖动 / 限流这类瞬时故障。
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        # 延迟导入：跑测试（MockLLM）时不强制要求安装 openai SDK
        from openai import OpenAI

        # 优先级：显式参数 > 环境变量 > 官方默认值
        self.model = model or os.environ.get("MODEL_NAME", "deepseek-chat")
        self.max_retries = max_retries
        # temperature 调低：ReAct 格式输出需要稳定，而不是发散
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=timeout,
        )

    def chat(self, messages: list[dict]) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                content = resp.choices[0].message.content
                # 模型偶尔返回空 content，统一按空串处理，交给上层解析兜底
                return content or ""
            except Exception as e:  # 网络错误/限流/5xx 统一重试，参数错误重试也无害
                last_err = e
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s 指数退避
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次）：%s；%ds 后重试",
                    attempt, self.max_retries, e, wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
        # 重试耗尽后抛出，由 Runtime 捕获并给用户一个可读的错误回复
        raise RuntimeError(f"LLM 调用连续失败 {self.max_retries} 次") from last_err


class MockLLM:
    """测试与离线演示用的假 LLM，不发任何网络请求。

    两种用法：
    1. script：预设一组回复，按调用顺序依次弹出（用于测试精确编排 loop 走向）；
    2. rule：传一个函数，根据收到的 messages 动态生成回复（用于 CLI --mock 演示）。
    calls 记录了每次收到的 messages，方便测试断言"第 N 步给模型看了什么"。
    """

    def __init__(
        self,
        script: list[str] | None = None,
        rule: Callable[[list[dict]], str] | None = None,
    ) -> None:
        self._script = list(script or [])
        self._rule = rule
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        if self._rule is not None:
            return self._rule(messages)
        if not self._script:
            raise RuntimeError("MockLLM 脚本已耗尽：被调用的次数超过预设回复数")
        return self._script.pop(0)
