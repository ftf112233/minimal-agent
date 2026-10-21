"""工具注册机制。

每个工具向 Registry 注册四样东西：名称、描述、参数 JSON Schema、处理函数。
- 名称+描述+Schema 会被渲染进 system prompt，LLM 基于它们自主决策调用哪个工具；
- 处理函数是真正的执行体，签名统一为 handler(arguments: dict, ctx: ToolContext) -> str。

设计取舍：参数校验只做"required 字段存在 + 顶层是 dict"的轻量检查，
不引入 jsonschema 库——面试场景下少一个依赖、校验逻辑一眼能讲完。
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # 避免循环导入：Session 只在类型注解里用
    from agent.session import Session

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """工具内部主动抛出的业务错误（如参数非法）。

    与意外异常区分开：ToolError 的 message 会直接作为 Observation 反馈给模型，
    让它修正参数重试；意外异常则记日志后给模型一个通用错误提示。
    """


@dataclass
class ToolContext:
    """工具执行时的上下文。

    目前只带 session：像 todo 这类工具需要把数据挂在当前会话上，
    天然实现"不同窗口的待办互不干扰"。
    """

    session: "Session | None" = None


ToolHandler = Callable[[dict, ToolContext], str]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema（type=object，含 properties/required）
    handler: ToolHandler
    timeout: float | None = None  # 单工具超时覆盖值；None 表示用注册表默认值


# 默认超时（秒）。环境变量可覆盖，方便测试里调小、线上调大。
DEFAULT_TOOL_TIMEOUT = float(os.environ.get("TOOL_TIMEOUT", "10.0"))


@dataclass
class ToolRegistry:
    """工具登记表。用 dict 保持 O(1) 查找；注册顺序即 prompt 中的展示顺序。"""

    _tools: dict[str, Tool] = field(default_factory=dict)
    default_timeout: float = DEFAULT_TOOL_TIMEOUT
    # 工具在线程池里执行，才能在 Windows 上也支持超时中断（signal.alarm 仅 Unix 可用）
    _executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")
    )

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas_prompt(self) -> str:
        """把所有工具的 Schema 渲染成给 LLM 看的文本块。"""
        import json

        blocks = []
        for tool in self._tools.values():
            schema = json.dumps(tool.parameters, ensure_ascii=False)
            blocks.append(f"- {tool.name}: {tool.description}\n  参数 Schema: {schema}")
        return "\n".join(blocks)

    def execute(self, name: str, arguments: dict | None, ctx: ToolContext) -> str:
        """执行工具并把结果（或错误）转成字符串 Observation。

        约定：永不向调用方抛异常。错误也是信息——以 "Error: ..." 形式喂回给
        模型，让它自己修正，这比直接让 loop 崩溃更符合 Agent 的工作方式。
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: 未知工具 '{name}'。可用工具：{', '.join(self.names())}"

        args = arguments or {}
        if not isinstance(args, dict):
            return f"Error: 工具 '{name}' 的参数必须是 JSON 对象，收到: {type(args).__name__}"

        # 轻量校验：检查 Schema 里声明的 required 字段
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            return f"Error: 工具 '{name}' 缺少必填参数: {', '.join(missing)}"

        timeout = tool.timeout if tool.timeout is not None else self.default_timeout

        # 超时控制：把 handler 丢进线程池，主线程只等 timeout 秒。
        # 为什么用线程池而不是 signal.alarm：signal 模块在 Windows 上没有 SIGALRM，
        # 而本项目要跨平台可跑。代价见下面的注释——超时后无法真正杀死线程。
        future = self._executor.submit(tool.handler, args, ctx)
        try:
            result = future.result(timeout=timeout)
        except FuturesTimeoutError:
            # 已知取舍：Python 线程无法被强制杀掉，卡住的 handler 会在后台继续跑
            # （线程池有上限 8，极端情况下并发打满会排队）。真正的强隔离要换子进程，
            # 对笔试题属于过度设计——这里明确了边界并写进注释。
            logger.warning("工具 %s 执行超时（>%.1fs）", name, timeout)
            return (f"Error: 工具 '{name}' 执行超时（>{timeout:.1f}s），已放弃本次调用。"
                    "请改用其他工具，或简化参数后重试。")
        except ToolError as e:
            # 业务错误：原样反馈给模型，信息最有价值
            return f"Error: {e}"
        except Exception as e:  # 意外 bug：记全量日志，给模型脱敏后的提示
            logger.exception("工具 %s 执行异常", name)
            return f"Error: 工具 '{name}' 执行失败: {type(e).__name__}: {e}"

        logger.info("工具 %s 执行成功，参数=%s", name, args)
        return str(result)
