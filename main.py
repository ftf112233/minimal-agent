
"""CLI 入口：多 session 对话。

用法：
    python main.py                 # 真实 DeepSeek API（需在 .env 配 DEEPSEEK_API_KEY）
    python main.py --session s1    # 指定初始会话
    python main.py --mock          # 离线演示模式（MockLLM，无需 API Key）

会话内命令：
    /switch <id>   切换会话（即"开另一个窗口"，数据隔离）
    /sessions      列出所有会话
    /trace         查看当前会话的执行轨迹
    /state         查看当前会话的工具状态（如待办）
    /new           开启一个新会话（自动生成 id）
    /quit          退出
"""
from __future__ import annotations

import argparse
import itertools
import logging
import os
import sys

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv 不是硬依赖：没有它时用系统环境变量即可
    def load_dotenv() -> bool:
        return False

from agent.context import ContextManager
from agent.llm import DeepSeekClient, MockLLM
from agent.runtime import AgentRuntime
from agent.session import SessionStore
from agent.tools import build_default_registry


def _mock_rule(messages: list[dict]) -> str:
    """离线演示用的规则 LLM：根据最后一条消息决定行为。

    看见 Observation 就给 Final Answer（模拟"根据工具结果收尾"）；
    否则按关键词选择工具。它证明了 loop 的正确性与 LLM 无关。
    """
    last = messages[-1]["content"]
    if last.startswith("Observation:"):
        return f"Thought: 工具已返回结果，可以回答了。\nFinal Answer: 根据工具结果 => {last}"
    if any(k in last for k in ("算", "计算", "+", "*", "/")):
        # 提取一个可算的表达式；提取不到就用默认表达式
        expr = "".join(ch for ch in last if ch.isdigit() or ch in "+-*/().% ") or "1+1"
        return (f'Thought: 用户要计算，应调用 calculator。\nAction: calculator\n'
                f'Action Input: {{"expression": "{expr.strip()}"}}')
    if "天气" in last:
        city = next((c for c in ("广州", "厦门", "北京", "上海", "深圳") if c in last), "广州")
        return (f'Thought: 用户问天气，应调用 weather。\nAction: weather\n'
                f'Action Input: {{"city": "{city}"}}')
    if "待办" in last or "todo" in last.lower():
        return ('Thought: 用户要记待办，应调用 todo。\nAction: todo\n'
                'Action Input: {"action": "add", "content": "' + last[:20] + '"}')
    if "搜索" in last or "查一下" in last:
        return ('Thought: 用户要搜索。\nAction: search\n'
                'Action Input: {"query": "agent"}')
    return f"Thought: 无需工具，直接回复。\nFinal Answer: 你说的是：{last}"


def build_runtime(mock: bool, persist_dir: str | None) -> AgentRuntime:
    tools = build_default_registry()
    store = SessionStore(persist_dir=persist_dir)
    if mock:
        llm = MockLLM(rule=_mock_rule)
    else:
        if not os.environ.get("DEEPSEEK_API_KEY"):
            # 早失败、给清楚指引，比请求打到一半报 401 友好
            print("错误：未找到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入 Key，"
                  "或使用 --mock 离线演示。", file=sys.stderr)
            sys.exit(2)
        llm = DeepSeekClient()
    return AgentRuntime(llm=llm, tools=tools, store=store, context=ContextManager())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="minimal-agent CLI")
    parser.add_argument("--session", default="default", help="初始会话 id")
    parser.add_argument("--mock", action="store_true", help="离线演示模式，不调真实 API")
    parser.add_argument("--no-persist", action="store_true", help="禁用会话落盘（纯内存）")
    args = parser.parse_args()

    load_dotenv()
    runtime = build_runtime(
        mock=args.mock,
        persist_dir=None if args.no_persist else os.path.join("data", "sessions"),
    )

    current = args.session
    counter = itertools.count(1)  # /new 自动生成 session id 用
    print(f"minimal-agent 已启动（{'mock 演示' if args.mock else 'DeepSeek 真实 API'}），"
          f"当前会话: {current}。输入 /help 查看命令。")

    while True:
        try:
            text = input(f"[{current}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd == "/quit":
                break
            elif cmd == "/switch" and arg:
                current = arg.strip()
                print(f"已切换到会话: {current}")
            elif cmd == "/sessions":
                print("会话列表:", ", ".join(runtime.store.list_ids()) or "（空）")
            elif cmd == "/new":
                current = f"s{next(counter)}"
                print(f"已创建并切换到新会话: {current}")
            elif cmd == "/trace":
                session = runtime.store.get_or_create(current)
                for t in session.trace:
                    print(f"  step {t['step']} [{t['ts']}] action={t['action']} "
                          f"final={t['is_final']}\n    {t['llm_output'][:120]}")
                if not session.trace:
                    print("（当前会话还没有执行轨迹）")
            elif cmd == "/state":
                session = runtime.store.get_or_create(current)
                print("会话状态:", session.state or "（空）")
            elif cmd == "/help":
                print(__doc__)
            else:
                print("未知命令，输入 /help 查看。")
            continue

        answer = runtime.handle(current, text)

        print(f"Agent: {answer}")


if __name__ == "__main__":
    main()
