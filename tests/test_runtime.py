"""Runtime 主循环测试：用脚本化 MockLLM 精确编排 loop 走向。

这些测试对应题目的核心考察点：
基本循环四步、工具调用回灌、解析失败纠正、异常兜底、
最大步数保护、最大轮次限制、多 session 隔离。
"""
import unittest

from agent.context import ContextConfig, ContextManager
from agent.llm import MockLLM
from agent.runtime import AgentRuntime
from agent.session import SessionStore
from agent.tools import build_default_registry


def _runtime(script, max_steps=8, max_rounds=20):
    llm = MockLLM(script=script)
    rt = AgentRuntime(
        llm=llm,
        tools=build_default_registry(),
        store=SessionStore(),
        context=ContextManager(ContextConfig(max_rounds=max_rounds)),
        max_steps=max_steps,
    )
    return rt, llm


class TestRuntimeLoop(unittest.TestCase):

    def test_basic_loop_tool_then_final(self):
        """完整走一遍四步循环：输入 -> 决策调工具 -> 执行 -> 依据结果给答案。"""
        rt, llm = _runtime([
            'Thought: 需要算一下\nAction: calculator\nAction Input: {"expression": "3+5"}',
            'Thought: 工具返回 8\nFinal Answer: 3+5 等于 8',
        ])
        answer = rt.handle("s1", "3+5等于几")
        self.assertEqual(answer, "3+5 等于 8")

        session = rt.store.get_or_create("s1")
        # 工具结果必须以 Observation 形式回灌进了历史
        obs = [m for m in session.history if m["kind"] == "observation"]
        self.assertEqual(len(obs), 1)
        self.assertIn("3+5 = 8", obs[0]["content"])
        # 第二次 LLM 调用时必须已经带上 Observation（Step 4 的依据）
        self.assertTrue(any(
            "Observation" in m["content"] for m in llm.calls[1] if m["role"] == "user"
        ))
        # trace 记录了两步
        self.assertEqual(len(session.trace), 2)
        self.assertEqual(session.trace[0]["action"], "calculator")
        self.assertTrue(session.trace[1]["is_final"])

    def test_direct_answer_without_tool(self):
        rt, _ = _runtime(["Thought: 简单问题\nFinal Answer: 你好！"])
        self.assertEqual(rt.handle("s1", "你好"), "你好！")

    def test_tool_error_fed_back_and_recovered(self):
        """工具报错不崩 loop：错误作为 Observation 回灌，模型修正后成功。"""
        rt, _ = _runtime([
            'Thought: 调用\nAction: calculator\nAction Input: {"expression": "1/0"}',
            'Thought: 出错了，换个问法\nFinal Answer: 除数不能为 0，请换个表达式',
        ])
        answer = rt.handle("s1", "算一下 1/0")
        self.assertIn("除数不能为 0", answer)

    def test_unknown_tool_error(self):
        rt, _ = _runtime([
            'Thought: 试试\nAction: fly_to_moon\nAction Input: {}',
            'Thought: 工具不存在\nFinal Answer: 没有那个工具',
        ])
        self.assertEqual(rt.handle("s1", "带我去月球"), "没有那个工具")


class TestRuntimeFaultTolerance(unittest.TestCase):

    def test_malformed_output_nudged_once_then_ok(self):
        """输出不合规 -> 注入纠正提示 -> 模型重来。"""
        rt, llm = _runtime([
            "我就随便说说，不按格式",
            "Thought: 抱歉，重来\nFinal Answer: 正经回答",
        ])
        self.assertEqual(rt.handle("s1", "问题"), "正经回答")
        # 第二次调用的 messages 里应该包含格式纠正提示
        self.assertTrue(any("不符合约定格式" in m["content"] for m in llm.calls[1]))

    def test_malformed_twice_falls_back_to_raw(self):
        """纠正后仍不合规：把原文兜底返回，不让用户干等。"""
        rt, _ = _runtime(["随便输出1", "随便输出2"])
        self.assertEqual(rt.handle("s1", "问题"), "随便输出2")

    def test_max_steps_protection(self):
        """模型一直调工具不收手：达到 max_steps 必须退出并明确告知。"""
        rt, _ = _runtime(
            ['Thought: 再想想\nAction: calculator\nAction Input: {"expression": "1+1"}'] * 10,
            max_steps=3,
        )
        answer = rt.handle("s1", "一直算")
        self.assertIn("最大推理步数", answer)

    def test_max_rounds_limit(self):
        """最大轮次限制：达到后拒绝新输入并提示。"""
        rt, _ = _runtime(
            ["Thought: ok\nFinal Answer: r1", "Thought: ok\nFinal Answer: r2"],
            max_rounds=2,
        )
        self.assertEqual(rt.handle("s1", "第1轮"), "r1")
        self.assertEqual(rt.handle("s1", "第2轮"), "r2")
        self.assertIn("最大轮次", rt.handle("s1", "第3轮"))

    def test_llm_fatal_error_returns_readable_message(self):
        """LLM 连续失败（如欠费/断网）：给可读回复，不抛穿透。"""
        class BrokenLLM:
            def chat(self, messages):
                raise RuntimeError("LLM 调用连续失败 3 次")

        rt = AgentRuntime(llm=BrokenLLM(), tools=build_default_registry(),
                          store=SessionStore())
        answer = rt.handle("s1", "你好")
        self.assertIn("错误", answer)


class TestRuntimeSessionSemantics(unittest.TestCase):

    def test_follow_up_with_context(self):
        """追问：第二轮的 messages 必须包含第一轮的历史。"""
        rt, llm = _runtime([
            "Thought: ok\nFinal Answer: 你住广州",
            "Thought: 上文说了\nFinal Answer: 你之前说你住广州",
        ])
        rt.handle("s1", "我住广州")
        rt.handle("s1", "我住哪？")  # 追问
        second_call = llm.calls[1]
        self.assertTrue(any("我住广州" in m["content"] for m in second_call))

    def test_sessions_isolated_through_runtime(self):
        """端到端验证窗口隔离：s1 的待办和历史不进 s2。"""
        rt, _ = _runtime([
            'Thought: 记\nAction: todo\nAction Input: {"action": "add", "content": "窗口1待办"}',
            'Thought: 好\nFinal Answer: 已记录',
            'Thought: 查\nAction: todo\nAction Input: {"action": "list"}',
            'Thought: 好\nFinal Answer: 列表见上',
        ])
        rt.handle("s1", "记个待办")
        rt.handle("s2", "看我的待办")
        s1 = rt.store.get_or_create("s1")
        s2 = rt.store.get_or_create("s2")
        self.assertEqual(s1.state["todos"][0]["content"], "窗口1待办")
        self.assertEqual(s2.state.get("todos", []), [])


if __name__ == "__main__":
    unittest.main()
