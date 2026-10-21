"""工具层测试：计算正确性与安全性、注册表错误兜底、todo 的会话隔离、超时控制。"""
import time
import unittest

from agent.session import Session
from agent.tools import Tool, ToolContext, ToolRegistry, build_default_registry


def _reg():
    return build_default_registry()


class TestCalculator(unittest.TestCase):

    def test_basic(self):
        reg = _reg()
        self.assertEqual(
            reg.execute("calculator", {"expression": "1+2*3"}, ToolContext()),
            "1+2*3 = 7",
        )
        self.assertIn("3.0", reg.execute("calculator", {"expression": "(10-4)/2"}, ToolContext()))

    def test_rejects_injection(self):
        """白名单求值：变量、函数调用、dunder 都必须被拒绝，而不是被 eval。"""
        reg = _reg()
        for evil in ("__import__('os')", "open('x')", "x+1", "1; rm -rf /"):
            result = reg.execute("calculator", {"expression": evil}, ToolContext())
            self.assertTrue(result.startswith("Error:"), evil)

    def test_division_by_zero_returns_error_not_crash(self):
        result = _reg().execute("calculator", {"expression": "1/0"}, ToolContext())
        self.assertTrue(result.startswith("Error:"))

    def test_missing_param(self):
        result = _reg().execute("calculator", {}, ToolContext())
        self.assertIn("缺少必填参数", result)


class TestRegistryFallback(unittest.TestCase):

    def test_unknown_tool(self):
        result = _reg().execute("nonexist", {}, ToolContext())
        self.assertIn("未知工具", result)
        self.assertIn("calculator", result)  # 提示里带可用工具清单

    def test_non_dict_arguments(self):
        result = _reg().execute("calculator", ["not", "a", "dict"], ToolContext())
        self.assertTrue(result.startswith("Error:"))


class TestMockTools(unittest.TestCase):

    def test_search_mock_hit_and_miss(self):
        reg = _reg()
        self.assertIn("ReAct", reg.execute("search", {"query": "什么是 react"}, ToolContext()))
        self.assertIn("未收录", reg.execute("search", {"query": "随机无命中词"}, ToolContext()))

    def test_weather_mock(self):
        reg = _reg()
        self.assertIn("广州", reg.execute("weather", {"city": "广州"}, ToolContext()))
        self.assertIn("暂无", reg.execute("weather", {"city": "拉萨"}, ToolContext()))


class TestTodo(unittest.TestCase):
    """todo 是有状态工具：数据挂在 Session.state 上，按会话隔离。"""

    def test_add_list_done_flow(self):
        reg = _reg()
        ctx = ToolContext(session=Session(session_id="s1"))
        self.assertIn("#1", reg.execute("todo", {"action": "add", "content": "写周报"}, ctx))
        self.assertIn("#2", reg.execute("todo", {"action": "add", "content": "查天气"}, ctx))
        listing = reg.execute("todo", {"action": "list"}, ctx)
        self.assertIn("写周报", listing)
        self.assertIn("查天气", listing)
        self.assertIn("已完成待办 #1", reg.execute("todo", {"action": "done", "id": 1}, ctx))
        self.assertIn("[x]", reg.execute("todo", {"action": "list"}, ctx))

    def test_isolated_between_sessions(self):
        """窗口 1 的待办不能出现在窗口 2，这是 session 隔离的核心语义。"""
        reg = _reg()
        s1, s2 = Session(session_id="s1"), Session(session_id="s2")
        reg.execute("todo", {"action": "add", "content": "窗口1的事"}, ToolContext(session=s1))
        listing_s2 = reg.execute("todo", {"action": "list"}, ToolContext(session=s2))
        self.assertNotIn("窗口1的事", listing_s2)
        self.assertIn("没有待办", listing_s2)

    def test_bad_action_and_missing_id(self):
        reg = _reg()
        ctx = ToolContext(session=Session(session_id="s1"))
        self.assertTrue(reg.execute("todo", {"action": "bad"}, ctx).startswith("Error:"))
        self.assertTrue(reg.execute("todo", {"action": "done", "id": 99}, ctx).startswith("Error:"))


class TestToolTimeout(unittest.TestCase):
    """超时控制：慢工具不能拖死整个 loop，超时转成 Error 回灌给模型。"""

    @staticmethod
    def _slow_registry(timeout: float, sleep_seconds: float = 1.0) -> ToolRegistry:
        reg = ToolRegistry(default_timeout=timeout)
        reg.register(Tool(
            name="slow",
            description="慢工具（测试用）",
            parameters={"type": "object", "properties": {}},
            handler=lambda args, ctx: (time.sleep(sleep_seconds), "睡醒了")[1],
        ))
        return reg

    def test_timeout_returns_error_observation(self):
        reg = self._slow_registry(timeout=0.1, sleep_seconds=2.0)
        result = reg.execute("slow", {}, ToolContext())
        self.assertIn("超时", result)
        self.assertTrue(result.startswith("Error:"))

    def test_fast_tool_not_affected(self):
        """正常执行的工具不受超时机制影响。"""
        reg = self._slow_registry(timeout=5.0, sleep_seconds=0.01)
        self.assertEqual(reg.execute("slow", {}, ToolContext()), "睡醒了")

    def test_per_tool_timeout_overrides_default(self):
        """单个工具可以用自己的 timeout 覆盖注册表默认值。"""
        reg = ToolRegistry(default_timeout=5.0)
        reg.register(Tool(
            name="urgent",
            description="要求快速返回的工具",
            parameters={"type": "object", "properties": {}},
            handler=lambda args, ctx: (time.sleep(2.0), "完成")[1],
            timeout=0.1,  # 覆盖默认的 5s
        ))
        self.assertIn("超时", reg.execute("urgent", {}, ToolContext()))


if __name__ == "__main__":
    unittest.main()
