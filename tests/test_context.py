"""Context 管理测试：压缩触发、摘要注入位置、降级截断。"""
import unittest

from agent.context import ContextConfig, ContextManager
from agent.llm import MockLLM
from agent.session import Session


def _session_with_history(n: int) -> Session:
    """造一个含 n 条消息的会话（用户输入与助手回复交替）。"""
    s = Session(session_id="s")
    for i in range(n):
        if i % 2 == 0:
            s.add_input(f"问题 {i}")
        else:
            s.add_assistant(f"回答 {i}")
    return s


class TestContextManager(unittest.TestCase):

    def test_no_compress_below_threshold(self):
        mgr = ContextManager(ContextConfig(compress_threshold=12, recent_window=6))
        s = _session_with_history(10)
        self.assertFalse(mgr.maybe_compress(s, MockLLM(script=["摘要"])))
        self.assertEqual(len(s.history), 10)
        self.assertEqual(s.summary, "")

    def test_compress_when_over_threshold(self):
        """超过阈值：最旧的部分压成摘要，最近窗口保留原文。"""
        mgr = ContextManager(ContextConfig(compress_threshold=12, recent_window=6))
        s = _session_with_history(14)  # 14 > 12，压缩前 8 条，留 6 条
        llm = MockLLM(script=["早期在聊天气和周报。"])
        self.assertTrue(mgr.maybe_compress(s, llm))
        self.assertEqual(len(s.history), 6)
        self.assertEqual(s.summary, "早期在聊天气和周报。")
        self.assertEqual(s.history[0]["content"], "问题 8")  # 最近窗口的原文完好

    def test_summary_injected_into_system_prompt(self):
        """memory 的放置方式：摘要进 system 尾部，而不是混在用户消息里。"""
        mgr = ContextManager()
        s = _session_with_history(2)
        s.summary = "用户叫张三"
        messages = mgr.build_messages(s, "SYS")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("SYS", messages[0]["content"])
        self.assertIn("用户叫张三", messages[0]["content"])
        # 历史原文按 role/content 传递，内部 kind 字段不外泄给 API
        self.assertEqual(set(messages[1]), {"role", "content"})

    def test_compress_fallback_when_llm_fails(self):
        """LLM 摘要失败时降级为文本截断，对话不能崩。"""
        mgr = ContextManager(ContextConfig(compress_threshold=4, recent_window=2))
        s = _session_with_history(6)

        class FailingLLM:
            def chat(self, messages):
                raise RuntimeError("服务不可用")

        self.assertTrue(mgr.maybe_compress(s, FailingLLM()))
        self.assertTrue(s.summary)  # 有兜底摘要
        self.assertEqual(len(s.history), 2)

    def test_truncate_observation_short_text_unchanged(self):
        mgr = ContextManager()
        self.assertEqual(mgr.truncate_observation("短结果"), "短结果")

    def test_truncate_observation_keeps_head_and_tail(self):
        """超长结果截断：保留头尾（结论常在末尾），中间标出省略了多少字符。"""
        mgr = ContextManager(ContextConfig(max_observation_chars=200))
        text = "H" * 500 + "TAIL_MARKER"
        result = mgr.truncate_observation(text)
        # 正文部分不超过 200 字符，外加一行省略标记（长度随省略字数浮动）
        self.assertLess(len(result) - len("...[工具结果过长，已省略  ]..."), 200 + 20)
        self.assertIn("已省略", result)
        self.assertIn("TAIL_MARKER", result)  # 尾部结论保留
        self.assertTrue(result.startswith("HHH"))  # 头部保留

    def test_truncate_disabled_when_limit_zero(self):
        mgr = ContextManager(ContextConfig(max_observation_chars=0))
        self.assertEqual(mgr.truncate_observation("x" * 9999), "x" * 9999)

    def test_compress_merges_old_summary(self):
        """多次压缩时，新摘要要和旧摘要合并而不是覆盖丢失。"""
        mgr = ContextManager(ContextConfig(compress_threshold=4, recent_window=2))
        s = _session_with_history(6)
        s.summary = "旧摘要：用户在广州"
        llm = MockLLM(rule=lambda msgs: "合并摘要：用户在广州，聊了天气"
                      if "用户在广州" in msgs[0]["content"] else "不应该走到这里")
        mgr.maybe_compress(s, llm)
        self.assertIn("用户在广州", s.summary)


if __name__ == "__main__":
    unittest.main()
