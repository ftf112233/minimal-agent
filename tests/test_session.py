"""Session 管理测试：隔离、轮次统计、持久化恢复。"""
import tempfile
import unittest
from pathlib import Path

from agent.session import SessionStore


class TestSessionStore(unittest.TestCase):

    def test_sessions_are_isolated(self):
        """两个窗口各自累计历史，互不可见。"""
        store = SessionStore()
        s1 = store.get_or_create("win1")
        s2 = store.get_or_create("win2")
        s1.add_input("查天气")
        s1.add_assistant("Thought: ...")
        self.assertEqual(len(s1.history), 2)
        self.assertEqual(len(s2.history), 0)
        self.assertIsNot(s1.state, s2.state)  # 工具状态也必须是两份

    def test_same_id_returns_same_session(self):
        """"随时接着窗口 1 继续聊" = 同一 id 拿回同一个对象。"""
        store = SessionStore()
        self.assertIs(store.get_or_create("win1"), store.get_or_create("win1"))

    def test_user_rounds_only_count_input(self):
        store = SessionStore()
        s = store.get_or_create("s")
        s.add_input("问题1")
        s.add_observation("工具结果")   # 工具结果不算用户轮次
        s.add_assistant("回答")
        s.add_input("问题2")
        self.assertEqual(s.user_rounds, 2)

    def test_empty_session_id_rejected(self):
        with self.assertRaises(ValueError):
            SessionStore().get_or_create("  ")

    def test_persistence_roundtrip(self):
        """落盘后重建 Store，历史/摘要/状态都能恢复（接着聊的前提）。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(persist_dir=Path(tmp))
            s = store.get_or_create("win1")
            s.add_input("记个待办")
            s.state["todos"] = [{"id": 1, "content": "写周报", "done": False}]
            s.summary = "早期对话摘要"
            store.save(s)

            store2 = SessionStore(persist_dir=Path(tmp))  # 模拟进程重启
            restored = store2.get_or_create("win1")
            self.assertEqual(restored.history, s.history)
            self.assertEqual(restored.state["todos"][0]["content"], "写周报")
            self.assertEqual(restored.summary, "早期对话摘要")

    def test_corrupted_file_falls_back_to_new_session(self):
        """会话文件损坏时不应崩溃，按新会话处理。"""
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(persist_dir=Path(tmp))
            (Path(tmp) / "bad.json").write_text("{not valid json", encoding="utf-8")
            s = store.get_or_create("bad")
            self.assertEqual(s.history, [])


if __name__ == "__main__":
    unittest.main()
