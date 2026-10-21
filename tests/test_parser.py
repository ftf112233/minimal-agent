"""解析器测试：覆盖标准格式与真实模型的常见"不守规矩"输出。"""
import unittest

from agent.parser import parse_llm_output


class TestParser(unittest.TestCase):

    def test_parse_action_standard(self):
        raw = ('Thought: 需要计算 3+5\n'
               'Action: calculator\n'
               'Action Input: {"expression": "3+5"}')
        p = parse_llm_output(raw)
        self.assertTrue(p.parse_ok)
        self.assertEqual(p.action, "calculator")
        self.assertEqual(p.action_input, {"expression": "3+5"})
        self.assertEqual(p.thought, "需要计算 3+5")
        self.assertIsNone(p.final_answer)

    def test_parse_final_answer(self):
        p = parse_llm_output("Thought: 已知答案\nFinal Answer: 3+5 等于 8")
        self.assertTrue(p.parse_ok)
        self.assertEqual(p.final_answer, "3+5 等于 8")
        self.assertIsNone(p.action)

    def test_parse_final_answer_multiline(self):
        p = parse_llm_output("Thought: 整理一下\nFinal Answer: 第一行\n第二行\n\n第三行")
        self.assertEqual(p.final_answer, "第一行\n第二行\n\n第三行")

    def test_parse_action_input_in_code_fence(self):
        """模型常把 JSON 包进 markdown 代码块，要能剥壳。"""
        raw = ('Thought: 查天气\nAction: weather\n'
               'Action Input: ```json\n{"city": "广州"}\n```')
        p = parse_llm_output(raw)
        self.assertEqual(p.action, "weather")
        self.assertEqual(p.action_input, {"city": "广州"})

    def test_parse_action_input_with_trailing_garbage(self):
        """JSON 后面跟了多余文本时，raw_decode 只取第一个完整对象。"""
        raw = 'Action: calculator\nAction Input: {"expression": "1+1"} 以上是参数'
        p = parse_llm_output(raw)
        self.assertEqual(p.action_input, {"expression": "1+1"})

    def test_parse_fullwidth_colon_and_case(self):
        """全角冒号、大小写混用都要容忍。"""
        raw = 'THOUGHT：想想\nACTION: calculator\nACTION INPUT: {"expression": "2*3"}'
        p = parse_llm_output(raw)
        self.assertTrue(p.parse_ok)
        self.assertEqual(p.action, "calculator")

    def test_parse_action_without_input(self):
        """有 Action 但参数解析失败：parse_ok=True，action_input=None，
        交给 Runtime 以空参数调用，工具层会返回缺少必填参数的错误。"""
        p = parse_llm_output("Thought: 调用\nAction: calculator\nAction Input: 这不是JSON")
        self.assertTrue(p.parse_ok)
        self.assertEqual(p.action, "calculator")
        self.assertIsNone(p.action_input)

    def test_parse_garbage_not_ok(self):
        p = parse_llm_output("我今天心情很好，不想按格式说话")
        self.assertFalse(p.parse_ok)
        self.assertIsNone(p.action)
        self.assertIsNone(p.final_answer)

    def test_parse_empty(self):
        self.assertFalse(parse_llm_output("").parse_ok)
        self.assertFalse(parse_llm_output("   \n ").parse_ok)


if __name__ == "__main__":
    unittest.main()
