"""ReAct 风格 LLM 输出解析。

与模型约定的输出格式（写死在 system prompt 里）只有两种：

形式一，调用工具：
    Thought: <推理过程>
    Action: <工具名>
    Action Input: <JSON 对象>

形式二，最终回复：
    Thought: <推理过程>
    Final Answer: <给用户的话>

解析的核心原则是"宽容"：真实模型不会 100% 守格式，常见偏差包括
JSON 包在 ```json 代码块里、Action Input 跨多行、字段名大小写/冒号全半角不一致。
所以这里不用整串正则一把梭，而是分字段切 + json.JSONDecoder.raw_decode
只取"第一个完整 JSON 对象"，后面的杂质直接忽略。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    """一次模型输出的结构化结果。

    final_answer 非 None 表示模型想结束 loop；
    action 非 None 表示模型想调用工具；
    两者都为空（parse_ok=False）表示输出完全不合规，由 Runtime 决定补救。
    """

    thought: str | None = None
    action: str | None = None
    action_input: dict | None = None
    final_answer: str | None = None
    parse_ok: bool = False


# 允许中英文冒号、大小写不敏感——这些都是真实模型输出里见过的偏差
_THOUGHT_RE = re.compile(
    r"Thought\s*[:：]\s*(.*?)(?=Action\s*[:：]|Final Answer\s*[:：]|$)",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_RE = re.compile(r"Final Answer\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)
_ACTION_RE = re.compile(r"Action\s*[:：]\s*([A-Za-z_][\w\-]*)", re.IGNORECASE)
_ACTION_INPUT_RE = re.compile(r"Action Input\s*[:：]\s*(.*)", re.IGNORECASE | re.DOTALL)
# ```json ... ``` 或 ``` ... ``` 代码块，内部才是 JSON
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json(text: str) -> dict | None:
    """从一段可能带杂质的文本里提取第一个完整 JSON 对象。

    用 json.JSONDecoder().raw_decode 而不是 json.loads(整段)：
    raw_decode 允许 JSON 后面跟着多余文本（模型常干的事），
    我们从每个 '{' 出现的位置尝试解码，第一个成功的即为所求。
    """
    candidate = text.strip()
    # 如果包在 markdown 代码块里，先剥壳
    fence = _FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", candidate):
        try:
            obj, _end = decoder.raw_decode(candidate, match.start())
        except json.JSONDecodeError:
            continue  # 这个 '{' 不是合法 JSON 起点，试下一个
        if isinstance(obj, dict):
            return obj
    return None


def parse_llm_output(raw: str) -> ParsedOutput:
    """把模型原始文本解析成 ParsedOutput。永不抛异常——解析失败也是合法结果。"""
    result = ParsedOutput()
    if not raw or not raw.strip():
        return result  # 空输出：parse_ok=False，走纠正流程

    text = raw.strip()

    thought = _THOUGHT_RE.search(text)
    if thought:
        result.thought = thought.group(1).strip() or None

    # 优先判 Final Answer：一旦模型给了最终答案，即使前面胡言乱语也算完成任务
    final = _FINAL_RE.search(text)
    if final:
        result.final_answer = final.group(1).strip()
        result.parse_ok = True
        return result

    # 再看是不是工具调用
    action = _ACTION_RE.search(text)
    if action:
        result.action = action.group(1).strip()
        input_match = _ACTION_INPUT_RE.search(text)
        if input_match:
            result.action_input = _extract_json(input_match.group(1))
        # Action 存在但参数解析失败时，action_input 留 None：
        # Runtime 会把它当作"参数缺失"，以空参数调用并把错误反馈给模型
        result.parse_ok = True
        return result

    # 既无 Final Answer 也无 Action：不合规输出
    return result
