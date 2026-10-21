"""内置工具：calculator / search / weather / todo。

题目要求至少三个：calculator、search（可 mock）、read_docs/todo/weather 任选一。
这里实现四个，其中 search 和 weather 是明确标注的 mock（返回内置数据），
todo 是真实可用的——它把数据挂在 Session.state 上，正好演示
"窗口 1 记天气待办、窗口 2 记周报待办互不干扰"的 session 隔离语义。
"""
from __future__ import annotations

import ast
import operator

from .registry import Tool, ToolContext, ToolError, ToolRegistry

# ---------------------------------------------------------------- calculator

# 白名单运算符：只允许纯算术，从根上杜绝 eval("...") 的代码注入风险
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    """递归求值 AST。任何不在白名单里的节点类型都直接拒绝。"""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError("表达式只能包含数字字面量")
    if isinstance(node, ast.BinOp):
        fn = _BIN_OPS.get(type(node.op))
        if fn is None:
            raise ToolError(f"不支持的运算符: {type(node.op).__name__}")
        return fn(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY_OPS.get(type(node.op))
        if fn is None:
            raise ToolError(f"不支持的一元运算符: {type(node.op).__name__}")
        return fn(_eval_node(node.operand))
    # Name（变量）、Call（函数调用）等全部走到这里被拒绝
    raise ToolError(f"表达式含有不允许的内容: {type(node).__name__}")


def calculator_handler(args: dict, ctx: ToolContext) -> str:
    expr = args.get("expression")
    if not isinstance(expr, str) or not expr.strip():
        raise ToolError("参数 expression 必须是非空字符串")
    expr = expr.strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        raise ToolError(f"表达式语法错误: {expr!r}")
    result = _eval_node(tree)
    return f"{expr} = {result}"


# ------------------------------------------------------------------- search

# mock 语料：key 词命中即返回，演示"工具结果进入 loop"即可，不接真实搜索引擎
_SEARCH_CORPUS = [
    (("react", "推理"), "ReAct（Reasoning + Acting）是 2022 年提出的范式：让模型交替输出"
     "思考（Thought）与行动（Action），根据工具返回的观察（Observation）继续推理，直到给出最终答案。"),
    (("deepseek",), "DeepSeek 是深度求索公司的大模型，官方 API 兼容 OpenAI 协议，"
     " base_url 为 https://api.deepseek.com，常用模型名 deepseek-chat。"),
    (("agent", "智能体"), "Agent（智能体）的核心循环：接收输入 → 决策（回复或调用工具）→"
     " 执行工具 → 依据结果继续循环或输出答案。"),
]


def search_handler(args: dict, ctx: ToolContext) -> str:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ToolError("参数 query 必须是非空字符串")
    query = query.strip().lower()
    for keywords, answer in _SEARCH_CORPUS:
        if any(k in query for k in keywords):
            return f"[mock 搜索结果] {answer}"
    return f"[mock 搜索结果] 未收录与 {query!r} 相关的条目（这是内置 mock，非真实搜索引擎）"


# ------------------------------------------------------------------ weather

_WEATHER_DATA = {
    "广州": "晴，28~35°C，南风 2 级，午后有雷阵雨",
    "厦门": "多云，26~32°C，东北风 3 级",
    "北京": "晴，18~29°C，北风 1 级",
    "上海": "阴，24~30°C，东风 2 级",
    "深圳": "阵雨，27~33°C，南风 3 级",
}


def weather_handler(args: dict, ctx: ToolContext) -> str:
    city = args.get("city")
    if not isinstance(city, str) or not city.strip():
        raise ToolError("参数 city 必须是非空字符串")
    city = city.strip()
    info = _WEATHER_DATA.get(city)
    if info is None:
        return f"[mock 天气] 暂无 {city} 的数据（内置城市：{ '、'.join(_WEATHER_DATA) }）"
    return f"[mock 天气] {city}：{info}"


# --------------------------------------------------------------------- todo
# todo 是"有状态工具"：数据存在 Session.state["todos"]，天然按会话隔离。

def _todos(ctx: ToolContext) -> list[dict]:
    if ctx.session is None:
        raise ToolError("todo 工具需要会话上下文")
    # setdefault：首次使用时初始化空列表，挂在 session.state 上随之持久化
    return ctx.session.state.setdefault("todos", [])


def todo_handler(args: dict, ctx: ToolContext) -> str:
    action = args.get("action")
    items = _todos(ctx)

    if action == "add":
        content = args.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ToolError("action=add 需要提供非空参数 content")
        item = {"id": len(items) + 1, "content": content.strip(), "done": False}
        items.append(item)
        return f"已添加待办 #{item['id']}: {item['content']}"

    if action == "list":
        if not items:
            return "当前会话没有待办事项"
        lines = [
            f"#{i['id']} [{'x' if i['done'] else ' '}] {i['content']}" for i in items
        ]
        return "待办列表：\n" + "\n".join(lines)

    if action == "done":
        todo_id = args.get("id")
        for item in items:
            if item["id"] == todo_id:
                item["done"] = True
                return f"已完成待办 #{todo_id}: {item['content']}"
        raise ToolError(f"找不到 id={todo_id} 的待办")

    if action == "clear":
        items.clear()
        return "已清空当前会话的待办列表"

    raise ToolError("action 必须是 add / list / done / clear 之一")


# ------------------------------------------------------- 注册表（统一出口）

def build_default_registry() -> ToolRegistry:
    """组装默认工具集。新增工具只需在这里加一条 register。"""
    registry = ToolRegistry()
    registry.register(Tool(
        name="calculator",
        description="计算数学表达式，支持 + - * / // % ** 和括号。输入必须是纯算术表达式。",
        parameters={
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "要计算的算术表达式，如 (3+5)*2"},
            },
            "required": ["expression"],
        },
        handler=calculator_handler,
    ))
    registry.register(Tool(
        name="search",
        description="搜索资料（内置 mock 数据，用于演示工具调用流程）。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
        handler=search_handler,
    ))
    registry.register(Tool(
        name="weather",
        description="查询城市天气（内置 mock 数据）。",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 广州"},
            },
            "required": ["city"],
        },
        handler=weather_handler,
    ))
    registry.register(Tool(
        name="todo",
        description="管理当前会话的待办事项：add 添加、list 查看、done 完成、clear 清空。",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "done", "clear"]},
                "content": {"type": "string", "description": "待办内容（action=add 时必填）"},
                "id": {"type": "integer", "description": "待办 id（action=done 时必填）"},
            },
            "required": ["action"],
        },
        handler=todo_handler,
    ))
    return registry
