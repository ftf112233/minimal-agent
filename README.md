# minimal-agent：从零实现一个最小可用 Agent

笔试提交：不依赖任何 Agent 框架（langgraph / openhands 等），核心 Runtime 全部自研。
唯一的第三方运行依赖是 `openai` SDK——它只是 DeepSeek API（OpenAI 兼容协议）的 HTTP 封装，
不参与任何 Agent 逻辑。

## 运行方式

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入真实 DEEPSEEK_API_KEY

python main.py                  # 真实 DeepSeek API
python main.py --session win1   # 指定会话
python main.py --mock           # 离线演示模式：MockLLM 按规则走完整 loop，无需 Key

# 测试（零第三方依赖，基于标准库 unittest）
python -m unittest discover -s tests -v
```

会话内命令：`/switch <id>`（切换窗口）、`/sessions`、`/new`、`/trace`（执行轨迹）、`/state`（工具状态）、`/quit`。

## 真实 LLM API 实测记录（model=deepseek-chat）

以下命令真实跑通（未做任何删改，工具调用与会话隔离均为模型自主决策的结果）：

```
$ python main.py --session win1

[win1] > 帮我算一下 (128+374)*2 等于多少
Agent: (128+374)*2 = 1004                    ← 调用 calculator

[win1] > 广州今天天气怎么样，适合出门吗
Agent: 广州今天天气：晴，午后有雷阵雨，28~35°C，南风 2 级。
       白天适合出门，但气温偏高注意防晒补水，午后记得带伞防雷阵雨。   ← 调用 weather 后基于结果给建议

[win1] > 帮我记个待办：明天带伞
Agent: 已帮你记下待办 #1 明天带伞。
       结合刚才广州午后有雷阵雨的天气，这个提醒很及时～          ← 调用 todo，且复用了上一步的天气结论

/switch win2                                  ← 切换窗口
[win2] > 帮我记个待办：周五前交周报
Agent: 已帮你记下待办 #1：周五前交周报。
[win2] > 我有哪些待办
Agent: #1 [未完成] 周五前交周报               ← 只有周报，没有 win1 的"带伞"（会话隔离）

/switch win1                                  ← 切回 win1 继续聊
[win1] > 我刚才查的那个城市天气怎么样
Agent: 你刚才查的是广州，天气情况如下：晴，午后有雷阵雨...      ← 追问：记住了历史
[win1] > 我有哪些待办
Agent: #1 [未完成] 明天带伞                    ← win1 的待办完好，未被 win2 污染
```

这段记录同时验证了四件事：工具自主决策、工具结果进入推理、追问带上下文、
多窗口会话隔离。

## 系统设计

### 模块划分

```
agent/
├── llm.py       LLMClient 协议 + DeepSeekClient（真实 API，重试退避）+ MockLLM（测试/演示）
├── parser.py    ReAct 输出解析：提取 Thought / Action / Action Input / Final Answer
├── runtime.py   核心四步循环、容错（格式纠正、最大步数）、trace
├── session.py   Session（历史/摘要/工具状态/轨迹）+ SessionStore（隔离、锁、JSON 持久化）
├── context.py   最大轮次、最近窗口、超阈值 LLM 摘要压缩（失败降级截断）
└── tools/       ToolRegistry（名称/描述/JSON Schema 注册）+ calculator/search/weather/todo
```

### 核心数据结构

- **Message**：`{role, content, kind}`。`kind ∈ input / observation / assistant`，
  轮次统计只认 `input`；Observation 以 `role=user` + `Observation:` 前缀发送
  （OpenAI 兼容协议的 `role=tool` 必须搭配 `tool_call_id`，ReAct 文本模式下用不了，
  伪装成 user 消息是业界最小实现的通行做法）。
- **Session**：`history`（原文）、`summary`（压缩后的长期记忆）、`state`（工具数据，如 todos）、`trace`（每步轨迹）。四者隔离 = 窗口隔离。
- **Tool**：`name / description / parameters(JSON Schema) / handler`。
  Schema 渲染进 system prompt，LLM 基于它自主决策；注册表做 required 字段轻量校验。
- **ParsedOutput**：`thought / action / action_input / final_answer / parse_ok`。

### 推理与工具调用流程（Loop）

```
用户输入 ──> history 追加 input
  │
  ▼ 每步：
maybe_compress（超阈值先压缩）
  │──> build_messages（system + 摘要 + 历史原文）
  │──> LLM 调用（失败重试 3 次，指数退避）
  │──> parse：
  │      Final Answer ──> 返回给用户，落盘，结束
  │      Action ──> ToolRegistry.execute ──> Observation 回灌 history ──> 继续循环
  │      不合规 ──> 注入格式纠正提示（一轮最多一次）；仍不合规则原文兜底返回
  │
  └ 超过 max_steps（默认 8）──> 明确告知用户，退出
```

工具错误不中断循环：`"Error: ..."` 作为 Observation 回灌，模型拿到错误信息自行修正参数重试。

### 状态管理与并发

- SessionStore：内存 dict + 一把 `threading.RLock`，单机 CLI 下读写线程安全；
  可选 JSON 落盘（先写 `.tmp` 再原子替换，防半截文件），进程重启后可恢复会话。
- 取舍：没有引入 asyncio/多 worker——CLI 场景同步模型最简单、最好讲；
  多实例部署时应把 SessionStore 换成 Redis/DB（属于架构题范畴，不过度设计）。

### 性能与容错考虑

- LLM 是唯一的慢调用和故障点：超时 60s、重试 3 次、指数退避；连续失败给可读回复不穿透。
- `temperature=0.2`：ReAct 格式输出求稳不求发散。
- 压缩只在 history 超过阈值时触发（每步 O(1) 判断），最近 6 条保留原文保证追问指代不失真。
- calculator 用 AST 白名单求值而非 `eval`：从根上消除代码注入面。
- **工具超时**：handler 放进线程池执行，主线程最多等 `timeout`（默认 10s，可用
  `TOOL_TIMEOUT` 环境变量或单个工具的 `timeout` 字段覆盖），超时转成 `Error:` 回灌给模型，
  让它换工具或简化参数。用线程池而非 `signal.alarm` 是因为后者在 Windows 上不可用。
  已知边界：Python 线程无法强制杀死，超时后卡住的 handler 仍在后台跑完——真正的强隔离
  要用子进程，对本实现属于过度设计（取舍已写进代码注释）。
- **Observation 截断**：单个工具结果超过 `max_observation_chars`（默认 2000）时截断后再入
  history，防止一个返回几十 KB 的工具把 context 撑爆。保留头尾而非只留头部——工具的结论
  常出现在末尾。

## Memory 的召回时机与放置方式（题目点名要求）

**放什么**：两级记忆——① `summary`：早期对话的 LLM 摘要（长期记忆）；
② 最近窗口的对话原文（短期记忆，含用户输入、模型 Thought、工具 Observation）。

**召回时机**：每次 loop 调用 LLM 前，在 `build_messages` 时注入——摘要是"被动全量召回"，
不做向量检索（题目说复杂压缩不用实现，检索式召回同理超纲）。

**放置方式**：摘要拼接在 **system prompt 尾部**（`## 早期对话摘要`）。理由：
位置最靠前、最稳定，不会被滑动窗口截掉；system 角色对模型约束力最强；
不污染 user/assistant 的消息序列，追问指代关系保持原样。

**压缩触发**：`len(history) > compress_threshold(12)` 时，把最旧的 `len-recent_window(6)` 条
交给 LLM 与旧摘要**合并**生成新摘要（不覆盖丢失）；LLM 摘要失败时降级为尾部 2000 字符截断。

**最大轮次**：`user_rounds >= max_rounds(20)` 时拒绝新输入并提示开新会话——硬上限防爆 context。

## 测试

49 个用例（`python -m unittest discover -s tests -v`）：
parser 9 个（标准/代码块/全角冒号/尾随杂质/垃圾输出）、tools 14 个（含注入拒绝、除零、
todo 会话隔离、**工具超时与单工具覆盖**）、session 6 个（隔离、持久化、损坏文件兜底）、
context 8 个（压缩触发、摘要注入、降级、合并、**Observation 截断**）、
runtime 12 个（完整 loop、Observation 回灌断言、格式纠正、最大步数/轮次、LLM 全挂兜底、
**超长结果截断**、端到端会话隔离）。

## 关键假设与不确定点

1. **输出协议选了 ReAct 文本解析而非原生 function calling**：题目明确要求"实现 LLM 输出的
   解析逻辑，提取思考过程、工具调用或最终答案"，原生 tool calling 会把这一考察点跳过。
   代价是对模型格式遵从度有要求，用纠正提示 + 兜底缓解。
2. **语言题目未指定**：选 Python——Agent Runtime 自研最直观、依赖最少。
3. search / weather 是 mock（题目允许），calculator / todo 是真实实现。
4. Session 持久化用 JSON 文件而非 SQLite/Redis：演示"可恢复"足够，少依赖。
