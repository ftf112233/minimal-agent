"""Session 管理。

对应题目里的场景：用户 A 开窗口 1 查天气记待办、开窗口 2 写周报记待办，
两个窗口是独立 session，可以随时切回去继续聊，互不影响。

落到代码上：
- 每个 Session 有独立的 history（对话历史）、state（工具状态，如 todos）、
  summary（压缩后的长期记忆）、trace（执行轨迹）；
- SessionStore 用 dict + 一把锁管理所有 session，CLI 单进程够用；
- 可选 JSON 文件持久化，进程重启后能"接着上次的窗口继续聊"。

消息结构说明：history 里每条是 dict(role, content, kind)。
- kind=input       真实用户输入，role=user
- kind=observation 工具结果，role=user，content 带 "Observation: " 前缀
  （OpenAI 兼容协议里 role=tool 必须搭配 tool_call_id，ReAct 文本模式下
   用不了，所以业界最小实现普遍把 Observation 伪装成 user 消息）
- kind=assistant   模型原始输出（Thought/Action/... 原文），role=assistant
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Session:
    session_id: str
    history: list[dict] = field(default_factory=list)
    summary: str = ""                # 压缩后的早期对话（长期记忆），注入 system prompt
    state: dict = field(default_factory=dict)   # 工具侧状态，如 todos
    trace: list[dict] = field(default_factory=list)  # 每步执行轨迹
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # ---- 历史追加：三个入口语义分明，context 统计轮次时只认 input ----
    def add_input(self, text: str) -> None:
        self.history.append({"role": "user", "content": text, "kind": "input"})
        self.updated_at = time.time()

    def add_observation(self, text: str) -> None:
        self.history.append({
            "role": "user",
            "content": f"Observation: {text}",
            "kind": "observation",
        })
        self.updated_at = time.time()

    def add_assistant(self, raw: str) -> None:
        self.history.append({"role": "assistant", "content": raw, "kind": "assistant"})
        self.updated_at = time.time()

    @property
    def user_rounds(self) -> int:
        """用户轮次 = kind=input 的消息数。Observation 不算一轮。"""
        return sum(1 for m in self.history if m.get("kind") == "input")


class SessionStore:
    """会话仓库：内存 dict 为主存储，可选 JSON 文件持久化。

    并发处理：一把 threading.RLock 保护所有读写。
    单机 CLI 场景下这是正确且够用的方案；多实例部署时要换成
    Redis/DB 做外部存储（那属于架构题第 1 题的范畴，这里不过度设计）。
    """

    def __init__(self, persist_dir: str | Path | None = None) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._dir = Path(persist_dir) if persist_dir else None
        if self._dir:
            self._dir.mkdir(parents=True, exist_ok=True)

    def get_or_create(self, session_id: str) -> Session:
        if not session_id or not session_id.strip():
            raise ValueError("session_id 不能为空")
        session_id = session_id.strip()
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = self._load(session_id) or Session(session_id=session_id)
                self._sessions[session_id] = session
            return session

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions or (
                self._dir is not None and self._file(session_id).exists()
            )

    def list_ids(self) -> list[str]:
        with self._lock:
            ids = set(self._sessions)
            if self._dir:
                ids.update(p.stem for p in self._dir.glob("*.json"))
            return sorted(ids)

    def save(self, session: Session) -> None:
        if not self._dir:
            return  # 纯内存模式：进程内隔离仍然成立，只是重启丢失
        with self._lock:
            data = {
                "session_id": session.session_id,
                "history": session.history,
                "summary": session.summary,
                "state": session.state,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
            }
            # 先写临时文件再替换，避免写一半进程崩溃留下半个 JSON
            tmp = self._file(session.session_id).with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self._file(session.session_id))

    # ---- 内部 ----
    def _file(self, session_id: str) -> Path:
        assert self._dir is not None
        return self._dir / f"{session_id}.json"

    def _load(self, session_id: str) -> Session | None:
        if not self._dir or not self._file(session_id).exists():
            return None
        try:
            data = json.loads(self._file(session_id).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("会话文件损坏，按新会话处理: %s (%s)", session_id, e)
            return None
        return Session(
            session_id=data["session_id"],
            history=data.get("history", []),
            summary=data.get("summary", ""),
            state=data.get("state", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
        )
