from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import string
import threading
from typing import Any
import uuid

from monitor.schemas import DistressEvent, MonitorState

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class SessionStore:
    def __init__(self, sessions_path: Path, subscriptions_path: Path) -> None:
        self._sessions_path = sessions_path
        self._subscriptions_path = subscriptions_path
        self._lock = threading.Lock()

        self._sessions: dict[str, dict[str, Any]] = {}
        self._pairing_code_index: dict[str, str] = {}
        self._by_session: dict[str, list[int]] = {}
        self._by_chat: dict[str, str] = {}
        self._events_by_session: dict[str, deque[dict[str, Any]]] = {}
        self._active_session_id: str | None = None

        self._load()

    def create_or_resume_session(self, session_id: str | None = None) -> dict[str, str]:
        with self._lock:
            sid = (session_id or "").strip()
            now_iso = _utc_now_iso()

            if sid and sid in self._sessions:
                self._sessions[sid]["lastSeen"] = now_iso
            else:
                sid = str(uuid.uuid4())
                pairing_code = self._generate_pairing_code_locked()
                self._sessions[sid] = {
                    "pairingCode": pairing_code,
                    "createdAt": now_iso,
                    "lastSeen": now_iso,
                    "latestVitals": None,
                    "latestRisk": None,
                    "latestState": None,
                }
                self._pairing_code_index[pairing_code] = sid
                self._events_by_session[sid] = deque(maxlen=80)
                self._save_sessions_locked()

            self._active_session_id = sid
            self._save_sessions_locked()
            return {
                "sessionId": sid,
                "pairingCode": str(self._sessions[sid]["pairingCode"]),
            }

    def set_active_session(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._active_session_id = session_id
                self._sessions[session_id]["lastSeen"] = _utc_now_iso()
                self._save_sessions_locked()

    def active_session_id(self) -> str | None:
        with self._lock:
            return self._active_session_id

    def apply_state_update(
        self,
        session_id: str,
        state: MonitorState,
        event: DistressEvent | None,
    ) -> None:
        with self._lock:
            if session_id not in self._sessions:
                return
            self._sessions[session_id]["lastSeen"] = _utc_now_iso()
            self._sessions[session_id]["latestVitals"] = {
                "bpm": state.bpm,
                "hr_confidence": state.hr_confidence,
                "fps": state.fps,
                "timestamp": state.timestamp,
            }
            self._sessions[session_id]["latestRisk"] = {
                "status": state.status,
                "distress_score": state.distress_score,
                "fall_risk": state.fall_risk,
                "faint_risk": state.faint_risk,
                "posture": state.posture,
                "timestamp": state.timestamp,
            }
            self._sessions[session_id]["latestState"] = state.to_dict()
            if event is not None:
                if session_id not in self._events_by_session:
                    self._events_by_session[session_id] = deque(maxlen=80)
                event_payload = event.to_dict()
                event_payload["sessionId"] = session_id
                self._events_by_session[session_id].appendleft(event_payload)

    def get_session_status(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            events = list(self._events_by_session.get(session_id, deque()))[:30]
            return {
                "sessionId": session_id,
                "pairingCode": session.get("pairingCode"),
                "createdAt": session.get("createdAt"),
                "lastSeen": session.get("lastSeen"),
                "latestVitals": session.get("latestVitals"),
                "latestRisk": session.get("latestRisk"),
                "latestState": session.get("latestState"),
                "events": events,
            }

    def get_session_by_code(self, pairing_code: str) -> str | None:
        with self._lock:
            return self._pairing_code_index.get(pairing_code)

    def pair_chat_to_session(self, chat_id: int, session_id: str) -> bool:
        with self._lock:
            if session_id not in self._sessions:
                return False

            chat_key = str(int(chat_id))
            prev_session = self._by_chat.get(chat_key)
            if prev_session == session_id:
                return True

            if prev_session is not None and prev_session in self._by_session:
                self._by_session[prev_session] = [
                    cid for cid in self._by_session[prev_session] if cid != int(chat_id)
                ]

            self._by_chat[chat_key] = session_id
            bucket = self._by_session.setdefault(session_id, [])
            if int(chat_id) not in bucket:
                bucket.append(int(chat_id))
            self._save_subscriptions_locked()
            return True

    def unpair_chat(self, chat_id: int) -> bool:
        with self._lock:
            chat_key = str(int(chat_id))
            session_id = self._by_chat.pop(chat_key, None)
            if session_id is None:
                return False
            if session_id in self._by_session:
                self._by_session[session_id] = [
                    cid for cid in self._by_session[session_id] if cid != int(chat_id)
                ]
            self._save_subscriptions_locked()
            return True

    def get_chat_session(self, chat_id: int) -> str | None:
        with self._lock:
            return self._by_chat.get(str(int(chat_id)))

    def get_chat_ids_for_session(self, session_id: str) -> list[int]:
        with self._lock:
            return list(self._by_session.get(session_id, []))

    def get_pairing_code(self, session_id: str) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            pairing_code = session.get("pairingCode")
            return str(pairing_code) if pairing_code is not None else None

    def _generate_pairing_code_locked(self) -> str:
        for _ in range(1000):
            code = "".join(random.choice(string.digits) for _ in range(6))
            if code not in self._pairing_code_index:
                return code
        raise RuntimeError("Could not generate unique pairing code")

    def _load(self) -> None:
        if self._sessions_path.exists():
            try:
                raw = json.loads(self._sessions_path.read_text(encoding="utf-8"))
                self._sessions = dict(raw.get("sessions", {}))
                self._pairing_code_index = {
                    str(k): str(v) for k, v in dict(raw.get("pairingCodeIndex", {})).items()
                }
                self._active_session_id = raw.get("activeSessionId")
                for sid in self._sessions:
                    self._events_by_session[sid] = deque(maxlen=80)
            except Exception:
                self._sessions = {}
                self._pairing_code_index = {}
                self._active_session_id = None
                logger.exception("Failed loading monitor sessions from %s", self._sessions_path)

        if self._subscriptions_path.exists():
            try:
                raw = json.loads(self._subscriptions_path.read_text(encoding="utf-8"))
                by_session = dict(raw.get("bySession", {}))
                self._by_session = {
                    str(session_id): [int(cid) for cid in chat_ids]
                    for session_id, chat_ids in by_session.items()
                }
                self._by_chat = {str(k): str(v) for k, v in dict(raw.get("byChat", {})).items()}
            except Exception:
                self._by_session = {}
                self._by_chat = {}
                logger.exception("Failed loading Telegram subscriptions from %s", self._subscriptions_path)

    def _save_sessions_locked(self) -> None:
        payload = {
            "sessions": self._sessions,
            "pairingCodeIndex": self._pairing_code_index,
            "activeSessionId": self._active_session_id,
        }
        try:
            self._sessions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed saving monitor sessions to %s", self._sessions_path)

    def _save_subscriptions_locked(self) -> None:
        payload = {
            "bySession": self._by_session,
            "byChat": self._by_chat,
        }
        try:
            self._subscriptions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed saving Telegram subscriptions to %s", self._subscriptions_path)
