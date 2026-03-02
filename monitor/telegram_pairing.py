from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from monitor.config import Settings
from monitor.session_store import SessionStore

logger = logging.getLogger(__name__)

RED_EVENT_TYPES = {"fall", "cardiac_distress"}
FAINT_REPEAT_INTERVAL_SEC = 10.0
POLL_LOCK_FILE = ".telegram_poll.lock"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class AlertPayload:
    session_id: str
    pairing_code: str | None
    alert_type: str
    reason: str
    hr: float | None
    confidence: float | None
    timestamp: str
    caregiver_url: str


class TelegramPairingBot:
    def __init__(self, settings: Settings, store: SessionStore) -> None:
        self._settings = settings
        self._store = store
        self._token = (settings.telegram_bot_token or "").strip()
        self._enabled = bool(self._token)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset: int | None = None
        self._lock = threading.Lock()
        self._session_red_active: dict[str, bool] = {}
        self._session_red_context: dict[str, tuple[str, str]] = {}
        self._session_faint_active: dict[str, bool] = {}
        self._last_sent_by_pair: dict[str, float] = {}
        self._last_faint_sent_by_pair: dict[str, float] = {}
        self._blocked_until_ts = 0.0
        self._last_conflict_log_ts = 0.0
        self._owns_poll_lock = False
        self._poll_lock_path = Path(__file__).resolve().parent.parent / POLL_LOCK_FILE

        if self._enabled:
            logger.info("Telegram pairing bot enabled (polling)")
        else:
            logger.info("Telegram pairing bot disabled (missing TELEGRAM_BOT_TOKEN)")

    def start(self) -> None:
        if not self._enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        if not self._acquire_poll_lock():
            logger.warning(
                "Telegram polling not started: another local instance already holds %s",
                self._poll_lock_path,
            )
            return
        # Polling cannot run while webhook mode is active.
        self._delete_webhook()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="telegram-polling", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._release_poll_lock()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "polling": bool(self._thread and self._thread.is_alive()),
            "poll_lock": self._owns_poll_lock,
        }

    def handle_state_update(
        self,
        session_id: str,
        status: str,
        event_type: str | None,
        reason: str | None,
        hr: float | None,
        confidence: float | None,
        timestamp: str,
        faint_type: str | None,
        faint_risk: float | None,
    ) -> None:
        if not self._enabled:
            return

        with self._lock:
            if event_type in RED_EVENT_TYPES:
                self._session_red_context[session_id] = (event_type.upper(), reason or "Detected distress condition.")

            red_context = self._session_red_context.get(session_id)
            is_red = status == "critical" and red_context is not None
            was_red = self._session_red_active.get(session_id, False)
            transition_to_red = (not was_red) and is_red
            self._session_red_active[session_id] = is_red

            chat_ids = self._store.get_chat_ids_for_session(session_id)
            if not chat_ids:
                return

            pairing_code = self._store.get_pairing_code(session_id)
            caregiver_url = self._build_caregiver_url(session_id)
            now_ts = time.time()

            if is_red and red_context is not None:
                alert_type, alert_reason = red_context
                for chat_id in chat_ids:
                    pair_key = f"{session_id}:{chat_id}"
                    last_sent = self._last_sent_by_pair.get(pair_key)
                    cooldown_ok = (
                        last_sent is None
                        or now_ts - last_sent >= self._settings.telegram_alert_cooldown_sec
                    )
                    if not transition_to_red and not cooldown_ok:
                        continue
                    payload = AlertPayload(
                        session_id=session_id,
                        pairing_code=pairing_code,
                        alert_type=alert_type,
                        reason=alert_reason,
                        hr=hr,
                        confidence=confidence,
                        timestamp=timestamp,
                        caregiver_url=caregiver_url,
                    )
                    sent = self._send_alert(chat_id, payload)
                    if sent:
                        self._last_sent_by_pair[pair_key] = now_ts
            elif status != "critical":
                self._session_red_context.pop(session_id, None)

            # While fainting remains detected, keep sending repeating alerts.
            faint_kind_from_event: str | None = None
            if event_type and event_type.startswith("faint_"):
                faint_kind_from_event = event_type.removeprefix("faint_")

            faint_kind = (faint_type or faint_kind_from_event or "").strip()
            faint_detected = bool(faint_kind)
            self._session_faint_active[session_id] = faint_detected
            if not faint_detected:
                return

            risk_value = 0.0 if faint_risk is None else float(faint_risk)
            faint_alert_type = f"FAINT_{faint_kind.upper()}"
            faint_reason = (
                f"Fainting still detected ({faint_kind}, risk={risk_value:.2f})."
                if faint_kind
                else f"Fainting still detected (risk={risk_value:.2f})."
            )
            for chat_id in chat_ids:
                pair_key = f"{session_id}:{chat_id}"
                last_faint_sent = self._last_faint_sent_by_pair.get(pair_key)
                if last_faint_sent is not None and now_ts - last_faint_sent < FAINT_REPEAT_INTERVAL_SEC:
                    continue
                payload = AlertPayload(
                    session_id=session_id,
                    pairing_code=pairing_code,
                    alert_type=faint_alert_type,
                    reason=faint_reason,
                    hr=hr,
                    confidence=confidence,
                    timestamp=timestamp,
                    caregiver_url=caregiver_url,
                )
                sent = self._send_alert(chat_id, payload)
                if sent:
                    self._last_faint_sent_by_pair[pair_key] = now_ts

    def send_test_alert(self, session_id: str) -> tuple[bool, str]:
        if not self._enabled:
            return False, "TELEGRAM_BOT_TOKEN is not set"
        status = self._store.get_session_status(session_id)
        if status is None:
            return False, "Session not found"

        chat_ids = self._store.get_chat_ids_for_session(session_id)
        if not chat_ids:
            return (
                False,
                "No caregiver paired to this session. In Telegram: /pair <6-digit-code>.",
            )

        payload = AlertPayload(
            session_id=session_id,
            pairing_code=self._store.get_pairing_code(session_id),
            alert_type="CARDIAC_DISTRESS",
            reason="Manual test alert from monitor API.",
            hr=32.0,
            confidence=0.99,
            timestamp=_utc_now_iso(),
            caregiver_url=self._build_caregiver_url(session_id),
        )

        any_sent = False
        now_ts = time.time()
        for chat_id in chat_ids:
            if self._send_alert(chat_id, payload):
                self._last_sent_by_pair[f"{session_id}:{chat_id}"] = now_ts
                any_sent = True
        if not any_sent:
            return False, "Telegram send failed"
        return True, "sent"

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                for update in self._get_updates():
                    self._handle_update(update)
            except urllib_error.HTTPError as exc:
                details = exc.read().decode("utf-8", errors="replace")
                if exc.code == 409:
                    self._handle_polling_conflict(details)
                    time.sleep(2.0)
                    continue
                logger.error("Telegram polling HTTP error %s: %s", exc.code, details)
                time.sleep(2.0)
            except Exception:
                logger.exception("Telegram polling loop error")
                time.sleep(2.0)

    def _get_updates(self) -> list[dict[str, Any]]:
        params = {"timeout": 25}
        if self._offset is not None:
            params["offset"] = self._offset
        url = (
            f"https://api.telegram.org/bot{self._token}/getUpdates?"
            + urllib_parse.urlencode(params)
        )
        req = urllib_request.Request(url=url, method="GET")
        with urllib_request.urlopen(req, timeout=30) as res:
            raw = json.loads(res.read().decode("utf-8", errors="replace"))
        if not raw.get("ok"):
            return []
        updates = list(raw.get("result", []))
        if updates:
            self._offset = int(updates[-1].get("update_id", 0)) + 1
        return updates

    def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message") or {}
        text = str(message.get("text", "")).strip()
        chat = message.get("chat") or {}
        chat_id_raw = chat.get("id")
        if chat_id_raw is None:
            return
        try:
            chat_id = int(chat_id_raw)
        except (TypeError, ValueError):
            return

        if not text:
            return

        lower = text.lower()
        if lower.startswith("/start"):
            self._send_text(
                chat_id,
                "Welcome. Send /pair <6-digit-code> to link to an elderly monitor session.",
            )
            return
        if lower.startswith("/pair"):
            parts = text.split()
            if len(parts) < 2:
                self._send_text(chat_id, "Usage: /pair <6-digit-code>")
                return
            code = parts[1].strip()
            session_id = self._store.get_session_by_code(code)
            if session_id is None:
                self._send_text(chat_id, "Invalid pairing code. Please check and try again.")
                return
            ok = self._store.pair_chat_to_session(chat_id, session_id)
            if not ok:
                self._send_text(chat_id, "Pairing failed. Please try again.")
                return
            self._send_text(
                chat_id,
                f"Paired successfully. You will receive alerts for session {session_id}. "
                f"Dashboard: {self._build_caregiver_url(session_id)}",
            )
            return
        if lower.startswith("/status"):
            session_id = self._store.get_chat_session(chat_id)
            if session_id is None:
                self._send_text(chat_id, "No active pairing. Send /pair <6-digit-code>.")
                return
            self._send_text(
                chat_id,
                f"Paired session: {session_id}\nDashboard: {self._build_caregiver_url(session_id)}",
            )
            return
        if lower.startswith("/unpair"):
            ok = self._store.unpair_chat(chat_id)
            if ok:
                self._send_text(chat_id, "Unpaired successfully.")
            else:
                self._send_text(chat_id, "No pairing found for this chat.")
            return

    def _build_caregiver_url(self, session_id: str) -> str:
        public_base = (self._settings.app_public_url or "").rstrip("/")
        relative = f"/caregiver/{session_id}"
        if not public_base:
            return relative
        return f"{public_base}{relative}"

    def _send_alert(self, chat_id: int, payload: AlertPayload) -> bool:
        hr_text = "unknown" if payload.hr is None else f"{payload.hr:.1f} bpm"
        conf_text = "--" if payload.confidence is None else f"{payload.confidence:.2f}"
        lines = [
            "\U0001F6A8 RED ALERT",
            f"Session: {payload.session_id}",
            f"Pairing Code: {payload.pairing_code or '--'}",
            f"Type: {payload.alert_type}",
            f"Reason: {payload.reason}",
            f"HR: {hr_text}",
            f"Confidence: {conf_text}",
            f"Time: {payload.timestamp}",
            f"Link: {payload.caregiver_url}",
        ]
        return self._send_text(chat_id, "\n".join(lines))

    def _send_text(self, chat_id: int, text: str) -> bool:
        if time.time() < self._blocked_until_ts:
            return False
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        body = json.dumps(
            {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
        ).encode("utf-8")
        req = urllib_request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=8) as res:
                return 200 <= res.status < 300
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                try:
                    payload = json.loads(details)
                    retry_after = float(
                        (payload.get("parameters") or {}).get("retry_after", 1.0)
                    )
                except Exception:
                    retry_after = 1.0
                self._blocked_until_ts = max(
                    self._blocked_until_ts,
                    time.time() + max(1.0, retry_after),
                )
            logger.error("Telegram API HTTP error %s: %s", exc.code, details)
            return False
        except Exception:
            logger.exception("Telegram API sendMessage failed")
            return False

    def _handle_polling_conflict(self, details: str) -> None:
        now = time.time()
        if now - self._last_conflict_log_ts >= 300.0:
            self._last_conflict_log_ts = now
            logger.warning(
                "Telegram polling conflict (409). Another bot instance with this token is polling. Details: %s",
                details,
            )

        details_lower = details.lower()
        if "webhook" in details_lower:
            self._delete_webhook()
        # Reset offset after conflict so polling can recover cleanly.
        self._offset = None

    def _delete_webhook(self) -> bool:
        url = f"https://api.telegram.org/bot{self._token}/deleteWebhook"
        body = json.dumps({"drop_pending_updates": False}).encode("utf-8")
        req = urllib_request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=10) as res:
                raw = json.loads(res.read().decode("utf-8", errors="replace"))
            ok = bool(raw.get("ok"))
            if ok:
                logger.info("Telegram webhook cleared for polling mode")
            return ok
        except urllib_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            logger.warning("Telegram deleteWebhook HTTP error %s: %s", exc.code, details)
            return False
        except Exception:
            logger.exception("Telegram deleteWebhook failed")
            return False

    def _acquire_poll_lock(self) -> bool:
        # Ensure only one local process polls Telegram for this project.
        for _ in range(2):
            try:
                fd = os.open(str(self._poll_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(str(os.getpid()))
                self._owns_poll_lock = True
                return True
            except FileExistsError:
                lock_pid = self._read_lock_pid()
                if lock_pid is None or not self._pid_alive(lock_pid):
                    try:
                        self._poll_lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    except Exception:
                        logger.exception("Failed to remove stale Telegram poll lock")
                        return False
                    continue
                return False
            except Exception:
                logger.exception("Failed to acquire Telegram poll lock")
                return False
        return False

    def _release_poll_lock(self) -> None:
        if not self._owns_poll_lock:
            return
        try:
            lock_pid = self._read_lock_pid()
            if lock_pid is None or lock_pid == os.getpid():
                self._poll_lock_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to release Telegram poll lock")
        finally:
            self._owns_poll_lock = False

    def _read_lock_pid(self) -> int | None:
        try:
            raw = self._poll_lock_path.read_text(encoding="utf-8").strip()
            if not raw:
                return None
            return int(raw)
        except Exception:
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
