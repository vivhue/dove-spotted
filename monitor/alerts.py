from __future__ import annotations

import logging
import time

from monitor.config import Settings
from monitor.schemas import DistressEvent, MonitorState

try:
    from twilio.rest import Client
except Exception:  # pragma: no cover - optional import behavior
    Client = None

logger = logging.getLogger(__name__)


class AlertManager:
    """Sends SMS alerts for critical events when Twilio is configured."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._last_sms_by_type: dict[str, float] = {}

        self._enabled = (
            Client is not None
            and bool(settings.sms_recipients)
            and bool(settings.twilio_account_sid)
            and bool(settings.twilio_auth_token)
            and bool(settings.twilio_from_number)
        )
        self._client = None
        if self._enabled:
            self._client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
            logger.info("SMS alerting enabled for %d recipients", len(settings.sms_recipients))
        else:
            logger.info("SMS alerting disabled (missing Twilio config or dependency)")

    def notify(self, event: DistressEvent, state: MonitorState) -> None:
        if event.severity != "critical":
            return
        if not self._enabled or self._client is None:
            return

        now_ts = time.time()
        last_ts = self._last_sms_by_type.get(event.event_type)
        if last_ts is not None and now_ts - last_ts < self._settings.alert_cooldown_sec:
            return

        body = self._message_body(event, state)
        for recipient in self._settings.sms_recipients:
            try:
                self._client.messages.create(
                    from_=self._settings.twilio_from_number,
                    to=recipient,
                    body=body,
                )
            except Exception as exc:  # pragma: no cover - network side effect
                logger.exception("Failed to send SMS to %s: %s", recipient, exc)

        self._last_sms_by_type[event.event_type] = now_ts

    @staticmethod
    def _message_body(event: DistressEvent, state: MonitorState) -> str:
        bpm_text = "unknown" if state.bpm is None else f"{state.bpm:.1f}"
        return (
            f"[DoveSpotted] {event.event_type.upper()} {event.severity.upper()} | "
            f"BPM={bpm_text} posture={state.posture} fall_risk={state.fall_risk:.2f} | "
            f"{event.message}"
        )

