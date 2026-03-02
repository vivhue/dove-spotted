from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _read_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _load_local_dotenv() -> None:
    dotenv_path = Path(".env")
    if not dotenv_path.exists():
        return
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)
    except Exception:
        # Keep settings load resilient even if .env is malformed.
        pass


@dataclass(frozen=True)
class Settings:
    camera_index: int
    simulate_mode: bool
    faint_high_sensitivity: bool
    warning_low_bpm: float
    warning_high_bpm: float
    critical_low_bpm: float
    critical_high_bpm: float
    alert_cooldown_sec: int
    emit_interval_sec: float
    sms_recipients: tuple[str, ...]
    twilio_account_sid: str | None
    twilio_auth_token: str | None
    twilio_from_number: str | None
    telegram_bot_token: str | None
    telegram_alert_cooldown_sec: int
    app_public_url: str | None


def load_settings() -> Settings:
    # Load local .env for development; existing OS env vars still take precedence.
    _load_local_dotenv()
    recipients_raw = os.getenv("SMS_RECIPIENTS", "")
    recipients = tuple(number.strip() for number in recipients_raw.split(",") if number.strip())

    return Settings(
        camera_index=_read_int("CAMERA_INDEX", 0),
        simulate_mode=_read_bool("SIMULATE_MODE", False),
        faint_high_sensitivity=_read_bool("FAINT_HIGH_SENSITIVITY", False),
        warning_low_bpm=_read_float("WARNING_LOW_BPM", 50.0),
        warning_high_bpm=_read_float("WARNING_HIGH_BPM", 120.0),
        critical_low_bpm=_read_float("CRITICAL_LOW_BPM", 45.0),
        critical_high_bpm=_read_float("CRITICAL_HIGH_BPM", 130.0),
        alert_cooldown_sec=_read_int("ALERT_COOLDOWN_SEC", 120),
        emit_interval_sec=_read_float("EMIT_INTERVAL_SEC", 0.5),
        sms_recipients=recipients,
        twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        twilio_from_number=os.getenv("TWILIO_FROM_NUMBER"),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN"),
        telegram_alert_cooldown_sec=_read_int("TELEGRAM_ALERT_COOLDOWN_SEC", 120),
        app_public_url=os.getenv("APP_PUBLIC_URL"),
    )
