from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class HRMeasurement:
    bpm: float | None
    confidence: float


@dataclass
class PoseMeasurement:
    posture: str
    fall_risk: float
    fall_detected: bool
    torso_angle: float | None
    person_box: tuple[int, int, int, int] | None = None
    person_present: bool = False
    person_source: str | None = None
    faint_risk: float = 0.0
    faint_detected: bool = False
    faint_type: str | None = None
    seated_slump_score: float = 0.0


@dataclass
class MonitorState:
    timestamp: str
    source: str
    bpm: float | None
    hr_confidence: float
    posture: str
    torso_angle: float | None
    person_present: bool
    person_source: str | None
    fall_risk: float
    faint_risk: float
    faint_type: str | None
    seated_slump_score: float
    distress_score: float
    status: str
    fps: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def boot(cls) -> "MonitorState":
        return cls(
            timestamp=utc_now_iso(),
            source="bootstrap",
            bpm=None,
            hr_confidence=0.0,
            posture="unknown",
            torso_angle=None,
            person_present=False,
            person_source=None,
            fall_risk=0.0,
            faint_risk=0.0,
            faint_type=None,
            seated_slump_score=0.0,
            distress_score=0.0,
            status="initializing",
            fps=0.0,
        )


@dataclass
class DistressEvent:
    event_id: str
    timestamp: str
    event_type: str
    severity: str
    message: str
    bpm: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(
        cls,
        event_type: str,
        severity: str,
        message: str,
        bpm: float | None,
    ) -> "DistressEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=utc_now_iso(),
            event_type=event_type,
            severity=severity,
            message=message,
            bpm=bpm,
        )
