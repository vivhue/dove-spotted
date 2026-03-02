from __future__ import annotations

import time

from monitor.config import Settings
from monitor.schemas import DistressEvent, HRMeasurement, MonitorState, PoseMeasurement


class SignalFusion:
    """Combines heart-rate and posture signals into one risk score and events."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._abnormal_hr_streak = 0
        self._person_present_streak = 0
        self._faint_confirm_streak = 0
        self._faint_candidate_start_ts: float | None = None
        self._last_event_by_type: dict[str, float] = {}
        self._hr_event_streak_threshold = 6
        self._person_confirm_streak_threshold = 1 if settings.faint_high_sensitivity else 2
        self._faint_confirm_streak_threshold = 2 if settings.faint_high_sensitivity else 5
        self._faint_confirm_min_sec = 0.45 if settings.faint_high_sensitivity else 0.90
        self._faint_synth_threshold = 0.36 if settings.faint_high_sensitivity else 0.65

    def update(
        self,
        timestamp: float,
        source: str,
        hr: HRMeasurement,
        pose: PoseMeasurement,
        fps: float,
    ) -> tuple[MonitorState, DistressEvent | None]:
        cardiac_risk = self._cardiac_risk(hr)
        if cardiac_risk >= 0.8:
            self._abnormal_hr_streak += 1
        else:
            self._abnormal_hr_streak = max(0, self._abnormal_hr_streak - 1)

        fall_risk = pose.fall_risk
        faint_risk = pose.faint_risk

        if pose.person_present:
            self._person_present_streak += 1
        else:
            self._person_present_streak = 0
        person_confirmed = self._person_present_streak >= self._person_confirm_streak_threshold

        if not person_confirmed:
            self._faint_confirm_streak = 0
            self._faint_candidate_start_ts = None

        if pose.faint_detected and pose.faint_risk >= (0.25 if self.settings.faint_high_sensitivity else 0.45):
            self._faint_confirm_streak += 1
            if self._faint_candidate_start_ts is None:
                self._faint_candidate_start_ts = timestamp
        else:
            self._faint_confirm_streak = max(0, self._faint_confirm_streak - 1)
            if self._faint_confirm_streak == 0:
                self._faint_candidate_start_ts = None

        confirm_age_sec = (
            0.0
            if self._faint_candidate_start_ts is None
            else max(0.0, timestamp - self._faint_candidate_start_ts)
        )

        seated_candidate = (
            pose.faint_type == "seated_slump"
            or (pose.posture == "sitting" and pose.faint_risk >= (0.50 if self.settings.faint_high_sensitivity else 0.58))
        )
        confirm_streak_threshold = self._faint_confirm_streak_threshold
        confirm_min_sec = self._faint_confirm_min_sec
        if seated_candidate:
            confirm_streak_threshold = max(2, confirm_streak_threshold - 2)
            confirm_min_sec = max(0.40, confirm_min_sec - 0.40)

        faint_confirmed = (
            self._faint_confirm_streak >= confirm_streak_threshold
            and confirm_age_sec >= confirm_min_sec
        )

        effective_faint_risk = faint_risk if (person_confirmed and faint_confirmed) else min(faint_risk, 0.44)
        distress_score = max(cardiac_risk, fall_risk, effective_faint_risk)

        status = "normal"
        if distress_score >= 0.75:
            status = "critical"
        elif distress_score >= 0.45:
            status = "warning"

        if pose.fall_detected or (pose.faint_detected and person_confirmed and faint_confirmed):
            status = "critical"

        state = MonitorState(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp)),
            source=source,
            bpm=hr.bpm,
            hr_confidence=hr.confidence,
            posture=pose.posture,
            torso_angle=pose.torso_angle,
            person_present=pose.person_present,
            person_source=pose.person_source,
            fall_risk=round(fall_risk, 3),
            faint_risk=round(faint_risk, 3),
            faint_type=pose.faint_type,
            seated_slump_score=round(pose.seated_slump_score, 3),
            distress_score=round(distress_score, 3),
            status=status,
            fps=round(fps, 1),
        )

        event: DistressEvent | None = None
        # High-sensitivity fallback: if person is present and faint risk is high,
        # synthesize a faint type even when upstream classifier is uncertain.
        synth_threshold = self._faint_synth_threshold
        if pose.posture == "sitting":
            synth_threshold = max(0.30, synth_threshold - 0.06)

        if person_confirmed and pose.faint_type is None and pose.faint_risk >= synth_threshold:
            if pose.fall_risk >= 0.40 or pose.posture == "lying":
                pose.faint_type = "ground_fall"
            elif pose.posture == "sitting":
                pose.faint_type = "seated_slump"
            else:
                pose.faint_type = "standing_faint"
            pose.faint_detected = True

        if pose.faint_detected and pose.faint_type and person_confirmed and faint_confirmed:
            faint_event_type = f"faint_{pose.faint_type}"
            if self._event_allowed(faint_event_type, timestamp):
                message = (
                    f"Possible faint detected ({pose.faint_type}, posture={pose.posture}, "
                    f"faint_risk={pose.faint_risk:.2f}, fall_risk={pose.fall_risk:.2f})."
                )
                event = DistressEvent.create(
                    event_type=faint_event_type,
                    severity="critical",
                    message=message,
                    bpm=hr.bpm,
                )
                self._last_event_by_type[faint_event_type] = timestamp
        elif pose.fall_detected and self._event_allowed("fall", timestamp):
            message = f"Possible fall detected (posture={pose.posture}, risk={pose.fall_risk:.2f})."
            event = DistressEvent.create(
                event_type="fall",
                severity="critical",
                message=message,
                bpm=hr.bpm,
            )
            self._last_event_by_type["fall"] = timestamp
        elif (
            cardiac_risk >= 0.8
            and self._abnormal_hr_streak >= self._hr_event_streak_threshold
            and self._event_allowed("cardiac_distress", timestamp)
        ):
            bpm_text = f"{hr.bpm:.1f}" if hr.bpm is not None else "unknown"
            message = f"Possible cardiac distress (BPM={bpm_text}, confidence={hr.confidence:.2f})."
            event = DistressEvent.create(
                event_type="cardiac_distress",
                severity="critical",
                message=message,
                bpm=hr.bpm,
            )
            self._last_event_by_type["cardiac_distress"] = timestamp
            self._abnormal_hr_streak = 0

        return state, event

    def _cardiac_risk(self, hr: HRMeasurement) -> float:
        if hr.bpm is None or hr.confidence < 0.35:
            return 0.0

        if hr.bpm <= self.settings.critical_low_bpm or hr.bpm >= self.settings.critical_high_bpm:
            return 0.9
        if hr.bpm <= self.settings.warning_low_bpm or hr.bpm >= self.settings.warning_high_bpm:
            return 0.6
        return 0.0

    def _event_allowed(self, event_type: str, now_ts: float) -> bool:
        last_ts = self._last_event_by_type.get(event_type)
        if last_ts is None:
            return True
        return now_ts - last_ts >= self.settings.alert_cooldown_sec
