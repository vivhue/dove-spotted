from __future__ import annotations

from collections import deque
import math

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency behavior
    cv2 = None

from monitor.schemas import PoseMeasurement

try:
    import mediapipe as mp
except Exception:  # pragma: no cover - import-time optional dependency behavior
    mp = None


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


class PoseEstimator:
    """Pose-based posture and fall risk estimator."""

    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError("mediapipe is required for pose estimation")
        if cv2 is None:
            raise RuntimeError("opencv-python is required for pose estimation")

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._prev_hip_y: float | None = None
        self._prev_shoulder_y: float | None = None
        self._prev_ts: float | None = None
        self._history: deque[tuple[float, str, float, float, float]] = deque()

    def estimate(self, frame_bgr, timestamp: float) -> PoseMeasurement:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return PoseMeasurement(
                posture="unknown",
                fall_risk=0.0,
                fall_detected=False,
                torso_angle=None,
                person_box=None,
                person_present=False,
                person_source=None,
            )

        lm = result.pose_landmarks.landmark

        left_shoulder = (lm[11].x, lm[11].y)
        right_shoulder = (lm[12].x, lm[12].y)
        left_hip = (lm[23].x, lm[23].y)
        right_hip = (lm[24].x, lm[24].y)

        shoulder_mid = _midpoint(left_shoulder, right_shoulder)
        hip_mid = _midpoint(left_hip, right_hip)
        torso_dx = shoulder_mid[0] - hip_mid[0]
        torso_dy = shoulder_mid[1] - hip_mid[1]

        # Angle relative to vertical axis; larger means more horizontal.
        torso_angle = abs(math.degrees(math.atan2(torso_dx, -torso_dy + 1e-6)))
        hip_y = hip_mid[1]
        shoulder_y = shoulder_mid[1]

        dt = 1e-6
        hip_velocity = 0.0
        shoulder_velocity = 0.0
        if self._prev_hip_y is not None and self._prev_ts is not None:
            dt = max(timestamp - self._prev_ts, 1e-6)
            hip_velocity = (hip_y - self._prev_hip_y) / dt
            if self._prev_shoulder_y is not None:
                shoulder_velocity = (shoulder_y - self._prev_shoulder_y) / dt

        self._prev_hip_y = hip_y
        self._prev_shoulder_y = shoulder_y
        self._prev_ts = timestamp

        if torso_angle >= 65.0:
            posture = "lying"
        elif torso_angle < 35.0 and hip_y < 0.72:
            posture = "standing"
        elif torso_angle < 45.0 and hip_y >= 0.72:
            posture = "sitting"
        else:
            posture = "transitioning"

        fall_risk = 0.0
        if torso_angle > 55.0:
            fall_risk += 0.45
        if hip_y > 0.80:
            fall_risk += 0.25
        if hip_velocity > 0.80:
            fall_risk += 0.35
        if posture == "lying":
            fall_risk += 0.10
        fall_risk = min(1.0, fall_risk)

        fall_detected = (posture == "lying" and fall_risk >= 0.75) or fall_risk > 0.9

        h_img, w_img = frame_bgr.shape[:2]
        xs = [p.x for p in lm if 0.0 <= p.x <= 1.0]
        ys = [p.y for p in lm if 0.0 <= p.y <= 1.0]
        person_box = None
        person_present = False
        person_source = None
        if xs and ys:
            x1 = max(0, int(min(xs) * w_img) - 12)
            y1 = max(0, int(min(ys) * h_img) - 12)
            x2 = min(w_img - 1, int(max(xs) * w_img) + 12)
            y2 = min(h_img - 1, int(max(ys) * h_img) + 12)
            if x2 > x1 and y2 > y1:
                person_box = (x1, y1, x2, y2)
                person_present = True
                person_source = "pose"

        faint_risk, faint_detected, faint_type, seated_slump_score = self._classify_faint(
            timestamp=timestamp,
            posture=posture,
            hip_y=hip_y,
            shoulder_y=shoulder_y,
            torso_angle=torso_angle,
            hip_velocity=hip_velocity,
            shoulder_velocity=shoulder_velocity,
            fall_risk=fall_risk,
            fall_detected=fall_detected,
        )

        self._append_history(timestamp, posture, hip_y, shoulder_y, torso_angle)

        return PoseMeasurement(
            posture=posture,
            fall_risk=round(fall_risk, 3),
            fall_detected=fall_detected,
            torso_angle=round(torso_angle, 1),
            person_box=person_box,
            person_present=person_present,
            person_source=person_source,
            slanting=(torso_angle >= 40.0 and posture != "lying"),
            faint_risk=round(faint_risk, 3),
            faint_detected=faint_detected,
            faint_type=faint_type,
            seated_slump_score=round(seated_slump_score, 3),
        )

    def close(self) -> None:
        self._pose.close()

    def _append_history(
        self,
        timestamp: float,
        posture: str,
        hip_y: float,
        shoulder_y: float,
        torso_angle: float,
    ) -> None:
        self._history.append((timestamp, posture, hip_y, shoulder_y, torso_angle))
        while self._history and timestamp - self._history[0][0] > 4.0:
            self._history.popleft()

    def _classify_faint(
        self,
        timestamp: float,
        posture: str,
        hip_y: float,
        shoulder_y: float,
        torso_angle: float,
        hip_velocity: float,
        shoulder_velocity: float,
        fall_risk: float,
        fall_detected: bool,
    ) -> tuple[float, bool, str | None, float]:
        if not self._history:
            return 0.0, False, None, 0.0

        history = [row for row in self._history if timestamp - row[0] <= 3.0]
        if not history:
            return 0.0, False, None, 0.0

        baseline = history[0]
        base_hip = baseline[2]
        base_shoulder = baseline[3]
        base_angle = baseline[4]

        hip_drop = max(0.0, hip_y - base_hip)
        shoulder_drop = max(0.0, shoulder_y - base_shoulder)
        angle_change = max(0.0, torso_angle - base_angle)

        standing_ratio = self._posture_ratio(history, "standing")
        sitting_ratio = self._posture_ratio(history, "sitting")

        standing_score = 0.0
        if standing_ratio >= 0.25:
            standing_score += min(0.32, hip_drop / 0.20 * 0.32)
            standing_score += min(0.22, shoulder_drop / 0.16 * 0.22)
            standing_score += min(0.24, max(0.0, hip_velocity) / 1.15 * 0.24)
            standing_score += min(0.18, angle_change / 32.0 * 0.18)
            if posture in {"transitioning", "lying"}:
                standing_score += 0.10
        standing_score = min(1.0, standing_score)

        ground_score = 0.0
        ground_score += min(0.44, fall_risk / 1.0 * 0.44)
        if posture == "lying":
            ground_score += 0.22
        ground_score += min(0.20, max(0.0, hip_velocity) / 1.20 * 0.20)
        ground_score += min(0.20, hip_drop / 0.24 * 0.20)
        ground_score = min(1.0, ground_score)

        slump_score = 0.0
        if sitting_ratio >= 0.30:
            slump_score += min(0.34, shoulder_drop / 0.14 * 0.34)
            slump_score += min(0.22, max(0.0, shoulder_velocity) / 1.05 * 0.22)
            slump_score += min(0.24, angle_change / 30.0 * 0.24)
            if posture in {"sitting", "transitioning"}:
                slump_score += 0.10
            if hip_drop < 0.10:
                slump_score += 0.10
        slump_score = min(1.0, slump_score)

        scores = {
            "standing_faint": standing_score,
            "ground_fall": ground_score,
        }
        faint_risk = max(scores.values())

        detected_type: str | None = None
        if (
            standing_score >= 0.62
            and posture in {"transitioning", "lying"}
            and (hip_velocity > 0.28 or angle_change > 12.0 or hip_drop > 0.10)
        ):
            detected_type = "standing_faint"
        elif ground_score >= 0.62 and (fall_detected or posture == "lying" or hip_drop > 0.12):
            detected_type = "ground_fall"
        return faint_risk, detected_type is not None, detected_type, slump_score

    @staticmethod
    def _posture_ratio(history: list[tuple[float, str, float, float, float]], name: str) -> float:
        if not history:
            return 0.0
        count = sum(1 for row in history if row[1] == name)
        return count / len(history)
