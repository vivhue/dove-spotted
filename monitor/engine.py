from __future__ import annotations

from collections import deque
import logging
import random
import threading
import time
from typing import Callable

import numpy as np
try:
    import cv2
except Exception:  # pragma: no cover - optional dependency behavior
    cv2 = None

from monitor.alerts import AlertManager
from monitor.body_tracker import BodyTracker
from monitor.config import Settings
from monitor.fusion import SignalFusion
from monitor.pose import PoseEstimator
from monitor.rppg import RPPGEstimator
from monitor.schemas import DistressEvent, HRMeasurement, MonitorState, PoseMeasurement

logger = logging.getLogger(__name__)

UpdateCallback = Callable[[MonitorState, DistressEvent | None], None]
FrameCallback = Callable[[bytes], None]


class MonitoringEngine:
    def __init__(
        self,
        settings: Settings,
        on_update: UpdateCallback,
        on_preview: FrameCallback | None = None,
    ) -> None:
        self._settings = settings
        self._on_update = on_update
        self._on_preview = on_preview
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._fusion = SignalFusion(settings)
        self._alerts = AlertManager(settings)
        self._rppg = RPPGEstimator()

        self._pose: PoseEstimator | None = None
        self._body_tracker: BodyTracker | None = BodyTracker() if cv2 is not None else None
        self._frame_index = 0
        self._last_face: tuple[int, int, int, int] | None = None
        self._last_body_center_y: float | None = None
        self._last_body_height: float | None = None
        self._last_body_ts: float | None = None
        self._box_drop_streak = 0
        self._body_motion_history: deque[
            tuple[float, float, float, float, float, float, float]
        ] = deque()
        self._head_rel_history: deque[tuple[float, float, float, float]] = deque()
        self._fainted_banner_until_ts = 0.0
        self._last_emit_ts = 0.0
        self._last_preview_ts = 0.0
        self._preview_interval_sec = 0.15
        self._active_camera_index: int | None = None
        if cv2 is not None:
            self._face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            self._face_detector = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="monitoring-engine", daemon=True)
        self._thread.start()
        logger.info("Monitoring engine started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        logger.info("Monitoring engine stopped")

    def _run(self) -> None:
        try:
            self._run_with_sources()
        except Exception as exc:
            logger.exception("Monitoring loop crashed. Falling back to simulation mode.")
            self._run_simulation(reason=self._format_engine_error_reason(exc))

    def _run_with_sources(self) -> None:
        if self._settings.simulate_mode:
            logger.warning("Running in simulation mode")
            self._run_simulation(reason="forced")
            return

        if cv2 is None:
            logger.error("OpenCV (cv2) is not installed. Falling back to simulation mode.")
            self._run_simulation(reason="opencv_missing")
            return

        pose_available = True
        try:
            self._pose = PoseEstimator()
        except RuntimeError as exc:
            pose_available = False
            logger.warning("Pose estimator unavailable (%s). Continuing with heart-rate only.", exc)
            self._pose = None
        except Exception:
            pose_available = False
            logger.exception("Pose estimator failed during initialization. Continuing with heart-rate only.")
            self._pose = None

        camera = self._open_camera()
        if camera is None:
            logger.error("Could not open any webcam. Falling back to simulation mode.")
            self._run_simulation(reason="camera_unavailable")
            return
        cap, selected_index = camera
        self._active_camera_index = selected_index

        prev_ts = time.time()
        source = f"camera:{selected_index}" if pose_available else f"camera:{selected_index}:hr_only"
        first_frame_deadline = time.time() + 8.0
        read_failures = 0

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or getattr(frame, "size", 0) == 0:
                    read_failures += 1
                    if time.time() > first_frame_deadline and read_failures > 80:
                        logger.error("Webcam opened but no frames received. Switching to simulation mode.")
                        self._run_simulation(reason="camera_no_frames")
                        return
                    time.sleep(0.05)
                    continue

                read_failures = 0
                now_ts = time.time()
                dt = max(now_ts - prev_ts, 1e-6)
                prev_ts = now_ts
                fps = 1.0 / dt

                hr = self._estimate_hr(frame, now_ts)
                pose = self._estimate_pose(frame, now_ts)
                pose = self._track_body_box(frame, pose, now_ts)
                state, event = self._fusion.update(now_ts, source, hr, pose, fps)
                self._emit_preview(frame, pose, source, state.status, event)

                if self._should_emit(now_ts, event):
                    self._on_update(state, event)
                if event is not None:
                    self._alerts.notify(event, state)
        finally:
            cap.release()

    def _should_emit(self, timestamp: float, event: DistressEvent | None) -> bool:
        if event is not None:
            self._last_emit_ts = timestamp
            return True
        if timestamp - self._last_emit_ts >= self._settings.emit_interval_sec:
            self._last_emit_ts = timestamp
            return True
        return False

    def _estimate_hr(self, frame_bgr, timestamp: float) -> HRMeasurement:
        if cv2 is None or self._face_detector is None:
            return HRMeasurement(bpm=None, confidence=0.0)
        try:
            self._frame_index += 1
            if self._frame_index % 5 == 0 or self._last_face is None:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = self._face_detector.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(80, 80),
                )
                if len(faces) > 0:
                    self._last_face = max(faces, key=lambda b: b[2] * b[3])
                else:
                    self._last_face = None

            if self._last_face is None:
                return HRMeasurement(bpm=None, confidence=0.0)

            x, y, w, h = self._last_face
            h_img, w_img = frame_bgr.shape[:2]

            fx1 = max(0, x + int(w * 0.20))
            fx2 = min(w_img, x + int(w * 0.80))
            fy1 = max(0, y + int(h * 0.12))
            fy2 = min(h_img, y + int(h * 0.32))
            if fx2 <= fx1 or fy2 <= fy1:
                return HRMeasurement(bpm=None, confidence=0.0)

            roi = frame_bgr[fy1:fy2, fx1:fx2]
            if roi.size == 0:
                return HRMeasurement(bpm=None, confidence=0.0)

            green_mean = float(np.mean(roi[:, :, 1]))
            return self._rppg.update(green_mean, timestamp)
        except Exception:
            logger.exception("Heart-rate estimation failed")
            return HRMeasurement(bpm=None, confidence=0.0)

    def _estimate_pose(self, frame_bgr, timestamp: float) -> PoseMeasurement:
        if self._pose is None:
            return PoseMeasurement(posture="unknown", fall_risk=0.0, fall_detected=False, torso_angle=None)
        try:
            return self._pose.estimate(frame_bgr, timestamp)
        except Exception:
            logger.exception("Pose estimation failed")
            return PoseMeasurement(posture="unknown", fall_risk=0.0, fall_detected=False, torso_angle=None)

    def _track_body_box(self, frame_bgr, pose: PoseMeasurement, timestamp: float) -> PoseMeasurement:
        if self._body_tracker is None:
            return pose
        try:
            result = self._body_tracker.update(frame_bgr, pose.person_box)
            pose.person_box = result.box
            pose.person_present = result.box is not None
            pose.person_source = result.source
            self._augment_faint_from_body_motion(frame_bgr, pose, timestamp)
        except Exception:
            logger.exception("Body tracking failed")
        return pose

    def _augment_faint_from_body_motion(
        self,
        frame_bgr,
        pose: PoseMeasurement,
        timestamp: float,
    ) -> None:
        # Fallback faint estimation when pose landmarks are unavailable (hr-only mode).
        if pose.person_box is None:
            self._box_drop_streak = 0
            self._last_body_center_y = None
            self._last_body_height = None
            self._last_body_ts = None
            self._head_rel_history.clear()
            pose.seated_slump_score = 0.0
            return

        h_img = float(frame_bgr.shape[0])
        w_img = float(frame_bgr.shape[1])
        x1, y1, x2, y2 = pose.person_box
        center_x = ((x1 + x2) * 0.5) / w_img
        center_y = ((y1 + y2) * 0.5) / h_img
        body_h = max(1.0, float(y2 - y1))
        body_w = max(1.0, float(x2 - x1))
        body_h_norm = body_h / h_img
        top_norm = y1 / h_img
        bottom_norm = y2 / h_img

        if self._last_body_center_y is None or self._last_body_ts is None:
            self._last_body_center_y = center_y
            self._last_body_height = body_h
            self._last_body_ts = timestamp
            return

        dt = max(timestamp - self._last_body_ts, 1e-6)
        drop_speed = (center_y - self._last_body_center_y) / dt
        height_change = 0.0
        if self._last_body_height is not None:
            height_change = max(0.0, (self._last_body_height - body_h) / max(self._last_body_height, 1.0))

        if drop_speed > 0.30:
            self._box_drop_streak += 1
        else:
            self._box_drop_streak = max(0, self._box_drop_streak - 1)

        self._last_body_center_y = center_y
        self._last_body_height = body_h
        self._last_body_ts = timestamp
        self._body_motion_history.append(
            (timestamp, center_x, center_y, body_h, body_w, bottom_norm, top_norm)
        )
        while self._body_motion_history and timestamp - self._body_motion_history[0][0] > 4.0:
            self._body_motion_history.popleft()

        # If pose landmarks are available, prefer pose-based faint inference.
        if pose.torso_angle is not None:
            return

        # Infer coarse posture from box shape when pose landmarks are missing.
        aspect = body_h / max(body_w, 1.0)
        if pose.posture == "unknown":
            if aspect < 0.72:
                pose.posture = "lying"
            elif aspect < 1.05:
                pose.posture = "sitting"
            else:
                pose.posture = "standing"

        history = [row for row in self._body_motion_history if timestamp - row[0] <= 3.0]
        if history:
            base_center_x = history[0][1]
            base_center_y = history[0][2]
            base_bottom = history[0][5]
            base_top = history[0][6]
            peak_height = max(row[3] for row in history)
            total_drop = max(0.0, center_y - base_center_y)
            body_compression = max(0.0, (peak_height - body_h) / max(peak_height, 1.0))
            lateral_shift = abs(center_x - base_center_x)
            top_drop = max(0.0, top_norm - base_top)
            bottom_drift = abs(bottom_norm - base_bottom)
        else:
            total_drop = 0.0
            body_compression = 0.0
            lateral_shift = 0.0
            top_drop = 0.0
            bottom_drift = 0.0

        upper_body_framed = (
            body_h_norm <= 0.72
            and bottom_norm >= 0.72
            and center_y >= 0.50
            and aspect >= 0.95
        )
        seated_like = pose.posture == "sitting" or (
            pose.posture == "unknown" and 0.74 <= aspect <= 1.24 and center_y > 0.50
        ) or upper_body_framed
        high_sens = self._settings.faint_high_sensitivity
        seat_anchor = bottom_drift <= (0.20 if high_sens else 0.15)
        if pose.posture in {"unknown", "standing"} and upper_body_framed and seat_anchor:
            pose.posture = "sitting"
        box_slump_score, box_slump_detected = self._seated_upper_body_slump_score(
            history=history,
            lateral_shift=lateral_shift,
            top_drop=top_drop,
            body_compression=body_compression,
            seat_anchor=seat_anchor,
            seated_like=seated_like,
        )
        head_slump_score, head_slump_detected = self._head_slump_score(
            pose_box=pose.person_box,
            timestamp=timestamp,
            seated_like=seated_like,
            seat_anchor=seat_anchor,
            lateral_shift=lateral_shift,
            top_drop=top_drop,
        )

        speed_norm = 0.45 if high_sens else 0.65
        drop_norm = 0.08 if high_sens else 0.12
        compress_norm = 0.24 if high_sens else 0.34
        center_gate = 0.62 if high_sens else 0.70
        height_gate = 0.06 if high_sens else 0.10
        streak_gate = 1 if high_sens else 2
        detect_risk_gate = 0.25 if high_sens else 0.55
        detect_drop_gate = 0.05 if high_sens else 0.10

        risk = 0.0
        risk += min(0.42, max(0.0, drop_speed) / speed_norm * 0.42)
        risk += min(0.24, total_drop / drop_norm * 0.24)
        risk += min(0.22, body_compression / compress_norm * 0.22)
        if center_y > center_gate:
            risk += 0.12
        if height_change > height_gate:
            risk += min(0.12, height_change / 0.20 * 0.12)
        if self._box_drop_streak >= streak_gate:
            risk += 0.12
        combined_slump = 0.0
        if seated_like:
            combined_slump = max(
                box_slump_score,
                head_slump_score,
                min(1.0, box_slump_score * 0.65 + head_slump_score * 0.55),
            )
            # Reduce seated false positives from generic box jitter and prefer slump-specific cues.
            risk = max(min(risk, 0.44), min(1.0, combined_slump))
        pose.seated_slump_score = round(combined_slump, 3)
        risk = min(1.0, risk)

        pose.faint_risk = max(pose.faint_risk, round(risk, 3))
        pose.fall_risk = max(pose.fall_risk, round(risk * 0.9, 3))

        seated_motion_confirmed = seated_like and (
            box_slump_detected
            or head_slump_detected
            or (body_compression > 0.13 and top_drop > 0.045 and seat_anchor)
            or (
                combined_slump >= (0.56 if high_sens else 0.58)
                and seat_anchor
                and (top_drop > 0.04 or lateral_shift > 0.07)
            )
        )

        generic_motion_detected = (
            risk >= detect_risk_gate and (self._box_drop_streak >= streak_gate or total_drop > detect_drop_gate)
        )
        if seated_like and not seated_motion_confirmed:
            generic_motion_detected = False

        if generic_motion_detected:
            pose.faint_detected = True
            if pose.faint_type is None:
                if pose.posture == "lying" or (center_y > 0.70 and body_compression > 0.18):
                    pose.faint_type = "ground_fall"
                elif seated_motion_confirmed:
                    pose.faint_type = "seated_slump"
                else:
                    pose.faint_type = "standing_faint"

        if seated_like and (head_slump_detected or box_slump_detected):
            pose.faint_detected = True
            pose.faint_type = "seated_slump"
            pose.faint_risk = max(
                pose.faint_risk,
                round(max(head_slump_score, box_slump_score), 3),
            )
            # Seated slump is upper-body dominant, so keep fall risk lower.
            pose.fall_risk = min(pose.fall_risk, 0.55)

    def _seated_upper_body_slump_score(
        self,
        history: list[tuple[float, float, float, float, float, float, float]],
        lateral_shift: float,
        top_drop: float,
        body_compression: float,
        seat_anchor: bool,
        seated_like: bool,
    ) -> tuple[float, bool]:
        if not seated_like or len(history) < 3:
            return 0.0, False

        window = history[-6:]
        t0 = window[0][0]
        t1 = window[-1][0]
        dt = max(t1 - t0, 1e-6)

        cx0 = window[0][1]
        cy0 = window[0][2]
        cx1 = window[-1][1]
        cy1 = window[-1][2]
        motion_speed = float(np.sqrt((cx1 - cx0) ** 2 + (cy1 - cy0) ** 2)) / dt

        top_span = max(row[6] for row in window) - min(row[6] for row in window)
        side_span = max(row[1] for row in window) - min(row[1] for row in window)

        score = 0.0
        score += min(0.38, top_drop / 0.08 * 0.38)
        score += min(0.28, lateral_shift / 0.11 * 0.28)
        score += min(0.20, body_compression / 0.17 * 0.20)
        score += min(0.12, motion_speed / 0.12 * 0.12)
        if seat_anchor:
            score += 0.08
        if top_span >= 0.05 or side_span >= 0.07:
            score += 0.08
        score = min(1.0, score)

        detected = (
            score >= 0.56
            and seat_anchor
            and dt <= 4.2
            and (top_drop >= 0.05 or lateral_shift >= 0.09 or body_compression >= 0.14)
            and (top_span >= 0.03 or side_span >= 0.04)
        )
        return score, detected

    def _head_slump_score(
        self,
        pose_box: tuple[int, int, int, int],
        timestamp: float,
        seated_like: bool,
        seat_anchor: bool,
        lateral_shift: float,
        top_drop: float,
    ) -> tuple[float, bool]:
        if self._last_face is None:
            self._head_rel_history.clear()
            return 0.0, False

        px1, py1, px2, py2 = pose_box
        bw = float(max(1, px2 - px1))
        bh = float(max(1, py2 - py1))

        fx, fy, fw, fh = self._last_face
        fcx = fx + fw * 0.5
        fcy = fy + fh * 0.5

        # Require the face to be within or near the tracked body box.
        if fcx < px1 - bw * 0.20 or fcx > px2 + bw * 0.20 or fcy < py1 - bh * 0.20 or fcy > py2 + bh * 0.25:
            self._head_rel_history.clear()
            return 0.0, False

        rel_x = (fcx - (px1 + bw * 0.5)) / bw
        rel_y = (fcy - (py1 + bh * 0.5)) / bh
        face_ratio = float(fh) / bh

        self._head_rel_history.append((timestamp, rel_x, rel_y, face_ratio))
        while self._head_rel_history and timestamp - self._head_rel_history[0][0] > 3.5:
            self._head_rel_history.popleft()

        if len(self._head_rel_history) < 3:
            return 0.0, False

        baseline_window = list(self._head_rel_history)[:-2]
        if len(baseline_window) < 2:
            baseline_window = list(self._head_rel_history)[:2]
        if not baseline_window:
            return 0.0, False

        base_t = baseline_window[0][0]
        base_x = float(np.median([row[1] for row in baseline_window]))
        base_y = float(np.median([row[2] for row in baseline_window]))
        base_ratio = float(np.median([row[3] for row in baseline_window]))

        recent_window = list(self._head_rel_history)[-3:]
        cur_x = float(np.median([row[1] for row in recent_window]))
        cur_y = float(np.median([row[2] for row in recent_window]))
        cur_ratio = float(np.median([row[3] for row in recent_window]))
        dt = max(timestamp - base_t, 1e-6)

        dx = cur_x - base_x
        dy = cur_y - base_y
        dsize = cur_ratio - base_ratio

        sideways = abs(dx)
        downward = max(0.0, dy)
        size_change = abs(dsize)
        displacement = float(np.sqrt(dx * dx + dy * dy))
        speed = displacement / dt

        if not seated_like:
            return 0.0, False

        score = 0.0
        score += min(0.38, sideways / 0.18 * 0.38)
        score += min(0.24, downward / 0.12 * 0.24)
        score += min(0.20, size_change / 0.20 * 0.20)
        score += min(0.14, speed / 0.16 * 0.14)
        if seat_anchor:
            score += 0.08
        if lateral_shift >= 0.08 or top_drop >= 0.05:
            score += 0.08
        score = min(1.0, score)

        detected = (
            score >= 0.58
            and dt <= 4.2
            and seat_anchor
            and (sideways >= 0.09 or downward >= 0.07 or size_change >= 0.13)
        )
        return score, detected

    def _run_simulation(self, reason: str) -> None:
        rng = random.Random(42)
        source = f"simulation:{reason}"
        start = time.time()
        prev_ts = start

        while not self._stop.is_set():
            now_ts = time.time()
            dt = max(now_ts - prev_ts, 1e-6)
            prev_ts = now_ts
            fps = 1.0 / dt

            elapsed = now_ts - start
            heart_base = 74.0 + 7.0 * np.sin(elapsed / 5.0)
            bpm = heart_base + rng.uniform(-2.5, 2.5)
            confidence = 0.72 + rng.uniform(-0.08, 0.08)

            posture = "standing"
            fall_risk = max(0.0, min(0.3, 0.2 + 0.1 * np.sin(elapsed / 3.0)))
            torso_angle = 12.0 + 5.0 * np.sin(elapsed / 4.0)
            fall_detected = False
            faint_risk = 0.12
            faint_detected = False
            faint_type = None

            # Simulate occasional high-risk scenarios.
            if 18.0 <= (elapsed % 85.0) <= 24.0:
                posture = "sitting"
                torso_angle = 52.0
                fall_risk = 0.36
                faint_risk = 0.84
                faint_detected = True
                faint_type = "seated_slump"
            if 40.0 <= (elapsed % 90.0) <= 55.0:
                bpm = 138.0 + rng.uniform(-3.0, 3.0)
                confidence = 0.78
            if 70.0 <= (elapsed % 110.0) <= 76.0:
                posture = "lying"
                torso_angle = 74.0
                fall_risk = 0.93
                fall_detected = True
                faint_risk = 0.96
                faint_detected = True
                faint_type = "ground_fall"

            hr = HRMeasurement(bpm=round(float(bpm), 1), confidence=round(float(confidence), 3))
            pose = PoseMeasurement(
                posture=posture,
                fall_risk=round(float(fall_risk), 3),
                fall_detected=fall_detected,
                torso_angle=round(float(torso_angle), 1),
                person_present=True,
                person_source="sim",
                faint_risk=round(float(faint_risk), 3),
                faint_detected=faint_detected,
                faint_type=faint_type,
                seated_slump_score=round(float(faint_risk if faint_type == "seated_slump" else 0.0), 3),
            )

            state, event = self._fusion.update(now_ts, source, hr, pose, fps)
            self._emit_simulation_preview(source, state.status)
            if self._should_emit(now_ts, event):
                self._on_update(state, event)
            if event is not None:
                self._alerts.notify(event, state)

            time.sleep(0.25)

    def _emit_preview(
        self,
        frame_bgr,
        pose: PoseMeasurement,
        source: str,
        status: str,
        event: DistressEvent | None,
    ) -> None:
        if cv2 is None or self._on_preview is None:
            return

        now_ts = time.time()
        if now_ts - self._last_preview_ts < self._preview_interval_sec:
            return
        self._last_preview_ts = now_ts

        try:
            preview = frame_bgr.copy()

            if pose.person_box is not None:
                x1, y1, x2, y2 = pose.person_box
                person_source = (pose.person_source or "person").lower()
                color = (39, 174, 96)
                if person_source == "tracker":
                    color = (230, 126, 34)
                elif person_source == "hog":
                    color = (155, 89, 182)

                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    preview,
                    f"person ({person_source})",
                    (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            if self._last_face is not None:
                x, y, w, h = self._last_face
                cv2.rectangle(preview, (x, y), (x + w, y + h), (52, 152, 219), 2)
                cv2.putText(
                    preview,
                    "face",
                    (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (52, 152, 219),
                    2,
                    cv2.LINE_AA,
                )

                h_img, w_img = preview.shape[:2]
                fx1 = max(0, x + int(w * 0.20))
                fx2 = min(w_img - 1, x + int(w * 0.80))
                fy1 = max(0, y + int(h * 0.12))
                fy2 = min(h_img - 1, y + int(h * 0.32))
                if fx2 > fx1 and fy2 > fy1:
                    cv2.rectangle(preview, (fx1, fy1), (fx2, fy2), (241, 196, 15), 2)
                    cv2.putText(
                        preview,
                        "forehead ROI",
                        (fx1, max(18, fy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (241, 196, 15),
                        2,
                        cv2.LINE_AA,
                    )

            banner = f"{source} | {status}"
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 34), (20, 20, 20), -1)
            cv2.putText(
                preview,
                banner,
                (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            faint_confirmed = (
                (event is not None and event.event_type.startswith("faint_"))
                or (status == "critical" and pose.faint_detected and pose.faint_risk >= 0.60)
            )
            if faint_confirmed and pose.faint_type:
                self._fainted_banner_until_ts = max(self._fainted_banner_until_ts, now_ts + 3.0)
                cv2.putText(
                    preview,
                    f"FAINT: {pose.faint_type}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (32, 105, 247),
                    2,
                    cv2.LINE_AA,
                )

            if now_ts < self._fainted_banner_until_ts:
                h_img, w_img = preview.shape[:2]
                box_h = max(90, int(h_img * 0.18))
                y1 = max(40, (h_img // 2) - (box_h // 2))
                y2 = min(h_img - 10, y1 + box_h)

                overlay = preview.copy()
                cv2.rectangle(overlay, (20, y1), (w_img - 20, y2), (20, 20, 220), -1)
                cv2.addWeighted(overlay, 0.42, preview, 0.58, 0, preview)

                text = "FAINTED"
                scale = max(1.2, min(2.6, w_img / 420.0))
                thickness = max(2, int(scale * 2))
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                tx = max(26, (w_img - tw) // 2)
                ty = y1 + ((y2 - y1 + th) // 2) - 6
                cv2.putText(
                    preview,
                    text,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    (245, 245, 245),
                    thickness,
                    cv2.LINE_AA,
                )

            ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                self._on_preview(encoded.tobytes())
        except Exception:
            logger.exception("Preview frame generation failed")

    def _emit_simulation_preview(self, source: str, status: str) -> None:
        if cv2 is None or self._on_preview is None:
            return

        now_ts = time.time()
        if now_ts - self._last_preview_ts < self._preview_interval_sec:
            return
        self._last_preview_ts = now_ts

        try:
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
            canvas[:] = (34, 44, 59)
            cv2.putText(
                canvas,
                "Camera feed unavailable",
                (130, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Running in simulation mode",
                (145, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (190, 205, 220),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(canvas, (110, 265), (530, 390), (82, 121, 170), 2)
            cv2.putText(
                canvas,
                "person (simulated)",
                (235, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (82, 121, 170),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{source} | {status}",
                (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (220, 230, 238),
                2,
                cv2.LINE_AA,
            )

            ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                self._on_preview(encoded.tobytes())
        except Exception:
            logger.exception("Simulation preview frame generation failed")

    def _open_camera(self) -> tuple["cv2.VideoCapture", int] | None:
        if cv2 is None:
            return None

        preferred = self._settings.camera_index
        candidates: list[int] = []
        if preferred >= 0:
            candidates.append(preferred)
        candidates.extend(index for index in range(5) if index != preferred)

        for index in candidates:
            try:
                cap = self._try_open_index(index)
            except Exception:
                logger.exception("Camera probe failed for index %s", index)
                cap = None
            if cap is not None:
                logger.info("Using webcam index %s", index)
                return cap, index
        return None

    @staticmethod
    def _try_open_index(index: int):
        if cv2 is None:
            return None

        backends = [None]
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)

        for backend in backends:
            try:
                cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception:
                logger.exception("cv2.VideoCapture failed for index=%s backend=%s", index, backend)
        return None

    @staticmethod
    def _format_engine_error_reason(exc: Exception) -> str:
        name = exc.__class__.__name__.lower()
        slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        if not slug:
            slug = "unknown"
        return f"engine_error_{slug}"
