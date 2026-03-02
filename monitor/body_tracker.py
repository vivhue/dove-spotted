from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency behavior
    cv2 = None


@dataclass
class BodyTrackResult:
    box: tuple[int, int, int, int] | None
    source: str


class BodyTracker:
    """Hybrid whole-body tracker for front/back/side body orientations."""

    def __init__(self) -> None:
        self._frame_index = 0
        self._last_box: tuple[int, int, int, int] | None = None
        self._last_source = "none"
        self._tracker = None
        self._frames_since_anchor = 0
        self._max_tracker_only_frames = 90

        if cv2 is not None:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        else:
            self._hog = None

    def update(
        self,
        frame_bgr,
        pose_box: tuple[int, int, int, int] | None,
    ) -> BodyTrackResult:
        self._frame_index += 1

        track_box = self._update_tracker(frame_bgr)
        if track_box is not None:
            if self._frames_since_anchor <= self._max_tracker_only_frames:
                self._last_box = track_box
                self._last_source = "tracker"
                self._frames_since_anchor += 1
            else:
                track_box = None

        if pose_box is not None and self._is_valid_box(pose_box, frame_bgr.shape[1], frame_bgr.shape[0]):
            if track_box is None or self._iou(track_box, pose_box) < 0.35:
                self._start_tracker(frame_bgr, pose_box)
            self._last_box = pose_box
            self._last_source = "pose"
            self._frames_since_anchor = 0
            return BodyTrackResult(box=pose_box, source="pose")

        need_hog = (
            self._frame_index % 10 == 0
            or self._last_box is None
            or self._frames_since_anchor > self._max_tracker_only_frames
        )
        if need_hog:
            hog_box = self._detect_hog(frame_bgr)
            if hog_box is not None:
                self._start_tracker(frame_bgr, hog_box)
                self._last_box = hog_box
                self._last_source = "hog"
                self._frames_since_anchor = 0
                return BodyTrackResult(box=hog_box, source="hog")

        if track_box is not None:
            return BodyTrackResult(box=track_box, source="tracker")

        if self._last_box is not None:
            return BodyTrackResult(box=self._last_box, source=self._last_source)
        return BodyTrackResult(box=None, source="none")

    def _update_tracker(self, frame_bgr) -> tuple[int, int, int, int] | None:
        if self._tracker is None:
            return None
        try:
            ok, box_xywh = self._tracker.update(frame_bgr)
        except Exception:
            return None
        if not ok:
            return None

        x, y, w, h = box_xywh
        x1 = int(max(0, x))
        y1 = int(max(0, y))
        x2 = int(max(x1 + 1, x + w))
        y2 = int(max(y1 + 1, y + h))
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _start_tracker(self, frame_bgr, box_xyxy: tuple[int, int, int, int]) -> None:
        tracker = self._create_tracker()
        if tracker is None:
            self._tracker = None
            return

        x1, y1, x2, y2 = box_xyxy
        box_xywh = (float(x1), float(y1), float(max(2, x2 - x1)), float(max(2, y2 - y1)))
        try:
            ok = tracker.init(frame_bgr, box_xywh)
        except Exception:
            ok = False
        if ok:
            self._tracker = tracker
        else:
            self._tracker = None

    def _detect_hog(self, frame_bgr) -> tuple[int, int, int, int] | None:
        if cv2 is None or self._hog is None:
            return None
        try:
            boxes, weights = self._hog.detectMultiScale(
                frame_bgr,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.03,
            )
        except Exception:
            return None

        best_score = -1.0
        best_box: tuple[int, int, int, int] | None = None
        for i, rect in enumerate(boxes):
            x, y, w, h = rect
            score = float(weights[i]) if i < len(weights) else 0.0
            if w <= 30 or h <= 60:
                continue
            if score > best_score:
                best_score = score
                best_box = (int(x), int(y), int(x + w), int(y + h))
        return best_box

    @staticmethod
    def _create_tracker() -> Any:
        if cv2 is None:
            return None

        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        if hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        return None

    @staticmethod
    def _is_valid_box(box: tuple[int, int, int, int], width: int, height: int) -> bool:
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return False
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        return (x2 - x1) >= 30 and (y2 - y1) >= 50

    @staticmethod
    def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float((ax2 - ax1) * (ay2 - ay1))
        area_b = float((bx2 - bx1) * (by2 - by1))
        union = area_a + area_b - inter + 1e-9
        return inter / union
