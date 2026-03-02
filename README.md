# Dove Spotted

Webcam-only monitoring service that:

- Estimates heart rate continuously with rPPG (facial color signal).
- Tracks posture and fall risk with pose estimation.
- Tracks full-body box with hybrid pose + tracker + detector fallback.
- Fuses both streams into a single distress score.
- Pushes live updates to a web dashboard.
- Sends SMS alerts for critical events.
- Tracks fainting patterns: standing collapse, ground fall, and seated slump.

## Architecture

- `monitor/rppg.py`: temporal rPPG estimator from forehead ROI.
- `monitor/pose.py`: MediaPipe posture + fall-risk heuristics.
- `monitor/body_tracker.py`: whole-body tracking (pose + CSRT/KCF + HOG fallback).
- `monitor/fusion.py`: multi-signal risk fusion and event detection.
- `monitor/alerts.py`: Twilio SMS alerts with cooldown.
- `monitor/engine.py`: webcam loop and orchestration.
- `main.py`: FastAPI API + WebSocket + dashboard serving.
- `web/index.html`: caregiver dashboard UI.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables (optional defaults shown):

```bash
cp .env.example .env
# PowerShell: Copy-Item .env.example .env
```

Then export values in your shell (or set them in your process manager):

```bash
CAMERA_INDEX=0
SIMULATE_MODE=false
FAINT_HIGH_SENSITIVITY=false
WARNING_LOW_BPM=50
WARNING_HIGH_BPM=120
CRITICAL_LOW_BPM=45
CRITICAL_HIGH_BPM=130
ALERT_COOLDOWN_SEC=120
SMS_RECIPIENTS=+15551234567,+15559876543
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+15550001111
```

4. Run:

```bash
uvicorn main:app --reload
```

5. Open dashboard:

```text
http://127.0.0.1:8000
```

## API

- `GET /api/state`: latest fused monitoring state.
- `GET /api/events`: recent detected events.
- `GET /video_feed`: MJPEG live feed with person/face/forehead overlays.
- `GET /api/frame.jpg`: latest preview frame (single JPEG snapshot endpoint).
- `WS /ws/live`: real-time stream of state + events.

`/api/state` includes:
- `person_present`: whether a person is currently confirmed in-frame.
- `person_source`: `pose`, `tracker`, `hog`, or `null`.
- `faint_risk`: fused pose-based faint risk score `[0.0, 1.0]`.
- `seated_slump_score`: seated upper-body slump score `[0.0, 1.0]` for live debugging/tuning.
- `faint_type`: one of `standing_faint`, `ground_fall`, `seated_slump`, or `null`.

Faint alerts are only emitted when a person is confirmed across consecutive frames.
When pose is unavailable (`camera:*:hr_only`), faint risk falls back to body-box motion cues (rapid/cumulative drop + box compression).
`FAINT_HIGH_SENSITIVITY=false` (default) uses a balanced profile to reduce false positives.
Set `FAINT_HIGH_SENSITIVITY=true` for aggressive demo sensitivity.
When fainting is detected, the live feed shows a large `FAINTED` banner for visibility.

## Notes

- Set `SIMULATE_MODE=true` to run without a camera.
- If Twilio credentials are missing, SMS sending is skipped automatically.
- This project is for prototyping/research only and is not a medical device.

## Troubleshooting Webcam

- Check the dashboard footer `Source`:
  - `camera:0`, `camera:1`, etc. means webcam mode is active.
  - `simulation:camera_unavailable` means no camera could be opened.
  - `simulation:opencv_missing` means OpenCV is not installed in your Python env.
  - `simulation:camera_no_frames` means camera opened but returned no images.
  - `simulation:engine_error_*` means camera pipeline crashed; suffix shows exception class.
- Try `CAMERA_INDEX=0`, then `1`, then `2` if your webcam is not detected.
- Close other apps that may lock the camera (Zoom, Teams, browser tabs).
- On Windows, grant camera access to desktop apps in system privacy settings.
- The dashboard now includes a live preview panel fed from `/video_feed`.
