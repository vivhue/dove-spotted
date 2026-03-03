# Dove Spotted

Webcam-only monitoring service that:

- Estimates heart rate continuously with rPPG (facial color signal).
- Tracks posture and fall risk with pose estimation.
- Tracks full-body box with hybrid pose + tracker + detector fallback.
- Fuses both streams into a single distress score.
- Pushes live updates to a web dashboard.
- Sends Telegram alerts for RED events (fall / cardiac distress).
- Tracks fainting patterns: standing collapse, ground fall, and seated slump.

## Architecture

- `monitor/rppg.py`: temporal rPPG estimator from forehead ROI.
- `monitor/pose.py`: MediaPipe posture + fall-risk heuristics.
- `monitor/body_tracker.py`: whole-body tracking (pose + CSRT/KCF + HOG fallback).
- `monitor/fusion.py`: multi-signal risk fusion and event detection.
- `monitor/alerts.py`: Twilio SMS alerts with cooldown (optional).
- `monitor/session_store.py`: monitor sessions, pairing codes, and chat subscriptions.
- `monitor/telegram_pairing.py`: Telegram polling bot and RED alert delivery.
- `monitor/engine.py`: webcam loop and orchestration.
- `main.py`: FastAPI API + WebSocket + dashboard serving.
- `web/monitor.html`: elderly monitor page (webcam + pairing code).
- `web/caregiver.html`: caregiver session dashboard.

## Quick Start
1. Create a .env file in the root folder with contents:
TELEGRAM_BOT_TOKEN=8623880967:AAE_HHLQuvWppt-M62DhxiGnsBnzUW8r7Nc
APP_PUBLIC_URL=http://127.0.0.1:8000

2. Create and activate a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
uvicorn main:app --reload
```

If `uvicorn` is not found:

```bash
python -m uvicorn main:app --reload
```

5. Open monitor:

```text
http://127.0.0.1:8000/monitor
```

6. Open `http://127.0.0.1:8000/monitor` and note the 6-digit pairing code.
7. In Telegram, open bot chat '@Motiventra_Bot' and send:

```bash
/pair <code>
```

8. The website will access your webcam to check detect a person's fainting spells. If detected, an alert will be sent to the caregiver (telegram paired with the bot) to notify them of possible signs. 

Telegram send logic:
- Monitor session shows a pairing code; caregiver uses `/pair <code>`.
- Bot commands: `/start`, `/pair <code>`, `/status`, `/unpair`.
- Sends for RED session events (`fall` or `cardiac_distress`) to caregivers paired to that session only.
- Sends on RED transition (`previous != RED`, `current == RED`), or while RED persists after cooldown.
- While `faint_type` remains detected, Telegram sends faint alerts at most once every 10 seconds.
- Run only one backend instance per Telegram bot token. This app enforces a local poll lock (`.telegram_poll.lock`) to avoid `409 Conflict`.
- Cooldown is per (`sessionId`, `chatId`) via `TELEGRAM_ALERT_COOLDOWN_SEC`.
- Local demo persistence files:
  - `monitorSessions.json` (sessions + pairingCodeIndex)
  - `telegramSubscriptions.json` (bySession + byChat)

## API

- `GET /api/state`: latest fused monitoring state.
- `GET /api/events`: recent detected events.
- `POST /api/session/create`: create/resume monitor session; returns `{sessionId, pairingCode, caregiverUrl}`.
- `GET /api/session/{sessionId}/status`: session latest vitals/risk/events.
- `GET /api/telegram/status`: Telegram bot enabled + caregiver registration status.
- `POST /api/test-alert?sessionId=...`: force a RED Telegram test alert for one session.
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
- If `TELEGRAM_BOT_TOKEN` is missing, Telegram sending is skipped automatically.
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
- On macOS, grant camera access to Terminal/iTerm/Python in Privacy & Security > Camera.
- The dashboard now includes a live preview panel fed from `/video_feed`.
