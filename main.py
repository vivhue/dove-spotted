from __future__ import annotations

import asyncio
from collections import deque
import logging
from pathlib import Path
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from monitor.config import load_settings
from monitor.engine import MonitoringEngine
from monitor.schemas import DistressEvent, MonitorState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            sockets = list(self._connections)
        for socket in sockets:
            try:
                await socket.send_json(payload)
            except Exception:
                await self.disconnect(socket)


settings = load_settings()
manager = ConnectionManager()
state_lock = threading.Lock()
latest_state = MonitorState.boot()
recent_events: deque[DistressEvent] = deque(maxlen=200)
preview_lock = threading.Lock()
latest_preview_jpeg: bytes | None = None
loop_ref: asyncio.AbstractEventLoop | None = None

app = FastAPI(title="Dove Spotted Monitor", version="0.1.0")
engine: MonitoringEngine | None = None


class HeartIssueRequest(BaseModel):
    issue: str


class ManualBpmRequest(BaseModel):
    bpm: float


def _on_engine_update(state: MonitorState, event: DistressEvent | None) -> None:
    global latest_state
    with state_lock:
        latest_state = state
        if event is not None:
            recent_events.appendleft(event)

    payload = {"type": "update", "state": state.to_dict()}
    if event is not None:
        payload["event"] = event.to_dict()

    if loop_ref is not None and not loop_ref.is_closed():
        try:
            asyncio.run_coroutine_threadsafe(manager.broadcast(payload), loop_ref)
        except RuntimeError:
            logger.debug("Event loop unavailable during update broadcast")


def _on_preview_frame(frame_jpeg: bytes) -> None:
    global latest_preview_jpeg
    with preview_lock:
        latest_preview_jpeg = frame_jpeg


@app.on_event("startup")
async def on_startup() -> None:
    global loop_ref, engine
    loop_ref = asyncio.get_running_loop()
    engine = MonitoringEngine(
        settings=settings,
        on_update=_on_engine_update,
        on_preview=_on_preview_frame,
    )
    engine.start()
    logger.info("Server startup complete")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    if engine is not None:
        engine.stop()
    logger.info("Server shutdown complete")


@app.get("/")
async def dashboard() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def _preview_stream():
    while True:
        with preview_lock:
            frame = latest_preview_jpeg
        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(frame)).encode("ascii")
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(0.06)


@app.get("/video_feed")
async def video_feed() -> StreamingResponse:
    return StreamingResponse(
        _preview_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/frame.jpg")
async def frame_jpg() -> Response:
    with preview_lock:
        frame = latest_preview_jpeg
    if frame is None:
        return Response(status_code=204)
    return Response(
        content=frame,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    with state_lock:
        return latest_state.to_dict()


@app.get("/api/events")
async def get_events() -> list[dict[str, Any]]:
    with state_lock:
        return [event.to_dict() for event in list(recent_events)]


@app.post("/api/heart_issue")
async def set_heart_issue(payload: HeartIssueRequest) -> dict[str, Any]:
    if engine is None:
        raise HTTPException(status_code=503, detail="Monitoring engine unavailable")

    issue_raw = payload.issue.strip().lower()
    issue_map: dict[str, str | None] = {
        "vt": "vt",
        "vf": "vf",
        "asystole": "asystole",
        "none": None,
        "stop": None,
    }
    if issue_raw not in issue_map:
        raise HTTPException(status_code=400, detail="Unsupported heart issue mode")

    engine.set_heart_issue(issue_map[issue_raw])
    return {"heart_issue": engine.get_heart_issue()}


@app.post("/api/manual_bpm")
async def set_manual_bpm(payload: ManualBpmRequest) -> dict[str, Any]:
    if engine is None:
        raise HTTPException(status_code=503, detail="Monitoring engine unavailable")

    bpm = float(payload.bpm)
    if bpm < 0.0 or bpm > 200.0:
        raise HTTPException(status_code=400, detail="Manual BPM must be within 0 to 200")

    engine.set_manual_bpm(bpm)
    return {"manual_bpm": engine.get_manual_bpm()}


@app.websocket("/ws/live")
async def live_updates(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    with state_lock:
        snapshot = latest_state.to_dict()
        events = [event.to_dict() for event in list(recent_events)[:20]]
    await websocket.send_json({"type": "snapshot", "state": snapshot, "events": events})

    try:
        while True:
            # We do not require client input, but this keeps connection lifecycle explicit.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception:
        await manager.disconnect(websocket)
