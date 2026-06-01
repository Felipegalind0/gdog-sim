import asyncio
import json
import importlib
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from commands import _parse_command_payload


try:
    aiortc_mod = importlib.import_module("aiortc")
    RTCPeerConnection = aiortc_mod.RTCPeerConnection
    RTCSessionDescription = aiortc_mod.RTCSessionDescription
    HAS_WEBRTC = True
except Exception:
    RTCPeerConnection = None
    RTCSessionDescription = None
    HAS_WEBRTC = False
    print("aiortc not installed. WebRTC disabled. Using WebSockets as primary.")


def create_app(state, runtime_info=None, tune_registry=None):
    app = FastAPI()
    active_websockets = set()
    active_datachannels = set()
    peer_connections = set()
    broadcast_task = None
    recent_outgoing_events = deque(maxlen=1200)
    next_event_id = 0
    runtime_info_payload = runtime_info if isinstance(runtime_info, dict) else {}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _handle_incoming_payload(payload):
        # Handle tune commands before normal command parsing.
        tune_cmd = payload.get("tune_cmd")
        if tune_cmd is not None:
            _handle_tune_payload(payload, tune_cmd)
            return

        args = _parse_command_payload(payload)
        voice_cmd = payload.get("voice_cmd")
        voice_direction = payload.get("direction")
        voice_call_id = payload.get("call_id")
        try:
            voice_amount = float(payload.get("amount", 0.0))
        except Exception:
            voice_amount = 0.0
        state.update(
            *args,
            voice_cmd=voice_cmd,
            voice_direction=voice_direction,
            voice_amount=voice_amount,
            voice_call_id=voice_call_id,
        )

    def _handle_tune_payload(payload, tune_cmd):
        if tune_registry is None:
            state.push_outgoing({"type": "tune_list", "vars": []})
            return

        cmd = str(tune_cmd).strip().lower()
        if cmd == "list":
            state.push_outgoing({"type": "tune_list", "vars": tune_registry.list_all()})
        elif cmd == "set":
            name = str(payload.get("name", "")).strip()
            try:
                value = float(payload.get("value", 0))
            except (TypeError, ValueError):
                state.push_outgoing({"type": "tune_result", "name": name, "ok": False})
                return
            new_val = tune_registry.set(name, value)
            state.push_outgoing({
                "type": "tune_result",
                "name": name,
                "value": new_val,
                "ok": new_val is not None,
            })
        elif cmd == "reset":
            name = str(payload.get("name", "")).strip()
            if name:
                tune_registry.reset(name)
            else:
                tune_registry.reset()
            state.push_outgoing({"type": "tune_list", "vars": tune_registry.list_all()})
        else:
            state.push_outgoing({"type": "tune_list", "vars": tune_registry.list_all()})

    async def _broadcast_outgoing_loop():
        nonlocal next_event_id
        while True:
            messages = state.pop_outgoing_all()
            if messages:
                for message in messages:
                    try:
                        payload_obj = dict(message)
                    except Exception:
                        payload_obj = {"type": "unknown", "payload": str(message)}

                    next_event_id += 1
                    payload_obj["_event_id"] = int(next_event_id)
                    recent_outgoing_events.append(payload_obj)

                    try:
                        encoded = json.dumps(payload_obj)
                    except Exception:
                        continue

                    stale_websockets = []
                    for websocket in tuple(active_websockets):
                        try:
                            await websocket.send_text(encoded)
                        except Exception:
                            stale_websockets.append(websocket)
                    for websocket in stale_websockets:
                        active_websockets.discard(websocket)

                    stale_channels = []
                    for channel in tuple(active_datachannels):
                        try:
                            if getattr(channel, "readyState", None) != "open":
                                stale_channels.append(channel)
                                continue
                            channel.send(encoded)
                        except Exception:
                            stale_channels.append(channel)
                    for channel in stale_channels:
                        active_datachannels.discard(channel)

            await asyncio.sleep(0.05)

    @app.get("/events")
    async def events(since: int = 0, limit: int = 200):
        try:
            since_id = max(int(since), 0)
        except Exception:
            since_id = 0

        try:
            max_events = int(limit)
        except Exception:
            max_events = 200
        max_events = min(max(max_events, 1), 500)

        events_out = [
            evt
            for evt in recent_outgoing_events
            if int(evt.get("_event_id", 0)) > since_id
        ]
        if len(events_out) > max_events:
            events_out = events_out[-max_events:]

        return {
            "ok": True,
            "latest_event_id": int(next_event_id),
            "events": events_out,
        }

    @app.on_event("startup")
    async def _startup():
        nonlocal broadcast_task
        broadcast_task = asyncio.create_task(_broadcast_outgoing_loop())

    @app.on_event("shutdown")
    async def _shutdown():
        nonlocal broadcast_task

        if broadcast_task is not None:
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass
            broadcast_task = None

        for pc in tuple(peer_connections):
            try:
                await pc.close()
            except Exception:
                pass
        peer_connections.clear()
        active_datachannels.clear()
        active_websockets.clear()

    @app.get("/capabilities")
    async def capabilities():
        return {
            "webrtc": bool(HAS_WEBRTC),
            "runtime_info": runtime_info_payload,
        }

    @app.get("/tune")
    async def tune_list_endpoint():
        if tune_registry is None:
            return {"ok": True, "vars": []}
        return {"ok": True, "vars": tune_registry.list_all()}

    @app.post("/command")
    async def command(payload: dict):
        if not isinstance(payload, dict):
            return {"ok": False, "error": "Command payload must be a JSON object."}

        _handle_incoming_payload(payload)
        return {"ok": True}

    @app.post("/offer")
    async def offer(params: dict):
        if not HAS_WEBRTC:
            return {"error": "WebRTC not installed"}

        offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()
        peer_connections.add(pc)

        @pc.on("datachannel")
        def on_datachannel(channel):
            active_datachannels.add(channel)

            @channel.on("close")
            def on_close():
                active_datachannels.discard(channel)

            @channel.on("message")
            def on_message(message):
                try:
                    if isinstance(message, bytes):
                        message = message.decode("utf-8")
                    data = json.loads(message)
                    _handle_incoming_payload(data)
                except Exception:
                    pass

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                peer_connections.discard(pc)
                try:
                    await pc.close()
                except Exception:
                    pass

        await pc.setRemoteDescription(offer_sdp)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        active_websockets.add(websocket)
        print("WebSocket client connected")
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    parsed = json.loads(data)
                    _handle_incoming_payload(parsed)
                except Exception:
                    pass
        except WebSocketDisconnect:
            print("WebSocket client disconnected")
        finally:
            active_websockets.discard(websocket)

    return app


def run_server(state, host="0.0.0.0", port=8000, runtime_info=None, tune_registry=None):
    app = create_app(state, runtime_info=runtime_info, tune_registry=tune_registry)
    uvicorn.run(app, host=host, port=port, log_level="error")
