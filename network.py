import asyncio
import json
import importlib

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


def create_app(state):
    app = FastAPI()
    active_websockets = set()
    active_datachannels = set()
    peer_connections = set()
    broadcast_task = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _handle_incoming_payload(payload):
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

    async def _broadcast_outgoing_loop():
        while True:
            messages = state.pop_outgoing_all()
            if messages:
                for message in messages:
                    try:
                        encoded = json.dumps(message)
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
        return {"webrtc": bool(HAS_WEBRTC)}

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


def run_server(state, host="0.0.0.0", port=8000):
    app = create_app(state)
    uvicorn.run(app, host=host, port=port, log_level="error")
