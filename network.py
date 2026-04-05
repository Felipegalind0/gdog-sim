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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/offer")
    async def offer(params: dict):
        if not HAS_WEBRTC:
            return {"error": "WebRTC not installed"}

        offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
        pc = RTCPeerConnection()

        @pc.on("datachannel")
        def on_datachannel(channel):
            @channel.on("message")
            def on_message(message):
                try:
                    data = json.loads(message)
                    state.update(*_parse_command_payload(data))
                except Exception:
                    pass

        await pc.setRemoteDescription(offer_sdp)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        print("WebSocket client connected")
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    parsed = json.loads(data)
                    state.update(*_parse_command_payload(parsed))
                except Exception:
                    pass
        except WebSocketDisconnect:
            print("WebSocket client disconnected")

    return app


def run_server(state, host="0.0.0.0", port=8000):
    app = create_app(state)
    uvicorn.run(app, host=host, port=port, log_level="error")
