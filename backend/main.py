from backend.function import predict, predict_from_bytes, model_status, _load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import secrets
import os
from pathlib import Path
import json
import base64
from fastapi import WebSocket, WebSocketDisconnect
import asyncio
import datetime
import traceback

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.active_ws = {}
STATIC_DIR = Path("static")
IMAGES_DIR = STATIC_DIR / "images"

try:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    # If we can't create directories, raise a clear error early.
    raise RuntimeError(f"Failed to ensure static directories exist: {e}")

# Mount using an absolute path for robustness
app.mount("/static", StaticFiles(directory=str(STATIC_DIR.resolve())), name="static")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health_check():
    """Basic health endpoint that reports model load status.
    Returns 200 even when model is missing so orchestration can still probe
    the service; the payload includes model.loaded boolean.
    """
    try:
        status = model_status()
        return {"status": "ok", "model": status}
    except Exception as e:
        return {"status": "ok", "model": {"loaded": False, "path": "unknown", "error": str(e)}}


@app.on_event("startup")
async def try_eager_load_model():
    """Attempt to load model at startup if the weights are present. This is
    optional — failure will not stop the app. Useful when `models/` is
    mounted into the container so the model can be available immediately.
    """
    try:
        _load_model()
        print(f"Model loaded eagerly at startup: {model_status()}")
    except FileNotFoundError:
        print("Model weights not found at startup; will load lazily on first prediction.")
    except Exception as e:
        print("Unexpected error while eager-loading model:", e)


@app.post("/predict")
async def make_prediction(file: UploadFile = File(...)):
    file_name = file.filename

    if not os.path.exists("./static/images/"):
        os.makedirs("./static/images/")
    File_path = "./static/images/"

    extension = file_name.split(".")[-1]

    token_name = secrets.token_hex(16)+"." + extension
    generated_name = File_path + token_name
    print(f"Generated file name: {generated_name}")

    file_content = await file.read()

    with open(generated_name, "wb") as f:
        f.write(file_content)
    print(f"File saved at: {generated_name}")

    try:
        predicted_class, confidence = predict(generated_name)
        # predict may return (None, Exception) on error — normalize that
        if predicted_class is None and isinstance(confidence, Exception):
            raise HTTPException(status_code=500, detail=str(confidence))
        return {"predicted_class": predicted_class, "confidence": confidence}
    except RuntimeError as e:
        # RuntimeError used for missing model file; map to 503 Service Unavailable
        raise HTTPException(status_code=503, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    # determine client host (ip) for connection tracking
    client_host = "unknown"
    try:
        client_host = websocket.client[0]
    except Exception:
        pass

    # If an existing websocket from this client IP is active, close it to
    # avoid duplicate concurrent connections from the same client during dev
    # (e.g. auto-reloads or reconnect loops).
    try:
        existing = app.state.active_ws.get(client_host)
        if existing is not None and existing is not websocket:
            try:
                await existing.close()
            except Exception:
                pass
    except Exception:
        pass

    # register this websocket for the client host
    try:
        app.state.active_ws[client_host] = websocket
    except Exception:
        pass

    await websocket.accept()
    frame_count = 0
    predict_every_frames = 5
    predict_every_seconds = None
    running = False
    latest_frame_bytes: bytes | None = None
    predictor_task: asyncio.Task | None = None
    prediction_lock = asyncio.Lock()

    print("WebSocket /ws/predict connected")

    async def _run_prediction(frame_bytes: bytes):
        """Run prediction and send result over WebSocket."""
        try:
            predicted_class, confidence = await asyncio.get_running_loop().run_in_executor(
                None, predict_from_bytes, frame_bytes
            )
            payload = {
                "type": "prediction",
                "predicted_class": predicted_class,
                "confidence": confidence,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            await websocket.send_json(payload)
        except Exception as e:
            await websocket.send_json({"type": "error", "message": str(e)})

    async def send_prediction_every_interval(seconds: int):
        """Background task: predict on latest frame every N seconds."""
        try:
            while running:
                if latest_frame_bytes is None:
                    await asyncio.sleep(0.5)
                    continue
                async with prediction_lock:
                    # Use a copy to avoid mutation during sleep
                    frame_copy = latest_frame_bytes
                await _run_prediction(frame_copy)
                await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "message": "Background predictor crashed",
                "trace": traceback.format_exc()
            })

    try:
        while True:
            message = await websocket.receive()

            # --- Handle binary frame ---
            if message.get("type") == "websocket.receive" and "bytes" in message:
                latest_frame_bytes = message["bytes"]
                frame_count += 1
                await websocket.send_json({"type": "ack", "frame_count": frame_count})

                if running and predict_every_frames and frame_count % predict_every_frames == 0:
                    asyncio.create_task(_run_prediction(latest_frame_bytes))
                continue

            # --- Handle text message ---
            if message.get("type") == "websocket.receive" and "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON"})
                    continue

                # --- Control: start/stop ---
                if data.get("action") == "start":
                    if not running:
                        running = True
                        if predict_every_seconds:
                            predictor_task = asyncio.create_task(
                                send_prediction_every_interval(
                                    predict_every_seconds)
                            )
                        await websocket.send_json({"type": "status", "message": "analysis_started"})
                    else:
                        await websocket.send_json({"type": "status", "message": "already_running"})
                    continue

                if data.get("action") == "stop":
                    running = False
                    if predictor_task:
                        predictor_task.cancel()
                        predictor_task = None
                    await websocket.send_json({"type": "status", "message": "analysis_stopped"})
                    continue

                # --- Configure cadence ---
                if "predict_every" in data:
                    try:
                        val = int(data["predict_every"])
                        if val <= 0:
                            raise ValueError("Must be positive")
                        predict_every_frames = val
                        await websocket.send_json({"type": "status", "message": f"predict_every_frames set to {val}"})
                    except (ValueError, TypeError) as e:
                        await websocket.send_json({"type": "error", "message": f"Invalid predict_every: {e}"})
                    continue

                if "predict_every_seconds" in data:
                    try:
                        val = int(data["predict_every_seconds"])
                        if val <= 0:
                            raise ValueError("Must be positive")
                        predict_every_seconds = val
                        if running:
                            if predictor_task:
                                predictor_task.cancel()
                            predictor_task = asyncio.create_task(
                                send_prediction_every_interval(val)
                            )
                        await websocket.send_json({"type": "status", "message": f"predict_every_seconds set to {val}s"})
                    except (ValueError, TypeError) as e:
                        await websocket.send_json({"type": "error", "message": f"Invalid predict_every_seconds: {e}"})
                    continue

                # --- Handle base64 frame ---
                frame_b64 = data.get("frame")
                if isinstance(frame_b64, str):
                    try:
                        if frame_b64.startswith("data:"):
                            frame_b64 = frame_b64.split(",", 1)[1]
                        latest_frame_bytes = base64.b64decode(frame_b64)
                        frame_count += 1
                        await websocket.send_json({"type": "ack", "frame_count": frame_count})
                        if running and predict_every_frames and frame_count % predict_every_frames == 0:
                            asyncio.create_task(
                                _run_prediction(latest_frame_bytes))
                    except Exception as e:
                        await websocket.send_json({"type": "error", "message": f"Base64 decode failed: {e}"})
                    continue

                await websocket.send_json({"type": "error", "message": "Unknown command"})

    except WebSocketDisconnect:
        pass
    finally:
        running = False
        if predictor_task:
            predictor_task.cancel()
        # remove from active map if still registered
        try:
            if app.state.active_ws.get(client_host) is websocket:
                del app.state.active_ws[client_host]
        except Exception:
            pass

        print("WebSocket /ws/predict disconnected from", client_host)
