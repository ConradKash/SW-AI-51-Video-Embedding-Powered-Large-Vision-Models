from typing import Union
from backend.function import predict
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
import secrets
import os
from pathlib import Path
import json
import io
import base64
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image

app = FastAPI()

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

    predicted_class, confidence = predict(generated_name)
    return {"predicted_class": predicted_class, "confidence": confidence}


@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    """WebSocket endpoint that receives frames and returns predictions.

    Behavior:
    - Accepts either raw binary frames (JPEG/PNG bytes) or text messages containing
      JSON {"frame": "data:image/jpeg;base64,..."}.
    - Accepts an optional JSON message to set `predict_every` e.g. {"predict_every":5}.
    - Sends an acknowledgement for each frame and sends a prediction result every
      `predict_every` frames.
    """
    await websocket.accept()
    frame_count = 0
    predict_every = 5  # default: predict every 5th frame

    try:
        while True:
            message = await websocket.receive()

            # Text messages (JSON control or base64 frame)
            if "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except Exception:
                    await websocket.send_text(json.dumps({"error": "invalid json"}))
                    continue

                # Update control parameters
                if "predict_every" in payload:
                    try:
                        predict_every = int(payload["predict_every"])
                        await websocket.send_text(json.dumps({"info": f"predict_every set to {predict_every}"}))
                    except Exception:
                        await websocket.send_text(json.dumps({"error": "invalid predict_every"}))
                    continue

                # If payload contains a base64 frame
                if "frame" in payload:
                    b64 = payload["frame"].split(",")[-1]
                    try:
                        frame_bytes = base64.b64decode(b64)
                    except Exception:
                        await websocket.send_text(json.dumps({"error": "invalid base64 frame"}))
                        continue
                else:
                    await websocket.send_text(json.dumps({"error": "no frame in message"}))
                    continue

            # Binary frames sent directly
            elif "bytes" in message and message["bytes"] is not None:
                frame_bytes = message["bytes"]
            else:
                # Unknown message type
                await websocket.send_text(json.dumps({"error": "unsupported message type"}))
                continue

            # Open image and run prediction when appropriate
            try:
                img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"cannot open image: {e}"}))
                continue

            frame_count += 1

            if frame_count % predict_every == 0:
                try:
                    result = predict(img)
                    await websocket.send_text(json.dumps({"type": "prediction", "frame": frame_count, "result": result}))
                except Exception as e:
                    await websocket.send_text(json.dumps({"type": "prediction_error", "frame": frame_count, "error": str(e)}))
            else:
                # lightweight ack
                await websocket.send_text(json.dumps({"type": "ack", "frame": frame_count}))

    except WebSocketDisconnect:
        # client disconnected
        return
