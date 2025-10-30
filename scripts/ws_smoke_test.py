#!/usr/bin/env python3
"""Simple smoke test for the /ws/predict endpoint.

Usage:
    python scripts/ws_smoke_test.py --uri ws://localhost:8000/ws/predict --mode base64

The script will:
- connect to the websocket
- optionally send a control message (predict_every)
- send a few frames (either binary bytes or JSON text with base64 under key 'frame')
- print any messages received from the server
"""
import argparse
import asyncio
import base64
import json
from pathlib import Path

import websockets
from PIL import Image
import io
from typing import Optional


async def run(uri: str, mode: str, image_path: Optional[str], predict_every: int):
    print(f"Connecting to {uri} mode={mode}")
    async with websockets.connect(uri) as ws:
        # send control: predict_every (frames)
        if predict_every is not None:
            ctrl = {"predict_every": int(predict_every)}
            await ws.send(json.dumps(ctrl))
            print(f"Sent control: {ctrl}")

        # prepare a sample image
        if image_path:
            p = Path(image_path)
            img_bytes = p.read_bytes()
        else:
            # create a tiny RGB image
            im = Image.new("RGB", (224, 224), color=(120, 120, 120))
            bio = io.BytesIO()
            im.save(bio, format="JPEG")
            img_bytes = bio.getvalue()

        async def sender():
            # send 10 frames spaced a little
            for i in range(1, 11):
                if mode == "binary":
                    await ws.send(img_bytes)
                    print(f"Sent binary frame {i}")
                else:
                    b64 = base64.b64encode(img_bytes).decode("ascii")
                    payload = json.dumps({"frame": b64})
                    await ws.send(payload)
                    print(f"Sent base64 frame {i}")
                await asyncio.sleep(0.25)

            # after sending frames wait a bit and then close
            await asyncio.sleep(2)
            await ws.close()

        async def receiver():
            try:
                async for message in ws:
                    # websockets library returns str for text and bytes for binary
                    if isinstance(message, bytes):
                        print("Received binary message (len=", len(message), ")")
                    else:
                        try:
                            data = json.loads(message)
                            print("Received:", json.dumps(data, indent=2))
                        except Exception:
                            print("Received text:", message)
            except Exception as e:
                print("Receiver exiting, reason:", e)

        await asyncio.gather(sender(), receiver())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="ws://localhost:8000/ws/predict")
    parser.add_argument(
        "--mode", choices=["binary", "base64"], default="base64")
    parser.add_argument("--image", default=None)
    parser.add_argument("--predict-every", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(run(args.uri, args.mode, args.image, args.predict_every))
