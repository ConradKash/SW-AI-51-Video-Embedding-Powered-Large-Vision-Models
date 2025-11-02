## Running backend (uvicorn) and frontend (Streamlit) together with Docker

This repository contains a Dockerfile and a docker-compose configuration to run both the FastAPI backend and the Streamlit frontend in the same container. The container starts:

- FastAPI via `uvicorn` on port 8000
- Streamlit on port 8501

Build the image (from repo root):

```bash
docker build -t dental-app:latest .
```

Run with Docker:

```bash
docker run --rm -p 8000:8000 -p 8501:8501 dental-app:latest
```

Or using docker-compose (recommended for local development):

```bash
docker-compose up --build
```

## Running backend (uvicorn) and frontend (Streamlit) together with Docker

This repository includes a Dockerfile and optional `docker-compose.yml` to run both the FastAPI backend and the Streamlit frontend in a single container image. The container can start:

- FastAPI (uvicorn) on port 8000
- Streamlit on port 8501

This README documents how to build and run the image, how to provide the (large) model weights at runtime, and common troubleshooting tips (memory, missing model, logs).

Quick build

```bash
# from the repository root
docker build -t dental-app:latest .
```

Run (simple)

```bash
# starts both services in the container; logs will appear in the container stdout
docker run --rm -p 8000:8000 -p 8501:8501 dental-app:latest
```

Run with Supervisor (recommended for nicer process management)

```bash
docker run --rm -e USE_SUPERVISOR=1 -p 8000:8000 -p 8501:8501 dental-app:latest
```

Run with docker-compose (development)

```bash
docker-compose up --build
```

Model weights (recommended: mount at runtime)
--------------------------------------------

Large model files (PyTorch `.pth`) should usually be kept out of the image for development. Mount your local `models/` directory into the container at runtime and tell the backend where the file is located:

```bash
docker run --rm \
	-e USE_SUPERVISOR=1 \
	-e MODEL_FILE=/app/models/best_model_clip.pth \
	-p 8000:8000 -p 8501:8501 \
	-v "$(pwd)/models:/app/models:ro" \
	dental-app:latest
```

- Default model path (inside the container): `/app/models/best_model_clip.pth`.
- You can override the path with `MODEL_FILE` or `MODEL_PATH` environment variables.

Health & readiness
------------------

The backend exposes a simple health endpoint:

- `GET /health` — returns JSON listing whether the model is loaded and the model path. Example (when model missing):

```json
{ "status": "ok", "model": { "loaded": false, "path": "/app/models/best_model_clip.pth" } }
```

If you mount the models directory and set `MODEL_FILE`, the server will try to eagerly load the model on startup (if present) and `/health` should show `model.loaded: true`.

Predict endpoint behaviour
--------------------------

- `POST /predict` accepts a file upload and returns `predicted_class` and `confidence`.
- If the model artifact is missing, `/predict` returns HTTP 503 Service Unavailable with a helpful message instructing you to mount the `models/` directory or set `MODEL_FILE`.

Logs and where to look
----------------------

- When running with `start.sh` (default), logs are written inside the container to `/tmp/service_logs/uvicorn.log` and `/tmp/service_logs/streamlit.log` (and stdout/stderr forwarded to the container logs).
- When running with Supervisor (`USE_SUPERVISOR=1`), Supervisor writes per-process logs to `/var/log/uvicorn.log`, `/var/log/uvicorn.err.log`, `/var/log/streamlit.log`, `/var/log/streamlit.err.log` inside the container.
- To view container logs from the host:

```bash
docker logs <container-name-or-id>
```

Troubleshooting
---------------

- Build fails with `cannot allocate memory` during `apt-get` or `pip install`:
	- On macOS Docker Desktop increase the VM memory (Preferences → Resources → Memory) to 6–8 GB and retry.
	- Or build on a machine / CI runner with more memory.
	- Use `--prefer-binary` for pip to prefer wheels (reduces compile memory usage).

- `GET /health` shows `model.loaded: false` even though you mounted `./models`:
	- Confirm the mounted file is present in the container:

		```bash
		docker run --rm -v "$(pwd)/models:/app/models:ro" dental-app:latest ls -l /app/models
		```

	- Confirm you set `MODEL_FILE` to the correct internal path (e.g. `/app/models/best_model_clip.pth`).
	- Check container logs for the startup loader message — the server prints whether it loaded the model at startup.

- If `/predict` returns a 503 with a message about the model file not found, mount the model file or set `MODEL_FILE` as shown above.

Dev option: create a small dummy weights file (for UI dev)
------------------------------------------------------

If you don't have the real model or want to exercise the UI without heavy weights, you can create a lightweight dummy weights file locally (requires local Python + PyTorch). Example (run on your host, not inside the container):

```python
# scripts/create_dummy_model.py
import torch
from backend.function import CustomCLIPVisionTransformer

model = CustomCLIPVisionTransformer(num_classes=6)
torch.save(model.state_dict(), 'models/best_model_clip.pth')
```

Run it in your virtualenv where torch and the repo are available:

```bash
python scripts/create_dummy_model.py
```

Then mount `./models` into the container as described earlier.

Notes & caveats
---------------

- Bundling heavy ML weights inside the image is not recommended for development (large image size, long builds). Mount the weights at runtime instead.
- For production deployments consider a separated model storage (S3, mounted volume, or a model server) and load the model as part of a separate lifecycle operation.

If something fails or you want me to add an automated `scripts/create_dummy_model.py` file and a docker-compose override that mounts logs/models, tell me and I will add it.

Verification
------------

- Backend docs: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Frontend: http://localhost:8501
