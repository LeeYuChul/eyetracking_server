# Eye Tracking Frame Chat Server

Stateless FastAPI server for Figma frame-level eye-tracking analysis. It analyzes selected UI frames, returns heatmap and scanpath artifacts as base64 response data, and provides an SSE VLM chatbot grounded in one frame's original, heatmap overlay, and scanpath overlay images.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

## Run

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 3781
```

Health check:

```bash
curl http://127.0.0.1:3781/api/v1/health
```

## Docker Deploy

The container keeps the same host port, `3781`, so the existing Cloudflare Origin can continue to point to `http://127.0.0.1:3781`.

Build and deploy with the provided script:

```bash
chmod +x scripts/deploy_docker.sh
./scripts/deploy_docker.sh
```

The script does the following every time:

- stops the existing `eyetrack-analysis-server` container if it exists
- removes that container
- rebuilds `eyetrack-analysis-server:latest`
- starts a new container with `-p 3781:3781`
- uses `.env` for runtime variables
- uses `--gpus all` when the NVIDIA Docker runtime is available
- verifies `/api/v1/health`

The Docker build uses BuildKit cache mounts and `uv pip install` for faster dependency installation on repeated builds. Keep BuildKit enabled:

```bash
export DOCKER_BUILDKIT=1
```

Repeated builds reuse:

- Docker layer cache from `eyetrack-analysis-server:latest`
- BuildKit cache mounts for `/root/.cache/uv`
- BuildKit cache mounts for `/root/.cache/pip`

As long as `requirements.txt` does not change, dependency installation should be mostly cached. As long as `model/` does not change, the model weights layer should also be reused.

Manual Docker commands:

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from eyetrack-analysis-server:latest \
  -t eyetrack-analysis-server:latest .
docker stop eyetrack-analysis-server || true
docker rm eyetrack-analysis-server || true
docker run -d \
  --name eyetrack-analysis-server \
  --restart unless-stopped \
  --env-file .env \
  --gpus all \
  -p 3781:3781 \
  eyetrack-analysis-server:latest
```

Docker Compose:

```bash
DOCKER_BUILDKIT=1 docker compose up -d --build
```

If you use Compose and need to force replacement:

```bash
docker compose down
docker compose up -d --build
```

## Configuration

Runtime settings are read from `.env` using the `EYETRACK_` prefix.

```bash
EYETRACK_PORT=3781
EYETRACK_BASE_URL=https://eyetrack.newlearn.ai.kr
EYETRACK_STORAGE_POLICY=stateless_response_only
EYETRACK_MODEL_WEIGHTS_DIR=model/model_weights
EYETRACK_SALIENCY_MODEL_PATH=model/model_weights/saliency_models/UMSI++/umsi++.hdf5
EYETRACK_DEFAULT_MODEL_NAME=umsi++
EYETRACK_ALLOWED_MODEL_NAMES=["umsi++","heuristic"]
EYETRACK_MODEL_RUNTIME=auto
EYETRACK_DEVICE=cuda
EYETRACK_CUDA_DEVICE_INDEX=0
EYETRACK_MAX_FRAMES=15
EYETRACK_MAX_UPLOAD_BYTES=10485760
EYETRACK_MAX_TOTAL_UPLOAD_BYTES=104857600
EYETRACK_MODEL_IDLE_UNLOAD_SECONDS=0
EYETRACK_VLM_PROVIDER=ollama
EYETRACK_OLLAMA_BASE_URL=http://host.docker.internal:11434
EYETRACK_OLLAMA_MODEL=gemma4:26b
EYETRACK_OLLAMA_KEEP_ALIVE=1m
EYETRACK_OLLAMA_NUM_CTX=32768
EYETRACK_OLLAMA_NUM_PREDICT=768
EYETRACK_VLM_REQUEST_TIMEOUT_SECONDS=600
EYETRACK_VLM_IMAGE_CHUNK_SIZE=5
EYETRACK_VLM_MAX_IMAGE_SIDE=768
EYETRACK_VLM_IMAGE_JPEG_QUALITY=82
EYETRACK_OPENAI_API_KEY=
EYETRACK_OPENAI_MODEL=gpt-4.1-mini
EYETRACK_OPENAI_BASE_URL=https://api.openai.com/v1
```

## API Shape

The server is request-in/response-out. Clients must store returned bundles locally if they want to reuse them.

- `GET /api/v1/health` returns server, storage policy, heatmap, scanpath, and VLM provider status.
- `POST /api/v1/frames/analyze` accepts up to 15 frame images plus `frames_meta`, then returns per-frame heatmap and scanpath results.
- `POST /api/v1/frames/chat/stream` accepts one frame's original, heatmap overlay, scanpath overlay, metrics, and a user question, then streams progress and final answer events.
- `POST /api/v1/analyses` remains available as a legacy single-frame compatibility endpoint.

`POST /api/v1/frames/analyze` expects multipart fields:

- `files`: repeated PNG/JPEG frame files
- `frames_meta`: JSON array with `client_frame_id`, `figma_node_id`, `frame_name`, `width`, `height`, `file_key`, `order_index`
- `model_name`: optional `umsi++` or `heuristic`
- `options`: optional JSON object

Clients can select the analysis model with multipart field `model_name`.

Supported values:

- `umsi++`
- `heuristic` for lightweight local development and tests

The frame analysis response includes:

- per-frame `metrics` with scanpath length, fixation count, entropy, complexity, and fixation points
- per-frame `artifacts.original`, `heatmap`, `heatmap_overlay`, `scanpath_overlay`
- `model_info`

`POST /api/v1/frames/chat/stream` returns Server-Sent Events:

- `progress`: request stages visible to the plugin UI
- `thinking`: user-visible evidence processing updates
- `final`: JSON answer with conclusion, reasoning summary, evidence image roles, risk/confidence, caveat, and recommendations

The legacy single-frame response still includes:

- `report`
- `assets.heatmap_png_base64`
- `assets.overlay_png_base64`

## Notes

The UMSI++ HDF5 weights are discovered at startup. If a compatible TensorFlow/Keras runtime is installed and can load the file as a complete model, it is used for prediction. Otherwise the server runs in a weights-bound heuristic mode while still reporting the configured model path and HDF5 metadata in `/api/v1/health`.
