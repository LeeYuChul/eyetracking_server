# Eye-Tracking Heatmap Analysis Server

FastAPI MVP server for Figma eye-tracking heatmap analysis.

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
EYETRACK_MODEL_WEIGHTS_DIR=model/model_weights
EYETRACK_SALIENCY_MODEL_PATH=model/model_weights/saliency_models/UMSI++/umsi++.hdf5
EYETRACK_DEFAULT_MODEL_NAME=umsi++
EYETRACK_ALLOWED_MODEL_NAMES='["umsi++","heuristic"]'
EYETRACK_MODEL_RUNTIME=auto
EYETRACK_DEVICE=cuda
EYETRACK_CUDA_DEVICE_INDEX=0
```

## API Shape

`POST /api/v1/analyses` processes the uploaded image in memory and returns the report immediately. The server does not persist uploaded images, heatmaps, overlays, or reports. No per-job retrieval APIs are exposed.

Clients can select the analysis model with multipart field `model_name`.

Supported values:

- `umsi++`
- `heuristic`

The response includes:

- `report`
- `assets.heatmap_png_base64`
- `assets.overlay_png_base64`

## Notes

The UMSI++ HDF5 weights are discovered at startup. If a compatible TensorFlow/Keras runtime is installed and can load the file as a complete model, it is used for prediction. Otherwise the server runs in a weights-bound heuristic mode while still reporting the configured model path and HDF5 metadata in `/api/v1/health`.
