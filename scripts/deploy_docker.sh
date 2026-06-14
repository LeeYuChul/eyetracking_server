#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-eyetrack-analysis-server:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-eyetrack-analysis-server}"
HOST_PORT="${HOST_PORT:-3781}"
CONTAINER_PORT="${CONTAINER_PORT:-3781}"
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-1}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Stopping existing container: ${CONTAINER_NAME}"
  docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Removing existing container: ${CONTAINER_NAME}"
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "Building image: ${IMAGE_NAME}"
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  -t "${IMAGE_NAME}" \
  .

GPU_ARGS=()
if docker info --format '{{json .Runtimes}}' | grep -q 'nvidia'; then
  GPU_ARGS=(--gpus all)
else
  echo "NVIDIA Docker runtime was not detected. Starting without --gpus all."
fi

echo "Starting container: ${CONTAINER_NAME}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart unless-stopped \
  --env-file .env \
  --add-host=host.docker.internal:host-gateway \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  "${GPU_ARGS[@]}" \
  "${IMAGE_NAME}"

echo "Waiting for health check on http://127.0.0.1:${HOST_PORT}/api/v1/health"
for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:${HOST_PORT}/api/v1/health" >/dev/null; then
    echo "Deployment is healthy."
    curl -fsS "http://127.0.0.1:${HOST_PORT}/api/v1/health"
    echo
    exit 0
  fi
  sleep 2
done

echo "Deployment failed health check. Recent logs:"
docker logs --tail 100 "${CONTAINER_NAME}"
exit 1
