# syntax=docker/dockerfile:1.7
FROM tensorflow/tensorflow:2.18.0-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    EYETRACK_HOST=0.0.0.0 \
    EYETRACK_PORT=3781

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip uv \
    && uv pip install --system -r requirements.txt

COPY app ./app
COPY model ./model
COPY .env ./.env

EXPOSE 3781

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "3781"]
