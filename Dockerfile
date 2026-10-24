FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="Puffx AI Image Generator" \
      org.opencontainers.image.description="Local and offline AI image creation by Puffx Host" \
      org.opencontainers.image.vendor="Puffx Host"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app.py model_catalog.py .
COPY scripts/ scripts/

EXPOSE 7860
CMD ["python", "app.py"]
