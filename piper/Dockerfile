FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng git build-essential wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "piper-tts==1.2.0"

RUN mkdir -p /app/models

EXPOSE 5002

CMD ["sleep", "infinity"]
