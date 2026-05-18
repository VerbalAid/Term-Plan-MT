# MedDRA lookup webapp — Render Docker runtime (CMD = start command).
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-render.txt

COPY webapp ./webapp
COPY start.sh ./
RUN chmod +x start.sh

EXPOSE 8000
CMD ["bash", "start.sh"]
