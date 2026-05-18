# MedDRA lookup webapp only (Render Docker runtime).
FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements-render.txt

COPY webapp ./webapp
COPY start.sh .

RUN chmod +x start.sh

CMD ["bash", "start.sh"]
