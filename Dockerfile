# Dockerfile for Hoffman2 Chatbot

FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend

RUN python frontend/build_frontend.py

COPY app.py .
COPY hoffman2_chatbot_web.py .

COPY hoffman2_index ./hoffman2_index

EXPOSE 5000

CMD ["python", "app.py"]
